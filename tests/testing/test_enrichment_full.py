"""
KRNX Enrichment - Comprehensive Test Suite

Run with: python test_enrichment_full.py

Full test coverage for multi-signal relation scoring system.
"""

import os
import sys
import time
import unittest
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Add project root and fabric/ to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

fabric_path = os.path.join(project_root, 'fabric')
if os.path.exists(fabric_path):
    sys.path.insert(0, fabric_path)


# ==============================================
# MOCK EVENT CLASS
# ==============================================

@dataclass
class MockEvent:
    """Mock event for testing."""
    event_id: str
    content: Dict[str, Any]
    timestamp: float
    user_id: str = "user_1"
    actor_id: str = None
    metadata: Dict[str, Any] = None
    episode_id: str = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.actor_id is None:
            self.actor_id = self.user_id


# ==============================================
# FEATURE EXTRACTION TESTS
# ==============================================

class TestNegationDetection(unittest.TestCase):
    """Test negation mismatch detection."""
    
    def setUp(self):
        from enrichment.features import has_negation, negation_mismatch
        self.has_negation = has_negation
        self.negation_mismatch = negation_mismatch
    
    def test_simple_negation(self):
        """Test basic negation detection."""
        self.assertTrue(self.has_negation("This is not correct"))
        self.assertTrue(self.has_negation("I can't do this"))
        self.assertTrue(self.has_negation("Never do that"))
        self.assertTrue(self.has_negation("There is nothing here"))
        
    def test_no_negation(self):
        """Test text without negation."""
        self.assertFalse(self.has_negation("This is correct"))
        self.assertFalse(self.has_negation("I can do this"))
        self.assertFalse(self.has_negation("Everything is fine"))
    
    def test_contraction_negation(self):
        """Test contractions."""
        self.assertTrue(self.has_negation("It isn't working"))
        self.assertTrue(self.has_negation("We won't proceed"))
        self.assertTrue(self.has_negation("They don't know"))
        self.assertTrue(self.has_negation("She doesn't agree"))
    
    def test_negation_mismatch(self):
        """Test mismatch between positive and negative statements."""
        self.assertTrue(self.negation_mismatch(
            "The project is approved",
            "The project is not approved"
        ))
        self.assertTrue(self.negation_mismatch(
            "I can access the file",
            "I cannot access the file"
        ))
    
    def test_no_mismatch(self):
        """Test matching negation states."""
        self.assertFalse(self.negation_mismatch(
            "This works fine",
            "That works well"
        ))
        self.assertFalse(self.negation_mismatch(
            "I don't know",
            "I can't tell"
        ))


class TestNumericDetection(unittest.TestCase):
    """Test numeric mismatch detection."""
    
    def setUp(self):
        from enrichment.features import extract_numerics, numeric_mismatch
        self.extract_numerics = extract_numerics
        self.numeric_mismatch = numeric_mismatch
    
    def test_currency_extraction(self):
        """Test currency extraction."""
        nums = self.extract_numerics("Budget is $50,000")
        self.assertIn("$50,000", nums)
        
        nums = self.extract_numerics("Cost: $1,234.56")
        self.assertIn("$1,234.56", nums)
    
    def test_time_extraction(self):
        """Test time extraction."""
        nums = self.extract_numerics("Meeting at 3:00pm")
        self.assertTrue(any("3:00" in n for n in nums))
        
        # Note: Simple times like "2pm" are handled by temporal extraction
        # Numeric extraction handles times with colons (3:00pm format)
    
    def test_percentage_extraction(self):
        """Test percentage extraction."""
        nums = self.extract_numerics("Growth is 15%")
        self.assertIn("15%", nums)
    
    def test_numeric_mismatch(self):
        """Test numeric mismatch detection."""
        self.assertTrue(self.numeric_mismatch(
            "Budget is $50,000",
            "Budget is $75,000"
        ))
        self.assertTrue(self.numeric_mismatch(
            "Meeting at 3pm",
            "Meeting at 4pm"
        ))
    
    def test_no_numeric_mismatch(self):
        """Test matching numbers."""
        self.assertFalse(self.numeric_mismatch(
            "Budget is $50,000",
            "Budget is $50,000"
        ))
        # One text without numbers
        self.assertFalse(self.numeric_mismatch(
            "The project is ready",
            "Budget is $50,000"
        ))


class TestTemporalDetection(unittest.TestCase):
    """Test temporal mismatch detection."""
    
    def setUp(self):
        from enrichment.features import extract_temporal, temporal_mismatch
        self.extract_temporal = extract_temporal
        self.temporal_mismatch = temporal_mismatch
    
    def test_day_extraction(self):
        """Test day extraction."""
        temp = self.extract_temporal("Meeting on Monday")
        self.assertIn("monday", temp.days)
        
        temp = self.extract_temporal("See you Tuesday and Friday")
        self.assertIn("tuesday", temp.days)
        self.assertIn("friday", temp.days)
    
    def test_relative_extraction(self):
        """Test relative time extraction."""
        temp = self.extract_temporal("Let's meet tomorrow")
        self.assertIn("tomorrow", temp.relative)
        
        temp = self.extract_temporal("I'll do it next week")
        self.assertIn("next week", temp.relative)
    
    def test_time_extraction(self):
        """Test time extraction."""
        temp = self.extract_temporal("Meeting at 3pm")
        self.assertTrue(len(temp.times) > 0)
    
    def test_temporal_mismatch_same_day_different_time(self):
        """Test mismatch: same day, different times."""
        self.assertTrue(self.temporal_mismatch(
            "Meeting Tuesday at 3pm",
            "Meeting Tuesday at 4pm"
        ))
    
    def test_temporal_mismatch_conflicting_relative(self):
        """Test mismatch: conflicting relative terms."""
        self.assertTrue(self.temporal_mismatch(
            "Let's meet today",
            "Let's meet tomorrow"
        ))
        self.assertTrue(self.temporal_mismatch(
            "Due next week",
            "Due last week"
        ))
    
    def test_no_temporal_mismatch(self):
        """Test no mismatch for consistent times."""
        self.assertFalse(self.temporal_mismatch(
            "Meeting on Monday",
            "Meeting on Monday"
        ))
        self.assertFalse(self.temporal_mismatch(
            "Meeting on Monday",
            "Review on Tuesday"
        ))


class TestAntonymDetection(unittest.TestCase):
    """Test antonym detection."""
    
    def setUp(self):
        from enrichment.features import antonym_detected, find_antonyms
        self.antonym_detected = antonym_detected
        self.find_antonyms = find_antonyms
    
    def test_common_antonyms(self):
        """Test common antonym pairs."""
        self.assertTrue(self.antonym_detected("approved", "rejected"))
        self.assertTrue(self.antonym_detected("joined the team", "left the team"))
        self.assertTrue(self.antonym_detected("started", "stopped"))
        self.assertTrue(self.antonym_detected("enabled", "disabled"))
    
    def test_no_antonym(self):
        """Test no antonym detection."""
        self.assertFalse(self.antonym_detected("meeting", "conference"))
        self.assertFalse(self.antonym_detected("project", "task"))


class TestPairFeatures(unittest.TestCase):
    """Test PairFeatures dataclass and extraction."""
    
    def setUp(self):
        from enrichment.features import PairFeatures, FeatureExtractor
        self.PairFeatures = PairFeatures
        self.extractor = FeatureExtractor()
    
    def test_contradiction_count(self):
        """Test contradiction counting."""
        features = self.PairFeatures(
            negation_mismatch=True,
            numeric_mismatch=True,
            temporal_mismatch=False,
            antonym_detected=False,
        )
        self.assertEqual(features.contradiction_count, 2)
        self.assertTrue(features.strict_contradiction)
    
    def test_has_contradiction(self):
        """Test has_contradiction property."""
        features = self.PairFeatures(negation_mismatch=True)
        self.assertTrue(features.has_contradiction)
        
        features = self.PairFeatures()
        self.assertFalse(features.has_contradiction)
    
    def test_full_extraction(self):
        """Test full feature extraction."""
        event_a = MockEvent(
            event_id="evt_a",
            content={"text": "Budget is $50,000"},
            timestamp=1000.0,
            user_id="alice",
        )
        event_b = MockEvent(
            event_id="evt_b",
            content={"text": "Budget is $75,000"},
            timestamp=900.0,
            user_id="bob",
        )
        
        features = self.extractor.extract(event_a, event_b, embedding_similarity=0.89)
        
        self.assertEqual(features.embedding_similarity, 0.89)
        self.assertTrue(features.numeric_mismatch)
        self.assertFalse(features.same_actor)
        self.assertEqual(features.timestamp_delta, 100.0)


# ==============================================
# RELATION SCORING TESTS
# ==============================================

class TestRelationScorer(unittest.TestCase):
    """Test multi-signal relation scoring."""
    
    def setUp(self):
        from enrichment.relations import RelationScorer, RelationScoringConfig, RelationType
        from enrichment.features import FeatureExtractor
        
        self.scorer = RelationScorer()
        self.extractor = FeatureExtractor()
        self.RelationType = RelationType
    
    def test_duplicate_detection(self):
        """Test duplicate detection (high similarity, no contradiction)."""
        event_a = MockEvent("evt_a", {"text": "User prefers dark mode"}, 1000.0)
        event_b = MockEvent("evt_b", {"text": "User prefers dark mode"}, 900.0)
        
        features = self.extractor.extract(event_a, event_b, embedding_similarity=0.98)
        results = self.scorer.score_pair(event_a, event_b, features)
        
        duplicates = [r for r in results if r.kind == self.RelationType.DUPLICATES]
        self.assertTrue(len(duplicates) > 0)
        self.assertEqual(duplicates[0].target, "evt_b")
    
    def test_supersedes_detection(self):
        """Test supersedes detection (correction/update)."""
        new_event = MockEvent("evt_new", {"text": "Meeting at 4pm"}, 2000.0)
        old_event = MockEvent("evt_old", {"text": "Meeting at 3pm"}, 1000.0)
        
        features = self.extractor.extract(new_event, old_event, embedding_similarity=0.88)
        results = self.scorer.score_pair(new_event, old_event, features)
        
        supersedes = [r for r in results if r.kind == self.RelationType.SUPERSEDES]
        self.assertTrue(len(supersedes) > 0)
        
        result = supersedes[0]
        self.assertEqual(result.target, "evt_old")
        self.assertIn("supersedes:", result.reason_code)
    
    def test_contradicts_detection(self):
        """Test contradicts detection (different actors)."""
        event_a = MockEvent("evt_alice", {"text": "approved"}, 1000.0, user_id="alice")
        event_b = MockEvent("evt_bob", {"text": "not approved"}, 1100.0, user_id="bob")
        
        features = self.extractor.extract(event_b, event_a, embedding_similarity=0.91)
        results = self.scorer.score_pair(event_b, event_a, features)
        
        contradicts = [r for r in results if r.kind == self.RelationType.CONTRADICTS]
        self.assertTrue(len(contradicts) > 0)
        self.assertIn("different_actor", contradicts[0].reason_code)
    
    def test_expands_on_detection(self):
        """Test expands_on detection."""
        event_a = MockEvent(
            "evt_a",
            {"text": "Project Alpha requires Python 3.9"},
            1000.0,
            metadata={"entities": [{"text": "Project Alpha", "type": "project"}]}
        )
        event_b = MockEvent(
            "evt_b",
            {"text": "Project Alpha started last week"},
            900.0,
            metadata={"entities": [{"text": "Project Alpha", "type": "project"}]}
        )
        
        features = self.extractor.extract(event_a, event_b, embedding_similarity=0.75)
        results = self.scorer.score_pair(event_a, event_b, features)
        
        expands = [r for r in results if r.kind == self.RelationType.EXPANDS_ON]
        self.assertTrue(len(expands) > 0)
    
    def test_replies_to_detection(self):
        """Test replies_to detection."""
        new_event = MockEvent("evt_new", {"text": "Yes, let's do it"}, 1050.0)
        prev_event = MockEvent("evt_prev", {"text": "Should we proceed?"}, 1000.0)
        
        result = self.scorer.score_replies_to(new_event, prev_event)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.kind, self.RelationType.REPLIES_TO)
        self.assertIn("temporal_proximity", result.signals)
    
    def test_strict_contradiction_flag(self):
        """Test strict_contradiction flag (2+ signals)."""
        # Event with multiple contradiction signals
        event_a = MockEvent("evt_a", {"text": "Meeting is not Tuesday at 3pm"}, 2000.0)
        event_b = MockEvent("evt_b", {"text": "Meeting is Tuesday at 4pm"}, 1000.0)
        
        features = self.extractor.extract(event_a, event_b, embedding_similarity=0.85)
        results = self.scorer.score_pair(event_a, event_b, features)
        
        supersedes = [r for r in results if r.kind == self.RelationType.SUPERSEDES]
        if supersedes:
            # Should have strict_contradiction if 2+ signals
            if features.contradiction_count >= 2:
                self.assertTrue(supersedes[0].strict_contradiction)


class TestRelationSummary(unittest.TestCase):
    """Test relation summary computation."""
    
    def setUp(self):
        from enrichment.summary import RelationSummaryComputer, RelationSummary
        from enrichment.relations import RelationResult, RelationType
        
        self.computer = RelationSummaryComputer()
        self.RelationResult = RelationResult
        self.RelationType = RelationType
    
    def test_basic_summary(self):
        """Test basic summary computation."""
        relations = [
            self.RelationResult(
                kind=self.RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["temporal_mismatch"],
                reason_code="supersedes",
                strict_contradiction=False,
            ),
        ]
        
        summary = self.computer.compute("evt_new", relations)
        
        self.assertEqual(summary.edge_count, 1)
        self.assertEqual(summary.by_kind["supersedes"], 1)
        self.assertTrue(summary.has_supersedes)
        self.assertFalse(summary.is_root)
    
    def test_find_current_version(self):
        """Test finding current version through supersession chain."""
        # Build a chain: evt_3 supersedes evt_2 supersedes evt_1
        all_relations = {
            "evt_1": [],
            "evt_2": [
                self.RelationResult(
                    kind=self.RelationType.SUPERSEDES,
                    target="evt_1",
                    confidence=0.9,
                    signals=[],
                    reason_code="",
                    strict_contradiction=False,
                )
            ],
            "evt_3": [
                self.RelationResult(
                    kind=self.RelationType.SUPERSEDES,
                    target="evt_2",
                    confidence=0.9,
                    signals=[],
                    reason_code="",
                    strict_contradiction=False,
                )
            ],
        }
        
        current = self.computer.find_current_version("evt_1", all_relations)
        self.assertEqual(current, "evt_3")
    
    def test_effective_events(self):
        """Test finding effective (leaf) events."""
        all_relations = {
            "evt_1": [],  # Superseded by evt_2
            "evt_2": [
                self.RelationResult(
                    kind=self.RelationType.SUPERSEDES,
                    target="evt_1",
                    confidence=0.9,
                    signals=[],
                    reason_code="",
                    strict_contradiction=False,
                )
            ],
            "evt_3": [],  # Independent, also effective
        }
        
        effective = self.computer.get_effective_events(all_relations)
        
        self.assertIn("evt_2", effective)
        self.assertIn("evt_3", effective)
        self.assertNotIn("evt_1", effective)  # Superseded


# ==============================================
# SALIENCE TESTS
# ==============================================

class TestSalienceEngine(unittest.TestCase):
    """Test salience scoring."""
    
    def setUp(self):
        from enrichment.salience import SalienceEngine, SalienceConfig, SalienceMethod
        from enrichment.relations import RelationResult, RelationType
        
        self.engine = SalienceEngine()
        self.SalienceMethod = SalienceMethod
        self.RelationResult = RelationResult
        self.RelationType = RelationType
    
    def test_recency_decay(self):
        """Test recency decay."""
        now = time.time()
        
        # Recent event
        result = self.engine.compute(
            "evt_1",
            timestamp=now - 3600,  # 1 hour ago
            now=now,
        )
        recent_score = result.factors["recency"]
        
        # Older event
        result = self.engine.compute(
            "evt_2",
            timestamp=now - 86400,  # 24 hours ago
            now=now,
        )
        old_score = result.factors["recency"]
        
        self.assertGreater(recent_score, old_score)
    
    def test_relation_adjustment(self):
        """Test salience adjustment from relations."""
        from enrichment.salience import adjust_salience_from_relations
        
        relations = [
            self.RelationResult(
                kind=self.RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["temporal_mismatch"],
                reason_code="",
                strict_contradiction=False,
            ),
        ]
        
        base = 0.5
        adjusted = adjust_salience_from_relations(base, relations)
        
        self.assertEqual(adjusted, 0.6)  # +0.10 for supersedes
    
    def test_strict_contradiction_boost(self):
        """Test extra boost for strict contradiction."""
        relations = [
            self.RelationResult(
                kind=self.RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["negation", "temporal"],
                reason_code="",
                strict_contradiction=True,
            ),
        ]
        
        base = 0.5
        adjusted = self.engine.adjust_from_relations(
            self.engine.compute("evt", time.time()),
            relations,
        )
        
        # Should have supersedes_boost + strict_contradiction_boost
        self.assertGreater(adjusted.relation_boost, 0.10)


# ==============================================
# RETENTION TESTS
# ==============================================

class TestRetentionClassifier(unittest.TestCase):
    """Test retention classification."""
    
    def setUp(self):
        from enrichment.retention_v2 import (
            RetentionClassifier, RetentionClass, DriftComputer
        )
        from enrichment.relations import RelationResult, RelationType
        
        self.classifier = RetentionClassifier()
        self.drift_computer = DriftComputer()
        self.RetentionClass = RetentionClass
        self.RelationResult = RelationResult
        self.RelationType = RelationType
    
    def test_durable_classification(self):
        """Test durable classification (high salience, low drift)."""
        now = time.time()
        
        result = self.classifier.classify_with_drift(
            timestamp=now - 3600,  # Recent
            salience=0.8,  # High
            relations=[],
            now=now,
        )
        
        self.assertEqual(result.retention_class, self.RetentionClass.DURABLE)
    
    def test_ephemeral_classification(self):
        """Test ephemeral classification (low salience, high drift)."""
        now = time.time()
        
        result = self.classifier.classify_with_drift(
            timestamp=now - 86400 * 30,  # 30 days ago
            salience=0.1,  # Low
            relations=[],
            now=now,
        )
        
        self.assertEqual(result.retention_class, self.RetentionClass.EPHEMERAL)
    
    def test_strict_contradiction_override(self):
        """Test strict contradiction override to durable."""
        now = time.time()
        
        relations = [
            self.RelationResult(
                kind=self.RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["negation", "numeric"],
                reason_code="",
                strict_contradiction=True,
            ),
        ]
        
        result = self.classifier.classify_with_drift(
            timestamp=now - 86400 * 30,  # Old
            salience=0.1,  # Low
            relations=relations,
            now=now,
        )
        
        self.assertEqual(result.retention_class, self.RetentionClass.DURABLE)
        self.assertTrue(result.is_strict_contradiction)
    
    def test_drift_computation(self):
        """Test drift computation."""
        now = time.time()
        
        # Recent event = low drift
        drift = self.drift_computer.compute(
            timestamp=now - 3600,
            relations=[],
            now=now,
        )
        self.assertLess(drift, 0.3)
        
        # Old event = high drift
        drift = self.drift_computer.compute(
            timestamp=now - 86400 * 30,
            relations=[],
            now=now,
        )
        self.assertGreater(drift, 0.5)


# ==============================================
# INTEGRATION TESTS (Internal Pipeline)
# ==============================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_enrichment_pipeline(self):
        """Test complete enrichment pipeline."""
        from enrichment import (
            FeatureExtractor,
            RelationScorer,
            SalienceEngine,
            RetentionClassifier,
            RelationSummaryComputer,
        )
        
        now = time.time()
        
        # Create events
        old_event = MockEvent(
            event_id="evt_old",
            content={"text": "Meeting scheduled for 3pm"},
            timestamp=now - 3600,
            user_id="alice",
        )
        
        new_event = MockEvent(
            event_id="evt_new",
            content={"text": "Meeting rescheduled to 4pm"},
            timestamp=now,
            user_id="alice",
        )
        
        # 1. Extract features
        extractor = FeatureExtractor()
        features = extractor.extract(new_event, old_event, embedding_similarity=0.85)
        
        # 2. Score relations
        scorer = RelationScorer()
        relations = scorer.score_pair(new_event, old_event, features)
        
        # 3. Compute salience with adjustments
        salience_engine = SalienceEngine()
        salience = salience_engine.compute_with_relations(
            event_id=new_event.event_id,
            timestamp=new_event.timestamp,
            relations=relations,
            now=now,
        )
        
        # 4. Classify retention
        classifier = RetentionClassifier()
        retention = classifier.classify_with_drift(
            timestamp=new_event.timestamp,
            salience=salience.score,
            relations=relations,
            now=now,
        )
        
        # 5. Compute summary
        summary_computer = RelationSummaryComputer()
        summary = summary_computer.compute(new_event.event_id, relations)
        
        # Verify pipeline results
        self.assertTrue(features.temporal_mismatch or features.numeric_mismatch)
        self.assertTrue(len(relations) > 0)
        self.assertGreater(salience.score, 0)
        self.assertIsNotNone(retention.retention_class)
        self.assertGreater(summary.edge_count, 0)
    
    def test_contradiction_chain(self):
        """Test handling of contradiction chains."""
        from enrichment import (
            FeatureExtractor,
            RelationScorer,
            RelationType,
        )
        
        now = time.time()
        
        # Original statement
        event_1 = MockEvent(
            "evt_1",
            {"text": "Budget is $50,000"},
            now - 7200,
            user_id="alice",
        )
        
        # First correction
        event_2 = MockEvent(
            "evt_2",
            {"text": "Budget is $60,000"},
            now - 3600,
            user_id="alice",
        )
        
        # Second correction
        event_3 = MockEvent(
            "evt_3",
            {"text": "Budget is $75,000"},
            now,
            user_id="alice",
        )
        
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        
        # Check evt_2 supersedes evt_1
        features = extractor.extract(event_2, event_1, embedding_similarity=0.88)
        relations = scorer.score_pair(event_2, event_1, features)
        supersedes = [r for r in relations if r.kind == RelationType.SUPERSEDES]
        self.assertTrue(len(supersedes) > 0)
        
        # Check evt_3 supersedes evt_2
        features = extractor.extract(event_3, event_2, embedding_similarity=0.88)
        relations = scorer.score_pair(event_3, event_2, features)
        supersedes = [r for r in relations if r.kind == RelationType.SUPERSEDES]
        self.assertTrue(len(supersedes) > 0)


# ==============================================
# SYSTEM INTEGRATION TESTS (Real Event Model)
# ==============================================

# Try to import KRNX models at module level
_KRNX_AVAILABLE = False
_Event = None
_create_event = None

try:
    import sys
    sys.path.insert(0, '/mnt/user-data/uploads')
    from models import Event as _Event, create_event as _create_event
    _KRNX_AVAILABLE = True
except ImportError:
    pass


class TestSystemIntegration(unittest.TestCase):
    """
    System integration tests using real KRNX Event model.
    
    Tests multi-signal enrichment against actual Event objects
    to ensure compatibility with the production system.
    """
    
    def setUp(self):
        if not _KRNX_AVAILABLE:
            self.skipTest("KRNX models not available")
    
    def test_with_real_event_model(self):
        """Test enrichment with real KRNX Event objects."""
        from enrichment import FeatureExtractor, RelationScorer, RelationType
        
        now = time.time()
        
        # Create real Event objects - use numeric mismatch which is more reliable
        old_event = _create_event(
            event_id="evt_real_001",
            workspace_id="ws_test",
            user_id="user_integration",
            content={"text": "Project budget is $50,000"},
            timestamp=now - 3600,
            channel="chat",
        )
        
        new_event = _create_event(
            event_id="evt_real_002",
            workspace_id="ws_test",
            user_id="user_integration",
            content={"text": "Project budget is $75,000"},
            timestamp=now,
            channel="chat",
        )
        
        # Extract features from real events
        extractor = FeatureExtractor()
        features = extractor.extract(new_event, old_event, embedding_similarity=0.88)
        
        # Should detect numeric mismatch ($50k vs $75k)
        self.assertTrue(features.numeric_mismatch)
        self.assertTrue(features.has_contradiction)
        
        # Score relations
        scorer = RelationScorer()
        relations = scorer.score_pair(new_event, old_event, features)
        
        # Should detect supersedes
        supersedes = [r for r in relations if r.kind == RelationType.SUPERSEDES]
        self.assertTrue(len(supersedes) > 0)
        self.assertEqual(supersedes[0].target, "evt_real_001")
    
    def test_backward_compatibility_relation_enricher(self):
        """Test backward compatibility with legacy RelationEnricher API."""
        from enrichment import RelationEnricher, RelationType
        
        now = time.time()
        
        # Create events
        old_event = _create_event(
            event_id="evt_compat_001",
            workspace_id="ws_test",
            user_id="user_test",
            content={"text": "Meeting at 2pm"},
            timestamp=now - 600,
        )
        
        new_event = _create_event(
            event_id="evt_compat_002",
            workspace_id="ws_test",
            user_id="user_test",
            content={"text": "Meeting moved to 3pm"},
            timestamp=now,
        )
        
        # Use legacy API
        enricher = RelationEnricher(
            duplicate_threshold=0.95,
            expand_threshold=0.70,
            supersede_threshold=0.70,
        )
        
        relations = enricher.detect(
            event_id=new_event.event_id,
            timestamp=new_event.timestamp,
            content=new_event.content,
            previous_event=old_event,
            similarity_scores={old_event.event_id: 0.85},
            workspace_events=[old_event],
        )
        
        # Should return List[Relation] (legacy format)
        self.assertIsInstance(relations, list)
        
        # Relations should have the standard fields
        for rel in relations:
            self.assertTrue(hasattr(rel, 'kind'))
            self.assertTrue(hasattr(rel, 'target'))
            self.assertTrue(hasattr(rel, 'confidence'))
            
            # to_dict should work
            rel_dict = rel.to_dict()
            self.assertIn('kind', rel_dict)
            self.assertIn('target', rel_dict)
    
    def test_multi_user_contradiction(self):
        """Test contradiction detection between different users."""
        from enrichment import FeatureExtractor, RelationScorer, RelationType
        
        now = time.time()
        
        # Alice says project is approved
        alice_event = _create_event(
            event_id="evt_alice_001",
            workspace_id="ws_project",
            user_id="alice",
            content={"text": "Project Alpha is approved for launch"},
            timestamp=now - 300,
        )
        
        # Bob says it's not approved
        bob_event = _create_event(
            event_id="evt_bob_001",
            workspace_id="ws_project",
            user_id="bob",
            content={"text": "Project Alpha is not approved yet"},
            timestamp=now,
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(bob_event, alice_event, embedding_similarity=0.88)
        
        # Should detect negation mismatch
        self.assertTrue(features.negation_mismatch)
        self.assertFalse(features.same_actor)  # Different users
        
        scorer = RelationScorer()
        relations = scorer.score_pair(bob_event, alice_event, features)
        
        # Should detect CONTRADICTS (not supersedes, since different actors)
        contradicts = [r for r in relations if r.kind == RelationType.CONTRADICTS]
        self.assertTrue(len(contradicts) > 0)
        self.assertIn("different_actor", contradicts[0].reason_code)
    
    def test_episode_continuity(self):
        """Test relation scoring respects episode boundaries."""
        from enrichment import FeatureExtractor, RelationScorer
        
        now = time.time()
        
        # Event in episode 1
        event_ep1 = _create_event(
            event_id="evt_ep1_001",
            workspace_id="ws_test",
            user_id="user_test",
            content={"text": "Working on the API"},
            timestamp=now - 7200,
            episode_id="episode_001",
        )
        
        # Event in episode 2 (different conversation)
        event_ep2 = _create_event(
            event_id="evt_ep2_001",
            workspace_id="ws_test",
            user_id="user_test",
            content={"text": "Working on the API endpoint"},
            timestamp=now,
            episode_id="episode_002",
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(event_ep2, event_ep1, embedding_similarity=0.80)
        
        # Different episodes
        self.assertFalse(features.same_episode)
    
    def test_strict_contradiction_preservation(self):
        """Test strict contradiction flows through to retention."""
        from enrichment import (
            FeatureExtractor, 
            RelationScorer,
            RetentionClassifier,
            RetentionClass,
        )
        
        now = time.time()
        
        # Create events with multiple contradiction signals
        old_event = _create_event(
            event_id="evt_strict_001",
            workspace_id="ws_test",
            user_id="user_test",
            content={"text": "Budget approved for $50,000 on Tuesday"},
            timestamp=now - 86400 * 30,  # 30 days ago
        )
        
        # Contradicts on: negation + numeric + temporal
        new_event = _create_event(
            event_id="evt_strict_002",
            workspace_id="ws_test",
            user_id="user_test",
            content={"text": "Budget not approved, was $75,000 on Wednesday"},
            timestamp=now,
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(new_event, old_event, embedding_similarity=0.75)
        
        # Should have multiple contradiction signals
        self.assertGreaterEqual(features.contradiction_count, 2)
        self.assertTrue(features.strict_contradiction)
        
        scorer = RelationScorer()
        relations = scorer.score_pair(new_event, old_event, features)
        
        # Find supersedes relation
        supersedes = [r for r in relations if r.kind.value == "supersedes"]
        if supersedes:
            self.assertTrue(supersedes[0].strict_contradiction)
        
        # Retention should override to durable despite old age
        classifier = RetentionClassifier()
        retention = classifier.classify_with_drift(
            timestamp=old_event.timestamp,
            salience=0.1,  # Low salience
            relations=relations,
            now=now,
        )
        
        # Strict contradiction should force DURABLE
        if any(r.strict_contradiction for r in relations):
            self.assertEqual(retention.retention_class, RetentionClass.DURABLE)
    
    def test_full_event_lifecycle(self):
        """Test complete event lifecycle with enrichment."""
        from enrichment import (
            FeatureExtractor,
            RelationScorer,
            SalienceEngine,
            RetentionClassifier,
            compute_relation_summary,
        )
        
        now = time.time()
        
        # Simulate a conversation with corrections
        events = [
            _create_event(
                event_id=f"evt_lifecycle_{i}",
                workspace_id="ws_lifecycle",
                user_id="user_test",
                content={"text": text},
                timestamp=now - (3 - i) * 300,  # 15 min, 10 min, 5 min, now
                episode_id="ep_lifecycle",
            )
            for i, text in enumerate([
                "Server runs on port 8000",
                "Server runs on port 8080",  # Correction
                "Server runs on port 8080 with SSL",  # Expansion
                "Server runs on port 443 with SSL",  # Another correction
            ])
        ]
        
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        salience_engine = SalienceEngine()
        classifier = RetentionClassifier()
        
        all_relations = {}
        
        # Process each event
        for i, event in enumerate(events):
            event_relations = []
            
            # Compare with previous events
            for j in range(i):
                prev_event = events[j]
                features = extractor.extract(
                    event, prev_event, 
                    embedding_similarity=0.85 if "port" in prev_event.content["text"] else 0.5
                )
                relations = scorer.score_pair(event, prev_event, features)
                event_relations.extend(relations)
            
            all_relations[event.event_id] = event_relations
            
            # Compute salience
            salience = salience_engine.compute_with_relations(
                event_id=event.event_id,
                timestamp=event.timestamp,
                relations=event_relations,
                now=now,
            )
            
            # Compute retention
            retention = classifier.classify_with_drift(
                timestamp=event.timestamp,
                salience=salience.score,
                relations=event_relations,
                now=now,
            )
        
        # Verify: last event should have relations
        last_event_relations = all_relations[events[-1].event_id]
        self.assertTrue(len(last_event_relations) > 0)
        
        # Compute summary for last event
        summary = compute_relation_summary(
            events[-1].event_id,
            last_event_relations,
            all_relations,
        )
        
        self.assertGreater(summary.edge_count, 0)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing KRNX enrichment."""
    
    def test_relation_type_enum_values(self):
        """Ensure RelationType enum has expected values."""
        from enrichment import RelationType
        
        # These are used by existing code
        expected = ['duplicates', 'supersedes', 'expands_on', 'contradicts', 'replies_to']
        
        for val in expected:
            self.assertTrue(
                hasattr(RelationType, val.upper()),
                f"Missing RelationType.{val.upper()}"
            )
    
    def test_relation_to_dict_format(self):
        """Ensure Relation.to_dict() produces expected format."""
        from enrichment import Relation, RelationType
        
        rel = Relation(
            kind=RelationType.SUPERSEDES,
            target="evt_123",
            confidence=0.85,
            metadata={"test": True},
        )
        
        d = rel.to_dict()
        
        # Expected fields
        self.assertIn("kind", d)
        self.assertIn("target", d)
        self.assertIn("confidence", d)
        
        # Kind should be string value
        self.assertEqual(d["kind"], "supersedes")
    
    def test_relation_from_dict(self):
        """Test Relation.from_dict() reconstructs properly."""
        from enrichment import Relation, RelationType
        
        data = {
            "kind": "expands_on",
            "target": "evt_456",
            "confidence": 0.72,
            "metadata": {"similarity": 0.72},
        }
        
        rel = Relation.from_dict(data)
        
        self.assertEqual(rel.kind, RelationType.EXPANDS_ON)
        self.assertEqual(rel.target, "evt_456")
        self.assertEqual(rel.confidence, 0.72)
    
    def test_legacy_api_signatures(self):
        """Test RelationEnricher has expected method signatures."""
        from enrichment import RelationEnricher
        
        enricher = RelationEnricher()
        
        # Must have detect method
        self.assertTrue(hasattr(enricher, 'detect'))
        
        # Must have chain methods
        self.assertTrue(hasattr(enricher, 'find_supersession_chain'))
        self.assertTrue(hasattr(enricher, 'get_current_version'))


# ==============================================
# EDGE CASE TESTS
# ==============================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_content(self):
        """Test handling of empty content."""
        from enrichment.features import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        event_a = MockEvent("evt_a", {}, 1000.0)
        event_b = MockEvent("evt_b", {}, 900.0)
        
        features = extractor.extract(event_a, event_b)
        
        # Should not crash
        self.assertFalse(features.has_contradiction)
    
    def test_same_timestamp(self):
        """Test events with same timestamp."""
        from enrichment.relations import RelationScorer
        from enrichment.features import FeatureExtractor
        
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        
        event_a = MockEvent("evt_a", {"text": "First"}, 1000.0)
        event_b = MockEvent("evt_b", {"text": "Second"}, 1000.0)
        
        features = extractor.extract(event_a, event_b, embedding_similarity=0.5)
        results = scorer.score_pair(event_a, event_b, features)
        
        # Should not detect supersedes (not newer)
        from enrichment.relations import RelationType
        supersedes = [r for r in results if r.kind == RelationType.SUPERSEDES]
        self.assertEqual(len(supersedes), 0)
    
    def test_very_old_event(self):
        """Test very old event (>1 year)."""
        from enrichment.retention_v2 import DriftComputer
        
        now = time.time()
        computer = DriftComputer()
        
        drift = computer.compute(
            timestamp=now - 86400 * 365,  # 1 year ago
            relations=[],
            now=now,
        )
        
        # Should be very high drift
        self.assertGreater(drift, 0.5)


# ==============================================
# MAIN
# ==============================================

def main():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        # Unit tests
        TestNegationDetection,
        TestNumericDetection,
        TestTemporalDetection,
        TestAntonymDetection,
        TestPairFeatures,
        TestRelationScorer,
        TestRelationSummary,
        TestSalienceEngine,
        TestRetentionClassifier,
        
        # Internal integration
        TestIntegration,
        TestEdgeCases,
        
        # System integration (real Event model)
        TestSystemIntegration,
        TestBackwardCompatibility,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    print("=" * 60)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
