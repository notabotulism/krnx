"""
KRNX Enrichment - Quick Validation Test

Run with: python test_enrichment_short.py

Tests the core multi-signal relation scoring functionality.
"""

import os
import sys

# Add project root to path (works from D:\chillbot\chillbot)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add fabric/ so 'from enrichment import ...' works
fabric_path = os.path.join(project_root, 'fabric')
if os.path.exists(fabric_path):
    sys.path.insert(0, fabric_path)

from dataclasses import dataclass
from typing import Dict, Any, Optional


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
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ==============================================
# TEST: FEATURE EXTRACTION
# ==============================================

def test_negation_detection():
    """Test negation mismatch detection."""
    from enrichment.features import has_negation, negation_mismatch
    
    # Test has_negation
    assert has_negation("The project is not approved") == True
    assert has_negation("The project is approved") == False
    assert has_negation("I can't do this") == True
    assert has_negation("I can do this") == False
    
    # Test mismatch
    assert negation_mismatch("Project approved", "Project not approved") == True
    assert negation_mismatch("Project approved", "Project confirmed") == False
    
    print("✓ Negation detection works")


def test_numeric_detection():
    """Test numeric mismatch detection."""
    from enrichment.features import extract_numerics, numeric_mismatch
    
    # Test extraction
    nums = extract_numerics("Meeting at 3:00pm for $50,000 budget")
    assert "3:00pm" in nums or "3:00 pm" in nums.union({"3:00pm"})
    assert "$50,000" in nums
    
    # Test mismatch
    assert numeric_mismatch("Budget is $50,000", "Budget is $75,000") == True
    assert numeric_mismatch("Meeting at 3pm", "Meeting at 4pm") == True
    assert numeric_mismatch("Budget is $50,000", "Budget is $50,000") == False
    
    print("✓ Numeric detection works")


def test_temporal_detection():
    """Test temporal mismatch detection."""
    from enrichment.features import temporal_mismatch
    
    # Same day, different times
    assert temporal_mismatch(
        "Meeting Tuesday at 3pm",
        "Meeting Tuesday at 4pm"
    ) == True
    
    # Conflicting relative terms
    assert temporal_mismatch(
        "Let's meet today",
        "Let's meet tomorrow"
    ) == True
    
    # No conflict
    assert temporal_mismatch(
        "Meeting on Monday",
        "Meeting on Monday"
    ) == False
    
    print("✓ Temporal detection works")


def test_antonym_detection():
    """Test antonym detection."""
    from enrichment.features import antonym_detected
    
    assert antonym_detected("Project approved", "Project rejected") == True
    assert antonym_detected("User joined", "User left") == True
    assert antonym_detected("Task completed", "Task started") == False
    
    print("✓ Antonym detection works")


def test_pair_features():
    """Test full PairFeatures extraction."""
    from enrichment.features import FeatureExtractor, PairFeatures
    
    event_a = MockEvent(
        event_id="evt_001",
        content={"text": "Meeting is Tuesday at 3pm"},
        timestamp=1000.0,
    )
    
    event_b = MockEvent(
        event_id="evt_002",
        content={"text": "Meeting is Tuesday at 4pm"},
        timestamp=900.0,
    )
    
    extractor = FeatureExtractor()
    features = extractor.extract(event_a, event_b, embedding_similarity=0.94)
    
    assert isinstance(features, PairFeatures)
    assert features.embedding_similarity == 0.94
    assert features.temporal_mismatch == True  # Different times on same day
    assert features.has_contradiction == True
    assert features.timestamp_delta == 100.0
    
    print("✓ PairFeatures extraction works")


# ==============================================
# TEST: RELATION SCORING
# ==============================================

def test_relation_scoring():
    """Test multi-signal relation scoring."""
    from enrichment.features import FeatureExtractor
    from enrichment.relations import RelationScorer, RelationType
    
    # Event that supersedes another (correction)
    new_event = MockEvent(
        event_id="evt_new",
        content={"text": "Meeting moved to 4pm"},
        timestamp=2000.0,
    )
    
    old_event = MockEvent(
        event_id="evt_old",
        content={"text": "Meeting is at 3pm"},
        timestamp=1000.0,
    )
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(new_event, old_event, embedding_similarity=0.88)
    
    # Score relations
    scorer = RelationScorer()
    results = scorer.score_pair(new_event, old_event, features)
    
    # Should detect supersedes (high similarity + contradiction + newer)
    supersedes = [r for r in results if r.kind == RelationType.SUPERSEDES]
    assert len(supersedes) > 0, "Should detect supersedes relation"
    
    result = supersedes[0]
    assert result.target == "evt_old"
    assert "temporal_mismatch" in result.signals or "numeric_mismatch" in result.signals
    assert "supersedes:" in result.reason_code
    
    print("✓ Relation scoring works")
    print(f"  Detected: {result.kind.value} with confidence {result.confidence:.3f}")
    print(f"  Signals: {result.signals}")
    print(f"  Reason: {result.reason_code}")


def test_contradiction_detection():
    """Test that contradictions are properly detected."""
    from enrichment.features import FeatureExtractor
    from enrichment.relations import RelationScorer, RelationType
    
    # Two users giving conflicting information
    event_a = MockEvent(
        event_id="evt_alice",
        content={"text": "Project is approved"},
        timestamp=1000.0,
        user_id="alice",
    )
    
    event_b = MockEvent(
        event_id="evt_bob",
        content={"text": "Project is not approved"},
        timestamp=1100.0,
        user_id="bob",
    )
    
    extractor = FeatureExtractor()
    features = extractor.extract(event_b, event_a, embedding_similarity=0.91)
    
    # Should have negation mismatch
    assert features.negation_mismatch == True
    assert features.has_contradiction == True
    assert features.same_actor == False
    
    scorer = RelationScorer()
    results = scorer.score_pair(event_b, event_a, features)
    
    # Should detect contradicts (different actors)
    contradicts = [r for r in results if r.kind == RelationType.CONTRADICTS]
    assert len(contradicts) > 0, "Should detect contradicts relation"
    
    print("✓ Contradiction detection works")


# ==============================================
# TEST: SALIENCE ADJUSTMENT
# ==============================================

def test_salience_adjustment():
    """Test salience adjustment from relations."""
    from enrichment.salience import SalienceEngine, adjust_salience_from_relations
    from enrichment.relations import RelationResult, RelationType
    
    # Create mock relations
    relations = [
        RelationResult(
            kind=RelationType.SUPERSEDES,
            target="evt_old",
            confidence=0.85,
            signals=["temporal_mismatch"],
            reason_code="supersedes: temporal_mismatch + newer_event",
            strict_contradiction=False,
        ),
    ]
    
    # Base salience
    base_salience = 0.5
    
    # Adjust
    adjusted = adjust_salience_from_relations(base_salience, relations)
    
    # Should be boosted
    assert adjusted > base_salience
    assert adjusted == base_salience + 0.10  # supersedes_boost
    
    print("✓ Salience adjustment works")
    print(f"  Base: {base_salience} → Adjusted: {adjusted}")


# ==============================================
# TEST: RETENTION CLASSIFICATION
# ==============================================

def test_retention_classification():
    """Test drift × salience retention matrix."""
    from enrichment.retention_v2 import RetentionClassifier, RetentionClass
    from enrichment.relations import RelationResult, RelationType
    import time
    
    classifier = RetentionClassifier()
    now = time.time()
    
    # Test: High salience + low drift → durable
    result = classifier.classify_with_drift(
        timestamp=now - 3600,  # 1 hour ago
        salience=0.8,
        relations=[],
        now=now,
    )
    # Recent + high salience should be durable or consolidation_candidate
    assert result.retention_class in [RetentionClass.DURABLE, RetentionClass.CONSOLIDATION_CANDIDATE]
    
    # Test: Low salience + high drift → ephemeral
    result = classifier.classify_with_drift(
        timestamp=now - 86400 * 30,  # 30 days ago (ensures high drift > 0.6)
        salience=0.1,
        relations=[],
        now=now,
    )
    assert result.retention_class == RetentionClass.EPHEMERAL, f"Expected EPHEMERAL, got {result.retention_class} (drift={result.drift})"
    
    # Test: Strict contradiction override → durable
    relations = [
        RelationResult(
            kind=RelationType.SUPERSEDES,
            target="evt_old",
            confidence=0.9,
            signals=["negation_mismatch", "numeric_mismatch"],
            reason_code="supersedes: negation_mismatch + numeric_mismatch",
            strict_contradiction=True,  # 2+ signals
        ),
    ]
    
    result = classifier.classify_with_drift(
        timestamp=now - 86400 * 30,  # Old event
        salience=0.1,  # Low salience
        relations=relations,
        now=now,
    )
    # Strict contradiction should override to durable
    assert result.retention_class == RetentionClass.DURABLE
    assert result.is_strict_contradiction == True
    
    print("✓ Retention classification works")


# ==============================================
# TEST: RELATION SUMMARY
# ==============================================

def test_relation_summary():
    """Test relation graph summary computation."""
    from enrichment.summary import RelationSummaryComputer, RelationSummary
    from enrichment.relations import RelationResult, RelationType
    
    computer = RelationSummaryComputer()
    
    relations = [
        RelationResult(
            kind=RelationType.SUPERSEDES,
            target="evt_old1",
            confidence=0.9,
            signals=["temporal_mismatch"],
            reason_code="supersedes: temporal_mismatch",
            strict_contradiction=False,
        ),
        RelationResult(
            kind=RelationType.EXPANDS_ON,
            target="evt_related",
            confidence=0.7,
            signals=["entity_overlap"],
            reason_code="expands_on: entity overlap",
            strict_contradiction=False,
        ),
    ]
    
    summary = computer.compute("evt_new", relations)
    
    assert isinstance(summary, RelationSummary)
    assert summary.edge_count == 2
    assert summary.by_kind["supersedes"] == 1
    assert summary.by_kind["expands_on"] == 1
    assert summary.has_supersedes == True
    assert summary.is_root == False  # Has supersedes relation
    
    print("✓ Relation summary works")
    print(f"  Edges: {summary.edge_count}, By kind: {summary.by_kind}")


# ==============================================
# MAIN
# ==============================================

def main():
    """Run all quick tests."""
    print("\n" + "="*60)
    print("KRNX Multi-Signal Relation Scoring - Quick Tests")
    print("="*60 + "\n")
    
    tests = [
        test_negation_detection,
        test_numeric_detection,
        test_temporal_detection,
        test_antonym_detection,
        test_pair_features,
        test_relation_scoring,
        test_contradiction_detection,
        test_salience_adjustment,
        test_retention_classification,
        test_relation_summary,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
