"""
KRNX Unit Test Suite

Tests each component in isolation without external dependencies (Redis, filesystem).
Uses mocks and in-memory structures where necessary.

Run with: pytest tests/test_unit.py -v
"""

import pytest
import time
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch


# ==============================================
# SECTION 1: KERNEL MODELS
# ==============================================

class TestEvent:
    """Test kernel/models.py Event class"""
    
    def test_event_creation_basic(self):
        """Test basic event creation with required fields"""
        from chillbot.kernel.models import Event
        
        event = Event(
            event_id="evt_test123",
            workspace_id="workspace_1",
            user_id="user_1",
            session_id="session_1",
            content={"text": "Hello world"},
            timestamp=1700000000.0,
        )
        
        assert event.event_id == "evt_test123"
        assert event.workspace_id == "workspace_1"
        assert event.user_id == "user_1"
        assert event.content == {"text": "Hello world"}
        assert event.timestamp == 1700000000.0
    
    def test_event_creation_with_optional_fields(self):
        """Test event creation with all optional fields"""
        from chillbot.kernel.models import Event
        
        event = Event(
            event_id="evt_full",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"query": "test", "response": "answer"},
            timestamp=1700000000.0,
            channel="chat",
            ttl_seconds=3600,
            retention_class="standard",
            metadata={"model": "claude", "cost": 0.01},
            previous_hash="abc123",
        )
        
        assert event.channel == "chat"
        assert event.ttl_seconds == 3600
        assert event.retention_class == "standard"
        assert event.metadata["model"] == "claude"
        assert event.previous_hash == "abc123"
    
    def test_event_validation_empty_event_id(self):
        """Test that empty event_id raises ValueError"""
        from chillbot.kernel.models import Event
        
        with pytest.raises(ValueError, match="event_id is required"):
            Event(
                event_id="",
                workspace_id="ws",
                user_id="usr",
                session_id="sess",
                content={"text": "test"},
                timestamp=1700000000.0,
            )
    
    def test_event_validation_empty_workspace_id(self):
        """Test that empty workspace_id raises ValueError"""
        from chillbot.kernel.models import Event
        
        with pytest.raises(ValueError, match="workspace_id is required"):
            Event(
                event_id="evt_1",
                workspace_id="",
                user_id="usr",
                session_id="sess",
                content={"text": "test"},
                timestamp=1700000000.0,
            )
    
    def test_event_validation_invalid_timestamp(self):
        """Test that non-positive timestamp raises ValueError"""
        from chillbot.kernel.models import Event
        
        with pytest.raises(ValueError, match="timestamp must be positive"):
            Event(
                event_id="evt_1",
                workspace_id="ws",
                user_id="usr",
                session_id="sess",
                content={"text": "test"},
                timestamp=0,
            )
    
    def test_event_validation_invalid_content(self):
        """Test that non-dict content raises ValueError"""
        from chillbot.kernel.models import Event
        
        with pytest.raises(ValueError, match="content must be a dictionary"):
            Event(
                event_id="evt_1",
                workspace_id="ws",
                user_id="usr",
                session_id="sess",
                content="not a dict",
                timestamp=1700000000.0,
            )
    
    def test_event_to_dict(self):
        """Test Event.to_dict() serialization"""
        from chillbot.kernel.models import Event
        
        event = Event(
            event_id="evt_1",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "hello"},
            timestamp=1700000000.0,
            channel="chat",
        )
        
        d = event.to_dict()
        
        assert d["event_id"] == "evt_1"
        assert d["workspace_id"] == "ws"
        assert d["content"] == {"text": "hello"}
        assert d["channel"] == "chat"
    
    def test_event_to_json_from_json_roundtrip(self):
        """Test JSON serialization roundtrip"""
        from chillbot.kernel.models import Event
        
        original = Event(
            event_id="evt_roundtrip",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"query": "test", "nested": {"deep": True}},
            timestamp=1700000000.0,
            metadata={"key": "value"},
        )
        
        json_str = original.to_json()
        restored = Event.from_json(json_str)
        
        assert restored.event_id == original.event_id
        assert restored.content == original.content
        assert restored.metadata == original.metadata
        assert restored.timestamp == original.timestamp
    
    def test_event_from_dict_with_json_strings(self):
        """Test Event.from_dict() handles JSON string content from SQLite"""
        from chillbot.kernel.models import Event
        
        # Simulate SQLite row where content/metadata are JSON strings
        row_data = {
            "event_id": "evt_sql",
            "workspace_id": "ws",
            "user_id": "usr",
            "session_id": "sess",
            "content": '{"text": "from sqlite"}',  # JSON string
            "timestamp": 1700000000.0,
            "created_at": 1700000000.0,
            "metadata": '{"source": "test"}',  # JSON string
        }
        
        event = Event.from_dict(row_data)
        
        assert event.content == {"text": "from sqlite"}
        assert event.metadata == {"source": "test"}
    
    def test_event_age_calculations(self):
        """Test age calculation methods"""
        from chillbot.kernel.models import Event
        
        # Create event 1 day old
        one_day_ago = time.time() - 86400
        
        event = Event(
            event_id="evt_old",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "old"},
            timestamp=one_day_ago,
        )
        
        assert 86300 < event.get_age_seconds() < 86500  # ~1 day with tolerance
        assert 0.99 < event.get_age_days() < 1.01
        assert event.is_recent(hours=25) is True
        assert event.is_recent(hours=23) is False
    
    def test_event_ttl_expiration(self):
        """Test TTL expiration check"""
        from chillbot.kernel.models import Event
        
        # Expired event (TTL 1 second, created 2 seconds ago)
        expired = Event(
            event_id="evt_expired",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "expired"},
            timestamp=time.time() - 2,
            ttl_seconds=1,
        )
        
        # Fresh event (TTL 1 hour)
        fresh = Event(
            event_id="evt_fresh",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "fresh"},
            timestamp=time.time(),
            ttl_seconds=3600,
        )
        
        # No TTL event
        no_ttl = Event(
            event_id="evt_no_ttl",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "forever"},
            timestamp=time.time() - 1000000,
        )
        
        assert expired.is_expired() is True
        assert fresh.is_expired() is False
        assert no_ttl.is_expired() is False  # None TTL = never expires
    
    def test_event_hash_computation(self):
        """Test cryptographic hash computation"""
        from chillbot.kernel.models import Event
        
        event = Event(
            event_id="evt_hash",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "hash me"},
            timestamp=1700000000.0,
        )
        
        hash1 = event.compute_hash()
        hash2 = event.compute_hash()
        
        # Same event produces same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex
        
        # Different event produces different hash
        event2 = Event(
            event_id="evt_hash2",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "different"},
            timestamp=1700000000.0,
        )
        
        assert event.compute_hash() != event2.compute_hash()
    
    def test_event_hash_chain_verification(self):
        """Test hash-chain integrity verification"""
        from chillbot.kernel.models import Event
        
        # First event (no previous)
        event1 = Event(
            event_id="evt_chain_1",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "first"},
            timestamp=1700000000.0,
        )
        
        # Second event links to first
        event2 = Event(
            event_id="evt_chain_2",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "second"},
            timestamp=1700000001.0,
            previous_hash=event1.compute_hash(),
        )
        
        # Valid chain
        assert event1.verify_hash_chain(None) is True
        assert event2.verify_hash_chain(event1) is True
        
        # Invalid chain (wrong previous event)
        event3 = Event(
            event_id="evt_chain_3",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "third"},
            timestamp=1700000002.0,
        )
        assert event2.verify_hash_chain(event3) is False
    
    def test_event_equality_and_hash(self):
        """Test Event __eq__ and __hash__ for use in sets/dicts"""
        from chillbot.kernel.models import Event
        
        event1 = Event(
            event_id="evt_same",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "a"},
            timestamp=1700000000.0,
        )
        
        event2 = Event(
            event_id="evt_same",  # Same ID
            workspace_id="ws2",  # Different workspace
            user_id="usr2",
            session_id="sess2",
            content={"text": "b"},
            timestamp=1700000001.0,
        )
        
        event3 = Event(
            event_id="evt_different",
            workspace_id="ws",
            user_id="usr",
            session_id="sess",
            content={"text": "c"},
            timestamp=1700000000.0,
        )
        
        # Equality based on event_id only
        assert event1 == event2
        assert event1 != event3
        
        # Can be used in sets
        event_set = {event1, event2, event3}
        assert len(event_set) == 2  # event1 and event2 are same


class TestCreateEvent:
    """Test create_event convenience constructor"""
    
    def test_create_event_minimal(self):
        """Test create_event with minimal args"""
        from chillbot.kernel.models import create_event
        
        event = create_event(
            event_id="evt_1",
            workspace_id="ws",
            user_id="usr",
            content={"text": "hello"},
        )
        
        assert event.event_id == "evt_1"
        assert event.session_id == "ws_usr"  # Auto-generated
        assert event.timestamp > 0  # Auto-generated
    
    def test_create_event_with_metadata_kwargs(self):
        """Test create_event with metadata as kwargs"""
        from chillbot.kernel.models import create_event
        
        event = create_event(
            event_id="evt_meta",
            workspace_id="ws",
            user_id="usr",
            content={"query": "test"},
            channel="code",
            ttl_seconds=3600,
            retention_class="standard",
            model="claude-sonnet",
            cost=0.05,
            tokens=1000,
        )
        
        assert event.channel == "code"
        assert event.ttl_seconds == 3600
        assert event.metadata["model"] == "claude-sonnet"
        assert event.metadata["cost"] == 0.05
        assert event.metadata["tokens"] == 1000


class TestValidationHelpers:
    """Test validation helper functions"""
    
    def test_validate_event_id(self):
        """Test event_id validation"""
        from chillbot.kernel.models import validate_event_id
        
        assert validate_event_id("evt_123") is True
        assert validate_event_id("") is False
        assert validate_event_id("x" * 256) is False
        assert validate_event_id("x" * 255) is True
    
    def test_validate_workspace_id(self):
        """Test workspace_id validation"""
        from chillbot.kernel.models import validate_workspace_id
        
        assert validate_workspace_id("my-workspace") is True
        assert validate_workspace_id("") is False
    
    def test_validate_user_id(self):
        """Test user_id validation"""
        from chillbot.kernel.models import validate_user_id
        
        assert validate_user_id("user_123") is True
        assert validate_user_id("") is False


# ==============================================
# SECTION 2: ENRICHMENT SIGNALS (Lexicons)
# ==============================================

class TestNegationLexicon:
    """Test signals/loader.py NegationLexicon"""
    
    def test_negation_detection_basic(self):
        """Test basic negation detection"""
        from chillbot.fabric.enrichment.signals import get_negation_lexicon
        
        lexicon = get_negation_lexicon()
        
        assert lexicon.has_negation("This is not approved") is True
        assert lexicon.has_negation("I cannot do this") is True
        assert lexicon.has_negation("Never going to happen") is True
        assert lexicon.has_negation("This is approved") is False
        assert lexicon.has_negation("Yes we can") is False
    
    def test_negation_detection_contractions(self):
        """Test negation with contractions"""
        from chillbot.fabric.enrichment.signals import get_negation_lexicon
        
        lexicon = get_negation_lexicon()
        
        # Note: depends on tokenization - "doesn't" splits to "doesn" + "'t"
        # The lexicon has "n't" as a negator
        assert lexicon.has_negation("It doesn't work") is True
        assert lexicon.has_negation("They won't come") is True


class TestCancellationLexicon:
    """Test signals/loader.py CancellationLexicon"""
    
    def test_cancellation_verb_detection(self):
        """Test cancellation verb detection"""
        from chillbot.fabric.enrichment.signals import get_cancellation_lexicon
        
        lexicon = get_cancellation_lexicon()
        
        assert lexicon.has_cancellation("We need to cancel the meeting") is True
        assert lexicon.has_cancellation("The order was revoked") is True
        assert lexicon.has_cancellation("Please withdraw your request") is True
        assert lexicon.has_cancellation("The meeting is confirmed") is False
    
    def test_correction_phrase_detection(self):
        """Test correction phrase detection"""
        from chillbot.fabric.enrichment.signals import get_cancellation_lexicon
        
        lexicon = get_cancellation_lexicon()
        
        assert lexicon.has_correction("Actually, the budget is $50k") is True
        assert lexicon.has_correction("Correction: it should be 10") is True
        assert lexicon.has_correction("Scratch that, use the other one") is True
        assert lexicon.has_correction("The budget is $50k") is False


class TestAntonymLexicon:
    """Test signals/loader.py AntonymLexicon"""
    
    def test_antonym_lookup(self):
        """Test getting antonyms for a word"""
        from chillbot.fabric.enrichment.signals import get_antonym_lexicon
        
        lexicon = get_antonym_lexicon()
        
        approved_antonyms = lexicon.get_antonyms("approved")
        assert "rejected" in approved_antonyms
        
        started_antonyms = lexicon.get_antonyms("started")
        assert "stopped" in started_antonyms or "ended" in started_antonyms
    
    def test_antonym_pair_detection(self):
        """Test detecting antonym pairs between word sets"""
        from chillbot.fabric.enrichment.signals import get_antonym_lexicon
        
        lexicon = get_antonym_lexicon()
        
        # Has antonym pair
        assert lexicon.has_antonym_pair({"project", "approved"}, {"rejected", "today"}) is True
        assert lexicon.has_antonym_pair({"enabled", "feature"}, {"disabled", "flag"}) is True
        
        # No antonym pair
        assert lexicon.has_antonym_pair({"project", "budget"}, {"timeline", "meeting"}) is False
    
    def test_antonym_bidirectional(self):
        """Test that antonym lookup is bidirectional"""
        from chillbot.fabric.enrichment.signals import get_antonym_lexicon
        
        lexicon = get_antonym_lexicon()
        
        # If A is antonym of B, B should be antonym of A
        assert "rejected" in lexicon.get_antonyms("approved")
        assert "approved" in lexicon.get_antonyms("rejected")


class TestTemporalLexicon:
    """Test signals/loader.py TemporalLexicon"""
    
    def test_temporal_conflict_detection(self):
        """Test temporal conflict detection"""
        from chillbot.fabric.enrichment.signals import get_temporal_lexicon
        
        lexicon = get_temporal_lexicon()
        
        # Conflicting pairs
        assert lexicon.is_conflict("today", "tomorrow") is True
        assert lexicon.is_conflict("next week", "last week") is True
        assert lexicon.is_conflict("morning", "afternoon") is True
        
        # Non-conflicting
        assert lexicon.is_conflict("today", "today") is False
        assert lexicon.is_conflict("monday", "afternoon") is False


class TestReplyPatternLexicon:
    """Test signals/loader.py ReplyPatternLexicon"""
    
    def test_reply_pattern_detection(self):
        """Test reply pattern type detection"""
        from chillbot.fabric.enrichment.signals import get_reply_pattern_lexicon
        
        lexicon = get_reply_pattern_lexicon()
        
        assert lexicon.get_pattern_type("Thanks, got it!") == "acknowledgment"
        assert lexicon.get_pattern_type("Also, we need to discuss...") == "follow_up"
        assert lexicon.get_pattern_type("Actually, that's wrong") == "correction"
        assert lexicon.get_pattern_type("What time is the meeting?") == "question"
        assert lexicon.get_pattern_type("The project is ready.") is None


# ==============================================
# SECTION 3: ENRICHMENT FEATURES
# ==============================================

class TestFeatureExtraction:
    """Test fabric/enrichment/features.py"""
    
    def test_negation_mismatch_detection(self):
        """Test negation mismatch between texts"""
        from chillbot.fabric.enrichment.features import negation_mismatch
        
        assert negation_mismatch("Project is approved", "Project is not approved") is True
        assert negation_mismatch("User can access", "User cannot access") is True
        assert negation_mismatch("Project is approved", "Project is accepted") is False
        assert negation_mismatch("Not allowed", "Never permitted") is False  # Both negative
    
    def test_numeric_extraction(self):
        """Test numeric value extraction"""
        from chillbot.fabric.enrichment.features import extract_numerics
        
        # Currency
        nums = extract_numerics("Budget is $50,000")
        assert "$50,000" in nums
        
        # Percentages
        nums = extract_numerics("Growth is 15%")
        assert "15%" in nums
        
        # Times
        nums = extract_numerics("Meeting at 3:00pm")
        assert "3:00pm" in nums
        
        # Plain numbers
        nums = extract_numerics("We have 42 items")
        assert "42" in nums
    
    def test_numeric_mismatch_detection(self):
        """Test numeric mismatch between texts"""
        from chillbot.fabric.enrichment.features import numeric_mismatch
        
        assert numeric_mismatch("Budget is $50,000", "Budget is $75,000") is True
        assert numeric_mismatch("Meeting at 3pm", "Meeting at 4pm") is True
        assert numeric_mismatch("Budget is $50,000", "Budget is $50,000") is False
        assert numeric_mismatch("No numbers here", "None here either") is False
    
    def test_temporal_extraction(self):
        """Test temporal reference extraction"""
        from chillbot.fabric.enrichment.features import extract_temporal
        
        temp = extract_temporal("Meeting on Monday at 3:00pm next week")
        
        assert "monday" in temp.days
        assert "3:00pm" in temp.times
        assert "next week" in temp.relative
    
    def test_temporal_mismatch_detection(self):
        """Test temporal mismatch between texts"""
        from chillbot.fabric.enrichment.features import temporal_mismatch
        
        assert temporal_mismatch("Meeting today", "Meeting tomorrow") is True
        assert temporal_mismatch("Monday at 3pm", "Monday at 4pm") is True
        assert temporal_mismatch("Meeting today", "Meeting today") is False
    
    def test_antonym_detection_in_texts(self):
        """Test antonym detection between texts"""
        from chillbot.fabric.enrichment.features import antonym_detected
        
        assert antonym_detected("Project approved", "Project rejected") is True
        assert antonym_detected("Feature enabled", "Feature disabled") is True
        assert antonym_detected("Project approved", "Project completed") is False
    
    def test_entity_overlap_calculation(self):
        """Test entity overlap Jaccard similarity"""
        from chillbot.fabric.enrichment.features import entity_overlap
        
        entities_a = [{"id": "project_alpha"}, {"id": "user_bob"}]
        entities_b = [{"id": "project_alpha"}, {"id": "user_alice"}]
        
        overlap = entity_overlap(entities_a, entities_b)
        
        # Jaccard: 1 (alpha) / 3 (alpha, bob, alice) = 0.333
        assert 0.3 < overlap < 0.4
        
        # Perfect overlap
        assert entity_overlap(entities_a, entities_a) == 1.0
        
        # No overlap
        entities_c = [{"id": "other"}]
        assert entity_overlap(entities_a, entities_c) == 0.0
    
    def test_reply_pattern_detection(self):
        """Test reply pattern detection"""
        from chillbot.fabric.enrichment.features import detect_reply_pattern, is_reply_like
        
        assert detect_reply_pattern("Thanks!") == "acknowledgment"
        assert detect_reply_pattern("Actually, that's wrong") == "correction"
        assert is_reply_like("Got it, thanks!") is True
        assert is_reply_like("The project deadline is Friday.") is False


class TestPairFeatures:
    """Test PairFeatures dataclass"""
    
    def test_pair_features_contradiction_count(self):
        """Test contradiction count property"""
        from chillbot.fabric.enrichment.features import PairFeatures
        
        # No contradictions
        features = PairFeatures(embedding_similarity=0.8)
        assert features.contradiction_count == 0
        assert features.has_contradiction is False
        
        # Single contradiction
        features = PairFeatures(negation_mismatch=True)
        assert features.contradiction_count == 1
        assert features.has_contradiction is True
        assert features.strict_contradiction is False
        
        # Multiple contradictions (strict)
        features = PairFeatures(
            negation_mismatch=True,
            numeric_mismatch=True,
        )
        assert features.contradiction_count == 2
        assert features.strict_contradiction is True
    
    def test_pair_features_fired_signals(self):
        """Test getting fired signals"""
        from chillbot.fabric.enrichment.features import PairFeatures
        
        features = PairFeatures(
            negation_mismatch=True,
            numeric_mismatch=True,
            temporal_mismatch=False,
        )
        
        signals = features.get_fired_signals()
        
        assert "negation_mismatch" in signals
        assert "numeric_mismatch" in signals
        assert "temporal_mismatch" not in signals
    
    def test_pair_features_to_dict(self):
        """Test serialization to dict"""
        from chillbot.fabric.enrichment.features import PairFeatures
        
        features = PairFeatures(
            embedding_similarity=0.85,
            entity_overlap=0.5,
            negation_mismatch=True,
        )
        
        d = features.to_dict()
        
        assert d["embedding_similarity"] == 0.85
        assert d["entity_overlap"] == 0.5
        assert d["negation_mismatch"] is True
        assert d["contradiction_count"] == 1


class TestFeatureExtractor:
    """Test FeatureExtractor class"""
    
    def test_feature_extractor_basic(self):
        """Test feature extraction from event pair"""
        from chillbot.fabric.enrichment.features import FeatureExtractor
        
        # Create mock events
        @dataclass
        class MockEvent:
            event_id: str
            timestamp: float
            content: Dict[str, Any]
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        event_a = MockEvent(
            event_id="evt_1",
            timestamp=1700000001.0,
            content={"text": "Budget is $50,000"},
        )
        
        event_b = MockEvent(
            event_id="evt_2",
            timestamp=1700000000.0,
            content={"text": "Budget is $75,000"},
        )
        
        extractor = FeatureExtractor()
        features = extractor.extract(event_a, event_b, embedding_similarity=0.9)
        
        assert features.embedding_similarity == 0.9
        assert features.numeric_mismatch is True
        assert features.timestamp_delta == 1.0


# ==============================================
# SECTION 4: ENRICHMENT RELATIONS
# ==============================================

class TestRelationScorer:
    """Test fabric/enrichment/relations.py"""
    
    def test_relation_types(self):
        """Test RelationType enum values"""
        from chillbot.fabric.enrichment.relations import RelationType
        
        assert RelationType.DUPLICATES.value == "duplicates"
        assert RelationType.SUPERSEDES.value == "supersedes"
        assert RelationType.EXPANDS_ON.value == "expands_on"
        assert RelationType.CONTRADICTS.value == "contradicts"
        assert RelationType.REPLIES_TO.value == "replies_to"
    
    def test_relation_result_to_dict(self):
        """Test RelationResult serialization"""
        from chillbot.fabric.enrichment.relations import RelationResult, RelationType
        
        result = RelationResult(
            kind=RelationType.SUPERSEDES,
            target="evt_old",
            confidence=0.85,
            signals=["negation_mismatch", "newer_event"],
            reason_code="supersedes: negation_mismatch + newer_event",
            strict_contradiction=True,
        )
        
        d = result.to_dict()
        
        assert d["kind"] == "supersedes"
        assert d["target"] == "evt_old"
        assert d["confidence"] == 0.85
        assert "negation_mismatch" in d["signals"]
        assert d["strict_contradiction"] is True
    
    def test_scorer_duplicates_detection(self):
        """Test duplicate detection (high similarity, no contradiction)"""
        from chillbot.fabric.enrichment.relations import RelationScorer, RelationType
        from chillbot.fabric.enrichment.features import PairFeatures
        
        @dataclass
        class MockEvent:
            event_id: str
            timestamp: float
        
        new_event = MockEvent("evt_new", 1700000001.0)
        old_event = MockEvent("evt_old", 1700000000.0)
        
        features = PairFeatures(
            embedding_similarity=0.96,  # Above duplicate threshold
            negation_mismatch=False,
            numeric_mismatch=False,
        )
        
        scorer = RelationScorer()
        results = scorer.score_pair(new_event, old_event, features)
        
        # Should detect as duplicate
        duplicates = [r for r in results if r.kind == RelationType.DUPLICATES]
        assert len(duplicates) == 1
        assert duplicates[0].confidence >= 0.95
    
    def test_scorer_supersedes_detection(self):
        """Test supersedes detection (high similarity + contradiction + newer)"""
        from chillbot.fabric.enrichment.relations import RelationScorer, RelationType
        from chillbot.fabric.enrichment.features import PairFeatures
        
        @dataclass
        class MockEvent:
            event_id: str
            timestamp: float
        
        new_event = MockEvent("evt_new", 1700000001.0)
        old_event = MockEvent("evt_old", 1700000000.0)
        
        features = PairFeatures(
            embedding_similarity=0.85,
            negation_mismatch=True,  # Contradiction!
            numeric_mismatch=True,   # Double contradiction = strict
        )
        
        scorer = RelationScorer()
        results = scorer.score_pair(new_event, old_event, features)
        
        supersedes = [r for r in results if r.kind == RelationType.SUPERSEDES]
        assert len(supersedes) == 1
        assert supersedes[0].strict_contradiction is True
    
    def test_scorer_contradicts_detection(self):
        """Test contradicts detection (high similarity + contradiction + different actor)"""
        from chillbot.fabric.enrichment.relations import RelationScorer, RelationType
        from chillbot.fabric.enrichment.features import PairFeatures
        
        @dataclass
        class MockEvent:
            event_id: str
            timestamp: float
        
        new_event = MockEvent("evt_new", 1700000001.0)
        old_event = MockEvent("evt_old", 1700000000.0)
        
        features = PairFeatures(
            embedding_similarity=0.85,
            negation_mismatch=True,
            same_actor=False,  # Different actors
        )
        
        scorer = RelationScorer()
        results = scorer.score_pair(new_event, old_event, features)
        
        contradicts = [r for r in results if r.kind == RelationType.CONTRADICTS]
        assert len(contradicts) == 1
    
    def test_scorer_replies_to_detection(self):
        """Test replies_to detection (temporal proximity)"""
        from chillbot.fabric.enrichment.relations import RelationScorer, RelationType
        
        @dataclass
        class MockEvent:
            event_id: str
            timestamp: float
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        new_event = MockEvent("evt_new", 1700000010.0)  # 10 seconds later
        prev_event = MockEvent("evt_prev", 1700000000.0)
        
        scorer = RelationScorer()
        result = scorer.score_replies_to(new_event, prev_event)
        
        assert result is not None
        assert result.kind == RelationType.REPLIES_TO
        assert result.target == "evt_prev"


# ==============================================
# SECTION 5: ENRICHMENT SALIENCE
# ==============================================

class TestSalienceEngine:
    """Test fabric/enrichment/salience.py"""
    
    def test_salience_method_enum(self):
        """Test SalienceMethod enum"""
        from chillbot.fabric.enrichment.salience import SalienceMethod
        
        assert SalienceMethod.RECENCY.value == "recency"
        assert SalienceMethod.COMPOSITE.value == "composite"
    
    def test_salience_computation_composite(self):
        """Test composite salience computation"""
        from chillbot.fabric.enrichment.salience import SalienceEngine, SalienceMethod
        
        engine = SalienceEngine()
        now = time.time()
        
        result = engine.compute(
            event_id="evt_1",
            timestamp=now - 3600,  # 1 hour old
            access_count=5,
            avg_similarity=0.8,
            structural_score=0.6,
            method=SalienceMethod.COMPOSITE,
            now=now,
        )
        
        assert 0.0 <= result.score <= 1.0
        assert result.method == "composite"
        assert "recency" in result.factors
        assert "frequency" in result.factors
        assert "semantic" in result.factors
        assert "structural" in result.factors
    
    def test_salience_recency_decay(self):
        """Test recency decay over time"""
        from chillbot.fabric.enrichment.salience import SalienceEngine, SalienceMethod
        
        engine = SalienceEngine()
        now = time.time()
        
        # Recent event (just created)
        recent = engine.compute(
            event_id="evt_1",
            timestamp=now,
            method=SalienceMethod.RECENCY,
            now=now,
        )
        
        # Old event (1 day old)
        old = engine.compute(
            event_id="evt_2",
            timestamp=now - 86400,
            method=SalienceMethod.RECENCY,
            now=now,
        )
        
        # Recent should have higher recency score
        assert recent.score > old.score
    
    def test_salience_adjustment_from_relations(self):
        """Test salience adjustment from relations"""
        from chillbot.fabric.enrichment.salience import SalienceEngine, SalienceResult
        from chillbot.fabric.enrichment.relations import RelationResult, RelationType
        
        engine = SalienceEngine()
        
        base_salience = SalienceResult(
            score=0.5,
            base_score=0.5,
        )
        
        # Add supersedes relation (should boost)
        relations = [
            RelationResult(
                kind=RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["negation_mismatch"],
                reason_code="test",
                strict_contradiction=True,
            )
        ]
        
        adjusted = engine.adjust_from_relations(base_salience, relations)
        
        # Should be boosted
        assert adjusted.score > base_salience.score
        assert adjusted.relation_boost > 0
    
    def test_salience_spec_dict_output(self):
        """Test spec-compliant dictionary output"""
        from chillbot.fabric.enrichment.salience import SalienceEngine
        
        engine = SalienceEngine()
        
        result = engine.compute(
            event_id="evt_1",
            timestamp=time.time(),
            access_count=10,
            avg_similarity=0.7,
            structural_score=0.5,
        )
        
        spec = result.to_spec_dict()
        
        assert "semantic" in spec
        assert "recency" in spec
        assert "frequency" in spec
        assert "structural" in spec
        assert "final" in spec


# ==============================================
# SECTION 6: ENRICHMENT RETENTION
# ==============================================

class TestRetentionClassifier:
    """Test fabric/enrichment/retention_v2.py"""
    
    def test_retention_class_enum(self):
        """Test RetentionClass enum"""
        from chillbot.fabric.enrichment.retention_v2 import RetentionClass
        
        assert RetentionClass.EPHEMERAL.value == "ephemeral"
        assert RetentionClass.DURABLE.value == "durable"
        assert RetentionClass.MERGE_CANDIDATE.value == "merge_candidate"
    
    def test_drift_computation_time_based(self):
        """Test drift computation (time-based)"""
        from chillbot.fabric.enrichment.retention_v2 import DriftComputer
        
        computer = DriftComputer()
        now = time.time()
        
        # Recent event (low drift)
        recent_drift = computer.compute(
            timestamp=now - 3600,  # 1 hour ago
            relations=[],
            now=now,
        )
        
        # Old event (high drift)
        old_drift = computer.compute(
            timestamp=now - 86400 * 7,  # 7 days ago
            relations=[],
            now=now,
        )
        
        assert recent_drift < old_drift
        assert 0.0 <= recent_drift <= 1.0
        assert 0.0 <= old_drift <= 1.0
    
    def test_retention_classification_matrix(self):
        """Test drift × salience matrix classification"""
        from chillbot.fabric.enrichment.retention_v2 import RetentionClassifier, RetentionClass
        
        classifier = RetentionClassifier()
        
        # High drift + Low salience = EPHEMERAL
        result = classifier.classify(
            salience=0.1,
            drift=0.8,
            relations=[],
        )
        assert result.retention_class == RetentionClass.EPHEMERAL
        
        # Low drift + High salience = DURABLE
        result = classifier.classify(
            salience=0.8,
            drift=0.1,
            relations=[],
        )
        assert result.retention_class == RetentionClass.DURABLE
        
        # Low drift + Low salience = MERGE_CANDIDATE
        result = classifier.classify(
            salience=0.1,
            drift=0.1,
            relations=[],
        )
        assert result.retention_class == RetentionClass.MERGE_CANDIDATE
    
    def test_retention_strict_contradiction_override(self):
        """Test that strict contradiction overrides to DURABLE"""
        from chillbot.fabric.enrichment.retention_v2 import RetentionClassifier, RetentionClass
        from chillbot.fabric.enrichment.relations import RelationResult, RelationType
        
        classifier = RetentionClassifier()
        
        # Would normally be EPHEMERAL (high drift, low salience)
        # But strict_contradiction should override to DURABLE
        relations = [
            RelationResult(
                kind=RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["negation", "numeric"],
                reason_code="test",
                strict_contradiction=True,
            )
        ]
        
        result = classifier.classify(
            salience=0.1,
            drift=0.8,
            relations=relations,
        )
        
        assert result.retention_class == RetentionClass.DURABLE
        assert result.is_strict_contradiction is True


# ==============================================
# SECTION 7: ENRICHMENT STRUCTURAL
# ==============================================

class TestStructuralAnalysis:
    """Test fabric/enrichment/structural.py"""
    
    def test_density_computation(self):
        """Test event density computation"""
        from chillbot.fabric.enrichment.structural import DensityComputer
        
        @dataclass
        class MockEvent:
            timestamp: float
        
        computer = DensityComputer()
        now = time.time()
        
        # Create 5 events in the last minute
        recent_events = [MockEvent(now - i * 10) for i in range(5)]
        
        result = computer.compute(
            event_timestamp=now,
            recent_events=recent_events,
            window_seconds=60,
        )
        
        assert result.events_per_minute == 5.0
        assert result.density_class in ["high", "normal", "low"]
    
    def test_boundary_detection_time_gap(self):
        """Test episode boundary detection by time gap"""
        from chillbot.fabric.enrichment.structural import BoundaryDetector
        
        @dataclass
        class MockEvent:
            timestamp: float
        
        detector = BoundaryDetector()
        
        # Event 10 minutes after previous (> 5 min threshold)
        result = detector.detect(
            event_timestamp=1700000600.0,
            previous_event=MockEvent(1700000000.0),
        )
        
        assert result.is_boundary is True
        assert result.gap_seconds == 600.0
        
        # Event 1 minute after previous (< 5 min threshold)
        result = detector.detect(
            event_timestamp=1700000060.0,
            previous_event=MockEvent(1700000000.0),
        )
        
        assert result.is_boundary is False
    
    def test_boundary_detection_no_previous(self):
        """Test boundary detection with no previous event"""
        from chillbot.fabric.enrichment.structural import BoundaryDetector
        
        detector = BoundaryDetector()
        
        result = detector.detect(
            event_timestamp=1700000000.0,
            previous_event=None,
        )
        
        assert result.is_boundary is True
        assert result.reason == "no_previous_event"
    
    def test_structural_salience_computation(self):
        """Test structural salience scoring"""
        from chillbot.fabric.enrichment.structural import StructuralSalienceComputer
        
        computer = StructuralSalienceComputer()
        
        # Episode boundary (should boost)
        boundary_result = computer.compute(is_boundary=True)
        
        # Regular event
        regular_result = computer.compute(is_boundary=False)
        
        assert boundary_result.score > regular_result.score
        assert "first_in_episode" in boundary_result.factors
    
    def test_structural_analyzer_unified(self):
        """Test unified structural analyzer"""
        from chillbot.fabric.enrichment.structural import StructuralAnalyzer
        
        @dataclass
        class MockEvent:
            timestamp: float
        
        analyzer = StructuralAnalyzer()
        
        event = MockEvent(time.time())
        prev_event = MockEvent(time.time() - 60)
        recent_events = [MockEvent(time.time() - i * 10) for i in range(3)]
        
        analysis = analyzer.analyze(
            event=event,
            previous_event=prev_event,
            recent_events=recent_events,
        )
        
        assert analysis.density is not None
        assert analysis.boundary is not None
        assert analysis.salience is not None


# ==============================================
# SECTION 8: ENRICHMENT SCHEMA
# ==============================================

class TestEnrichmentSchema:
    """Test fabric/enrichment/schema.py"""
    
    def test_salience_output_to_dict(self):
        """Test SalienceOutput serialization"""
        from chillbot.fabric.enrichment.schema import SalienceOutput
        
        output = SalienceOutput(
            semantic=0.82,
            recency=0.40,
            frequency=0.10,
            structural=0.15,
            final=0.67,
        )
        
        d = output.to_dict()
        
        assert d["semantic"] == 0.82
        assert d["final"] == 0.67
    
    def test_relation_output_from_relation_result(self):
        """Test RelationOutput creation from internal RelationResult"""
        from chillbot.fabric.enrichment.schema import RelationOutput
        from chillbot.fabric.enrichment.relations import RelationResult, RelationType
        
        internal = RelationResult(
            kind=RelationType.SUPERSEDES,
            target="evt_old",
            confidence=0.85,
            signals=["negation_mismatch", "numeric_mismatch"],
            reason_code="test",
            strict_contradiction=True,
        )
        
        output = RelationOutput.from_relation_result(internal)
        
        assert output.type == "supersedes"
        assert output.target_event_id == "evt_old"
        assert "contradiction: negation" in output.signals
        assert output.reason_code == "UPDATE_NUMERIC"  # Generated from signals
    
    def test_temporal_output(self):
        """Test TemporalOutput"""
        from chillbot.fabric.enrichment.schema import TemporalOutput
        
        output = TemporalOutput(
            episode_id="ep_004",
            is_boundary=False,
            age_seconds=123400,
            drift_factor=0.31,
        )
        
        d = output.to_dict()
        
        assert d["episode_id"] == "ep_004"
        assert d["is_boundary"] is False
        assert d["drift_factor"] == 0.31
    
    def test_enriched_metadata_v2_spec_output(self):
        """Test EnrichedMetadataV2 spec-compliant output"""
        from chillbot.fabric.enrichment.schema import (
            EnrichedMetadataV2,
            SalienceOutput,
            RelationOutput,
            TemporalOutput,
        )
        
        metadata = EnrichedMetadataV2(
            salience=SalienceOutput(
                semantic=0.82,
                recency=0.40,
                frequency=0.10,
                structural=0.15,
                final=0.67,
            ),
            relations=[
                RelationOutput(
                    type="supersedes",
                    target_event_id="evt_2381",
                    confidence=0.78,
                    signals=["contradiction: numeric mismatch"],
                    reason_code="UPDATE_NUMERIC",
                )
            ],
            retention_class="durable",
            temporal=TemporalOutput(
                episode_id="ep_004",
                is_boundary=False,
                age_seconds=123400,
                drift_factor=0.31,
            ),
            confidence=0.92,
        )
        
        d = metadata.to_dict()
        
        # Check structure matches spec
        assert "salience" in d
        assert d["salience"]["final"] == 0.67
        assert "relations" in d
        assert len(d["relations"]) == 1
        assert d["relations"][0]["type"] == "supersedes"
        assert d["retention_class"] == "durable"
        assert d["temporal"]["episode_id"] == "ep_004"
        assert d["confidence"] == 0.92
    
    def test_metadata_builder(self):
        """Test MetadataBuilder fluent interface"""
        from chillbot.fabric.enrichment.schema import MetadataBuilder
        from chillbot.fabric.enrichment.relations import RelationResult, RelationType
        
        relations = [
            RelationResult(
                kind=RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["negation_mismatch"],
                reason_code="test",
                strict_contradiction=False,
            )
        ]
        
        metadata = (
            MetadataBuilder()
            .with_salience(semantic=0.8, recency=0.6, frequency=0.2, structural=0.5)
            .with_relations(relations)
            .with_retention("durable")
            .with_temporal(episode_id="ep_001", is_boundary=False, drift_factor=0.3)
            .with_confidence(0.95)
            .build()
        )
        
        assert metadata.salience.semantic == 0.8
        assert len(metadata.relations) == 1
        assert metadata.retention_class == "durable"
        assert metadata.confidence == 0.95


# ==============================================
# SECTION 9: FABRIC CONTEXT
# ==============================================

class TestContextBuilder:
    """Test fabric/context.py"""
    
    def test_context_config_defaults(self):
        """Test ContextConfig default values"""
        from chillbot.fabric.context import ContextConfig
        
        config = ContextConfig()
        
        assert config.max_tokens == 4000
        assert config.reserve_tokens == 500
        assert config.include_timestamps is True
    
    def test_context_builder_text_format(self):
        """Test building text context"""
        from chillbot.fabric.context import ContextBuilder
        
        @dataclass
        class MockMemory:
            event_id: str
            content: str
            timestamp: float
            score: float
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        builder = ContextBuilder(max_tokens=1000)
        
        memories = [
            MockMemory("evt_1", "User loves hiking", time.time() - 3600, 0.9),
            MockMemory("evt_2", "User mentioned dogs", time.time() - 7200, 0.8),
        ]
        
        context = builder.build(memories, query="outdoor activities", format="text")
        
        assert isinstance(context, str)
        assert "hiking" in context
        assert "outdoor activities" in context
    
    def test_context_builder_json_format(self):
        """Test building JSON context"""
        from chillbot.fabric.context import ContextBuilder
        
        @dataclass
        class MockMemory:
            event_id: str
            content: str
            timestamp: float
            score: float
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        builder = ContextBuilder(max_tokens=1000)
        
        memories = [
            MockMemory("evt_1", "User loves hiking", time.time(), 0.9),
        ]
        
        context = builder.build(memories, query="test", format="json")
        
        assert isinstance(context, dict)
        assert "query" in context
        assert "memories" in context
        assert context["count"] == 1
    
    def test_context_builder_messages_format(self):
        """Test building chat messages format"""
        from chillbot.fabric.context import ContextBuilder
        
        @dataclass
        class MockMemory:
            event_id: str
            content: str
            timestamp: float
            score: float
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        builder = ContextBuilder(max_tokens=1000)
        
        memories = [
            MockMemory("evt_1", "User loves hiking", time.time(), 0.9),
        ]
        
        messages = builder.build(memories, query="test", format="messages")
        
        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert messages[0]["role"] == "system"
    
    def test_context_builder_token_limiting(self):
        """Test that context respects token limits"""
        from chillbot.fabric.context import ContextBuilder
        
        @dataclass
        class MockMemory:
            event_id: str
            content: str
            timestamp: float
            score: float
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        builder = ContextBuilder(max_tokens=100)  # Very small limit
        
        # Create many memories with lots of content
        memories = [
            MockMemory(f"evt_{i}", "x" * 500, time.time() - i, 0.9 - i * 0.1)
            for i in range(10)
        ]
        
        context = builder.build(memories, query="test", format="text")
        
        # Should be truncated to respect token limit
        # 100 tokens ≈ 400 characters
        assert len(context) < 2000  # Well under 10 * 500 = 5000
    
    def test_context_estimate_size(self):
        """Test context size estimation"""
        from chillbot.fabric.context import ContextBuilder
        
        @dataclass
        class MockMemory:
            event_id: str
            content: str
            timestamp: float
            score: float
            metadata: Dict[str, Any] = None
            
            def __post_init__(self):
                if self.metadata is None:
                    self.metadata = {}
        
        builder = ContextBuilder(max_tokens=4000)
        
        memories = [
            MockMemory("evt_1", "Short content", time.time(), 0.9),
            MockMemory("evt_2", "Another short one", time.time(), 0.8),
        ]
        
        estimate = builder.estimate_context_size(memories)
        
        assert "total_tokens" in estimate
        assert "available_tokens" in estimate
        assert "fits" in estimate
        assert estimate["fits"] is True


# ==============================================
# SECTION 10: FABRIC IDENTITY
# ==============================================

class TestIdentityResolver:
    """Test fabric/identity.py"""
    
    def test_identity_types(self):
        """Test IdentityType enum"""
        from chillbot.fabric.identity import IdentityType
        
        assert IdentityType.USER.value == "user"
        assert IdentityType.AGENT.value == "agent"
        assert IdentityType.SERVICE.value == "service"
    
    def test_identity_creation(self):
        """Test Identity dataclass"""
        from chillbot.fabric.identity import Identity, IdentityType
        
        identity = Identity(
            id="user_123",
            type=IdentityType.USER,
            workspace_id="workspace_1",
            user_id="user_123",
            role="developer",
        )
        
        assert identity.id == "user_123"
        assert identity.workspace_id == "workspace_1"
        assert identity.role == "developer"
    
    def test_identity_scope_path(self):
        """Test Identity.scope_path property"""
        from chillbot.fabric.identity import Identity, IdentityType
        
        identity = Identity(
            id="user_123",
            type=IdentityType.USER,
            workspace_id="workspace_1",
            user_id="user_123",
            org_id="acme",
            project_id="alpha",
        )
        
        path = identity.scope_path
        
        assert "org:acme" in path
        assert "project:alpha" in path
        assert "workspace:workspace_1" in path
        assert "user:user_123" in path
    
    def test_resolver_agent_registration(self):
        """Test agent registration and resolution"""
        from chillbot.fabric.identity import IdentityResolver, IdentityType
        
        resolver = IdentityResolver()
        
        resolver.register_agent(
            agent_id="coder-1",
            workspace_id="project-x",
            role="coder",
            permissions=["read", "write"],
        )
        
        identity = resolver.resolve("coder-1")
        
        assert identity.type == IdentityType.AGENT
        assert identity.workspace_id == "project-x"
        assert identity.role == "coder"
        assert "read" in identity.permissions
    
    def test_resolver_scope_parsing(self):
        """Test scope path resolution"""
        from chillbot.fabric.identity import IdentityResolver
        
        resolver = IdentityResolver()
        
        identity = resolver.resolve_scope("org:acme/project:alpha/user:bob")
        
        assert identity.org_id == "acme"
        assert identity.project_id == "alpha"
        assert identity.user_id == "bob"
    
    def test_resolver_workspace_resolution(self):
        """Test workspace ID resolution from parts"""
        from chillbot.fabric.identity import IdentityResolver
        
        resolver = IdentityResolver(default_workspace="fallback")
        
        # Explicit workspace
        assert resolver.resolve_workspace(workspace_id="explicit") == "explicit"
        
        # From org + project
        assert resolver.resolve_workspace(org_id="acme", project_id="alpha") == "acme_alpha"
        
        # From project only
        assert resolver.resolve_workspace(project_id="alpha") == "alpha"
        
        # Fallback to default
        assert resolver.resolve_workspace() == "fallback"
    
    def test_parse_scope_function(self):
        """Test parse_scope helper function"""
        from chillbot.fabric.identity import parse_scope
        
        result = parse_scope("org:acme/project:alpha/user:bob/workspace:main")
        
        assert result["org"] == "acme"
        assert result["project"] == "alpha"
        assert result["user"] == "bob"
        assert result["workspace"] == "main"


# ==============================================
# SECTION 11: FABRIC RETENTION POLICIES
# ==============================================

class TestRetentionManager:
    """Test fabric/retention.py"""
    
    def test_retention_action_enum(self):
        """Test RetentionAction enum"""
        from chillbot.fabric.retention import RetentionAction
        
        assert RetentionAction.KEEP.value == "keep"
        assert RetentionAction.DELETE.value == "delete"
        assert RetentionAction.ARCHIVE.value == "archive"
    
    def test_retention_class_enum(self):
        """Test RetentionClass enum"""
        from chillbot.fabric.retention import RetentionClass
        
        assert RetentionClass.EPHEMERAL.value == "ephemeral"
        assert RetentionClass.STANDARD.value == "standard"
        assert RetentionClass.PERMANENT.value == "permanent"
    
    def test_retention_policy_creation(self):
        """Test RetentionPolicy creation"""
        from chillbot.fabric.retention import RetentionPolicy, RetentionAction
        
        policy = RetentionPolicy(
            name="test_policy",
            ttl_seconds=3600,
            action=RetentionAction.DELETE,
            priority=100,
        )
        
        assert policy.name == "test_policy"
        assert policy.ttl_seconds == 3600
        assert policy.action == RetentionAction.DELETE
    
    def test_retention_policy_matches(self):
        """Test policy matching logic"""
        from chillbot.fabric.retention import RetentionPolicy, RetentionClass, RetentionAction
        
        @dataclass
        class MockMemory:
            retention_class: str
            channel: str
        
        policy = RetentionPolicy(
            name="ephemeral_cleanup",
            retention_class=RetentionClass.EPHEMERAL,
            action=RetentionAction.DELETE,
        )
        
        ephemeral_memory = MockMemory(retention_class="ephemeral", channel="chat")
        standard_memory = MockMemory(retention_class="standard", channel="chat")
        
        assert policy.matches(ephemeral_memory) is True
        assert policy.matches(standard_memory) is False
    
    def test_retention_policy_should_trigger(self):
        """Test TTL trigger logic"""
        from chillbot.fabric.retention import RetentionPolicy, RetentionAction
        
        @dataclass
        class MockMemory:
            timestamp: float
        
        policy = RetentionPolicy(
            name="quick_expire",
            ttl_seconds=60,  # 1 minute
            action=RetentionAction.DELETE,
        )
        
        now = time.time()
        
        # Old memory (should trigger)
        old = MockMemory(timestamp=now - 120)  # 2 minutes old
        assert policy.should_trigger(old, now) is True
        
        # Fresh memory (should not trigger)
        fresh = MockMemory(timestamp=now - 30)  # 30 seconds old
        assert policy.should_trigger(fresh, now) is False
    
    def test_retention_manager_default_policies(self):
        """Test RetentionManager has default policies"""
        from chillbot.fabric.retention import RetentionManager
        
        manager = RetentionManager()
        policies = manager.list_policies()
        
        policy_names = [p.name for p in policies]
        
        assert "ephemeral_cleanup" in policy_names
        assert "standard_archive" in policy_names
        assert "permanent_keep" in policy_names
    
    def test_retention_manager_evaluate(self):
        """Test memory evaluation against policies"""
        from chillbot.fabric.retention import RetentionManager, RetentionAction
        
        @dataclass
        class MockMemory:
            event_id: str
            timestamp: float
            retention_class: str = None
            ttl_seconds: int = None
        
        manager = RetentionManager()
        now = time.time()
        
        # Ephemeral, old memory should be deleted
        ephemeral_old = MockMemory(
            event_id="evt_1",
            timestamp=now - 7200,  # 2 hours old
            retention_class="ephemeral",
        )
        
        evaluation = manager.evaluate(ephemeral_old, now)
        assert evaluation.action == RetentionAction.DELETE
        
        # Permanent memory should be kept
        permanent = MockMemory(
            event_id="evt_2",
            timestamp=now - 86400 * 365,  # 1 year old
            retention_class="permanent",
        )
        
        evaluation = manager.evaluate(permanent, now)
        assert evaluation.action == RetentionAction.KEEP
    
    def test_retention_manager_memory_ttl_override(self):
        """Test that memory's own TTL takes precedence"""
        from chillbot.fabric.retention import RetentionManager, RetentionAction
        
        @dataclass
        class MockMemory:
            event_id: str
            timestamp: float
            ttl_seconds: int
        
        manager = RetentionManager()
        now = time.time()
        
        # Memory with 1-second TTL, 2 seconds old
        expired = MockMemory(
            event_id="evt_ttl",
            timestamp=now - 2,
            ttl_seconds=1,
        )
        
        evaluation = manager.evaluate(expired, now)
        assert evaluation.action == RetentionAction.DELETE
        assert evaluation.policy_name == "memory_ttl"
    
    def test_convenience_policy_creators(self):
        """Test create_ttl_policy and create_archive_policy"""
        from chillbot.fabric.retention import (
            create_ttl_policy,
            create_archive_policy,
            RetentionAction,
        )
        
        ttl_policy = create_ttl_policy("quick", ttl_seconds=60, channels=["temp"])
        
        assert ttl_policy.name == "quick"
        assert ttl_policy.ttl_seconds == 60
        assert ttl_policy.action == RetentionAction.DELETE
        assert "temp" in ttl_policy.channels
        
        archive_policy = create_archive_policy("cold", days=30)
        
        assert archive_policy.archive_after_days == 30
        assert archive_policy.action == RetentionAction.ARCHIVE


# ==============================================
# SECTION 12: ENTITY EXTRACTOR
# ==============================================

class TestEntityExtractor:
    """Test fabric/enrichment/entities.py"""
    
    def test_entity_types(self):
        """Test EntityType enum"""
        from chillbot.fabric.enrichment.entities import EntityType
        
        assert EntityType.USER.value == "user"
        assert EntityType.PROJECT.value == "project"
        assert EntityType.CODE.value == "code"
        assert EntityType.CONCEPT.value == "concept"
    
    def test_extract_user_mentions(self):
        """Test @mention extraction"""
        from chillbot.fabric.enrichment.entities import EntityExtractor, EntityType
        
        extractor = EntityExtractor()
        
        entities = extractor.extract("Hey @alice and @bob, please review")
        
        users = [e for e in entities if e.type == EntityType.USER]
        user_ids = [u.id for u in users]
        
        assert "alice" in user_ids
        assert "bob" in user_ids
    
    def test_extract_project_references(self):
        """Test project reference extraction"""
        from chillbot.fabric.enrichment.entities import EntityExtractor, EntityType
        
        extractor = EntityExtractor()
        
        entities = extractor.extract("Working on project:alpha today")
        
        projects = [e for e in entities if e.type == EntityType.PROJECT]
        assert any(p.id == "alpha" for p in projects)
    
    def test_extract_file_paths(self):
        """Test file path extraction"""
        from chillbot.fabric.enrichment.entities import EntityExtractor, EntityType
        
        extractor = EntityExtractor()
        
        entities = extractor.extract("Edit /src/main.py please")
        
        files = [e for e in entities if e.type == EntityType.FILE]
        assert len(files) >= 1
    
    def test_extract_urls(self):
        """Test URL extraction"""
        from chillbot.fabric.enrichment.entities import EntityExtractor, EntityType
        
        extractor = EntityExtractor()
        
        entities = extractor.extract("Check https://example.com/api/docs")
        
        urls = [e for e in entities if e.type == EntityType.URL]
        assert len(urls) == 1
        assert "example.com" in urls[0].id
    
    def test_extract_hashtags(self):
        """Test #hashtag extraction"""
        from chillbot.fabric.enrichment.entities import EntityExtractor, EntityType
        
        extractor = EntityExtractor()
        
        entities = extractor.extract("This is #important and #urgent")
        
        concepts = [e for e in entities if e.type == EntityType.CONCEPT]
        concept_ids = [c.id for c in concepts]
        
        assert "important" in concept_ids
        assert "urgent" in concept_ids
    
    def test_extract_code_identifiers(self):
        """Test code identifier extraction"""
        from chillbot.fabric.enrichment.entities import EntityExtractor, EntityType
        
        extractor = EntityExtractor()
        
        entities = extractor.extract("def process_data() in DataService class")
        
        code = [e for e in entities if e.type == EntityType.CODE]
        code_ids = [c.id for c in code]
        
        assert "process_data" in code_ids
        assert "DataService" in code_ids
    
    def test_extract_typed_convenience(self):
        """Test extract_users/projects/concepts convenience methods"""
        from chillbot.fabric.enrichment.entities import EntityExtractor
        
        extractor = EntityExtractor()
        
        text = "@alice working on project:alpha #review"
        
        users = extractor.extract_users(text)
        projects = extractor.extract_projects(text)
        concepts = extractor.extract_concepts(text)
        
        assert "alice" in users
        assert "alpha" in projects
        assert "review" in concepts
    
    def test_entity_to_dict(self):
        """Test Entity.to_dict()"""
        from chillbot.fabric.enrichment.entities import Entity, EntityType
        
        entity = Entity(
            type=EntityType.USER,
            id="alice",
            raw="@alice",
            confidence=0.9,
        )
        
        d = entity.to_dict()
        
        assert d["type"] == "user"
        assert d["id"] == "alice"
        assert d["raw"] == "@alice"
        assert d["confidence"] == 0.9


# ==============================================
# SECTION 13: RELATION SUMMARY
# ==============================================

class TestRelationSummary:
    """Test fabric/enrichment/summary.py"""
    
    def test_relation_summary_basic(self):
        """Test RelationSummary creation"""
        from chillbot.fabric.enrichment.summary import RelationSummary
        
        summary = RelationSummary(
            edge_count=3,
            by_kind={"supersedes": 1, "expands_on": 2},
            chain_depth=2,
            is_leaf=True,
            is_root=False,
        )
        
        assert summary.edge_count == 3
        assert summary.chain_depth == 2
        assert summary.is_leaf is True
    
    def test_relation_summary_to_dict(self):
        """Test RelationSummary serialization"""
        from chillbot.fabric.enrichment.summary import RelationSummary
        
        summary = RelationSummary(
            edge_count=1,
            by_kind={"supersedes": 1},
            has_supersedes=True,
            has_strict_contradiction=True,
        )
        
        d = summary.to_dict()
        
        assert d["edge_count"] == 1
        assert d["has_supersedes"] is True
        assert d["has_strict_contradiction"] is True
    
    def test_relation_summary_computer_basic(self):
        """Test RelationSummaryComputer"""
        from chillbot.fabric.enrichment.summary import RelationSummaryComputer
        from chillbot.fabric.enrichment.relations import RelationResult, RelationType
        
        computer = RelationSummaryComputer()
        
        relations = [
            RelationResult(
                kind=RelationType.SUPERSEDES,
                target="evt_old",
                confidence=0.9,
                signals=["negation_mismatch", "numeric_mismatch"],
                reason_code="test",
                strict_contradiction=True,
            ),
            RelationResult(
                kind=RelationType.EXPANDS_ON,
                target="evt_related",
                confidence=0.7,
                signals=[],
                reason_code="test",
                strict_contradiction=False,
            ),
        ]
        
        summary = computer.compute(
            event_id="evt_new",
            relations=relations,
        )
        
        assert summary.edge_count == 2
        assert summary.by_kind.get("supersedes", 0) == 1
        assert summary.by_kind.get("expands_on", 0) == 1
        assert summary.has_supersedes is True
        assert summary.has_strict_contradiction is True


# ==============================================
# RUN TESTS
# ==============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
