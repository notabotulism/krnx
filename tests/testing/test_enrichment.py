#!/usr/bin/env python3
"""
Test the metadata enrichment pipeline.

Usage:
    python3 test_enrichment.py
"""

import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, '/home/claude/chillbot')

from fabric.enrichment import (
    MetadataEnricher,
    EnrichmentConfig,
    EnrichmentContext,
    EntityExtractor,
    EntityType,
    RelationEnricher,
    RelationType,
)


# ==============================================
# MOCK EVENT FOR TESTING
# ==============================================

@dataclass
class MockEvent:
    """Mock event for testing."""
    event_id: str
    workspace_id: str
    user_id: str
    timestamp: float
    content: Dict[str, Any]
    metadata: Dict[str, Any] = None
    retention_class: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def test_temporal_enrichment():
    """Test temporal metadata (episode threading, time gaps)."""
    print("\n=== TEST: Temporal Enrichment ===")
    
    enricher = MetadataEnricher(config=EnrichmentConfig(
        episode_gap_threshold=300,  # 5 min
        enable_salience=False,
        enable_relations=False,
        enable_entities=False,
        enable_retention=False,
    ))
    
    now = time.time()
    
    # First event (no previous)
    event1 = MockEvent(
        event_id="evt_001",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 600,  # 10 min ago
        content={"text": "Hello world"},
    )
    
    result1 = enricher.enrich(event1, EnrichmentContext(now=now))
    print(f"Event 1: gap={result1.time_gap_seconds}, episode={result1.episode_id}")
    assert result1.time_gap_seconds is None  # No previous
    assert result1.episode_id is not None
    
    # Second event (same episode)
    event2 = MockEvent(
        event_id="evt_002",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 540,  # 9 min ago (1 min after event1)
        content={"text": "How are you?"},
    )
    event1.metadata["episode_id"] = result1.episode_id
    
    result2 = enricher.enrich(event2, EnrichmentContext(
        previous_event=event1,
        now=now,
    ))
    print(f"Event 2: gap={result2.time_gap_seconds}s, episode={result2.episode_id}")
    assert result2.time_gap_seconds == 60  # 1 min gap
    assert result2.episode_id == result1.episode_id  # Same episode
    
    # Third event (new episode - gap > threshold)
    event3 = MockEvent(
        event_id="evt_003",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 100,  # 100s ago (440s after event2 = 7+ min gap)
        content={"text": "Back after a break"},
    )
    event2.metadata["episode_id"] = result2.episode_id
    
    result3 = enricher.enrich(event3, EnrichmentContext(
        previous_event=event2,
        now=now,
    ))
    print(f"Event 3: gap={result3.time_gap_seconds}s, episode={result3.episode_id}")
    assert result3.time_gap_seconds == 440  # 7+ min gap
    assert result3.episode_id != result2.episode_id  # New episode!
    
    print("✓ Temporal enrichment working!")


def test_salience_scoring():
    """Test salience/importance scoring."""
    print("\n=== TEST: Salience Scoring ===")
    
    enricher = MetadataEnricher(config=EnrichmentConfig(
        recency_half_life_hours=24,
        weight_recency=0.4,
        weight_frequency=0.3,
        weight_semantic=0.3,
        enable_temporal=False,
        enable_relations=False,
        enable_entities=False,
        enable_retention=False,
    ))
    
    now = time.time()
    
    # Recent event, no access
    recent = MockEvent(
        event_id="evt_recent",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 3600,  # 1 hour ago
        content={"text": "Recent message"},
    )
    
    result_recent = enricher.enrich(recent, EnrichmentContext(
        access_count=0,
        now=now,
    ))
    print(f"Recent (1h, 0 access): salience={result_recent.salience_score:.3f}")
    print(f"  factors: {result_recent.salience_factors}")
    
    # Old event, high access
    old = MockEvent(
        event_id="evt_old",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 86400 * 7,  # 7 days ago
        content={"text": "Old but popular message"},
    )
    
    result_old = enricher.enrich(old, EnrichmentContext(
        access_count=50,
        now=now,
    ))
    print(f"Old (7d, 50 access): salience={result_old.salience_score:.3f}")
    print(f"  factors: {result_old.salience_factors}")
    
    # Recent should have higher recency factor
    assert result_recent.salience_factors["recency"] > result_old.salience_factors["recency"]
    
    # Old should have higher frequency factor
    assert result_old.salience_factors["frequency"] > result_recent.salience_factors["frequency"]
    
    print("✓ Salience scoring working!")


def test_entity_extraction():
    """Test entity extraction from content."""
    print("\n=== TEST: Entity Extraction ===")
    
    extractor = EntityExtractor()
    
    # Test various entity types
    content = {
        "text": """
        Hey @john, check out Project Alpha!
        The code is in /home/user/project/main.py
        See https://github.com/example/repo for details.
        This relates to #security and #performance.
        Created by user:alice in workspace:production.
        References evt_abc123 from earlier.
        The UserManager class handles this.
        """
    }
    
    entities = extractor.extract(content)
    
    print(f"Extracted {len(entities)} entities:")
    for e in entities:
        print(f"  {e.type.value}: {e.id} (raw: {e.raw})")
    
    # Verify we got expected types
    types = {e.type for e in entities}
    assert EntityType.USER in types, "Should find @john"
    assert EntityType.PROJECT in types, "Should find Project Alpha"
    assert EntityType.FILE in types, "Should find file path"
    assert EntityType.URL in types, "Should find URL"
    assert EntityType.CONCEPT in types, "Should find #hashtags"
    
    print("✓ Entity extraction working!")


def test_relation_detection():
    """Test relationship detection."""
    print("\n=== TEST: Relation Detection ===")
    
    enricher = RelationEnricher(
        duplicate_threshold=0.95,
        expand_threshold=0.70,
        supersede_threshold=0.90,
    )
    
    now = time.time()
    
    # Previous event
    prev_event = MockEvent(
        event_id="evt_001",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 60,  # 1 min ago
        content={"text": "Original message"},
    )
    
    # Current event with similarity scores
    current_id = "evt_002"
    similarity_scores = {
        "evt_001": 0.92,  # High similarity (supersedes candidate)
        "evt_old": 0.75,  # Moderate (expands_on)
        "evt_dup": 0.97,  # Very high (duplicate)
    }
    
    # Create workspace events for timestamp lookup
    workspace_events = [
        prev_event,
        MockEvent("evt_old", "test", "user1", now - 86400, {"text": "old"}),
        MockEvent("evt_dup", "test", "user1", now - 3600, {"text": "dup"}),
    ]
    
    relations = enricher.detect(
        event_id=current_id,
        timestamp=now,
        content={"text": "Updated message"},
        previous_event=prev_event,
        similarity_scores=similarity_scores,
        workspace_events=workspace_events,
    )
    
    print(f"Detected {len(relations)} relations:")
    for r in relations:
        print(f"  {r.kind.value} -> {r.target} (confidence: {r.confidence:.2f})")
    
    # Should have replies_to, supersedes, expands_on, duplicates
    kinds = {r.kind for r in relations}
    assert RelationType.REPLIES_TO in kinds, "Should detect replies_to"
    assert RelationType.DUPLICATES in kinds, "Should detect duplicate (0.97)"
    assert RelationType.SUPERSEDES in kinds, "Should detect supersedes (0.92 + newer)"
    assert RelationType.EXPANDS_ON in kinds, "Should detect expands_on (0.75)"
    
    print("✓ Relation detection working!")


def test_supersedes_chain():
    """Test supersession chain following."""
    print("\n=== TEST: Supersession Chain ===")
    
    from fabric.enrichment.relations import Relation, RelationType
    
    enricher = RelationEnricher()
    
    # Build a correction chain: evt_003 supersedes evt_002 supersedes evt_001
    events_relations = {
        "evt_003": [
            Relation(kind=RelationType.SUPERSEDES, target="evt_002", confidence=0.95),
        ],
        "evt_002": [
            Relation(kind=RelationType.SUPERSEDES, target="evt_001", confidence=0.92),
        ],
        "evt_001": [],  # Original
    }
    
    # Find chain from newest
    chain = enricher.find_supersession_chain("evt_003", events_relations)
    print(f"Chain from evt_003: {chain}")
    assert chain == ["evt_003", "evt_002", "evt_001"]
    
    # Find current version of original
    current = enricher.get_current_version("evt_001", events_relations)
    print(f"Current version of evt_001: {current}")
    assert current == "evt_003"
    
    print("✓ Supersession chain working!")


def test_full_pipeline():
    """Test the complete enrichment pipeline."""
    print("\n=== TEST: Full Pipeline ===")
    
    enricher = MetadataEnricher(config=EnrichmentConfig(
        episode_gap_threshold=300,
        recency_half_life_hours=24,
        enable_entity_extraction=True,
    ))
    
    now = time.time()
    
    prev_event = MockEvent(
        event_id="evt_prev",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 120,
        content={"text": "Previous message about Project Beta"},
        metadata={"episode_id": "ep_test123"},
    )
    
    current_event = MockEvent(
        event_id="evt_current",
        workspace_id="test",
        user_id="user1",
        timestamp=now - 60,
        content={
            "text": "Hey @bob, here's the update on Project Beta. Check #security issues.",
            "supersedes": "evt_old_info",
        },
    )
    
    context = EnrichmentContext(
        previous_event=prev_event,
        access_count=5,
        similarity_scores={
            "evt_prev": 0.85,
            "evt_other": 0.45,
        },
        workspace_events=[prev_event],
        now=now,
    )
    
    result = enricher.enrich(current_event, context)
    
    print("\nEnriched metadata:")
    print(f"  Temporal:")
    print(f"    - time_gap: {result.time_gap_seconds}s")
    print(f"    - episode: {result.episode_id}")
    print(f"  Salience:")
    print(f"    - score: {result.salience_score:.3f}")
    print(f"    - factors: {result.salience_factors}")
    print(f"  Relations ({len(result.relations)}):")
    for r in result.relations:
        print(f"    - {r}")
    print(f"  Entities ({len(result.entities)}):")
    for e in result.entities:
        print(f"    - {e}")
    print(f"  Retention:")
    print(f"    - class: {result.retention_class}")
    print(f"    - consolidation_candidate: {result.consolidation_candidate}")
    print(f"  Enrichment sources: {result.enrichment_sources}")
    
    # Verify all enrichment stages ran
    assert "temporal" in result.enrichment_sources
    assert "salience" in result.enrichment_sources
    assert "relations" in result.enrichment_sources
    assert "entities" in result.enrichment_sources
    assert "retention" in result.enrichment_sources
    
    # Verify we got entities
    assert len(result.entities) > 0, "Should extract entities"
    
    # Verify we got the explicit supersedes relation
    supersedes = [r for r in result.relations if r.get("kind") == "supersedes"]
    assert len(supersedes) > 0, "Should detect explicit supersedes"
    
    print("\n✓ Full pipeline working!")
    
    # Show the final dict format
    print("\n=== Final Metadata Dict ===")
    import json
    print(json.dumps(result.to_dict(), indent=2, default=str))


def main():
    """Run all tests."""
    print("=" * 60)
    print("KRNX Metadata Enrichment Tests")
    print("=" * 60)
    
    try:
        test_temporal_enrichment()
        test_salience_scoring()
        test_entity_extraction()
        test_relation_detection()
        test_supersedes_chain()
        test_full_pipeline()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
