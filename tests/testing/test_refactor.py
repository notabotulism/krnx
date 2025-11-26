"""
KRNX Enrichment v3.1 - Refactor Test

Tests the refactored enrichment module with:
- External JSON lexicons
- Spec-compliant output schema
- Structural signals
"""

import sys
import os
import time

# Add parent dir to path so we can import as a package
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)

# Rename this directory temporarily for import
_pkg_name = "enrichment"
if os.path.basename(_this_dir) != _pkg_name:
    # We're in krnx_refactor, rename for testing
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enrichment",
        os.path.join(_this_dir, "__init__.py"),
        submodule_search_locations=[_this_dir]
    )
    enrichment = importlib.util.module_from_spec(spec)
    sys.modules["enrichment"] = enrichment
    
    # Also set up submodules
    for submod in ["signals", "features", "relations", "structural", "salience", 
                   "schema", "retention_v2", "summary", "cross_encoder", "temporal", "entities"]:
        subpath = os.path.join(_this_dir, f"{submod}.py")
        if os.path.exists(subpath):
            subspec = importlib.util.spec_from_file_location(
                f"enrichment.{submod}", subpath,
                submodule_search_locations=[_this_dir]
            )
            submodule = importlib.util.module_from_spec(subspec)
            sys.modules[f"enrichment.{submod}"] = submodule
    
    # Handle signals subpackage
    signals_dir = os.path.join(_this_dir, "signals")
    if os.path.isdir(signals_dir):
        sigspec = importlib.util.spec_from_file_location(
            "enrichment.signals",
            os.path.join(signals_dir, "__init__.py"),
            submodule_search_locations=[signals_dir]
        )
        signals_mod = importlib.util.module_from_spec(sigspec)
        sys.modules["enrichment.signals"] = signals_mod


def test_lexicon_loading():
    """Test external lexicons load correctly."""
    print("\n=== Test: Lexicon Loading ===")
    
    from signals import (
        get_negation_lexicon,
        get_antonym_lexicon,
        get_temporal_lexicon,
        get_reply_pattern_lexicon,
    )
    
    # Negation
    neg = get_negation_lexicon()
    assert neg.has_negation("I do not agree")
    assert not neg.has_negation("I agree")
    print("  ✓ Negation lexicon works")
    
    # Antonyms
    ant = get_antonym_lexicon()
    assert ant.has_antonym_pair({"approved"}, {"rejected"})
    assert not ant.has_antonym_pair({"hello"}, {"world"})
    print("  ✓ Antonym lexicon works")
    
    # Temporal
    temp = get_temporal_lexicon()
    assert temp.is_conflict("today", "tomorrow")
    assert not temp.is_conflict("monday", "tuesday")
    print("  ✓ Temporal lexicon works")
    
    # Reply patterns
    reply = get_reply_pattern_lexicon()
    assert reply.get_pattern_type("thanks!") == "acknowledgment"
    assert reply.get_pattern_type("actually, let me correct that") == "correction"
    print("  ✓ Reply pattern lexicon works")
    
    print("  [OK] All lexicons loaded and functional")


def test_feature_extraction():
    """Test feature extraction with external lexicons."""
    print("\n=== Test: Feature Extraction ===")
    
    from features import (
        negation_mismatch,
        numeric_mismatch,
        temporal_mismatch,
        antonym_detected,
        cancellation_mismatch,
        detect_reply_pattern,
    )
    
    # Negation
    assert negation_mismatch("Project approved", "Project not approved")
    print("  ✓ Negation mismatch detection")
    
    # Numeric
    assert numeric_mismatch("Budget is $50,000", "Budget is $75,000")
    print("  ✓ Numeric mismatch detection")
    
    # Temporal
    assert temporal_mismatch("Meeting at 3pm Monday", "Meeting at 4pm Monday")
    print("  ✓ Temporal mismatch detection")
    
    # Antonyms
    assert antonym_detected("The task succeeded", "The task failed")
    print("  ✓ Antonym detection")
    
    # Cancellation
    assert cancellation_mismatch("Actually, the meeting is cancelled", "Meeting at 3pm")
    print("  ✓ Cancellation detection")
    
    # Reply patterns
    assert detect_reply_pattern("Thanks!") == "acknowledgment"
    assert detect_reply_pattern("What time?") == "question"
    print("  ✓ Reply pattern detection")
    
    print("  [OK] All feature extraction working")


def test_pair_features():
    """Test PairFeatures extraction."""
    print("\n=== Test: PairFeatures ===")
    
    from features import FeatureExtractor, PairFeatures
    
    class MockEvent:
        def __init__(self, content, timestamp=0, user_id="user1"):
            self.content = content
            self.timestamp = timestamp
            self.user_id = user_id
            self.metadata = {}
    
    extractor = FeatureExtractor()
    
    # Test contradiction detection
    event_a = MockEvent("Budget is $75,000 and project is not approved")
    event_b = MockEvent("Budget is $50,000 and project is approved")
    
    features = extractor.extract(event_a, event_b, embedding_similarity=0.85)
    
    assert features.negation_mismatch
    assert features.numeric_mismatch
    assert features.antonym_detected
    assert features.has_contradiction
    assert features.strict_contradiction  # 3+ signals
    
    print(f"  Contradiction count: {features.contradiction_count}")
    print(f"  Fired signals: {features.get_fired_signals()}")
    print("  [OK] PairFeatures extraction working")


def test_structural_signals():
    """Test structural signal computation."""
    print("\n=== Test: Structural Signals ===")
    
    from structural import (
        StructuralAnalyzer,
        compute_event_density,
        is_episode_boundary,
        compute_structural_salience,
    )
    
    class MockEvent:
        def __init__(self, timestamp):
            self.timestamp = timestamp
    
    # Event density
    now = time.time()
    recent = [MockEvent(now - 10), MockEvent(now - 20), MockEvent(now - 30)]
    density = compute_event_density(now, recent, window_seconds=60)
    print(f"  Event density: {density:.2f} events/min")
    
    # Episode boundary
    prev = MockEvent(now - 600)  # 10 min ago
    assert is_episode_boundary(now, prev, gap_threshold=300)
    print("  ✓ Episode boundary detection (10min gap)")
    
    prev = MockEvent(now - 60)  # 1 min ago
    assert not is_episode_boundary(now, prev, gap_threshold=300)
    print("  ✓ Episode continuation (1min gap)")
    
    # Structural salience
    salience = compute_structural_salience(
        is_boundary=True,
        is_correction=True,
        relation_count=3,
    )
    print(f"  Structural salience: {salience:.3f}")
    assert salience > 0.5  # Should be boosted
    
    print("  [OK] Structural signals working")


def test_salience_breakdown():
    """Test salience with component breakdown."""
    print("\n=== Test: Salience Breakdown ===")
    
    from salience import SalienceEngine, compute_salience_breakdown
    
    now = time.time()
    
    # Get breakdown
    breakdown = compute_salience_breakdown(
        timestamp=now - 3600,  # 1 hour old
        access_count=5,
        avg_similarity=0.7,
        structural_score=0.6,
        now=now,
    )
    
    print(f"  Components:")
    print(f"    semantic:   {breakdown['semantic']:.4f}")
    print(f"    recency:    {breakdown['recency']:.4f}")
    print(f"    frequency:  {breakdown['frequency']:.4f}")
    print(f"    structural: {breakdown['structural']:.4f}")
    print(f"    final:      {breakdown['final']:.4f}")
    
    # Verify all components present
    assert "semantic" in breakdown
    assert "recency" in breakdown
    assert "frequency" in breakdown
    assert "structural" in breakdown
    assert "final" in breakdown
    
    print("  [OK] Salience breakdown working")


def test_schema_output():
    """Test spec-compliant schema output."""
    print("\n=== Test: Schema Output ===")
    
    from schema import MetadataBuilder, EnrichedMetadataV2
    from relations import RelationType, RelationResult
    
    # Create mock relation
    relation = RelationResult(
        kind=RelationType.SUPERSEDES,
        target="evt_001",
        confidence=0.85,
        signals=["numeric_mismatch", "negation_mismatch"],
        reason_code="supersedes: numeric_mismatch + negation_mismatch + newer_event",
        strict_contradiction=True,
    )
    
    # Build metadata
    metadata = (
        MetadataBuilder()
        .with_salience(semantic=0.82, recency=0.40, frequency=0.10, structural=0.15)
        .with_relations([relation])
        .with_retention("durable")
        .with_temporal(
            episode_id="ep_004",
            is_boundary=False,
            age_seconds=123400,
            drift_factor=0.31,
        )
        .with_confidence(0.92)
        .build()
    )
    
    # Get spec-compliant output
    output = metadata.to_dict()
    
    print("  Output structure:")
    print(f"    salience: {output['salience']}")
    print(f"    relations: {len(output['relations'])} relation(s)")
    print(f"    retention_class: {output['retention_class']}")
    print(f"    temporal: {output['temporal']}")
    print(f"    confidence: {output['confidence']}")
    
    # Verify structure matches spec
    assert "salience" in output
    assert "semantic" in output["salience"]
    assert "recency" in output["salience"]
    assert "frequency" in output["salience"]
    assert "structural" in output["salience"]
    assert "final" in output["salience"]
    
    assert "relations" in output
    assert "retention_class" in output
    
    assert "temporal" in output
    assert "episode_id" in output["temporal"]
    assert "is_boundary" in output["temporal"]
    assert "drift_factor" in output["temporal"]
    
    assert "confidence" in output
    
    print("  [OK] Schema output matches spec")


def test_full_pipeline():
    """Test full enrichment pipeline."""
    print("\n=== Test: Full Pipeline ===")
    
    from features import FeatureExtractor
    from relations import RelationScorer
    from salience import SalienceEngine
    from retention_v2 import RetentionClassifier
    from structural import StructuralAnalyzer
    from schema import MetadataBuilder
    
    class MockEvent:
        def __init__(self, event_id, content, timestamp, user_id="user1"):
            self.event_id = event_id
            self.content = content
            self.timestamp = timestamp
            self.user_id = user_id
            self.metadata = {}
    
    now = time.time()
    
    # Create events
    old_event = MockEvent("evt_001", "Budget is $50,000", now - 3600, "user1")
    new_event = MockEvent("evt_002", "Budget is now $75,000", now, "user1")
    
    # 1. Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(new_event, old_event, embedding_similarity=0.85)
    print(f"  Features: contradiction_count={features.contradiction_count}")
    
    # 2. Score relations
    scorer = RelationScorer()
    relations = scorer.score_pair(new_event, old_event, features)
    print(f"  Relations: {[r.kind.value for r in relations]}")
    
    # 3. Structural analysis
    analyzer = StructuralAnalyzer()
    structural = analyzer.analyze(
        new_event,
        previous_event=old_event,
        is_correction=features.cancellation_detected,
        relation_count=len(relations),
    )
    print(f"  Structural salience: {structural.salience.score:.3f}")
    
    # 4. Compute salience
    salience_engine = SalienceEngine()
    salience = salience_engine.compute_with_relations(
        event_id=new_event.event_id,
        timestamp=new_event.timestamp,
        relations=relations,
        avg_similarity=0.85,
        structural_score=structural.salience.score,
        now=now,
    )
    print(f"  Final salience: {salience.score:.3f}")
    
    # 5. Classify retention
    classifier = RetentionClassifier()
    retention = classifier.classify_with_drift(
        timestamp=new_event.timestamp,
        salience=salience.score,
        relations=relations,
        now=now,
    )
    print(f"  Retention class: {retention.retention_class.value}")
    
    # 6. Build spec-compliant output
    metadata = (
        MetadataBuilder()
        .with_salience(
            semantic=salience.semantic,
            recency=salience.recency,
            frequency=salience.frequency,
            structural=salience.structural,
            final=salience.score,
        )
        .with_relations(relations)
        .with_retention(retention.retention_class.value)
        .with_temporal(
            episode_id="ep_001",
            is_boundary=structural.boundary.is_boundary,
            age_seconds=0,
            drift_factor=retention.drift,
        )
        .with_confidence(0.95)
        .build()
    )
    
    output = metadata.to_dict()
    print(f"\n  Final output:")
    import json
    print(json.dumps(output, indent=2))
    
    print("\n  [OK] Full pipeline working")


def main():
    """Run all tests."""
    print("=" * 60)
    print("KRNX Enrichment v3.1 - Refactor Tests")
    print("=" * 60)
    
    tests = [
        test_lexicon_loading,
        test_feature_extraction,
        test_pair_features,
        test_structural_signals,
        test_salience_breakdown,
        test_schema_output,
        test_full_pipeline,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
