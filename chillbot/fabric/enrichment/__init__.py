"""
KRNX Enrichment - Multi-Signal Metadata Enrichment v3.1

Transforms raw events into intelligent memory through deterministic enrichment.
No LLMs. No hallucination. Just composable, transparent scoring.

v3.1 Changes:
- External JSON lexicons (signals/*.json)
- User overrides (~/.krnx/signals/)
- Spec-compliant output schema
- Structural salience component
- Event density and boundary detection
- Reply pattern detection

Components:
- signals/: External lexicons (negation, antonyms, cancellation, temporal)
- features.py: Multi-signal feature extraction
- relations.py: Relation scoring with reason codes
- cross_encoder.py: High-accuracy semantic reranking
- summary.py: Relation graph summary computation
- salience.py: Importance scoring with 4 components
- structural.py: Event density, boundary, structural salience
- retention_v2.py: Drift × salience retention classification
- schema.py: Spec-compliant output format

Usage:
    from enrichment import (
        FeatureExtractor,
        RelationScorer,
        SalienceEngine,
        RetentionClassifier,
        MetadataBuilder,
    )
    
    # Extract features from event pair
    extractor = FeatureExtractor()
    features = extractor.extract(new_event, old_event, similarity=0.85)
    
    # Score relations
    scorer = RelationScorer()
    relations = scorer.score_pair(new_event, old_event, features)
    
    # Build spec-compliant metadata
    metadata = (
        MetadataBuilder()
        .with_salience(semantic=0.8, recency=0.6, frequency=0.2, structural=0.5)
        .with_relations(relations)
        .with_retention("durable")
        .with_temporal(episode_id="ep_001", is_boundary=False, drift_factor=0.3)
        .build()
    )
    
    # Get spec-compliant output
    output = metadata.to_dict()

Constitution Compliance:
- Purely optional (kernel works without enrichment)
- Purely deterministic (no LLM calls)
- Metadata only (no deletion, no merging)
- No autonomous behavior
- No side effects
"""

# ==============================================
# SIGNALS (External Lexicons)
# ==============================================

from .signals import (
    # Data classes
    NegationLexicon,
    CancellationLexicon,
    AntonymLexicon,
    TemporalLexicon,
    ReplyPatternLexicon,
    
    # Loader
    LexiconLoader,
    
    # Global access
    get_loader,
    get_negation_lexicon,
    get_cancellation_lexicon,
    get_antonym_lexicon,
    get_temporal_lexicon,
    get_reply_pattern_lexicon,
    reload_lexicons,
)

# ==============================================
# FEATURES (Multi-Signal Extraction)
# ==============================================

from .features import (
    # Core dataclass
    PairFeatures,
    
    # Extractor
    FeatureExtractor,
    extract_pair_features,
    
    # Individual detectors
    has_negation,
    negation_mismatch,
    has_cancellation,
    has_correction,
    cancellation_mismatch,
    extract_numerics,
    numeric_mismatch,
    extract_temporal,
    temporal_mismatch,
    find_antonyms,
    antonym_detected,
    entity_overlap,
    detect_reply_pattern,
    is_reply_like,
    
    # Types
    TemporalInfo,
)

# ==============================================
# RELATIONS (Multi-Signal Scoring)
# ==============================================

from .relations import (
    # Types
    RelationType,
    RelationResult,
    RelationScoringConfig,
    
    # Scorer
    RelationScorer,
    
    # Legacy compatibility
    Relation,
    RelationEnricher,
)

# ==============================================
# STRUCTURAL (Position-Based Signals)
# ==============================================

from .structural import (
    # Config
    StructuralConfig,
    
    # Results
    DensityResult,
    BoundaryResult,
    StructuralSalienceResult,
    StructuralAnalysis,
    
    # Computers
    DensityComputer,
    BoundaryDetector,
    StructuralSalienceComputer,
    StructuralAnalyzer,
    
    # Convenience
    compute_event_density,
    is_episode_boundary,
    compute_structural_salience,
)

# ==============================================
# CROSS-ENCODER (High-Accuracy Reranking)
# ==============================================

from .cross_encoder import (
    # Config
    CrossEncoderConfig,
    
    # Classes
    CrossEncoderBase,
    SentenceTransformerCrossEncoder,
    CrossEncoderReranker,
    
    # Convenience
    get_reranker,
    rerank_candidates,
)

# ==============================================
# SUMMARY (Relation Graph Summary)
# ==============================================

from .summary import (
    RelationSummary,
    RelationSummaryComputer,
    compute_relation_summary,
    get_current_version,
)

# ==============================================
# SALIENCE (Importance Scoring)
# ==============================================

from .salience import (
    # Config and types
    SalienceConfig,
    SalienceMethod,
    SalienceResult,
    
    # Engine
    SalienceEngine,
    
    # Convenience
    compute_salience,
    adjust_salience_from_relations,
    compute_salience_breakdown,
)

# ==============================================
# RETENTION (Drift × Salience Classification)
# ==============================================

from .retention_v2 import (
    # Classes
    RetentionClass,
    RetentionConfig,
    RetentionResult,
    
    # Engines
    DriftComputer,
    RetentionClassifier,
    
    # Convenience
    compute_retention_class,
    compute_drift,
)

# ==============================================
# SCHEMA (Spec-Compliant Output)
# ==============================================

from .schema import (
    # Output types
    SalienceOutput,
    RelationOutput,
    TemporalOutput,
    EnrichedMetadataV2,
    
    # Builder
    MetadataBuilder,
)

# ==============================================
# VERSION
# ==============================================

__version__ = "3.1.0"

# ==============================================
# ALL EXPORTS
# ==============================================

__all__ = [
    # Version
    "__version__",
    
    # Signals (lexicons)
    "NegationLexicon",
    "CancellationLexicon",
    "AntonymLexicon",
    "TemporalLexicon",
    "ReplyPatternLexicon",
    "LexiconLoader",
    "get_loader",
    "get_negation_lexicon",
    "get_cancellation_lexicon",
    "get_antonym_lexicon",
    "get_temporal_lexicon",
    "get_reply_pattern_lexicon",
    "reload_lexicons",
    
    # Features
    "PairFeatures",
    "FeatureExtractor",
    "extract_pair_features",
    "has_negation",
    "negation_mismatch",
    "has_cancellation",
    "has_correction",
    "cancellation_mismatch",
    "extract_numerics",
    "numeric_mismatch",
    "extract_temporal",
    "temporal_mismatch",
    "find_antonyms",
    "antonym_detected",
    "entity_overlap",
    "detect_reply_pattern",
    "is_reply_like",
    "TemporalInfo",
    
    # Relations
    "RelationType",
    "RelationResult",
    "RelationScoringConfig",
    "RelationScorer",
    "Relation",
    "RelationEnricher",
    
    # Structural
    "StructuralConfig",
    "DensityResult",
    "BoundaryResult",
    "StructuralSalienceResult",
    "StructuralAnalysis",
    "DensityComputer",
    "BoundaryDetector",
    "StructuralSalienceComputer",
    "StructuralAnalyzer",
    "compute_event_density",
    "is_episode_boundary",
    "compute_structural_salience",
    
    # Cross-encoder
    "CrossEncoderConfig",
    "CrossEncoderBase",
    "SentenceTransformerCrossEncoder",
    "CrossEncoderReranker",
    "get_reranker",
    "rerank_candidates",
    
    # Summary
    "RelationSummary",
    "RelationSummaryComputer",
    "compute_relation_summary",
    "get_current_version",
    
    # Salience
    "SalienceConfig",
    "SalienceMethod",
    "SalienceResult",
    "SalienceEngine",
    "compute_salience",
    "adjust_salience_from_relations",
    "compute_salience_breakdown",
    
    # Retention
    "RetentionClass",
    "RetentionConfig",
    "RetentionResult",
    "DriftComputer",
    "RetentionClassifier",
    "compute_retention_class",
    "compute_drift",
    
    # Schema (spec-compliant output)
    "SalienceOutput",
    "RelationOutput",
    "TemporalOutput",
    "EnrichedMetadataV2",
    "MetadataBuilder",
]
