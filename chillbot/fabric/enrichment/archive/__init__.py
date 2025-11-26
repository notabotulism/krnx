"""
KRNX Fabric - Metadata Enrichment

Pure, deterministic metadata enrichment for events.
Constitution-safe: no side effects, no deletion, no autonomous behavior.

Usage:
    from chillbot.fabric.enrichment import MetadataEnricher, EnrichmentConfig
    
    enricher = MetadataEnricher()
    
    # Enrich single event
    enriched_metadata = enricher.enrich(event, context)
    
    # Enrich with custom config
    enricher = MetadataEnricher(config=EnrichmentConfig(
        episode_gap_threshold=300,
        enable_entity_extraction=True,
    ))
"""

from .pipeline import (
    MetadataEnricher,
    EnrichmentConfig,
    EnrichmentContext,
    EnrichedMetadata,
)

from .temporal import (
    TemporalEnricher,
    EpisodeTracker,
)

from .relations import (
    RelationEnricher,
    Relation,
    RelationType,
)

from .entities import (
    EntityExtractor,
    Entity,
    EntityType,
)

from .retention import (
    RetentionSignaler,
    RetentionSignals,
)

__all__ = [
    # Main pipeline
    "MetadataEnricher",
    "EnrichmentConfig", 
    "EnrichmentContext",
    "EnrichedMetadata",
    
    # Temporal
    "TemporalEnricher",
    "EpisodeTracker",
    
    # Relations
    "RelationEnricher",
    "Relation",
    "RelationType",
    
    # Entities
    "EntityExtractor",
    "Entity",
    "EntityType",
    
    # Retention
    "RetentionSignaler",
    "RetentionSignals",
]
