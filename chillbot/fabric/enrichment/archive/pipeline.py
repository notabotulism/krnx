"""
KRNX Fabric - Enrichment Pipeline

Main orchestrator for metadata enrichment.
Runs all enrichment stages in order, returns unified metadata.

Constitution Compliance:
- Purely optional
- Purely deterministic  
- Metadata only — no deletion, no merging, no summarization
- No autonomous behavior
- No implicit jobs
- No side effects outside metadata

Usage:
    enricher = MetadataEnricher()
    
    # Enrich with context (previous events, embeddings, etc.)
    context = EnrichmentContext(
        previous_event=last_event,
        workspace_events=recent_events,
        query_embedding=embedding,
    )
    
    metadata = enricher.enrich(event, context)
    event.metadata.update(metadata.to_dict())
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .temporal import TemporalEnricher
from .relations import RelationEnricher, Relation
from .entities import EntityExtractor, Entity
from .retention import RetentionSignaler, RetentionSignals

# Import salience from compute - handle various import contexts
import sys
import os

# Add parent dirs to path for standalone testing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_fabric_dir = os.path.dirname(_current_dir)
_chillbot_dir = os.path.dirname(_fabric_dir)
if _chillbot_dir not in sys.path:
    sys.path.insert(0, _chillbot_dir)

from compute.salience import SalienceEngine, SalienceConfig, SalienceMethod

logger = logging.getLogger(__name__)


# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class EnrichmentConfig:
    """
    Configuration for metadata enrichment.
    
    All settings optional with sane defaults.
    """
    # Temporal
    episode_gap_threshold: float = 300.0  # 5 min = new episode
    
    # Salience (recency)
    recency_half_life_hours: float = 24.0
    recency_floor: float = 0.1
    
    # Salience weights
    weight_recency: float = 0.4
    weight_frequency: float = 0.2
    weight_semantic: float = 0.4
    
    # Relations
    duplicate_similarity_threshold: float = 0.95
    expand_similarity_threshold: float = 0.70
    supersede_similarity_threshold: float = 0.90  # + recency check
    
    # Entities
    enable_entity_extraction: bool = True
    entity_patterns: Optional[Dict[str, str]] = None  # Custom regex patterns
    
    # Retention signals
    ephemeral_ttl_hours: float = 1.0
    consolidation_age_days: float = 7.0
    consolidation_salience_ceiling: float = 0.3
    
    # Feature flags
    enable_temporal: bool = True
    enable_salience: bool = True
    enable_relations: bool = True
    enable_entities: bool = True
    enable_retention: bool = True
    
    # Provenance
    enricher_version: str = "1.0.0"


# ==============================================
# CONTEXT (Input to enrichment)
# ==============================================

@dataclass
class EnrichmentContext:
    """
    Context for enrichment — what the enricher needs to compute metadata.
    
    Provided by the caller (fabric orchestrator).
    """
    # Previous event in this workspace/user stream
    previous_event: Optional[Any] = None
    
    # Recent events for relation detection
    workspace_events: List[Any] = field(default_factory=list)
    
    # Embeddings for semantic comparison
    event_embedding: Optional[List[float]] = None
    workspace_embeddings: Optional[Dict[str, List[float]]] = None  # event_id -> embedding
    
    # Similarity scores (pre-computed if available)
    similarity_scores: Optional[Dict[str, float]] = None  # event_id -> similarity
    
    # Access counts (from kernel stats)
    access_count: int = 0
    
    # Current time (for deterministic testing)
    now: Optional[float] = None
    
    def get_now(self) -> float:
        return self.now or time.time()


# ==============================================
# OUTPUT (Enriched metadata)
# ==============================================

@dataclass
class EnrichedMetadata:
    """
    Complete enriched metadata for an event.
    
    Maps directly to the v1 metadata schema.
    """
    # Temporal
    time_gap_seconds: Optional[float] = None
    episode_id: Optional[str] = None
    last_access: Optional[float] = None
    
    # Importance (salience)
    salience_score: Optional[float] = None
    salience_factors: Dict[str, float] = field(default_factory=dict)
    salience_computed_at: Optional[float] = None
    access_count: int = 0
    
    # Relations
    relations: List[Dict[str, str]] = field(default_factory=list)
    
    # Entities
    entities: List[Dict[str, str]] = field(default_factory=list)
    
    # Retention signals
    retention_class: Optional[str] = None
    consolidation_candidate: bool = False
    expiry_suggested_at: Optional[float] = None
    
    # Provenance
    enrichment_version: str = "1.0.0"
    enrichment_computed_at: Optional[float] = None
    enrichment_sources: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for event metadata."""
        result = {}
        
        # Temporal
        if self.time_gap_seconds is not None:
            result["time_gap_seconds"] = self.time_gap_seconds
        if self.episode_id:
            result["episode_id"] = self.episode_id
        if self.last_access:
            result["last_access"] = self.last_access
        
        # Importance
        if self.salience_score is not None:
            result["salience_score"] = self.salience_score
            result["salience_factors"] = self.salience_factors
            result["salience_computed_at"] = self.salience_computed_at
        result["access_count"] = self.access_count
        
        # Relations
        if self.relations:
            result["relations"] = self.relations
        
        # Entities
        if self.entities:
            result["entities"] = self.entities
        
        # Retention
        if self.retention_class:
            result["retention_class"] = self.retention_class
        result["consolidation_candidate"] = self.consolidation_candidate
        if self.expiry_suggested_at:
            result["expiry_suggested_at"] = self.expiry_suggested_at
        
        # Provenance
        result["enrichment"] = {
            "version": self.enrichment_version,
            "computed_at": self.enrichment_computed_at,
            "sources": self.enrichment_sources,
        }
        
        return result
    
    def to_nested_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary matching v1 schema structure."""
        return {
            "temporal": {
                "time_gap_seconds": self.time_gap_seconds,
                "episode_id": self.episode_id,
                "last_access": self.last_access,
            },
            "importance": {
                "salience_score": self.salience_score,
                "salience_factors": self.salience_factors,
                "salience_computed_at": self.salience_computed_at,
                "access_count": self.access_count,
            },
            "relations": self.relations,
            "entities": self.entities,
            "retention": {
                "retention_class": self.retention_class,
                "consolidation_candidate": self.consolidation_candidate,
                "expiry_suggested_at": self.expiry_suggested_at,
            },
            "provenance": {
                "enrichment_version": self.enrichment_version,
                "enrichment_computed_at": self.enrichment_computed_at,
                "enrichment_sources": self.enrichment_sources,
            },
        }


# ==============================================
# MAIN ENRICHER
# ==============================================

class MetadataEnricher:
    """
    Unified metadata enrichment pipeline.
    
    Runs all enrichment stages:
    1. Temporal (episode threading, time gaps)
    2. Salience (importance scoring)
    3. Relations (replies_to, duplicates, expands_on, supersedes)
    4. Entities (extract referenced entities)
    5. Retention (consolidation signals)
    
    Constitution-safe: Pure transform, no side effects.
    """
    
    def __init__(self, config: Optional[EnrichmentConfig] = None):
        """
        Initialize enricher with configuration.
        
        Args:
            config: Enrichment configuration (uses defaults if not provided)
        """
        self.config = config or EnrichmentConfig()
        
        # Initialize sub-enrichers
        self._temporal = TemporalEnricher(
            episode_gap_threshold=self.config.episode_gap_threshold
        )
        
        # Salience engine (from compute layer)
        salience_config = SalienceConfig(
            recency_halflife=self.config.recency_half_life_hours * 3600,
            recency_min=self.config.recency_floor,
            recency_weight=self.config.weight_recency,
            frequency_weight=self.config.weight_frequency,
            semantic_weight=self.config.weight_semantic,
        )
        self._salience = SalienceEngine(config=salience_config)
        
        self._relations = RelationEnricher(
            duplicate_threshold=self.config.duplicate_similarity_threshold,
            expand_threshold=self.config.expand_similarity_threshold,
            supersede_threshold=self.config.supersede_similarity_threshold,
        )
        
        self._entities = EntityExtractor(
            custom_patterns=self.config.entity_patterns
        )
        
        self._retention = RetentionSignaler(
            ephemeral_ttl_hours=self.config.ephemeral_ttl_hours,
            consolidation_age_days=self.config.consolidation_age_days,
            consolidation_salience_ceiling=self.config.consolidation_salience_ceiling,
        )
        
        logger.debug(f"[ENRICHER] Initialized with config: {self.config}")
    
    def enrich(
        self,
        event: Any,
        context: Optional[EnrichmentContext] = None,
    ) -> EnrichedMetadata:
        """
        Enrich an event with computed metadata.
        
        Args:
            event: Event object (must have event_id, timestamp, content)
            context: Enrichment context (previous events, embeddings, etc.)
        
        Returns:
            EnrichedMetadata with all computed fields
        """
        context = context or EnrichmentContext()
        now = context.get_now()
        
        result = EnrichedMetadata(
            enrichment_version=self.config.enricher_version,
            enrichment_computed_at=now,
            access_count=context.access_count,
        )
        
        # Get event properties
        event_id = getattr(event, 'event_id', str(id(event)))
        timestamp = getattr(event, 'timestamp', now)
        content = getattr(event, 'content', {})
        
        # ==============================================
        # 1. TEMPORAL ENRICHMENT
        # ==============================================
        if self.config.enable_temporal:
            temporal = self._temporal.enrich(
                timestamp=timestamp,
                previous_event=context.previous_event,
                now=now,
            )
            result.time_gap_seconds = temporal.get("time_gap_seconds")
            result.episode_id = temporal.get("episode_id")
            result.last_access = now
            result.enrichment_sources.append("temporal")
        
        # ==============================================
        # 2. SALIENCE SCORING
        # ==============================================
        if self.config.enable_salience:
            # Compute average similarity if embeddings available
            avg_similarity = 0.0
            if context.similarity_scores:
                scores = list(context.similarity_scores.values())
                if scores:
                    avg_similarity = sum(scores) / len(scores)
            
            salience = self._salience.compute(
                event_id=event_id,
                timestamp=timestamp,
                access_count=context.access_count,
                avg_similarity=avg_similarity,
                method=SalienceMethod.COMPOSITE,
                now=now,
            )
            
            result.salience_score = salience.score
            result.salience_factors = salience.factors
            result.salience_computed_at = salience.computed_at
            result.enrichment_sources.append("salience")
        
        # ==============================================
        # 3. RELATION DETECTION
        # ==============================================
        if self.config.enable_relations:
            relations = self._relations.detect(
                event_id=event_id,
                timestamp=timestamp,
                content=content,
                previous_event=context.previous_event,
                similarity_scores=context.similarity_scores,
                workspace_events=context.workspace_events,
            )
            
            result.relations = [r.to_dict() for r in relations]
            if relations:
                result.enrichment_sources.append("relations")
        
        # ==============================================
        # 4. ENTITY EXTRACTION
        # ==============================================
        if self.config.enable_entities:
            entities = self._entities.extract(content)
            result.entities = [e.to_dict() for e in entities]
            if entities:
                result.enrichment_sources.append("entities")
        
        # ==============================================
        # 5. RETENTION SIGNALS
        # ==============================================
        if self.config.enable_retention:
            retention = self._retention.compute(
                timestamp=timestamp,
                salience_score=result.salience_score,
                retention_class=getattr(event, 'retention_class', None),
                now=now,
            )
            
            result.retention_class = retention.retention_class
            result.consolidation_candidate = retention.consolidation_candidate
            result.expiry_suggested_at = retention.expiry_suggested_at
            result.enrichment_sources.append("retention")
        
        logger.debug(
            f"[ENRICHER] Enriched {event_id}: "
            f"salience={result.salience_score if result.salience_score else 'N/A'}, "
            f"relations={len(result.relations)}, "
            f"entities={len(result.entities)}"
        )
        
        return result
    
    def enrich_batch(
        self,
        events: List[Any],
        contexts: Optional[List[EnrichmentContext]] = None,
    ) -> List[EnrichedMetadata]:
        """
        Enrich multiple events.
        
        Args:
            events: List of events
            contexts: Optional list of contexts (one per event)
        
        Returns:
            List of EnrichedMetadata
        """
        contexts = contexts or [EnrichmentContext() for _ in events]
        
        if len(contexts) != len(events):
            raise ValueError("contexts must match events length")
        
        return [
            self.enrich(event, context)
            for event, context in zip(events, contexts)
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enricher statistics."""
        return {
            "config": {
                "episode_gap_threshold": self.config.episode_gap_threshold,
                "recency_half_life_hours": self.config.recency_half_life_hours,
                "weight_recency": self.config.weight_recency,
                "weight_frequency": self.config.weight_frequency,
                "weight_semantic": self.config.weight_semantic,
            },
            "enabled": {
                "temporal": self.config.enable_temporal,
                "salience": self.config.enable_salience,
                "relations": self.config.enable_relations,
                "entities": self.config.enable_entities,
                "retention": self.config.enable_retention,
            },
        }
    
    def __repr__(self) -> str:
        enabled = []
        if self.config.enable_temporal:
            enabled.append("temporal")
        if self.config.enable_salience:
            enabled.append("salience")
        if self.config.enable_relations:
            enabled.append("relations")
        if self.config.enable_entities:
            enabled.append("entities")
        if self.config.enable_retention:
            enabled.append("retention")
        
        return f"MetadataEnricher(stages={enabled})"
