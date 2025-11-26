"""
KRNX Enrichment - Spec-Compliant Output Schema

Defines the unified metadata output schema that matches the KRNX Constitution
Addendum specification exactly.

Output Shape:
{
  "salience": {
    "semantic": 0.82,
    "recency": 0.40,
    "frequency": 0.10,
    "structural": 0.15,
    "final": 0.67
  },
  "relations": [
    {
      "type": "supersedes",
      "target_event_id": "evt_2381",
      "confidence": 0.78,
      "signals": ["contradiction: numeric mismatch", "newer_timestamp"],
      "reason_code": "UPDATE_NUMERIC"
    }
  ],
  "retention_class": "durable",
  "temporal": {
    "episode_id": "ep_004",
    "is_boundary": false,
    "age_seconds": 123400,
    "drift_factor": 0.31
  },
  "confidence": 0.92
}

All fields are deterministic and testable.
No LLM inference is used in metadata creation.

Constitution-safe: Pure, deterministic, descriptive only.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .relations import RelationResult, RelationType


# ==============================================
# SALIENCE OUTPUT
# ==============================================

@dataclass
class SalienceOutput:
    """
    Spec-compliant salience output with component breakdown.
    
    Components must sum to produce 'final' (weighted average).
    """
    semantic: float = 0.0       # Similarity to recent queries
    recency: float = 0.0        # Time-based decay
    frequency: float = 0.0      # Access count normalized
    structural: float = 0.0     # Position/relation importance
    final: float = 0.0          # Weighted combination
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to spec-compliant dictionary."""
        return {
            "semantic": round(self.semantic, 4),
            "recency": round(self.recency, 4),
            "frequency": round(self.frequency, 4),
            "structural": round(self.structural, 4),
            "final": round(self.final, 4),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "SalienceOutput":
        """Create from dictionary."""
        return cls(
            semantic=data.get("semantic", 0.0),
            recency=data.get("recency", 0.0),
            frequency=data.get("frequency", 0.0),
            structural=data.get("structural", 0.0),
            final=data.get("final", 0.0),
        )


# ==============================================
# RELATION OUTPUT
# ==============================================

@dataclass
class RelationOutput:
    """
    Spec-compliant relation output.
    
    Maps to:
    {
      "type": "supersedes",
      "target_event_id": "evt_2381",
      "confidence": 0.78,
      "signals": ["contradiction: numeric mismatch", "newer_timestamp"],
      "reason_code": "UPDATE_NUMERIC"
    }
    """
    type: str                   # supersedes, contradicts, duplicates, etc.
    target_event_id: str        # Target event ID
    confidence: float           # 0.0-1.0
    signals: List[str]          # Which signals fired
    reason_code: str            # Human-readable cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to spec-compliant dictionary."""
        return {
            "type": self.type,
            "target_event_id": self.target_event_id,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
            "reason_code": self.reason_code,
        }
    
    @classmethod
    def from_relation_result(cls, result: RelationResult) -> "RelationOutput":
        """Convert from internal RelationResult."""
        # Map signals to spec format
        spec_signals = []
        for signal in result.signals:
            if signal == "negation_mismatch":
                spec_signals.append("contradiction: negation")
            elif signal == "numeric_mismatch":
                spec_signals.append("contradiction: numeric mismatch")
            elif signal == "temporal_mismatch":
                spec_signals.append("contradiction: temporal mismatch")
            elif signal == "antonym_detected":
                spec_signals.append("contradiction: antonym")
            elif signal == "cancellation_detected":
                spec_signals.append("contradiction: cancellation")
            else:
                spec_signals.append(signal)
        
        # Generate reason code
        reason_code = cls._generate_reason_code(result)
        
        return cls(
            type=result.kind.value,
            target_event_id=result.target,
            confidence=result.confidence,
            signals=spec_signals,
            reason_code=reason_code,
        )
    
    @staticmethod
    def _generate_reason_code(result: RelationResult) -> str:
        """Generate standardized reason code from result."""
        kind = result.kind
        signals = result.signals
        
        # Map to standardized codes
        if kind == RelationType.SUPERSEDES:
            if "numeric_mismatch" in signals:
                return "UPDATE_NUMERIC"
            elif "temporal_mismatch" in signals:
                return "UPDATE_TEMPORAL"
            elif "negation_mismatch" in signals:
                return "CANCELLATION_NEGATION"
            elif "cancellation_detected" in signals:
                return "CANCELLATION_VERB"
            else:
                return "UPDATE_GENERAL"
        
        elif kind == RelationType.CONTRADICTS:
            if "negation_mismatch" in signals:
                return "CONFLICT_NEGATION"
            elif "numeric_mismatch" in signals:
                return "CONFLICT_NUMERIC"
            elif "antonym_detected" in signals:
                return "CONFLICT_ANTONYM"
            else:
                return "CONFLICT_GENERAL"
        
        elif kind == RelationType.DUPLICATES:
            return "DUPLICATE_SEMANTIC"
        
        elif kind == RelationType.EXPANDS_ON:
            return "EXPANSION_RELATED"
        
        elif kind == RelationType.REPLIES_TO:
            return "REPLY_STRUCTURAL"
        
        else:
            return f"{kind.value.upper()}_DETECTED"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationOutput":
        """Create from dictionary."""
        return cls(
            type=data["type"],
            target_event_id=data["target_event_id"],
            confidence=data.get("confidence", 1.0),
            signals=data.get("signals", []),
            reason_code=data.get("reason_code", ""),
        )


# ==============================================
# TEMPORAL OUTPUT
# ==============================================

@dataclass
class TemporalOutput:
    """
    Spec-compliant temporal output.
    
    Maps to:
    {
      "episode_id": "ep_004",
      "is_boundary": false,
      "age_seconds": 123400,
      "drift_factor": 0.31
    }
    """
    episode_id: Optional[str] = None
    is_boundary: bool = False           # True if this starts a new episode
    age_seconds: float = 0.0            # Seconds since creation
    drift_factor: float = 0.0           # Context drift [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to spec-compliant dictionary."""
        return {
            "episode_id": self.episode_id,
            "is_boundary": self.is_boundary,
            "age_seconds": round(self.age_seconds, 2),
            "drift_factor": round(self.drift_factor, 4),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalOutput":
        """Create from dictionary."""
        return cls(
            episode_id=data.get("episode_id"),
            is_boundary=data.get("is_boundary", False),
            age_seconds=data.get("age_seconds", 0.0),
            drift_factor=data.get("drift_factor", 0.0),
        )


# ==============================================
# UNIFIED METADATA OUTPUT
# ==============================================

@dataclass
class EnrichedMetadataV2:
    """
    Spec-compliant unified metadata output.
    
    This is the top-level structure that matches the Constitution Addendum.
    """
    salience: SalienceOutput = field(default_factory=SalienceOutput)
    relations: List[RelationOutput] = field(default_factory=list)
    retention_class: str = "durable"
    temporal: TemporalOutput = field(default_factory=TemporalOutput)
    confidence: float = 1.0             # Overall enrichment confidence
    
    # Provenance (not in spec, but useful)
    enrichment_version: str = "3.0.0"
    computed_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to spec-compliant dictionary.
        
        This is the exact format specified in the Constitution Addendum.
        """
        return {
            "salience": self.salience.to_dict(),
            "relations": [r.to_dict() for r in self.relations],
            "retention_class": self.retention_class,
            "temporal": self.temporal.to_dict(),
            "confidence": round(self.confidence, 4),
        }
    
    def to_full_dict(self) -> Dict[str, Any]:
        """
        Convert to full dictionary including provenance.
        
        Includes additional metadata not in minimal spec.
        """
        result = self.to_dict()
        result["_provenance"] = {
            "version": self.enrichment_version,
            "computed_at": self.computed_at,
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnrichedMetadataV2":
        """Create from dictionary."""
        provenance = data.get("_provenance", {})
        
        return cls(
            salience=SalienceOutput.from_dict(data.get("salience", {})),
            relations=[
                RelationOutput.from_dict(r) 
                for r in data.get("relations", [])
            ],
            retention_class=data.get("retention_class", "durable"),
            temporal=TemporalOutput.from_dict(data.get("temporal", {})),
            confidence=data.get("confidence", 1.0),
            enrichment_version=provenance.get("version", "3.0.0"),
            computed_at=provenance.get("computed_at", time.time()),
        )


# ==============================================
# BUILDER FOR ASSEMBLY
# ==============================================

class MetadataBuilder:
    """
    Builds EnrichedMetadataV2 from component results.
    
    This is the assembly point where individual enrichment
    outputs are combined into the spec-compliant format.
    """
    
    def __init__(self):
        """Initialize builder."""
        self._salience: Optional[SalienceOutput] = None
        self._relations: List[RelationOutput] = []
        self._retention_class: str = "durable"
        self._temporal: Optional[TemporalOutput] = None
        self._confidence: float = 1.0
    
    def with_salience(
        self,
        semantic: float = 0.0,
        recency: float = 0.0,
        frequency: float = 0.0,
        structural: float = 0.0,
        final: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> "MetadataBuilder":
        """
        Set salience scores.
        
        Args:
            semantic: Semantic similarity score
            recency: Recency decay score
            frequency: Frequency score
            structural: Structural importance score
            final: Final score (computed if not provided)
            weights: Optional component weights
        
        Returns:
            Self for chaining
        """
        # Default weights
        if weights is None:
            weights = {
                "semantic": 0.4,
                "recency": 0.3,
                "frequency": 0.15,
                "structural": 0.15,
            }
        
        # Compute final if not provided
        if final is None:
            final = (
                weights.get("semantic", 0.4) * semantic +
                weights.get("recency", 0.3) * recency +
                weights.get("frequency", 0.15) * frequency +
                weights.get("structural", 0.15) * structural
            )
        
        self._salience = SalienceOutput(
            semantic=semantic,
            recency=recency,
            frequency=frequency,
            structural=structural,
            final=final,
        )
        return self
    
    def with_relations(
        self,
        relations: List[RelationResult],
    ) -> "MetadataBuilder":
        """
        Set relations from internal RelationResult list.
        
        Args:
            relations: List of RelationResult from scorer
        
        Returns:
            Self for chaining
        """
        self._relations = [
            RelationOutput.from_relation_result(r)
            for r in relations
        ]
        return self
    
    def with_retention(
        self,
        retention_class: str,
    ) -> "MetadataBuilder":
        """
        Set retention class.
        
        Args:
            retention_class: One of ephemeral, durable, etc.
        
        Returns:
            Self for chaining
        """
        self._retention_class = retention_class
        return self
    
    def with_temporal(
        self,
        episode_id: Optional[str] = None,
        is_boundary: bool = False,
        age_seconds: float = 0.0,
        drift_factor: float = 0.0,
    ) -> "MetadataBuilder":
        """
        Set temporal metadata.
        
        Args:
            episode_id: Episode identifier
            is_boundary: Whether this is an episode boundary
            age_seconds: Event age in seconds
            drift_factor: Context drift factor
        
        Returns:
            Self for chaining
        """
        self._temporal = TemporalOutput(
            episode_id=episode_id,
            is_boundary=is_boundary,
            age_seconds=age_seconds,
            drift_factor=drift_factor,
        )
        return self
    
    def with_confidence(
        self,
        confidence: float,
    ) -> "MetadataBuilder":
        """
        Set overall confidence.
        
        Args:
            confidence: Confidence score [0, 1]
        
        Returns:
            Self for chaining
        """
        self._confidence = confidence
        return self
    
    def build(self) -> EnrichedMetadataV2:
        """
        Build the final EnrichedMetadataV2.
        
        Returns:
            Complete metadata object
        """
        return EnrichedMetadataV2(
            salience=self._salience or SalienceOutput(),
            relations=self._relations,
            retention_class=self._retention_class,
            temporal=self._temporal or TemporalOutput(),
            confidence=self._confidence,
        )


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Output types
    'SalienceOutput',
    'RelationOutput',
    'TemporalOutput',
    'EnrichedMetadataV2',
    
    # Builder
    'MetadataBuilder',
]
