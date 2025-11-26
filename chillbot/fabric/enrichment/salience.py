"""
KRNX Enrichment - Salience Scoring v2

Computes and adjusts salience (importance) scores for events.
Salience determines which memories surface during recall.

v2 Changes:
- Added structural component
- Exposed all component scores (semantic, recency, frequency, structural)
- Spec-compliant output format
- Integration with structural analyzer

Components:
- semantic: Similarity to recent queries/context
- recency: Time-based decay
- frequency: Access count normalized
- structural: Position-based importance (boundary, corrections, density)

Adjustments from relations:
- +0.10 for supersedes (corrections are important)
- +0.05 for contradicts (conflicts need attention)
- +0.05 * confidence for expands_on (related content)
- +0.05 if strict_contradiction (high-confidence signals)

Constitution-safe: Pure, deterministic, no side effects.
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

from .relations import RelationType, RelationResult

logger = logging.getLogger(__name__)


# ==============================================
# CONFIGURATION
# ==============================================

class SalienceMethod(Enum):
    """Methods for computing salience."""
    RECENCY = "recency"             # Time-based decay only
    FREQUENCY = "frequency"         # Access count only
    SEMANTIC = "semantic"           # Similarity only
    STRUCTURAL = "structural"       # Position-based only
    COMPOSITE = "composite"         # Weighted combination (default)


@dataclass
class SalienceConfig:
    """Configuration for salience computation."""
    
    # Recency decay
    recency_halflife: float = 86400.0  # 24 hours in seconds
    recency_min: float = 0.1           # Minimum recency score
    
    # Component weights (should sum to 1.0)
    recency_weight: float = 0.30
    frequency_weight: float = 0.15
    semantic_weight: float = 0.40
    structural_weight: float = 0.15
    
    # Frequency normalization
    frequency_max: int = 100           # Max access count for normalization
    
    # Relation adjustments
    supersedes_boost: float = 0.10     # Corrections are important
    contradicts_boost: float = 0.05    # Conflicts need attention
    expands_on_factor: float = 0.05    # Related content (multiplied by confidence)
    strict_contradiction_boost: float = 0.05  # High-confidence signals
    
    # Bounds
    min_salience: float = 0.0
    max_salience: float = 1.0


# ==============================================
# SALIENCE RESULT
# ==============================================

@dataclass
class SalienceResult:
    """
    Result of salience computation.
    
    Now includes all four component scores for spec compliance.
    """
    score: float                        # Final salience [0, 1]
    factors: Dict[str, float] = field(default_factory=dict)  # Component scores
    computed_at: float = field(default_factory=time.time)
    method: str = "composite"
    
    # Adjustment info
    base_score: float = 0.0             # Score before relation adjustments
    relation_boost: float = 0.0         # Total boost from relations
    
    # Component breakdown (spec-compliant)
    semantic: float = 0.0
    recency: float = 0.0
    frequency: float = 0.0
    structural: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 4),
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "computed_at": self.computed_at,
            "method": self.method,
            "base_score": round(self.base_score, 4),
            "relation_boost": round(self.relation_boost, 4),
        }
    
    def to_spec_dict(self) -> Dict[str, float]:
        """
        Convert to spec-compliant dictionary.
        
        Returns:
            {
              "semantic": 0.82,
              "recency": 0.40,
              "frequency": 0.10,
              "structural": 0.15,
              "final": 0.67
            }
        """
        return {
            "semantic": round(self.semantic, 4),
            "recency": round(self.recency, 4),
            "frequency": round(self.frequency, 4),
            "structural": round(self.structural, 4),
            "final": round(self.score, 4),
        }


# ==============================================
# SALIENCE ENGINE
# ==============================================

class SalienceEngine:
    """
    Computes salience scores for events.
    
    Salience determines which memories surface during recall.
    Higher salience = more likely to be retrieved.
    """
    
    def __init__(self, config: Optional[SalienceConfig] = None):
        """
        Initialize salience engine.
        
        Args:
            config: Salience configuration
        """
        self.config = config or SalienceConfig()
    
    def compute(
        self,
        event_id: str,
        timestamp: float,
        access_count: int = 0,
        avg_similarity: float = 0.0,
        structural_score: float = 0.5,
        method: SalienceMethod = SalienceMethod.COMPOSITE,
        now: Optional[float] = None,
    ) -> SalienceResult:
        """
        Compute salience for an event.
        
        Args:
            event_id: Event identifier
            timestamp: Event creation timestamp
            access_count: How many times event was accessed
            avg_similarity: Average similarity to recent queries (semantic)
            structural_score: Structural salience from position analysis
            method: Computation method
            now: Current time (for testing)
        
        Returns:
            SalienceResult with score and factors
        """
        now = now or time.time()
        
        # Compute individual factors
        recency = self._compute_recency(timestamp, now)
        frequency = self._compute_frequency(access_count)
        semantic = avg_similarity  # Already [0, 1]
        structural = structural_score  # From StructuralAnalyzer
        
        factors = {
            "recency": recency,
            "frequency": frequency,
            "semantic": semantic,
            "structural": structural,
        }
        
        # Compute final score based on method
        if method == SalienceMethod.RECENCY:
            score = recency
        elif method == SalienceMethod.FREQUENCY:
            score = frequency
        elif method == SalienceMethod.SEMANTIC:
            score = semantic
        elif method == SalienceMethod.STRUCTURAL:
            score = structural
        else:  # COMPOSITE
            score = (
                self.config.recency_weight * recency +
                self.config.frequency_weight * frequency +
                self.config.semantic_weight * semantic +
                self.config.structural_weight * structural
            )
        
        # Clamp to bounds
        score = max(self.config.min_salience, min(self.config.max_salience, score))
        
        return SalienceResult(
            score=score,
            factors=factors,
            computed_at=now,
            method=method.value,
            base_score=score,
            relation_boost=0.0,
            # Component breakdown
            semantic=semantic,
            recency=recency,
            frequency=frequency,
            structural=structural,
        )
    
    def _compute_recency(self, timestamp: float, now: float) -> float:
        """
        Compute recency factor using exponential decay.
        
        Uses half-life decay: score = min + (1-min) * 2^(-age/halflife)
        """
        age_seconds = now - timestamp
        
        if age_seconds <= 0:
            return 1.0
        
        decay = math.pow(2, -age_seconds / self.config.recency_halflife)
        
        # Scale to [min, 1]
        return self.config.recency_min + (1 - self.config.recency_min) * decay
    
    def _compute_frequency(self, access_count: int) -> float:
        """
        Compute frequency factor with logarithmic scaling.
        
        Uses log scaling to prevent high-access items from dominating.
        """
        if access_count <= 0:
            return 0.0
        
        # Log scaling: log(1 + count) / log(1 + max)
        return math.log(1 + access_count) / math.log(1 + self.config.frequency_max)
    
    def adjust_from_relations(
        self,
        salience: SalienceResult,
        relations: List[RelationResult],
    ) -> SalienceResult:
        """
        Adjust salience based on detected relations.
        
        Adjustments:
        - +0.10 for supersedes (corrections are important)
        - +0.05 for contradicts (conflicts need attention)
        - +0.05 * confidence for expands_on
        - +0.05 if strict_contradiction (high-confidence signals)
        
        Args:
            salience: Base salience result
            relations: Detected relations for this event
        
        Returns:
            Adjusted SalienceResult
        """
        if not relations:
            return salience
        
        boost = 0.0
        boost_reasons = []
        
        for rel in relations:
            if rel.kind == RelationType.SUPERSEDES:
                boost += self.config.supersedes_boost
                boost_reasons.append(f"supersedes:+{self.config.supersedes_boost}")
            
            elif rel.kind == RelationType.CONTRADICTS:
                boost += self.config.contradicts_boost
                boost_reasons.append(f"contradicts:+{self.config.contradicts_boost}")
            
            elif rel.kind == RelationType.EXPANDS_ON:
                expansion_boost = self.config.expands_on_factor * rel.confidence
                boost += expansion_boost
                boost_reasons.append(f"expands_on:+{expansion_boost:.3f}")
            
            # Additional boost for strict contradiction
            if rel.strict_contradiction:
                boost += self.config.strict_contradiction_boost
                boost_reasons.append(f"strict:+{self.config.strict_contradiction_boost}")
        
        # Create adjusted result
        adjusted_score = min(
            salience.score + boost,
            self.config.max_salience
        )
        
        # Copy factors and add relation info
        new_factors = dict(salience.factors)
        new_factors["relation_boost"] = boost
        if boost_reasons:
            new_factors["boost_reasons"] = boost_reasons
        
        return SalienceResult(
            score=adjusted_score,
            factors=new_factors,
            computed_at=salience.computed_at,
            method=salience.method,
            base_score=salience.base_score,
            relation_boost=boost,
            # Preserve component breakdown
            semantic=salience.semantic,
            recency=salience.recency,
            frequency=salience.frequency,
            structural=salience.structural,
        )
    
    def compute_with_relations(
        self,
        event_id: str,
        timestamp: float,
        relations: List[RelationResult],
        access_count: int = 0,
        avg_similarity: float = 0.0,
        structural_score: float = 0.5,
        method: SalienceMethod = SalienceMethod.COMPOSITE,
        now: Optional[float] = None,
    ) -> SalienceResult:
        """
        Compute salience with relation adjustments in one call.
        
        Args:
            event_id: Event identifier
            timestamp: Event creation timestamp
            relations: Detected relations
            access_count: Access count
            avg_similarity: Average similarity (semantic)
            structural_score: Structural salience
            method: Computation method
            now: Current time
        
        Returns:
            SalienceResult with relation adjustments applied
        """
        # Compute base salience
        base = self.compute(
            event_id=event_id,
            timestamp=timestamp,
            access_count=access_count,
            avg_similarity=avg_similarity,
            structural_score=structural_score,
            method=method,
            now=now,
        )
        
        # Apply relation adjustments
        return self.adjust_from_relations(base, relations)


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def compute_salience(
    timestamp: float,
    access_count: int = 0,
    avg_similarity: float = 0.0,
    structural_score: float = 0.5,
    now: Optional[float] = None,
) -> float:
    """
    Convenience function to compute simple salience score.
    
    Args:
        timestamp: Event timestamp
        access_count: Access count
        avg_similarity: Average similarity (semantic)
        structural_score: Structural salience
        now: Current time
    
    Returns:
        Salience score [0, 1]
    """
    engine = SalienceEngine()
    result = engine.compute(
        event_id="",
        timestamp=timestamp,
        access_count=access_count,
        avg_similarity=avg_similarity,
        structural_score=structural_score,
        now=now,
    )
    return result.score


def adjust_salience_from_relations(
    base_salience: float,
    relations: List[RelationResult],
    config: Optional[SalienceConfig] = None,
) -> float:
    """
    Convenience function to adjust salience from relations.
    
    Args:
        base_salience: Base salience score
        relations: Detected relations
        config: Optional configuration
    
    Returns:
        Adjusted salience score
    """
    config = config or SalienceConfig()
    
    # Create a minimal SalienceResult
    base = SalienceResult(
        score=base_salience,
        base_score=base_salience,
    )
    
    engine = SalienceEngine(config)
    adjusted = engine.adjust_from_relations(base, relations)
    
    return adjusted.score


def compute_salience_breakdown(
    timestamp: float,
    access_count: int = 0,
    avg_similarity: float = 0.0,
    structural_score: float = 0.5,
    now: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute salience with full component breakdown.
    
    Args:
        timestamp: Event timestamp
        access_count: Access count
        avg_similarity: Average similarity (semantic)
        structural_score: Structural salience
        now: Current time
    
    Returns:
        Spec-compliant dictionary with all components
    """
    engine = SalienceEngine()
    result = engine.compute(
        event_id="",
        timestamp=timestamp,
        access_count=access_count,
        avg_similarity=avg_similarity,
        structural_score=structural_score,
        now=now,
    )
    return result.to_spec_dict()


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Config and types
    'SalienceConfig',
    'SalienceMethod',
    'SalienceResult',
    
    # Engine
    'SalienceEngine',
    
    # Convenience
    'compute_salience',
    'adjust_salience_from_relations',
    'compute_salience_breakdown',
]
