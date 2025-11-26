"""
KRNX Compute - Salience Scoring

Computes importance scores for memories. Pure compute primitive -
provides methods, not opinions about what scores mean.

Scoring methods:
- recency: Time-based decay
- frequency: Access count based
- semantic: Vector space centrality
- explicit: User-provided scores
- composite: Weighted combination

Usage:
    engine = SalienceEngine()
    
    # Compute score for an event
    score = engine.compute(
        event_id="evt_123",
        timestamp=1704067200,
        access_count=5,
        method=SalienceMethod.COMPOSITE
    )
    
    print(f"Score: {score.score:.3f}")
    print(f"Factors: {score.factors}")
"""

import math
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SalienceMethod(Enum):
    """Available salience scoring methods."""
    RECENCY = "recency"           # Pure time decay
    FREQUENCY = "frequency"       # Access count based
    SEMANTIC = "semantic"         # Vector space centrality
    EXPLICIT = "explicit"         # User-provided score
    COMPOSITE = "composite"       # Weighted combination


@dataclass
class SalienceScore:
    """Result of salience computation."""
    event_id: str
    method: SalienceMethod
    score: float                    # 0.0 to 1.0
    computed_at: float
    factors: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "method": self.method.value,
            "score": self.score,
            "computed_at": self.computed_at,
            "factors": self.factors,
        }


@dataclass
class SalienceConfig:
    """Configuration for salience computation."""
    # Time decay parameters
    recency_halflife: float = 86400 * 7     # 7 days
    recency_min: float = 0.01               # Minimum recency score
    
    # Frequency parameters
    frequency_max: int = 100                # Access count for max score
    frequency_log_base: float = 10.0        # Logarithmic scaling base
    
    # Semantic parameters (for vector centrality)
    semantic_top_k: int = 50                # Neighbors to consider
    
    # Composite weights (must sum to 1.0)
    recency_weight: float = 0.4
    frequency_weight: float = 0.3
    semantic_weight: float = 0.3
    
    def validate(self):
        """Validate configuration."""
        total = self.recency_weight + self.frequency_weight + self.semantic_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Composite weights must sum to 1.0, got {total}")


class SalienceEngine:
    """
    Compute salience (importance) scores for memories.
    
    This is a pure compute primitive. It provides scoring methods,
    but does NOT decide what scores mean or how to use them.
    Applications decide thresholds and policies.
    """
    
    def __init__(self, config: Optional[SalienceConfig] = None):
        """
        Initialize salience engine.
        
        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or SalienceConfig()
        self.config.validate()
    
    def recency_score(
        self,
        timestamp: float,
        now: Optional[float] = None,
    ) -> float:
        """
        Compute recency score using exponential decay.
        
        Newer = higher score. Uses half-life decay model.
        
        Args:
            timestamp: Event timestamp
            now: Current time (defaults to time.time())
        
        Returns:
            Score from 0.0 to 1.0
        """
        now = now or time.time()
        
        if timestamp > now:
            return 1.0
        
        age = now - timestamp
        halflife = self.config.recency_halflife
        
        # Exponential decay: score = 0.5^(age / halflife)
        score = math.pow(0.5, age / halflife)
        
        # Apply minimum
        return max(self.config.recency_min, score)
    
    def frequency_score(
        self,
        access_count: int,
    ) -> float:
        """
        Compute frequency score based on access count.
        
        More accessed = higher score. Uses logarithmic scaling
        to prevent runaway scores.
        
        Args:
            access_count: Number of times memory was accessed
        
        Returns:
            Score from 0.0 to 1.0
        """
        if access_count <= 0:
            return 0.0
        
        max_count = self.config.frequency_max
        log_base = self.config.frequency_log_base
        
        # Logarithmic scaling: log(count + 1) / log(max + 1)
        score = math.log(access_count + 1, log_base) / math.log(max_count + 1, log_base)
        
        return min(1.0, max(0.0, score))
    
    def semantic_score(
        self,
        avg_similarity: float,
    ) -> float:
        """
        Compute semantic score based on vector space centrality.
        
        More central (similar to many others) = higher score.
        
        Args:
            avg_similarity: Average similarity to other memories
        
        Returns:
            Score from 0.0 to 1.0
        """
        # Already in 0-1 range from cosine similarity
        return max(0.0, min(1.0, avg_similarity))
    
    def explicit_score(
        self,
        score: float,
    ) -> float:
        """
        Normalize an explicitly provided score.
        
        Args:
            score: User-provided score (any range)
        
        Returns:
            Score clamped to 0.0 to 1.0
        """
        return max(0.0, min(1.0, score))
    
    def composite_score(
        self,
        recency: float,
        frequency: float,
        semantic: float,
    ) -> float:
        """
        Compute weighted combination of all factors.
        
        Args:
            recency: Recency score (0-1)
            frequency: Frequency score (0-1)
            semantic: Semantic score (0-1)
        
        Returns:
            Weighted composite score (0-1)
        """
        return (
            self.config.recency_weight * recency +
            self.config.frequency_weight * frequency +
            self.config.semantic_weight * semantic
        )
    
    def compute(
        self,
        event_id: str,
        timestamp: float,
        access_count: int = 0,
        avg_similarity: float = 0.0,
        explicit_score: Optional[float] = None,
        method: SalienceMethod = SalienceMethod.COMPOSITE,
        now: Optional[float] = None,
    ) -> SalienceScore:
        """
        Compute salience score for an event.
        
        Args:
            event_id: Event identifier
            timestamp: Event creation timestamp
            access_count: Number of times accessed
            avg_similarity: Average similarity to other memories
            explicit_score: User-provided score (for EXPLICIT method)
            method: Scoring method to use
            now: Current time for recency calculation
        
        Returns:
            SalienceScore with score and factor breakdown
        """
        now = now or time.time()
        
        # Compute individual factors
        recency = self.recency_score(timestamp, now)
        frequency = self.frequency_score(access_count)
        semantic = self.semantic_score(avg_similarity)
        
        factors = {
            "recency": recency,
            "frequency": frequency,
            "semantic": semantic,
        }
        
        # Compute final score based on method
        if method == SalienceMethod.RECENCY:
            score = recency
        elif method == SalienceMethod.FREQUENCY:
            score = frequency
        elif method == SalienceMethod.SEMANTIC:
            score = semantic
        elif method == SalienceMethod.EXPLICIT:
            if explicit_score is None:
                raise ValueError("explicit_score required for EXPLICIT method")
            score = self.explicit_score(explicit_score)
            factors["explicit"] = score
        elif method == SalienceMethod.COMPOSITE:
            score = self.composite_score(recency, frequency, semantic)
        else:
            score = self.composite_score(recency, frequency, semantic)
        
        return SalienceScore(
            event_id=event_id,
            method=method,
            score=score,
            computed_at=now,
            factors=factors,
        )
    
    def compute_batch(
        self,
        events: List[Dict[str, Any]],
        method: SalienceMethod = SalienceMethod.COMPOSITE,
        now: Optional[float] = None,
    ) -> List[SalienceScore]:
        """
        Compute salience scores for multiple events.
        
        Args:
            events: List of event dicts with keys:
                - event_id: str
                - timestamp: float
                - access_count: int (optional)
                - avg_similarity: float (optional)
            method: Scoring method to use
            now: Current time for recency calculation
        
        Returns:
            List of SalienceScore objects
        """
        now = now or time.time()
        
        return [
            self.compute(
                event_id=e["event_id"],
                timestamp=e["timestamp"],
                access_count=e.get("access_count", 0),
                avg_similarity=e.get("avg_similarity", 0.0),
                method=method,
                now=now,
            )
            for e in events
        ]
    
    def rank_by_salience(
        self,
        events: List[Dict[str, Any]],
        method: SalienceMethod = SalienceMethod.COMPOSITE,
        now: Optional[float] = None,
    ) -> List[tuple]:
        """
        Rank events by salience score.
        
        Args:
            events: List of event dicts
            method: Scoring method
            now: Current time
        
        Returns:
            List of (event_dict, SalienceScore) tuples, sorted by score descending
        """
        scores = self.compute_batch(events, method, now)
        paired = list(zip(events, scores))
        paired.sort(key=lambda x: x[1].score, reverse=True)
        return paired
    
    def decay_curve(
        self,
        hours: int = 168,
        interval: int = 1,
    ) -> List[tuple]:
        """
        Generate decay curve for visualization.
        
        Args:
            hours: Total hours to plot
            interval: Hour interval between points
        
        Returns:
            List of (hours_ago, score) tuples
        """
        now = time.time()
        points = []
        
        for h in range(0, hours + 1, interval):
            timestamp = now - (h * 3600)
            score = self.recency_score(timestamp, now)
            points.append((h, score))
        
        return points
    
    def threshold_age(
        self,
        threshold: float = 0.5,
    ) -> float:
        """
        Calculate age (in seconds) at which recency score hits threshold.
        
        Args:
            threshold: Score threshold (0-1)
        
        Returns:
            Age in seconds
        """
        halflife = self.config.recency_halflife
        
        # Solve: threshold = 0.5^(age / halflife)
        # age = halflife * log(threshold) / log(0.5)
        if threshold <= 0 or threshold >= 1:
            return float('inf')
        
        return halflife * math.log(threshold) / math.log(0.5)
    
    def update_config(
        self,
        recency_weight: Optional[float] = None,
        frequency_weight: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        recency_halflife: Optional[float] = None,
    ):
        """
        Update configuration parameters.
        
        Args:
            recency_weight: New recency weight
            frequency_weight: New frequency weight
            semantic_weight: New semantic weight
            recency_halflife: New recency half-life in seconds
        """
        if recency_weight is not None:
            self.config.recency_weight = recency_weight
        if frequency_weight is not None:
            self.config.frequency_weight = frequency_weight
        if semantic_weight is not None:
            self.config.semantic_weight = semantic_weight
        if recency_halflife is not None:
            self.config.recency_halflife = recency_halflife
        
        self.config.validate()
    
    def __repr__(self) -> str:
        return (
            f"SalienceEngine("
            f"halflife={self.config.recency_halflife/86400:.1f}d, "
            f"weights=[r:{self.config.recency_weight}, "
            f"f:{self.config.frequency_weight}, "
            f"s:{self.config.semantic_weight}])"
        )
