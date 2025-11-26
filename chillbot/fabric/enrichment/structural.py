"""
KRNX Enrichment - Structural Signal Computation

Computes structural signals that aren't captured by semantic similarity:
- Event density (events per minute in window)
- Episode boundary detection
- Structural salience (position-based importance)
- Reply chain detection

These signals help downstream systems understand:
- Conversation pacing
- Topic boundaries
- Event importance based on position

Constitution-safe: Pure, deterministic, no side effects.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class StructuralConfig:
    """Configuration for structural signal computation."""
    
    # Episode boundary detection
    episode_gap_threshold: float = 300.0    # 5 min = new episode
    
    # Event density windows
    density_window_seconds: float = 60.0    # 1 minute window
    high_density_threshold: float = 10.0    # >10 events/min = high density
    low_density_threshold: float = 1.0      # <1 event/min = low density
    
    # Structural salience
    first_in_episode_boost: float = 0.2     # First event in episode
    reply_to_question_boost: float = 0.1    # Reply to a question
    correction_boost: float = 0.15          # Correction/update event
    
    # Reply chain
    max_reply_chain_depth: int = 10         # Max chain to follow


# ==============================================
# EVENT DENSITY
# ==============================================

@dataclass
class DensityResult:
    """Result of event density computation."""
    events_per_minute: float
    density_class: str          # 'high', 'normal', 'low'
    window_seconds: float
    event_count_in_window: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "events_per_minute": round(self.events_per_minute, 2),
            "density_class": self.density_class,
            "window_seconds": self.window_seconds,
            "event_count_in_window": self.event_count_in_window,
        }


class DensityComputer:
    """
    Computes event density in a time window.
    
    High density = rapid conversation (e.g., Q&A session)
    Low density = sparse events (e.g., daily notes)
    """
    
    def __init__(self, config: Optional[StructuralConfig] = None):
        """Initialize density computer."""
        self.config = config or StructuralConfig()
    
    def compute(
        self,
        event_timestamp: float,
        recent_events: List[Any],
        window_seconds: Optional[float] = None,
    ) -> DensityResult:
        """
        Compute event density around a timestamp.
        
        Args:
            event_timestamp: Timestamp of the event
            recent_events: List of recent events with timestamps
            window_seconds: Window size (default from config)
        
        Returns:
            DensityResult with events per minute
        """
        window = window_seconds or self.config.density_window_seconds
        
        # Count events in window
        window_start = event_timestamp - window
        count = 0
        
        for event in recent_events:
            ts = getattr(event, 'timestamp', 0)
            if window_start <= ts <= event_timestamp:
                count += 1
        
        # Compute rate
        minutes = window / 60.0
        events_per_minute = count / minutes if minutes > 0 else 0.0
        
        # Classify density
        if events_per_minute >= self.config.high_density_threshold:
            density_class = "high"
        elif events_per_minute <= self.config.low_density_threshold:
            density_class = "low"
        else:
            density_class = "normal"
        
        return DensityResult(
            events_per_minute=events_per_minute,
            density_class=density_class,
            window_seconds=window,
            event_count_in_window=count,
        )


# ==============================================
# EPISODE BOUNDARY DETECTION
# ==============================================

@dataclass
class BoundaryResult:
    """Result of episode boundary detection."""
    is_boundary: bool
    gap_seconds: Optional[float]
    reason: str
    new_episode_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_boundary": self.is_boundary,
            "gap_seconds": self.gap_seconds,
            "reason": self.reason,
            "new_episode_id": self.new_episode_id,
        }


class BoundaryDetector:
    """
    Detects episode boundaries.
    
    An episode boundary occurs when:
    - Time gap exceeds threshold
    - Topic appears to change significantly
    - User explicitly starts new conversation
    """
    
    def __init__(self, config: Optional[StructuralConfig] = None):
        """Initialize boundary detector."""
        self.config = config or StructuralConfig()
    
    def detect(
        self,
        event_timestamp: float,
        previous_event: Optional[Any] = None,
        topic_similarity: Optional[float] = None,
    ) -> BoundaryResult:
        """
        Detect if event marks an episode boundary.
        
        Args:
            event_timestamp: Event timestamp
            previous_event: Previous event (if any)
            topic_similarity: Semantic similarity to previous (optional)
        
        Returns:
            BoundaryResult
        """
        # No previous event = definitely a boundary
        if previous_event is None:
            return BoundaryResult(
                is_boundary=True,
                gap_seconds=None,
                reason="no_previous_event",
            )
        
        # Compute time gap
        prev_ts = getattr(previous_event, 'timestamp', 0)
        gap = event_timestamp - prev_ts
        
        # Time-based boundary
        if gap >= self.config.episode_gap_threshold:
            return BoundaryResult(
                is_boundary=True,
                gap_seconds=gap,
                reason=f"time_gap_exceeded ({gap:.0f}s > {self.config.episode_gap_threshold}s)",
            )
        
        # Topic-based boundary (if similarity provided)
        if topic_similarity is not None and topic_similarity < 0.3:
            return BoundaryResult(
                is_boundary=True,
                gap_seconds=gap,
                reason=f"topic_shift (similarity={topic_similarity:.2f})",
            )
        
        # Not a boundary
        return BoundaryResult(
            is_boundary=False,
            gap_seconds=gap,
            reason="continuation",
        )


# ==============================================
# STRUCTURAL SALIENCE
# ==============================================

@dataclass
class StructuralSalienceResult:
    """Result of structural salience computation."""
    score: float                # [0, 1]
    factors: Dict[str, float]   # Component scores
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 4),
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
        }


class StructuralSalienceComputer:
    """
    Computes structural salience (position-based importance).
    
    Factors:
    - First in episode (high importance)
    - Reply to question (response is important)
    - Correction event (updates are important)
    - High-density context (rapid exchange = less individual importance)
    """
    
    def __init__(self, config: Optional[StructuralConfig] = None):
        """Initialize structural salience computer."""
        self.config = config or StructuralConfig()
    
    def compute(
        self,
        is_boundary: bool = False,
        is_correction: bool = False,
        reply_pattern: Optional[str] = None,
        density_class: str = "normal",
        relation_count: int = 0,
    ) -> StructuralSalienceResult:
        """
        Compute structural salience.
        
        Args:
            is_boundary: Whether this is an episode boundary
            is_correction: Whether this is a correction/update
            reply_pattern: Reply pattern type if detected
            density_class: Event density class
            relation_count: Number of relations detected
        
        Returns:
            StructuralSalienceResult
        """
        factors = {}
        score = 0.5  # Base score
        
        # Episode boundary boost
        if is_boundary:
            factors["first_in_episode"] = self.config.first_in_episode_boost
            score += self.config.first_in_episode_boost
        
        # Correction boost
        if is_correction:
            factors["correction"] = self.config.correction_boost
            score += self.config.correction_boost
        
        # Reply pattern adjustments
        if reply_pattern == "correction":
            factors["reply_correction"] = 0.1
            score += 0.1
        elif reply_pattern == "question":
            factors["question"] = 0.05
            score += 0.05
        elif reply_pattern == "acknowledgment":
            factors["acknowledgment"] = -0.05
            score -= 0.05
        
        # Density adjustments
        if density_class == "high":
            factors["high_density_penalty"] = -0.1
            score -= 0.1
        elif density_class == "low":
            factors["low_density_bonus"] = 0.05
            score += 0.05
        
        # Relation count bonus (more connected = more important)
        if relation_count > 0:
            relation_bonus = min(0.1, relation_count * 0.02)
            factors["relation_connections"] = relation_bonus
            score += relation_bonus
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return StructuralSalienceResult(
            score=score,
            factors=factors,
        )


# ==============================================
# UNIFIED STRUCTURAL ANALYZER
# ==============================================

@dataclass
class StructuralAnalysis:
    """Complete structural analysis result."""
    density: DensityResult
    boundary: BoundaryResult
    salience: StructuralSalienceResult
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "density": self.density.to_dict(),
            "boundary": self.boundary.to_dict(),
            "salience": self.salience.to_dict(),
        }


class StructuralAnalyzer:
    """
    Unified structural signal analyzer.
    
    Combines density, boundary, and salience computation.
    """
    
    def __init__(self, config: Optional[StructuralConfig] = None):
        """Initialize structural analyzer."""
        self.config = config or StructuralConfig()
        self._density = DensityComputer(config)
        self._boundary = BoundaryDetector(config)
        self._salience = StructuralSalienceComputer(config)
    
    def analyze(
        self,
        event: Any,
        previous_event: Optional[Any] = None,
        recent_events: Optional[List[Any]] = None,
        topic_similarity: Optional[float] = None,
        is_correction: bool = False,
        reply_pattern: Optional[str] = None,
        relation_count: int = 0,
    ) -> StructuralAnalysis:
        """
        Perform complete structural analysis.
        
        Args:
            event: The event to analyze
            previous_event: Previous event (for boundary detection)
            recent_events: Recent events (for density computation)
            topic_similarity: Semantic similarity to previous
            is_correction: Whether this is a correction
            reply_pattern: Detected reply pattern
            relation_count: Number of relations
        
        Returns:
            StructuralAnalysis with all signals
        """
        timestamp = getattr(event, 'timestamp', 0)
        recent = recent_events or []
        
        # Compute density
        density = self._density.compute(
            event_timestamp=timestamp,
            recent_events=recent,
        )
        
        # Detect boundary
        boundary = self._boundary.detect(
            event_timestamp=timestamp,
            previous_event=previous_event,
            topic_similarity=topic_similarity,
        )
        
        # Compute structural salience
        salience = self._salience.compute(
            is_boundary=boundary.is_boundary,
            is_correction=is_correction,
            reply_pattern=reply_pattern,
            density_class=density.density_class,
            relation_count=relation_count,
        )
        
        return StructuralAnalysis(
            density=density,
            boundary=boundary,
            salience=salience,
        )


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def compute_event_density(
    event_timestamp: float,
    recent_events: List[Any],
    window_seconds: float = 60.0,
) -> float:
    """
    Convenience function to compute events per minute.
    
    Args:
        event_timestamp: Event timestamp
        recent_events: Recent events
        window_seconds: Window size
    
    Returns:
        Events per minute
    """
    computer = DensityComputer()
    result = computer.compute(event_timestamp, recent_events, window_seconds)
    return result.events_per_minute


def is_episode_boundary(
    event_timestamp: float,
    previous_event: Optional[Any] = None,
    gap_threshold: float = 300.0,
) -> bool:
    """
    Convenience function to check episode boundary.
    
    Args:
        event_timestamp: Event timestamp
        previous_event: Previous event
        gap_threshold: Gap threshold in seconds
    
    Returns:
        True if this is an episode boundary
    """
    config = StructuralConfig(episode_gap_threshold=gap_threshold)
    detector = BoundaryDetector(config)
    result = detector.detect(event_timestamp, previous_event)
    return result.is_boundary


def compute_structural_salience(
    is_boundary: bool = False,
    is_correction: bool = False,
    relation_count: int = 0,
) -> float:
    """
    Convenience function to compute structural salience.
    
    Args:
        is_boundary: Episode boundary
        is_correction: Correction event
        relation_count: Number of relations
    
    Returns:
        Structural salience score [0, 1]
    """
    computer = StructuralSalienceComputer()
    result = computer.compute(
        is_boundary=is_boundary,
        is_correction=is_correction,
        relation_count=relation_count,
    )
    return result.score


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Config
    'StructuralConfig',
    
    # Results
    'DensityResult',
    'BoundaryResult',
    'StructuralSalienceResult',
    'StructuralAnalysis',
    
    # Computers
    'DensityComputer',
    'BoundaryDetector',
    'StructuralSalienceComputer',
    'StructuralAnalyzer',
    
    # Convenience
    'compute_event_density',
    'is_episode_boundary',
    'compute_structural_salience',
]
