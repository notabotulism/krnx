"""
KRNX Enrichment - Retention Classification v2

Computes retention_class using drift × salience matrix.
Drift measures how much context has changed since the event.

Matrix:
                    Low Salience        High Salience
    High Drift      ephemeral           consolidation_candidate
    Low Drift       merge_candidate     durable

Override: strict_contradiction → always durable

Constitution-safe: Signals only, no actions, no deletion.
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

from .relations import RelationResult, RelationType

logger = logging.getLogger(__name__)


# ==============================================
# RETENTION CLASSES
# ==============================================

class RetentionClass(Enum):
    """Retention classification for events."""
    EPHEMERAL = "ephemeral"                 # Low salience + high drift → expire soon
    MERGE_CANDIDATE = "merge_candidate"     # Low salience + low drift → consolidate
    CONSOLIDATION_CANDIDATE = "consolidation_candidate"  # High salience + high drift
    DURABLE = "durable"                     # High salience + low drift → keep forever
    
    # Special classes
    PERMANENT = "permanent"                 # Explicitly marked permanent
    AUDIT = "audit"                         # Compliance retention


# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class RetentionConfig:
    """Configuration for retention classification."""
    
    # Salience thresholds
    salience_high_threshold: float = 0.5    # Above = high salience
    salience_low_threshold: float = 0.3     # Below = low salience
    
    # Drift thresholds
    drift_high_threshold: float = 0.55      # Above = high drift (lowered to allow pure time-based drift)
    drift_low_threshold: float = 0.3        # Below = low drift
    
    # Time-based drift decay
    drift_halflife_hours: float = 24.0      # How fast drift increases
    
    # Overrides
    strict_contradiction_class: RetentionClass = RetentionClass.DURABLE


# ==============================================
# DRIFT COMPUTATION
# ==============================================

class DriftComputer:
    """
    Computes context drift for events.
    
    Drift measures how much the surrounding context has changed
    since the event was created. High drift = context moved on.
    
    Factors:
    - Time since creation (older = more drift)
    - Supersession depth (more corrections = more drift)
    - Episode changes (different conversation = more drift)
    """
    
    def __init__(self, config: Optional[RetentionConfig] = None):
        """Initialize drift computer."""
        self.config = config or RetentionConfig()
    
    def compute(
        self,
        timestamp: float,
        relations: List[RelationResult],
        current_episode_id: Optional[str] = None,
        event_episode_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> float:
        """
        Compute drift score for an event.
        
        Args:
            timestamp: Event creation timestamp
            relations: Event's relations
            current_episode_id: Current conversation episode
            event_episode_id: Episode when event was created
            now: Current time
        
        Returns:
            Drift score [0, 1]
        """
        now = now or time.time()
        
        # Time-based drift (exponential approach to 1)
        age_hours = (now - timestamp) / 3600
        halflife = self.config.drift_halflife_hours
        time_drift = 1 - (0.5 ** (age_hours / halflife))
        
        # Supersession drift (if this event is superseded, drift increases)
        supersession_drift = 0.0
        for rel in relations:
            if rel.kind == RelationType.SUPERSEDES:
                # Each level of supersession adds drift
                supersession_drift += 0.1 * rel.confidence
        supersession_drift = min(supersession_drift, 0.5)  # Cap at 0.5
        
        # Episode drift (different episode = more drift)
        episode_drift = 0.0
        if current_episode_id and event_episode_id:
            if current_episode_id != event_episode_id:
                episode_drift = 0.2  # Flat boost for different episode
        
        # Combine (weighted average)
        drift = (
            0.6 * time_drift +
            0.25 * supersession_drift +
            0.15 * episode_drift
        )
        
        return min(1.0, drift)
    
    def compute_batch(
        self,
        events: List[Dict[str, Any]],
        current_episode_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute drift for multiple events.
        
        Args:
            events: List of dicts with timestamp, relations, episode_id
            current_episode_id: Current episode
            now: Current time
        
        Returns:
            Dict of event_id -> drift score
        """
        now = now or time.time()
        results = {}
        
        for event in events:
            event_id = event.get("event_id", str(id(event)))
            timestamp = event.get("timestamp", now)
            relations = event.get("relations", [])
            episode_id = event.get("episode_id")
            
            drift = self.compute(
                timestamp=timestamp,
                relations=relations,
                current_episode_id=current_episode_id,
                event_episode_id=episode_id,
                now=now,
            )
            
            results[event_id] = drift
        
        return results


# ==============================================
# RETENTION RESULT
# ==============================================

@dataclass
class RetentionResult:
    """Result of retention classification."""
    retention_class: RetentionClass
    salience: float
    drift: float
    reason: str
    
    # Flags
    is_strict_contradiction: bool = False
    has_override: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retention_class": self.retention_class.value,
            "salience": round(self.salience, 4),
            "drift": round(self.drift, 4),
            "reason": self.reason,
            "is_strict_contradiction": self.is_strict_contradiction,
            "has_override": self.has_override,
        }


# ==============================================
# RETENTION CLASSIFIER
# ==============================================

class RetentionClassifier:
    """
    Classifies events into retention classes using drift × salience matrix.
    
    Matrix:
                        Low Salience        High Salience
        High Drift      ephemeral           consolidation_candidate
        Low Drift       merge_candidate     durable
    
    Override: strict_contradiction → always durable
    """
    
    def __init__(self, config: Optional[RetentionConfig] = None):
        """Initialize retention classifier."""
        self.config = config or RetentionConfig()
        self._drift_computer = DriftComputer(config)
    
    def classify(
        self,
        salience: float,
        drift: float,
        relations: List[RelationResult],
        explicit_class: Optional[str] = None,
    ) -> RetentionResult:
        """
        Classify an event into a retention class.
        
        Args:
            salience: Salience score [0, 1]
            drift: Drift score [0, 1]
            relations: Event's relations (for strict_contradiction check)
            explicit_class: Explicit retention class (overrides matrix)
        
        Returns:
            RetentionResult with classification
        """
        # Check for explicit override
        if explicit_class:
            try:
                explicit = RetentionClass(explicit_class)
                return RetentionResult(
                    retention_class=explicit,
                    salience=salience,
                    drift=drift,
                    reason=f"explicit: {explicit_class}",
                    has_override=True,
                )
            except ValueError:
                pass  # Invalid explicit class, continue with matrix
        
        # Check for strict contradiction override
        has_strict = any(r.strict_contradiction for r in relations)
        if has_strict:
            return RetentionResult(
                retention_class=self.config.strict_contradiction_class,
                salience=salience,
                drift=drift,
                reason="strict_contradiction override",
                is_strict_contradiction=True,
                has_override=True,
            )
        
        # Apply drift × salience matrix
        high_salience = salience >= self.config.salience_high_threshold
        low_salience = salience < self.config.salience_low_threshold
        high_drift = drift >= self.config.drift_high_threshold
        low_drift = drift < self.config.drift_low_threshold
        
        if high_drift and low_salience:
            return RetentionResult(
                retention_class=RetentionClass.EPHEMERAL,
                salience=salience,
                drift=drift,
                reason=f"matrix: high_drift({drift:.2f}) + low_salience({salience:.2f})",
            )
        
        elif high_drift and high_salience:
            return RetentionResult(
                retention_class=RetentionClass.CONSOLIDATION_CANDIDATE,
                salience=salience,
                drift=drift,
                reason=f"matrix: high_drift({drift:.2f}) + high_salience({salience:.2f})",
            )
        
        elif low_drift and low_salience:
            return RetentionResult(
                retention_class=RetentionClass.MERGE_CANDIDATE,
                salience=salience,
                drift=drift,
                reason=f"matrix: low_drift({drift:.2f}) + low_salience({salience:.2f})",
            )
        
        elif low_drift and high_salience:
            return RetentionResult(
                retention_class=RetentionClass.DURABLE,
                salience=salience,
                drift=drift,
                reason=f"matrix: low_drift({drift:.2f}) + high_salience({salience:.2f})",
            )
        
        else:
            # Middle ground - default to merge_candidate (conservative)
            return RetentionResult(
                retention_class=RetentionClass.MERGE_CANDIDATE,
                salience=salience,
                drift=drift,
                reason=f"matrix: mid_drift({drift:.2f}) + mid_salience({salience:.2f})",
            )
    
    def classify_with_drift(
        self,
        timestamp: float,
        salience: float,
        relations: List[RelationResult],
        current_episode_id: Optional[str] = None,
        event_episode_id: Optional[str] = None,
        explicit_class: Optional[str] = None,
        now: Optional[float] = None,
    ) -> RetentionResult:
        """
        Classify event with automatic drift computation.
        
        Args:
            timestamp: Event creation timestamp
            salience: Salience score
            relations: Event's relations
            current_episode_id: Current conversation episode
            event_episode_id: Episode when event was created
            explicit_class: Explicit retention class
            now: Current time
        
        Returns:
            RetentionResult with classification
        """
        # Compute drift
        drift = self._drift_computer.compute(
            timestamp=timestamp,
            relations=relations,
            current_episode_id=current_episode_id,
            event_episode_id=event_episode_id,
            now=now,
        )
        
        # Classify
        return self.classify(
            salience=salience,
            drift=drift,
            relations=relations,
            explicit_class=explicit_class,
        )
    
    def batch_classify(
        self,
        events: List[Dict[str, Any]],
        current_episode_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, RetentionResult]:
        """
        Classify multiple events.
        
        Args:
            events: List of dicts with timestamp, salience, relations, etc.
            current_episode_id: Current episode
            now: Current time
        
        Returns:
            Dict of event_id -> RetentionResult
        """
        now = now or time.time()
        results = {}
        
        for event in events:
            event_id = event.get("event_id", str(id(event)))
            
            result = self.classify_with_drift(
                timestamp=event.get("timestamp", now),
                salience=event.get("salience", 0.5),
                relations=event.get("relations", []),
                current_episode_id=current_episode_id,
                event_episode_id=event.get("episode_id"),
                explicit_class=event.get("retention_class"),
                now=now,
            )
            
            results[event_id] = result
        
        return results


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def compute_retention_class(
    timestamp: float,
    salience: float,
    relations: List[RelationResult],
    now: Optional[float] = None,
) -> str:
    """
    Convenience function to compute retention class.
    
    Args:
        timestamp: Event timestamp
        salience: Salience score
        relations: Event relations
        now: Current time
    
    Returns:
        Retention class string
    """
    classifier = RetentionClassifier()
    result = classifier.classify_with_drift(
        timestamp=timestamp,
        salience=salience,
        relations=relations,
        now=now,
    )
    return result.retention_class.value


def compute_drift(
    timestamp: float,
    relations: List[RelationResult],
    now: Optional[float] = None,
) -> float:
    """
    Convenience function to compute drift.
    
    Args:
        timestamp: Event timestamp
        relations: Event relations
        now: Current time
    
    Returns:
        Drift score [0, 1]
    """
    computer = DriftComputer()
    return computer.compute(
        timestamp=timestamp,
        relations=relations,
        now=now,
    )


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Classes
    'RetentionClass',
    'RetentionConfig',
    'RetentionResult',
    
    # Engines
    'DriftComputer',
    'RetentionClassifier',
    
    # Convenience
    'compute_retention_class',
    'compute_drift',
]
