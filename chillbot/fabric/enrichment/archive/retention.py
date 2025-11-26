"""
KRNX Fabric - Retention Signaler

Computes retention SIGNALS (not policies).
Signals are metadata suggestions - the application decides what to do.

Signals:
- retention_class: ephemeral, standard, audit, archive
- consolidation_candidate: True if old + low salience
- expiry_suggested_at: When this event could be expired

Constitution-safe:
- Signals only, no actions
- No deletion
- No autonomous behavior
- Application decides what signals mean

Usage:
    signaler = RetentionSignaler()
    
    signals = signaler.compute(
        timestamp=event.timestamp,
        salience_score=0.25,
        retention_class="standard",
    )
    
    if signals.consolidation_candidate:
        # Application decides to consolidate
        pass
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class RetentionSignals:
    """Retention signals for an event."""
    retention_class: Optional[str] = None
    consolidation_candidate: bool = False
    expiry_suggested_at: Optional[float] = None
    
    # Diagnostic info
    age_days: float = 0.0
    salience_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "consolidation_candidate": self.consolidation_candidate,
        }
        if self.retention_class:
            result["retention_class"] = self.retention_class
        if self.expiry_suggested_at:
            result["expiry_suggested_at"] = self.expiry_suggested_at
        return result


class RetentionSignaler:
    """
    Computes retention signals for events.
    
    Signals are suggestions based on:
    - Event age
    - Salience score
    - Retention class
    
    Application decides what to do with signals.
    """
    
    def __init__(
        self,
        ephemeral_ttl_hours: float = 1.0,
        consolidation_age_days: float = 7.0,
        consolidation_salience_ceiling: float = 0.3,
        archive_age_days: float = 30.0,
    ):
        """
        Initialize retention signaler.
        
        Args:
            ephemeral_ttl_hours: TTL for ephemeral events
            consolidation_age_days: Age before consolidation candidate
            consolidation_salience_ceiling: Max salience for consolidation
            archive_age_days: Age before archive suggestion
        """
        self.ephemeral_ttl_hours = ephemeral_ttl_hours
        self.consolidation_age_days = consolidation_age_days
        self.consolidation_salience_ceiling = consolidation_salience_ceiling
        self.archive_age_days = archive_age_days
    
    def compute(
        self,
        timestamp: float,
        salience_score: Optional[float] = None,
        retention_class: Optional[str] = None,
        now: Optional[float] = None,
    ) -> RetentionSignals:
        """
        Compute retention signals for an event.
        
        Args:
            timestamp: Event creation timestamp
            salience_score: Current salience (0-1)
            retention_class: Event's retention class
            now: Current time (for testing)
        
        Returns:
            RetentionSignals with computed values
        """
        now = now or time.time()
        age_seconds = now - timestamp
        age_days = age_seconds / 86400
        
        signals = RetentionSignals(
            retention_class=retention_class,
            age_days=age_days,
            salience_score=salience_score,
        )
        
        # Handle by retention class
        if retention_class == "ephemeral":
            signals = self._compute_ephemeral(signals, age_seconds, now)
        elif retention_class == "permanent":
            # Never expire, never consolidate
            pass
        elif retention_class == "audit":
            # Archive after threshold, never delete
            if age_days >= self.archive_age_days:
                signals.retention_class = "archive"
        else:
            # Standard retention
            signals = self._compute_standard(signals, age_days, salience_score, now)
        
        return signals
    
    def _compute_ephemeral(
        self,
        signals: RetentionSignals,
        age_seconds: float,
        now: float,
    ) -> RetentionSignals:
        """Compute signals for ephemeral events."""
        ttl_seconds = self.ephemeral_ttl_hours * 3600
        
        if age_seconds >= ttl_seconds:
            signals.expiry_suggested_at = now
        else:
            signals.expiry_suggested_at = now + (ttl_seconds - age_seconds)
        
        return signals
    
    def _compute_standard(
        self,
        signals: RetentionSignals,
        age_days: float,
        salience_score: Optional[float],
        now: float,
    ) -> RetentionSignals:
        """Compute signals for standard events."""
        
        # Consolidation candidate: old + low salience
        if age_days >= self.consolidation_age_days:
            if salience_score is not None and salience_score <= self.consolidation_salience_ceiling:
                signals.consolidation_candidate = True
        
        # Archive suggestion
        if age_days >= self.archive_age_days:
            signals.retention_class = "archive"
        
        return signals
    
    def should_consolidate(
        self,
        timestamp: float,
        salience_score: float,
        now: Optional[float] = None,
    ) -> bool:
        """
        Quick check if an event is a consolidation candidate.
        
        Args:
            timestamp: Event timestamp
            salience_score: Current salience
            now: Current time
        
        Returns:
            True if consolidation candidate
        """
        signals = self.compute(
            timestamp=timestamp,
            salience_score=salience_score,
            now=now,
        )
        return signals.consolidation_candidate
    
    def batch_compute(
        self,
        events: list,
        now: Optional[float] = None,
    ) -> list:
        """
        Compute signals for multiple events.
        
        Args:
            events: List of dicts with timestamp, salience_score, retention_class
            now: Current time
        
        Returns:
            List of RetentionSignals
        """
        now = now or time.time()
        
        return [
            self.compute(
                timestamp=e.get("timestamp", now),
                salience_score=e.get("salience_score"),
                retention_class=e.get("retention_class"),
                now=now,
            )
            for e in events
        ]
    
    def find_consolidation_candidates(
        self,
        events: list,
        now: Optional[float] = None,
    ) -> list:
        """
        Find events that are consolidation candidates.
        
        Args:
            events: List of event dicts
            now: Current time
        
        Returns:
            List of event dicts that are candidates
        """
        now = now or time.time()
        candidates = []
        
        for e in events:
            signals = self.compute(
                timestamp=e.get("timestamp", now),
                salience_score=e.get("salience_score"),
                retention_class=e.get("retention_class"),
                now=now,
            )
            if signals.consolidation_candidate:
                candidates.append(e)
        
        return candidates
    
    def get_stats(self) -> Dict[str, Any]:
        """Get signaler configuration."""
        return {
            "ephemeral_ttl_hours": self.ephemeral_ttl_hours,
            "consolidation_age_days": self.consolidation_age_days,
            "consolidation_salience_ceiling": self.consolidation_salience_ceiling,
            "archive_age_days": self.archive_age_days,
        }
    
    def __repr__(self) -> str:
        return (
            f"RetentionSignaler("
            f"consolidate_after={self.consolidation_age_days}d, "
            f"salience<{self.consolidation_salience_ceiling})"
        )
