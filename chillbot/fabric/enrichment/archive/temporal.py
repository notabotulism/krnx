"""
KRNX Fabric - Temporal Enricher

Computes temporal metadata:
- time_gap_seconds: Time since previous event
- episode_id: Groups events into conversation episodes
- last_access: Timestamp of enrichment

Episode threading rule:
- If gap < threshold → same episode
- If gap >= threshold → new episode

Constitution-safe: Pure, deterministic, no side effects.
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class EpisodeTracker:
    """
    Tracks episode IDs across events.
    
    Stateless per-call, but can maintain state across
    a session if needed.
    """
    
    def __init__(self, gap_threshold: float = 300.0):
        """
        Initialize episode tracker.
        
        Args:
            gap_threshold: Seconds of silence before new episode (default 5 min)
        """
        self.gap_threshold = gap_threshold
        self._current_episode: Optional[str] = None
        self._last_timestamp: Optional[float] = None
    
    def get_episode(
        self,
        timestamp: float,
        previous_timestamp: Optional[float] = None,
        previous_episode_id: Optional[str] = None,
    ) -> str:
        """
        Get episode ID for a timestamp.
        
        Args:
            timestamp: Current event timestamp
            previous_timestamp: Previous event timestamp
            previous_episode_id: Previous event's episode ID
        
        Returns:
            Episode ID (new or continued)
        """
        # If no previous event, start new episode
        if previous_timestamp is None:
            return self._new_episode()
        
        # Calculate gap
        gap = timestamp - previous_timestamp
        
        # If gap exceeds threshold, new episode
        if gap >= self.gap_threshold:
            return self._new_episode()
        
        # Continue previous episode
        if previous_episode_id:
            return previous_episode_id
        
        # Fallback: new episode
        return self._new_episode()
    
    def _new_episode(self) -> str:
        """Generate new episode ID."""
        return f"ep_{uuid.uuid4().hex[:12]}"
    
    def reset(self):
        """Reset tracker state."""
        self._current_episode = None
        self._last_timestamp = None


class TemporalEnricher:
    """
    Enriches events with temporal metadata.
    
    Computes:
    - time_gap_seconds: Seconds since previous event
    - episode_id: Conversation episode grouping
    - last_access: When this enrichment was computed
    """
    
    def __init__(self, episode_gap_threshold: float = 300.0):
        """
        Initialize temporal enricher.
        
        Args:
            episode_gap_threshold: Seconds before new episode (default 5 min)
        """
        self.episode_tracker = EpisodeTracker(gap_threshold=episode_gap_threshold)
    
    def enrich(
        self,
        timestamp: float,
        previous_event: Optional[Any] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute temporal metadata for an event.
        
        Args:
            timestamp: Event timestamp
            previous_event: Previous event in stream (if any)
            now: Current time (for deterministic testing)
        
        Returns:
            Dict with temporal metadata:
            - time_gap_seconds: float or None
            - episode_id: str
            - last_access: float
        """
        now = now or time.time()
        
        # Extract previous event info
        prev_timestamp = None
        prev_episode_id = None
        
        if previous_event is not None:
            prev_timestamp = getattr(previous_event, 'timestamp', None)
            
            # Check metadata for episode_id
            prev_metadata = getattr(previous_event, 'metadata', {})
            if isinstance(prev_metadata, dict):
                prev_episode_id = prev_metadata.get('episode_id')
        
        # Calculate time gap
        time_gap = None
        if prev_timestamp is not None:
            time_gap = timestamp - prev_timestamp
        
        # Get episode ID
        episode_id = self.episode_tracker.get_episode(
            timestamp=timestamp,
            previous_timestamp=prev_timestamp,
            previous_episode_id=prev_episode_id,
        )
        
        return {
            "time_gap_seconds": time_gap,
            "episode_id": episode_id,
            "last_access": now,
        }
    
    def compute_gap(
        self,
        timestamp: float,
        previous_timestamp: float,
    ) -> float:
        """
        Compute time gap between two timestamps.
        
        Args:
            timestamp: Current timestamp
            previous_timestamp: Previous timestamp
        
        Returns:
            Gap in seconds
        """
        return timestamp - previous_timestamp
    
    def is_new_episode(
        self,
        gap_seconds: float,
    ) -> bool:
        """
        Check if a gap indicates a new episode.
        
        Args:
            gap_seconds: Time gap in seconds
        
        Returns:
            True if new episode should start
        """
        return gap_seconds >= self.episode_tracker.gap_threshold
    
    def __repr__(self) -> str:
        threshold = self.episode_tracker.gap_threshold
        return f"TemporalEnricher(episode_gap={threshold}s)"
