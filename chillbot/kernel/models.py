"""
KRNX Models - Core Data Structures

Pure kernel types for event storage and temporal replay.
No application logic - just immutable event primitives.

Philosophy:
- Events are append-only, immutable records
- Content is generic (no opinions on structure)
- Timestamps enable temporal replay
- Metadata is extensible

Constitution Compliance:
- 6.1 Channels: channel field for event filtering
- 6.3 Retention: ttl_seconds, retention_class fields
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import time
import hashlib


# ==============================================
# EVENT (Core Data Structure)
# ==============================================

@dataclass(frozen=True)
class Event:
    """
    Immutable event record.
    
    The fundamental unit of KRNX memory.
    Represents one interaction with arbitrary content.
    
    Stored in:
    - STM (Redis): 0-24 hours (hot, fast)
    - LTM Warm (SQLite): 0-30 days (queryable)
    - LTM Cold (SQLite): 30+ days (compressed, forever)
    
    Design philosophy:
    - Immutable: Once written, never modified
    - Complete: Contains everything needed to reconstruct interaction
    - Temporal: Timestamp is critical for replay
    - Generic: No assumptions about content structure
    """
    
    # Identity
    event_id: str
    workspace_id: str
    user_id: str
    session_id: str
    
    # Content (generic - no opinions)
    content: Dict[str, Any]
    
    # Timing
    timestamp: float         # Unix timestamp (high precision)
    
    # Channel for filtering (Constitution 6.1)
    channel: Optional[str] = None  # e.g., 'chat', 'code', 'system', 'audit'
    
    # Hash-chain provenance (cryptographic integrity)
    previous_hash: Optional[str] = None  # SHA-256 of previous event in workspace:user
    
    # Retention primitives (Constitution 6.3)
    ttl_seconds: Optional[int] = None  # Time-to-live (None = forever)
    retention_class: Optional[str] = None  # e.g., 'ephemeral', 'standard', 'permanent'
    
    # Extensible metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle (set by system)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate event on creation"""
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.workspace_id:
            raise ValueError("workspace_id is required")
        if not self.user_id:
            raise ValueError("user_id is required")
        if self.timestamp <= 0:
            raise ValueError("timestamp must be positive")
        if not isinstance(self.content, dict):
            raise ValueError("content must be a dictionary")
    
    # ==============================================
    # SERIALIZATION
    # ==============================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for storage.
        
        Content and metadata remain as nested dicts.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """
        Create Event from dictionary.
        
        Handles JSON string conversion for content/metadata from SQLite.
        """
        # Handle content (might be JSON string from SQLite)
        if isinstance(data.get('content'), str):
            data['content'] = json.loads(data['content'])
        
        # Handle metadata (might be JSON string from SQLite or missing)
        if 'metadata' not in data or data['metadata'] is None:
            data['metadata'] = {}
        elif isinstance(data.get('metadata'), str):
            data['metadata'] = json.loads(data['metadata'])
        
        # Ensure optional fields are present (might be NULL from SQLite)
        if 'previous_hash' not in data:
            data['previous_hash'] = None
        if 'channel' not in data:
            data['channel'] = None
        if 'ttl_seconds' not in data:
            data['ttl_seconds'] = None
        if 'retention_class' not in data:
            data['retention_class'] = None
        
        return cls(**data)
    
    def to_json(self) -> str:
        """Serialize to JSON string (optimized - skip intermediate dict)"""
        return json.dumps({
            'event_id': self.event_id,
            'workspace_id': self.workspace_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'content': self.content,
            'timestamp': self.timestamp,
            'channel': self.channel,
            'previous_hash': self.previous_hash,
            'ttl_seconds': self.ttl_seconds,
            'retention_class': self.retention_class,
            'metadata': self.metadata,
            'created_at': self.created_at
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserialize from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    # ==============================================
    # HELPERS
    # ==============================================
    
    def get_age_seconds(self) -> float:
        """How old is this event (in seconds)?"""
        return time.time() - self.timestamp
    
    def get_age_days(self) -> float:
        """How old is this event (in days)?"""
        return self.get_age_seconds() / 86400
    
    def is_recent(self, hours: int = 24) -> bool:
        """Is this event within the last N hours?"""
        return self.get_age_seconds() < (hours * 3600)
    
    def should_archive(self, days: int = 30) -> bool:
        """Should this event be moved to cold storage?"""
        return self.get_age_days() > days
    
    def is_expired(self) -> bool:
        """Check if event has exceeded its TTL (Constitution 6.3)"""
        if self.ttl_seconds is None:
            return False  # No TTL = never expires
        return self.get_age_seconds() > self.ttl_seconds
    
    # ==============================================
    # HASH-CHAIN PROVENANCE (Cryptographic Integrity)
    # ==============================================
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of this event.
        
        Hash includes all immutable fields for tamper detection.
        Excludes previous_hash to avoid circular dependency.
        
        Returns:
            Hex string of SHA-256 hash
        """
        # Create canonical representation (deterministic ordering)
        canonical = {
            'event_id': self.event_id,
            'workspace_id': self.workspace_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'channel': self.channel,
            'content': json.dumps(self.content, sort_keys=True),
            'metadata': json.dumps(self.metadata, sort_keys=True),
            'created_at': self.created_at
        }
        
        # Compute hash
        canonical_bytes = json.dumps(canonical, sort_keys=True).encode('utf-8')
        return hashlib.sha256(canonical_bytes).hexdigest()
    
    def verify_hash_chain(self, previous_event: Optional['Event']) -> bool:
        """
        Verify this event's hash-chain link to previous event.
        
        Args:
            previous_event: The event that should precede this one
        
        Returns:
            True if hash-chain is valid, False otherwise
        """
        if previous_event is None:
            # First event in chain - should have no previous_hash
            return self.previous_hash is None
        
        # Verify previous_hash matches previous event's hash
        expected_hash = previous_event.compute_hash()
        return self.previous_hash == expected_hash
    
    def __repr__(self) -> str:
        """Human-readable representation"""
        age = self.get_age_days()
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        channel_str = f", channel={self.channel}" if self.channel else ""
        return (
            f"Event(id={self.event_id[:8]}..., "
            f"workspace={self.workspace_id}{channel_str}, "
            f"content={content_preview}, "
            f"age={age:.1f}d)"
        )
    
    def __hash__(self) -> int:
        """Hash based on event_id for use in sets/dicts"""
        return hash(self.event_id)
    
    def __eq__(self, other) -> bool:
        """Equality based on event_id"""
        if not isinstance(other, Event):
            return False
        return self.event_id == other.event_id


# ==============================================
# VALIDATION HELPERS
# ==============================================

def validate_event_id(event_id: str) -> bool:
    """Validate event_id format"""
    return len(event_id) > 0 and len(event_id) < 256


def validate_workspace_id(workspace_id: str) -> bool:
    """Validate workspace_id format"""
    return len(workspace_id) > 0 and len(workspace_id) < 256


def validate_user_id(user_id: str) -> bool:
    """Validate user_id format"""
    return len(user_id) > 0 and len(user_id) < 256


def validate_channel(channel: str) -> bool:
    """Validate channel format (Constitution 6.1)"""
    if channel is None:
        return True  # Optional field
    return len(channel) > 0 and len(channel) < 128


# ==============================================
# CONVENIENCE CONSTRUCTORS
# ==============================================

def create_event(
    event_id: str,
    workspace_id: str,
    user_id: str,
    content: Dict[str, Any],
    session_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    channel: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    retention_class: Optional[str] = None,
    **metadata
) -> Event:
    """
    Convenience constructor for creating events.
    
    Usage:
        event = create_event(
            event_id="evt_123",
            workspace_id="project_x",
            user_id="user_1",
            content={
                "query": "Create FastAPI endpoint",
                "response": "Here's the code...",
                "type": "code_generation"
            },
            channel="code",  # Constitution 6.1
            ttl_seconds=86400,  # Constitution 6.3 - expires in 24h
            retention_class="standard",  # Constitution 6.3
            model="claude-sonnet-4",
            cost=0.05
        )
    
    Args:
        event_id: Unique event identifier
        workspace_id: Workspace identifier
        user_id: User identifier
        content: Event content (arbitrary structure)
        session_id: Optional session ID (defaults to workspace_user)
        timestamp: Optional timestamp (defaults to now)
        channel: Optional channel for filtering (Constitution 6.1)
        ttl_seconds: Optional TTL in seconds (Constitution 6.3)
        retention_class: Optional retention class (Constitution 6.3)
        **metadata: Additional metadata fields
    
    Returns:
        Event instance
    """
    return Event(
        event_id=event_id,
        workspace_id=workspace_id,
        user_id=user_id,
        session_id=session_id or f"{workspace_id}_{user_id}",
        content=content,
        timestamp=timestamp or time.time(),
        channel=channel,
        ttl_seconds=ttl_seconds,
        retention_class=retention_class,
        metadata=metadata
    )


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'Event',
    'create_event',
    'validate_event_id',
    'validate_workspace_id',
    'validate_user_id',
    'validate_channel'
]
