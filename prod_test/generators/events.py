"""
Event Generator

Factory for creating test events with various patterns.
"""

import time
import uuid
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from chillbot.kernel.models import Event


@dataclass
class EventPattern:
    """Configuration for event generation pattern."""
    event_type: str
    content_template: Dict[str, Any]
    timestamp_jitter_ms: float = 0
    metadata_template: Optional[Dict[str, Any]] = None


class EventGenerator:
    """
    Factory for generating test events.
    
    Supports:
    - Sequential events with proper hash chaining
    - Concurrent event streams
    - Time-distributed events
    - Pattern-based content generation
    """
    
    def __init__(
        self,
        workspace_id: str,
        user_id: str,
        session_id: Optional[str] = None,
    ):
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self._sequence = 0
        self._last_hash: Optional[str] = None
    
    def create_event(
        self,
        content: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        ttl_seconds: Optional[int] = None,
        retention_class: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        link_to_previous: bool = False,
    ) -> Event:
        """
        Create a single event.
        
        Args:
            content: Event content (default: generated)
            timestamp: Event timestamp (default: now)
            ttl_seconds: Optional TTL
            retention_class: Optional retention class
            channel: Optional channel
            metadata: Optional metadata
            link_to_previous: Whether to set previous_hash
            
        Returns:
            Event instance
        """
        self._sequence += 1
        
        if content is None:
            content = {
                'type': 'generated',
                'sequence': self._sequence,
                'random': uuid.uuid4().hex[:8],
            }
        
        event = Event(
            event_id=f"gen-{self._sequence:06d}-{uuid.uuid4().hex[:8]}",
            workspace_id=self.workspace_id,
            user_id=self.user_id,
            session_id=self.session_id,
            content=content,
            timestamp=timestamp or time.time(),
            created_at=time.time(),
            previous_hash=self._last_hash if link_to_previous else None,
            channel=channel,
            ttl_seconds=ttl_seconds,
            retention_class=retention_class,
            metadata=metadata or {},
        )
        
        if link_to_previous:
            self._last_hash = event.compute_hash()
        
        return event
    
    def create_chain(
        self,
        count: int,
        content_generator: Optional[callable] = None,
        base_timestamp: Optional[float] = None,
        timestamp_increment: float = 0.001,
    ) -> List[Event]:
        """
        Create a chain of events with proper hash linking.
        
        Args:
            count: Number of events to create
            content_generator: Optional callable(index) -> content
            base_timestamp: Starting timestamp
            timestamp_increment: Time between events in seconds
            
        Returns:
            List of linked events
        """
        events = []
        ts = base_timestamp or time.time()
        
        for i in range(count):
            if content_generator:
                content = content_generator(i)
            else:
                content = {'type': 'chain', 'index': i}
            
            event = self.create_event(
                content=content,
                timestamp=ts + (i * timestamp_increment),
                link_to_previous=True,
            )
            events.append(event)
        
        return events
    
    def create_batch(
        self,
        count: int,
        content_generator: Optional[callable] = None,
        timestamp_range: Optional[tuple] = None,
    ) -> List[Event]:
        """
        Create a batch of independent events (no hash linking).
        
        Args:
            count: Number of events
            content_generator: Optional callable(index) -> content
            timestamp_range: Optional (start, end) for random timestamps
            
        Returns:
            List of events
        """
        events = []
        
        for i in range(count):
            if content_generator:
                content = content_generator(i)
            else:
                content = {'type': 'batch', 'index': i}
            
            if timestamp_range:
                ts = random.uniform(timestamp_range[0], timestamp_range[1])
            else:
                ts = time.time()
            
            event = self.create_event(
                content=content,
                timestamp=ts,
                link_to_previous=False,
            )
            events.append(event)
        
        return events
    
    def create_conversation(
        self,
        turns: int,
        user_name: str = "User",
        assistant_name: str = "Assistant",
    ) -> List[Event]:
        """
        Create a simulated conversation.
        
        Args:
            turns: Number of back-and-forth turns
            user_name: Name for user messages
            assistant_name: Name for assistant messages
            
        Returns:
            List of conversation events
        """
        events = []
        ts = time.time()
        
        for i in range(turns * 2):
            if i % 2 == 0:
                role = "user"
                name = user_name
                text = f"Message {i//2 + 1} from {user_name}"
            else:
                role = "assistant"
                name = assistant_name
                text = f"Response {i//2 + 1} from {assistant_name}"
            
            content = {
                'type': 'message',
                'role': role,
                'name': name,
                'text': text,
                'turn': i // 2,
            }
            
            event = self.create_event(
                content=content,
                timestamp=ts + (i * 0.5),  # 500ms between messages
                link_to_previous=True,
            )
            events.append(event)
        
        return events
    
    def create_with_needle(
        self,
        haystack_size: int,
        needle_content: Dict[str, Any],
        needle_position: str = "middle",
    ) -> tuple:
        """
        Create haystack with a needle at specified position.
        
        Args:
            haystack_size: Total number of events
            needle_content: Content of the needle event
            needle_position: "early", "middle", or "late"
            
        Returns:
            (events, needle_index) tuple
        """
        # Calculate needle position
        if needle_position == "early":
            needle_idx = haystack_size // 10
        elif needle_position == "late":
            needle_idx = haystack_size - (haystack_size // 10)
        else:  # middle
            needle_idx = haystack_size // 2
        
        events = []
        ts = time.time()
        
        for i in range(haystack_size):
            if i == needle_idx:
                content = needle_content
            else:
                content = {
                    'type': 'haystack',
                    'index': i,
                    'filler': f"This is filler content for event {i}",
                }
            
            event = self.create_event(
                content=content,
                timestamp=ts + (i * 0.001),
                link_to_previous=True,
            )
            events.append(event)
        
        return events, needle_idx
    
    def reset_chain(self) -> None:
        """Reset hash chain (for starting new independent chain)."""
        self._last_hash = None
        self._sequence = 0
