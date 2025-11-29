"""
Base Adapter Protocol

Abstract interface for memory system adapters used in comparative testing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AdapterEvent:
    """Normalized event format for adapters."""
    id: str
    content: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    events: List[AdapterEvent]
    latency_ms: float
    metadata: Dict[str, Any]


class BaseAdapter(ABC):
    """
    Abstract base class for memory system adapters.
    
    All adapters must implement these methods to enable
    fair comparison in Layer 3 demos.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""
        pass
    
    @abstractmethod
    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Adapter-specific configuration dict
        """
        pass
    
    @abstractmethod
    def teardown(self) -> None:
        """Clean up resources."""
        pass
    
    @abstractmethod
    def write(
        self,
        workspace_id: str,
        user_id: str,
        content: Dict[str, Any],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Write an event to the memory system.
        
        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            content: Event content
            timestamp: Optional timestamp (defaults to now)
            metadata: Optional metadata
            
        Returns:
            Event ID
        """
        pass
    
    @abstractmethod
    def query(
        self,
        workspace_id: str,
        user_id: str,
        query_text: Optional[str] = None,
        limit: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Query events from the memory system.
        
        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            query_text: Optional semantic query
            limit: Maximum results to return
            start_time: Optional time filter start
            end_time: Optional time filter end
            
        Returns:
            RetrievalResult with events and metadata
        """
        pass
    
    @abstractmethod
    def replay_to(
        self,
        workspace_id: str,
        user_id: str,
        timestamp: float,
    ) -> RetrievalResult:
        """
        Replay all events up to a timestamp.
        
        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            timestamp: Replay cutoff timestamp
            
        Returns:
            RetrievalResult with events in chronological order
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get adapter statistics.
        
        Returns:
            Dict with adapter-specific stats
        """
        pass
    
    def supports_temporal_replay(self) -> bool:
        """
        Whether this adapter supports true temporal replay.
        
        Override in subclasses - default is False.
        KRNX returns True, RAG returns False.
        """
        return False
    
    def supports_hash_chain(self) -> bool:
        """
        Whether this adapter supports hash chain verification.
        
        Override in subclasses - default is False.
        KRNX returns True, others return False.
        """
        return False
