"""
KRNX Adapter

Full KRNX kernel adapter for comparative testing.
"""

import time
import uuid
from typing import Dict, Any, Optional, List

from chillbot.kernel.controller import KRNXController
from chillbot.kernel.models import Event

from .base import BaseAdapter, AdapterEvent, RetrievalResult


class KRNXAdapter(BaseAdapter):
    """
    KRNX kernel adapter.
    
    Provides full temporal memory capabilities including:
    - Hash chain verification
    - Temporal replay
    - Event ordering guarantees
    """
    
    def __init__(self):
        self._kernel: Optional[KRNXController] = None
        self._config: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        return "KRNX"
    
    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize KRNX kernel.
        
        Config options:
            data_path: str - Path for SQLite storage
            redis_host: str - Redis host (default: localhost)
            redis_port: int - Redis port (default: 6379)
            enable_hash_chain: bool - Enable hash chain (default: True)
        """
        self._config = config
        
        self._kernel = KRNXController(
            data_path=config.get('data_path', './krnx-test-data'),
            redis_host=config.get('redis_host', 'localhost'),
            redis_port=config.get('redis_port', 6379),
            redis_password=config.get('redis_password'),
            enable_backpressure=config.get('enable_backpressure', False),
            enable_async_worker=config.get('enable_async_worker', True),
            enable_hash_chain=config.get('enable_hash_chain', True),
        )
    
    def teardown(self) -> None:
        """Shutdown kernel."""
        if self._kernel:
            self._kernel.shutdown(timeout=10)
            self._kernel = None
    
    def write(
        self,
        workspace_id: str,
        user_id: str,
        content: Dict[str, Any],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write event to KRNX."""
        if not self._kernel:
            raise RuntimeError("Adapter not initialized")
        
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        ts = timestamp or time.time()
        
        event = Event(
            event_id=event_id,
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=f"adapter-session-{workspace_id}",
            content=content,
            timestamp=ts,
            created_at=time.time(),
            metadata=metadata or {},
        )
        
        self._kernel.write_event(workspace_id, user_id, event)
        return event_id
    
    def query(
        self,
        workspace_id: str,
        user_id: str,
        query_text: Optional[str] = None,
        limit: int = 10,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> RetrievalResult:
        """Query events from KRNX."""
        if not self._kernel:
            raise RuntimeError("Adapter not initialized")
        
        start = time.perf_counter()
        
        events = self._kernel.query_events(
            workspace_id=workspace_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        adapter_events = [
            AdapterEvent(
                id=e.event_id,
                content=e.content,
                timestamp=e.timestamp,
                metadata=e.metadata,
            )
            for e in events
        ]
        
        return RetrievalResult(
            events=adapter_events,
            latency_ms=latency_ms,
            metadata={'source': 'ltm', 'query_text': query_text},
        )
    
    def replay_to(
        self,
        workspace_id: str,
        user_id: str,
        timestamp: float,
    ) -> RetrievalResult:
        """
        Replay events up to timestamp.
        
        THIS IS THE KRNX DIFFERENTIATOR.
        """
        if not self._kernel:
            raise RuntimeError("Adapter not initialized")
        
        start = time.perf_counter()
        
        events = self._kernel.replay_to_timestamp(
            workspace_id=workspace_id,
            user_id=user_id,
            timestamp=timestamp,
        )
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        adapter_events = [
            AdapterEvent(
                id=e.event_id,
                content=e.content,
                timestamp=e.timestamp,
                metadata=e.metadata,
            )
            for e in events
        ]
        
        return RetrievalResult(
            events=adapter_events,
            latency_ms=latency_ms,
            metadata={'replay_timestamp': timestamp},
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get KRNX stats."""
        if not self._kernel:
            return {'error': 'not initialized'}
        
        ltm_stats = self._kernel.ltm.get_stats()
        worker_metrics = self._kernel.get_worker_metrics()
        
        return {
            'adapter': self.name,
            'ltm': ltm_stats,
            'worker': worker_metrics.to_dict(),
        }
    
    def supports_temporal_replay(self) -> bool:
        """KRNX supports true temporal replay."""
        return True
    
    def supports_hash_chain(self) -> bool:
        """KRNX supports hash chain verification."""
        return True
    
    def verify_hash_chain(
        self,
        workspace_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        KRNX-specific: Verify hash chain integrity.
        """
        if not self._kernel:
            raise RuntimeError("Adapter not initialized")
        
        return self._kernel.verify_hash_chain(workspace_id, user_id)
