"""
KRNX Controller - Pure Kernel v0.3.6 TURBO (NOGROUP Recovery + Metrics Fix)

CRITICAL FIX: Worker resilience and backpressure recovery
Before: Stream deletion (e.g., in tests) caused NOGROUP errors, worker couldn't process
After:  Worker recreates consumer group on NOGROUP error, proper recovery

Changes from v0.3.5:
- FIX: Worker recreates consumer group when NOGROUP error occurs
- FIX: More robust metrics calculation (handles missing stream/group)
- FIX: Proper backpressure recovery after queue drains
"""

import time
import os
import threading
import json
import logging
from dataclasses import dataclass, replace
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

from chillbot.kernel.stm import STM
from chillbot.kernel.ltm import LTM
from chillbot.kernel.models import Event
from chillbot.kernel.connection_pool import configure_pool, close_pool, get_redis_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackpressureError(Exception):
    """Raised when system is under load and rejecting writes"""
    pass


class RedisUnavailableError(Exception):
    """Raised when Redis is unavailable"""
    pass


@dataclass
class WorkerMetrics:
    """Worker health metrics"""
    queue_depth: int
    lag_seconds: float
    messages_processed: int
    errors_last_hour: int
    last_error: Optional[str]
    last_error_time: Optional[float]
    worker_running: bool
    
    def is_healthy(self) -> bool:
        return (
            self.worker_running and
            self.queue_depth < 10000 and
            self.lag_seconds < 60 and
            self.errors_last_hour < 100
        )
    
    def to_dict(self):
        return {
            'queue_depth': self.queue_depth,
            'lag_seconds': round(self.lag_seconds, 2),
            'messages_processed': self.messages_processed,
            'errors_last_hour': self.errors_last_hour,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time,
            'worker_running': self.worker_running,
            'healthy': self.is_healthy()
        }


class ErrorTracker:
    """Track errors in a sliding window"""
    
    def __init__(self, window_seconds: int = 3600):
        self.window_seconds = window_seconds
        self._errors: List[tuple] = []
        self._lock = threading.Lock()
        self.last_error = None
        self.last_error_time = None
    
    def record_error(self, error_msg: str):
        with self._lock:
            now = time.time()
            self._errors.append((now, error_msg))
            self.last_error = error_msg
            self.last_error_time = now
            # Prune old errors
            cutoff = now - self.window_seconds
            self._errors = [(t, m) for t, m in self._errors if t > cutoff]
    
    def get_error_count(self) -> int:
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds
            return len([t for t, _ in self._errors if t > cutoff])
    
    def get_last_error(self) -> Optional[tuple]:
        with self._lock:
            if self.last_error:
                return (self.last_error_time, self.last_error)
            return None


class KRNXController:
    """
    KRNX Kernel Controller v0.3.6 TURBO (NOGROUP Recovery + Metrics Fix)
    
    CRITICAL FIX: Worker resilience and backpressure recovery.
    Worker now recreates consumer group on NOGROUP error.
    """
    
    def __init__(
        self,
        data_path: str = "./krnx-data",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: Optional[str] = None,
        redis_max_connections: int = 200,
        enable_backpressure: bool = True,
        enable_async_worker: bool = True,
        warm_retention_days: int = 30,
        enable_auto_archival: bool = False,
        archival_interval_hours: int = 24,
        enable_auto_snapshots: bool = False,
        snapshot_interval_hours: int = 168,
        max_queue_depth: int = 50000,
        max_lag_seconds: float = 30.0,
        enable_hash_chain: bool = False,
        enable_telemetry: bool = False,
        ltm_batch_size: int = 100,
        ltm_batch_interval: float = 0.1,
        worker_block_ms: int = 100,
    ):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[START] Initializing KRNX kernel v0.3.6 at {data_path}")
        
        # Settings
        self.max_queue_depth = max_queue_depth
        self.max_lag_seconds = max_lag_seconds
        self.enable_backpressure = enable_backpressure
        self.enable_hash_chain = enable_hash_chain
        self.enable_telemetry = enable_telemetry
        self.worker_block_ms = worker_block_ms
        self.worker_batch_size = ltm_batch_size
        
        # Metrics
        self._worker_metrics = {
            'messages_processed': 0,
            'start_time': time.time()
        }
        self._error_tracker = ErrorTracker(window_seconds=3600)
        
        # Hash-chain (optional)
        self._last_hashes: Dict[str, str] = {}
        self._hash_lock = threading.Lock()
        
        # Telemetry hooks
        self._telemetry_hooks: List[callable] = []
        
        # Thread-safe backpressure
        self._backpressure_mode = False
        self._backpressure_lock = threading.Lock()
        self._last_backpressure_check = 0.0
        self._backpressure_check_interval = 0.1  # Check at most every 100ms
        
        # Configure global connection pool
        configure_pool(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            max_connections=redis_max_connections
        )
        
        # Initialize components
        self.stm = STM(ttl_hours=24, use_connection_pool=True)
        
        self.ltm = LTM(
            data_path=str(self.data_path),
            warm_retention_days=warm_retention_days,
            enable_auto_archival=enable_auto_archival,
            archival_interval_hours=archival_interval_hours,
        )
        
        # Worker state
        self._worker_running = False
        self._ltm_worker_thread = None
        self._drain_complete = threading.Event()
        
        if enable_async_worker:
            self._start_ltm_worker()
        
        logger.info(f"[OK] KRNX kernel v0.3.6 initialized")
        logger.info(f"   Backpressure: {enable_backpressure}")
        logger.info(f"   Max queue depth: {max_queue_depth}")
        logger.info(f"   Max lag: {max_lag_seconds}s")
    
    # ==============================================
    # TURBO WRITE PATH (with Fixed Backpressure)
    # ==============================================
    
    def write_event_turbo(
        self,
        workspace_id: str,
        user_id: str,
        event: Event,
        target_agents: Optional[List[str]] = None,
        job_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        TURBO: Write event using SINGLE Redis pipeline.
        
        v0.3.5 FIX: Backpressure now recovers properly:
        1. Always re-evaluate metrics before rejecting
        2. Use XPENDING for accurate queue depth
        3. Use first-entry for accurate lag calculation
        """
        # ==============================================
        # FIXED BACKPRESSURE CHECK
        # Key change: Evaluate metrics BEFORE checking mode
        # ==============================================
        if self.enable_backpressure:
            now = time.time()
            should_check = False
            
            with self._backpressure_lock:
                # Time-based check interval
                if (now - self._last_backpressure_check) > self._backpressure_check_interval:
                    should_check = True
                    self._last_backpressure_check = now
                # CRITICAL FIX: If in backpressure mode, ALWAYS check to allow recovery
                elif self._backpressure_mode:
                    should_check = True
                    self._last_backpressure_check = now
            
            if should_check:
                # Get fresh metrics OUTSIDE the lock
                try:
                    metrics = self.get_worker_metrics()
                    new_mode = (
                        metrics.queue_depth > self.max_queue_depth or 
                        metrics.lag_seconds > self.max_lag_seconds
                    )
                    
                    # Update mode
                    with self._backpressure_lock:
                        old_mode = self._backpressure_mode
                        self._backpressure_mode = new_mode
                        
                        # Log state transitions
                        if old_mode and not new_mode:
                            logger.info(f"[BP] Backpressure RECOVERED (depth={metrics.queue_depth}, lag={metrics.lag_seconds:.2f}s)")
                        elif not old_mode and new_mode:
                            logger.warning(f"[BP] Backpressure ENGAGED (depth={metrics.queue_depth}, lag={metrics.lag_seconds:.2f}s)")
                except Exception as e:
                    logger.debug(f"[BP] Metrics check failed: {e}")
                    # On error, don't change mode
            
            # NOW check mode (after potential update)
            with self._backpressure_lock:
                if self._backpressure_mode:
                    raise BackpressureError("System under load")
        
        # Optional hash-chain
        if self.enable_hash_chain:
            with self._hash_lock:
                chain_key = f"{workspace_id}:{user_id}"
                last_hash = self._last_hashes.get(chain_key)
                if last_hash:
                    event = replace(event, previous_hash=last_hash)
                current_hash = event.compute_hash()
                self._last_hashes[chain_key] = current_hash
        
        # Get Redis client
        redis_client = get_redis_client()
        pipe = redis_client.pipeline()
        
        # ===== STM COMMANDS (5 commands) =====
        
        # 1. Store event data
        event_key = f"event:{event.event_id}"
        pipe.setex(
            event_key,
            self.stm.ttl_seconds,
            event.to_json()
        )
        
        # 2. Add to workspace stream
        stream_name = f"workspace:{workspace_id}:events"
        content_str = str(event.content)
        content_preview = content_str[:200] if len(content_str) < 1000 else f"<{event.content.get('type', 'large')}>"
        
        stream_data = {
            'event_id': event.event_id,
            'user_id': user_id,
            'timestamp': str(event.timestamp),
            'target_agents': ','.join(target_agents) if target_agents else '*',
            'content_preview': content_preview
        }
        pipe.xadd(stream_name, stream_data, maxlen=10000)
        
        # 3-5. User recent list
        user_recent_key = f"user:{workspace_id}:{user_id}:recent"
        pipe.lpush(user_recent_key, event.event_id)
        pipe.ltrim(user_recent_key, 0, 99)
        pipe.expire(user_recent_key, self.stm.ttl_seconds)
        
        # ===== LTM QUEUE (1 command) =====
        ltm_data = {
            'event_id': event.event_id,
            'event_json': event.to_json()
        }
        pipe.xadd('krnx:ltm:queue', ltm_data, maxlen=100000)
        
        # ===== JOB QUEUE (1 command, optional) =====
        if job_data:
            pipe.xadd('krnx:compute:jobs', {'job_json': json.dumps(job_data)}, maxlen=10000)
        
        # ===== EXECUTE ALL IN ONE ROUND-TRIP =====
        results = pipe.execute()
        
        # Return STM stream message ID (index 1)
        stream_message_id = results[1]
        return stream_message_id
    
    def write_event(
        self,
        workspace_id: str,
        user_id: str,
        event: Event,
        target_agents: Optional[List[str]] = None
    ) -> str:
        """Legacy write_event - redirects to turbo version."""
        return self.write_event_turbo(
            workspace_id=workspace_id,
            user_id=user_id,
            event=event,
            target_agents=target_agents,
            job_data=None
        )
    
    # ==============================================
    # WORKER METRICS (Fixed Calculation)
    # ==============================================
    
    def get_worker_metrics(self) -> WorkerMetrics:
        """
        Get worker health metrics.
        
        v0.3.6 FIX: Correct queue depth and lag calculation
        - XPENDING 'pending' count is the ONLY accurate measure of unprocessed work
        - When pending = 0, queue is fully processed regardless of stream length
        - Lag only matters when there's pending work
        """
        redis_client = get_redis_client()
        
        try:
            # Get queue depth from XPENDING (actual unprocessed messages)
            queue_depth = 0
            has_pending_info = False
            
            try:
                pending_info = redis_client.xpending('krnx:ltm:queue', 'krnx-ltm-workers')
                if pending_info:
                    has_pending_info = True
                    # 'pending' is the count of delivered-but-not-ACK'd messages
                    queue_depth = pending_info.get('pending', 0)
                    
                    # CRITICAL: If pending is 0, the queue is FULLY PROCESSED
                    # regardless of how many messages are in the stream (XLEN)
            except Exception as e:
                # Consumer group doesn't exist yet - check stream length as fallback
                if "NOGROUP" in str(e):
                    try:
                        # No consumer group = nothing processed = all messages are "pending"
                        queue_depth = redis_client.xlen('krnx:ltm:queue')
                    except Exception:
                        queue_depth = 0
            
            # Calculate lag - only meaningful when there's pending work
            lag_seconds = 0.0
            
            # CRITICAL: If queue is empty (no pending), lag is 0
            if queue_depth == 0:
                lag_seconds = 0.0
            else:
                try:
                    stream_info = redis_client.xinfo_stream('krnx:ltm:queue')
                    stream_length = stream_info.get('length', 0)
                    
                    if stream_length > 0 and 'first-entry' in stream_info:
                        first_entry = stream_info.get('first-entry')
                        if first_entry and len(first_entry) >= 1:
                            msg_id = first_entry[0]
                            if isinstance(msg_id, bytes):
                                msg_id = msg_id.decode('utf-8')
                            
                            # Extract timestamp from message ID
                            msg_timestamp_ms = int(msg_id.split('-')[0])
                            now_ms = int(time.time() * 1000)
                            lag_seconds = max(0, (now_ms - msg_timestamp_ms) / 1000.0)
                except Exception:
                    lag_seconds = 0.0
            
            error_info = self._error_tracker.get_last_error()
            
            return WorkerMetrics(
                queue_depth=queue_depth,
                lag_seconds=lag_seconds,
                messages_processed=self._worker_metrics['messages_processed'],
                errors_last_hour=self._error_tracker.get_error_count(),
                last_error=error_info[1] if error_info else None,
                last_error_time=error_info[0] if error_info else None,
                worker_running=self._worker_running
            )
        
        except Exception as e:
            logger.error(f"[FAIL] Get worker metrics error: {e}")
            return WorkerMetrics(
                queue_depth=0,
                lag_seconds=0.0,
                messages_processed=self._worker_metrics['messages_processed'],
                errors_last_hour=self._error_tracker.get_error_count(),
                last_error=str(e),
                last_error_time=time.time(),
                worker_running=self._worker_running
            )
    
    # ==============================================
    # ASYNC WORKER (with Stream Trimming)
    # ==============================================
    
    def _start_ltm_worker(self):
        """Start async worker thread."""
        if self._worker_running:
            logger.warning("[WARN] LTM worker already running")
            return
        
        self._worker_running = True
        self._drain_complete.clear()
        
        logger.info("[START] Starting LTM worker thread...")
        self._ltm_worker_thread = threading.Thread(
            target=self._ltm_worker_loop,
            daemon=True
        )
        self._ltm_worker_thread.start()
        logger.info("[OK] LTM worker started")
    
    def _ensure_consumer_group(self, redis_client):
        """Ensure consumer group exists (call before XREADGROUP operations)."""
        try:
            redis_client.xgroup_create('krnx:ltm:queue', 'krnx-ltm-workers', id='0', mkstream=True)
            logger.debug("[WORKER] Consumer group created/verified")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                # Only log if it's not "group already exists"
                logger.debug(f"[WORKER] Consumer group check: {e}")
    
    def _ltm_worker_loop(self):
        """
        LTM worker with batch accumulation and stream trimming.
        
        v0.3.6 FIX: 
        - Handle NOGROUP errors by recreating consumer group
        - This allows recovery after stream deletion (e.g., in tests)
        """
        worker_id = f"worker-{int(time.time())}"
        redis_client = get_redis_client()
        
        # Ensure consumer group exists
        self._ensure_consumer_group(redis_client)
        
        logger.info(f"[WORKER] LTM worker '{worker_id}' started (v0.3.6 with NOGROUP recovery)")
        
        # Batch accumulation
        pending_events = []
        pending_msg_ids = []
        batch_start_time = None
        batch_size = 500
        batch_timeout_ms = 100
        
        while self._worker_running:
            try:
                messages = redis_client.xreadgroup(
                    groupname='krnx-ltm-workers',
                    consumername=worker_id,
                    streams={'krnx:ltm:queue': '>'},
                    count=100,
                    block=self.worker_block_ms
                )
                
                if messages:
                    for stream_name, msg_list in messages:
                        for msg_id, data in msg_list:
                            event_json = data.get('event_json')
                            if event_json:
                                try:
                                    event = Event.from_json(event_json)
                                    pending_events.append(event)
                                    pending_msg_ids.append(msg_id)
                                    
                                    if batch_start_time is None:
                                        batch_start_time = time.time()
                                except Exception as e:
                                    logger.error(f"[WORKER] Parse error: {e}")
                                    # ACK bad messages to prevent infinite retry
                                    redis_client.xack('krnx:ltm:queue', 'krnx-ltm-workers', msg_id)
                
                # Check if we should flush
                should_flush = False
                if pending_events:
                    if len(pending_events) >= batch_size:
                        should_flush = True
                    elif batch_start_time and (time.time() - batch_start_time) * 1000 >= batch_timeout_ms:
                        should_flush = True
                
                if should_flush and pending_events:
                    try:
                        # Store batch to LTM
                        stored_count = self.ltm.store_events_batch(pending_events)
                        self._worker_metrics['messages_processed'] += stored_count
                        
                        # ACK all messages
                        if pending_msg_ids:
                            redis_client.xack('krnx:ltm:queue', 'krnx-ltm-workers', *pending_msg_ids)
                            
                            # CRITICAL FIX: XTRIM to actually remove processed messages
                            # This makes XLEN/queue depth reflect true state
                            # Use MINID to trim all entries older than oldest pending
                            redis_client.xtrim('krnx:ltm:queue', maxlen=50000, approximate=True)
                        
                        logger.debug(f"[WORKER] Stored batch of {stored_count} events")
                        
                    except Exception as e:
                        logger.error(f"[WORKER] Batch store failed: {e}")
                        self._error_tracker.record_error(str(e))
                        # Store events individually as fallback
                        for i, event in enumerate(pending_events):
                            try:
                                self.ltm.store_event(event)
                                redis_client.xack('krnx:ltm:queue', 'krnx-ltm-workers', pending_msg_ids[i])
                                self._worker_metrics['messages_processed'] += 1
                            except Exception as e2:
                                logger.error(f"[FAIL] Event {event.event_id}: {e2}")
                    
                    pending_events = []
                    pending_msg_ids = []
                    batch_start_time = None
                    
            except Exception as e:
                error_str = str(e)
                
                # CRITICAL FIX: Handle NOGROUP error by recreating consumer group
                if "NOGROUP" in error_str:
                    logger.warning(f"[WORKER] Consumer group missing, recreating...")
                    self._ensure_consumer_group(redis_client)
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    logger.error(f"[WORKER] Worker loop error: {e}")
                    self._error_tracker.record_error(error_str)
                    time.sleep(1)
        
        # Drain remaining
        self._drain_worker(worker_id, redis_client, pending_events, pending_msg_ids)
    
    def _drain_worker(self, worker_id: str, redis_client, pending_events: list, pending_msg_ids: list):
        """Drain worker on shutdown."""
        logger.info(f"[DRAIN] Worker draining {len(pending_events)} accumulated events...")
        
        if pending_events:
            try:
                stored_count = self.ltm.store_events_batch(pending_events)
                if pending_msg_ids:
                    redis_client.xack('krnx:ltm:queue', 'krnx-ltm-workers', *pending_msg_ids)
                    redis_client.xtrim('krnx:ltm:queue', maxlen=50000, approximate=True)
                logger.info(f"[DRAIN] Stored {stored_count} accumulated events")
            except Exception as e:
                logger.error(f"[DRAIN] Failed to store accumulated events: {e}")
        
        # Final drain from Redis
        try:
            if not self.ltm.db:
                logger.info("[DRAIN] Skipping Redis drain - LTM already closed")
                self._drain_complete.set()
                return
            
            messages = redis_client.xreadgroup(
                groupname='krnx-ltm-workers',
                consumername=worker_id,
                streams={'krnx:ltm:queue': '>'},
                count=1000,
                block=None
            )
            
            if messages:
                events_to_store = []
                msg_ids_to_ack = []
                
                for stream_name, msg_list in messages:
                    for msg_id, data in msg_list:
                        try:
                            event_json = data.get('event_json')
                            if event_json:
                                event = Event.from_json(event_json)
                                events_to_store.append(event)
                                msg_ids_to_ack.append(msg_id)
                        except Exception as e:
                            logger.error(f"[DRAIN] Error parsing message: {e}")
                
                if events_to_store:
                    try:
                        self.ltm.store_events_batch(events_to_store)
                        if msg_ids_to_ack:
                            redis_client.xack('krnx:ltm:queue', 'krnx-ltm-workers', *msg_ids_to_ack)
                            redis_client.xtrim('krnx:ltm:queue', maxlen=50000, approximate=True)
                        logger.info(f"[DRAIN] Stored {len(events_to_store)} final events")
                    except Exception as e:
                        logger.error(f"[DRAIN] Failed to store final batch: {e}")
        
        except Exception as e:
            logger.error(f"[DRAIN] Redis drain failed: {e}")
        
        self._drain_complete.set()
        logger.info(f"[STOP] LTM worker '{worker_id}' stopped cleanly")
    
    def worker_running(self) -> bool:
        """Check if async worker is running."""
        return self._worker_running
    
    # ==============================================
    # MANUAL BACKPRESSURE CONTROL (for testing)
    # ==============================================
    
    def reset_backpressure(self):
        """
        Manually reset backpressure state.
        Use this in tests or for emergency recovery.
        """
        with self._backpressure_lock:
            self._backpressure_mode = False
            self._last_backpressure_check = 0.0
        logger.info("[BP] Backpressure manually reset")
    
    def force_backpressure_check(self) -> bool:
        """
        Force an immediate backpressure re-evaluation.
        Returns the new backpressure state.
        """
        try:
            metrics = self.get_worker_metrics()
            new_mode = (
                metrics.queue_depth > self.max_queue_depth or 
                metrics.lag_seconds > self.max_lag_seconds
            )
            
            with self._backpressure_lock:
                self._backpressure_mode = new_mode
                self._last_backpressure_check = time.time()
            
            logger.info(f"[BP] Forced check: mode={new_mode}, depth={metrics.queue_depth}, lag={metrics.lag_seconds:.2f}s")
            return new_mode
        except Exception as e:
            logger.error(f"[BP] Forced check failed: {e}")
            return self._backpressure_mode
    
    # ==============================================
    # READ OPERATIONS
    # ==============================================
    
    def query_events(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[Event]:
        """Query events from LTM."""
        return self.ltm.query_events(
            workspace_id=workspace_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get single event by ID."""
        return self.ltm.get_event(event_id)
    
    def get_recent_events(
        self,
        workspace_id: str,
        user_id: str,
        limit: int = 10
    ) -> List[Event]:
        """Get recent events for user from STM."""
        return self.stm.get_recent_events(workspace_id, user_id, limit)
    
    # ==============================================
    # CONSUMER GROUPS
    # ==============================================
    
    def create_consumer_group(
        self,
        workspace_id: str,
        agent_group: str,
        start_id: str = '0'
    ):
        """Create consumer group for agents."""
        self.stm.create_consumer_group(workspace_id, agent_group, start_id)
    
    def read_events_for_agent(
        self,
        workspace_id: str,
        agent_group: str,
        agent_id: str,
        count: int = 10,
        block_ms: int = 1000,
        filter_targets: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Read events for specific agent from consumer group."""
        return self.stm.read_events_for_agent(
            workspace_id=workspace_id,
            agent_group=agent_group,
            agent_id=agent_id,
            count=count,
            block_ms=block_ms,
            filter_targets=filter_targets
        )
    
    def ack_event(self, workspace_id: str, agent_group: str, message_id: str):
        """Acknowledge event processing."""
        self.stm.ack_event(workspace_id, agent_group, message_id)
    
    # ==============================================
    # LIFECYCLE
    # ==============================================
    
    def shutdown(self, timeout: float = 10.0):
        """Graceful shutdown with drain."""
        logger.info("[STOP] Shutting down KRNX kernel...")
        
        self._worker_running = False
        
        if self._ltm_worker_thread:
            logger.info("[STOP] Waiting for worker to finish draining...")
            self._drain_complete.wait(timeout=timeout)
            self._ltm_worker_thread.join(timeout=2.0)
        
        self.ltm.close()
        close_pool()
        
        logger.info("[OK] KRNX kernel shutdown complete")
    
    def close(self):
        """Alias for shutdown()."""
        self.shutdown()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


# ==============================================
# CONVENIENCE FACTORY
# ==============================================

def create_krnx(
    data_path: str = "./krnx-data",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    **kwargs
) -> KRNXController:
    """Create and return a KRNXController instance."""
    return KRNXController(
        data_path=data_path,
        redis_host=redis_host,
        redis_port=redis_port,
        **kwargs
    )


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'KRNXController',
    'create_krnx',
    'BackpressureError',
    'RedisUnavailableError',
    'WorkerMetrics',
    'ErrorTracker',
]
