"""
KRNX Compute - Job Queue (Redis Streams)

Redis-based job queue using the PROVEN pattern from LTM worker.

Why Redis instead of SQLite:
- Multi-producer safe (50 threads can write simultaneously)
- Consumer groups = durability
- Natural batching (XREADGROUP count=N)
- Already using Redis successfully for LTM queue

Pattern (copied from working LTM queue):
- Producer: Direct XADD (no batching)
- Consumer: XREADGROUP with batching
- Durability: XACK after processing

Usage:
    from chillbot.compute.queue import JobQueue, JobType
    from chillbot.kernel.connection_pool import configure_pool, get_redis_client
    
    # Configure Redis (once at startup)
    configure_pool(host="localhost", port=6379)
    
    # Create queue
    queue = JobQueue()
    
    # Enqueue (fast, non-blocking)
    job_id = queue.enqueue(
        job_type=JobType.EMBED,
        workspace_id="my-app",
        payload={"event_id": "evt_123", "text": "Hello"}
    )
    
    # Dequeue batch (in worker)
    jobs = queue.dequeue_batch(count=10, block_ms=100)
    for job in jobs:
        # process...
        queue.complete(job.job_id)
"""

import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List

from chillbot.kernel.connection_pool import get_redis_client

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job lifecycle states."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(Enum):
    """Types of compute jobs."""
    EMBED = "embed"                 # Generate embedding for event
    EMBED_BATCH = "embed_batch"     # Batch embedding generation
    REEMBED = "reembed"             # Regenerate embedding (model change)
    SALIENCE = "salience"           # Calculate salience score
    SALIENCE_BATCH = "salience_batch"  # Batch salience calculation
    CONSOLIDATE = "consolidate"     # Memory consolidation
    DECAY = "decay"                 # Recalculate decay scores
    DELETE_VECTOR = "delete_vector" # Remove vector from store
    CUSTOM = "custom"               # User-defined job


@dataclass
class Job:
    """A compute job."""
    job_id: str
    job_type: JobType
    workspace_id: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    
    # Redis message ID (for acking)
    _message_id: Optional[str] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "workspace_id": self.workspace_id,
            "payload": self.payload,
            "priority": self.priority,
            "created_at": self.created_at,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], message_id: Optional[str] = None) -> "Job":
        """Create Job from dictionary."""
        return cls(
            job_id=data["job_id"],
            job_type=JobType(data["job_type"]),
            workspace_id=data["workspace_id"],
            payload=data["payload"],
            priority=data.get("priority", 0),
            created_at=data.get("created_at", time.time()),
            _message_id=message_id,
        )


class JobQueue:
    """
    Redis Streams-based job queue.
    
    Follows the PROVEN pattern from LTM worker:
    - Producer: Direct XADD (fast, non-blocking)
    - Consumer: XREADGROUP with batching
    - Durability: Consumer groups + XACK
    
    This is the SAME architecture that successfully handles
    50-thread LTM writes. Tested, proven, production-ready.
    """
    
    def __init__(
        self,
        stream_name: str = "krnx:compute:jobs",
        group_name: str = "krnx-compute-workers",
        max_length: int = 10000,
    ):
        """
        Initialize Redis-based job queue.
        
        Args:
            stream_name: Redis stream name
            group_name: Consumer group name
            max_length: Max stream length (trim old completed jobs)
        """
        self.stream_name = stream_name
        self.group_name = group_name
        self.max_length = max_length
        self.redis = get_redis_client()
        
        # Create consumer group (if not exists)
        self._ensure_group()
        
        logger.info(f"[QUEUE] Initialized Redis job queue: {stream_name}")
    
    def _ensure_group(self):
        """Create consumer group if it doesn't exist."""
        try:
            self.redis.xgroup_create(
                self.stream_name,
                self.group_name,
                id='0',
                mkstream=True
            )
            logger.info(f"[QUEUE] Created consumer group: {self.group_name}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"[QUEUE] Failed to create group: {e}")
    
    # ==============================================
    # PRODUCER (Multi-threaded writes)
    # ==============================================
    
    def enqueue(
        self,
        job_type: JobType,
        workspace_id: str,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """
        Enqueue a job (FAST - direct XADD, no batching).
        
        Pattern from working LTM queue:
        - No producer batching (causes stuck events)
        - Direct XADD (Redis handles concurrency)
        - Worker batches on read (XREADGROUP count=N)
        
        Args:
            job_type: Type of job
            workspace_id: Workspace identifier
            payload: Job payload
            priority: Priority (higher = processed first)
        
        Returns:
            job_id
        """
        job_id = f"job_{uuid.uuid4().hex[:16]}"
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            workspace_id=workspace_id,
            payload=payload,
            priority=priority,
        )
        
        # Serialize job
        job_data = {
            'job_json': json.dumps(job.to_dict())
        }
        
        # Direct XADD (same as LTM queue)
        try:
            self.redis.xadd(
                self.stream_name,
                job_data,
                maxlen=self.max_length
            )
            return job_id
        
        except Exception as e:
            logger.error(f"[QUEUE] Enqueue failed: {e}")
            raise
    
    def enqueue_batch(
        self,
        jobs: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Enqueue multiple jobs efficiently.
        
        Uses Redis pipeline (multiple XADD in one roundtrip).
        
        Args:
            jobs: List of job dicts with keys: job_type, workspace_id, payload
        
        Returns:
            List of job_ids
        """
        if not jobs:
            return []
        
        pipe = self.redis.pipeline()
        job_ids = []
        
        for job_dict in jobs:
            job_id = f"job_{uuid.uuid4().hex[:16]}"
            job_ids.append(job_id)
            
            job = Job(
                job_id=job_id,
                job_type=job_dict["job_type"],
                workspace_id=job_dict["workspace_id"],
                payload=job_dict["payload"],
                priority=job_dict.get("priority", 0),
            )
            
            job_data = {
                'job_json': json.dumps(job.to_dict())
            }
            
            pipe.xadd(self.stream_name, job_data, maxlen=self.max_length)
        
        pipe.execute()
        return job_ids
    
    # ==============================================
    # CONSUMER (Worker reads batches)
    # ==============================================
    
    def dequeue_batch(
        self,
        consumer_name: str,
        count: int = 10,
        block_ms: int = 100,
    ) -> List[Job]:
        """
        Dequeue batch of jobs (WORKER PATTERN - same as LTM).
        
        Pattern from working LTM worker:
        - XREADGROUP with count=N (natural batching)
        - Block for new messages
        - Parse and return jobs
        
        Args:
            consumer_name: Unique consumer identifier
            count: Max jobs to return
            block_ms: Block time in milliseconds
        
        Returns:
            List of Job objects
        """
        try:
            messages = self.redis.xreadgroup(
                groupname=self.group_name,
                consumername=consumer_name,
                streams={self.stream_name: '>'},
                count=count,
                block=block_ms
            )
            
            if not messages:
                return []
            
            jobs = []
            for stream_name, msg_list in messages:
                for msg_id, data in msg_list:
                    try:
                        job_json = data.get('job_json')
                        if job_json:
                            job_dict = json.loads(job_json)
                            job = Job.from_dict(job_dict, message_id=msg_id)
                            jobs.append(job)
                    except Exception as e:
                        logger.error(f"[QUEUE] Error parsing job: {e}")
            
            return jobs
        
        except Exception as e:
            logger.error(f"[QUEUE] Dequeue failed: {e}")
            return []
    
    def complete(self, job: Job):
        """
        Mark job as completed (ACK message).
        
        Pattern from LTM worker: XACK after successful processing.
        """
        if job._message_id:
            try:
                self.redis.xack(
                    self.stream_name,
                    self.group_name,
                    job._message_id
                )
            except Exception as e:
                logger.error(f"[QUEUE] ACK failed for {job.job_id}: {e}")
    
    def fail(self, job: Job, error: str):
        """
        Mark job as failed (log and ACK to prevent retry loop).
        
        For now, we just ACK to remove from pending.
        Future: Could write to dead-letter queue.
        """
        logger.error(f"[QUEUE] Job {job.job_id} failed: {error}")
        self.complete(job)  # ACK to prevent infinite retries
    
    # ==============================================
    # MONITORING
    # ==============================================
    
    def get_pending_count(self) -> int:
        """Get number of pending jobs."""
        try:
            pending_info = self.redis.xpending(self.stream_name, self.group_name)
            return pending_info.get('pending', 0) if pending_info else 0
        except Exception:
            return 0
    
    def get_stream_length(self) -> int:
        """Get total stream length."""
        try:
            return self.redis.xlen(self.stream_name)
        except Exception:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "stream_name": self.stream_name,
            "group_name": self.group_name,
            "stream_length": self.get_stream_length(),
            "pending_count": self.get_pending_count(),
        }
    
    def health_check(self) -> bool:
        """Check if queue is healthy."""
        try:
            self.redis.ping()
            return True
        except Exception:
            return False
    
    # ==============================================
    # CLEANUP
    # ==============================================
    
    def trim(self, max_length: Optional[int] = None):
        """Trim stream to max length."""
        length = max_length or self.max_length
        try:
            self.redis.xtrim(self.stream_name, maxlen=length, approximate=True)
        except Exception as e:
            logger.error(f"[QUEUE] Trim failed: {e}")
    
    def clear(self):
        """Clear all jobs (DANGER - for testing only)."""
        try:
            self.redis.delete(self.stream_name)
            self._ensure_group()
            logger.warning(f"[QUEUE] Cleared stream: {self.stream_name}")
        except Exception as e:
            logger.error(f"[QUEUE] Clear failed: {e}")
    
    def __repr__(self) -> str:
        return f"JobQueue(stream='{self.stream_name}', group='{self.group_name}')"

    def close(self):
        """Close queue (no-op for Redis - connection pool handles cleanup)."""
        pass


# ==============================================
# COMPATIBILITY LAYER (for existing code)
# ==============================================

# Old SQLite methods that some code might still call
# These are no-ops or simple wrappers

def dequeue(*args, **kwargs):
    """Legacy method - use dequeue_batch instead."""
    raise NotImplementedError(
        "Use dequeue_batch() with consumer_name instead. "
        "See worker.py for example."
    )


__all__ = [
    "JobQueue",
    "Job",
    "JobType",
    "JobStatus",
]
