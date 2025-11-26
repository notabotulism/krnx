"""
KRNX Compute - Worker (Redis Streams)

Background worker using Redis-based job queue.
Follows the PROVEN pattern from LTM worker.

Key changes from SQLite version:
- Uses XREADGROUP (same as LTM worker)
- Consumer name for tracking
- No stale job reset needed (Redis handles via XPENDING)
- XACK after processing (durability)

Usage:
    from chillbot.compute import ComputeWorker, WorkerConfig
    from chillbot.compute.queue import JobQueue
    from chillbot.compute.embeddings import EmbeddingEngine
    from chillbot.compute.vectors import VectorStore
    
    # Create components
    queue = JobQueue()
    embeddings = EmbeddingEngine()
    vectors = VectorStore()
    
    # Create worker
    worker = ComputeWorker(
        queue=queue,
        embeddings=embeddings,
        vectors=vectors,
        config=WorkerConfig(batch_size=100)  # Batch on read!
    )
    
    # Run
    import asyncio
    asyncio.run(worker.run())
"""

import asyncio
import logging
import time
import signal
import uuid
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from chillbot.compute.queue import JobQueue, Job, JobType
from chillbot.compute.embeddings import EmbeddingEngine
from chillbot.compute.vectors import VectorStore
from chillbot.compute.salience import SalienceEngine, SalienceMethod

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Worker statistics."""
    started_at: float = 0.0
    jobs_processed: int = 0
    jobs_failed: int = 0
    embeddings_generated: int = 0
    vectors_indexed: int = 0
    salience_computed: int = 0
    last_job_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        runtime = time.time() - self.started_at if self.started_at else 0
        jobs_per_sec = self.jobs_processed / runtime if runtime > 0 else 0
        
        return {
            "runtime_seconds": round(runtime, 2),
            "jobs_processed": self.jobs_processed,
            "jobs_failed": self.jobs_failed,
            "embeddings_generated": self.embeddings_generated,
            "vectors_indexed": self.vectors_indexed,
            "salience_computed": self.salience_computed,
            "jobs_per_second": round(jobs_per_sec, 3),
            "last_job_at": self.last_job_at,
        }


@dataclass
class WorkerConfig:
    """Worker configuration."""
    batch_size: int = 100                   # Jobs per batch (INCREASED - Redis handles it)
    poll_interval: float = 1.0              # Seconds between polls when idle
    busy_poll_interval: float = 0.01        # Seconds between polls when busy
    block_ms: int = 100                     # Redis block time in milliseconds
    cleanup_interval: float = 3600.0        # Cleanup old jobs every hour
    max_batch_embed: int = 100              # Max texts per batch embedding
    vector_batch_size: int = 50             # Max vectors per batch index


class ComputeWorker:
    """
    Background worker for compute jobs (Redis Streams).
    
    PATTERN (same as LTM worker):
    - XREADGROUP count=N (batch on read)
    - Process batch
    - XACK after success
    
    This is the PROVEN pattern from controller.py LTM worker.
    Handles 50-thread writes with no contention.
    """
    
    def __init__(
        self,
        queue: JobQueue,
        embeddings: EmbeddingEngine,
        vectors: VectorStore,
        salience: Optional[SalienceEngine] = None,
        config: Optional[WorkerConfig] = None,
        kernel_client: Optional[Any] = None,
    ):
        """
        Initialize compute worker.
        
        Args:
            queue: Redis-based job queue
            embeddings: Embedding engine
            vectors: Vector store
            salience: Salience engine (optional)
            config: Worker configuration
            kernel_client: KRNX client for fetching events (optional)
        """
        self.queue = queue
        self.embeddings = embeddings
        self.vectors = vectors
        self.salience = salience or SalienceEngine()
        self.config = config or WorkerConfig()
        self.kernel = kernel_client
        
        self._running = False
        self._stats = WorkerStats()
        self._last_cleanup = 0.0
        
        # Generate unique consumer name (like LTM worker)
        self._consumer_name = f"worker-{uuid.uuid4().hex[:8]}"
        
        # Custom job handlers
        self._handlers: Dict[JobType, Callable] = {}
    
    def register_handler(
        self,
        job_type: JobType,
        handler: Callable,
    ):
        """
        Register custom job handler.
        
        Args:
            job_type: Job type to handle
            handler: Async function(job) -> None
        """
        self._handlers[job_type] = handler
    
    async def run(self):
        """
        Main worker loop (SAME PATTERN as LTM worker).
        
        Pattern from controller.py line 602-678:
        - XREADGROUP with count=N (batching)
        - Process each job
        - XACK after success
        - Continue until stopped
        """
        self._running = True
        self._stats = WorkerStats(started_at=time.time())
        
        logger.info(f"[WORKER] Starting compute worker '{self._consumer_name}'")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Block time: {self.config.block_ms}ms")
        
        # Setup signal handlers
        def handle_signal(sig, frame):
            logger.info(f"[WORKER] Received signal {sig}, stopping...")
            self._running = False
        
        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except ValueError:
            pass
        
        try:
            while self._running:
                try:
                    # Dequeue batch (SAME AS LTM WORKER)
                    jobs = self.queue.dequeue_batch(
                        consumer_name=self._consumer_name,
                        count=self.config.batch_size,
                        block_ms=self.config.block_ms
                    )
                    
                    if not jobs:
                        # No jobs - check if we should cleanup
                        await self._maybe_cleanup()
                        continue
                    
                    # Process batch
                    for job in jobs:
                        await self._process_job(job)
                    
                    # Short sleep between batches
                    if len(jobs) < self.config.batch_size:
                        # Partial batch = queue is draining
                        await asyncio.sleep(self.config.poll_interval)
                    else:
                        # Full batch = queue is busy, keep processing
                        await asyncio.sleep(self.config.busy_poll_interval)
                    
                except asyncio.CancelledError:
                    logger.info("[WORKER] Cancelled, stopping...")
                    break
                except Exception as e:
                    logger.error(f"[WORKER] Loop error: {e}")
                    self._stats.jobs_failed += 1
                    await asyncio.sleep(self.config.poll_interval)
        
        finally:
            logger.info(f"[WORKER] Stopped '{self._consumer_name}'")
            logger.info(f"   Processed: {self._stats.jobs_processed} jobs")
            logger.info(f"   Failed: {self._stats.jobs_failed} jobs")
            logger.info(f"   Embeddings: {self._stats.embeddings_generated}")
    
    async def _process_job(self, job: Job):
        """Process a single job."""
        try:
            logger.debug(f"[WORKER] Processing {job.job_type.value}: {job.job_id}")
            
            # Check for custom handler
            if job.job_type in self._handlers:
                await self._handlers[job.job_type](job)
            elif job.job_type == JobType.EMBED:
                await self._handle_embed(job)
            elif job.job_type == JobType.EMBED_BATCH:
                await self._handle_embed_batch(job)
            elif job.job_type == JobType.SALIENCE:
                await self._handle_salience(job)
            elif job.job_type == JobType.SALIENCE_BATCH:
                await self._handle_salience_batch(job)
            elif job.job_type == JobType.DELETE_VECTOR:
                await self._handle_delete_vector(job)
            elif job.job_type == JobType.CONSOLIDATE:
                await self._handle_consolidate(job)
            elif job.job_type == JobType.CUSTOM:
                await self._handle_custom(job)
            else:
                logger.warning(f"[WORKER] Unknown job type: {job.job_type}")
            
            # XACK (SAME AS LTM WORKER - line 659)
            self.queue.complete(job)
            self._stats.jobs_processed += 1
            self._stats.last_job_at = time.time()
            
        except Exception as e:
            logger.error(f"[WORKER] Job {job.job_id} failed: {e}")
            self.queue.fail(job, str(e))
            self._stats.jobs_failed += 1
    
    async def _handle_embed(self, job: Job):
        """Handle single embedding job."""
        workspace_id = job.workspace_id
        payload = job.payload
        
        # Extract text
        text = payload.get("text")
        if not text:
            raise ValueError("No text in payload")
        
        # Generate embedding (sync, but fast)
        vector = self.embeddings.embed(text)
        
        # Ensure collection exists (cached - fast path)
        self.vectors.ensure_collection(workspace_id, self.embeddings.dimension)
        
        # Index vector
        self.vectors.index(
            workspace_id=workspace_id,
            id=payload.get("event_id", job.job_id),
            vector=vector,
            payload=payload.get("metadata", {}),
        )
        
        self._stats.embeddings_generated += 1
        self._stats.vectors_indexed += 1
    
    async def _handle_embed_batch(self, job: Job):
        """Handle batch embedding job (more efficient)."""
        workspace_id = job.workspace_id
        payload = job.payload
        
        texts = payload.get("texts", [])
        event_ids = payload.get("event_ids", [])
        metadatas = payload.get("metadatas", [{}] * len(texts))
        
        if not texts:
            return
        
        # Batch generate embeddings
        vectors = self.embeddings.embed_batch(texts, show_progress=False)
        
        # Ensure collection
        self.vectors.ensure_collection(workspace_id, self.embeddings.dimension)
        
        # Batch index
        vector_data = [
            {
                "id": event_ids[i] if i < len(event_ids) else f"vec_{i}",
                "vector": vectors[i],
                "payload": metadatas[i] if i < len(metadatas) else {},
            }
            for i in range(len(vectors))
        ]
        
        self.vectors.index_batch(workspace_id, vector_data, batch_size=self.config.vector_batch_size)
        
        self._stats.embeddings_generated += len(vectors)
        self._stats.vectors_indexed += len(vectors)
    
    async def _handle_salience(self, job: Job):
        """Handle salience scoring job."""
        payload = job.payload
        
        score = self.salience.compute(
            event_id=payload.get("event_id"),
            timestamp=payload.get("timestamp", time.time()),
            access_count=payload.get("access_count", 0),
            avg_similarity=payload.get("avg_similarity", 0.0),
            method=SalienceMethod.COMPOSITE,
        )
        
        self._stats.salience_computed += 1
        
        # Store score somewhere (TBD - depends on architecture)
        logger.debug(f"[WORKER] Computed salience: {score.score:.3f}")
    
    async def _handle_salience_batch(self, job: Job):
        """Handle batch salience scoring."""
        payload = job.payload
        events = payload.get("events", [])
        
        if not events:
            return
        
        scores = self.salience.compute_batch(events, method=SalienceMethod.COMPOSITE)
        self._stats.salience_computed += len(scores)
    
    async def _handle_delete_vector(self, job: Job):
        """Handle vector deletion."""
        workspace_id = job.workspace_id
        payload = job.payload
        
        vector_id = payload.get("vector_id") or payload.get("event_id")
        if vector_id:
            self.vectors.delete(workspace_id, vector_id)
    
    async def _handle_consolidate(self, job: Job):
        """Handle memory consolidation (placeholder)."""
        logger.debug(f"[WORKER] Consolidation not yet implemented")
    
    async def _handle_custom(self, job: Job):
        """Handle custom job type."""
        logger.debug(f"[WORKER] Custom job: {job.payload}")
    
    async def _maybe_cleanup(self):
        """Periodic cleanup (optional)."""
        now = time.time()
        if now - self._last_cleanup > self.config.cleanup_interval:
            self._last_cleanup = now
            # Could trim old completed jobs here
            logger.debug("[WORKER] Periodic cleanup")
    
    def stop(self):
        """Stop worker gracefully."""
        self._running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return self._stats.to_dict()


__all__ = [
    "ComputeWorker",
    "WorkerConfig",
    "WorkerStats",
]
