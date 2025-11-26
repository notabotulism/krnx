"""
KRNX Compute Subsystem

Background processing for memory operations:
- Job queue (SQLite-backed, durable)
- Embedding generation (sentence-transformers)
- Vector storage (Qdrant)
- Salience scoring
- Worker loop

Usage:
    from chillbot.compute import JobQueue, ComputeWorker, EmbeddingEngine, VectorStore
    
    # Create components
    queue = JobQueue("jobs.db")
    embeddings = EmbeddingEngine()
    vectors = VectorStore()
    
    # Enqueue a job
    job_id = queue.enqueue(
        job_type=JobType.EMBED,
        workspace_id="my-app",
        payload={"event_id": "evt_123", "text": "Hello world"}
    )
    
    # Run worker
    worker = ComputeWorker(queue, embeddings, vectors)
    await worker.run()
"""

__version__ = "0.1.0"

from chillbot.compute.queue import JobQueue, Job, JobType, JobStatus
from chillbot.compute.embeddings import EmbeddingEngine
from chillbot.compute.vectors import VectorStore, VectorMatch
from chillbot.compute.salience import SalienceEngine, SalienceScore, SalienceMethod
from chillbot.compute.worker import ComputeWorker

__all__ = [
    # Queue
    "JobQueue",
    "Job",
    "JobType",
    "JobStatus",
    # Embeddings
    "EmbeddingEngine",
    # Vectors
    "VectorStore",
    "VectorMatch",
    # Salience
    "SalienceEngine",
    "SalienceScore",
    "SalienceMethod",
    # Worker
    "ComputeWorker",
]
