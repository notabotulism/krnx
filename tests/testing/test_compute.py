"""
KRNX Compute - Tests

Comprehensive tests for the compute subsystem:
- Job queue operations
- Embedding generation
- Vector store operations
- Salience scoring
- Worker integration

Run with: pytest test_compute.py -v
"""

import asyncio
import time
import tempfile
import os
import pytest
from pathlib import Path

# Import compute components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute.queue import JobQueue, Job, JobType, JobStatus
from compute.embeddings import EmbeddingEngine
from compute.vectors import VectorStore, VectorStoreBackend, VectorMatch
from compute.salience import SalienceEngine, SalienceMethod, SalienceConfig
from compute.worker import ComputeWorker, WorkerConfig


# ============================================
# JOB QUEUE TESTS
# ============================================

class TestJobQueue:
    """Tests for SQLite-backed job queue."""
    
    def setup_method(self):
        """Create temp database for each test."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        self.queue = JobQueue(self.temp_file.name)
    
    def teardown_method(self):
        """Cleanup temp database."""
        self.queue.close()
        os.unlink(self.temp_file.name)
    
    def test_enqueue_dequeue(self):
        """Basic enqueue and dequeue."""
        job_id = self.queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test-ws",
            payload={"event_id": "evt_1", "text": "Hello"},
        )
        
        assert job_id.startswith("job_")
        
        jobs = self.queue.dequeue(batch_size=1)
        assert len(jobs) == 1
        assert jobs[0].job_id == job_id
        assert jobs[0].job_type == JobType.EMBED
        assert jobs[0].status == JobStatus.RUNNING
        assert jobs[0].payload["text"] == "Hello"
    
    def test_priority_ordering(self):
        """Higher priority jobs dequeued first."""
        # Enqueue low priority first
        low_id = self.queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test-ws",
            payload={"priority": "low"},
            priority=0,
        )
        
        # Enqueue high priority second
        high_id = self.queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test-ws",
            payload={"priority": "high"},
            priority=10,
        )
        
        # High priority should come first
        jobs = self.queue.dequeue(batch_size=2)
        assert jobs[0].job_id == high_id
        assert jobs[1].job_id == low_id
    
    def test_complete_and_fail(self):
        """Complete and fail job handling."""
        job_id = self.queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test-ws",
            payload={},
        )
        
        # Dequeue
        jobs = self.queue.dequeue()
        assert jobs[0].status == JobStatus.RUNNING
        
        # Complete
        self.queue.complete(job_id)
        job = self.queue.get(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
    
    def test_retry_on_failure(self):
        """Jobs retry on failure up to max_retries."""
        job_id = self.queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test-ws",
            payload={},
            max_retries=2,
        )
        
        # First attempt
        jobs = self.queue.dequeue()
        self.queue.fail(job_id, "Error 1")
        
        job = self.queue.get(job_id)
        assert job.status == JobStatus.PENDING
        assert job.retries == 1
        
        # Second attempt
        jobs = self.queue.dequeue()
        self.queue.fail(job_id, "Error 2")
        
        job = self.queue.get(job_id)
        assert job.status == JobStatus.PENDING
        assert job.retries == 2
        
        # Third attempt - should fail permanently
        jobs = self.queue.dequeue()
        self.queue.fail(job_id, "Error 3")
        
        job = self.queue.get(job_id)
        assert job.status == JobStatus.FAILED
        assert job.retries == 2
        assert job.error == "Error 3"
    
    def test_batch_enqueue(self):
        """Batch enqueue multiple jobs."""
        jobs_data = [
            (JobType.EMBED, "ws1", {"id": 1}, 0),
            (JobType.EMBED, "ws1", {"id": 2}, 5),
            (JobType.SALIENCE, "ws1", {"id": 3}, 0),
        ]
        
        job_ids = self.queue.enqueue_batch(jobs_data)
        assert len(job_ids) == 3
        
        assert self.queue.pending_count() == 3
    
    def test_stats(self):
        """Queue statistics."""
        # Enqueue some jobs
        for i in range(5):
            self.queue.enqueue(JobType.EMBED, "ws1", {"i": i})
        
        for i in range(3):
            self.queue.enqueue(JobType.SALIENCE, "ws1", {"i": i})
        
        # Complete some
        jobs = self.queue.dequeue(batch_size=2)
        for job in jobs:
            self.queue.complete(job.job_id)
        
        stats = self.queue.stats()
        assert stats["total"] == 8
        assert stats["by_status"]["completed"] == 2
        assert stats["by_status"]["pending"] == 6
    
    def test_cleanup(self):
        """Cleanup old jobs."""
        # Create a completed job with old timestamp
        job_id = self.queue.enqueue(JobType.EMBED, "ws1", {})
        self.queue.dequeue()
        self.queue.complete(job_id)
        
        # Modify completed_at to be old (hacky but works for test)
        with self.queue._get_conn() as conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ? WHERE job_id = ?",
                (time.time() - 100000, job_id)
            )
            conn.commit()
        
        # Cleanup with short threshold
        deleted = self.queue.cleanup(completed_older_than=1)
        assert deleted == 1
    
    def test_workspace_delete(self):
        """Delete all jobs for a workspace."""
        # Create jobs in different workspaces
        self.queue.enqueue(JobType.EMBED, "ws1", {})
        self.queue.enqueue(JobType.EMBED, "ws1", {})
        self.queue.enqueue(JobType.EMBED, "ws2", {})
        
        assert self.queue.pending_count() == 3
        
        deleted = self.queue.delete_workspace_jobs("ws1")
        assert deleted == 2
        assert self.queue.pending_count() == 1


# ============================================
# EMBEDDING ENGINE TESTS
# ============================================

class TestEmbeddingEngine:
    """Tests for embedding generation."""
    
    def setup_method(self):
        """Create embedding engine."""
        # Use default model - will lazy load
        self.engine = EmbeddingEngine()
    
    def test_dimension(self):
        """Check dimension matches model."""
        assert self.engine.dimension == 384  # all-MiniLM-L6-v2
    
    def test_embed_single(self):
        """Embed single text."""
        vector = self.engine.embed("Hello world")
        
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)
    
    def test_embed_empty(self):
        """Empty text returns zero vector."""
        vector = self.engine.embed("")
        assert all(v == 0.0 for v in vector)
        
        vector = self.engine.embed("   ")
        assert all(v == 0.0 for v in vector)
    
    def test_embed_batch(self):
        """Batch embedding."""
        texts = ["Hello", "World", "Test"]
        vectors = self.engine.embed_batch(texts)
        
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)
    
    def test_embed_batch_with_empty(self):
        """Batch with empty texts."""
        texts = ["Hello", "", "Test"]
        vectors = self.engine.embed_batch(texts)
        
        assert len(vectors) == 3
        assert all(vectors[1][i] == 0.0 for i in range(384))
    
    def test_similarity(self):
        """Similarity computation."""
        vec1 = self.engine.embed("I love dogs")
        vec2 = self.engine.embed("I adore puppies")
        vec3 = self.engine.embed("The weather is sunny")
        
        # Similar texts should have higher similarity
        sim_related = self.engine.similarity(vec1, vec2)
        sim_unrelated = self.engine.similarity(vec1, vec3)
        
        assert sim_related > sim_unrelated
        assert sim_related > 0.5  # Should be reasonably similar
    
    def test_extract_text(self):
        """Text extraction from various formats."""
        # String
        assert self.engine.extract_text("Hello") == "Hello"
        
        # Dict with text field
        assert self.engine.extract_text({"text": "Hello"}) == "Hello"
        
        # Dict with message field
        assert self.engine.extract_text({"message": "Hi"}) == "Hi"
        
        # Chat format
        assert self.engine.extract_text({"role": "user", "content": "Hey"}) == "Hey"


# ============================================
# VECTOR STORE TESTS
# ============================================

class TestVectorStore:
    """Tests for vector storage (using in-memory backend)."""
    
    def setup_method(self):
        """Create in-memory vector store."""
        self.store = VectorStore(backend=VectorStoreBackend.MEMORY)
    
    def test_ensure_collection(self):
        """Create collection."""
        self.store.ensure_collection("test-ws", dimension=384)
        assert self.store.collection_exists("test-ws")
    
    def test_index_and_search(self):
        """Index and search vectors."""
        self.store.ensure_collection("test-ws", dimension=3)
        
        # Index some vectors
        self.store.index("test-ws", "v1", [1.0, 0.0, 0.0], {"label": "x"})
        self.store.index("test-ws", "v2", [0.0, 1.0, 0.0], {"label": "y"})
        self.store.index("test-ws", "v3", [1.0, 1.0, 0.0], {"label": "xy"})
        
        # Search
        results = self.store.search("test-ws", [1.0, 0.0, 0.0], top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "v1"  # Exact match
        assert results[0].score > 0.99
    
    def test_batch_index(self):
        """Batch indexing."""
        self.store.ensure_collection("test-ws", dimension=3)
        
        points = [
            ("v1", [1.0, 0.0, 0.0], {"i": 1}),
            ("v2", [0.0, 1.0, 0.0], {"i": 2}),
            ("v3", [0.0, 0.0, 1.0], {"i": 3}),
        ]
        
        self.store.index_batch("test-ws", points)
        
        assert self.store.count("test-ws") == 3
    
    def test_get_and_delete(self):
        """Get and delete vectors."""
        self.store.ensure_collection("test-ws", dimension=3)
        self.store.index("test-ws", "v1", [1.0, 0.0, 0.0], {"label": "test"})
        
        # Get
        result = self.store.get("test-ws", "v1")
        assert result is not None
        assert result.payload["label"] == "test"
        
        # Delete
        self.store.delete("test-ws", "v1")
        result = self.store.get("test-ws", "v1")
        assert result is None
    
    def test_delete_collection(self):
        """Delete entire collection."""
        self.store.ensure_collection("test-ws", dimension=3)
        self.store.index("test-ws", "v1", [1.0, 0.0, 0.0], {})
        
        self.store.delete_collection("test-ws")
        assert not self.store.collection_exists("test-ws")
    
    def test_score_threshold(self):
        """Search with score threshold."""
        self.store.ensure_collection("test-ws", dimension=3)
        
        self.store.index("test-ws", "v1", [1.0, 0.0, 0.0], {})
        self.store.index("test-ws", "v2", [0.0, 1.0, 0.0], {})  # Orthogonal
        
        # Search with high threshold
        results = self.store.search(
            "test-ws",
            [1.0, 0.0, 0.0],
            top_k=10,
            score_threshold=0.9,
        )
        
        assert len(results) == 1
        assert results[0].id == "v1"


# ============================================
# SALIENCE ENGINE TESTS
# ============================================

class TestSalienceEngine:
    """Tests for salience scoring."""
    
    def setup_method(self):
        """Create salience engine."""
        self.engine = SalienceEngine()
    
    def test_recency_score(self):
        """Recency scoring with time decay."""
        now = time.time()
        
        # Just now - high score
        score_now = self.engine.recency_score(now, now)
        assert score_now > 0.99
        
        # 7 days ago (halflife) - should be ~0.5
        score_week = self.engine.recency_score(now - 7 * 86400, now)
        assert 0.4 < score_week < 0.6
        
        # 14 days ago - should be ~0.25
        score_2weeks = self.engine.recency_score(now - 14 * 86400, now)
        assert 0.2 < score_2weeks < 0.3
    
    def test_frequency_score(self):
        """Frequency scoring."""
        # No accesses
        assert self.engine.frequency_score(0) == 0.0
        
        # Some accesses
        score_5 = self.engine.frequency_score(5)
        score_50 = self.engine.frequency_score(50)
        score_100 = self.engine.frequency_score(100)
        
        # Should increase with count
        assert score_5 < score_50 < score_100
        
        # 100 should be near max
        assert score_100 > 0.9
    
    def test_composite_score(self):
        """Composite weighted scoring."""
        score = self.engine.compute(
            event_id="evt_1",
            timestamp=time.time(),  # Now
            access_count=50,
            avg_similarity=0.8,
            method=SalienceMethod.COMPOSITE,
        )
        
        assert 0 < score.score <= 1
        assert "recency" in score.factors
        assert "frequency" in score.factors
        assert "semantic" in score.factors
    
    def test_explicit_score(self):
        """Explicit user-provided scores."""
        score = self.engine.compute(
            event_id="evt_1",
            timestamp=time.time(),
            explicit_score=0.75,
            method=SalienceMethod.EXPLICIT,
        )
        
        assert score.score == 0.75
    
    def test_batch_compute(self):
        """Batch salience computation."""
        now = time.time()
        events = [
            {"event_id": "e1", "timestamp": now, "access_count": 0},
            {"event_id": "e2", "timestamp": now - 86400, "access_count": 10},
            {"event_id": "e3", "timestamp": now - 604800, "access_count": 100},
        ]
        
        scores = self.engine.compute_batch(events)
        
        assert len(scores) == 3
        # Newer with fewer accesses vs older with more accesses
        # Depends on weights, but all should be valid
        assert all(0 < s.score <= 1 for s in scores)
    
    def test_rank_by_salience(self):
        """Ranking by salience."""
        now = time.time()
        events = [
            {"event_id": "old", "timestamp": now - 604800, "access_count": 0},
            {"event_id": "new", "timestamp": now, "access_count": 0},
            {"event_id": "mid", "timestamp": now - 86400, "access_count": 50},
        ]
        
        ranked = self.engine.rank_by_salience(events)
        
        # New should rank highest (recency dominates with 0.4 weight)
        assert ranked[0][0]["event_id"] == "new"
    
    def test_threshold_age(self):
        """Calculate age at threshold."""
        # At halflife, score should be 0.5
        halflife = self.engine.config.recency_halflife
        age_at_half = self.engine.threshold_age(0.5)
        
        assert abs(age_at_half - halflife) < 1  # Within 1 second


# ============================================
# WORKER INTEGRATION TESTS
# ============================================

class TestComputeWorker:
    """Tests for compute worker."""
    
    def setup_method(self):
        """Create worker components."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_file.close()
        
        self.queue = JobQueue(self.temp_file.name)
        self.embeddings = EmbeddingEngine()
        self.vectors = VectorStore(backend=VectorStoreBackend.MEMORY)
        self.salience = SalienceEngine()
        
        self.worker = ComputeWorker(
            queue=self.queue,
            embeddings=self.embeddings,
            vectors=self.vectors,
            salience=self.salience,
            config=WorkerConfig(
                batch_size=5,
                poll_interval=0.1,
            ),
        )
    
    def teardown_method(self):
        """Cleanup."""
        self.queue.close()
        os.unlink(self.temp_file.name)
    
    @pytest.mark.asyncio
    async def test_process_embed_job(self):
        """Process single embedding job."""
        # Enqueue job
        self.queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test-ws",
            payload={
                "event_id": "evt_1",
                "text": "Hello world",
                "metadata": {"user_id": "user_1"},
            },
        )
        
        # Process one batch
        jobs = self.queue.dequeue(batch_size=1)
        assert len(jobs) == 1
        
        await self.worker._process_job(jobs[0])
        
        # Check vector was indexed
        result = self.vectors.get("test-ws", "evt_1")
        assert result is not None
        assert result.payload["text_preview"] == "Hello world"
        
        # Check job completed
        job = self.queue.get(jobs[0].job_id)
        assert job.status == JobStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_process_batch_embed_job(self):
        """Process batch embedding job."""
        self.queue.enqueue(
            job_type=JobType.EMBED_BATCH,
            workspace_id="test-ws",
            payload={
                "items": [
                    {"event_id": "evt_1", "text": "Hello"},
                    {"event_id": "evt_2", "text": "World"},
                    {"event_id": "evt_3", "text": "Test"},
                ],
            },
        )
        
        jobs = self.queue.dequeue(batch_size=1)
        await self.worker._process_job(jobs[0])
        
        # Check all vectors indexed
        assert self.vectors.count("test-ws") == 3
    
    @pytest.mark.asyncio
    async def test_process_delete_vector_job(self):
        """Process vector deletion job."""
        # First index a vector
        self.vectors.ensure_collection("test-ws", 384)
        self.vectors.index("test-ws", "evt_1", [0.1] * 384, {})
        
        assert self.vectors.get("test-ws", "evt_1") is not None
        
        # Enqueue delete job
        self.queue.enqueue(
            job_type=JobType.DELETE_VECTOR,
            workspace_id="test-ws",
            payload={"event_id": "evt_1"},
        )
        
        jobs = self.queue.dequeue(batch_size=1)
        await self.worker._process_job(jobs[0])
        
        # Check vector deleted
        assert self.vectors.get("test-ws", "evt_1") is None
    
    @pytest.mark.asyncio
    async def test_worker_stats(self):
        """Worker statistics tracking."""
        # Enqueue some jobs
        for i in range(3):
            self.queue.enqueue(
                job_type=JobType.EMBED,
                workspace_id="test-ws",
                payload={"event_id": f"evt_{i}", "text": f"Text {i}"},
            )
        
        # Process all
        jobs = self.queue.dequeue(batch_size=3)
        for job in jobs:
            await self.worker._process_job(job)
        
        stats = self.worker.get_stats_dict()
        assert stats["jobs_processed"] == 3
        assert stats["embeddings_generated"] == 3
        assert stats["vectors_indexed"] == 3


# ============================================
# RUN TESTS
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
