"""
KRNX E2E Test Suite 1: Comprehensive Functionality Tests

Tests each component in ISOLATION to verify correct behavior.

Components tested:
- Kernel: STM, LTM, Controller
- Compute: Embeddings, Vectors, Queue, Salience
- Fabric: Orchestrator, Context, Identity, Retention
- Enrichment: Features, Relations, Salience, Structural

Run:
    cd /mnt/d/chillbot
    python3 -m pytest chillbot/tests/test_e2e_functionality.py -v
    
    # Or standalone:
    python3 chillbot/tests/test_e2e_functionality.py
"""

import sys
import os
import time
import json
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Path setup
_this_file = os.path.abspath(__file__)
_tests_dir = os.path.dirname(_this_file)
_chillbot_dir = os.path.dirname(_tests_dir)
_root_dir = os.path.dirname(_chillbot_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


# ==============================================
# TEST CONFIGURATION
# ==============================================

@dataclass
class TestConfig:
    """Test configuration."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    qdrant_url: str = "http://localhost:6333"
    temp_dir: str = ""
    
    def __post_init__(self):
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="krnx_test_")


CONFIG = TestConfig()


# ==============================================
# TEST UTILITIES
# ==============================================

class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[str] = []
    
    def record_pass(self, name: str):
        self.passed += 1
        print(f"  ✓ {name}")
    
    def record_fail(self, name: str, error: str):
        self.failed += 1
        self.errors.append(f"{name}: {error}")
        print(f"  ✗ {name}: {error}")
    
    def record_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⊘ {name}: SKIPPED ({reason})")
    
    def summary(self) -> str:
        total = self.passed + self.failed + self.skipped
        return f"Results: {self.passed}/{total} passed, {self.failed} failed, {self.skipped} skipped"


def check_redis_available() -> bool:
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host=CONFIG.redis_host, port=CONFIG.redis_port)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


def check_qdrant_available() -> bool:
    """Check if Qdrant is available."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=CONFIG.qdrant_url)
        client.get_collections()
        client.close()
        return True
    except Exception:
        return False


# ==============================================
# 1. KERNEL TESTS
# ==============================================

def test_kernel_models(results: TestResult):
    """Test kernel data models."""
    print("\n=== Kernel Models ===")
    
    try:
        from chillbot.kernel.models import Event, create_event
        
        # Test Event creation
        event = create_event(
            event_id="evt_test_001",
            workspace_id="test_workspace",
            user_id="test_user",
            content={"text": "Hello, world!"},
            channel="test",
            ttl_seconds=3600,
        )
        
        assert event.event_id == "evt_test_001"
        assert event.workspace_id == "test_workspace"
        assert event.content["text"] == "Hello, world!"
        results.record_pass("Event creation")
        
        # Test serialization
        event_dict = event.to_dict()
        assert "event_id" in event_dict
        assert "content" in event_dict
        results.record_pass("Event to_dict")
        
        event_json = event.to_json()
        restored = Event.from_json(event_json)
        assert restored.event_id == event.event_id
        results.record_pass("Event JSON serialization")
        
        # Test hash computation
        event_hash = event.compute_hash()
        assert len(event_hash) == 64  # SHA-256 hex
        results.record_pass("Event hash computation")
        
        # Test TTL expiration check
        assert not event.is_expired()
        results.record_pass("Event TTL check")
        
    except Exception as e:
        results.record_fail("Kernel models", str(e))


def test_kernel_stm(results: TestResult):
    """Test STM (Short-Term Memory)."""
    print("\n=== Kernel STM ===")
    
    if not check_redis_available():
        results.record_skip("STM tests", "Redis not available")
        return
    
    try:
        from chillbot.kernel.connection_pool import configure_pool, close_pool
        from chillbot.kernel.stm import STM
        from chillbot.kernel.models import create_event
        
        # Configure connection pool
        configure_pool(host=CONFIG.redis_host, port=CONFIG.redis_port)
        
        stm = STM(ttl_hours=1, use_connection_pool=True)
        results.record_pass("STM initialization")
        
        # Test write
        event = create_event(
            event_id=f"evt_stm_test_{int(time.time())}",
            workspace_id="test_workspace",
            user_id="test_user",
            content={"text": "STM test event"},
        )
        
        msg_id = stm.write_event("test_workspace", "test_user", event)
        assert msg_id is not None
        results.record_pass("STM write_event")
        
        # Test read
        retrieved = stm.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == event.event_id
        results.record_pass("STM get_event")
        
        # Test query
        events = stm.get_events("test_workspace", "test_user", limit=10)
        assert len(events) > 0
        results.record_pass("STM get_events")
        
        # Test stats
        stats = stm.get_stats()
        assert "connected" in stats
        results.record_pass("STM get_stats")
        
        # Cleanup
        stm.delete_user("test_workspace", "test_user")
        close_pool()
        results.record_pass("STM cleanup")
        
    except Exception as e:
        results.record_fail("STM tests", str(e))
        try:
            close_pool()
        except:
            pass


def test_kernel_ltm(results: TestResult):
    """Test LTM (Long-Term Memory)."""
    print("\n=== Kernel LTM ===")
    
    try:
        from chillbot.kernel.ltm import LTM
        from chillbot.kernel.models import create_event
        
        # Create LTM with temp directory
        ltm_path = os.path.join(CONFIG.temp_dir, "ltm_test")
        ltm = LTM(data_path=ltm_path, high_throughput_mode=False)
        results.record_pass("LTM initialization")
        
        # Test single event store
        event = create_event(
            event_id=f"evt_ltm_test_{int(time.time())}",
            workspace_id="test_workspace",
            user_id="test_user",
            content={"text": "LTM test event"},
        )
        
        ltm.store_event(event)
        results.record_pass("LTM store_event")
        
        # Test retrieval
        retrieved = ltm.get_event(event.event_id)
        assert retrieved is not None
        assert retrieved.event_id == event.event_id
        results.record_pass("LTM get_event")
        
        # Test batch store
        batch_events = [
            create_event(
                event_id=f"evt_ltm_batch_{i}_{int(time.time())}",
                workspace_id="test_workspace",
                user_id="test_user",
                content={"text": f"Batch event {i}"},
            )
            for i in range(10)
        ]
        
        stored = ltm.store_events_batch(batch_events)
        assert stored == 10
        results.record_pass("LTM store_events_batch")
        
        # Test query
        events = ltm.query_events("test_workspace", user_id="test_user", limit=20)
        assert len(events) >= 10
        results.record_pass("LTM query_events")
        
        # Test stats
        stats = ltm.get_stats()
        assert "warm_events" in stats
        results.record_pass("LTM get_stats")
        
        # Test integrity
        integrity = ltm.verify_integrity()
        assert integrity["healthy"]
        results.record_pass("LTM verify_integrity")
        
        # Cleanup
        ltm.close()
        results.record_pass("LTM cleanup")
        
    except Exception as e:
        results.record_fail("LTM tests", str(e))


def test_kernel_controller(results: TestResult):
    """Test KRNXController."""
    print("\n=== Kernel Controller ===")
    
    if not check_redis_available():
        results.record_skip("Controller tests", "Redis not available")
        return
    
    try:
        from chillbot.kernel.controller import KRNXController, create_krnx
        from chillbot.kernel.models import create_event
        
        # Create controller
        controller_path = os.path.join(CONFIG.temp_dir, "controller_test")
        controller = create_krnx(
            data_path=controller_path,
            redis_host=CONFIG.redis_host,
            redis_port=CONFIG.redis_port,
            enable_async_worker=False,  # Disable for testing
        )
        results.record_pass("Controller initialization")
        
        # Test turbo write
        event = create_event(
            event_id=f"evt_ctrl_test_{int(time.time())}",
            workspace_id="test_workspace",
            user_id="test_user",
            content={"text": "Controller test event"},
        )
        
        event_id = controller.write_event_turbo(
            workspace_id="test_workspace",
            user_id="test_user",
            event=event,
        )
        assert event_id == event.event_id
        results.record_pass("Controller write_event_turbo")
        
        # Test metrics
        metrics = controller.get_worker_metrics()
        assert hasattr(metrics, "queue_depth")
        results.record_pass("Controller get_worker_metrics")
        
        # Cleanup
        controller.shutdown(timeout=5.0)
        results.record_pass("Controller shutdown")
        
    except Exception as e:
        results.record_fail("Controller tests", str(e))


# ==============================================
# 2. COMPUTE TESTS
# ==============================================

def test_compute_embeddings(results: TestResult):
    """Test embedding engine."""
    print("\n=== Compute Embeddings ===")
    
    try:
        from chillbot.compute.embeddings import EmbeddingEngine
        
        engine = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        results.record_pass("EmbeddingEngine initialization")
        
        # Test dimension
        assert engine.dimension == 384
        results.record_pass("Embedding dimension")
        
        # Test single embedding
        vector = engine.embed("Hello, world!")
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)
        results.record_pass("Single embedding")
        
        # Test batch embedding
        texts = ["Hello", "World", "Test"]
        vectors = engine.embed_batch(texts)
        assert len(vectors) == 3
        assert all(len(v) == 384 for v in vectors)
        results.record_pass("Batch embedding")
        
        # Test similarity
        vec1 = engine.embed("I love dogs")
        vec2 = engine.embed("I adore puppies")
        vec3 = engine.embed("The weather is nice")
        
        sim_related = engine.similarity(vec1, vec2)
        sim_unrelated = engine.similarity(vec1, vec3)
        
        assert sim_related > sim_unrelated
        results.record_pass("Embedding similarity")
        
        # Test empty text handling
        empty_vec = engine.embed("")
        assert len(empty_vec) == 384
        assert all(v == 0.0 for v in empty_vec)
        results.record_pass("Empty text handling")
        
    except ImportError as e:
        results.record_skip("Embeddings tests", f"sentence-transformers not installed: {e}")
    except Exception as e:
        results.record_fail("Embeddings tests", str(e))


def test_compute_vectors(results: TestResult):
    """Test vector store."""
    print("\n=== Compute Vectors ===")
    
    try:
        from chillbot.compute.vectors import VectorStore, VectorStoreBackend
        
        # Test with in-memory backend (no Qdrant needed)
        vectors = VectorStore(backend=VectorStoreBackend.MEMORY)
        results.record_pass("VectorStore initialization (memory)")
        
        # Test collection creation
        vectors.ensure_collection("test_workspace", dimension=384)
        results.record_pass("ensure_collection")
        
        # Test indexing
        test_vector = [0.1] * 384
        vectors.index(
            workspace_id="test_workspace",
            id="vec_001",
            vector=test_vector,
            payload={"text": "Test document"},
        )
        results.record_pass("Vector index")
        
        # Test batch indexing
        batch_data = [
            {"id": f"vec_{i}", "vector": [0.1 + i * 0.01] * 384, "payload": {"text": f"Doc {i}"}}
            for i in range(10)
        ]
        vectors.index_batch("test_workspace", batch_data)
        results.record_pass("Vector index_batch")
        
        # Test search
        query_vector = [0.1] * 384
        matches = vectors.search("test_workspace", query_vector, top_k=5)
        assert len(matches) > 0
        results.record_pass("Vector search")
        
        # Test count
        count = vectors.count("test_workspace")
        assert count >= 10
        results.record_pass("Vector count")
        
        # Test delete
        vectors.delete("test_workspace", "vec_001")
        results.record_pass("Vector delete")
        
        # Test Qdrant backend if available
        if check_qdrant_available():
            qdrant_vectors = VectorStore(
                url=CONFIG.qdrant_url,
                backend=VectorStoreBackend.QDRANT,
            )
            assert qdrant_vectors.health_check()
            results.record_pass("VectorStore Qdrant health_check")
            qdrant_vectors.close()
        else:
            results.record_skip("Qdrant tests", "Qdrant not available")
        
    except Exception as e:
        results.record_fail("Vectors tests", str(e))


def test_compute_queue(results: TestResult):
    """Test job queue."""
    print("\n=== Compute Queue ===")
    
    if not check_redis_available():
        results.record_skip("Queue tests", "Redis not available")
        return
    
    try:
        from chillbot.kernel.connection_pool import configure_pool, close_pool
        from chillbot.compute.queue import JobQueue, JobType
        
        configure_pool(host=CONFIG.redis_host, port=CONFIG.redis_port)
        
        queue = JobQueue(stream_name="krnx:test:jobs", group_name="test-workers")
        results.record_pass("JobQueue initialization")
        
        # Test enqueue
        job_id = queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test_workspace",
            payload={"text": "Test job", "event_id": "evt_001"},
        )
        assert job_id.startswith("job_")
        results.record_pass("JobQueue enqueue")
        
        # Test batch enqueue
        jobs = [
            {"job_type": JobType.EMBED, "workspace_id": "test", "payload": {"i": i}}
            for i in range(5)
        ]
        job_ids = queue.enqueue_batch(jobs)
        assert len(job_ids) == 5
        results.record_pass("JobQueue enqueue_batch")
        
        # Test dequeue
        dequeued = queue.dequeue_batch(
            consumer_name="test-consumer",
            count=10,
            block_ms=100,
        )
        assert len(dequeued) >= 1
        results.record_pass("JobQueue dequeue_batch")
        
        # Test complete
        for job in dequeued:
            queue.complete(job)
        results.record_pass("JobQueue complete")
        
        # Test stats
        stats = queue.get_stats()
        assert "stream_name" in stats
        results.record_pass("JobQueue get_stats")
        
        # Cleanup
        queue.clear()
        close_pool()
        results.record_pass("JobQueue cleanup")
        
    except Exception as e:
        results.record_fail("Queue tests", str(e))
        try:
            close_pool()
        except:
            pass


def test_compute_salience(results: TestResult):
    """Test salience scoring."""
    print("\n=== Compute Salience ===")
    
    try:
        from chillbot.compute.salience import SalienceEngine, SalienceMethod
        
        engine = SalienceEngine()
        results.record_pass("SalienceEngine initialization")
        
        now = time.time()
        
        # Test recency score
        recent_score = engine.recency_score(now - 3600)  # 1 hour ago
        old_score = engine.recency_score(now - 86400 * 7)  # 1 week ago
        assert recent_score > old_score
        results.record_pass("Recency scoring")
        
        # Test frequency score
        low_freq = engine.frequency_score(1)
        high_freq = engine.frequency_score(50)
        assert high_freq > low_freq
        results.record_pass("Frequency scoring")
        
        # Test composite score
        score = engine.compute(
            event_id="evt_001",
            timestamp=now - 3600,
            access_count=10,
            avg_similarity=0.7,
            method=SalienceMethod.COMPOSITE,
        )
        assert 0 <= score.score <= 1
        assert "recency" in score.factors
        results.record_pass("Composite scoring")
        
        # Test batch scoring
        events = [
            {"event_id": f"evt_{i}", "timestamp": now - i * 3600, "access_count": i}
            for i in range(5)
        ]
        scores = engine.compute_batch(events)
        assert len(scores) == 5
        results.record_pass("Batch scoring")
        
    except Exception as e:
        results.record_fail("Salience tests", str(e))


# ==============================================
# 3. FABRIC TESTS
# ==============================================

def test_fabric_context(results: TestResult):
    """Test context builder."""
    print("\n=== Fabric Context ===")
    
    try:
        from chillbot.fabric.context import ContextBuilder, ContextConfig
        
        builder = ContextBuilder(max_tokens=4000)
        results.record_pass("ContextBuilder initialization")
        
        # Create mock memories
        class MockMemory:
            def __init__(self, event_id, content, timestamp, score):
                self.event_id = event_id
                self.content = content
                self.timestamp = timestamp
                self.score = score
                self.metadata = {}
        
        memories = [
            MockMemory("evt_1", {"text": "User likes hiking"}, time.time() - 3600, 0.9),
            MockMemory("evt_2", {"text": "User prefers mountains"}, time.time() - 7200, 0.8),
            MockMemory("evt_3", {"text": "User enjoys nature"}, time.time() - 86400, 0.7),
        ]
        
        # Test text format
        text_ctx = builder.build(memories, "plan a trip", format="text")
        assert isinstance(text_ctx, str)
        assert len(text_ctx) > 0
        results.record_pass("Context build (text)")
        
        # Test JSON format
        json_ctx = builder.build(memories, "plan a trip", format="json")
        assert isinstance(json_ctx, dict)
        assert "memories" in json_ctx
        results.record_pass("Context build (json)")
        
        # Test messages format
        msg_ctx = builder.build(memories, "plan a trip", format="messages")
        assert isinstance(msg_ctx, list)
        assert len(msg_ctx) > 0
        results.record_pass("Context build (messages)")
        
        # Test size estimation
        size = builder.estimate_context_size(memories)
        assert "total_tokens" in size
        results.record_pass("Context size estimation")
        
    except Exception as e:
        results.record_fail("Context tests", str(e))


def test_fabric_identity(results: TestResult):
    """Test identity resolver."""
    print("\n=== Fabric Identity ===")
    
    try:
        from chillbot.fabric.identity import IdentityResolver, IdentityType
        
        resolver = IdentityResolver(default_workspace="default", default_user="anonymous")
        results.record_pass("IdentityResolver initialization")
        
        # Test agent registration
        resolver.register_agent(
            agent_id="coder-1",
            workspace_id="project-x",
            role="coder",
            permissions=["read", "write"],
        )
        results.record_pass("Agent registration")
        
        # Test agent resolution
        identity = resolver.resolve("coder-1")
        assert identity.workspace_id == "project-x"
        assert identity.type == IdentityType.AGENT
        results.record_pass("Agent resolution")
        
        # Test scope resolution
        identity = resolver.resolve_scope("org:acme/project:alpha/user:bob")
        assert identity.org_id == "acme"
        assert identity.project_id == "alpha"
        assert identity.user_id == "bob"
        results.record_pass("Scope resolution")
        
        # Test validation
        errors = resolver.validate_identity(identity)
        assert len(errors) == 0
        results.record_pass("Identity validation")
        
    except Exception as e:
        results.record_fail("Identity tests", str(e))


def test_fabric_retention(results: TestResult):
    """Test retention manager."""
    print("\n=== Fabric Retention ===")
    
    try:
        from chillbot.fabric.retention import (
            RetentionManager, RetentionPolicy, RetentionAction, RetentionClass
        )
        
        manager = RetentionManager()
        results.record_pass("RetentionManager initialization")
        
        # Test default policies exist
        policies = manager.list_policies()
        assert len(policies) > 0
        results.record_pass("Default policies loaded")
        
        # Test custom policy
        manager.add_policy(RetentionPolicy(
            name="test_ttl",
            ttl_seconds=60,
            action=RetentionAction.DELETE,
            priority=50,
        ))
        results.record_pass("Custom policy added")
        
        # Test evaluation
        class MockMemory:
            def __init__(self, timestamp, retention_class=None):
                self.event_id = "evt_test"
                self.timestamp = timestamp
                self.retention_class = retention_class
        
        # Fresh memory - should keep
        fresh = MockMemory(time.time())
        eval_result = manager.evaluate(fresh)
        assert eval_result.action == RetentionAction.KEEP
        results.record_pass("Fresh memory evaluation")
        
        # Permanent memory - should keep regardless of age
        old_permanent = MockMemory(time.time() - 86400 * 365, "permanent")
        eval_result = manager.evaluate(old_permanent)
        assert eval_result.action == RetentionAction.KEEP
        results.record_pass("Permanent memory evaluation")
        
    except Exception as e:
        results.record_fail("Retention tests", str(e))


# ==============================================
# 4. ENRICHMENT TESTS
# ==============================================

def test_enrichment_features(results: TestResult):
    """Test feature extraction."""
    print("\n=== Enrichment Features ===")
    
    try:
        from chillbot.fabric.enrichment.features import (
            FeatureExtractor,
            negation_mismatch,
            numeric_mismatch,
            antonym_detected,
        )
        
        extractor = FeatureExtractor()
        results.record_pass("FeatureExtractor initialization")
        
        # Test negation detection
        assert negation_mismatch("I agree", "I do not agree")
        results.record_pass("Negation mismatch")
        
        # Test numeric detection
        assert numeric_mismatch("Budget is $50,000", "Budget is $75,000")
        results.record_pass("Numeric mismatch")
        
        # Test antonym detection
        assert antonym_detected("The project succeeded", "The project failed")
        results.record_pass("Antonym detection")
        
        # Test full extraction
        class MockEvent:
            def __init__(self, content, timestamp=0):
                self.content = content
                self.timestamp = timestamp
                self.user_id = "user1"
                self.metadata = {}
        
        event_a = MockEvent("Budget is $75,000")
        event_b = MockEvent("Budget is $50,000")
        
        features = extractor.extract(event_a, event_b, embedding_similarity=0.85)
        assert features.numeric_mismatch
        assert features.embedding_similarity == 0.85
        results.record_pass("Full feature extraction")
        
    except Exception as e:
        results.record_fail("Feature tests", str(e))


def test_enrichment_relations(results: TestResult):
    """Test relation scoring."""
    print("\n=== Enrichment Relations ===")
    
    try:
        from chillbot.fabric.enrichment.relations import RelationScorer, RelationType
        from chillbot.fabric.enrichment.features import FeatureExtractor
        
        scorer = RelationScorer()
        extractor = FeatureExtractor()
        results.record_pass("RelationScorer initialization")
        
        class MockEvent:
            def __init__(self, event_id, content, timestamp):
                self.event_id = event_id
                self.content = content
                self.timestamp = timestamp
                self.user_id = "user1"
                self.metadata = {}
        
        # Test supersedes relation
        old_event = MockEvent("evt_001", "Budget is $50,000", time.time() - 3600)
        new_event = MockEvent("evt_002", "Budget is now $75,000", time.time())
        
        features = extractor.extract(new_event, old_event, embedding_similarity=0.85)
        relations = scorer.score_pair(new_event, old_event, features)
        
        # Should detect supersedes due to numeric mismatch
        supersedes = [r for r in relations if r.kind == RelationType.SUPERSEDES]
        assert len(supersedes) > 0
        results.record_pass("Supersedes relation detection")
        
    except Exception as e:
        results.record_fail("Relation tests", str(e))


def test_enrichment_structural(results: TestResult):
    """Test structural analysis."""
    print("\n=== Enrichment Structural ===")
    
    try:
        from chillbot.fabric.enrichment.structural import (
            StructuralAnalyzer,
            compute_event_density,
            is_episode_boundary,
            compute_structural_salience,
        )
        
        analyzer = StructuralAnalyzer()
        results.record_pass("StructuralAnalyzer initialization")
        
        class MockEvent:
            def __init__(self, timestamp):
                self.timestamp = timestamp
        
        now = time.time()
        
        # Test event density
        recent_events = [MockEvent(now - i * 10) for i in range(5)]
        density = compute_event_density(now, recent_events, window_seconds=60)
        assert density > 0
        results.record_pass("Event density computation")
        
        # Test episode boundary
        prev_event = MockEvent(now - 600)  # 10 min gap
        assert is_episode_boundary(now, prev_event, gap_threshold=300)
        results.record_pass("Episode boundary detection")
        
        # Test structural salience
        salience = compute_structural_salience(
            is_boundary=True,
            is_correction=False,
            relation_count=2,
        )
        assert 0 <= salience <= 1
        results.record_pass("Structural salience computation")
        
    except Exception as e:
        results.record_fail("Structural tests", str(e))


def test_enrichment_salience(results: TestResult):
    """Test enrichment salience."""
    print("\n=== Enrichment Salience ===")
    
    try:
        from chillbot.fabric.enrichment.salience import (
            SalienceEngine,
            compute_salience_breakdown,
        )
        
        engine = SalienceEngine()
        results.record_pass("SalienceEngine initialization")
        
        now = time.time()
        
        # Test breakdown computation
        breakdown = compute_salience_breakdown(
            timestamp=now - 3600,
            access_count=5,
            avg_similarity=0.7,
            structural_score=0.6,
            now=now,
        )
        
        assert "semantic" in breakdown
        assert "recency" in breakdown
        assert "frequency" in breakdown
        assert "structural" in breakdown
        assert "final" in breakdown
        results.record_pass("Salience breakdown")
        
    except Exception as e:
        results.record_fail("Salience tests", str(e))


def test_enrichment_schema(results: TestResult):
    """Test schema output."""
    print("\n=== Enrichment Schema ===")
    
    try:
        from chillbot.fabric.enrichment.schema import MetadataBuilder
        from chillbot.fabric.enrichment.relations import RelationType, RelationResult
        
        # Create mock relation
        relation = RelationResult(
            kind=RelationType.SUPERSEDES,
            target="evt_001",
            confidence=0.85,
            signals=["numeric_mismatch"],
            reason_code="UPDATE_NUMERIC",
            strict_contradiction=False,
        )
        
        # Build metadata
        metadata = (
            MetadataBuilder()
            .with_salience(semantic=0.8, recency=0.6, frequency=0.2, structural=0.5)
            .with_relations([relation])
            .with_retention("durable")
            .with_temporal(episode_id="ep_001", is_boundary=False, drift_factor=0.1)
            .with_confidence(0.95)
            .build()
        )
        
        output = metadata.to_dict()
        
        assert "salience" in output
        assert "relations" in output
        assert "retention_class" in output
        assert "temporal" in output
        assert "confidence" in output
        results.record_pass("Schema output generation")
        
    except Exception as e:
        results.record_fail("Schema tests", str(e))


# ==============================================
# MAIN
# ==============================================

def main():
    """Run all functionality tests."""
    print("=" * 70)
    print("KRNX E2E Test Suite 1: Comprehensive Functionality Tests")
    print("=" * 70)
    print(f"Temp directory: {CONFIG.temp_dir}")
    print(f"Redis: {CONFIG.redis_host}:{CONFIG.redis_port} ({'available' if check_redis_available() else 'unavailable'})")
    print(f"Qdrant: {CONFIG.qdrant_url} ({'available' if check_qdrant_available() else 'unavailable'})")
    
    results = TestResult()
    
    # Run all test groups
    test_groups = [
        # Kernel
        test_kernel_models,
        test_kernel_stm,
        test_kernel_ltm,
        test_kernel_controller,
        # Compute
        test_compute_embeddings,
        test_compute_vectors,
        test_compute_queue,
        test_compute_salience,
        # Fabric
        test_fabric_context,
        test_fabric_identity,
        test_fabric_retention,
        # Enrichment
        test_enrichment_features,
        test_enrichment_relations,
        test_enrichment_structural,
        test_enrichment_salience,
        test_enrichment_schema,
    ]
    
    for test_fn in test_groups:
        try:
            test_fn(results)
        except Exception as e:
            results.record_fail(test_fn.__name__, f"Uncaught exception: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(results.summary())
    
    if results.errors:
        print("\nErrors:")
        for error in results.errors:
            print(f"  - {error}")
    
    print("=" * 70)
    
    # Cleanup temp directory
    import shutil
    try:
        shutil.rmtree(CONFIG.temp_dir)
    except:
        pass
    
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
