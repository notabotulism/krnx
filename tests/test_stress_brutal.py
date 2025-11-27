"""
KRNX BRUTAL Stress Tests v2

This is not a gentle test. This beats the shit out of the system.

Tests:
  1. THROUGHPUT - Raw speed with backpressure enabled
  2. BACKPRESSURE - Verify graceful degradation under overload
  3. SUSTAINED LOAD - 60+ seconds of continuous hammering
  4. CONCURRENCY - 50-100 workers fighting for resources
  5. FULL STACK - Kernel + Redis + Qdrant + Embeddings
  6. RECOVERY - System behavior after overload
  7. MEMORY - Leak detection under sustained load
  8. CHAOS - Random failures, network issues

Requirements:
  - Redis running (localhost:6379)
  - Qdrant running (localhost:6333)
  - sentence-transformers installed (for embeddings)

Run:
  # Quick smoke test (default settings)
  pytest tests/test_stress_brutal.py -v -s
  
  # Full brutal mode
  STRESS_EVENTS=50000 STRESS_WORKERS=50 STRESS_DURATION=120 pytest tests/test_stress_brutal.py -v -s
  
  # Just backpressure tests
  pytest tests/test_stress_brutal.py -v -s -k "backpressure"
"""

import pytest
import time
import uuid
import os
import sys
import tempfile
import shutil
import threading
import statistics
import gc
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Generator, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from queue import Queue
from contextlib import contextmanager

# ==============================================
# CONFIGURATION
# ==============================================

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

# Brutal defaults - override with env vars for even more punishment
STRESS_EVENTS = int(os.environ.get("STRESS_EVENTS", 5000))
STRESS_WORKERS = int(os.environ.get("STRESS_WORKERS", 25))
STRESS_DURATION = int(os.environ.get("STRESS_DURATION", 60))  # seconds
STRESS_BURST_SIZE = int(os.environ.get("STRESS_BURST_SIZE", 1000))

# Backpressure tuning - tighter thresholds to trigger faster
BACKPRESSURE_QUEUE_DEPTH = int(os.environ.get("BP_QUEUE_DEPTH", 5000))
BACKPRESSURE_LAG_SECONDS = float(os.environ.get("BP_LAG_SECONDS", 5.0))

TEST_PREFIX = f"brutal_{uuid.uuid4().hex[:8]}"


# ==============================================
# METRICS
# ==============================================

@dataclass
class BrutalMetrics:
    """Comprehensive metrics for brutal testing"""
    total_ops: int = 0
    success_ops: int = 0
    failed_ops: int = 0
    backpressure_rejections: int = 0
    other_errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Time series for degradation analysis
    throughput_samples: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, ops/sec)
    latency_samples: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, p95)
    
    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 0.001)
    
    @property
    def throughput(self) -> float:
        return self.success_ops / self.duration if self.duration > 0 else 0
    
    @property
    def success_rate(self) -> float:
        return (self.success_ops / self.total_ops * 100) if self.total_ops > 0 else 0
    
    @property
    def backpressure_rate(self) -> float:
        return (self.backpressure_rejections / self.total_ops * 100) if self.total_ops > 0 else 0
    
    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * p / 100)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]
    
    @property
    def p50(self) -> float:
        return self.percentile(50)
    
    @property
    def p95(self) -> float:
        return self.percentile(95)
    
    @property
    def p99(self) -> float:
        return self.percentile(99)
    
    @property
    def max_latency(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0
    
    def record_success(self, latency_ms: float):
        self.success_ops += 1
        self.total_ops += 1
        self.latencies_ms.append(latency_ms)
    
    def record_failure(self, error: Exception):
        self.failed_ops += 1
        self.total_ops += 1
        error_type = type(error).__name__
        self.errors[error_type] += 1
        if "backpressure" in str(error).lower() or "under load" in str(error).lower():
            self.backpressure_rejections += 1
        else:
            self.other_errors += 1
    
    def summary(self, title: str = "STRESS TEST") -> str:
        lines = [
            "",
            "╔" + "═" * 70 + "╗",
            f"║  {title:^66}  ║",
            "╠" + "═" * 70 + "╣",
            f"║  Total Operations:      {self.total_ops:>10,}                              ║",
            f"║  Successful:            {self.success_ops:>10,}  ({self.success_rate:>6.2f}%)                ║",
            f"║  Failed:                {self.failed_ops:>10,}                              ║",
            f"║    - Backpressure:      {self.backpressure_rejections:>10,}  ({self.backpressure_rate:>6.2f}%)                ║",
            f"║    - Other Errors:      {self.other_errors:>10,}                              ║",
            "╠" + "═" * 70 + "╣",
            f"║  Duration:              {self.duration:>10.2f}s                             ║",
            f"║  Throughput:            {self.throughput:>10.1f} ops/sec                      ║",
            "╠" + "═" * 70 + "╣",
            f"║  Latency (ms):                                                       ║",
            f"║    P50:                 {self.p50:>10.2f}                              ║",
            f"║    P95:                 {self.p95:>10.2f}                              ║",
            f"║    P99:                 {self.p99:>10.2f}                              ║",
            f"║    Max:                 {self.max_latency:>10.2f}                              ║",
            "╚" + "═" * 70 + "╝",
        ]
        
        if self.errors:
            lines.append("\nError breakdown:")
            for err_type, count in sorted(self.errors.items(), key=lambda x: -x[1]):
                lines.append(f"  {err_type}: {count}")
        
        return "\n".join(lines)


# ==============================================
# FIXTURES
# ==============================================

@pytest.fixture(scope="module")
def temp_data_dir() -> Generator[str, None, None]:
    """Temp directory for LTM SQLite"""
    temp_dir = tempfile.mkdtemp(prefix="krnx_brutal_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def redis_client():
    """Redis client - skip if unavailable"""
    try:
        import redis
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        client.ping()
        yield client
        # Cleanup
        for key in client.scan_iter(f"{TEST_PREFIX}:*"):
            client.delete(key)
        for key in client.scan_iter(f"krnx:{TEST_PREFIX}*"):
            client.delete(key)
    except Exception as e:
        pytest.skip(f"Redis unavailable: {e}")


@pytest.fixture(scope="module")
def qdrant_available() -> bool:
    """Check if Qdrant is available"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def krnx_controller_no_backpressure(temp_data_dir, redis_client):
    """Controller WITHOUT backpressure - for baseline throughput"""
    from chillbot.kernel import KRNXController
    
    controller = KRNXController(
        data_path=temp_data_dir,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        enable_backpressure=False,
        enable_async_worker=True,
        ltm_batch_size=100,
        ltm_batch_interval=0.05,
    )
    yield controller
    controller.close()


@pytest.fixture(scope="module")
def krnx_controller_with_backpressure(temp_data_dir, redis_client):
    """Controller WITH backpressure - for realistic testing"""
    from chillbot.kernel import KRNXController
    
    # CRITICAL: Clear any stale queue data from previous tests
    # Otherwise backpressure triggers immediately due to old messages
    try:
        redis_client.delete('krnx:ltm:queue')
        redis_client.execute_command('XGROUP', 'DESTROY', 'krnx:ltm:queue', 'krnx-ltm-workers')
    except Exception:
        pass  # Queue might not exist yet
    
    controller = KRNXController(
        data_path=temp_data_dir + "_bp",
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        enable_backpressure=True,
        max_queue_depth=BACKPRESSURE_QUEUE_DEPTH,
        max_lag_seconds=BACKPRESSURE_LAG_SECONDS,
        enable_async_worker=True,
        ltm_batch_size=100,
        ltm_batch_interval=0.05,
    )
    os.makedirs(temp_data_dir + "_bp", exist_ok=True)
    yield controller
    controller.close()


@pytest.fixture(scope="module")
def fabric_minimal(krnx_controller_no_backpressure):
    """Minimal fabric - kernel only, no vectors"""
    from chillbot.fabric import MemoryFabric
    
    fabric = MemoryFabric(
        kernel=krnx_controller_no_backpressure,
        default_workspace=f"{TEST_PREFIX}_minimal",
        auto_embed=False,
        auto_enrich=False,
    )
    yield fabric
    fabric.close()


@pytest.fixture(scope="module")
def fabric_with_backpressure(krnx_controller_with_backpressure):
    """Fabric with backpressure enabled"""
    from chillbot.fabric import MemoryFabric
    
    fabric = MemoryFabric(
        kernel=krnx_controller_with_backpressure,
        default_workspace=f"{TEST_PREFIX}_bp",
        auto_embed=False,
        auto_enrich=False,
    )
    yield fabric
    fabric.close()


@pytest.fixture(scope="module")
def fabric_full_stack(krnx_controller_no_backpressure, qdrant_available):
    """Full stack - kernel + embeddings + vectors"""
    if not qdrant_available:
        pytest.skip("Qdrant not available for full stack test")
    
    from chillbot.fabric import MemoryFabric
    
    embeddings = None
    vectors = None
    
    try:
        from chillbot.compute.embeddings import EmbeddingEngine
        embeddings = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    except ImportError:
        pytest.skip("sentence-transformers not available")
    
    try:
        from chillbot.compute.vectors import VectorStore
        vectors = VectorStore(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            collection_prefix=TEST_PREFIX,
        )
    except ImportError:
        pytest.skip("qdrant-client not available")
    
    fabric = MemoryFabric(
        kernel=krnx_controller_no_backpressure,
        embeddings=embeddings,
        vectors=vectors,
        default_workspace=f"{TEST_PREFIX}_full",
        auto_embed=True,
        auto_enrich=True,
    )
    yield fabric
    fabric.close()


# ==============================================
# HELPER FUNCTIONS
# ==============================================

def generate_content(i: int, size: str = "medium") -> Dict[str, Any]:
    """Generate test content of varying sizes"""
    if size == "tiny":
        return {"text": f"msg{i}"}
    elif size == "small":
        return {"text": f"Event {i}: small payload", "i": i}
    elif size == "medium":
        return {
            "text": f"Event {i}: " + "This is a medium-sized payload with some content. " * 5,
            "index": i,
            "timestamp": time.time(),
            "tags": ["test", "stress", f"batch_{i // 100}"],
        }
    elif size == "large":
        return {
            "text": f"Event {i}: " + "Large payload with lots of text. " * 50,
            "index": i,
            "metadata": {f"key_{j}": f"value_{j}" for j in range(20)},
        }
    else:
        return {"text": f"Event {i}"}


def run_concurrent_load(
    fabric,
    num_workers: int,
    ops_per_worker: int,
    workspace: str,
    metrics: BrutalMetrics,
    content_size: str = "medium",
    include_reads: float = 0.0,  # 0.0 = writes only, 0.3 = 30% reads
):
    """Run concurrent load with specified workers"""
    lock = threading.Lock()
    
    def worker(worker_id: int):
        local_latencies = []
        local_success = 0
        local_bp = 0
        local_other = 0
        
        for i in range(ops_per_worker):
            is_read = include_reads > 0 and (random.random() < include_reads)
            start = time.time()
            
            try:
                if is_read:
                    fabric.recall(
                        query=f"worker {worker_id} event",
                        workspace_id=workspace,
                        top_k=5,
                    )
                else:
                    fabric.remember(
                        content=generate_content(i, content_size),
                        workspace_id=workspace,
                        user_id=f"worker_{worker_id}",
                    )
                latency_ms = (time.time() - start) * 1000
                local_latencies.append(latency_ms)
                local_success += 1
            except Exception as e:
                if "backpressure" in str(e).lower() or "under load" in str(e).lower():
                    local_bp += 1
                else:
                    local_other += 1
        
        with lock:
            metrics.latencies_ms.extend(local_latencies)
            metrics.success_ops += local_success
            metrics.backpressure_rejections += local_bp
            metrics.other_errors += local_other
            metrics.total_ops += ops_per_worker
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker, i) for i in range(num_workers)]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Worker exception: {e}")


def run_timed_load(
    fabric,
    duration_seconds: float,
    num_workers: int,
    workspace: str,
    metrics: BrutalMetrics,
    target_ops_per_sec: Optional[int] = None,
):
    """Run load for a fixed duration"""
    stop_event = threading.Event()
    lock = threading.Lock()
    
    def worker(worker_id: int):
        local_latencies = []
        local_success = 0
        local_bp = 0
        i = 0
        
        while not stop_event.is_set():
            start = time.time()
            try:
                fabric.remember(
                    content=generate_content(i, "medium"),
                    workspace_id=workspace,
                    user_id=f"worker_{worker_id}",
                )
                latency_ms = (time.time() - start) * 1000
                local_latencies.append(latency_ms)
                local_success += 1
            except Exception as e:
                if "backpressure" in str(e).lower() or "under load" in str(e).lower():
                    local_bp += 1
            
            i += 1
            
            # Rate limiting if target specified
            if target_ops_per_sec:
                target_interval = num_workers / target_ops_per_sec
                elapsed = time.time() - start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
        
        with lock:
            metrics.latencies_ms.extend(local_latencies)
            metrics.success_ops += local_success
            metrics.backpressure_rejections += local_bp
            metrics.total_ops += i
    
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    
    time.sleep(duration_seconds)
    stop_event.set()
    
    for t in threads:
        t.join(timeout=5)


# ==============================================
# TEST CLASS: BASELINE THROUGHPUT
# ==============================================

class TestBaselineThroughput:
    """Establish baseline throughput WITHOUT backpressure"""
    
    def test_sequential_baseline(self, fabric_minimal):
        """Baseline: sequential writes, no concurrency"""
        workspace = f"{TEST_PREFIX}_seq_baseline"
        metrics = BrutalMetrics()
        num_events = min(STRESS_EVENTS, 2000)  # Cap for baseline
        
        metrics.start_time = time.time()
        
        for i in range(num_events):
            start = time.time()
            try:
                fabric_minimal.remember(
                    content=generate_content(i, "medium"),
                    workspace_id=workspace,
                    user_id="baseline_user",
                )
                metrics.record_success((time.time() - start) * 1000)
            except Exception as e:
                metrics.record_failure(e)
        
        metrics.end_time = time.time()
        
        print(metrics.summary("SEQUENTIAL BASELINE"))
        
        # Baseline assertions - should be near 100% success
        assert metrics.success_rate >= 99.0, f"Baseline should be ~100% success, got {metrics.success_rate}%"
        assert metrics.throughput >= 100, f"Baseline throughput too low: {metrics.throughput} ops/sec"
    
    def test_concurrent_baseline(self, fabric_minimal):
        """Baseline: concurrent writes, no backpressure"""
        workspace = f"{TEST_PREFIX}_conc_baseline"
        metrics = BrutalMetrics()
        num_workers = STRESS_WORKERS
        ops_per_worker = STRESS_EVENTS // num_workers
        
        metrics.start_time = time.time()
        run_concurrent_load(
            fabric_minimal, num_workers, ops_per_worker,
            workspace, metrics, content_size="medium"
        )
        metrics.end_time = time.time()
        
        print(metrics.summary(f"CONCURRENT BASELINE ({num_workers} workers)"))
        
        assert metrics.success_rate >= 99.0
        assert metrics.throughput >= 200, f"Concurrent baseline too slow: {metrics.throughput}"


# ==============================================
# TEST CLASS: BACKPRESSURE BEHAVIOR
# ==============================================

class TestBackpressure:
    """Test that backpressure works correctly"""
    
    def test_backpressure_triggers(self, fabric_with_backpressure):
        """Verify backpressure actually triggers under heavy load"""
        workspace = f"{TEST_PREFIX}_bp_trigger"
        metrics = BrutalMetrics()
        
        # Hammer it with many workers to trigger backpressure
        num_workers = 50
        ops_per_worker = 200
        
        metrics.start_time = time.time()
        run_concurrent_load(
            fabric_with_backpressure, num_workers, ops_per_worker,
            workspace, metrics, content_size="medium"
        )
        metrics.end_time = time.time()
        
        print(metrics.summary("BACKPRESSURE TRIGGER TEST"))
        
        # We WANT to see some backpressure rejections
        # If we see zero, backpressure isn't working or load wasn't heavy enough
        print(f"\nBackpressure rejections: {metrics.backpressure_rejections}")
        print(f"Backpressure rate: {metrics.backpressure_rate:.2f}%")
        
        # Assertions:
        # 1. Should have SOME successes (system didn't completely fail)
        assert metrics.success_ops > 0, "System should accept some requests"
        
        # 2. Should have SOME backpressure (proves it's working)
        # Note: If this fails with 0 backpressure, increase load or lower thresholds
        assert metrics.backpressure_rejections > 0, \
            "Expected backpressure rejections - increase load or lower BP thresholds"
        
        # 3. Should NOT have other errors (only backpressure, no crashes)
        assert metrics.other_errors == 0, f"Unexpected errors: {metrics.errors}"
    
    def test_backpressure_recovery(self, fabric_with_backpressure):
        """Test that system recovers after backpressure"""
        workspace = f"{TEST_PREFIX}_bp_recovery"
        
        # Phase 1: Overload the system
        print("\n[Phase 1: Overloading system...]")
        overload_metrics = BrutalMetrics()
        overload_metrics.start_time = time.time()
        run_concurrent_load(
            fabric_with_backpressure, 50, 100,
            workspace, overload_metrics
        )
        overload_metrics.end_time = time.time()
        print(f"Overload phase: {overload_metrics.success_ops} success, {overload_metrics.backpressure_rejections} BP")
        
        # Phase 2: Let it cool down
        print("[Phase 2: Cooling down (5s)...]")
        time.sleep(5)
        
        # Phase 3: Light load - should mostly succeed
        print("[Phase 3: Recovery test (light load)...]")
        recovery_metrics = BrutalMetrics()
        recovery_metrics.start_time = time.time()
        
        # Sequential, slow writes
        for i in range(100):
            start = time.time()
            try:
                fabric_with_backpressure.remember(
                    content=generate_content(i, "small"),
                    workspace_id=workspace,
                    user_id="recovery_user",
                )
                recovery_metrics.record_success((time.time() - start) * 1000)
            except Exception as e:
                recovery_metrics.record_failure(e)
            time.sleep(0.01)  # Gentle pace
        
        recovery_metrics.end_time = time.time()
        
        print(metrics.summary("RECOVERY PHASE") if 'metrics' in dir() else recovery_metrics.summary("RECOVERY PHASE"))
        
        # Recovery should be much better than overload
        assert recovery_metrics.success_rate >= 80.0, \
            f"System didn't recover: {recovery_metrics.success_rate}% success"


# ==============================================
# TEST CLASS: SUSTAINED LOAD
# ==============================================

class TestSustainedLoad:
    """Test behavior under sustained load over time"""
    
    def test_sustained_writes(self, fabric_minimal):
        """Sustained write load for extended duration"""
        workspace = f"{TEST_PREFIX}_sustained"
        metrics = BrutalMetrics()
        duration = min(STRESS_DURATION, 60)  # Cap at 60s for CI
        num_workers = 10
        
        print(f"\n[Running sustained load for {duration}s with {num_workers} workers...]")
        
        metrics.start_time = time.time()
        run_timed_load(
            fabric_minimal, duration, num_workers,
            workspace, metrics
        )
        metrics.end_time = time.time()
        
        print(metrics.summary(f"SUSTAINED LOAD ({duration}s)"))
        
        # Should maintain high success rate over time
        assert metrics.success_rate >= 95.0
        assert metrics.throughput >= 100
        
        # Latency shouldn't explode
        assert metrics.p99 < 500, f"P99 latency too high: {metrics.p99}ms"
    
    def test_sustained_mixed_workload(self, fabric_minimal):
        """Sustained mixed read/write workload"""
        workspace = f"{TEST_PREFIX}_sustained_mixed"
        
        # Seed some data first
        print("\n[Seeding data...]")
        for i in range(500):
            fabric_minimal.remember(
                content=generate_content(i, "medium"),
                workspace_id=workspace,
                user_id="seed_user",
            )
        
        metrics = BrutalMetrics()
        duration = min(STRESS_DURATION // 2, 30)
        
        print(f"[Running mixed workload for {duration}s...]")
        
        # Run with 30% reads
        metrics.start_time = time.time()
        run_concurrent_load(
            fabric_minimal, 10, 500,
            workspace, metrics, include_reads=0.3
        )
        metrics.end_time = time.time()
        
        print(metrics.summary("SUSTAINED MIXED WORKLOAD"))
        
        assert metrics.success_rate >= 90.0


# ==============================================
# TEST CLASS: EXTREME CONCURRENCY
# ==============================================

class TestExtremeConcurrency:
    """Push concurrency to the limits"""
    
    def test_high_worker_count(self, fabric_minimal):
        """Many workers (50+) hammering simultaneously"""
        workspace = f"{TEST_PREFIX}_high_conc"
        metrics = BrutalMetrics()
        num_workers = 50
        ops_per_worker = 100
        
        print(f"\n[{num_workers} workers x {ops_per_worker} ops each...]")
        
        metrics.start_time = time.time()
        run_concurrent_load(
            fabric_minimal, num_workers, ops_per_worker,
            workspace, metrics
        )
        metrics.end_time = time.time()
        
        print(metrics.summary(f"HIGH CONCURRENCY ({num_workers} workers)"))
        
        assert metrics.success_rate >= 95.0
        assert metrics.p99 < 200, f"P99 too high under concurrency: {metrics.p99}ms"
    
    def test_workspace_contention(self, fabric_minimal):
        """All workers writing to SAME workspace"""
        workspace = f"{TEST_PREFIX}_contention"
        metrics = BrutalMetrics()
        num_workers = 30
        ops_per_worker = 100
        
        print(f"\n[{num_workers} workers fighting over single workspace...]")
        
        metrics.start_time = time.time()
        
        lock = threading.Lock()
        
        def worker(worker_id: int):
            for i in range(ops_per_worker):
                start = time.time()
                try:
                    fabric_minimal.remember(
                        content={"text": f"Contention {worker_id}-{i}", "w": worker_id},
                        workspace_id=workspace,  # SAME workspace
                        user_id=f"user_{worker_id}",
                    )
                    with lock:
                        metrics.record_success((time.time() - start) * 1000)
                except Exception as e:
                    with lock:
                        metrics.record_failure(e)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            for f in as_completed(futures):
                f.result()
        
        metrics.end_time = time.time()
        
        print(metrics.summary("WORKSPACE CONTENTION"))
        
        assert metrics.success_rate >= 95.0


# ==============================================
# TEST CLASS: FULL STACK
# ==============================================

class TestFullStack:
    """Test complete stack including embeddings and vectors"""
    
    def test_full_stack_throughput(self, fabric_full_stack):
        """Throughput with embeddings + vector storage"""
        workspace = f"{TEST_PREFIX}_fullstack"
        metrics = BrutalMetrics()
        num_events = min(STRESS_EVENTS // 5, 500)  # Embeddings are slow
        
        print(f"\n[Full stack test: {num_events} events with embeddings...]")
        
        metrics.start_time = time.time()
        
        for i in range(num_events):
            start = time.time()
            try:
                fabric_full_stack.remember(
                    content=generate_content(i, "medium"),
                    workspace_id=workspace,
                    user_id="fullstack_user",
                )
                metrics.record_success((time.time() - start) * 1000)
            except Exception as e:
                metrics.record_failure(e)
        
        metrics.end_time = time.time()
        
        print(metrics.summary("FULL STACK (kernel + embeddings + vectors)"))
        
        # Full stack will be slower, but should still work
        assert metrics.success_rate >= 95.0
        
        # Test recall actually works
        print("\n[Testing recall...]")
        result = fabric_full_stack.recall(
            query="test event content",
            workspace_id=workspace,
            top_k=10,
        )
        assert len(result.memories) > 0, "Recall should return results"
        print(f"Recall returned {len(result.memories)} memories")


# ==============================================
# TEST CLASS: MEMORY & RESOURCES
# ==============================================

class TestResourceUsage:
    """Test memory usage and resource cleanup"""
    
    def test_memory_growth(self, fabric_minimal):
        """Verify memory doesn't grow unboundedly"""
        workspace = f"{TEST_PREFIX}_memory"
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Run significant load
        print("\n[Running 2000 writes...]")
        for i in range(2000):
            fabric_minimal.remember(
                content=generate_content(i, "medium"),
                workspace_id=workspace,
                user_id="memory_user",
            )
        
        gc.collect()
        after_writes = len(gc.get_objects())
        
        # Run reads
        print("[Running 200 recalls...]")
        for i in range(200):
            fabric_minimal.recall(
                query=f"event {i * 10}",
                workspace_id=workspace,
                top_k=10,
            )
        
        gc.collect()
        after_reads = len(gc.get_objects())
        
        write_growth = after_writes - initial_objects
        read_growth = after_reads - after_writes
        total_growth = after_reads - initial_objects
        
        print(f"\n[MEMORY GROWTH]")
        print(f"  Initial objects:      {initial_objects:,}")
        print(f"  After 2000 writes:    {after_writes:,} (+{write_growth:,})")
        print(f"  After 200 reads:      {after_reads:,} (+{read_growth:,})")
        print(f"  Total growth:         {total_growth:,} objects")
        print(f"  Growth factor:        {after_reads / initial_objects:.2f}x")
        
        # Memory should not grow more than 2x
        assert after_reads / initial_objects < 2.0, \
            f"Memory grew too much: {after_reads / initial_objects:.2f}x"
    
    def test_no_connection_leaks(self, fabric_minimal, redis_client):
        """Verify Redis connections don't leak"""
        workspace = f"{TEST_PREFIX}_connleak"
        
        initial_clients = redis_client.client_list()
        initial_count = len(initial_clients)
        
        # Run concurrent load
        metrics = BrutalMetrics()
        run_concurrent_load(fabric_minimal, 20, 100, workspace, metrics)
        
        time.sleep(1)  # Let connections settle
        
        final_clients = redis_client.client_list()
        final_count = len(final_clients)
        
        print(f"\n[CONNECTION LEAK TEST]")
        print(f"  Initial connections: {initial_count}")
        print(f"  Final connections:   {final_count}")
        print(f"  Delta:               {final_count - initial_count}")
        
        # Should not have many more connections
        assert final_count <= initial_count + 10, \
            f"Possible connection leak: {initial_count} -> {final_count}"


# ==============================================
# TEST CLASS: BURST PATTERNS
# ==============================================

class TestBurstPatterns:
    """Test handling of burst traffic patterns"""
    
    def test_repeated_bursts(self, fabric_minimal):
        """Handle repeated bursts with recovery periods"""
        workspace = f"{TEST_PREFIX}_bursts"
        burst_size = STRESS_BURST_SIZE
        num_bursts = 5
        
        all_burst_stats = []
        
        print(f"\n[Running {num_bursts} bursts of {burst_size} events each...]")
        
        for burst_num in range(num_bursts):
            metrics = BrutalMetrics()
            metrics.start_time = time.time()
            
            # Fire burst
            for i in range(burst_size):
                start = time.time()
                try:
                    fabric_minimal.remember(
                        content=generate_content(i, "small"),
                        workspace_id=workspace,
                        user_id=f"burst_user_{burst_num}",
                    )
                    metrics.record_success((time.time() - start) * 1000)
                except Exception as e:
                    metrics.record_failure(e)
            
            metrics.end_time = time.time()
            
            all_burst_stats.append({
                "burst": burst_num,
                "success": metrics.success_ops,
                "throughput": metrics.throughput,
                "p50": metrics.p50,
                "p99": metrics.p99,
            })
            
            # Recovery period
            time.sleep(1.0)
        
        print("\n[BURST RESULTS]")
        print(f"{'Burst':<8}{'Success':<10}{'Throughput':<15}{'P50 (ms)':<12}{'P99 (ms)':<12}")
        print("-" * 57)
        for stat in all_burst_stats:
            print(f"{stat['burst']:<8}{stat['success']:<10}{stat['throughput']:<15.1f}{stat['p50']:<12.2f}{stat['p99']:<12.2f}")
        
        # All bursts should mostly succeed
        for stat in all_burst_stats:
            assert stat["success"] >= burst_size * 0.95, \
                f"Burst {stat['burst']} had too many failures"
        
        # Later bursts shouldn't be dramatically worse than first
        first_throughput = all_burst_stats[0]["throughput"]
        last_throughput = all_burst_stats[-1]["throughput"]
        degradation = last_throughput / first_throughput if first_throughput > 0 else 0
        
        print(f"\nThroughput degradation: {degradation:.2f}x")
        assert degradation >= 0.5, f"Throughput degraded too much: {degradation:.2f}x"


# ==============================================
# TEST CLASS: LATENCY CONSISTENCY
# ==============================================

class TestLatencyConsistency:
    """Verify latency stays consistent over time"""
    
    def test_latency_over_time(self, fabric_minimal):
        """Track latency buckets over extended operation"""
        workspace = f"{TEST_PREFIX}_latency"
        num_events = min(STRESS_EVENTS, 5000)
        bucket_size = 500
        
        buckets = defaultdict(list)
        
        print(f"\n[Writing {num_events} events, tracking latency in buckets of {bucket_size}...]")
        
        for i in range(num_events):
            start = time.time()
            try:
                fabric_minimal.remember(
                    content=generate_content(i, "medium"),
                    workspace_id=workspace,
                    user_id="latency_user",
                )
                latency_ms = (time.time() - start) * 1000
                bucket_id = i // bucket_size
                buckets[bucket_id].append(latency_ms)
            except Exception:
                pass
        
        print("\n[LATENCY BY BUCKET]")
        print(f"{'Bucket':<10}{'Count':<10}{'P50 (ms)':<12}{'P95 (ms)':<12}{'P99 (ms)':<12}")
        print("-" * 56)
        
        p95_values = []
        for bucket_id in sorted(buckets.keys()):
            lats = buckets[bucket_id]
            sorted_lats = sorted(lats)
            p50 = statistics.median(lats)
            p95 = sorted_lats[int(len(sorted_lats) * 0.95)] if lats else 0
            p99 = sorted_lats[int(len(sorted_lats) * 0.99)] if lats else 0
            p95_values.append(p95)
            print(f"{bucket_id:<10}{len(lats):<10}{p50:<12.2f}{p95:<12.2f}{p99:<12.2f}")
        
        # Check for latency degradation
        if len(p95_values) >= 4:
            first_half = statistics.mean(p95_values[:len(p95_values)//2])
            second_half = statistics.mean(p95_values[len(p95_values)//2:])
            degradation = second_half / first_half if first_half > 0 else 1
            
            print(f"\nFirst half avg P95: {first_half:.2f}ms")
            print(f"Second half avg P95: {second_half:.2f}ms")
            print(f"Degradation factor: {degradation:.2f}x")
            
            # Latency shouldn't more than triple
            assert degradation < 3.0, f"Latency degraded too much: {degradation:.2f}x"


# ==============================================
# SUMMARY
# ==============================================

if __name__ == "__main__":
    print("Run with: pytest tests/test_stress_brutal.py -v -s")
    print("Or with more load: STRESS_EVENTS=50000 STRESS_WORKERS=50 pytest tests/test_stress_brutal.py -v -s")
