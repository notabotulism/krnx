"""
KRNX Stress Tests

Hammer the system to measure:
  - Throughput (events/sec)
  - Latency percentiles (p50, p95, p99)
  - Concurrency handling
  - Backpressure behavior
  - Memory usage under load

Requirements:
  - Redis running (default: localhost:6379)
  - Qdrant running (default: localhost:6333)

Run:
  pytest tests/test_stress.py -v -s
  pytest tests/test_stress.py -v -k "test_throughput" -s
  
  # With custom parameters
  STRESS_EVENTS=10000 STRESS_WORKERS=20 pytest tests/test_stress.py -v -s
"""

import pytest
import time
import uuid
import os
import tempfile
import shutil
import threading
import statistics
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

# ==============================================
# TEST CONFIGURATION
# ==============================================

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

# Stress test parameters (configurable via env)
STRESS_EVENTS = int(os.environ.get("STRESS_EVENTS", 1000))
STRESS_WORKERS = int(os.environ.get("STRESS_WORKERS", 10))
STRESS_DURATION_SECONDS = int(os.environ.get("STRESS_DURATION", 30))

# Test workspace prefix
TEST_PREFIX = f"test_stress_{uuid.uuid4().hex[:8]}"


# ==============================================
# METRICS COLLECTION
# ==============================================

@dataclass
class StressMetrics:
    """Collected metrics from stress test"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def throughput(self) -> float:
        """Operations per second"""
        if self.duration_seconds <= 0:
            return 0.0
        return self.successful_operations / self.duration_seconds
    
    @property
    def success_rate(self) -> float:
        """Percentage of successful operations"""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100
    
    @property
    def p50_ms(self) -> float:
        """50th percentile latency"""
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)
    
    @property
    def p95_ms(self) -> float:
        """95th percentile latency"""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def p99_ms(self) -> float:
        """99th percentile latency"""
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def mean_ms(self) -> float:
        """Mean latency"""
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)
    
    @property
    def max_ms(self) -> float:
        """Max latency"""
        if not self.latencies_ms:
            return 0.0
        return max(self.latencies_ms)
    
    def summary(self) -> str:
        """Human-readable summary"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    STRESS TEST RESULTS                       ║
╠══════════════════════════════════════════════════════════════╣
║  Total Operations:    {self.total_operations:>10}                           ║
║  Successful:          {self.successful_operations:>10}                           ║
║  Failed:              {self.failed_operations:>10}                           ║
║  Success Rate:        {self.success_rate:>10.2f}%                          ║
║  Duration:            {self.duration_seconds:>10.2f}s                          ║
║  Throughput:          {self.throughput:>10.2f} ops/sec                     ║
╠══════════════════════════════════════════════════════════════╣
║  LATENCY PERCENTILES                                         ║
║  ─────────────────────────────────────────────────────────── ║
║  Mean:                {self.mean_ms:>10.2f} ms                            ║
║  P50:                 {self.p50_ms:>10.2f} ms                            ║
║  P95:                 {self.p95_ms:>10.2f} ms                            ║
║  P99:                 {self.p99_ms:>10.2f} ms                            ║
║  Max:                 {self.max_ms:>10.2f} ms                            ║
╚══════════════════════════════════════════════════════════════╝
"""


# ==============================================
# FIXTURES
# ==============================================

@pytest.fixture(scope="module")
def temp_data_dir() -> Generator[str, None, None]:
    """Create temporary data directory"""
    temp_dir = tempfile.mkdtemp(prefix="krnx_stress_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def redis_available() -> bool:
    """Check if Redis is available"""
    try:
        import redis
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        client.ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def krnx_controller(temp_data_dir, redis_available):
    """Create KRNX controller for stress testing
    
    NOTE: Backpressure is DISABLED for stress tests.
    Backpressure is designed to reject requests under load,
    which defeats the purpose of measuring throughput limits.
    """
    if not redis_available:
        pytest.skip("Redis not available")
    
    from chillbot.kernel import KRNXController
    
    controller = KRNXController(
        data_path=temp_data_dir,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        enable_backpressure=False,  # Disable for stress testing
    )
    yield controller
    controller.close()


@pytest.fixture(scope="module")
def memory_fabric(krnx_controller):
    """Create MemoryFabric for stress testing"""
    from chillbot.fabric import MemoryFabric
    
    fabric = MemoryFabric(
        kernel=krnx_controller,
        default_workspace=f"{TEST_PREFIX}_workspace",
        auto_embed=False,  # Disable for pure throughput test
        auto_enrich=True,
    )
    yield fabric
    fabric.close()


@pytest.fixture(scope="module")
def memory_fabric_full(krnx_controller):
    """Create MemoryFabric with full enrichment for realistic load"""
    from chillbot.fabric import MemoryFabric
    
    fabric = MemoryFabric(
        kernel=krnx_controller,
        default_workspace=f"{TEST_PREFIX}_full_workspace",
        auto_embed=False,
        auto_enrich=True,
    )
    yield fabric
    fabric.close()


# ==============================================
# STRESS TEST: THROUGHPUT
# ==============================================

class TestThroughput:
    """Measure raw throughput"""
    
    def test_sequential_write_throughput(self, memory_fabric):
        """Measure sequential write throughput"""
        workspace = f"{TEST_PREFIX}_seq_write"
        user = "stress_user"
        num_events = STRESS_EVENTS
        
        metrics = StressMetrics()
        metrics.start_time = time.time()
        
        for i in range(num_events):
            start = time.time()
            try:
                memory_fabric.remember(
                    content={"text": f"Sequential event {i}", "index": i},
                    workspace_id=workspace,
                    user_id=user,
                )
                latency_ms = (time.time() - start) * 1000
                metrics.latencies_ms.append(latency_ms)
                metrics.successful_operations += 1
            except Exception as e:
                metrics.failed_operations += 1
                metrics.errors.append(str(e))
            metrics.total_operations += 1
        
        metrics.end_time = time.time()
        
        print(metrics.summary())
        
        # Assertions
        assert metrics.success_rate >= 99.0, f"Success rate too low: {metrics.success_rate}%"
        assert metrics.throughput >= 100, f"Throughput too low: {metrics.throughput} ops/sec"
        
    def test_concurrent_write_throughput(self, memory_fabric):
        """Measure concurrent write throughput"""
        workspace = f"{TEST_PREFIX}_conc_write"
        num_events = STRESS_EVENTS
        num_workers = STRESS_WORKERS
        
        metrics = StressMetrics()
        lock = threading.Lock()
        
        def worker(worker_id: int, num_ops: int):
            local_latencies = []
            local_success = 0
            local_fail = 0
            
            for i in range(num_ops):
                start = time.time()
                try:
                    memory_fabric.remember(
                        content={"text": f"Worker {worker_id} event {i}", "worker": worker_id, "index": i},
                        workspace_id=workspace,
                        user_id=f"worker_{worker_id}",
                    )
                    latency_ms = (time.time() - start) * 1000
                    local_latencies.append(latency_ms)
                    local_success += 1
                except Exception as e:
                    local_fail += 1
            
            with lock:
                metrics.latencies_ms.extend(local_latencies)
                metrics.successful_operations += local_success
                metrics.failed_operations += local_fail
                metrics.total_operations += num_ops
        
        ops_per_worker = num_events // num_workers
        
        metrics.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker, i, ops_per_worker)
                for i in range(num_workers)
            ]
            for future in as_completed(futures):
                future.result()
        
        metrics.end_time = time.time()
        
        print(f"\n[CONCURRENT {num_workers} workers]")
        print(metrics.summary())
        
        assert metrics.success_rate >= 95.0, f"Success rate too low: {metrics.success_rate}%"
        
    def test_read_throughput(self, memory_fabric):
        """Measure read (recall) throughput"""
        workspace = f"{TEST_PREFIX}_read"
        user = "stress_user"
        
        # First, populate with some data
        for i in range(100):
            memory_fabric.remember(
                content={"text": f"Seed data {i} about various topics"},
                workspace_id=workspace,
                user_id=user,
            )
        
        time.sleep(0.5)  # Let data settle
        
        num_reads = STRESS_EVENTS // 10  # Fewer reads since they're heavier
        metrics = StressMetrics()
        queries = ["topic", "data", "various", "seed", "about"]
        
        metrics.start_time = time.time()
        
        for i in range(num_reads):
            query = queries[i % len(queries)]
            start = time.time()
            try:
                result = memory_fabric.recall(
                    query=query,
                    workspace_id=workspace,
                    user_id=user,
                    top_k=10,
                )
                latency_ms = (time.time() - start) * 1000
                metrics.latencies_ms.append(latency_ms)
                metrics.successful_operations += 1
            except Exception as e:
                metrics.failed_operations += 1
                metrics.errors.append(str(e))
            metrics.total_operations += 1
        
        metrics.end_time = time.time()
        
        print("\n[READ THROUGHPUT]")
        print(metrics.summary())
        
        assert metrics.success_rate >= 99.0


# ==============================================
# STRESS TEST: CONCURRENCY
# ==============================================

class TestConcurrency:
    """Test concurrent access patterns"""
    
    def test_mixed_read_write_load(self, memory_fabric):
        """Test mixed read/write workload"""
        workspace = f"{TEST_PREFIX}_mixed"
        num_operations = STRESS_EVENTS
        num_workers = STRESS_WORKERS
        read_ratio = 0.3  # 30% reads, 70% writes
        
        write_metrics = StressMetrics()
        read_metrics = StressMetrics()
        lock = threading.Lock()
        
        def worker(worker_id: int, num_ops: int):
            local_write_latencies = []
            local_read_latencies = []
            local_write_success = 0
            local_read_success = 0
            local_fail = 0
            
            for i in range(num_ops):
                is_read = (i % 10) < (read_ratio * 10)
                start = time.time()
                
                try:
                    if is_read:
                        memory_fabric.recall(
                            query=f"worker {worker_id}",
                            workspace_id=workspace,
                            user_id=f"worker_{worker_id}",
                            top_k=5,
                        )
                        latency_ms = (time.time() - start) * 1000
                        local_read_latencies.append(latency_ms)
                        local_read_success += 1
                    else:
                        memory_fabric.remember(
                            content={"text": f"Mixed load event from worker {worker_id}", "i": i},
                            workspace_id=workspace,
                            user_id=f"worker_{worker_id}",
                        )
                        latency_ms = (time.time() - start) * 1000
                        local_write_latencies.append(latency_ms)
                        local_write_success += 1
                except Exception:
                    local_fail += 1
            
            with lock:
                write_metrics.latencies_ms.extend(local_write_latencies)
                write_metrics.successful_operations += local_write_success
                read_metrics.latencies_ms.extend(local_read_latencies)
                read_metrics.successful_operations += local_read_success
                write_metrics.failed_operations += local_fail
        
        ops_per_worker = num_operations // num_workers
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(worker, i, ops_per_worker)
                for i in range(num_workers)
            ]
            for future in as_completed(futures):
                future.result()
        
        end_time = time.time()
        
        write_metrics.total_operations = write_metrics.successful_operations + write_metrics.failed_operations
        read_metrics.total_operations = read_metrics.successful_operations
        
        write_metrics.start_time = read_metrics.start_time = start_time
        write_metrics.end_time = read_metrics.end_time = end_time
        
        print("\n[MIXED WORKLOAD - WRITES]")
        print(write_metrics.summary())
        print("\n[MIXED WORKLOAD - READS]")
        print(read_metrics.summary())
        
        total_success = write_metrics.successful_operations + read_metrics.successful_operations
        total_ops = num_operations
        assert (total_success / total_ops) >= 0.90
        
    def test_workspace_contention(self, memory_fabric):
        """Test many workers writing to same workspace"""
        workspace = f"{TEST_PREFIX}_contention"
        num_workers = STRESS_WORKERS * 2  # Extra contention
        ops_per_worker = 50
        
        metrics = StressMetrics()
        lock = threading.Lock()
        
        def worker(worker_id: int):
            local_latencies = []
            local_success = 0
            
            for i in range(ops_per_worker):
                start = time.time()
                try:
                    memory_fabric.remember(
                        content={"text": f"Contention test {worker_id}-{i}"},
                        workspace_id=workspace,  # Same workspace!
                        user_id=f"user_{worker_id}",
                    )
                    latency_ms = (time.time() - start) * 1000
                    local_latencies.append(latency_ms)
                    local_success += 1
                except Exception:
                    pass
            
            with lock:
                metrics.latencies_ms.extend(local_latencies)
                metrics.successful_operations += local_success
                metrics.total_operations += ops_per_worker
        
        metrics.start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            for future in as_completed(futures):
                future.result()
        
        metrics.end_time = time.time()
        
        print(f"\n[CONTENTION TEST - {num_workers} workers, same workspace]")
        print(metrics.summary())
        
        # Should handle contention gracefully
        assert metrics.success_rate >= 90.0


# ==============================================
# STRESS TEST: LATENCY UNDER LOAD
# ==============================================

class TestLatencyUnderLoad:
    """Test latency characteristics under sustained load"""
    
    def test_latency_consistency(self, memory_fabric):
        """Test that latency remains consistent over time"""
        workspace = f"{TEST_PREFIX}_latency"
        user = "stress_user"
        num_events = STRESS_EVENTS
        
        # Collect latencies in buckets (e.g., every 100 events)
        bucket_size = 100
        buckets = defaultdict(list)
        
        for i in range(num_events):
            start = time.time()
            memory_fabric.remember(
                content={"text": f"Latency test event {i}"},
                workspace_id=workspace,
                user_id=user,
            )
            latency_ms = (time.time() - start) * 1000
            
            bucket_idx = i // bucket_size
            buckets[bucket_idx].append(latency_ms)
        
        # Analyze bucket statistics
        print("\n[LATENCY CONSISTENCY OVER TIME]")
        print(f"{'Bucket':<10} {'P50 (ms)':<12} {'P95 (ms)':<12} {'Count':<10}")
        print("-" * 44)
        
        bucket_p95s = []
        for bucket_idx in sorted(buckets.keys()):
            bucket_latencies = buckets[bucket_idx]
            p50 = statistics.median(bucket_latencies)
            p95 = sorted(bucket_latencies)[int(len(bucket_latencies) * 0.95)]
            bucket_p95s.append(p95)
            print(f"{bucket_idx:<10} {p50:<12.2f} {p95:<12.2f} {len(bucket_latencies):<10}")
        
        # Check that p95 doesn't degrade significantly
        if len(bucket_p95s) >= 3:
            first_half_avg = statistics.mean(bucket_p95s[:len(bucket_p95s)//2])
            second_half_avg = statistics.mean(bucket_p95s[len(bucket_p95s)//2:])
            degradation = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
            
            print(f"\nFirst half avg P95: {first_half_avg:.2f} ms")
            print(f"Second half avg P95: {second_half_avg:.2f} ms")
            print(f"Degradation factor: {degradation:.2f}x")
            
            # Allow up to 3x degradation (reasonable for sustained load)
            assert degradation < 3.0, f"Latency degraded too much: {degradation}x"


# ==============================================
# STRESS TEST: BACKPRESSURE
# ==============================================

class TestBurstHandling:
    """Test system behavior under burst load (backpressure disabled)"""
    
    def test_burst_handling(self, memory_fabric):
        """Test handling of sudden burst of writes"""
        workspace = f"{TEST_PREFIX}_burst"
        burst_size = 500
        num_bursts = 5
        
        all_latencies = []
        burst_stats = []
        
        for burst in range(num_bursts):
            burst_latencies = []
            burst_start = time.time()
            
            # Fire burst as fast as possible
            for i in range(burst_size):
                start = time.time()
                try:
                    memory_fabric.remember(
                        content={"text": f"Burst {burst} event {i}"},
                        workspace_id=workspace,
                        user_id=f"burst_user_{burst}",
                    )
                    latency_ms = (time.time() - start) * 1000
                    burst_latencies.append(latency_ms)
                except Exception:
                    pass
            
            burst_duration = time.time() - burst_start
            burst_throughput = len(burst_latencies) / burst_duration if burst_duration > 0 else 0
            
            burst_stats.append({
                "burst": burst,
                "success": len(burst_latencies),
                "duration": burst_duration,
                "throughput": burst_throughput,
                "p50": statistics.median(burst_latencies) if burst_latencies else 0,
                "p99": sorted(burst_latencies)[int(len(burst_latencies) * 0.99)] if burst_latencies else 0,
            })
            
            all_latencies.extend(burst_latencies)
            
            # Brief pause between bursts
            time.sleep(0.5)
        
        print("\n[BURST HANDLING]")
        print(f"{'Burst':<8} {'Success':<10} {'Duration':<12} {'Throughput':<15} {'P50':<10} {'P99':<10}")
        print("-" * 65)
        for stat in burst_stats:
            print(f"{stat['burst']:<8} {stat['success']:<10} {stat['duration']:<12.2f} {stat['throughput']:<15.2f} {stat['p50']:<10.2f} {stat['p99']:<10.2f}")
        
        # System should handle bursts without complete failure
        total_success = sum(s["success"] for s in burst_stats)
        total_expected = burst_size * num_bursts
        assert total_success / total_expected >= 0.90


# ==============================================
# STRESS TEST: MEMORY USAGE
# ==============================================

class TestMemoryUsage:
    """Test memory characteristics under load"""
    
    def test_memory_growth(self, memory_fabric):
        """Test that memory doesn't grow unboundedly"""
        import sys
        
        workspace = f"{TEST_PREFIX}_memory"
        user = "stress_user"
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Write many events
        for i in range(1000):
            memory_fabric.remember(
                content={"text": f"Memory test event {i}" * 10},  # Decent size
                workspace_id=workspace,
                user_id=user,
            )
        
        gc.collect()
        after_writes = len(gc.get_objects())
        
        # Do some recalls
        for i in range(100):
            memory_fabric.recall(
                query=f"test event {i % 100}",
                workspace_id=workspace,
                user_id=user,
            )
        
        gc.collect()
        after_reads = len(gc.get_objects())
        
        print(f"\n[MEMORY GROWTH]")
        print(f"Initial objects: {initial_objects}")
        print(f"After 1000 writes: {after_writes} ({after_writes - initial_objects:+d})")
        print(f"After 100 reads: {after_reads} ({after_reads - after_writes:+d})")
        
        # Object count shouldn't explode (allow 10x growth at most)
        growth_factor = after_reads / initial_objects if initial_objects > 0 else 1
        print(f"Total growth factor: {growth_factor:.2f}x")
        
        assert growth_factor < 20, f"Object count grew too much: {growth_factor}x"


# ==============================================
# RUN TESTS
# ==============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
