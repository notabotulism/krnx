"""
KRNX E2E Test Suite 3: Stress Tests

Tests the system under EXTREME LOAD:
- High concurrency (50-100+ threads)
- Rapid writes/reads
- Large payloads
- Connection pool exhaustion
- Backpressure handling

Measures:
- Throughput (events/sec)
- Latency percentiles (p50, p95, p99)
- Error rates
- Resource usage

Run:
    cd /mnt/d/chillbot
    python3 chillbot/tests/test_e2e_stress.py

    # With custom parameters:
    python3 chillbot/tests/test_e2e_stress.py --threads 50 --events 1000 --duration 60
"""

import sys
import os
import time
import json
import tempfile
import threading
import statistics
import argparse
import random
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path setup
_this_file = os.path.abspath(__file__)
_tests_dir = os.path.dirname(_this_file)
_chillbot_dir = os.path.dirname(_tests_dir)
_root_dir = os.path.dirname(_chillbot_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


# ==============================================
# STRESS TEST CONFIGURATION
# ==============================================

@dataclass
class StressConfig:
    """Stress test configuration."""
    # Connection settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    qdrant_url: str = "http://localhost:6333"
    
    # Test parameters
    num_threads: int = 50
    events_per_thread: int = 100
    test_duration_seconds: int = 0  # 0 = run until events complete
    
    # Workload mix
    write_percentage: float = 80.0  # % of operations that are writes
    read_percentage: float = 20.0
    
    # Payload settings
    min_payload_size: int = 100
    max_payload_size: int = 5000
    
    # Thresholds
    max_error_rate: float = 1.0  # Max acceptable error rate %
    max_p99_latency_ms: float = 100.0  # Max acceptable p99 latency
    min_throughput: float = 500.0  # Min acceptable events/sec
    
    # Paths
    temp_dir: str = ""
    test_workspace: str = "stress_test"
    
    def __post_init__(self):
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="krnx_stress_")


# ==============================================
# METRICS COLLECTOR
# ==============================================

@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_type: str
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Thread-safe metrics collection."""
    
    def __init__(self):
        self._metrics: List[OperationMetrics] = []
        self._lock = threading.Lock()
        self._start_time = 0.0
        self._end_time = 0.0
    
    def start(self):
        self._start_time = time.time()
    
    def stop(self):
        self._end_time = time.time()
    
    def record(self, metric: OperationMetrics):
        with self._lock:
            self._metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        with self._lock:
            metrics = list(self._metrics)
        
        if not metrics:
            return {"error": "No metrics collected"}
        
        # Separate by operation type
        by_type = defaultdict(list)
        for m in metrics:
            by_type[m.operation_type].append(m)
        
        duration = self._end_time - self._start_time
        total_ops = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        failed = total_ops - successful
        
        summary = {
            "duration_seconds": round(duration, 2),
            "total_operations": total_ops,
            "successful": successful,
            "failed": failed,
            "error_rate_percent": round(failed / total_ops * 100, 2) if total_ops > 0 else 0,
            "throughput_ops_sec": round(total_ops / duration, 2) if duration > 0 else 0,
            "by_operation": {},
        }
        
        # Per-operation stats
        for op_type, op_metrics in by_type.items():
            latencies = [m.latency_ms for m in op_metrics if m.success]
            
            if latencies:
                sorted_latencies = sorted(latencies)
                p50_idx = int(len(sorted_latencies) * 0.50)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p99_idx = int(len(sorted_latencies) * 0.99)
                
                summary["by_operation"][op_type] = {
                    "count": len(op_metrics),
                    "success": sum(1 for m in op_metrics if m.success),
                    "failed": sum(1 for m in op_metrics if not m.success),
                    "latency_ms": {
                        "min": round(min(latencies), 2),
                        "max": round(max(latencies), 2),
                        "avg": round(statistics.mean(latencies), 2),
                        "p50": round(sorted_latencies[p50_idx], 2),
                        "p95": round(sorted_latencies[p95_idx], 2),
                        "p99": round(sorted_latencies[min(p99_idx, len(sorted_latencies)-1)], 2),
                    },
                }
            else:
                summary["by_operation"][op_type] = {
                    "count": len(op_metrics),
                    "success": 0,
                    "failed": len(op_metrics),
                    "latency_ms": None,
                }
        
        # Collect errors
        errors = [m.error for m in metrics if m.error]
        error_counts = defaultdict(int)
        for e in errors:
            error_counts[e[:100]] += 1  # Truncate long errors
        
        summary["top_errors"] = dict(sorted(error_counts.items(), key=lambda x: -x[1])[:5])
        
        return summary
    
    def get_errors(self) -> List[str]:
        """Get all error messages."""
        with self._lock:
            return [m.error for m in self._metrics if m.error]


# ==============================================
# STRESS TEST RUNNER
# ==============================================

class StressTestRunner:
    """Executes stress tests."""
    
    def __init__(self, config: StressConfig):
        self.config = config
        self.kernel = None
        self.embeddings = None
        self.vectors = None
        self.metrics = MetricsCollector()
        self._stop_flag = threading.Event()
        self._event_ids: List[str] = []
        self._event_ids_lock = threading.Lock()
    
    def setup(self):
        """Initialize components."""
        print("\n=== Setting up stress test ===")
        
        from chillbot.kernel.connection_pool import configure_pool
        from chillbot.kernel.controller import KRNXController
        from chillbot.compute.embeddings import EmbeddingEngine
        from chillbot.compute.vectors import VectorStore, VectorStoreBackend
        
        # Configure Redis with high connection limit
        configure_pool(
            host=self.config.redis_host,
            port=self.config.redis_port,
            max_connections=300,  # High limit for stress test
        )
        print(f"  ✓ Redis pool configured (max_connections=300)")
        
        # Initialize kernel
        self.kernel = KRNXController(
            data_path=os.path.join(self.config.temp_dir, "stress_kernel"),
            redis_host=self.config.redis_host,
            redis_port=self.config.redis_port,
            enable_async_worker=True,
            enable_backpressure=True,
            max_queue_depth=100000,
            max_lag_seconds=60.0,
            ltm_batch_size=200,
            ltm_batch_interval=0.05,
        )
        print(f"  ✓ Kernel initialized (backpressure enabled)")
        
        # Initialize embeddings
        try:
            self.embeddings = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
            print(f"  ✓ Embeddings initialized")
        except:
            self.embeddings = None
            print(f"  ⊘ Embeddings skipped")
        
        # Initialize vectors
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=self.config.qdrant_url)
            client.get_collections()
            self.vectors = VectorStore(url=self.config.qdrant_url)
            if self.embeddings:
                self.vectors.ensure_collection(self.config.test_workspace, self.embeddings.dimension)
            print(f"  ✓ Vectors initialized (Qdrant)")
        except:
            self.vectors = VectorStore(backend=VectorStoreBackend.MEMORY)
            if self.embeddings:
                self.vectors.ensure_collection(self.config.test_workspace, self.embeddings.dimension)
            print(f"  ✓ Vectors initialized (Memory)")
        
        print("  Setup complete!\n")
    
    def teardown(self):
        """Cleanup."""
        print("\n=== Tearing down stress test ===")
        
        from chillbot.kernel.connection_pool import close_pool
        
        if self.kernel:
            self.kernel.shutdown(timeout=30.0)
        
        if self.vectors:
            try:
                self.vectors.delete_collection(self.config.test_workspace)
            except:
                pass
            self.vectors.close()
        
        close_pool()
        
        import shutil
        try:
            shutil.rmtree(self.config.temp_dir)
        except:
            pass
        
        print("  Teardown complete!\n")
    
    def generate_payload(self) -> str:
        """Generate random payload of variable size."""
        size = random.randint(self.config.min_payload_size, self.config.max_payload_size)
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=size))
    
    def worker_write(self, thread_id: int, num_events: int):
        """Worker function for write operations."""
        from chillbot.kernel.models import create_event
        
        for i in range(num_events):
            if self._stop_flag.is_set():
                break
            
            event_id = f"evt_stress_{thread_id}_{i}_{int(time.time()*1000000)}"
            payload = self.generate_payload()
            
            start = time.time()
            try:
                event = create_event(
                    event_id=event_id,
                    workspace_id=self.config.test_workspace,
                    user_id=f"user_{thread_id % 10}",
                    content={"text": payload, "thread_id": thread_id, "seq": i},
                )
                
                self.kernel.write_event_turbo(
                    workspace_id=self.config.test_workspace,
                    user_id=f"user_{thread_id % 10}",
                    event=event,
                )
                
                latency = (time.time() - start) * 1000
                self.metrics.record(OperationMetrics(
                    operation_type="write",
                    latency_ms=latency,
                    success=True,
                ))
                
                with self._event_ids_lock:
                    self._event_ids.append(event_id)
                
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.metrics.record(OperationMetrics(
                    operation_type="write",
                    latency_ms=latency,
                    success=False,
                    error=str(e)[:200],
                ))
    
    def worker_read(self, thread_id: int, num_reads: int):
        """Worker function for read operations."""
        for i in range(num_reads):
            if self._stop_flag.is_set():
                break
            
            start = time.time()
            try:
                # Get a random event ID if available
                event_id = None
                with self._event_ids_lock:
                    if self._event_ids:
                        event_id = random.choice(self._event_ids)
                
                if event_id:
                    event = self.kernel.stm.get_event(event_id)
                    success = event is not None
                else:
                    # Query recent events
                    events = self.kernel.stm.get_events(
                        self.config.test_workspace,
                        f"user_{thread_id % 10}",
                        limit=10,
                    )
                    success = True
                
                latency = (time.time() - start) * 1000
                self.metrics.record(OperationMetrics(
                    operation_type="read",
                    latency_ms=latency,
                    success=success,
                ))
                
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.metrics.record(OperationMetrics(
                    operation_type="read",
                    latency_ms=latency,
                    success=False,
                    error=str(e)[:200],
                ))
    
    def worker_mixed(self, thread_id: int, num_ops: int):
        """Worker function for mixed workload."""
        from chillbot.kernel.models import create_event
        
        for i in range(num_ops):
            if self._stop_flag.is_set():
                break
            
            # Decide operation type
            if random.random() * 100 < self.config.write_percentage:
                self.worker_write(thread_id, 1)
            else:
                self.worker_read(thread_id, 1)
    
    def run_concurrent_writes(self) -> Dict[str, Any]:
        """Test: High-concurrency writes."""
        print("\n=== Stress Test: Concurrent Writes ===")
        print(f"    Threads: {self.config.num_threads}")
        print(f"    Events per thread: {self.config.events_per_thread}")
        print(f"    Total events: {self.config.num_threads * self.config.events_per_thread}")
        
        self.metrics = MetricsCollector()
        self._event_ids = []
        self._stop_flag.clear()
        
        self.metrics.start()
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [
                executor.submit(self.worker_write, i, self.config.events_per_thread)
                for i in range(self.config.num_threads)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"    Thread error: {e}")
        
        self.metrics.stop()
        
        return self.metrics.get_summary()
    
    def run_concurrent_reads(self) -> Dict[str, Any]:
        """Test: High-concurrency reads."""
        print("\n=== Stress Test: Concurrent Reads ===")
        print(f"    Threads: {self.config.num_threads}")
        print(f"    Reads per thread: {self.config.events_per_thread}")
        
        self.metrics = MetricsCollector()
        self._stop_flag.clear()
        
        self.metrics.start()
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [
                executor.submit(self.worker_read, i, self.config.events_per_thread)
                for i in range(self.config.num_threads)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"    Thread error: {e}")
        
        self.metrics.stop()
        
        return self.metrics.get_summary()
    
    def run_mixed_workload(self) -> Dict[str, Any]:
        """Test: Mixed read/write workload."""
        print("\n=== Stress Test: Mixed Workload ===")
        print(f"    Threads: {self.config.num_threads}")
        print(f"    Ops per thread: {self.config.events_per_thread}")
        print(f"    Write/Read ratio: {self.config.write_percentage}%/{self.config.read_percentage}%")
        
        self.metrics = MetricsCollector()
        self._stop_flag.clear()
        
        self.metrics.start()
        
        with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
            futures = [
                executor.submit(self.worker_mixed, i, self.config.events_per_thread)
                for i in range(self.config.num_threads)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"    Thread error: {e}")
        
        self.metrics.stop()
        
        return self.metrics.get_summary()
    
    def run_burst_test(self) -> Dict[str, Any]:
        """Test: Burst traffic pattern."""
        print("\n=== Stress Test: Burst Traffic ===")
        
        num_bursts = 5
        threads_per_burst = self.config.num_threads
        events_per_burst = self.config.events_per_thread // 2
        
        print(f"    Bursts: {num_bursts}")
        print(f"    Threads per burst: {threads_per_burst}")
        print(f"    Events per burst: {threads_per_burst * events_per_burst}")
        
        self.metrics = MetricsCollector()
        self._event_ids = []
        self._stop_flag.clear()
        
        self.metrics.start()
        
        for burst in range(num_bursts):
            print(f"    Burst {burst + 1}/{num_bursts}...")
            
            with ThreadPoolExecutor(max_workers=threads_per_burst) as executor:
                futures = [
                    executor.submit(self.worker_write, i + burst * threads_per_burst, events_per_burst)
                    for i in range(threads_per_burst)
                ]
                
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        pass
            
            # Brief pause between bursts
            time.sleep(0.5)
        
        self.metrics.stop()
        
        return self.metrics.get_summary()
    
    def run_connection_exhaustion_test(self) -> Dict[str, Any]:
        """Test: Try to exhaust connection pool."""
        print("\n=== Stress Test: Connection Exhaustion ===")
        
        # Use more threads than connection pool size
        num_threads = 400
        events_per_thread = 10
        
        print(f"    Threads: {num_threads} (> pool size of 300)")
        print(f"    Events per thread: {events_per_thread}")
        
        self.metrics = MetricsCollector()
        self._event_ids = []
        self._stop_flag.clear()
        
        self.metrics.start()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.worker_write, i, events_per_thread)
                for i in range(num_threads)
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    pass
        
        self.metrics.stop()
        
        return self.metrics.get_summary()


def print_summary(name: str, summary: Dict[str, Any], config: StressConfig):
    """Print formatted test summary."""
    print(f"\n    Results for {name}:")
    print(f"    {'─' * 50}")
    print(f"    Duration:    {summary['duration_seconds']}s")
    print(f"    Operations:  {summary['total_operations']}")
    print(f"    Throughput:  {summary['throughput_ops_sec']} ops/sec")
    print(f"    Error Rate:  {summary['error_rate_percent']}%")
    
    if "by_operation" in summary:
        for op_type, stats in summary["by_operation"].items():
            print(f"\n    {op_type.upper()}:")
            print(f"      Count:   {stats['count']} ({stats['success']} success, {stats['failed']} failed)")
            if stats.get("latency_ms"):
                lat = stats["latency_ms"]
                print(f"      Latency: p50={lat['p50']}ms, p95={lat['p95']}ms, p99={lat['p99']}ms")
    
    if summary.get("top_errors"):
        print(f"\n    Top Errors:")
        for error, count in list(summary["top_errors"].items())[:3]:
            print(f"      [{count}x] {error[:60]}...")
    
    # Pass/Fail assessment
    passed = True
    failures = []
    
    if summary["error_rate_percent"] > config.max_error_rate:
        passed = False
        failures.append(f"Error rate {summary['error_rate_percent']}% > {config.max_error_rate}%")
    
    if summary["throughput_ops_sec"] < config.min_throughput:
        passed = False
        failures.append(f"Throughput {summary['throughput_ops_sec']} < {config.min_throughput} ops/sec")
    
    for op_type, stats in summary.get("by_operation", {}).items():
        if stats.get("latency_ms") and stats["latency_ms"].get("p99"):
            if stats["latency_ms"]["p99"] > config.max_p99_latency_ms:
                passed = False
                failures.append(f"{op_type} p99 latency {stats['latency_ms']['p99']}ms > {config.max_p99_latency_ms}ms")
    
    print(f"\n    {'✓ PASS' if passed else '✗ FAIL'}")
    if failures:
        for f in failures:
            print(f"      - {f}")
    
    return passed


# ==============================================
# MAIN
# ==============================================

def main():
    parser = argparse.ArgumentParser(description="KRNX Stress Tests")
    parser.add_argument("--threads", type=int, default=50, help="Number of concurrent threads")
    parser.add_argument("--events", type=int, default=100, help="Events per thread")
    parser.add_argument("--duration", type=int, default=0, help="Test duration (0=unlimited)")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333")
    args = parser.parse_args()
    
    config = StressConfig(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        qdrant_url=args.qdrant_url,
        num_threads=args.threads,
        events_per_thread=args.events,
        test_duration_seconds=args.duration,
    )
    
    print("=" * 70)
    print("KRNX E2E Test Suite 3: Stress Tests")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Threads:           {config.num_threads}")
    print(f"  Events per thread: {config.events_per_thread}")
    print(f"  Payload size:      {config.min_payload_size}-{config.max_payload_size} bytes")
    print(f"  Error threshold:   {config.max_error_rate}%")
    print(f"  P99 threshold:     {config.max_p99_latency_ms}ms")
    print(f"  Min throughput:    {config.min_throughput} ops/sec")
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host=config.redis_host, port=config.redis_port)
        r.ping()
        r.close()
    except:
        print("\n[ERROR] Redis is not available. Stress tests require Redis.")
        return 1
    
    runner = StressTestRunner(config)
    results = {}
    all_passed = True
    
    try:
        runner.setup()
        
        # Run stress tests
        tests = [
            ("Concurrent Writes", runner.run_concurrent_writes),
            ("Concurrent Reads", runner.run_concurrent_reads),
            ("Mixed Workload", runner.run_mixed_workload),
            ("Burst Traffic", runner.run_burst_test),
            ("Connection Exhaustion", runner.run_connection_exhaustion_test),
        ]
        
        for name, test_fn in tests:
            summary = test_fn()
            results[name] = summary
            passed = print_summary(name, summary, config)
            if not passed:
                all_passed = False
        
    except Exception as e:
        print(f"\n[FATAL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        runner.teardown()
    
    # Final summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)
    
    for name, summary in results.items():
        status = "✓" if summary["error_rate_percent"] <= config.max_error_rate else "✗"
        print(f"  {status} {name}: {summary['throughput_ops_sec']} ops/sec, {summary['error_rate_percent']}% errors")
    
    print("\n" + "=" * 70)
    print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
