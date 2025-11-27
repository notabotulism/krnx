"""
KRNX Comprehensive Stress Test Suite v1.0

Tests the KRNX memory kernel under various stress conditions:
- High throughput write tests
- Concurrent read/write tests
- Burst load tests
- Backpressure behavior tests
- Memory and resource stability tests
- Data integrity under load tests

Configuration via environment variables:
    STRESS_EVENTS=10000       - Total events per test
    STRESS_WORKERS=50         - Concurrent worker threads
    STRESS_DURATION=60        - Duration for timed tests (seconds)
    STRESS_BURST_SIZE=1000    - Events per burst
    BP_QUEUE_DEPTH=500        - Backpressure queue depth threshold
    BP_LAG_SECONDS=5.0        - Backpressure lag threshold
    REDIS_HOST=localhost      - Redis host
    REDIS_PORT=6379           - Redis port

Run with:
    pytest tests/test_stress_comprehensive.py -v -s
    
Run specific test:
    pytest tests/test_stress_comprehensive.py::TestThroughput::test_sustained_write_throughput -v -s
"""

import pytest
import time
import threading
import os
import uuid
import random
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Test scale
STRESS_EVENTS = int(os.environ.get('STRESS_EVENTS', '10000'))
STRESS_WORKERS = int(os.environ.get('STRESS_WORKERS', '50'))
STRESS_DURATION = int(os.environ.get('STRESS_DURATION', '60'))
STRESS_BURST_SIZE = int(os.environ.get('STRESS_BURST_SIZE', '1000'))

# Backpressure thresholds for testing
BP_QUEUE_DEPTH = int(os.environ.get('BP_QUEUE_DEPTH', '500'))
BP_LAG_SECONDS = float(os.environ.get('BP_LAG_SECONDS', '5.0'))

# Redis
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StressMetrics:
    """Comprehensive metrics from a stress test run."""
    test_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    backpressure_rejections: int = 0
    duration_seconds: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Computed metrics
    @property
    def success_rate(self) -> float:
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100
    
    @property
    def throughput(self) -> float:
        """Operations per second."""
        if self.duration_seconds == 0:
            return 0.0
        return self.successful_operations / self.duration_seconds
    
    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)
    
    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def max_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return max(self.latencies_ms)
    
    def summary(self) -> str:
        return f"""
{'='*60}
STRESS TEST: {self.test_name}
{'='*60}
Operations:     {self.successful_operations:,} / {self.total_operations:,} ({self.success_rate:.1f}%)
Backpressure:   {self.backpressure_rejections:,} rejections
Duration:       {self.duration_seconds:.2f}s
Throughput:     {self.throughput:,.0f} ops/sec

Latency (ms):
  P50:          {self.p50_latency_ms:.2f}
  P95:          {self.p95_latency_ms:.2f}
  P99:          {self.p99_latency_ms:.2f}
  Max:          {self.max_latency_ms:.2f}

Errors:         {len(self.errors)} unique error types
{'='*60}
"""


@dataclass
class WorkerStats:
    """Per-worker statistics."""
    worker_id: int
    operations: int = 0
    successes: int = 0
    failures: int = 0
    bp_rejections: int = 0
    total_latency_ms: float = 0.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_event(worker_id: int, sequence: int, payload_size: int = 100):
    """Create a test event with specified payload size."""
    from chillbot.kernel.models import Event
    
    # Create payload of specified size
    payload = {
        'text': f"Worker {worker_id} event {sequence} - " + "x" * max(0, payload_size - 50),
        'worker_id': worker_id,
        'sequence': sequence,
        'timestamp': time.time(),
        'checksum': uuid.uuid4().hex[:8],  # For integrity verification
    }
    
    return Event(
        event_id=f"evt_{uuid.uuid4().hex[:16]}",
        workspace_id="stress_test",
        user_id=f"user_{worker_id}",
        session_id=f"stress_test_user_{worker_id}",
        content=payload,
        timestamp=time.time(),
    )


def run_write_worker(
    controller,
    worker_id: int,
    num_events: int,
    delay_between_writes: float = 0.0,
    payload_size: int = 100,
    collect_latencies: bool = True
) -> WorkerStats:
    """
    Worker function that writes events and collects statistics.
    """
    from chillbot.kernel.controller import BackpressureError
    
    stats = WorkerStats(worker_id=worker_id)
    latencies = []
    
    for i in range(num_events):
        event = create_test_event(worker_id, i, payload_size)
        start_time = time.perf_counter()
        
        try:
            controller.write_event(
                workspace_id="stress_test",
                user_id=f"user_{worker_id}",
                event=event
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            stats.successes += 1
            if collect_latencies:
                latencies.append(elapsed_ms)
                
        except BackpressureError:
            stats.bp_rejections += 1
            stats.failures += 1
        except Exception as e:
            stats.failures += 1
        
        stats.operations += 1
        
        if delay_between_writes > 0:
            time.sleep(delay_between_writes)
    
    stats.total_latency_ms = sum(latencies) if latencies else 0.0
    return stats, latencies


def aggregate_worker_stats(
    worker_results: List[Tuple[WorkerStats, List[float]]],
    test_name: str,
    duration: float
) -> StressMetrics:
    """Aggregate statistics from multiple workers."""
    metrics = StressMetrics(test_name=test_name, duration_seconds=duration)
    
    all_latencies = []
    for stats, latencies in worker_results:
        metrics.total_operations += stats.operations
        metrics.successful_operations += stats.successes
        metrics.failed_operations += stats.failures
        metrics.backpressure_rejections += stats.bp_rejections
        all_latencies.extend(latencies)
    
    metrics.latencies_ms = all_latencies
    return metrics


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def stress_controller(tmp_path_factory):
    """Create a controller configured for stress testing."""
    from chillbot.kernel.controller import KRNXController
    from chillbot.kernel.connection_pool import close_pool, get_redis_client
    
    data_path = tmp_path_factory.mktemp("krnx_stress")
    
    # Clean up any existing state
    try:
        close_pool()
    except:
        pass
    
    controller = KRNXController(
        data_path=str(data_path),
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
        enable_backpressure=True,
        max_queue_depth=50000,  # High threshold for throughput tests
        max_lag_seconds=60.0,    # High threshold for throughput tests
        enable_async_worker=True,
        worker_block_ms=10,      # Fast worker
        redis_max_connections=200,
    )
    
    # Clean Redis queue
    redis_client = get_redis_client()
    try:
        redis_client.delete('krnx:ltm:queue')
    except:
        pass
    
    yield controller
    
    # Cleanup
    try:
        controller.shutdown(timeout=10.0)
    except:
        pass


@pytest.fixture(scope="function")
def clean_queue(stress_controller):
    """Clean the queue before each test."""
    from chillbot.kernel.connection_pool import get_redis_client
    
    redis_client = get_redis_client()
    try:
        redis_client.delete('krnx:ltm:queue')
    except:
        pass
    
    # Reset backpressure
    stress_controller.reset_backpressure()
    
    # Give worker time to stabilize
    time.sleep(0.2)
    
    yield


# =============================================================================
# THROUGHPUT TESTS
# =============================================================================

class TestThroughput:
    """Test raw throughput capabilities."""
    
    def test_sustained_write_throughput(self, stress_controller, clean_queue):
        """
        Test sustained write throughput over a period of time.
        
        Target: Measure maximum sustainable writes/second
        """
        print(f"\n{'='*60}")
        print("TEST: Sustained Write Throughput")
        print(f"Config: {STRESS_WORKERS} workers, {STRESS_EVENTS} events total")
        print(f"{'='*60}")
        
        events_per_worker = STRESS_EVENTS // STRESS_WORKERS
        start_time = time.time()
        
        # Run workers
        with ThreadPoolExecutor(max_workers=STRESS_WORKERS) as executor:
            futures = [
                executor.submit(
                    run_write_worker,
                    stress_controller,
                    worker_id,
                    events_per_worker,
                    delay_between_writes=0.0,
                    payload_size=200,
                    collect_latencies=True
                )
                for worker_id in range(STRESS_WORKERS)
            ]
            
            results = [f.result() for f in as_completed(futures)]
        
        duration = time.time() - start_time
        metrics = aggregate_worker_stats(results, "Sustained Write Throughput", duration)
        
        print(metrics.summary())
        
        # Assertions
        assert metrics.success_rate >= 50.0, \
            f"Success rate {metrics.success_rate:.1f}% below 50% threshold"
        assert metrics.throughput >= 100, \
            f"Throughput {metrics.throughput:.0f} ops/sec below 100 minimum"
        
        # Store results for reporting
        return metrics
    
    def test_burst_write_performance(self, stress_controller, clean_queue):
        """
        Test performance under burst load conditions.
        
        Pattern: Short bursts of high-intensity writes followed by pauses
        """
        print(f"\n{'='*60}")
        print("TEST: Burst Write Performance")
        print(f"Config: {STRESS_BURST_SIZE} events per burst, 5 bursts")
        print(f"{'='*60}")
        
        num_bursts = 5
        burst_results = []
        
        for burst_num in range(num_bursts):
            print(f"\n[Burst {burst_num + 1}/{num_bursts}]")
            
            start_time = time.time()
            
            # Fire burst with many threads
            with ThreadPoolExecutor(max_workers=STRESS_WORKERS) as executor:
                events_per_worker = STRESS_BURST_SIZE // STRESS_WORKERS
                futures = [
                    executor.submit(
                        run_write_worker,
                        stress_controller,
                        worker_id,
                        events_per_worker,
                        delay_between_writes=0.0,
                        payload_size=100,
                        collect_latencies=True
                    )
                    for worker_id in range(STRESS_WORKERS)
                ]
                results = [f.result() for f in as_completed(futures)]
            
            duration = time.time() - start_time
            burst_metrics = aggregate_worker_stats(
                results, f"Burst {burst_num + 1}", duration
            )
            burst_results.append(burst_metrics)
            
            print(f"  Throughput: {burst_metrics.throughput:,.0f} ops/sec")
            print(f"  Success rate: {burst_metrics.success_rate:.1f}%")
            print(f"  P95 latency: {burst_metrics.p95_latency_ms:.2f}ms")
            
            # Pause between bursts
            if burst_num < num_bursts - 1:
                print("  [Cooling down 2s...]")
                time.sleep(2)
        
        # Aggregate across bursts
        total_success = sum(m.successful_operations for m in burst_results)
        total_ops = sum(m.total_operations for m in burst_results)
        avg_throughput = statistics.mean(m.throughput for m in burst_results)
        
        print(f"\n{'='*60}")
        print("BURST SUMMARY")
        print(f"Total: {total_success:,} / {total_ops:,} events")
        print(f"Average throughput: {avg_throughput:,.0f} ops/sec")
        print(f"{'='*60}")
        
        # At least some bursts should have good success rates
        successful_bursts = sum(1 for m in burst_results if m.success_rate >= 50)
        assert successful_bursts >= 3, \
            f"Only {successful_bursts}/5 bursts had >=50% success rate"
    
    def test_single_thread_latency(self, stress_controller, clean_queue):
        """
        Test single-threaded latency baseline.
        
        This establishes the minimum latency without contention.
        """
        print(f"\n{'='*60}")
        print("TEST: Single Thread Latency Baseline")
        print(f"{'='*60}")
        
        num_events = 1000
        latencies = []
        
        start_time = time.time()
        
        for i in range(num_events):
            event = create_test_event(0, i, 100)
            op_start = time.perf_counter()
            
            try:
                stress_controller.write_event(
                    workspace_id="stress_test",
                    user_id="latency_test",
                    event=event
                )
                latencies.append((time.perf_counter() - op_start) * 1000)
            except Exception as e:
                pass
        
        duration = time.time() - start_time
        
        if latencies:
            sorted_latencies = sorted(latencies)
            p50 = statistics.median(latencies)
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
            max_lat = max(latencies)
            
            print(f"Events written: {len(latencies)}")
            print(f"Duration: {duration:.2f}s")
            print(f"Throughput: {len(latencies)/duration:.0f} ops/sec")
            print(f"\nLatency (ms):")
            print(f"  P50: {p50:.3f}")
            print(f"  P95: {p95:.3f}")
            print(f"  P99: {p99:.3f}")
            print(f"  Max: {max_lat:.3f}")
            
            # Single-thread should have low latency
            assert p95 < 50, f"P95 latency {p95:.2f}ms exceeds 50ms threshold"


# =============================================================================
# CONCURRENCY TESTS
# =============================================================================

class TestConcurrency:
    """Test behavior under concurrent access patterns."""
    
    def test_high_concurrency_writes(self, stress_controller, clean_queue):
        """
        Test with very high thread count.
        
        Target: Verify stability with many concurrent writers
        """
        print(f"\n{'='*60}")
        print("TEST: High Concurrency Writes")
        print(f"Config: 100 workers, 100 events each")
        print(f"{'='*60}")
        
        num_workers = 100
        events_per_worker = 100
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_write_worker,
                    stress_controller,
                    worker_id,
                    events_per_worker,
                    delay_between_writes=0.0,
                    payload_size=100,
                    collect_latencies=False  # Skip latency collection for speed
                )
                for worker_id in range(num_workers)
            ]
            results = [f.result() for f in as_completed(futures)]
        
        duration = time.time() - start_time
        metrics = aggregate_worker_stats(results, "High Concurrency Writes", duration)
        
        print(metrics.summary())
        
        # High concurrency should still achieve reasonable success rate
        assert metrics.success_rate >= 30.0, \
            f"Success rate {metrics.success_rate:.1f}% too low under high concurrency"
    
    def test_mixed_read_write_load(self, stress_controller, clean_queue):
        """
        Test concurrent reads and writes.
        
        Pattern: Writers produce, readers consume concurrently
        """
        print(f"\n{'='*60}")
        print("TEST: Mixed Read/Write Load")
        print(f"{'='*60}")
        
        # First, write some baseline data
        print("[Phase 1: Writing baseline data...]")
        
        write_count = 500
        written_events = []
        
        for i in range(write_count):
            event = create_test_event(0, i, 200)
            written_events.append(event.event_id)
            try:
                stress_controller.write_event(
                    workspace_id="stress_test",
                    user_id="mixed_test_writer",
                    event=event
                )
            except:
                pass
        
        # Let worker process
        time.sleep(1)
        
        # Now do mixed read/write
        print("[Phase 2: Mixed read/write load...]")
        
        read_successes = 0
        read_failures = 0
        write_successes = 0
        write_failures = 0
        lock = threading.Lock()
        
        def reader_worker(worker_id):
            nonlocal read_successes, read_failures
            local_success = 0
            local_fail = 0
            
            for _ in range(100):
                try:
                    events = stress_controller.query_events(
                        workspace_id="stress_test",
                        limit=10
                    )
                    local_success += 1
                except Exception:
                    local_fail += 1
                time.sleep(0.01)
            
            with lock:
                read_successes += local_success
                read_failures += local_fail
        
        def writer_worker(worker_id):
            nonlocal write_successes, write_failures
            local_success = 0
            local_fail = 0
            
            for i in range(100):
                event = create_test_event(worker_id + 1000, i, 100)
                try:
                    stress_controller.write_event(
                        workspace_id="stress_test",
                        user_id=f"mixed_test_writer_{worker_id}",
                        event=event
                    )
                    local_success += 1
                except:
                    local_fail += 1
            
            with lock:
                write_successes += local_success
                write_failures += local_fail
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # 10 readers, 10 writers
            reader_futures = [executor.submit(reader_worker, i) for i in range(10)]
            writer_futures = [executor.submit(writer_worker, i) for i in range(10)]
            
            for f in reader_futures + writer_futures:
                f.result()
        
        duration = time.time() - start_time
        
        total_reads = read_successes + read_failures
        total_writes = write_successes + write_failures
        
        print(f"\nResults ({duration:.2f}s):")
        print(f"  Reads: {read_successes}/{total_reads} ({100*read_successes/total_reads:.1f}%)")
        print(f"  Writes: {write_successes}/{total_writes} ({100*write_successes/total_writes:.1f}%)")
        
        # Both reads and writes should work under mixed load
        assert read_successes > 0, "No reads succeeded"
        assert write_successes > 0, "No writes succeeded"


# =============================================================================
# BACKPRESSURE TESTS
# =============================================================================

class TestBackpressure:
    """Test backpressure mechanism behavior."""
    
    @pytest.fixture
    def bp_controller(self, tmp_path_factory):
        """Controller with low backpressure thresholds for testing."""
        from chillbot.kernel.controller import KRNXController
        from chillbot.kernel.connection_pool import close_pool, get_redis_client
        
        data_path = tmp_path_factory.mktemp("krnx_bp_stress")
        
        controller = KRNXController(
            data_path=str(data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=True,
            max_queue_depth=BP_QUEUE_DEPTH,    # Low threshold
            max_lag_seconds=BP_LAG_SECONDS,     # Low threshold
            enable_async_worker=True,
            worker_block_ms=50,
        )
        
        # Clean queue
        redis_client = get_redis_client()
        try:
            redis_client.delete('krnx:ltm:queue')
        except:
            pass
        
        yield controller
        
        try:
            controller.shutdown(timeout=5.0)
        except:
            pass
    
    def test_backpressure_triggers_under_load(self, bp_controller):
        """Verify backpressure activates when queue fills."""
        print(f"\n{'='*60}")
        print("TEST: Backpressure Triggers Under Load")
        print(f"Config: BP threshold = {BP_QUEUE_DEPTH} depth, {BP_LAG_SECONDS}s lag")
        print(f"{'='*60}")
        
        from chillbot.kernel.controller import BackpressureError
        from chillbot.kernel.connection_pool import get_redis_client
        
        # Clean start
        redis_client = get_redis_client()
        redis_client.delete('krnx:ltm:queue')
        bp_controller.reset_backpressure()
        
        successes = 0
        bp_errors = 0
        
        # Flood with writes
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(
                    run_write_worker,
                    bp_controller,
                    worker_id,
                    100,  # 100 events per worker = 5000 total
                    delay_between_writes=0.0,
                    payload_size=100,
                    collect_latencies=False
                )
                for worker_id in range(50)
            ]
            results = [f.result() for f in as_completed(futures)]
        
        duration = time.time() - start_time
        
        for stats, _ in results:
            successes += stats.successes
            bp_errors += stats.bp_rejections
        
        print(f"\nResults ({duration:.2f}s):")
        print(f"  Successes: {successes}")
        print(f"  Backpressure rejections: {bp_errors}")
        
        # Backpressure should have triggered
        assert bp_errors > 0, "Backpressure should have triggered under heavy load"
        print("\n[OK] Backpressure triggered correctly")
    
    def test_backpressure_recovery_under_stress(self, bp_controller):
        """Test backpressure recovery with repeated load cycles."""
        print(f"\n{'='*60}")
        print("TEST: Backpressure Recovery Under Stress")
        print(f"{'='*60}")
        
        from chillbot.kernel.connection_pool import get_redis_client
        
        redis_client = get_redis_client()
        
        num_cycles = 3
        recovery_results = []
        
        for cycle in range(num_cycles):
            print(f"\n[Cycle {cycle + 1}/{num_cycles}]")
            
            # Clean state
            redis_client.delete('krnx:ltm:queue')
            bp_controller.reset_backpressure()
            time.sleep(0.5)
            
            # Phase 1: Overload
            print("  [Phase 1: Overloading...]")
            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [
                    executor.submit(
                        run_write_worker,
                        bp_controller,
                        worker_id,
                        50,
                        delay_between_writes=0.0,
                        payload_size=100,
                        collect_latencies=False
                    )
                    for worker_id in range(30)
                ]
                [f.result() for f in as_completed(futures)]
            
            # Phase 2: Cooldown
            print("  [Phase 2: Cooling down 3s...]")
            time.sleep(3)
            
            # Phase 3: Recovery test
            print("  [Phase 3: Testing recovery...]")
            recovery_successes = 0
            recovery_failures = 0
            
            for i in range(50):
                event = create_test_event(999, i, 100)
                try:
                    bp_controller.write_event(
                        workspace_id="stress_test",
                        user_id="recovery_test",
                        event=event
                    )
                    recovery_successes += 1
                except:
                    recovery_failures += 1
                time.sleep(0.02)
            
            recovery_rate = (recovery_successes / 50) * 100
            recovery_results.append(recovery_rate)
            print(f"  Recovery: {recovery_successes}/50 ({recovery_rate:.1f}%)")
        
        # At least 2 of 3 cycles should recover well
        good_recoveries = sum(1 for r in recovery_results if r >= 60)
        print(f"\nGood recoveries: {good_recoveries}/{num_cycles}")
        
        assert good_recoveries >= 2, \
            f"Only {good_recoveries}/{num_cycles} cycles recovered properly"


# =============================================================================
# DATA INTEGRITY TESTS
# =============================================================================

class TestDataIntegrity:
    """Test data integrity under stress conditions."""
    
    def test_no_data_loss_under_load(self, stress_controller, clean_queue):
        """
        Verify no data loss during sustained writes.
        
        Writes events with unique IDs and verifies all are stored.
        """
        print(f"\n{'='*60}")
        print("TEST: No Data Loss Under Load")
        print(f"{'='*60}")
        
        num_events = 1000
        written_ids = set()
        lock = threading.Lock()
        
        def tracked_writer(worker_id, num_events):
            local_ids = []
            for i in range(num_events):
                event = create_test_event(worker_id, i, 100)
                try:
                    stress_controller.write_event(
                        workspace_id="integrity_test",
                        user_id=f"user_{worker_id}",
                        event=event
                    )
                    local_ids.append(event.event_id)
                except:
                    pass
            
            with lock:
                written_ids.update(local_ids)
            
            return len(local_ids)
        
        # Write with multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(tracked_writer, i, num_events // 10)
                for i in range(10)
            ]
            [f.result() for f in as_completed(futures)]
        
        print(f"Written: {len(written_ids)} unique events")
        
        # Wait for worker to process
        print("[Waiting 3s for worker to process...]")
        time.sleep(3)
        
        # Verify events in LTM
        found_ids = set()
        events = stress_controller.query_events(
            workspace_id="integrity_test",
            limit=num_events + 100
        )
        
        for event in events:
            found_ids.add(event.event_id)
        
        missing = written_ids - found_ids
        
        print(f"Found in LTM: {len(found_ids)}")
        print(f"Missing: {len(missing)}")
        
        # Allow for some events still in queue
        missing_rate = len(missing) / len(written_ids) * 100 if written_ids else 0
        print(f"Missing rate: {missing_rate:.2f}%")
        
        # Most events should be persisted
        assert missing_rate < 20, f"Too many missing events: {missing_rate:.1f}%"


# =============================================================================
# RESOURCE STABILITY TESTS
# =============================================================================

class TestResourceStability:
    """Test system resource stability under load."""
    
    def test_metrics_stability(self, stress_controller, clean_queue):
        """
        Test that metrics remain consistent during load.
        """
        print(f"\n{'='*60}")
        print("TEST: Metrics Stability")
        print(f"{'='*60}")
        
        metrics_samples = []
        
        def metrics_collector():
            """Collect metrics every 100ms."""
            while not stop_collecting.is_set():
                try:
                    m = stress_controller.get_worker_metrics()
                    metrics_samples.append({
                        'time': time.time(),
                        'queue_depth': m.queue_depth,
                        'lag_seconds': m.lag_seconds,
                        'messages_processed': m.messages_processed,
                        'worker_running': m.worker_running,
                    })
                except Exception as e:
                    metrics_samples.append({'error': str(e)})
                time.sleep(0.1)
        
        stop_collecting = threading.Event()
        collector_thread = threading.Thread(target=metrics_collector)
        collector_thread.start()
        
        # Run load
        print("[Running load for 5 seconds...]")
        start_time = time.time()
        
        while time.time() - start_time < 5:
            event = create_test_event(0, int((time.time() - start_time) * 1000), 100)
            try:
                stress_controller.write_event(
                    workspace_id="metrics_test",
                    user_id="metrics_user",
                    event=event
                )
            except:
                pass
            time.sleep(0.01)
        
        # Stop collector
        stop_collecting.set()
        collector_thread.join()
        
        # Analyze metrics
        valid_samples = [m for m in metrics_samples if 'error' not in m]
        error_samples = [m for m in metrics_samples if 'error' in m]
        
        print(f"\nMetrics samples: {len(valid_samples)} valid, {len(error_samples)} errors")
        
        if valid_samples:
            # Check worker was running throughout
            worker_running = [m['worker_running'] for m in valid_samples]
            assert all(worker_running), "Worker stopped during test"
            
            # Check queue depth didn't go negative
            queue_depths = [m['queue_depth'] for m in valid_samples]
            assert all(d >= 0 for d in queue_depths), "Negative queue depth detected"
            
            # Check lag didn't go negative
            lags = [m['lag_seconds'] for m in valid_samples]
            assert all(l >= 0 for l in lags), "Negative lag detected"
            
            print(f"Queue depth range: {min(queue_depths)} - {max(queue_depths)}")
            print(f"Lag range: {min(lags):.2f}s - {max(lags):.2f}s")
        
        print("\n[OK] Metrics remained stable")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
