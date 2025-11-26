"""
KRNX E2E Test Suite 4: Soak Test

Long-running test (hours/overnight) to detect:
- Memory leaks
- Connection leaks
- File handle exhaustion
- Performance degradation over time
- Resource accumulation issues

Runs sustained moderate load while monitoring:
- Memory usage (RSS, heap)
- Connection counts (Redis, Qdrant)
- File descriptors
- Response time trends
- Error rate trends

Run:
    cd /mnt/d/chillbot
    python3 chillbot/tests/test_e2e_soak.py --duration 3600  # 1 hour
    python3 chillbot/tests/test_e2e_soak.py --duration 28800 # 8 hours (overnight)

Output:
    - Console: Live progress updates
    - File: soak_test_report_{timestamp}.json
"""

import sys
import os
import time
import json
import tempfile
import threading
import psutil
import argparse
import signal
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import deque
import statistics

# Path setup
_this_file = os.path.abspath(__file__)
_tests_dir = os.path.dirname(_this_file)
_chillbot_dir = os.path.dirname(_tests_dir)
_root_dir = os.path.dirname(_chillbot_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


# ==============================================
# SOAK TEST CONFIGURATION
# ==============================================

@dataclass
class SoakConfig:
    """Soak test configuration."""
    # Connection settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    qdrant_url: str = "http://localhost:6333"
    
    # Test duration
    duration_seconds: int = 3600  # 1 hour default
    
    # Load parameters
    target_ops_per_second: int = 100
    num_worker_threads: int = 10
    
    # Monitoring
    sample_interval_seconds: int = 10  # How often to collect metrics
    report_interval_seconds: int = 60  # How often to print progress
    
    # Thresholds
    max_memory_growth_mb: float = 500.0  # Max acceptable memory growth
    max_latency_degradation_percent: float = 50.0  # Max latency increase
    max_error_rate_percent: float = 1.0
    
    # Paths
    temp_dir: str = ""
    test_workspace: str = "soak_test"
    output_dir: str = "."
    
    def __post_init__(self):
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="krnx_soak_")


# ==============================================
# RESOURCE MONITOR
# ==============================================

@dataclass
class ResourceSample:
    """Single resource measurement."""
    timestamp: float
    memory_rss_mb: float
    memory_vms_mb: float
    cpu_percent: float
    open_files: int
    thread_count: int
    redis_connections: int
    latency_avg_ms: float
    latency_p99_ms: float
    ops_per_second: float
    error_rate_percent: float


class ResourceMonitor:
    """Monitor system resources over time."""
    
    def __init__(self, config: SoakConfig):
        self.config = config
        self.process = psutil.Process()
        self._samples: List[ResourceSample] = []
        self._recent_latencies: deque = deque(maxlen=1000)
        self._recent_errors: deque = deque(maxlen=1000)
        self._ops_count = 0
        self._last_sample_time = 0.0
        self._last_ops_count = 0
        self._lock = threading.Lock()
    
    def record_operation(self, latency_ms: float, success: bool):
        """Record a single operation."""
        with self._lock:
            self._recent_latencies.append(latency_ms)
            self._recent_errors.append(0 if success else 1)
            self._ops_count += 1
    
    def _get_redis_connections(self) -> int:
        """Get current Redis connection count."""
        try:
            import redis
            r = redis.Redis(host=self.config.redis_host, port=self.config.redis_port)
            info = r.info("clients")
            r.close()
            return info.get("connected_clients", 0)
        except:
            return -1
    
    def sample(self) -> ResourceSample:
        """Collect current resource metrics."""
        now = time.time()
        
        # Process metrics
        mem_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        try:
            open_files = len(self.process.open_files())
        except:
            open_files = -1
        
        thread_count = self.process.num_threads()
        
        # Calculate latency stats
        with self._lock:
            latencies = list(self._recent_latencies)
            errors = list(self._recent_errors)
            current_ops = self._ops_count
        
        if latencies:
            latency_avg = statistics.mean(latencies)
            sorted_lat = sorted(latencies)
            p99_idx = int(len(sorted_lat) * 0.99)
            latency_p99 = sorted_lat[min(p99_idx, len(sorted_lat) - 1)]
        else:
            latency_avg = 0
            latency_p99 = 0
        
        # Calculate ops/sec
        time_delta = now - self._last_sample_time if self._last_sample_time > 0 else 1
        ops_delta = current_ops - self._last_ops_count
        ops_per_second = ops_delta / time_delta if time_delta > 0 else 0
        
        # Calculate error rate
        if errors:
            error_rate = (sum(errors) / len(errors)) * 100
        else:
            error_rate = 0
        
        self._last_sample_time = now
        self._last_ops_count = current_ops
        
        sample = ResourceSample(
            timestamp=now,
            memory_rss_mb=mem_info.rss / (1024 * 1024),
            memory_vms_mb=mem_info.vms / (1024 * 1024),
            cpu_percent=cpu_percent,
            open_files=open_files,
            thread_count=thread_count,
            redis_connections=self._get_redis_connections(),
            latency_avg_ms=round(latency_avg, 2),
            latency_p99_ms=round(latency_p99, 2),
            ops_per_second=round(ops_per_second, 2),
            error_rate_percent=round(error_rate, 2),
        )
        
        self._samples.append(sample)
        return sample
    
    def get_samples(self) -> List[ResourceSample]:
        return self._samples
    
    def get_analysis(self) -> Dict[str, Any]:
        """Analyze collected samples for anomalies."""
        if len(self._samples) < 2:
            return {"error": "Not enough samples"}
        
        first_samples = self._samples[:10]
        last_samples = self._samples[-10:]
        
        # Memory growth
        initial_memory = statistics.mean([s.memory_rss_mb for s in first_samples])
        final_memory = statistics.mean([s.memory_rss_mb for s in last_samples])
        memory_growth = final_memory - initial_memory
        
        # Latency degradation
        initial_latency = statistics.mean([s.latency_avg_ms for s in first_samples if s.latency_avg_ms > 0] or [1])
        final_latency = statistics.mean([s.latency_avg_ms for s in last_samples if s.latency_avg_ms > 0] or [1])
        latency_change_percent = ((final_latency - initial_latency) / initial_latency) * 100 if initial_latency > 0 else 0
        
        # Error rate trend
        initial_errors = statistics.mean([s.error_rate_percent for s in first_samples])
        final_errors = statistics.mean([s.error_rate_percent for s in last_samples])
        
        # Connection count trend
        initial_conns = statistics.mean([s.redis_connections for s in first_samples if s.redis_connections >= 0] or [0])
        final_conns = statistics.mean([s.redis_connections for s in last_samples if s.redis_connections >= 0] or [0])
        
        # Overall stats
        all_latencies = [s.latency_avg_ms for s in self._samples if s.latency_avg_ms > 0]
        all_ops = [s.ops_per_second for s in self._samples if s.ops_per_second > 0]
        all_memory = [s.memory_rss_mb for s in self._samples]
        
        return {
            "duration_seconds": self._samples[-1].timestamp - self._samples[0].timestamp,
            "total_samples": len(self._samples),
            
            "memory": {
                "initial_mb": round(initial_memory, 2),
                "final_mb": round(final_memory, 2),
                "growth_mb": round(memory_growth, 2),
                "peak_mb": round(max(all_memory), 2),
                "leak_detected": memory_growth > self.config.max_memory_growth_mb,
            },
            
            "latency": {
                "initial_avg_ms": round(initial_latency, 2),
                "final_avg_ms": round(final_latency, 2),
                "change_percent": round(latency_change_percent, 2),
                "overall_avg_ms": round(statistics.mean(all_latencies), 2) if all_latencies else 0,
                "degradation_detected": latency_change_percent > self.config.max_latency_degradation_percent,
            },
            
            "errors": {
                "initial_rate_percent": round(initial_errors, 2),
                "final_rate_percent": round(final_errors, 2),
                "threshold_exceeded": final_errors > self.config.max_error_rate_percent,
            },
            
            "connections": {
                "initial": round(initial_conns),
                "final": round(final_conns),
                "growth": round(final_conns - initial_conns),
            },
            
            "throughput": {
                "avg_ops_per_second": round(statistics.mean(all_ops), 2) if all_ops else 0,
                "min_ops_per_second": round(min(all_ops), 2) if all_ops else 0,
                "max_ops_per_second": round(max(all_ops), 2) if all_ops else 0,
            },
        }


# ==============================================
# SOAK TEST RUNNER
# ==============================================

class SoakTestRunner:
    """Executes long-running soak test."""
    
    def __init__(self, config: SoakConfig):
        self.config = config
        self.kernel = None
        self.embeddings = None
        self.vectors = None
        self.monitor = ResourceMonitor(config)
        self._stop_flag = threading.Event()
        self._workers: List[threading.Thread] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._report_thread: Optional[threading.Thread] = None
    
    def setup(self):
        """Initialize components."""
        print("\n=== Setting up soak test ===")
        
        from chillbot.kernel.connection_pool import configure_pool
        from chillbot.kernel.controller import KRNXController
        from chillbot.compute.vectors import VectorStore, VectorStoreBackend
        
        configure_pool(
            host=self.config.redis_host,
            port=self.config.redis_port,
            max_connections=100,
        )
        print(f"  ✓ Redis pool configured")
        
        self.kernel = KRNXController(
            data_path=os.path.join(self.config.temp_dir, "soak_kernel"),
            redis_host=self.config.redis_host,
            redis_port=self.config.redis_port,
            enable_async_worker=True,
            enable_backpressure=True,
            ltm_batch_size=50,
        )
        print(f"  ✓ Kernel initialized")
        
        self.vectors = VectorStore(backend=VectorStoreBackend.MEMORY)
        print(f"  ✓ Vectors initialized (Memory)")
        
        print("  Setup complete!\n")
    
    def teardown(self):
        """Cleanup."""
        print("\n=== Tearing down soak test ===")
        
        from chillbot.kernel.connection_pool import close_pool
        
        if self.kernel:
            self.kernel.shutdown(timeout=30.0)
        
        if self.vectors:
            self.vectors.close()
        
        close_pool()
        
        import shutil
        try:
            shutil.rmtree(self.config.temp_dir)
        except:
            pass
        
        print("  Teardown complete!\n")
    
    def _worker_loop(self, worker_id: int):
        """Worker thread main loop."""
        from chillbot.kernel.models import create_event
        import random
        import string
        
        ops_per_worker = self.config.target_ops_per_second / self.config.num_worker_threads
        sleep_time = 1.0 / ops_per_worker if ops_per_worker > 0 else 0.1
        
        seq = 0
        while not self._stop_flag.is_set():
            start = time.time()
            
            try:
                # Generate event
                payload = ''.join(random.choices(string.ascii_letters, k=500))
                event = create_event(
                    event_id=f"evt_soak_{worker_id}_{seq}_{int(time.time()*1000000)}",
                    workspace_id=self.config.test_workspace,
                    user_id=f"user_{worker_id}",
                    content={"text": payload},
                )
                
                # Write
                self.kernel.write_event_turbo(
                    workspace_id=self.config.test_workspace,
                    user_id=f"user_{worker_id}",
                    event=event,
                )
                
                latency = (time.time() - start) * 1000
                self.monitor.record_operation(latency, True)
                
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.monitor.record_operation(latency, False)
            
            seq += 1
            
            # Throttle to target rate
            elapsed = time.time() - start
            if elapsed < sleep_time:
                time.sleep(sleep_time - elapsed)
    
    def _monitor_loop(self):
        """Resource monitoring thread."""
        while not self._stop_flag.is_set():
            self.monitor.sample()
            time.sleep(self.config.sample_interval_seconds)
    
    def _report_loop(self, start_time: float):
        """Progress reporting thread."""
        while not self._stop_flag.is_set():
            elapsed = time.time() - start_time
            remaining = self.config.duration_seconds - elapsed
            
            if len(self.monitor._samples) > 0:
                latest = self.monitor._samples[-1]
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Elapsed: {int(elapsed)}s | "
                      f"Remaining: {int(remaining)}s | "
                      f"Ops/s: {latest.ops_per_second:.0f} | "
                      f"Lat: {latest.latency_avg_ms:.1f}ms | "
                      f"Mem: {latest.memory_rss_mb:.0f}MB | "
                      f"Err: {latest.error_rate_percent:.1f}%",
                      end="", flush=True)
            
            time.sleep(self.config.report_interval_seconds)
    
    def run(self) -> Dict[str, Any]:
        """Execute soak test."""
        print(f"\n{'=' * 70}")
        print(f"SOAK TEST STARTED")
        print(f"Duration: {self.config.duration_seconds}s ({self.config.duration_seconds/3600:.1f} hours)")
        print(f"Target load: {self.config.target_ops_per_second} ops/sec")
        print(f"Workers: {self.config.num_worker_threads}")
        print(f"{'=' * 70}\n")
        
        start_time = time.time()
        
        # Start monitor thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        # Start report thread
        self._report_thread = threading.Thread(target=self._report_loop, args=(start_time,), daemon=True)
        self._report_thread.start()
        
        # Start worker threads
        for i in range(self.config.num_worker_threads):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self._workers.append(worker)
        
        # Wait for duration
        try:
            while time.time() - start_time < self.config.duration_seconds:
                if self._stop_flag.is_set():
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        # Stop everything
        self._stop_flag.set()
        print("\n\nStopping workers...")
        
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        # Final sample
        self.monitor.sample()
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate final report."""
        analysis = self.monitor.get_analysis()
        
        report = {
            "test_info": {
                "start_time": datetime.fromtimestamp(self.monitor._samples[0].timestamp).isoformat() if self.monitor._samples else None,
                "end_time": datetime.fromtimestamp(self.monitor._samples[-1].timestamp).isoformat() if self.monitor._samples else None,
                "config": {
                    "duration_seconds": self.config.duration_seconds,
                    "target_ops_per_second": self.config.target_ops_per_second,
                    "num_worker_threads": self.config.num_worker_threads,
                },
            },
            "analysis": analysis,
            "verdict": {
                "passed": True,
                "failures": [],
            },
            "samples": [asdict(s) for s in self.monitor._samples[-100:]],  # Last 100 samples
        }
        
        # Determine pass/fail
        if analysis.get("memory", {}).get("leak_detected"):
            report["verdict"]["passed"] = False
            report["verdict"]["failures"].append(
                f"Memory leak detected: {analysis['memory']['growth_mb']}MB growth"
            )
        
        if analysis.get("latency", {}).get("degradation_detected"):
            report["verdict"]["passed"] = False
            report["verdict"]["failures"].append(
                f"Latency degradation: {analysis['latency']['change_percent']}% increase"
            )
        
        if analysis.get("errors", {}).get("threshold_exceeded"):
            report["verdict"]["passed"] = False
            report["verdict"]["failures"].append(
                f"Error rate exceeded: {analysis['errors']['final_rate_percent']}%"
            )
        
        return report


def print_report(report: Dict[str, Any]):
    """Print formatted report."""
    print("\n" + "=" * 70)
    print("SOAK TEST REPORT")
    print("=" * 70)
    
    info = report.get("test_info", {})
    print(f"\nTest Duration: {info.get('config', {}).get('duration_seconds', 0)}s")
    print(f"Start: {info.get('start_time', 'N/A')}")
    print(f"End: {info.get('end_time', 'N/A')}")
    
    analysis = report.get("analysis", {})
    
    print(f"\n--- Memory ---")
    mem = analysis.get("memory", {})
    print(f"  Initial:     {mem.get('initial_mb', 0):.1f} MB")
    print(f"  Final:       {mem.get('final_mb', 0):.1f} MB")
    print(f"  Growth:      {mem.get('growth_mb', 0):.1f} MB")
    print(f"  Peak:        {mem.get('peak_mb', 0):.1f} MB")
    print(f"  Leak:        {'⚠ YES' if mem.get('leak_detected') else '✓ No'}")
    
    print(f"\n--- Latency ---")
    lat = analysis.get("latency", {})
    print(f"  Initial avg: {lat.get('initial_avg_ms', 0):.2f} ms")
    print(f"  Final avg:   {lat.get('final_avg_ms', 0):.2f} ms")
    print(f"  Change:      {lat.get('change_percent', 0):.1f}%")
    print(f"  Degradation: {'⚠ YES' if lat.get('degradation_detected') else '✓ No'}")
    
    print(f"\n--- Errors ---")
    err = analysis.get("errors", {})
    print(f"  Initial rate: {err.get('initial_rate_percent', 0):.2f}%")
    print(f"  Final rate:   {err.get('final_rate_percent', 0):.2f}%")
    print(f"  Threshold:    {'⚠ EXCEEDED' if err.get('threshold_exceeded') else '✓ OK'}")
    
    print(f"\n--- Throughput ---")
    tput = analysis.get("throughput", {})
    print(f"  Average: {tput.get('avg_ops_per_second', 0):.1f} ops/sec")
    print(f"  Min:     {tput.get('min_ops_per_second', 0):.1f} ops/sec")
    print(f"  Max:     {tput.get('max_ops_per_second', 0):.1f} ops/sec")
    
    print(f"\n--- Connections ---")
    conn = analysis.get("connections", {})
    print(f"  Initial: {conn.get('initial', 0)}")
    print(f"  Final:   {conn.get('final', 0)}")
    print(f"  Growth:  {conn.get('growth', 0)}")
    
    verdict = report.get("verdict", {})
    print(f"\n{'=' * 70}")
    if verdict.get("passed"):
        print("VERDICT: ✓ PASSED")
    else:
        print("VERDICT: ✗ FAILED")
        for failure in verdict.get("failures", []):
            print(f"  - {failure}")
    print("=" * 70)


# ==============================================
# MAIN
# ==============================================

def main():
    parser = argparse.ArgumentParser(description="KRNX Soak Test")
    parser.add_argument("--duration", type=int, default=3600, help="Test duration in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--ops-per-sec", type=int, default=100, help="Target operations per second")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker threads")
    parser.add_argument("--redis-host", type=str, default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--output", type=str, default=".", help="Output directory for report")
    args = parser.parse_args()
    
    config = SoakConfig(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        duration_seconds=args.duration,
        target_ops_per_second=args.ops_per_sec,
        num_worker_threads=args.workers,
        output_dir=args.output,
    )
    
    print("=" * 70)
    print("KRNX E2E Test Suite 4: Soak Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Duration:     {config.duration_seconds}s ({config.duration_seconds/3600:.1f} hours)")
    print(f"  Target load:  {config.target_ops_per_second} ops/sec")
    print(f"  Workers:      {config.num_worker_threads}")
    print(f"  Memory limit: {config.max_memory_growth_mb} MB growth")
    print(f"  Latency limit: {config.max_latency_degradation_percent}% degradation")
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host=config.redis_host, port=config.redis_port)
        r.ping()
        r.close()
    except:
        print("\n[ERROR] Redis is not available. Soak test requires Redis.")
        return 1
    
    # Handle Ctrl+C gracefully
    runner = SoakTestRunner(config)
    
    def signal_handler(sig, frame):
        print("\n\nReceived interrupt signal, stopping test...")
        runner._stop_flag.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        runner.setup()
        report = runner.run()
    except Exception as e:
        print(f"\n[FATAL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        runner.teardown()
    
    # Print report
    print_report(report)
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(config.output_dir, f"soak_test_report_{timestamp}.json")
    
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")
    except Exception as e:
        print(f"\nFailed to save report: {e}")
    
    return 0 if report.get("verdict", {}).get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
