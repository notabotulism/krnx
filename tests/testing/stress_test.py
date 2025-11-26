#!/usr/bin/env python3
"""
KRNX Stress Test - 50 Threads × 1000 Events

Clean, comprehensive stress test for validating production readiness.

Usage:
    python stress_test.py                    # Default: 50 threads × 1000 events
    python stress_test.py --threads 10       # Custom thread count
    python stress_test.py --events 100       # Custom events per thread
    python stress_test.py --quick            # Quick test: 5 threads × 100 events

Requirements:
    - Redis running on localhost:6379
    - Qdrant running on localhost:6333 (optional, for vector tests)
"""

import argparse
import os
import shutil
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Ensure clean import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestResult:
    """Results from a single thread."""
    thread_id: int
    events_sent: int = 0
    errors: List[str] = field(default_factory=list)
    duration: float = 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.events_sent + len(self.errors)
        return self.events_sent / total if total > 0 else 0.0


@dataclass 
class StressTestResults:
    """Aggregated stress test results."""
    threads: int
    events_per_thread: int
    total_events: int
    successful_events: int
    failed_events: int
    total_duration: float
    thread_results: List[TestResult]
    
    @property
    def throughput(self) -> float:
        """Events per second."""
        return self.successful_events / self.total_duration if self.total_duration > 0 else 0
    
    @property
    def success_rate(self) -> float:
        return self.successful_events / self.total_events if self.total_events > 0 else 0
    
    @property
    def error_summary(self) -> Dict[str, int]:
        """Group errors by type."""
        errors = defaultdict(int)
        for tr in self.thread_results:
            for err in tr.errors:
                # Extract error type
                err_type = err.split(":")[0] if ":" in err else err[:50]
                errors[err_type] += 1
        return dict(errors)
    
    def print_report(self):
        """Print formatted test report."""
        print("\n" + "=" * 60)
        print("KRNX STRESS TEST RESULTS")
        print("=" * 60)
        
        print(f"\nConfiguration:")
        print(f"  Threads:          {self.threads}")
        print(f"  Events/thread:    {self.events_per_thread}")
        print(f"  Total events:     {self.total_events}")
        
        print(f"\nResults:")
        print(f"  Successful:       {self.successful_events}")
        print(f"  Failed:           {self.failed_events}")
        print(f"  Success rate:     {self.success_rate * 100:.2f}%")
        print(f"  Duration:         {self.total_duration:.2f}s")
        print(f"  Throughput:       {self.throughput:.0f} events/sec")
        
        if self.failed_events > 0:
            print(f"\nError Summary:")
            for err_type, count in sorted(self.error_summary.items(), key=lambda x: -x[1]):
                print(f"  {err_type}: {count}")
        
        # Status
        print("\n" + "-" * 60)
        if self.failed_events == 0:
            print("✅ PASSED - Zero errors!")
        elif self.success_rate >= 0.99:
            print(f"⚠️  MARGINAL - {self.failed_events} errors ({100 - self.success_rate*100:.2f}%)")
        else:
            print(f"❌ FAILED - {self.failed_events} errors ({100 - self.success_rate*100:.1f}%)")
        print("=" * 60 + "\n")


def clear_test_data(data_path: str):
    """Remove test data directory."""
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        print(f"[CLEANUP] Removed {data_path}")


def clear_redis_streams():
    """Clear Redis streams used by KRNX."""
    try:
        from chillbot.kernel.connection_pool import configure_pool, get_redis_client
        
        # Configure pool if not already done
        try:
            configure_pool(host="localhost", port=6379)
        except Exception:
            pass  # Already configured
        
        redis = get_redis_client()
        
        # Clear streams
        streams = [
            "krnx:ltm:queue",
            "krnx:compute:jobs",
        ]
        
        for stream in streams:
            try:
                redis.delete(stream)
                print(f"[CLEANUP] Cleared {stream}")
            except Exception as e:
                print(f"[CLEANUP] Could not clear {stream}: {e}")
        
        # Clear any workspace keys
        keys = redis.keys("krnx:*stress*")
        if keys:
            redis.delete(*keys)
            print(f"[CLEANUP] Cleared {len(keys)} workspace keys")
            
    except Exception as e:
        print(f"[WARN] Redis cleanup failed: {e}")


def run_stress_test(
    threads: int = 50,
    events_per_thread: int = 1000,
    data_path: str = "./stress-test-data",
    verbose: bool = False
) -> StressTestResults:
    """
    Run the stress test.
    
    Args:
        threads: Number of concurrent threads
        events_per_thread: Events each thread will write
        data_path: Path for test data
        verbose: Print progress updates
    
    Returns:
        StressTestResults with full metrics
    """
    total_events = threads * events_per_thread
    print(f"\n🚀 Starting stress test: {threads} threads × {events_per_thread} events = {total_events} total\n")
    
    # Cleanup
    clear_test_data(data_path)
    clear_redis_streams()
    
    # Import after cleanup to ensure fresh state
    from chillbot import Memory
    
    # Initialize memory
    print("[INIT] Creating Memory instance...")
    memory = Memory(
        agent_id="stress-test",
        data_path=data_path,
        redis_host="localhost",
        redis_port=6379,
    )
    print("[INIT] Memory ready\n")
    
    # Thread results storage
    results: List[TestResult] = []
    results_lock = threading.Lock()
    
    # Progress tracking
    progress = {"completed": 0, "errors": 0}
    progress_lock = threading.Lock()
    
    def hammer(thread_id: int):
        """Worker thread that writes events."""
        result = TestResult(thread_id=thread_id)
        start = time.time()
        
        for i in range(events_per_thread):
            try:
                memory.remember(
                    content=f"Thread {thread_id} event {i}: stress test payload",
                    metadata={"thread_id": thread_id, "seq": i}
                )
                result.events_sent += 1
                
            except Exception as e:
                result.errors.append(f"{type(e).__name__}: {str(e)[:100]}")
        
        result.duration = time.time() - start
        
        with results_lock:
            results.append(result)
        
        with progress_lock:
            progress["completed"] += result.events_sent
            progress["errors"] += len(result.errors)
            
            if verbose or (thread_id % 10 == 0):
                pct = (progress["completed"] + progress["errors"]) / total_events * 100
                print(f"[PROGRESS] {pct:.1f}% - {progress['completed']} ok, {progress['errors']} errors")
    
    # Run threads
    print(f"[RUN] Spawning {threads} threads...")
    start_time = time.time()
    
    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=hammer, args=(i,), name=f"hammer-{i}")
        thread_list.append(t)
    
    # Start all threads
    for t in thread_list:
        t.start()
    
    # Wait for completion
    for t in thread_list:
        t.join()
    
    total_duration = time.time() - start_time
    
    # Aggregate results
    successful = sum(r.events_sent for r in results)
    failed = sum(len(r.errors) for r in results)
    
    # Close memory (triggers drain)
    print("\n[DRAIN] Closing memory, waiting for worker drain...")
    drain_start = time.time()
    memory.close()
    drain_duration = time.time() - drain_start
    print(f"[DRAIN] Completed in {drain_duration:.1f}s")
    
    return StressTestResults(
        threads=threads,
        events_per_thread=events_per_thread,
        total_events=total_events,
        successful_events=successful,
        failed_events=failed,
        total_duration=total_duration,
        thread_results=results,
    )


def main():
    parser = argparse.ArgumentParser(description="KRNX Stress Test")
    parser.add_argument("--threads", type=int, default=50, help="Number of threads")
    parser.add_argument("--events", type=int, default=1000, help="Events per thread")
    parser.add_argument("--data-path", type=str, default="./stress-test-data", help="Test data directory")
    parser.add_argument("--quick", action="store_true", help="Quick test (5 threads × 100 events)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--keep-data", action="store_true", help="Don't cleanup after test")
    
    args = parser.parse_args()
    
    if args.quick:
        args.threads = 5
        args.events = 100
    
    try:
        results = run_stress_test(
            threads=args.threads,
            events_per_thread=args.events,
            data_path=args.data_path,
            verbose=args.verbose,
        )
        
        results.print_report()
        
        # Cleanup
        if not args.keep_data:
            clear_test_data(args.data_path)
        
        # Exit code based on results
        if results.failed_events == 0:
            sys.exit(0)
        elif results.success_rate >= 0.99:
            sys.exit(1)  # Marginal
        else:
            sys.exit(2)  # Failed
            
    except KeyboardInterrupt:
        print("\n\n[ABORT] Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] Test crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
