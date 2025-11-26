#!/usr/bin/env python3
"""
TURBO Benchmark - Verify 3x throughput improvement

Before: 443 events/sec (3 Redis round-trips)
After:  1500+ events/sec (1 Redis round-trip)
"""

import os
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_all():
    data_path = os.environ.get("DATABASE_PATH", "./turbo-test")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    try:
        from chillbot.kernel.connection_pool import configure_pool, get_redis_client
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        try:
            configure_pool(host=redis_host, port=redis_port)
        except:
            pass
        redis = get_redis_client()
        for key in redis.keys("krnx:*"):
            redis.delete(key)
    except:
        pass

def main():
    print("="*60)
    print("TURBO BENCHMARK - Single-Pipeline Writes")
    print("="*60)
    
    clear_all()
    
    # Import optimized components
    from chillbot import Memory
    
    # Read from environment (for Docker) or use defaults (for local)
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    data_path = os.environ.get("DATABASE_PATH", "./turbo-test")
    
    print(f"Redis: {redis_host}:{redis_port}")
    print(f"Qdrant: {qdrant_url}")
    print(f"Data: {data_path}")
    
    memory = Memory(
        agent_id="turbo-test",
        data_path=data_path,
        redis_host=redis_host,
        redis_port=redis_port,
        qdrant_url=qdrant_url,
    )
    
    # Check if turbo is enabled
    if hasattr(memory._kernel, 'write_event_turbo'):
        print("✓ TURBO mode enabled (single-pipeline writes)")
    else:
        print("⚠️  TURBO mode NOT enabled - using legacy path")
        print("   Copy controller_turbo.py and orchestrator_turbo.py to enable")
    
    NUM_THREADS = 50
    EVENTS_PER_THREAD = 200
    TOTAL_EVENTS = NUM_THREADS * EVENTS_PER_THREAD
    
    print(f"\nRunning: {NUM_THREADS} threads × {EVENTS_PER_THREAD} events = {TOTAL_EVENTS} total\n")
    
    all_times = []
    times_lock = threading.Lock()
    
    # Use simple counter with minimal locking
    class AtomicCounter:
        def __init__(self):
            self._value = 0
            self._lock = threading.Lock()
        def increment(self):
            with self._lock:
                self._value += 1
                return self._value
        @property
        def value(self):
            return self._value
    
    progress_counter = AtomicCounter()
    error_counter = AtomicCounter()
    errors = []
    errors_lock = threading.Lock()
    
    def worker(thread_id):
        local_times = []  # Collect locally, batch update at end
        local_errors = []
        
        for i in range(EVENTS_PER_THREAD):
            start = time.time()
            try:
                memory.remember(
                    content=f"Thread {thread_id} event {i}",
                    metadata={"thread": thread_id, "event": i}
                )
                elapsed = (time.time() - start) * 1000
                local_times.append(elapsed)
                progress_counter.increment()
                    
            except Exception as e:
                local_errors.append(str(e))
                error_counter.increment()
        
        # Batch update times at end (less lock contention)
        with times_lock:
            all_times.extend(local_times)
        
        if local_errors:
            with errors_lock:
                errors.extend(local_errors)
    
    # Progress reporter
    stop_reporter = threading.Event()
    
    def reporter():
        last = 0
        while not stop_reporter.is_set():
            time.sleep(1)
            current = progress_counter.value
            rate = current - last
            elapsed = time.time() - start_time
            overall = current / elapsed if elapsed > 0 else 0
            err_count = error_counter.value
            print(f"  [{elapsed:>5.1f}s] {current:>5}/{TOTAL_EVENTS} | {rate:>4}/s recent | {overall:>6.0f}/s overall | {err_count} errors")
            last = current
    
    reporter_thread = threading.Thread(target=reporter, daemon=True)
    reporter_thread.start()
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(worker, i) for i in range(NUM_THREADS)]
        for f in as_completed(futures):
            f.result()
    
    stop_reporter.set()
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print("="*60)
    
    throughput = len(all_times) / total_time
    
    print(f"  Events:     {len(all_times)}")
    print(f"  Time:       {total_time:.2f}s")
    print(f"  Throughput: {throughput:.0f} events/sec")
    print(f"  Errors:     {len(errors)}")
    
    if all_times:
        sorted_times = sorted(all_times)
        avg = sum(all_times) / len(all_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        max_t = max(all_times)
        print(f"\n  Latency:")
        print(f"    Avg: {avg:.2f}ms")
        print(f"    P50: {p50:.2f}ms")
        print(f"    P95: {p95:.2f}ms")
        print(f"    P99: {p99:.2f}ms")
        print(f"    Max: {max_t:.1f}ms")
    
    print(f"\n{'='*60}")
    
    # Compare to baseline
    BASELINE = 443  # events/sec from previous test
    
    if throughput > BASELINE * 2:
        improvement = throughput / BASELINE
        print(f"✓ TURBO WORKING! {improvement:.1f}x faster than baseline ({BASELINE}/s)")
    elif throughput > BASELINE * 1.3:
        print(f"↑ Modest improvement: {throughput:.0f}/s vs {BASELINE}/s baseline")
    else:
        print(f"✗ No improvement: {throughput:.0f}/s vs {BASELINE}/s baseline")
        print("  Check that controller_turbo.py is installed")
    
    print("="*60)
    
    memory.close()
    data_path = os.environ.get("DATABASE_PATH", "./turbo-test")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)

if __name__ == "__main__":
    main()
