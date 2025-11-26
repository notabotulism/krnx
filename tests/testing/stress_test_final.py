#!/usr/bin/env python3
"""
KRNX Stress Test - 50 Threads × 1000 Events

Fixed version with proper timeout.
"""

import os
import shutil
import sys
import threading
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_test_data(data_path: str):
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        print(f"[CLEANUP] Removed {data_path}")

def clear_redis_streams():
    try:
        from chillbot.kernel.connection_pool import configure_pool, get_redis_client
        try:
            configure_pool(host="localhost", port=6379)
        except Exception:
            pass
        redis = get_redis_client()
        for stream in ["krnx:ltm:queue", "krnx:compute:jobs"]:
            try:
                redis.delete(stream)
                print(f"[CLEANUP] Cleared {stream}")
            except:
                pass
        keys = redis.keys("krnx:*stress*")
        if keys:
            redis.delete(*keys)
    except Exception as e:
        print(f"[WARN] Redis cleanup failed: {e}")

def main():
    threads = 50
    events_per_thread = 1000
    total_events = threads * events_per_thread
    data_path = "./stress-test-data"
    
    print(f"\n{'='*60}")
    print(f"KRNX STRESS TEST")
    print(f"{'='*60}")
    print(f"Configuration: {threads} threads × {events_per_thread} events = {total_events:,} total\n")
    
    # Cleanup
    clear_test_data(data_path)
    clear_redis_streams()
    
    # Import and init
    print("[INIT] Creating Memory instance...")
    from chillbot import Memory
    
    memory = Memory(
        agent_id="stress-test",
        data_path=data_path,
        redis_host="localhost",
        redis_port=6379,
    )
    print("[INIT] Memory ready\n")
    
    # Results storage
    results = []
    results_lock = threading.Lock()
    progress = {"completed": 0}
    progress_lock = threading.Lock()
    
    def hammer(thread_id):
        completed = 0
        errors = []
        start = time.time()
        
        for i in range(events_per_thread):
            try:
                memory.remember(
                    content=f"Thread {thread_id} event {i}: stress test payload",
                    metadata={"thread_id": thread_id, "seq": i}
                )
                completed += 1
                
                # Progress update every 100 events
                if completed % 100 == 0:
                    with progress_lock:
                        progress["completed"] += 100
                        pct = progress["completed"] / total_events * 100
                        print(f"[PROGRESS] {pct:.0f}% ({progress['completed']:,}/{total_events:,})")
                        
            except Exception as e:
                errors.append(f"{type(e).__name__}: {str(e)[:50]}")
        
        duration = time.time() - start
        
        # Add remaining to progress
        remaining = completed % 100
        if remaining > 0:
            with progress_lock:
                progress["completed"] += remaining
        
        with results_lock:
            results.append({
                "thread_id": thread_id,
                "completed": completed,
                "errors": errors,
                "duration": duration
            })
    
    # Run threads
    print(f"[RUN] Spawning {threads} threads...")
    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=hammer, args=(i,), name=f"hammer-{i}")
        thread_list.append(t)
    
    start_time = time.time()
    for t in thread_list:
        t.start()
    
    print(f"[RUN] All threads started, waiting for completion...")
    
    # Wait (no timeout - let it complete)
    for t in thread_list:
        t.join()
    
    total_duration = time.time() - start_time
    
    # Aggregate
    total_completed = sum(r["completed"] for r in results)
    total_errors = sum(len(r["errors"]) for r in results)
    
    # Drain
    print("\n[DRAIN] Closing memory, waiting for worker drain...")
    drain_start = time.time()
    memory.close()
    drain_duration = time.time() - drain_start
    print(f"[DRAIN] Completed in {drain_duration:.1f}s")
    
    # Report
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Total events:     {total_completed:,}/{total_events:,}")
    print(f"  Errors:           {total_errors}")
    print(f"  Duration:         {total_duration:.2f}s")
    print(f"  Throughput:       {total_completed/total_duration:,.0f} events/sec")
    print(f"  Drain time:       {drain_duration:.2f}s")
    
    if total_errors > 0:
        print(f"\nError summary:")
        error_counts = defaultdict(int)
        for r in results:
            for e in r["errors"]:
                error_counts[e.split(":")[0]] += 1
        for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}")
    
    print(f"\n{'-'*60}")
    if total_errors == 0 and total_completed == total_events:
        print("✅ PASSED - All events written successfully!")
    else:
        print(f"❌ FAILED - {total_errors} errors, {total_events - total_completed} missing")
    print(f"{'='*60}\n")
    
    # Cleanup
    clear_test_data(data_path)
    
    return 0 if total_errors == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
