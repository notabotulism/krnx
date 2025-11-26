#!/usr/bin/env python3
"""
KRNX Stress Test - DIAGNOSTIC VERSION

This version adds detailed logging to find exactly where the freeze occurs.
"""

import os
import shutil
import sys
import threading
import time

# Ensure clean import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_test_data(data_path: str):
    """Remove test data directory."""
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        print(f"[CLEANUP] Removed {data_path}")


def clear_redis_streams():
    """Clear Redis streams used by KRNX."""
    try:
        from chillbot.kernel.connection_pool import configure_pool, get_redis_client
        
        try:
            configure_pool(host="localhost", port=6379)
        except Exception:
            pass
        
        redis = get_redis_client()
        
        streams = ["krnx:ltm:queue", "krnx:compute:jobs"]
        for stream in streams:
            try:
                redis.delete(stream)
                print(f"[CLEANUP] Cleared {stream}")
            except Exception as e:
                print(f"[CLEANUP] Could not clear {stream}: {e}")
                
    except Exception as e:
        print(f"[WARN] Redis cleanup failed: {e}")


def main():
    data_path = "./stress-test-data"
    threads = 5  # Start small
    events_per_thread = 10
    
    print(f"\n🔍 DIAGNOSTIC STRESS TEST")
    print(f"   Threads: {threads}, Events/thread: {events_per_thread}\n")
    
    # Cleanup
    clear_test_data(data_path)
    clear_redis_streams()
    
    # Step 1: Import
    print("[STEP 1] Importing Memory...")
    from chillbot import Memory
    print("[STEP 1] ✓ Import complete")
    
    # Step 2: Initialize
    print("\n[STEP 2] Creating Memory instance...")
    memory = Memory(
        agent_id="stress-test",
        data_path=data_path,
        redis_host="localhost",
        redis_port=6379,
    )
    print("[STEP 2] ✓ Memory created")
    
    # Step 3: Test single write
    print("\n[STEP 3] Testing single remember()...")
    start = time.time()
    event_id = memory.remember(content="Test event 0", metadata={"test": True})
    print(f"[STEP 3] ✓ Single remember completed in {(time.time()-start)*1000:.1f}ms: {event_id}")
    
    # Step 4: Test 10 sequential writes
    print("\n[STEP 4] Testing 10 sequential remember() calls...")
    start = time.time()
    for i in range(10):
        memory.remember(content=f"Sequential event {i}", metadata={"seq": i})
    print(f"[STEP 4] ✓ 10 sequential writes completed in {(time.time()-start)*1000:.1f}ms")
    
    # Step 5: Test single thread with multiple writes
    print("\n[STEP 5] Testing single background thread...")
    results = {"completed": 0, "error": None}
    
    def single_thread_test():
        try:
            for i in range(events_per_thread):
                memory.remember(content=f"Thread test {i}", metadata={"i": i})
                results["completed"] += 1
        except Exception as e:
            results["error"] = str(e)
    
    t = threading.Thread(target=single_thread_test)
    start = time.time()
    t.start()
    t.join(timeout=10)
    
    if t.is_alive():
        print(f"[STEP 5] ✗ Thread hung! Completed {results['completed']}/{events_per_thread}")
    else:
        print(f"[STEP 5] ✓ Single thread completed {results['completed']} events in {(time.time()-start)*1000:.1f}ms")
    
    # Step 6: Test 5 concurrent threads
    print(f"\n[STEP 6] Testing {threads} concurrent threads...")
    
    thread_results = {}
    thread_lock = threading.Lock()
    
    def hammer(thread_id):
        completed = 0
        try:
            for i in range(events_per_thread):
                print(f"  [T{thread_id}] Writing event {i}...")
                memory.remember(
                    content=f"Thread {thread_id} event {i}",
                    metadata={"thread_id": thread_id, "seq": i}
                )
                completed += 1
        except Exception as e:
            with thread_lock:
                thread_results[thread_id] = {"completed": completed, "error": str(e)}
            return
        
        with thread_lock:
            thread_results[thread_id] = {"completed": completed, "error": None}
    
    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=hammer, args=(i,), name=f"hammer-{i}")
        thread_list.append(t)
    
    print(f"[STEP 6] Starting {threads} threads...")
    start = time.time()
    
    for t in thread_list:
        t.start()
        print(f"  Started {t.name}")
    
    print(f"[STEP 6] All threads started, waiting for completion...")
    
    # Wait with timeout
    for t in thread_list:
        t.join(timeout=30)
        if t.is_alive():
            print(f"  ✗ {t.name} is HUNG")
        else:
            print(f"  ✓ {t.name} completed")
    
    duration = time.time() - start
    total_completed = sum(r.get("completed", 0) for r in thread_results.values())
    total_expected = threads * events_per_thread
    
    print(f"\n[STEP 6] Results: {total_completed}/{total_expected} events in {duration:.2f}s")
    
    for tid, result in sorted(thread_results.items()):
        status = "✓" if result["completed"] == events_per_thread else "✗"
        error_msg = f" - ERROR: {result['error']}" if result["error"] else ""
        print(f"  Thread {tid}: {result['completed']}/{events_per_thread} {status}{error_msg}")
    
    # Cleanup
    print("\n[CLEANUP] Closing memory...")
    memory.close()
    clear_test_data(data_path)
    
    print("\n" + "="*50)
    if total_completed == total_expected:
        print("✅ DIAGNOSTIC PASSED")
    else:
        print("❌ DIAGNOSTIC FAILED - See above for details")
    print("="*50)


if __name__ == "__main__":
    main()
