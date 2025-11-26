#!/usr/bin/env python3
"""
KRNX Progressive Stress Test

Scales up to find where the system breaks.
"""

import os
import shutil
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_test_data(data_path: str):
    if os.path.exists(data_path):
        shutil.rmtree(data_path)

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
            except:
                pass
    except:
        pass

def run_test(threads: int, events_per_thread: int, timeout: int = 60):
    """Run a single stress test, returns (success, duration, completed, errors)"""
    data_path = "./stress-test-data"
    clear_test_data(data_path)
    clear_redis_streams()
    
    from chillbot import Memory
    
    memory = Memory(
        agent_id="stress-test",
        data_path=data_path,
        redis_host="localhost",
        redis_port=6379,
    )
    
    results = []
    results_lock = threading.Lock()
    
    def hammer(thread_id):
        completed = 0
        errors = []
        for i in range(events_per_thread):
            try:
                memory.remember(
                    content=f"Thread {thread_id} event {i}",
                    metadata={"thread_id": thread_id, "seq": i}
                )
                completed += 1
            except Exception as e:
                errors.append(str(e))
        
        with results_lock:
            results.append({"completed": completed, "errors": errors})
    
    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=hammer, args=(i,))
        thread_list.append(t)
    
    start = time.time()
    for t in thread_list:
        t.start()
    
    # Wait with timeout
    all_done = True
    for t in thread_list:
        t.join(timeout=timeout)
        if t.is_alive():
            all_done = False
    
    duration = time.time() - start
    
    total_completed = sum(r["completed"] for r in results)
    total_errors = sum(len(r["errors"]) for r in results)
    total_expected = threads * events_per_thread
    
    memory.close()
    clear_test_data(data_path)
    
    success = all_done and total_completed == total_expected
    return success, duration, total_completed, total_expected, total_errors


def main():
    print("\n" + "="*60)
    print("KRNX PROGRESSIVE STRESS TEST")
    print("="*60)
    
    # Test configurations: (threads, events_per_thread)
    configs = [
        (5, 100),      # 500 events
        (10, 100),     # 1,000 events
        (20, 100),     # 2,000 events
        (30, 100),     # 3,000 events
        (40, 100),     # 4,000 events
        (50, 100),     # 5,000 events
        (50, 500),     # 25,000 events
        (50, 1000),    # 50,000 events (original target)
    ]
    
    for threads, events in configs:
        total = threads * events
        print(f"\n[TEST] {threads} threads × {events} events = {total:,} total")
        print("       Running...", end=" ", flush=True)
        
        try:
            success, duration, completed, expected, errors = run_test(threads, events, timeout=120)
            
            if success:
                throughput = completed / duration
                print(f"✓ PASS")
                print(f"       Duration: {duration:.2f}s | Throughput: {throughput:,.0f} events/sec")
            else:
                print(f"✗ FAIL")
                print(f"       Completed: {completed:,}/{expected:,} | Errors: {errors} | Duration: {duration:.2f}s")
                print(f"\n⚠️  BREAKING POINT FOUND: {threads} threads × {events} events")
                break
                
        except Exception as e:
            print(f"✗ CRASH: {e}")
            print(f"\n⚠️  BREAKING POINT FOUND: {threads} threads × {events} events")
            break
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
