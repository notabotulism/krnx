#!/usr/bin/env python3
"""Quick diagnostic to see what's failing"""

import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_all():
    if os.path.exists("./diag-test"):
        shutil.rmtree("./diag-test")
    try:
        from chillbot.kernel.connection_pool import configure_pool, get_redis_client
        try:
            configure_pool(host="localhost", port=6379)
        except:
            pass
        redis = get_redis_client()
        for key in redis.keys("krnx:*"):
            redis.delete(key)
    except:
        pass

def main():
    print("="*60)
    print("ERROR DIAGNOSTIC")
    print("="*60)
    
    clear_all()
    
    from chillbot import Memory
    
    memory = Memory(
        agent_id="diag-test",
        data_path="./diag-test",
        redis_host="localhost",
        redis_port=6379,
        qdrant_url="http://localhost:6333",
    )
    
    # Check turbo status
    print(f"\nKernel type: {type(memory._kernel)}")
    print(f"Has write_event_turbo: {hasattr(memory._kernel, 'write_event_turbo')}")
    print(f"Orchestrator type: {type(memory._fabric)}")
    
    # Try single remember
    print("\n--- Test 1: Single remember() ---")
    try:
        event_id = memory.remember(content="Test event 1", metadata={"test": 1})
        print(f"✓ Success: {event_id}")
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
    
    # Try 10 sequential
    print("\n--- Test 2: 10 sequential remember() ---")
    errors = []
    for i in range(10):
        try:
            memory.remember(content=f"Test event {i}", metadata={"i": i})
        except Exception as e:
            errors.append(str(e))
    
    if errors:
        print(f"✗ {len(errors)} errors")
        print(f"  First error: {errors[0]}")
    else:
        print("✓ All 10 succeeded")
    
    # Try with threads
    print("\n--- Test 3: 5 threads × 10 events ---")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    thread_errors = []
    errors_lock = threading.Lock()
    
    def worker(tid):
        for i in range(10):
            try:
                memory.remember(content=f"Thread {tid} event {i}")
            except Exception as e:
                with errors_lock:
                    thread_errors.append(f"Thread {tid}: {e}")
    
    with ThreadPoolExecutor(max_workers=5) as ex:
        for tid in range(5):
            ex.submit(worker, tid)
    
    if thread_errors:
        print(f"✗ {len(thread_errors)} errors")
        # Show unique errors
        unique = list(set(thread_errors))[:5]
        for err in unique:
            print(f"  - {err[:100]}")
    else:
        print("✓ All 50 succeeded")
    
    memory.close()
    shutil.rmtree("./diag-test")

if __name__ == "__main__":
    main()
