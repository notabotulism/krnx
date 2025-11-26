#!/usr/bin/env python3
"""
Multi-threaded Pause Isolation - Find concurrency bottleneck
"""

import os
import shutil
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_all():
    if os.path.exists("./mt-pause-test"):
        shutil.rmtree("./mt-pause-test")
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


def test_concurrent_ltm(num_threads=50, events_per_thread=200):
    """Test LTM under concurrency"""
    print(f"\n{'='*60}")
    print(f"TEST: LTM with {num_threads} threads × {events_per_thread} events")
    print("="*60)
    
    clear_all()
    
    from chillbot.kernel.ltm import LTM
    from chillbot.kernel.models import Event
    import uuid
    
    ltm = LTM(data_path="./mt-pause-test", high_throughput_mode=True)
    
    # Shared timing data
    all_batch_times = []
    times_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress = {"completed": 0, "events": 0}
    
    def worker(thread_id):
        batch_times = []
        batch_size = 50
        
        for batch_num in range(events_per_thread // batch_size):
            events = [
                Event(
                    event_id=f"evt_{thread_id}_{batch_num}_{i}_{uuid.uuid4().hex[:8]}",
                    workspace_id="test",
                    user_id=f"user_{thread_id}",
                    session_id=f"sess_{thread_id}",
                    content={"text": f"Thread {thread_id} batch {batch_num} event {i}"},
                    timestamp=time.time(),
                )
                for i in range(batch_size)
            ]
            
            start = time.time()
            ltm.store_events_batch(events)
            batch_time = (time.time() - start) * 1000
            batch_times.append(batch_time)
        
        with times_lock:
            all_batch_times.extend(batch_times)
        
        with progress_lock:
            progress["completed"] += 1
            progress["events"] += events_per_thread
            if progress["completed"] % 10 == 0:
                print(f"  Progress: {progress['completed']}/{num_threads} threads done")
        
        return batch_times
    
    print(f"Starting {num_threads} threads...")
    overall_start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in as_completed(futures):
            f.result()  # Raise any exceptions
    
    total_time = time.time() - overall_start
    total_events = num_threads * events_per_thread
    
    avg_time = sum(all_batch_times) / len(all_batch_times)
    max_time = max(all_batch_times)
    slow_batches = [t for t in all_batch_times if t > 1000]
    very_slow = [t for t in all_batch_times if t > 5000]
    
    print(f"\nRESULTS (LTM concurrent):")
    print(f"  Total events: {total_events}")
    print(f"  Total time:   {total_time:.2f}s")
    print(f"  Throughput:   {total_events/total_time:.0f} events/sec")
    print(f"  Avg batch:    {avg_time:.1f}ms")
    print(f"  Max batch:    {max_time:.1f}ms")
    print(f"  Slow (>1s):   {len(slow_batches)} batches")
    print(f"  Very slow (>5s): {len(very_slow)} batches")
    
    if very_slow:
        print(f"  ⚠️  PAUSE DETECTED in LTM! {len(very_slow)} batches took >5s")
        print(f"      Worst times: {sorted(very_slow, reverse=True)[:5]}")
    
    ltm.close()
    shutil.rmtree("./mt-pause-test")
    
    return max_time < 5000


def test_concurrent_redis(num_threads=50, events_per_thread=200):
    """Test Redis under concurrency"""
    print(f"\n{'='*60}")
    print(f"TEST: Redis with {num_threads} threads × {events_per_thread} events")
    print("="*60)
    
    from chillbot.kernel.connection_pool import configure_pool, get_redis_client
    
    try:
        configure_pool(host="localhost", port=6379, max_connections=200)
    except:
        pass
    
    redis = get_redis_client()
    redis.delete("test:mt:stream")
    
    all_batch_times = []
    times_lock = threading.Lock()
    progress = {"completed": 0}
    progress_lock = threading.Lock()
    
    def worker(thread_id):
        batch_times = []
        batch_size = 50
        client = get_redis_client()
        
        for batch_num in range(events_per_thread // batch_size):
            start = time.time()
            pipe = client.pipeline()
            for i in range(batch_size):
                pipe.xadd("test:mt:stream", {"data": f"t{thread_id}_b{batch_num}_e{i}"}, maxlen=100000)
            pipe.execute()
            batch_time = (time.time() - start) * 1000
            batch_times.append(batch_time)
        
        with times_lock:
            all_batch_times.extend(batch_times)
        
        with progress_lock:
            progress["completed"] += 1
            if progress["completed"] % 10 == 0:
                print(f"  Progress: {progress['completed']}/{num_threads} threads done")
        
        return batch_times
    
    print(f"Starting {num_threads} threads...")
    overall_start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    
    total_time = time.time() - overall_start
    total_events = num_threads * events_per_thread
    
    avg_time = sum(all_batch_times) / len(all_batch_times)
    max_time = max(all_batch_times)
    slow_batches = [t for t in all_batch_times if t > 1000]
    
    print(f"\nRESULTS (Redis concurrent):")
    print(f"  Total events: {total_events}")
    print(f"  Total time:   {total_time:.2f}s")
    print(f"  Throughput:   {total_events/total_time:.0f} events/sec")
    print(f"  Avg batch:    {avg_time:.1f}ms")
    print(f"  Max batch:    {max_time:.1f}ms")
    print(f"  Slow (>1s):   {len(slow_batches)} batches")
    
    if max_time > 5000:
        print(f"  ⚠️  PAUSE DETECTED in Redis!")
    
    redis.delete("test:mt:stream")
    
    return max_time < 5000


def test_concurrent_memory(num_threads=50, events_per_thread=200):
    """Test full Memory under concurrency"""
    print(f"\n{'='*60}")
    print(f"TEST: Full Memory with {num_threads} threads × {events_per_thread} events")
    print("="*60)
    
    clear_all()
    
    from chillbot import Memory
    
    memory = Memory(
        agent_id="mt-pause-test",
        data_path="./mt-pause-test",
        redis_host="localhost",
        redis_port=6379,
        qdrant_url="http://localhost:6333",
    )
    
    all_event_times = []
    times_lock = threading.Lock()
    progress = {"completed": 0, "events": 0}
    progress_lock = threading.Lock()
    errors = []
    errors_lock = threading.Lock()
    
    # Track timing per-second for pause detection
    second_buckets = {}
    bucket_lock = threading.Lock()
    
    def worker(thread_id):
        event_times = []
        
        for i in range(events_per_thread):
            start = time.time()
            try:
                memory.remember(
                    content=f"Thread {thread_id} event {i}",
                    metadata={"thread": thread_id, "event": i}
                )
                elapsed = (time.time() - start) * 1000
                event_times.append(elapsed)
                
                # Track by second for pause detection
                second = int(time.time())
                with bucket_lock:
                    if second not in second_buckets:
                        second_buckets[second] = []
                    second_buckets[second].append(elapsed)
                    
            except Exception as e:
                with errors_lock:
                    errors.append(str(e))
        
        with times_lock:
            all_event_times.extend(event_times)
        
        with progress_lock:
            progress["completed"] += 1
            progress["events"] += events_per_thread
            if progress["completed"] % 10 == 0:
                elapsed = time.time() - overall_start
                rate = progress["events"] / elapsed if elapsed > 0 else 0
                print(f"  Progress: {progress['completed']}/{num_threads} threads | {rate:.0f} evt/s")
        
        return event_times
    
    print(f"Starting {num_threads} threads...")
    overall_start = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in as_completed(futures):
            f.result()
    
    total_time = time.time() - overall_start
    total_events = num_threads * events_per_thread
    
    avg_time = sum(all_event_times) / len(all_event_times)
    max_time = max(all_event_times)
    slow_events = [t for t in all_event_times if t > 1000]
    very_slow = [t for t in all_event_times if t > 5000]
    
    # Analyze per-second throughput for pauses
    sorted_seconds = sorted(second_buckets.keys())
    throughput_per_second = [(s, len(second_buckets[s])) for s in sorted_seconds]
    
    # Find gaps (potential pauses)
    low_throughput_seconds = [(s, count) for s, count in throughput_per_second if count < 50]
    
    print(f"\nRESULTS (Full Memory concurrent):")
    print(f"  Total events: {total_events}")
    print(f"  Total time:   {total_time:.2f}s")
    print(f"  Throughput:   {total_events/total_time:.0f} events/sec")
    print(f"  Avg event:    {avg_time:.1f}ms")
    print(f"  Max event:    {max_time:.1f}ms")
    print(f"  Slow (>1s):   {len(slow_events)} events")
    print(f"  Very slow (>5s): {len(very_slow)} events")
    print(f"  Errors:       {len(errors)}")
    
    if low_throughput_seconds:
        print(f"\n  ⚠️  LOW THROUGHPUT SECONDS (potential pauses):")
        for sec, count in low_throughput_seconds[:10]:
            print(f"      Second {sec - sorted_seconds[0]}: only {count} events")
    
    if very_slow:
        print(f"\n  ⚠️  PAUSE DETECTED! {len(very_slow)} events took >5s")
        print(f"      Worst times: {sorted(very_slow, reverse=True)[:5]}")
    
    memory.close()
    shutil.rmtree("./mt-pause-test")
    
    return max_time < 5000 and len(very_slow) == 0


def main():
    print("="*60)
    print("MULTI-THREADED PAUSE ISOLATION")
    print("="*60)
    print("Testing with 50 threads to find concurrency bottleneck\n")
    
    results = {}
    
    # Test each layer under concurrency
    results['ltm'] = test_concurrent_ltm(num_threads=50, events_per_thread=200)
    results['redis'] = test_concurrent_redis(num_threads=50, events_per_thread=200)
    results['memory'] = test_concurrent_memory(num_threads=50, events_per_thread=200)
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if not results['ltm']:
        print("❌ LTM (SQLite) has concurrency issues")
        print("   → The _write_lock is causing thread pile-up")
        print("   → Consider connection pooling or sharding")
    elif not results['redis']:
        print("❌ Redis has concurrency issues")
        print("   → Check connection pool size")
        print("   → Check Redis server memory")
    elif not results['memory']:
        print("❌ Full Memory stack has concurrency issues")
        print("   → Check job_queue.enqueue() under load")
        print("   → Check kernel.write_event() contention")
    else:
        print("✓ All components handle 50 threads well!")
    
    print("="*60)


if __name__ == "__main__":
    main()
