#!/usr/bin/env python3
"""
Isolate Pause Source - Times each component separately
"""

import os
import shutil
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clear_all():
    if os.path.exists("./pause-test-data"):
        shutil.rmtree("./pause-test-data")
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

def test_1_ltm_only():
    """Test LTM batch writes with NO other components"""
    print("\n" + "="*60)
    print("TEST 1: LTM ONLY (no Redis, no Qdrant)")
    print("="*60)
    
    from chillbot.kernel.ltm import LTM
    from chillbot.kernel.models import Event
    import uuid
    
    clear_all()
    ltm = LTM(data_path="./pause-test-data", high_throughput_mode=True)
    
    total_events = 10000
    batch_size = 500
    
    print(f"Writing {total_events} events in batches of {batch_size}...")
    
    overall_start = time.time()
    batch_times = []
    
    for batch_num in range(total_events // batch_size):
        events = [
            Event(
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
                workspace_id="test",
                user_id="user",
                session_id="sess",
                content={"text": f"Event {batch_num * batch_size + i}"},
                timestamp=time.time(),
            )
            for i in range(batch_size)
        ]
        
        batch_start = time.time()
        ltm.store_events_batch(events)
        batch_time = (time.time() - batch_start) * 1000
        batch_times.append(batch_time)
        
        total_so_far = (batch_num + 1) * batch_size
        if total_so_far % 2000 == 0:
            avg = sum(batch_times[-4:]) / len(batch_times[-4:])
            print(f"  {total_so_far:>6} events | last batch: {batch_time:>6.1f}ms | avg(4): {avg:.1f}ms")
    
    total_time = time.time() - overall_start
    
    # Check for outliers (potential pauses)
    avg_time = sum(batch_times) / len(batch_times)
    max_time = max(batch_times)
    outliers = [t for t in batch_times if t > avg_time * 3]
    
    print(f"\nRESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_events/total_time:.0f} events/sec")
    print(f"  Avg batch:  {avg_time:.1f}ms")
    print(f"  Max batch:  {max_time:.1f}ms")
    print(f"  Outliers:   {len(outliers)} batches > 3x avg")
    
    if max_time > 1000:
        print(f"  ⚠️  PAUSE DETECTED in LTM! Max batch took {max_time:.0f}ms")
    else:
        print(f"  ✓ LTM is clean (no pauses)")
    
    ltm.close()
    shutil.rmtree("./pause-test-data")
    
    return max_time < 1000


def test_2_redis_only():
    """Test Redis XADD with NO other components"""
    print("\n" + "="*60)
    print("TEST 2: REDIS ONLY (stream writes)")
    print("="*60)
    
    from chillbot.kernel.connection_pool import configure_pool, get_redis_client
    
    try:
        configure_pool(host="localhost", port=6379)
    except:
        pass
    
    redis = get_redis_client()
    redis.delete("test:pause:stream")
    
    total_events = 10000
    batch_size = 500
    
    print(f"Writing {total_events} events in batches of {batch_size}...")
    
    overall_start = time.time()
    batch_times = []
    
    for batch_num in range(total_events // batch_size):
        batch_start = time.time()
        
        pipe = redis.pipeline()
        for i in range(batch_size):
            pipe.xadd("test:pause:stream", {"data": f"event_{batch_num * batch_size + i}"}, maxlen=50000)
        pipe.execute()
        
        batch_time = (time.time() - batch_start) * 1000
        batch_times.append(batch_time)
        
        total_so_far = (batch_num + 1) * batch_size
        if total_so_far % 2000 == 0:
            avg = sum(batch_times[-4:]) / len(batch_times[-4:])
            print(f"  {total_so_far:>6} events | last batch: {batch_time:>6.1f}ms | avg(4): {avg:.1f}ms")
    
    total_time = time.time() - overall_start
    
    avg_time = sum(batch_times) / len(batch_times)
    max_time = max(batch_times)
    
    print(f"\nRESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_events/total_time:.0f} events/sec")
    print(f"  Avg batch:  {avg_time:.1f}ms")
    print(f"  Max batch:  {max_time:.1f}ms")
    
    if max_time > 500:
        print(f"  ⚠️  PAUSE DETECTED in Redis! Max batch took {max_time:.0f}ms")
    else:
        print(f"  ✓ Redis is clean (no pauses)")
    
    redis.delete("test:pause:stream")
    
    return max_time < 500


def test_3_memory_no_vectors():
    """Test full Memory path WITHOUT vectors"""
    print("\n" + "="*60)
    print("TEST 3: MEMORY (no vectors/embeddings)")
    print("="*60)
    
    clear_all()
    
    from chillbot import Memory
    
    memory = Memory(
        agent_id="pause-test",
        data_path="./pause-test-data",
        redis_host="localhost",
        redis_port=6379,
        qdrant_url=None,  # DISABLED
    )
    
    total_events = 5000
    
    print(f"Writing {total_events} events...")
    
    overall_start = time.time()
    event_times = []
    last_report = 0
    
    for i in range(total_events):
        start = time.time()
        memory.remember(content=f"Event {i}", metadata={"i": i})
        event_times.append((time.time() - start) * 1000)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - overall_start
            recent_avg = sum(event_times[-100:]) / 100
            print(f"  {i+1:>5} events | {(i+1)/elapsed:.0f} evt/s | recent avg: {recent_avg:.1f}ms")
    
    total_time = time.time() - overall_start
    
    avg_time = sum(event_times) / len(event_times)
    max_time = max(event_times)
    slow_events = [t for t in event_times if t > 100]
    
    print(f"\nRESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_events/total_time:.0f} events/sec")
    print(f"  Avg event:  {avg_time:.2f}ms")
    print(f"  Max event:  {max_time:.1f}ms")
    print(f"  Slow (>100ms): {len(slow_events)} events")
    
    if max_time > 1000:
        print(f"  ⚠️  PAUSE DETECTED! Max event took {max_time:.0f}ms")
    else:
        print(f"  ✓ Memory (no vectors) is clean")
    
    memory.close()
    shutil.rmtree("./pause-test-data")
    
    return max_time < 1000


def test_4_memory_with_vectors():
    """Test full Memory path WITH vectors"""
    print("\n" + "="*60)
    print("TEST 4: MEMORY (WITH vectors/embeddings)")
    print("="*60)
    
    clear_all()
    
    from chillbot import Memory
    
    memory = Memory(
        agent_id="pause-test",
        data_path="./pause-test-data",
        redis_host="localhost",
        redis_port=6379,
        qdrant_url="http://localhost:6333",  # ENABLED
    )
    
    total_events = 5000
    
    print(f"Writing {total_events} events...")
    
    overall_start = time.time()
    event_times = []
    
    for i in range(total_events):
        start = time.time()
        memory.remember(content=f"Event {i}", metadata={"i": i})
        event_times.append((time.time() - start) * 1000)
        
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - overall_start
            recent_avg = sum(event_times[-100:]) / 100
            print(f"  {i+1:>5} events | {(i+1)/elapsed:.0f} evt/s | recent avg: {recent_avg:.1f}ms")
    
    total_time = time.time() - overall_start
    
    avg_time = sum(event_times) / len(event_times)
    max_time = max(event_times)
    slow_events = [t for t in event_times if t > 100]
    
    print(f"\nRESULTS:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {total_events/total_time:.0f} events/sec")
    print(f"  Avg event:  {avg_time:.2f}ms")
    print(f"  Max event:  {max_time:.1f}ms")
    print(f"  Slow (>100ms): {len(slow_events)} events")
    
    if max_time > 1000:
        print(f"  ⚠️  PAUSE DETECTED with vectors! Max event took {max_time:.0f}ms")
    else:
        print(f"  ✓ Memory (with vectors) is clean")
    
    memory.close()
    shutil.rmtree("./pause-test-data")
    
    return max_time < 1000


def main():
    print("="*60)
    print("PAUSE ISOLATION TEST")
    print("="*60)
    print("This will identify exactly which component causes pauses.\n")
    
    results = {}
    
    results['ltm'] = test_1_ltm_only()
    results['redis'] = test_2_redis_only()
    results['memory_no_vec'] = test_3_memory_no_vectors()
    results['memory_with_vec'] = test_4_memory_with_vectors()
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if not results['ltm']:
        print("❌ PROBLEM: LTM (SQLite) is causing pauses")
        print("   → Check WAL checkpoint settings")
        print("   → Try: PRAGMA wal_autocheckpoint=0")
    elif not results['redis']:
        print("❌ PROBLEM: Redis is causing pauses")
        print("   → Check Redis memory/persistence settings")
        print("   → Try: CONFIG SET appendfsync no")
    elif not results['memory_no_vec']:
        print("❌ PROBLEM: Memory stack (without vectors) is causing pauses")
        print("   → Check controller worker loop")
        print("   → Check batch accumulation logic")
    elif not results['memory_with_vec']:
        print("❌ PROBLEM: Vectors/Embeddings are causing pauses")
        print("   → Check Qdrant connection")
        print("   → Check embedding model loading")
    else:
        print("✓ All components are clean - pause may be in multi-threaded scenario")
        print("   → Run full stress test to confirm")
    
    print("="*60)


if __name__ == "__main__":
    main()
