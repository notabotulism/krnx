#!/usr/bin/env python3
"""
Latency Diagnostic - Find where the 50ms is going

Tests each component in isolation to identify bottleneck.
"""

import os
import sys
import time
import threading
import uuid
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def bench(name, fn, iterations=1000):
    """Benchmark a function."""
    times = []
    for _ in range(iterations):
        start = time.time()
        fn()
        times.append((time.time() - start) * 1000)
    
    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    print(f"  {name}: avg={avg:.3f}ms p50={p50:.3f}ms p99={p99:.3f}ms")
    return avg

def main():
    print("=" * 60)
    print("LATENCY DIAGNOSTIC")
    print("=" * 60)
    
    # Setup
    from chillbot.kernel.connection_pool import configure_pool, get_redis_client
    configure_pool(host="localhost", port=6379, max_connections=200)
    
    redis_client = get_redis_client()
    redis_client.flushdb()
    
    print("\n1. BASELINE OPERATIONS (single-threaded)")
    print("-" * 40)
    
    # Test 1: Raw Redis PING
    bench("Redis PING", lambda: redis_client.ping())
    
    # Test 2: Redis SET
    bench("Redis SET", lambda: redis_client.set("test_key", "test_value"))
    
    # Test 3: Redis pipeline (similar to turbo write)
    def pipeline_test():
        pipe = redis_client.pipeline()
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        pipe.setex(f"event:{event_id}", 86400, '{"test": true}')
        pipe.xadd("test:stream", {"event_id": event_id}, maxlen=10000)
        pipe.lpush("test:recent", event_id)
        pipe.ltrim("test:recent", 0, 99)
        pipe.expire("test:recent", 86400)
        pipe.xadd("test:ltm:queue", {"event_id": event_id}, maxlen=10000)
        pipe.execute()
    
    bench("Redis 6-cmd pipeline", pipeline_test)
    
    # Test 4: UUID generation
    bench("UUID generation", lambda: f"evt_{uuid.uuid4().hex[:16]}")
    
    # Test 5: JSON serialization
    content = {"text": "Thread 1 event 1", "metadata": {"thread": 1, "event": 1}}
    bench("JSON dumps", lambda: json.dumps(content))
    
    # Test 6: Event object creation
    from chillbot.kernel.models import Event
    def create_event():
        return Event(
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            workspace_id="test",
            user_id="default",
            session_id="test_default",
            content={"text": "test"},
            timestamp=time.time(),
            metadata={},
        )
    bench("Event creation", create_event)
    
    # Test 7: Full turbo write (kernel only)
    from chillbot.kernel.controller import KRNXController
    import shutil
    
    if os.path.exists("./diag-test"):
        shutil.rmtree("./diag-test")
    
    kernel = KRNXController(
        data_path="./diag-test",
        redis_host="localhost",
        redis_port=6379,
        enable_backpressure=False,  # Disable for pure timing
    )
    
    def turbo_write():
        event = Event(
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            workspace_id="test",
            user_id="default",
            session_id="test_default",
            content={"text": "test"},
            timestamp=time.time(),
            metadata={},
        )
        kernel.write_event_turbo("test", "default", event)
    
    bench("Kernel turbo write", turbo_write)
    
    # Test 8: Full Memory.remember()
    from chillbot import Memory
    
    memory = Memory(
        agent_id="diag-test",
        data_path="./diag-test",
        redis_host="localhost",
        redis_port=6379,
        qdrant_url="http://localhost:6333",
    )
    
    def full_remember():
        memory.remember(content="Test event", metadata={"test": True})
    
    bench("Full Memory.remember()", full_remember, iterations=500)
    
    print("\n2. CONCURRENT OPERATIONS (50 threads)")
    print("-" * 40)
    
    # Test concurrent pipeline
    results = []
    results_lock = threading.Lock()
    
    def concurrent_pipeline_worker(iterations):
        local_times = []
        for _ in range(iterations):
            start = time.time()
            pipeline_test()
            local_times.append((time.time() - start) * 1000)
        with results_lock:
            results.extend(local_times)
    
    threads = []
    for _ in range(50):
        t = threading.Thread(target=concurrent_pipeline_worker, args=(100,))
        threads.append(t)
    
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    results.sort()
    total_time = time.time() - start
    throughput = len(results) / total_time
    print(f"  50-thread pipeline: avg={sum(results)/len(results):.2f}ms p50={results[len(results)//2]:.2f}ms p99={results[int(len(results)*0.99)]:.2f}ms")
    print(f"  Throughput: {throughput:.0f} ops/sec")
    
    # Test concurrent Memory.remember()
    results2 = []
    
    def concurrent_remember_worker(iterations):
        local_times = []
        for i in range(iterations):
            start = time.time()
            memory.remember(content=f"Concurrent event {i}", metadata={"i": i})
            local_times.append((time.time() - start) * 1000)
        with results_lock:
            results2.extend(local_times)
    
    threads = []
    for _ in range(50):
        t = threading.Thread(target=concurrent_remember_worker, args=(100,))
        threads.append(t)
    
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    results2.sort()
    total_time = time.time() - start
    throughput = len(results2) / total_time
    print(f"  50-thread remember: avg={sum(results2)/len(results2):.2f}ms p50={results2[len(results2)//2]:.2f}ms p99={results2[int(len(results2)*0.99)]:.2f}ms")
    print(f"  Throughput: {throughput:.0f} ops/sec")
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    pipeline_avg = sum(results) / len(results)
    remember_avg = sum(results2) / len(results2)
    overhead = remember_avg - pipeline_avg
    
    print(f"\n  Raw pipeline avg:     {pipeline_avg:.2f}ms")
    print(f"  Full remember avg:    {remember_avg:.2f}ms")
    print(f"  Overhead (non-Redis): {overhead:.2f}ms")
    
    if overhead > 10:
        print(f"\n  ⚠️  {overhead:.0f}ms overhead suggests Python-side bottleneck")
        print("     Check: imports in hot path, object creation, JSON serialization")
    elif pipeline_avg > 5:
        print(f"\n  ⚠️  {pipeline_avg:.0f}ms pipeline suggests Redis contention")
        print("     Check: connection pool size, Redis server performance")
    else:
        print(f"\n  ✓ Performance looks reasonable")
    
    # Cleanup
    memory.close()
    kernel.close()
    shutil.rmtree("./diag-test")

if __name__ == "__main__":
    main()
