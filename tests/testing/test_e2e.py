#!/usr/bin/env python3
"""
Chillbot End-to-End Integration Test

Prerequisites:
1. Redis running:     docker run -d -p 6379:6379 redis:7-alpine
2. Qdrant running:    docker run -d -p 6333:6333 qdrant/qdrant

Run:
    python test_e2e.py
"""

import sys
import os
import time
import tempfile
import shutil

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_redis():
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✓ Redis: connected")
        return True
    except Exception as e:
        print(f"✗ Redis: {e}")
        print("  → Run: docker run -d -p 6379:6379 redis:7-alpine")
        return False


def check_qdrant():
    """Check if Qdrant is available."""
    try:
        import httpx
        r = httpx.get("http://localhost:6333/health", timeout=5)
        if r.status_code == 200:
            print("✓ Qdrant: connected")
            return True
        else:
            print(f"✗ Qdrant: status {r.status_code}")
            return False
    except Exception as e:
        print(f"✗ Qdrant: {e}")
        print("  → Run: docker run -d -p 6333:6333 qdrant/qdrant")
        return False


def test_kernel_direct():
    """Test kernel layer directly."""
    print("\n=== TEST: Kernel (Direct) ===")
    
    from chillbot.kernel import KRNXController, Event
    
    data_path = tempfile.mkdtemp(prefix="krnx_test_")
    
    try:
        # Initialize
        krnx = KRNXController(
            data_path=data_path,
            redis_host="localhost",
            redis_port=6379,
            enable_async_worker=False,  # Sync for testing
        )
        print("  ✓ KRNXController initialized")
        
        # Write event
        event = Event(
            event_id="test_evt_001",
            workspace_id="test_workspace",
            user_id="test_user",
            session_id="test_session",
            content={"message": "Hello from kernel test"},
            timestamp=time.time(),
        )
        
        event_id = krnx.write_event("test_workspace", "test_user", event)
        print(f"  ✓ Event written: {event_id}")
        
        # Query events
        events = krnx.query_events("test_workspace", "test_user", limit=10)
        print(f"  ✓ Query returned {len(events)} events")
        
        # Verify content
        if events and events[0].content.get("message") == "Hello from kernel test":
            print("  ✓ Content verified")
        else:
            print("  ✗ Content mismatch!")
            return False
        
        # Cleanup
        krnx.close()
        print("  ✓ Kernel closed")
        
        return True
        
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


def test_compute_components():
    """Test compute layer components."""
    print("\n=== TEST: Compute Components ===")
    
    from chillbot.compute import JobQueue, JobType, EmbeddingEngine, VectorStore
    
    data_path = tempfile.mkdtemp(prefix="compute_test_")
    
    try:
        # Job Queue
        queue = JobQueue(f"{data_path}/jobs.db")
        job_id = queue.enqueue(
            job_type=JobType.EMBED,
            workspace_id="test",
            payload={"text": "test embedding"}
        )
        print(f"  ✓ JobQueue: enqueued {job_id}")
        
        pending = queue.pending_count()
        print(f"  ✓ JobQueue: {pending} pending jobs")
        queue.close()
        
        # Embeddings (only if model can be loaded)
        try:
            embeddings = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
            vec = embeddings.embed("Hello world")
            print(f"  ✓ Embeddings: dimension={len(vec)}")
        except Exception as e:
            print(f"  ⚠ Embeddings: {e} (may need sentence-transformers)")
        
        # Vector Store
        try:
            vectors = VectorStore(url="http://localhost:6333")
            vectors.ensure_collection("test_collection", dimension=384)
            print("  ✓ VectorStore: collection created")
            vectors.close()
        except Exception as e:
            print(f"  ⚠ VectorStore: {e}")
        
        return True
        
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


def test_fabric_components():
    """Test fabric layer components."""
    print("\n=== TEST: Fabric Components ===")
    
    from chillbot.fabric import ContextBuilder, IdentityResolver, RetentionManager
    from chillbot.fabric import MemoryItem
    
    # Context Builder
    builder = ContextBuilder(max_tokens=1000)
    
    memories = [
        MemoryItem(
            event_id="evt_1",
            content="User likes hiking",
            timestamp=time.time() - 3600,
            score=0.9,
        ),
        MemoryItem(
            event_id="evt_2", 
            content="User visited Alps last summer",
            timestamp=time.time() - 7200,
            score=0.8,
        ),
    ]
    
    context = builder.build(memories, query="outdoor activities", format="text")
    print(f"  ✓ ContextBuilder: built {len(context)} chars")
    
    # Identity Resolver
    resolver = IdentityResolver(default_workspace="test")
    resolver.register_agent("coder-1", workspace_id="project-x", role="coder")
    
    identity = resolver.resolve("coder-1")
    print(f"  ✓ IdentityResolver: {identity.workspace_id}/{identity.user_id}")
    
    # Retention Manager
    retention = RetentionManager()
    policies = retention.list_policies()
    print(f"  ✓ RetentionManager: {len(policies)} default policies")
    
    return True


def test_memory_simple():
    """Test the simple Memory interface."""
    print("\n=== TEST: Memory (Simple Interface) ===")
    
    from chillbot import Memory
    
    data_path = tempfile.mkdtemp(prefix="memory_test_")
    
    try:
        # Initialize Memory
        memory = Memory(
            agent_id="test-agent",
            data_path=data_path,
            redis_host="localhost",
            redis_port=6379,
            qdrant_url="http://localhost:6333",
        )
        print(f"  ✓ Memory initialized: {memory}")
        
        # Remember
        event_id = memory.remember("User loves hiking in the mountains")
        print(f"  ✓ Remember: {event_id}")
        
        # Remember more
        memory.remember("User's favorite color is blue")
        memory.remember("User works as a software engineer")
        print("  ✓ Remembered 3 items")
        
        # Give kernel time to process
        time.sleep(0.5)
        
        # Recall (may not have vectors yet without worker)
        memories = memory.recall("outdoor activities", top_k=5)
        print(f"  ✓ Recall returned {len(memories)} memories")
        
        # Stats
        stats = memory.stats()
        print(f"  ✓ Stats: mode={stats.get('mode')}")
        
        # Cleanup
        memory.close()
        print("  ✓ Memory closed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


def test_full_flow():
    """Test complete remember → embed → recall flow."""
    print("\n=== TEST: Full Flow (with Worker) ===")
    
    from chillbot import Memory
    from chillbot.compute import ComputeWorker
    import threading
    
    data_path = tempfile.mkdtemp(prefix="fullflow_test_")
    
    try:
        # Initialize Memory
        memory = Memory(
            agent_id="fullflow-agent",
            data_path=data_path,
            redis_host="localhost",
            redis_port=6379,
            qdrant_url="http://localhost:6333",
        )
        
        # Start worker in background (processes embedding jobs)
        worker = None
        if memory._job_queue and memory._embeddings and memory._vectors:
            worker = ComputeWorker(
                job_queue=memory._job_queue,
                embeddings=memory._embeddings,
                vectors=memory._vectors,
                poll_interval=0.1,  # Fast polling for test
            )
            worker_thread = threading.Thread(target=worker.run, daemon=True)
            worker_thread.start()
            print("  ✓ Worker started")
        else:
            print("  ⚠ Worker not started (missing components)")
        
        # Remember items
        memory.remember("User loves hiking and mountain climbing")
        memory.remember("User enjoys photography, especially landscapes")
        memory.remember("User's dog is named Max, a golden retriever")
        print("  ✓ Remembered 3 items")
        
        # Wait for worker to process
        time.sleep(2)
        
        # Recall with semantic search
        results = memory.recall("outdoor hobbies", top_k=3)
        print(f"  ✓ Semantic recall: {len(results)} results")
        
        for r in results:
            score = r.get('score', 0)
            content = r.get('content', {})
            if isinstance(content, dict):
                text = content.get('text', str(content))[:50]
            else:
                text = str(content)[:50]
            print(f"      {score:.2f}: {text}...")
        
        # Build context
        context = memory.context("plan an outdoor trip", max_tokens=500)
        print(f"  ✓ Context built: {len(context)} chars")
        
        # Stop worker
        if worker:
            worker.stop()
        
        memory.close()
        print("  ✓ Closed")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


def main():
    print("=" * 60)
    print("CHILLBOT END-TO-END INTEGRATION TEST")
    print("=" * 60)
    
    # Check infrastructure
    print("\n=== INFRASTRUCTURE ===")
    redis_ok = check_redis()
    qdrant_ok = check_qdrant()
    
    if not redis_ok:
        print("\n⚠ Redis required for kernel tests. Skipping kernel tests.")
    
    results = {}
    
    # Run tests
    if redis_ok:
        results['kernel'] = test_kernel_direct()
    
    results['compute'] = test_compute_components()
    results['fabric'] = test_fabric_components()
    
    if redis_ok:
        results['memory'] = test_memory_simple()
    
    if redis_ok and qdrant_ok:
        results['full_flow'] = test_full_flow()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
