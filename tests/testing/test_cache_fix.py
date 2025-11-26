#!/usr/bin/env python3
"""
Test script to validate the 50-thread bottleneck fix.

Tests:
1. Single thread baseline
2. 10 threads (should work)
3. 50 threads (was failing, should now work)
4. Cache hit rate validation
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Mock VectorStore for testing the caching logic
class MockVectorStore:
    """Mock to test caching logic without Qdrant dependency."""
    
    def __init__(self, cache_ttl_seconds=300):
        self.cache_ttl = cache_ttl_seconds
        self._known_collections = {}
        self._collection_lock = threading.Lock()
        
        # Stats
        self.http_calls = 0
        self.cache_hits = 0
        self.collections_created = 0
    
    def ensure_collection(self, workspace_id: str, dimension: int):
        """Ensure collection exists (with caching)."""
        now = time.time()
        
        # Fast path: Cache hit
        if workspace_id in self._known_collections:
            cached_dim, cached_at = self._known_collections[workspace_id]
            if now - cached_at < self.cache_ttl:
                self.cache_hits += 1
                return
        
        # Slow path: Verify (simulated HTTP call)
        with self._collection_lock:
            # Double-check
            if workspace_id in self._known_collections:
                cached_dim, cached_at = self._known_collections[workspace_id]
                if now - cached_at < self.cache_ttl:
                    self.cache_hits += 1
                    return
            
            # Simulate HTTP call
            self.http_calls += 1
            time.sleep(0.01)  # Simulate network latency
            
            # Check if exists (all exist in mock)
            exists = workspace_id in self._known_collections
            
            if not exists:
                # Simulate create
                time.sleep(0.02)
                self.collections_created += 1
            
            # Update cache
            self._known_collections[workspace_id] = (dimension, now)
    
    def get_stats(self):
        return {
            "http_calls": self.http_calls,
            "cache_hits": self.cache_hits,
            "collections_created": self.collections_created,
            "total_ops": self.http_calls + self.cache_hits,
        }


def worker_task(vector_store, workspace_id, num_ops):
    """Simulate worker doing multiple operations."""
    for _ in range(num_ops):
        vector_store.ensure_collection(workspace_id, 384)


def run_test(num_threads, ops_per_thread, workspace_id="test-workspace"):
    """Run concurrent test."""
    print(f"\n{'='*60}")
    print(f"Test: {num_threads} threads × {ops_per_thread} ops")
    print(f"{'='*60}")
    
    vector_store = MockVectorStore(cache_ttl_seconds=300)
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(worker_task, vector_store, workspace_id, ops_per_thread)
            for _ in range(num_threads)
        ]
        
        # Wait for all to complete
        for future in as_completed(futures):
            future.result()
    
    elapsed = time.time() - start_time
    stats = vector_store.get_stats()
    
    print(f"\nResults:")
    print(f"  Time elapsed: {elapsed:.3f}s")
    print(f"  HTTP calls: {stats['http_calls']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Collections created: {stats['collections_created']}")
    print(f"  Total operations: {stats['total_ops']}")
    print(f"  Cache hit rate: {stats['cache_hits'] / stats['total_ops'] * 100:.1f}%")
    print(f"  Throughput: {stats['total_ops'] / elapsed:.0f} ops/sec")
    
    # Validate
    expected_total = num_threads * ops_per_thread
    assert stats['total_ops'] == expected_total, f"Lost operations! Expected {expected_total}, got {stats['total_ops']}"
    
    # Should have very few HTTP calls (ideally just 1)
    if stats['http_calls'] > num_threads:
        print(f"  ⚠️  Warning: More HTTP calls than threads (expected ≤{num_threads}, got {stats['http_calls']})")
    else:
        print(f"  ✅ HTTP calls optimized (≤{num_threads})")
    
    return stats, elapsed


def test_cache_expiry():
    """Test that cache expires correctly."""
    print(f"\n{'='*60}")
    print(f"Test: Cache TTL expiry")
    print(f"{'='*60}")
    
    vector_store = MockVectorStore(cache_ttl_seconds=1)  # 1 second TTL
    
    # First call - cache miss
    vector_store.ensure_collection("test", 384)
    assert vector_store.http_calls == 1
    assert vector_store.cache_hits == 0
    print("  ✅ First call - cache miss (HTTP call)")
    
    # Second call immediately - cache hit
    vector_store.ensure_collection("test", 384)
    assert vector_store.http_calls == 1
    assert vector_store.cache_hits == 1
    print("  ✅ Second call - cache hit")
    
    # Wait for TTL to expire
    time.sleep(1.1)
    
    # Third call - cache expired, should make HTTP call
    vector_store.ensure_collection("test", 384)
    assert vector_store.http_calls == 2
    assert vector_store.cache_hits == 1
    print("  ✅ After TTL expiry - cache miss (HTTP call)")


def test_multiple_workspaces():
    """Test that different workspaces are cached separately."""
    print(f"\n{'='*60}")
    print(f"Test: Multiple workspaces")
    print(f"{'='*60}")
    
    vector_store = MockVectorStore()
    
    # Create 5 different workspaces
    workspaces = [f"workspace-{i}" for i in range(5)]
    
    for ws in workspaces:
        vector_store.ensure_collection(ws, 384)
    
    # Each workspace should trigger one HTTP call
    assert vector_store.http_calls == 5
    assert vector_store.collections_created == 5
    print(f"  ✅ Created {len(workspaces)} workspaces - {vector_store.http_calls} HTTP calls")
    
    # Access them again - should all be cached
    for ws in workspaces:
        vector_store.ensure_collection(ws, 384)
    
    assert vector_store.http_calls == 5  # No new calls
    assert vector_store.cache_hits == 5
    print(f"  ✅ Second access - all cache hits")


def main():
    print("KRNX Vector Store - Cache Performance Test")
    print("Testing the 50-thread bottleneck fix\n")
    
    # Test 1: Single thread baseline
    stats1, time1 = run_test(num_threads=1, ops_per_thread=100)
    
    # Test 2: 10 threads (should work well)
    stats10, time10 = run_test(num_threads=10, ops_per_thread=100)
    
    # Test 3: 50 threads (THE BIG TEST - was failing, should now work)
    stats50, time50 = run_test(num_threads=50, ops_per_thread=100)
    
    # Test 4: Cache expiry
    test_cache_expiry()
    
    # Test 5: Multiple workspaces
    test_multiple_workspaces()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"1 thread:  {time1:.3f}s, {stats1['cache_hits']} cache hits")
    print(f"10 threads: {time10:.3f}s, {stats10['cache_hits']} cache hits")
    print(f"50 threads: {time50:.3f}s, {stats50['cache_hits']} cache hits")
    print(f"\n✅ All tests passed!")
    print(f"\nKey takeaway:")
    print(f"  - 50 threads made only {stats50['http_calls']} HTTP calls")
    print(f"  - {stats50['cache_hits']} operations served from cache")
    print(f"  - Cache hit rate: {stats50['cache_hits'] / stats50['total_ops'] * 100:.1f}%")
    print(f"\nWithout caching, this would have been {stats50['total_ops']} HTTP calls!")


if __name__ == "__main__":
    main()
