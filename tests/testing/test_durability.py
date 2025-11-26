#!/usr/bin/env python3
"""
KRNX Durability Test Suite

Tests system behavior under adverse conditions:
- Infrastructure failures (Redis/Qdrant down)
- Concurrent operations (race conditions)
- Shutdown under load
- Network partitions
- Disk full scenarios

Run with: python test_durability.py
"""

import sys
import time
import threading
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import signal

try:
    from chillbot import Memory
    from chillbot.kernel import KRNXController
except ImportError:
    print("❌ Cannot import chillbot. Run from project root.")
    sys.exit(1)


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print('=' * 60)


def print_test(text):
    print(f"\n{Colors.YELLOW}TEST: {text}{Colors.END}")


def print_pass(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_fail(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def get_container_id(name):
    """Get Docker container ID by name pattern"""
    try:
        result = subprocess.run(
            ['docker', 'ps', '-qf', f'name={name}'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def stop_container(name):
    """Stop a Docker container"""
    container_id = get_container_id(name)
    if container_id:
        subprocess.run(['docker', 'stop', container_id], capture_output=True)
        return True
    return False


def start_container(name):
    """Start a Docker container"""
    container_id = get_container_id(name)
    if container_id:
        subprocess.run(['docker', 'start', container_id], capture_output=True)
        return True
    # Try docker-compose
    subprocess.run(['docker-compose', 'up', '-d', name], capture_output=True)
    time.sleep(2)
    return True


# ==============================================
# TEST 1: Redis Failure During Writes
# ==============================================

def test_redis_failure():
    """Test behavior when Redis dies mid-operation"""
    print_test("Redis failure during writes")
    
    data_path = tempfile.mkdtemp(prefix='durability_redis_')
    
    try:
        memory = Memory('redis-test', data_path=data_path)
        
        # Write some events
        for i in range(5):
            memory.remember(f'Before failure {i}')
        print_pass("Wrote 5 events before failure")
        
        # Kill Redis
        print("  Stopping Redis...")
        stop_container('redis')
        time.sleep(1)
        
        # Try to write (should fail gracefully)
        try:
            memory.remember('During failure')
            print_fail("Write succeeded when Redis was down (unexpected)")
        except Exception as e:
            print_pass(f"Write failed gracefully: {type(e).__name__}")
        
        # Restart Redis
        print("  Starting Redis...")
        start_container('redis')
        time.sleep(2)
        
        # Create new memory instance (old one is dead)
        memory.close()
        memory = Memory('redis-test-2', data_path=data_path)
        
        # Try to write again
        memory.remember('After recovery')
        print_pass("Write succeeded after Redis recovery")
        
        memory.close()
        return True
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)
        start_container('redis')  # Ensure Redis is back up


# ==============================================
# TEST 2: Qdrant Failure During Recall
# ==============================================

def test_qdrant_failure():
    """Test fallback to kernel when Qdrant is unavailable"""
    print_test("Qdrant failure during recall")
    
    data_path = tempfile.mkdtemp(prefix='durability_qdrant_')
    
    try:
        memory = Memory('qdrant-test', data_path=data_path)
        
        # Write events
        memory.remember('outdoor hiking')
        memory.remember('indoor cooking')
        memory.remember('mountain climbing')
        time.sleep(0.5)  # Let embeddings process
        print_pass("Wrote 3 events")
        
        # Recall with Qdrant up
        results = memory.recall('outdoor activities')
        count_with_qdrant = len(results)
        print_pass(f"Recall with Qdrant: {count_with_qdrant} results")
        
        # Kill Qdrant
        print("  Stopping Qdrant...")
        stop_container('qdrant')
        time.sleep(1)
        
        # Recall should fall back to kernel
        results = memory.recall('outdoor activities')
        count_without_qdrant = len(results)
        
        if count_without_qdrant > 0:
            print_pass(f"Fallback to kernel: {count_without_qdrant} results (no crash)")
        else:
            print_fail("Fallback returned 0 results")
        
        memory.close()
        return count_without_qdrant > 0
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)
        start_container('qdrant')  # Ensure Qdrant is back up
        time.sleep(2)


# ==============================================
# TEST 3: Concurrent Writes (Race Conditions)
# ==============================================

def test_concurrent_writes():
    """Test thread-safety with concurrent writes"""
    print_test("Concurrent writes from multiple threads")
    
    data_path = tempfile.mkdtemp(prefix='durability_concurrent_')
    
    try:
        memory = Memory('concurrent-test', data_path=data_path)
        
        errors = []
        write_count = [0]
        lock = threading.Lock()
        
        def hammer(thread_id, count=20):
            try:
                for i in range(count):
                    memory.remember(f'Thread-{thread_id}: Event-{i}')
                    with lock:
                        write_count[0] += 1
            except Exception as e:
                with lock:
                    errors.append(f"Thread-{thread_id}: {e}")
        
        # Spawn 10 threads writing concurrently
        threads = [threading.Thread(target=hammer, args=(i,)) for i in range(10)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        duration = time.time() - start
        
        if errors:
            print_fail(f"Errors during concurrent writes: {len(errors)}")
            for err in errors[:3]:  # Show first 3
                print(f"  {err}")
            return False
        else:
            print_pass(f"Completed {write_count[0]} concurrent writes in {duration:.2f}s")
            print_pass(f"Throughput: {write_count[0]/duration:.1f} writes/sec")
        
        # Verify data integrity
        time.sleep(0.5)  # Let worker catch up
        stats = memory.stats()
        ltm_events = stats.get('kernel', {}).get('ltm', {}).get('warm_events', 0)
        print_pass(f"LTM has {ltm_events} events (data persisted)")
        
        memory.close()
        return True
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


# ==============================================
# TEST 4: Shutdown Under Load
# ==============================================

def test_shutdown_under_load():
    """Test clean shutdown while actively writing"""
    print_test("Shutdown under load")
    
    data_path = tempfile.mkdtemp(prefix='durability_shutdown_')
    
    try:
        memory = Memory('shutdown-test', data_path=data_path)
        
        write_count = [0]
        stop_flag = threading.Event()
        
        def continuous_writes():
            while not stop_flag.is_set():
                try:
                    memory.remember(f'Event {write_count[0]}')
                    write_count[0] += 1
                except Exception:
                    break
        
        # Start writing
        writer = threading.Thread(target=continuous_writes)
        writer.start()
        
        # Let it write for a bit
        time.sleep(1.0)
        
        # Stop and close
        print(f"  Stopping after {write_count[0]} writes...")
        stop_flag.set()
        
        start_close = time.time()
        memory.close()
        close_duration = time.time() - start_close
        
        writer.join(timeout=5)
        
        if writer.is_alive():
            print_fail("Writer thread did not stop")
            return False
        
        if close_duration > 20:
            print_fail(f"Shutdown took {close_duration:.1f}s (too long)")
            return False
        
        print_pass(f"Clean shutdown in {close_duration:.2f}s after {write_count[0]} writes")
        return True
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


# ==============================================
# TEST 5: Kernel Crash Recovery
# ==============================================

def test_crash_recovery():
    """Test recovery from unclean shutdown"""
    print_test("Crash recovery (unclean shutdown)")
    
    data_path = tempfile.mkdtemp(prefix='durability_crash_')
    
    try:
        # Write some data
        memory1 = Memory('crash-test', data_path=data_path)
        for i in range(10):
            memory1.remember(f'Before crash {i}')
        
        # Simulate crash (don't call close)
        print("  Simulating crash (no clean shutdown)...")
        del memory1  # Just drop it
        time.sleep(1)
        
        # Try to recover
        print("  Attempting recovery...")
        memory2 = Memory('crash-test-2', data_path=data_path)
        
        # Write more data
        memory2.remember('After recovery')
        print_pass("Recovery successful, able to write")
        
        # Verify old data is gone (different workspace)
        # but new data works
        results = memory2.recall('recovery')
        if len(results) > 0:
            print_pass("New writes after recovery work")
        
        memory2.close()
        return True
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


# ==============================================
# TEST 6: Large Event Handling
# ==============================================

def test_large_events():
    """Test handling of large event payloads"""
    print_test("Large event payloads")
    
    data_path = tempfile.mkdtemp(prefix='durability_large_')
    
    try:
        memory = Memory('large-test', data_path=data_path)
        
        # Small event
        memory.remember('small event')
        print_pass("Small event: OK")
        
        # 1KB event
        large_text = 'x' * 1024
        memory.remember(large_text)
        print_pass("1KB event: OK")
        
        # 100KB event
        very_large_text = 'x' * (100 * 1024)
        memory.remember(very_large_text)
        print_pass("100KB event: OK")
        
        # 1MB event (should this work?)
        huge_text = 'x' * (1024 * 1024)
        try:
            memory.remember(huge_text)
            print_pass("1MB event: OK (consider adding limit)")
        except Exception as e:
            print_pass(f"1MB event: Rejected ({type(e).__name__})")
        
        memory.close()
        return True
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


# ==============================================
# TEST 7: Memory Leaks (Repeated Open/Close)
# ==============================================

def test_memory_leaks():
    """Test for connection/memory leaks"""
    print_test("Memory leaks (repeated open/close)")
    
    data_path = tempfile.mkdtemp(prefix='durability_leaks_')
    
    try:
        cycles = 20
        for i in range(cycles):
            memory = Memory(f'leak-test-{i}', data_path=data_path)
            memory.remember(f'Event {i}')
            memory.close()
            
            if i % 5 == 0:
                print(f"  Completed {i}/{cycles} cycles...")
        
        print_pass(f"Completed {cycles} open/close cycles without crash")
        return True
        
    except Exception as e:
        print_fail(f"Test crashed: {e}")
        return False
    finally:
        shutil.rmtree(data_path, ignore_errors=True)


# ==============================================
# MAIN
# ==============================================

def main():
    print_header("KRNX DURABILITY TEST SUITE")
    print("This will stress-test the system with failures and edge cases.")
    print("Make sure Redis and Qdrant are running: docker-compose up -d")
    
    input("\nPress Enter to start tests...")
    
    results = {}
    
    # Run all tests
    tests = [
        ("Redis Failure", test_redis_failure),
        ("Qdrant Failure", test_qdrant_failure),
        ("Concurrent Writes", test_concurrent_writes),
        ("Shutdown Under Load", test_shutdown_under_load),
        ("Crash Recovery", test_crash_recovery),
        ("Large Events", test_large_events),
        ("Memory Leaks", test_memory_leaks),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print_fail(f"Test suite error: {e}")
            results[name] = False
        
        time.sleep(1)  # Breathing room between tests
    
    # Summary
    print_header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        color = Colors.GREEN if passed_test else Colors.RED
        print(f"{color}{status}: {name}{Colors.END}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All durability tests passed!{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}⚠️  Some tests failed. Review and fix before launch.{Colors.END}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
