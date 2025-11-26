#!/usr/bin/env python3
"""
Minimal diagnostic script to find where the freeze occurs.
Run this INSTEAD of stress_test.py to isolate the issue.
"""

import sys
import time
import threading

print("[1] Starting diagnostic...")

# Test imports
print("[2] Importing Memory class...")
from chillbot.memory import Memory
print("[2] ✓ Import OK")

# Test Memory creation
print("[3] Creating Memory instance...")
start = time.time()
memory = Memory(agent_id="diag-test")
print(f"[3] ✓ Memory created in {time.time()-start:.2f}s")

# Test single write (main thread)
print("[4] Testing single remember() on main thread...")
start = time.time()
try:
    event_id = memory.remember(content="test event 1", metadata={"test": True})
    print(f"[4] ✓ Single write OK in {time.time()-start:.3f}s - event_id={event_id}")
except Exception as e:
    print(f"[4] ✗ Single write FAILED: {e}")
    sys.exit(1)

# Test second write (confirm not one-time issue)
print("[5] Testing second remember()...")
start = time.time()
try:
    event_id = memory.remember(content="test event 2", metadata={"test": True})
    print(f"[5] ✓ Second write OK in {time.time()-start:.3f}s")
except Exception as e:
    print(f"[5] ✗ Second write FAILED: {e}")

# Test threaded write (single thread)
print("[6] Testing threaded write (1 thread)...")
result = {"ok": False, "error": None, "time": 0}

def single_thread_test():
    start = time.time()
    try:
        memory.remember(content="thread test", metadata={"thread": True})
        result["ok"] = True
        result["time"] = time.time() - start
    except Exception as e:
        result["error"] = str(e)

t = threading.Thread(target=single_thread_test)
t.start()
t.join(timeout=10)

if t.is_alive():
    print("[6] ✗ Thread HUNG (still running after 10s)")
elif result["ok"]:
    print(f"[6] ✓ Thread write OK in {result['time']:.3f}s")
else:
    print(f"[6] ✗ Thread write FAILED: {result['error']}")

# Test 5 concurrent threads
print("[7] Testing 5 concurrent threads...")
results = []
results_lock = threading.Lock()

def multi_thread_test(thread_id):
    start = time.time()
    try:
        memory.remember(content=f"multi thread {thread_id}", metadata={"tid": thread_id})
        with results_lock:
            results.append(("ok", thread_id, time.time()-start))
    except Exception as e:
        with results_lock:
            results.append(("error", thread_id, str(e)))

threads = []
for i in range(5):
    t = threading.Thread(target=multi_thread_test, args=(i,))
    threads.append(t)

start = time.time()
for t in threads:
    t.start()

# Wait with timeout
for t in threads:
    t.join(timeout=30)

hung = [t for t in threads if t.is_alive()]
if hung:
    print(f"[7] ✗ {len(hung)} threads HUNG after 30s")
else:
    ok_count = sum(1 for r in results if r[0] == "ok")
    total_time = time.time() - start
    print(f"[7] ✓ {ok_count}/5 threads completed in {total_time:.2f}s")

# Test 50 threads (the actual stress scenario)
print("[8] Testing 50 concurrent threads (5 events each)...")
results2 = []
results2_lock = threading.Lock()
completed = {"count": 0}

def stress_test(thread_id):
    for i in range(5):
        try:
            memory.remember(content=f"stress {thread_id}-{i}", metadata={"tid": thread_id, "seq": i})
            with results2_lock:
                completed["count"] += 1
        except Exception as e:
            with results2_lock:
                results2.append(f"T{thread_id}: {e}")

threads2 = []
for i in range(50):
    t = threading.Thread(target=stress_test, args=(i,))
    threads2.append(t)

print("[8] Starting threads...")
start = time.time()
for t in threads2:
    t.start()

# Progress check every 5 seconds
deadline = time.time() + 60  # 1 minute timeout
while time.time() < deadline:
    time.sleep(5)
    alive = sum(1 for t in threads2 if t.is_alive())
    print(f"[8] Progress: {completed['count']}/250 events, {alive} threads still running")
    if alive == 0:
        break

# Final check
for t in threads2:
    t.join(timeout=1)

hung2 = [t for t in threads2 if t.is_alive()]
if hung2:
    print(f"[8] ✗ {len(hung2)} threads HUNG")
else:
    errors = len(results2)
    total_time = time.time() - start
    print(f"[8] ✓ {completed['count']}/250 events, {errors} errors, {total_time:.2f}s")

# Cleanup
print("\n[9] Closing memory...")
memory.close()
print("[9] ✓ Closed")

print("\n" + "="*50)
print("DIAGNOSTIC COMPLETE")
print("="*50)
