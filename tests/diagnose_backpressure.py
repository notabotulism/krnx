#!/usr/bin/env python3
"""
KRNX Backpressure Diagnostic Script

This script helps diagnose why backpressure tests are failing.
Run this to see the actual queue metrics during a load test.

Usage:
    python diagnose_backpressure.py
    
Requirements:
    - Redis running on localhost:6379
    - chillbot package installed
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def diagnose():
    """Run diagnostic tests on backpressure behavior."""
    
    # Setup
    from chillbot.kernel.connection_pool import configure_pool, get_redis_client
    
    configure_pool(host='localhost', port=6379)
    redis_client = get_redis_client()
    
    print("=" * 60)
    print("KRNX BACKPRESSURE DIAGNOSTIC")
    print("=" * 60)
    
    # Clean start
    redis_client.delete('krnx:ltm:queue')
    try:
        redis_client.xgroup_destroy('krnx:ltm:queue', 'krnx-ltm-workers')
    except:
        pass
    
    # Create consumer group
    try:
        redis_client.xgroup_create('krnx:ltm:queue', 'krnx-ltm-workers', id='0', mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            raise
    
    print("\n[Phase 1: Adding messages to queue]")
    
    # Add 1000 messages
    for i in range(1000):
        redis_client.xadd('krnx:ltm:queue', {
            'event_id': f'evt_{i}',
            'event_json': f'{{"id": {i}}}'
        })
    
    # Check metrics
    xlen = redis_client.xlen('krnx:ltm:queue')
    pending_info = redis_client.xpending('krnx:ltm:queue', 'krnx-ltm-workers')
    pending = pending_info.get('pending', 0) if pending_info else 0
    
    print(f"\nAfter adding 1000 messages:")
    print(f"  XLEN (stream length):  {xlen}")
    print(f"  XPENDING (in-flight):  {pending}")
    print(f"  TRUE BACKLOG:          {xlen}")
    
    print("\n[Phase 2: Simulating worker reading batch]")
    
    # Worker reads 100 messages
    messages = redis_client.xreadgroup(
        'krnx-ltm-workers', 'test-worker',
        streams={'krnx:ltm:queue': '>'},
        count=100
    )
    
    # Check metrics again
    xlen = redis_client.xlen('krnx:ltm:queue')
    pending_info = redis_client.xpending('krnx:ltm:queue', 'krnx-ltm-workers')
    pending = pending_info.get('pending', 0) if pending_info else 0
    
    print(f"\nAfter worker reads 100 messages (but hasn't ACK'd):")
    print(f"  XLEN (stream length):  {xlen}")
    print(f"  XPENDING (in-flight):  {pending}")
    print(f"  TRUE BACKLOG:          {xlen}")
    
    print("\n[Phase 3: Worker ACKs the batch]")
    
    # ACK the messages
    if messages:
        msg_ids = [msg[0] for stream, msgs in messages for msg in msgs]
        if msg_ids:
            redis_client.xack('krnx:ltm:queue', 'krnx-ltm-workers', *msg_ids)
    
    # Check metrics again
    xlen = redis_client.xlen('krnx:ltm:queue')
    pending_info = redis_client.xpending('krnx:ltm:queue', 'krnx-ltm-workers')
    pending = pending_info.get('pending', 0) if pending_info else 0
    
    print(f"\nAfter ACK (without XTRIM):")
    print(f"  XLEN (stream length):  {xlen}")
    print(f"  XPENDING (in-flight):  {pending}")
    print(f"  TRUE BACKLOG:          {xlen}")  # Still 1000!
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)
    
    if pending == 0 and xlen > 0:
        print("""
⚠️  BUG CONFIRMED: Using XPENDING for queue_depth is WRONG!

Current behavior:
  queue_depth = XPENDING.pending  # = 0 (nothing in-flight)
  
But actual backlog:
  true_backlog = XLEN  # = {} messages waiting!

The backpressure check thinks the queue is empty when it's not.

FIX:
  In controller.py get_worker_metrics():
  - Change: queue_depth = XPENDING.pending
  - To:     queue_depth = XLEN
  
This ensures backpressure reflects the TRUE queue depth.
""".format(xlen))
    
    print("\n[Phase 4: XTRIM demonstration]")
    
    # XTRIM to simulate worker cleanup
    redis_client.xtrim('krnx:ltm:queue', maxlen=900, approximate=False)
    
    xlen = redis_client.xlen('krnx:ltm:queue')
    print(f"\nAfter XTRIM to 900:")
    print(f"  XLEN (stream length):  {xlen}")
    
    # Clean up
    redis_client.delete('krnx:ltm:queue')
    
    print("\n" + "=" * 60)
    print("RECOMMENDATION:")
    print("=" * 60)
    print("""
For backpressure to work correctly:

1. Use XLEN for queue_depth (not XPENDING)
   - XLEN = total messages needing processing
   - XPENDING = messages currently being processed

2. XTRIM after ACK to keep XLEN accurate
   - The worker already does this (maxlen=50000)
   - But the key insight is: XLEN is the metric that matters

3. Alternative: Track "last_delivered_id" position
   - More complex but more accurate
   - For KRNX, XLEN is sufficient

Apply the fix in controller.py:
  queue_depth = redis_client.xlen('krnx:ltm:queue')
""")


def test_with_controller():
    """Test with actual KRNX controller to show the issue."""
    
    import tempfile
    from chillbot.kernel.controller import KRNXController, BackpressureError
    from chillbot.kernel.connection_pool import close_pool, get_redis_client
    
    print("\n" + "=" * 60)
    print("TESTING WITH ACTUAL CONTROLLER")
    print("=" * 60)
    
    # Clean up
    try:
        close_pool()
    except:
        pass
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create controller with low thresholds
        controller = KRNXController(
            data_path=tmpdir,
            redis_host='localhost',
            redis_port=6379,
            enable_backpressure=True,
            max_queue_depth=100,      # Low threshold for testing
            max_lag_seconds=5.0,
            enable_async_worker=True,
            worker_block_ms=1000,      # Slow worker to build up backlog
        )
        
        redis_client = get_redis_client()
        
        # Clean queue
        redis_client.delete('krnx:ltm:queue')
        controller.reset_backpressure()
        
        # Pause worker to build backlog
        time.sleep(0.5)
        
        print("\n[Writing 500 events rapidly...]")
        
        from chillbot.kernel.models import Event
        import uuid
        
        successes = 0
        bp_errors = 0
        
        for i in range(500):
            event = Event(
                event_id=f"evt_{uuid.uuid4().hex[:16]}",
                workspace_id="diag_test",
                user_id="test_user",
                session_id="test_session",
                content={"index": i},
                timestamp=time.time(),
            )
            try:
                controller.write_event("diag_test", "test_user", event)
                successes += 1
            except BackpressureError:
                bp_errors += 1
        
        # Get metrics
        metrics = controller.get_worker_metrics()
        xlen = redis_client.xlen('krnx:ltm:queue')
        
        print(f"\nResults:")
        print(f"  Successes:           {successes}")
        print(f"  Backpressure errors: {bp_errors}")
        print(f"  ")
        print(f"  metrics.queue_depth: {metrics.queue_depth}")
        print(f"  Actual XLEN:         {xlen}")
        print(f"  Worker running:      {metrics.worker_running}")
        
        if metrics.queue_depth < xlen:
            print(f"\n⚠️  MISMATCH: queue_depth ({metrics.queue_depth}) != XLEN ({xlen})")
            print("   This means backpressure is checking the wrong metric!")
        
        # Cleanup
        controller.shutdown(timeout=5.0)


if __name__ == "__main__":
    try:
        diagnose()
        print("\n" + "=" * 60)
        print()
        test_with_controller()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Run: pip install -e . (from chillbot directory)")
