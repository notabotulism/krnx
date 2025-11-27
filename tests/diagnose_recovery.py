#!/usr/bin/env python3
"""
Backpressure Recovery Test Diagnostic

This script replicates the failing test with detailed logging to see:
1. Is the worker actually processing?
2. Is XLEN decreasing over time?
3. What happens when we delete the queue mid-test?
4. Why isn't backpressure recovering?
"""

import time
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Test parameters (same as stress test)
BP_QUEUE_DEPTH = 500
BP_LAG_SECONDS = 5.0


def create_test_event(worker_id: int, sequence: int):
    """Create a test event."""
    from chillbot.kernel.models import Event
    
    return Event(
        event_id=f"evt_{uuid.uuid4().hex[:16]}",
        workspace_id="stress_test",
        user_id=f"user_{worker_id}",
        session_id=f"stress_test_user_{worker_id}",
        content={'text': f"Worker {worker_id} event {sequence}", 'data': 'x' * 50},
        timestamp=time.time(),
    )


def run_write_worker(controller, worker_id: int, num_events: int):
    """Write events, return (successes, bp_rejections)."""
    from chillbot.kernel.controller import BackpressureError
    
    successes = 0
    bp_rejections = 0
    
    for i in range(num_events):
        event = create_test_event(worker_id, i)
        try:
            controller.write_event("stress_test", f"user_{worker_id}", event)
            successes += 1
        except BackpressureError:
            bp_rejections += 1
        except Exception as e:
            pass
    
    return successes, bp_rejections


def monitor_queue(redis_client, duration_sec: float, interval: float = 0.2):
    """Monitor queue metrics over time."""
    print(f"\n{'Time':>6} | {'XLEN':>6} | {'Pending':>7} | {'Lag':>6}")
    print("-" * 40)
    
    start = time.time()
    samples = []
    
    while time.time() - start < duration_sec:
        try:
            xlen = redis_client.xlen('krnx:ltm:queue')
        except:
            xlen = 0
        
        pending = 0
        try:
            pending_info = redis_client.xpending('krnx:ltm:queue', 'krnx-ltm-workers')
            pending = pending_info.get('pending', 0) if pending_info else 0
        except:
            pass
        
        lag = 0.0
        try:
            stream_info = redis_client.xinfo_stream('krnx:ltm:queue')
            if stream_info.get('length', 0) > 0 and 'first-entry' in stream_info:
                first_entry = stream_info['first-entry']
                if first_entry:
                    msg_id = first_entry[0]
                    if isinstance(msg_id, bytes):
                        msg_id = msg_id.decode('utf-8')
                    msg_ts = int(msg_id.split('-')[0])
                    lag = (time.time() * 1000 - msg_ts) / 1000.0
        except:
            pass
        
        elapsed = time.time() - start
        print(f"{elapsed:>5.1f}s | {xlen:>6} | {pending:>7} | {lag:>5.1f}s")
        samples.append({'time': elapsed, 'xlen': xlen, 'pending': pending, 'lag': lag})
        
        time.sleep(interval)
    
    return samples


def main():
    print("=" * 60)
    print("BACKPRESSURE RECOVERY DIAGNOSTIC")
    print("=" * 60)
    
    from chillbot.kernel.controller import KRNXController, BackpressureError
    from chillbot.kernel.connection_pool import get_redis_client, close_pool
    
    # Clean start
    try:
        close_pool()
    except:
        pass
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n[1] Creating controller with BP thresholds:")
        print(f"    max_queue_depth = {BP_QUEUE_DEPTH}")
        print(f"    max_lag_seconds = {BP_LAG_SECONDS}")
        print(f"    worker_block_ms = 50")
        
        controller = KRNXController(
            data_path=tmpdir,
            redis_host='localhost',
            redis_port=6379,
            enable_backpressure=True,
            max_queue_depth=BP_QUEUE_DEPTH,
            max_lag_seconds=BP_LAG_SECONDS,
            enable_async_worker=True,
            worker_block_ms=50,  # Same as test
        )
        
        redis_client = get_redis_client()
        
        # Check worker is running
        print(f"\n[2] Worker status: {'RUNNING' if controller.worker_running() else 'NOT RUNNING'}")
        
        # Clean queue
        print(f"\n[3] Cleaning queue...")
        redis_client.delete('krnx:ltm:queue')
        controller.reset_backpressure()
        time.sleep(0.5)
        
        # Check worker after delete
        print(f"    Worker status after delete: {'RUNNING' if controller.worker_running() else 'NOT RUNNING'}")
        
        # Start monitoring in background
        print(f"\n[4] Starting queue monitor...")
        stop_monitor = threading.Event()
        monitor_samples = []
        
        def monitor_thread():
            nonlocal monitor_samples
            monitor_samples = monitor_queue(redis_client, duration_sec=10, interval=0.5)
        
        monitor = threading.Thread(target=monitor_thread)
        monitor.start()
        
        # Phase 1: Flood
        print(f"\n[5] PHASE 1: Flooding with 1500 events (30 workers x 50)...")
        flood_start = time.time()
        
        with ThreadPoolExecutor(max_workers=30) as executor:
            futures = [
                executor.submit(run_write_worker, controller, wid, 50)
                for wid in range(30)
            ]
            results = [f.result() for f in as_completed(futures)]
        
        total_success = sum(r[0] for r in results)
        total_bp = sum(r[1] for r in results)
        flood_duration = time.time() - flood_start
        
        print(f"    Flood complete in {flood_duration:.2f}s")
        print(f"    Successes: {total_success}, BP rejections: {total_bp}")
        
        # Check metrics right after flood
        metrics = controller.get_worker_metrics()
        print(f"\n[6] Metrics after flood:")
        print(f"    queue_depth (XLEN): {metrics.queue_depth}")
        print(f"    pending_depth (XPENDING): {metrics.pending_depth}")
        print(f"    lag_seconds: {metrics.lag_seconds:.2f}")
        print(f"    worker_running: {metrics.worker_running}")
        
        # Phase 2: Wait for drain
        print(f"\n[7] PHASE 2: Waiting 3s for queue to drain...")
        time.sleep(3)
        
        # Check metrics after wait
        metrics = controller.get_worker_metrics()
        print(f"\n[8] Metrics after 3s wait:")
        print(f"    queue_depth (XLEN): {metrics.queue_depth}")
        print(f"    pending_depth (XPENDING): {metrics.pending_depth}")  
        print(f"    lag_seconds: {metrics.lag_seconds:.2f}")
        print(f"    worker_running: {metrics.worker_running}")
        
        # Check backpressure state
        bp_engaged = metrics.queue_depth > BP_QUEUE_DEPTH or metrics.lag_seconds > BP_LAG_SECONDS
        print(f"\n    Backpressure should be: {'ENGAGED' if bp_engaged else 'DISENGAGED'}")
        print(f"    (queue_depth {metrics.queue_depth} > {BP_QUEUE_DEPTH}? {metrics.queue_depth > BP_QUEUE_DEPTH})")
        print(f"    (lag {metrics.lag_seconds:.1f} > {BP_LAG_SECONDS}? {metrics.lag_seconds > BP_LAG_SECONDS})")
        
        # Phase 3: Recovery test
        print(f"\n[9] PHASE 3: Testing recovery (50 writes with 20ms delay)...")
        recovery_successes = 0
        recovery_failures = 0
        
        for i in range(50):
            event = create_test_event(999, i)
            try:
                controller.write_event("stress_test", "recovery_test", event)
                recovery_successes += 1
            except BackpressureError:
                recovery_failures += 1
            except Exception as e:
                recovery_failures += 1
            time.sleep(0.02)
        
        recovery_rate = (recovery_successes / 50) * 100
        print(f"    Recovery: {recovery_successes}/50 ({recovery_rate:.1f}%)")
        
        # Wait for monitor to finish
        time.sleep(2)
        monitor.join(timeout=1)
        
        # Final metrics
        metrics = controller.get_worker_metrics()
        print(f"\n[10] Final metrics:")
        print(f"    queue_depth: {metrics.queue_depth}")
        print(f"    messages_processed: {metrics.messages_processed}")
        
        # Analysis
        print(f"\n{'='*60}")
        print("ANALYSIS")
        print("=" * 60)
        
        if recovery_rate < 60:
            print(f"\n❌ Recovery FAILED ({recovery_rate:.1f}% < 60%)")
            
            if metrics.queue_depth > BP_QUEUE_DEPTH:
                print(f"\n   ROOT CAUSE: Queue not draining fast enough")
                print(f"   Queue depth ({metrics.queue_depth}) still > threshold ({BP_QUEUE_DEPTH})")
                print(f"\n   Possible fixes:")
                print(f"   1. Increase worker speed (reduce worker_block_ms)")
                print(f"   2. Increase BP_QUEUE_DEPTH threshold")
                print(f"   3. Wait longer for drain")
                print(f"   4. Use more aggressive XTRIM")
            
            if not metrics.worker_running:
                print(f"\n   ROOT CAUSE: Worker stopped!")
                print(f"   The delete('krnx:ltm:queue') likely killed the consumer group")
        else:
            print(f"\n✅ Recovery PASSED ({recovery_rate:.1f}%)")
        
        # Cleanup
        print(f"\n[11] Shutting down...")
        controller.shutdown(timeout=5.0)
        
        print("\nDone!")


if __name__ == "__main__":
    main()
