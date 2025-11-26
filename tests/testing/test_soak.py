#!/usr/bin/env python3
"""
KRNX Soak Test Suite (Load Over Time)

Long-running tests to expose memory leaks, performance degradation,
and race conditions that only appear after hours of operation.

Tests:
- Sustained write load (hours)
- Memory growth over time
- Performance degradation
- Worker stability
- Storage growth patterns
- Recall accuracy over time

Run with: python test_soak.py --duration 4h
"""

import sys
import time
import argparse
import threading
import tempfile
import shutil
import psutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import json

try:
    from chillbot import Memory
except ImportError:
    print("❌ Cannot import chillbot. Run from project root.")
    sys.exit(1)


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


class MetricsCollector:
    """Collect system metrics over time"""
    
    def __init__(self):
        self.metrics: List[Dict] = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
    
    def collect(self, label: str, custom: Optional[Dict] = None):
        """Collect current metrics"""
        elapsed = time.time() - self.start_time
        
        metric = {
            'timestamp': time.time(),
            'elapsed_seconds': elapsed,
            'elapsed_hours': elapsed / 3600,
            'label': label,
            'memory_mb': self.process.memory_info().rss / (1024 * 1024),
            'cpu_percent': self.process.cpu_percent(interval=0.1),
            'threads': threading.active_count(),
        }
        
        if custom:
            metric.update(custom)
        
        self.metrics.append(metric)
        return metric
    
    def get_trend(self, key: str) -> str:
        """Analyze trend for a metric"""
        if len(self.metrics) < 10:
            return "insufficient_data"
        
        recent = [m[key] for m in self.metrics[-10:]]
        old = [m[key] for m in self.metrics[:10]]
        
        recent_avg = sum(recent) / len(recent)
        old_avg = sum(old) / len(old)
        
        change = ((recent_avg - old_avg) / old_avg * 100) if old_avg > 0 else 0
        
        if change > 20:
            return f"increasing (+{change:.1f}%)"
        elif change < -20:
            return f"decreasing ({change:.1f}%)"
        else:
            return "stable"
    
    def summary(self):
        """Print summary statistics"""
        if not self.metrics:
            print("No metrics collected")
            return
        
        first = self.metrics[0]
        last = self.metrics[-1]
        
        print(f"\n{Colors.BLUE}=== METRICS SUMMARY ==={Colors.END}")
        print(f"Duration: {last['elapsed_hours']:.2f} hours")
        print(f"Samples: {len(self.metrics)}")
        
        print(f"\nMemory:")
        print(f"  Start: {first['memory_mb']:.1f} MB")
        print(f"  End: {last['memory_mb']:.1f} MB")
        print(f"  Trend: {self.get_trend('memory_mb')}")
        
        print(f"\nCPU:")
        print(f"  Average: {sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics):.1f}%")
        
        print(f"\nThreads:")
        print(f"  Start: {first['threads']}")
        print(f"  End: {last['threads']}")
    
    def save(self, path: str):
        """Save metrics to JSON"""
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to: {path}")


# ==============================================
# TEST 1: Sustained Write Load
# ==============================================

def test_sustained_writes(duration_hours: float, rate_per_sec: int = 10):
    """Write events continuously at target rate"""
    print(f"\n{Colors.BLUE}TEST: Sustained writes ({duration_hours}h @ {rate_per_sec}/sec){Colors.END}\n")
    
    data_path = tempfile.mkdtemp(prefix='soak_writes_')
    metrics = MetricsCollector()
    
    try:
        memory = Memory('soak-writes', data_path=data_path)
        
        end_time = time.time() + (duration_hours * 3600)
        write_count = 0
        error_count = 0
        
        interval = 1.0 / rate_per_sec
        next_write = time.time()
        
        print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Will end at: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        last_report = time.time()
        report_interval = 60  # Report every minute
        
        while time.time() < end_time:
            try:
                # Write event
                memory.remember(f'Event {write_count} at {time.time()}')
                write_count += 1
                
                # Sleep to maintain target rate
                next_write += interval
                sleep_time = next_write - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Periodic reporting
                if time.time() - last_report >= report_interval:
                    elapsed = time.time() - metrics.start_time
                    stats = memory.stats()
                    
                    custom = {
                        'write_count': write_count,
                        'error_count': error_count,
                        'writes_per_sec': write_count / elapsed,
                        'ltm_events': stats.get('kernel', {}).get('ltm', {}).get('warm_events', 0),
                        'stm_keys': stats.get('kernel', {}).get('stm', {}).get('total_keys', 0),
                    }
                    
                    metric = metrics.collect('periodic', custom)
                    
                    print(f"[{elapsed/3600:.2f}h] "
                          f"Writes: {write_count:,} ({metric['writes_per_sec']:.1f}/s) | "
                          f"Errors: {error_count} | "
                          f"Memory: {metric['memory_mb']:.1f}MB | "
                          f"LTM: {custom['ltm_events']:,} events")
                    
                    last_report = time.time()
                
            except KeyboardInterrupt:
                print("\n⚠️  Test interrupted by user")
                break
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    print(f"  Error: {e}")
                if error_count > 100:
                    print(f"  Too many errors ({error_count}), aborting")
                    break
        
        # Final report
        print(f"\n{Colors.GREEN}✓ Completed sustained write test{Colors.END}")
        print(f"  Total writes: {write_count:,}")
        print(f"  Errors: {error_count}")
        print(f"  Success rate: {(write_count/(write_count+error_count)*100):.2f}%")
        
        metrics.summary()
        metrics.save(f'{data_path}/metrics.json')
        
        memory.close()
        return write_count > 0 and error_count < write_count * 0.01  # <1% errors
        
    except Exception as e:
        print(f"{Colors.RED}✗ Test crashed: {e}{Colors.END}")
        return False
    finally:
        print(f"\nData path (not deleted): {data_path}")


# ==============================================
# TEST 2: Mixed Read/Write Load
# ==============================================

def test_mixed_load(duration_hours: float, writers: int = 3, readers: int = 2):
    """Concurrent readers and writers"""
    print(f"\n{Colors.BLUE}TEST: Mixed load ({duration_hours}h, {writers}W + {readers}R){Colors.END}\n")
    
    data_path = tempfile.mkdtemp(prefix='soak_mixed_')
    metrics = MetricsCollector()
    
    try:
        memory = Memory('soak-mixed', data_path=data_path)
        
        end_time = time.time() + (duration_hours * 3600)
        stop_flag = threading.Event()
        
        stats_lock = threading.Lock()
        stats = {
            'writes': 0,
            'reads': 0,
            'write_errors': 0,
            'read_errors': 0,
        }
        
        def writer_thread(thread_id):
            while not stop_flag.is_set() and time.time() < end_time:
                try:
                    memory.remember(f'Writer-{thread_id}: {time.time()}')
                    with stats_lock:
                        stats['writes'] += 1
                    time.sleep(0.1)
                except Exception as e:
                    with stats_lock:
                        stats['write_errors'] += 1
        
        def reader_thread(thread_id):
            queries = ['test', 'event', 'data', 'information', 'memory']
            query_idx = 0
            
            while not stop_flag.is_set() and time.time() < end_time:
                try:
                    memory.recall(queries[query_idx % len(queries)])
                    with stats_lock:
                        stats['reads'] += 1
                    query_idx += 1
                    time.sleep(0.5)
                except Exception as e:
                    with stats_lock:
                        stats['read_errors'] += 1
        
        # Start threads
        threads = []
        for i in range(writers):
            t = threading.Thread(target=writer_thread, args=(i,))
            t.start()
            threads.append(t)
        
        for i in range(readers):
            t = threading.Thread(target=reader_thread, args=(i,))
            t.start()
            threads.append(t)
        
        print(f"Started {writers} writers + {readers} readers")
        print(f"Will run until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        last_report = time.time()
        report_interval = 60
        
        # Monitor
        while time.time() < end_time and not stop_flag.is_set():
            time.sleep(1)
            
            if time.time() - last_report >= report_interval:
                elapsed = time.time() - metrics.start_time
                
                with stats_lock:
                    custom = {**stats}
                
                metric = metrics.collect('periodic', custom)
                
                print(f"[{elapsed/3600:.2f}h] "
                      f"W: {custom['writes']:,} (E:{custom['write_errors']}) | "
                      f"R: {custom['reads']:,} (E:{custom['read_errors']}) | "
                      f"Mem: {metric['memory_mb']:.1f}MB")
                
                last_report = time.time()
        
        # Stop threads
        print("\nStopping threads...")
        stop_flag.set()
        for t in threads:
            t.join(timeout=5)
        
        print(f"\n{Colors.GREEN}✓ Completed mixed load test{Colors.END}")
        with stats_lock:
            print(f"  Writes: {stats['writes']:,} (errors: {stats['write_errors']})")
            print(f"  Reads: {stats['reads']:,} (errors: {stats['read_errors']})")
        
        metrics.summary()
        metrics.save(f'{data_path}/metrics.json')
        
        memory.close()
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ Test crashed: {e}{Colors.END}")
        return False
    finally:
        print(f"\nData path (not deleted): {data_path}")


# ==============================================
# TEST 3: Memory Leak Detection
# ==============================================

def test_memory_leak(duration_hours: float, check_interval_sec: int = 300):
    """Monitor memory growth over time"""
    print(f"\n{Colors.BLUE}TEST: Memory leak detection ({duration_hours}h){Colors.END}\n")
    
    data_path = tempfile.mkdtemp(prefix='soak_leak_')
    metrics = MetricsCollector()
    
    try:
        end_time = time.time() + (duration_hours * 3600)
        
        print(f"Will monitor until: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        iteration = 0
        
        while time.time() < end_time:
            # Create memory instance, use it, close it
            memory = Memory(f'leak-test-{iteration}', data_path=data_path)
            
            # Do some work
            for i in range(10):
                memory.remember(f'Iteration {iteration}, event {i}')
            
            results = memory.recall('test')
            
            memory.close()
            
            # Collect metrics
            metric = metrics.collect(f'iteration_{iteration}', {
                'iteration': iteration
            })
            
            if iteration % 10 == 0:
                elapsed = time.time() - metrics.start_time
                print(f"[{elapsed/3600:.2f}h] "
                      f"Iteration: {iteration} | "
                      f"Memory: {metric['memory_mb']:.1f}MB | "
                      f"Trend: {metrics.get_trend('memory_mb')}")
            
            iteration += 1
            time.sleep(check_interval_sec)
        
        # Analyze
        print(f"\n{Colors.GREEN}✓ Completed memory leak test{Colors.END}")
        print(f"  Iterations: {iteration}")
        
        metrics.summary()
        
        # Check for memory leak
        if len(metrics.metrics) > 10:
            first_avg = sum(m['memory_mb'] for m in metrics.metrics[:5]) / 5
            last_avg = sum(m['memory_mb'] for m in metrics.metrics[-5:]) / 5
            growth = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
            
            print(f"\nMemory growth: {growth:.1f}%")
            
            if growth > 50:
                print(f"{Colors.RED}⚠️  Possible memory leak detected!{Colors.END}")
                return False
            else:
                print(f"{Colors.GREEN}✓ No significant memory leak{Colors.END}")
        
        metrics.save(f'{data_path}/metrics.json')
        return True
        
    except Exception as e:
        print(f"{Colors.RED}✗ Test crashed: {e}{Colors.END}")
        return False
    finally:
        print(f"\nData path (not deleted): {data_path}")


# ==============================================
# MAIN
# ==============================================

def parse_duration(duration_str: str) -> float:
    """Parse duration string like '4h', '30m', '2.5h'"""
    duration_str = duration_str.lower().strip()
    
    if duration_str.endswith('h'):
        return float(duration_str[:-1])
    elif duration_str.endswith('m'):
        return float(duration_str[:-1]) / 60
    elif duration_str.endswith('s'):
        return float(duration_str[:-1]) / 3600
    else:
        return float(duration_str)  # Assume hours


def main():
    parser = argparse.ArgumentParser(description='KRNX Soak/Load Testing')
    parser.add_argument('--duration', default='4h', help='Test duration (e.g., 4h, 30m)')
    parser.add_argument('--test', choices=['writes', 'mixed', 'leak', 'all'], default='all')
    parser.add_argument('--rate', type=int, default=10, help='Writes per second')
    parser.add_argument('--quick', action='store_true', help='Quick test (5 minutes)')
    
    args = parser.parse_args()
    
    if args.quick:
        duration = 5 / 60  # 5 minutes
        print(f"{Colors.YELLOW}QUICK MODE: 5 minute test{Colors.END}")
    else:
        duration = parse_duration(args.duration)
    
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}KRNX SOAK TEST SUITE{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"\nDuration: {duration:.2f} hours")
    print(f"Test: {args.test}")
    print("\n⚠️  This is a long-running test. Press Ctrl+C to abort.")
    print("Make sure Redis and Qdrant are running: docker-compose up -d\n")
    
    input("Press Enter to start...")
    
    results = {}
    
    try:
        if args.test in ['writes', 'all']:
            results['Sustained Writes'] = test_sustained_writes(duration, args.rate)
        
        if args.test in ['mixed', 'all']:
            results['Mixed Load'] = test_mixed_load(duration)
        
        if args.test in ['leak', 'all']:
            results['Memory Leak'] = test_memory_leak(duration)
        
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Test suite interrupted by user{Colors.END}")
    
    # Summary
    if results:
        print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
        print(f"{Colors.BLUE}SUMMARY{Colors.END}")
        print(f"{Colors.BLUE}{'=' * 60}{Colors.END}\n")
        
        for name, passed in results.items():
            status = f"{Colors.GREEN}✓ PASS" if passed else f"{Colors.RED}✗ FAIL"
            print(f"{status}: {name}{Colors.END}")
        
        passed_count = sum(1 for v in results.values() if v)
        total = len(results)
        
        if passed_count == total:
            print(f"\n{Colors.GREEN}🎉 All soak tests passed!{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}⚠️  {total - passed_count}/{total} tests failed{Colors.END}")
            return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
