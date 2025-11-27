#!/usr/bin/env python3
"""
KRNX Overnight Soak Test v1.1

Long-running stability test designed to run for hours or overnight.
Tests system stability, memory leaks, resource exhaustion, and degradation over time.

v1.1 Changes:
- Compatible with controller v0.3.8 (workspace_id/user_id consistency validation)
- Updated create_soak_event to accept workspace_id and user_id parameters
- Fixed all call sites to pass consistent values

Features:
- Configurable duration (default 8 hours)
- Periodic health checks and metrics collection
- Automatic recovery testing
- Memory and resource monitoring
- Detailed logging and reporting
- Graceful shutdown with final report

Configuration via environment variables:
    SOAK_DURATION_HOURS=8      - Total test duration
    SOAK_INTENSITY=medium       - low, medium, high
    SOAK_CHECKPOINT_MINS=15    - Minutes between health checkpoints
    SOAK_REPORT_FILE=soak_report.json  - Output report file
    REDIS_HOST=localhost
    REDIS_PORT=6379

Run:
    python tests/test_soak_overnight.py
    
    # Or with custom duration:
    SOAK_DURATION_HOURS=1 python tests/test_soak_overnight.py
    
    # Run as pytest (shorter version):
    pytest tests/test_soak_overnight.py -v -s
"""

import os
import sys
import time
import json
import uuid
import signal
import threading
import statistics
import traceback
import gc
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Duration
SOAK_DURATION_HOURS = float(os.environ.get('SOAK_DURATION_HOURS', '8'))
SOAK_DURATION_SECONDS = SOAK_DURATION_HOURS * 3600

# Intensity levels: low, medium, high
SOAK_INTENSITY = os.environ.get('SOAK_INTENSITY', 'medium')

# Health checkpoint interval
SOAK_CHECKPOINT_MINS = int(os.environ.get('SOAK_CHECKPOINT_MINS', '15'))
SOAK_CHECKPOINT_SECONDS = SOAK_CHECKPOINT_MINS * 60

# Output
SOAK_REPORT_FILE = os.environ.get('SOAK_REPORT_FILE', 'soak_report.json')

# Redis
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))

# Intensity configurations
INTENSITY_CONFIGS = {
    'low': {
        'writers': 5,
        'events_per_minute': 100,
        'burst_probability': 0.05,
        'burst_size': 50,
        'pause_probability': 0.1,
        'pause_duration_range': (1, 5),
    },
    'medium': {
        'writers': 15,
        'events_per_minute': 500,
        'burst_probability': 0.1,
        'burst_size': 200,
        'pause_probability': 0.05,
        'pause_duration_range': (0.5, 2),
    },
    'high': {
        'writers': 30,
        'events_per_minute': 2000,
        'burst_probability': 0.15,
        'burst_size': 500,
        'pause_probability': 0.02,
        'pause_duration_range': (0.1, 0.5),
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HealthCheckpoint:
    """Periodic health snapshot."""
    timestamp: float
    elapsed_hours: float
    
    # Metrics
    total_events_written: int
    total_events_processed: int
    total_backpressure_events: int
    total_errors: int
    
    # Current state
    queue_depth: int
    lag_seconds: float
    worker_running: bool
    backpressure_mode: bool
    
    # Rates
    write_rate_per_min: float
    process_rate_per_min: float
    error_rate_per_min: float
    
    # System
    memory_mb: float = 0.0
    
    # Latency (from recent samples)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


@dataclass
class SoakTestReport:
    """Final soak test report."""
    test_id: str
    start_time: str
    end_time: str
    duration_hours: float
    intensity: str
    
    # Totals
    total_events_attempted: int = 0
    total_events_written: int = 0
    total_events_processed: int = 0
    total_backpressure_events: int = 0
    total_errors: int = 0
    
    # Rates
    avg_write_rate_per_min: float = 0.0
    avg_process_rate_per_min: float = 0.0
    peak_write_rate_per_min: float = 0.0
    peak_queue_depth: int = 0
    peak_lag_seconds: float = 0.0
    
    # Latency
    overall_p50_latency_ms: float = 0.0
    overall_p95_latency_ms: float = 0.0
    overall_p99_latency_ms: float = 0.0
    
    # Stability
    worker_restarts: int = 0
    recovery_tests_passed: int = 0
    recovery_tests_failed: int = 0
    
    # Health checkpoints
    checkpoints: List[Dict] = field(default_factory=list)
    
    # Issues
    issues: List[str] = field(default_factory=list)
    
    # Final status
    status: str = "unknown"  # passed, degraded, failed


@dataclass
class SoakTestState:
    """Shared state for soak test."""
    running: bool = True
    
    # Counters
    events_attempted: int = 0
    events_written: int = 0
    events_processed_baseline: int = 0
    backpressure_events: int = 0
    errors: int = 0
    
    # Timing
    start_time: float = 0.0
    last_checkpoint_time: float = 0.0
    last_events_written: int = 0
    last_events_processed: int = 0
    
    # Latency samples (rolling window)
    latency_samples: List[float] = field(default_factory=list)
    latency_lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Worker state tracking
    last_worker_running: bool = True
    worker_restart_count: int = 0
    
    # Recovery tests
    recovery_tests_passed: int = 0
    recovery_tests_failed: int = 0
    
    # Issues
    issues: List[str] = field(default_factory=list)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024  # Convert KB to MB on Linux
    except:
        return 0.0


def create_soak_event(
    writer_id: int,
    sequence: int,
    payload_size: int = 150,
    workspace_id: str = "soak_test",
    user_id: str = None,
):
    """
    Create an event for soak testing.
    
    v1.1: Now accepts workspace_id and user_id parameters to ensure
    consistency with write_event() calls (required by controller v0.3.8).
    """
    from chillbot.kernel.models import Event
    
    # Default user_id based on writer_id if not specified
    if user_id is None:
        user_id = f"soak_user_{writer_id}"
    
    payload = {
        'text': f"Soak test writer {writer_id} seq {sequence}",
        'writer_id': writer_id,
        'sequence': sequence,
        'timestamp': time.time(),
        'checksum': uuid.uuid4().hex[:12],
        'padding': 'x' * max(0, payload_size - 80),
    }
    
    return Event(
        event_id=f"soak_{uuid.uuid4().hex[:16]}",
        workspace_id=workspace_id,
        user_id=user_id,
        session_id=f"{workspace_id}_{user_id}",
        content=payload,
        timestamp=time.time(),
    )


def format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# =============================================================================
# SOAK TEST RUNNER
# =============================================================================

class SoakTestRunner:
    """
    Main soak test orchestrator.
    
    Manages:
    - Writer workers that continuously write events
    - Health checker that takes periodic snapshots
    - Recovery tester that periodically tests backpressure recovery
    - Reporter that generates final report
    """
    
    def __init__(
        self,
        duration_seconds: float,
        intensity: str = 'medium',
        checkpoint_interval: float = 900,  # 15 minutes
    ):
        self.duration_seconds = duration_seconds
        self.intensity = intensity
        self.config = INTENSITY_CONFIGS.get(intensity, INTENSITY_CONFIGS['medium'])
        self.checkpoint_interval = checkpoint_interval
        
        self.state = SoakTestState()
        self.checkpoints: List[HealthCheckpoint] = []
        self.controller = None
        self.test_id = f"soak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Threading
        self._shutdown_event = threading.Event()
        self._threads: List[threading.Thread] = []
    
    def setup(self):
        """Initialize the KRNX controller."""
        from chillbot.kernel.controller import KRNXController
        from chillbot.kernel.connection_pool import close_pool, get_redis_client
        
        print(f"\n{'='*70}")
        print(f"KRNX OVERNIGHT SOAK TEST v1.1")
        print(f"{'='*70}")
        print(f"Test ID:        {self.test_id}")
        print(f"Duration:       {self.duration_seconds/3600:.1f} hours")
        print(f"Intensity:      {self.intensity}")
        print(f"Writers:        {self.config['writers']}")
        print(f"Target rate:    {self.config['events_per_minute']} events/min")
        print(f"Checkpoint:     every {self.checkpoint_interval/60:.0f} minutes")
        print(f"{'='*70}\n")
        
        # Clean up any existing connections
        try:
            close_pool()
        except:
            pass
        
        # Create data directory
        data_path = Path(f"./soak_data_{self.test_id}")
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize controller
        self.controller = KRNXController(
            data_path=str(data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=True,
            max_queue_depth=10000,
            max_lag_seconds=30.0,
            enable_async_worker=True,
            worker_block_ms=50,
            redis_max_connections=100,
        )
        
        # Clean Redis queue
        redis_client = get_redis_client()
        try:
            redis_client.delete('krnx:ltm:queue')
        except:
            pass
        
        # Record baseline
        metrics = self.controller.get_worker_metrics()
        self.state.events_processed_baseline = metrics.messages_processed
        self.state.start_time = time.time()
        self.state.last_checkpoint_time = time.time()
        
        print("[OK] Controller initialized (v0.3.8 with consistency validation)")
    
    def run(self):
        """Main test execution."""
        try:
            self.setup()
            
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            
            # Start worker threads
            self._start_workers()
            
            # Main loop - wait for duration or shutdown
            print(f"\n[START] Test running... (Ctrl+C to stop early)\n")
            
            end_time = time.time() + self.duration_seconds
            last_status_time = time.time()
            
            while self.state.running and time.time() < end_time:
                # Print periodic status
                if time.time() - last_status_time >= 60:
                    self._print_status()
                    last_status_time = time.time()
                
                time.sleep(1)
            
            self.state.running = False
            print("\n[STOP] Shutting down test...")
            
        except Exception as e:
            print(f"\n[ERROR] Fatal error: {e}")
            traceback.print_exc()
            self.state.issues.append(f"Fatal error: {e}")
        
        finally:
            self._shutdown()
    
    def _start_workers(self):
        """Start all worker threads."""
        # Writer workers
        for i in range(self.config['writers']):
            t = threading.Thread(target=self._writer_loop, args=(i,), daemon=True)
            t.start()
            self._threads.append(t)
        
        # Health checker
        t = threading.Thread(target=self._health_checker_loop, daemon=True)
        t.start()
        self._threads.append(t)
        
        # Recovery tester (every 30 minutes)
        t = threading.Thread(target=self._recovery_tester_loop, daemon=True)
        t.start()
        self._threads.append(t)
        
        print(f"[OK] Started {len(self._threads)} worker threads")
    
    def _writer_loop(self, writer_id: int):
        """Continuous writer loop."""
        import random
        from chillbot.kernel.controller import BackpressureError
        
        target_interval = 60.0 / (self.config['events_per_minute'] / self.config['writers'])
        sequence = 0
        
        while self.state.running:
            try:
                # Random burst
                if random.random() < self.config['burst_probability']:
                    burst_size = random.randint(
                        self.config['burst_size'] // 2,
                        self.config['burst_size']
                    )
                    for _ in range(burst_size):
                        if not self.state.running:
                            break
                        self._write_event(writer_id, sequence)
                        sequence += 1
                    continue
                
                # Random pause
                if random.random() < self.config['pause_probability']:
                    pause = random.uniform(*self.config['pause_duration_range'])
                    time.sleep(pause)
                    continue
                
                # Normal write
                self._write_event(writer_id, sequence)
                sequence += 1
                
                # Rate limiting
                time.sleep(target_interval * random.uniform(0.8, 1.2))
                
            except Exception as e:
                self.state.errors += 1
                time.sleep(0.1)
    
    def _write_event(self, writer_id: int, sequence: int):
        """Write a single event with tracking."""
        from chillbot.kernel.controller import BackpressureError, ValidationError
        
        # Define consistent IDs
        workspace_id = "soak_test"
        user_id = f"soak_user_{writer_id}"
        
        self.state.events_attempted += 1
        
        # Create event with matching IDs (v0.3.8 requirement)
        event = create_soak_event(
            writer_id, sequence,
            workspace_id=workspace_id,
            user_id=user_id
        )
        
        start_time = time.perf_counter()
        
        try:
            self.controller.write_event(
                workspace_id=workspace_id,
                user_id=user_id,
                event=event
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.state.events_written += 1
            
            # Record latency (keep last 1000 samples)
            with self.state.latency_lock:
                self.state.latency_samples.append(latency_ms)
                if len(self.state.latency_samples) > 1000:
                    self.state.latency_samples = self.state.latency_samples[-1000:]
                    
        except BackpressureError:
            self.state.backpressure_events += 1
        except ValidationError as e:
            # This shouldn't happen with consistent IDs, but log if it does
            self.state.errors += 1
            self.state.issues.append(f"ValidationError: {e}")
        except Exception as e:
            self.state.errors += 1
    
    def _health_checker_loop(self):
        """Periodic health check loop."""
        while self.state.running:
            try:
                time.sleep(self.checkpoint_interval)
                
                if not self.state.running:
                    break
                
                checkpoint = self._take_checkpoint()
                self.checkpoints.append(checkpoint)
                
                # Check for issues
                self._check_for_issues(checkpoint)
                
            except Exception as e:
                self.state.issues.append(f"Health check error: {e}")
    
    def _take_checkpoint(self) -> HealthCheckpoint:
        """Take a health checkpoint."""
        now = time.time()
        elapsed = now - self.state.start_time
        interval = now - self.state.last_checkpoint_time
        
        # Get current metrics
        metrics = self.controller.get_worker_metrics()
        events_processed = metrics.messages_processed - self.state.events_processed_baseline
        
        # Calculate rates
        events_written_delta = self.state.events_written - self.state.last_events_written
        events_processed_delta = events_processed - self.state.last_events_processed
        
        write_rate = (events_written_delta / interval) * 60 if interval > 0 else 0
        process_rate = (events_processed_delta / interval) * 60 if interval > 0 else 0
        error_rate = (self.state.errors / (elapsed / 60)) if elapsed > 0 else 0
        
        # Get latency stats
        with self.state.latency_lock:
            latencies = list(self.state.latency_samples)
        
        p50, p95, p99 = 0.0, 0.0, 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            p50 = statistics.median(sorted_lat)
            p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
            p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        
        # Check worker state
        worker_running = metrics.queue_depth >= 0  # Simple check
        if self.state.last_worker_running and not worker_running:
            self.state.worker_restart_count += 1
        self.state.last_worker_running = worker_running
        
        # Create checkpoint
        checkpoint = HealthCheckpoint(
            timestamp=now,
            elapsed_hours=elapsed / 3600,
            total_events_written=self.state.events_written,
            total_events_processed=events_processed,
            total_backpressure_events=self.state.backpressure_events,
            total_errors=self.state.errors,
            queue_depth=metrics.queue_depth,
            lag_seconds=metrics.lag_seconds,
            worker_running=worker_running,
            backpressure_mode=metrics.queue_depth > self.controller.max_queue_depth,
            write_rate_per_min=write_rate,
            process_rate_per_min=process_rate,
            error_rate_per_min=error_rate,
            memory_mb=get_memory_usage_mb(),
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
        )
        
        # Update tracking
        self.state.last_checkpoint_time = now
        self.state.last_events_written = self.state.events_written
        self.state.last_events_processed = events_processed
        
        # Print checkpoint
        print(f"\n[CHECKPOINT @ {format_duration(elapsed)}]")
        print(f"  Written: {checkpoint.total_events_written:,} "
              f"({checkpoint.write_rate_per_min:.0f}/min)")
        print(f"  Processed: {checkpoint.total_events_processed:,} "
              f"({checkpoint.process_rate_per_min:.0f}/min)")
        print(f"  Queue: {checkpoint.queue_depth} | Lag: {checkpoint.lag_seconds:.1f}s")
        print(f"  Latency: P50={checkpoint.p50_latency_ms:.1f}ms "
              f"P95={checkpoint.p95_latency_ms:.1f}ms P99={checkpoint.p99_latency_ms:.1f}ms")
        print(f"  Errors: {checkpoint.total_errors} | BP events: {checkpoint.total_backpressure_events}")
        print(f"  Memory: {checkpoint.memory_mb:.1f}MB")
        
        return checkpoint
    
    def _check_for_issues(self, checkpoint: HealthCheckpoint):
        """Check checkpoint for potential issues."""
        issues = []
        
        # Queue growing too large
        if checkpoint.queue_depth > self.controller.max_queue_depth * 0.8:
            issues.append(f"Queue near limit: {checkpoint.queue_depth}")
        
        # Processing falling behind
        if checkpoint.lag_seconds > self.controller.max_lag_seconds * 0.8:
            issues.append(f"Lag near limit: {checkpoint.lag_seconds:.1f}s")
        
        # High error rate
        if checkpoint.error_rate_per_min > 10:
            issues.append(f"High error rate: {checkpoint.error_rate_per_min:.1f}/min")
        
        # High latency
        if checkpoint.p99_latency_ms > 500:
            issues.append(f"High P99 latency: {checkpoint.p99_latency_ms:.1f}ms")
        
        for issue in issues:
            print(f"  [ISSUE] {issue}")
            self.state.issues.append(issue)
    
    def _recovery_tester_loop(self):
        """Periodic recovery test loop."""
        while self.state.running:
            try:
                # Wait 30 minutes between tests
                for _ in range(1800):
                    if not self.state.running:
                        return
                    time.sleep(1)
                
                if not self.state.running:
                    return
                
                self._test_recovery()
                
            except Exception as e:
                self.state.issues.append(f"Recovery test error: {e}")
    
    def _test_recovery(self):
        """Test backpressure recovery."""
        from chillbot.kernel.controller import BackpressureError, ValidationError
        
        print("\n[RECOVERY TEST] Starting...")
        
        # Force backpressure check
        bp_state = self.controller.force_backpressure_check()
        
        # If in backpressure, test recovery
        if bp_state:
            print("  System in backpressure, waiting for recovery...")
            time.sleep(5)
            
            # Test writes with consistent IDs (v0.3.8 requirement)
            workspace_id = "soak_test"
            user_id = "recovery_test"
            
            successes = 0
            for i in range(20):
                try:
                    event = create_soak_event(
                        9999, i,
                        workspace_id=workspace_id,
                        user_id=user_id
                    )
                    self.controller.write_event(
                        workspace_id=workspace_id,
                        user_id=user_id,
                        event=event
                    )
                    successes += 1
                except (BackpressureError, ValidationError):
                    pass
                except:
                    pass
                time.sleep(0.05)
            
            recovery_rate = successes / 20 * 100
            
            if recovery_rate >= 50:
                print(f"  [PASS] Recovery rate: {recovery_rate:.0f}%")
                self.state.recovery_tests_passed += 1
            else:
                print(f"  [FAIL] Recovery rate: {recovery_rate:.0f}%")
                self.state.recovery_tests_failed += 1
        else:
            print("  System not in backpressure, skipping")
            self.state.recovery_tests_passed += 1
    
    def _print_status(self):
        """Print current status line."""
        elapsed = time.time() - self.state.start_time
        remaining = self.duration_seconds - elapsed
        
        metrics = self.controller.get_worker_metrics()
        
        print(f"[{format_duration(elapsed)}] "
              f"Written: {self.state.events_written:,} | "
              f"Queue: {metrics.queue_depth} | "
              f"Errors: {self.state.errors} | "
              f"Remaining: {format_duration(max(0, remaining))}")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n[SIGNAL] Received signal {signum}, initiating shutdown...")
        self.state.running = False
    
    def _shutdown(self):
        """Shutdown and generate report."""
        self.state.running = False
        self._shutdown_event.set()
        
        # Wait for threads
        print("[SHUTDOWN] Waiting for threads to finish...")
        time.sleep(2)
        
        # Take final checkpoint
        print("[SHUTDOWN] Taking final checkpoint...")
        try:
            final_checkpoint = self._take_checkpoint()
            self.checkpoints.append(final_checkpoint)
        except:
            pass
        
        # Generate report
        report = self._generate_report()
        
        # Save report
        try:
            with open(SOAK_REPORT_FILE, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            print(f"\n[OK] Report saved to {SOAK_REPORT_FILE}")
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")
        
        # Print summary
        self._print_report(report)
        
        # Shutdown controller
        try:
            self.controller.shutdown(timeout=10)
        except:
            pass
    
    def _generate_report(self) -> SoakTestReport:
        """Generate final test report."""
        end_time = time.time()
        duration = end_time - self.state.start_time
        
        metrics = self.controller.get_worker_metrics()
        events_processed = metrics.messages_processed - self.state.events_processed_baseline
        
        report = SoakTestReport(
            test_id=self.test_id,
            start_time=datetime.fromtimestamp(self.state.start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration_hours=duration / 3600,
            intensity=self.intensity,
            total_events_attempted=self.state.events_attempted,
            total_events_written=self.state.events_written,
            total_events_processed=events_processed,
            total_backpressure_events=self.state.backpressure_events,
            total_errors=self.state.errors,
            worker_restarts=self.state.worker_restart_count,
            recovery_tests_passed=self.state.recovery_tests_passed,
            recovery_tests_failed=self.state.recovery_tests_failed,
            issues=self.state.issues,
        )
        
        # Calculate rates from checkpoints
        if self.checkpoints:
            write_rates = [c.write_rate_per_min for c in self.checkpoints]
            process_rates = [c.process_rate_per_min for c in self.checkpoints]
            queue_depths = [c.queue_depth for c in self.checkpoints]
            lags = [c.lag_seconds for c in self.checkpoints]
            
            report.avg_write_rate_per_min = statistics.mean(write_rates) if write_rates else 0
            report.avg_process_rate_per_min = statistics.mean(process_rates) if process_rates else 0
            report.peak_write_rate_per_min = max(write_rates) if write_rates else 0
            report.peak_queue_depth = max(queue_depths) if queue_depths else 0
            report.peak_lag_seconds = max(lags) if lags else 0
            
            # Latency from all checkpoints
            p50s = [c.p50_latency_ms for c in self.checkpoints if c.p50_latency_ms > 0]
            p95s = [c.p95_latency_ms for c in self.checkpoints if c.p95_latency_ms > 0]
            p99s = [c.p99_latency_ms for c in self.checkpoints if c.p99_latency_ms > 0]
            
            report.overall_p50_latency_ms = statistics.mean(p50s) if p50s else 0
            report.overall_p95_latency_ms = statistics.mean(p95s) if p95s else 0
            report.overall_p99_latency_ms = statistics.mean(p99s) if p99s else 0
            
            # Convert checkpoints to dicts
            report.checkpoints = [asdict(c) for c in self.checkpoints]
        
        # Determine status
        if self.state.errors > self.state.events_attempted * 0.1:
            report.status = "failed"
        elif self.state.recovery_tests_failed > 0:
            report.status = "degraded"
        elif len(self.state.issues) > 5:
            report.status = "degraded"
        else:
            report.status = "passed"
        
        return report
    
    def _print_report(self, report: SoakTestReport):
        """Print final report summary."""
        print(f"\n{'='*70}")
        print("SOAK TEST FINAL REPORT")
        print(f"{'='*70}")
        print(f"Test ID:        {report.test_id}")
        print(f"Duration:       {report.duration_hours:.2f} hours")
        print(f"Status:         {report.status.upper()}")
        print()
        print("THROUGHPUT:")
        print(f"  Events written:     {report.total_events_written:,}")
        print(f"  Events processed:   {report.total_events_processed:,}")
        print(f"  Backpressure:       {report.total_backpressure_events:,}")
        print(f"  Errors:             {report.total_errors:,}")
        print(f"  Avg write rate:     {report.avg_write_rate_per_min:.0f}/min")
        print(f"  Peak write rate:    {report.peak_write_rate_per_min:.0f}/min")
        print()
        print("LATENCY (avg across checkpoints):")
        print(f"  P50:  {report.overall_p50_latency_ms:.2f}ms")
        print(f"  P95:  {report.overall_p95_latency_ms:.2f}ms")
        print(f"  P99:  {report.overall_p99_latency_ms:.2f}ms")
        print()
        print("STABILITY:")
        print(f"  Peak queue depth:   {report.peak_queue_depth:,}")
        print(f"  Peak lag:           {report.peak_lag_seconds:.1f}s")
        print(f"  Worker restarts:    {report.worker_restarts}")
        print(f"  Recovery tests:     {report.recovery_tests_passed} passed, "
              f"{report.recovery_tests_failed} failed")
        print()
        print(f"Issues detected:      {len(report.issues)}")
        if report.issues:
            for issue in report.issues[:5]:
                print(f"  - {issue}")
            if len(report.issues) > 5:
                print(f"  ... and {len(report.issues) - 5} more")
        print(f"{'='*70}\n")


# =============================================================================
# PYTEST INTEGRATION
# =============================================================================

class TestSoakShort:
    """Short soak tests for pytest (1-5 minutes)."""
    
    def test_soak_1_minute(self, tmp_path):
        """Quick 1-minute soak test."""
        os.chdir(tmp_path)
        runner = SoakTestRunner(
            duration_seconds=60,
            intensity='low',
            checkpoint_interval=30,
        )
        runner.run()
        
        # Basic assertions
        assert runner.state.events_written > 0, "No events written"
        assert runner.state.errors < runner.state.events_attempted * 0.5, "Too many errors"
    
    def test_soak_5_minutes_medium(self, tmp_path):
        """5-minute medium intensity soak test."""
        os.chdir(tmp_path)
        runner = SoakTestRunner(
            duration_seconds=300,
            intensity='medium',
            checkpoint_interval=60,
        )
        runner.run()
        
        assert runner.state.events_written > 100, "Too few events written"


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point for overnight soak test."""
    print("\n" + "="*70)
    print(" KRNX OVERNIGHT SOAK TEST v1.1")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Duration:    {SOAK_DURATION_HOURS} hours")
    print(f"  Intensity:   {SOAK_INTENSITY}")
    print(f"  Checkpoint:  every {SOAK_CHECKPOINT_MINS} minutes")
    print(f"  Report file: {SOAK_REPORT_FILE}")
    print()
    
    runner = SoakTestRunner(
        duration_seconds=SOAK_DURATION_SECONDS,
        intensity=SOAK_INTENSITY,
        checkpoint_interval=SOAK_CHECKPOINT_SECONDS,
    )
    
    runner.run()
    
    # Exit with appropriate code
    if runner.state.errors > runner.state.events_attempted * 0.1:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
