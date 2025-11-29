"""
Metrics Collector

Collects and aggregates test metrics for analysis.
"""

import time
import json
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class TestMetrics:
    """Metrics for a single test run."""
    test_name: str
    adapter_name: str
    start_time: float
    end_time: float
    duration_ms: float
    events_written: int = 0
    events_read: int = 0
    write_latency_ms: List[float] = field(default_factory=list)
    read_latency_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs."""
    test_name: str
    adapter_name: str
    run_count: int
    
    # Duration stats
    duration_mean_ms: float
    duration_std_ms: float
    duration_min_ms: float
    duration_max_ms: float
    
    # Write latency stats
    write_latency_mean_ms: float
    write_latency_p50_ms: float
    write_latency_p95_ms: float
    write_latency_p99_ms: float
    
    # Read latency stats
    read_latency_mean_ms: float
    read_latency_p50_ms: float
    read_latency_p95_ms: float
    read_latency_p99_ms: float
    
    # Throughput
    events_per_second: float
    
    # Errors
    total_errors: int
    error_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    Collects and aggregates test metrics.
    
    Usage:
        collector = MetricsCollector("test_k1_append_only")
        
        with collector.measure("KRNX"):
            # Run test
            for event in events:
                collector.record_write(latency_ms)
        
        metrics = collector.get_metrics()
        collector.save_report("reports/k1_results.json")
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self._runs: List[TestMetrics] = []
        self._current_run: Optional[TestMetrics] = None
    
    class _MeasureContext:
        """Context manager for measuring a test run."""
        
        def __init__(self, collector: 'MetricsCollector', adapter_name: str):
            self.collector = collector
            self.adapter_name = adapter_name
        
        def __enter__(self):
            self.collector._start_run(self.adapter_name)
            return self.collector
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.collector.record_error(str(exc_val))
            self.collector._end_run()
            return False
    
    def measure(self, adapter_name: str) -> _MeasureContext:
        """
        Context manager for measuring a test run.
        
        Args:
            adapter_name: Name of the adapter being tested
            
        Returns:
            Context manager
        """
        return self._MeasureContext(self, adapter_name)
    
    def _start_run(self, adapter_name: str) -> None:
        """Start a new measurement run."""
        self._current_run = TestMetrics(
            test_name=self.test_name,
            adapter_name=adapter_name,
            start_time=time.time(),
            end_time=0,
            duration_ms=0,
        )
    
    def _end_run(self) -> None:
        """End the current measurement run."""
        if self._current_run:
            self._current_run.end_time = time.time()
            self._current_run.duration_ms = (
                (self._current_run.end_time - self._current_run.start_time) * 1000
            )
            self._runs.append(self._current_run)
            self._current_run = None
    
    def record_write(self, latency_ms: float) -> None:
        """Record a write operation."""
        if self._current_run:
            self._current_run.events_written += 1
            self._current_run.write_latency_ms.append(latency_ms)
    
    def record_read(self, latency_ms: float, events_count: int = 1) -> None:
        """Record a read operation."""
        if self._current_run:
            self._current_run.events_read += events_count
            self._current_run.read_latency_ms.append(latency_ms)
    
    def record_error(self, error: str) -> None:
        """Record an error."""
        if self._current_run:
            self._current_run.errors.append(error)
    
    def record_custom(self, key: str, value: Any) -> None:
        """Record a custom metric."""
        if self._current_run:
            self._current_run.custom_metrics[key] = value
    
    def get_runs(self) -> List[TestMetrics]:
        """Get all recorded runs."""
        return self._runs.copy()
    
    def get_aggregated(self, adapter_name: Optional[str] = None) -> AggregatedMetrics:
        """
        Get aggregated metrics.
        
        Args:
            adapter_name: Filter to specific adapter (optional)
            
        Returns:
            AggregatedMetrics instance
        """
        runs = self._runs
        if adapter_name:
            runs = [r for r in runs if r.adapter_name == adapter_name]
        
        if not runs:
            raise ValueError("No runs to aggregate")
        
        # Duration stats
        durations = [r.duration_ms for r in runs]
        
        # Combine all latencies
        all_write_latencies = []
        all_read_latencies = []
        total_events = 0
        total_errors = 0
        total_duration = 0
        
        for run in runs:
            all_write_latencies.extend(run.write_latency_ms)
            all_read_latencies.extend(run.read_latency_ms)
            total_events += run.events_written + run.events_read
            total_errors += len(run.errors)
            total_duration += run.duration_ms
        
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        return AggregatedMetrics(
            test_name=self.test_name,
            adapter_name=adapter_name or runs[0].adapter_name,
            run_count=len(runs),
            
            duration_mean_ms=statistics.mean(durations),
            duration_std_ms=statistics.stdev(durations) if len(durations) > 1 else 0,
            duration_min_ms=min(durations),
            duration_max_ms=max(durations),
            
            write_latency_mean_ms=statistics.mean(all_write_latencies) if all_write_latencies else 0,
            write_latency_p50_ms=percentile(all_write_latencies, 50),
            write_latency_p95_ms=percentile(all_write_latencies, 95),
            write_latency_p99_ms=percentile(all_write_latencies, 99),
            
            read_latency_mean_ms=statistics.mean(all_read_latencies) if all_read_latencies else 0,
            read_latency_p50_ms=percentile(all_read_latencies, 50),
            read_latency_p95_ms=percentile(all_read_latencies, 95),
            read_latency_p99_ms=percentile(all_read_latencies, 99),
            
            events_per_second=total_events / (total_duration / 1000) if total_duration > 0 else 0,
            
            total_errors=total_errors,
            error_rate=total_errors / len(runs) if runs else 0,
        )
    
    def save_report(self, path: str) -> None:
        """
        Save metrics report to JSON file.
        
        Args:
            path: Output file path
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'test_name': self.test_name,
            'generated_at': datetime.now().isoformat(),
            'runs': [r.to_dict() for r in self._runs],
        }
        
        # Add aggregated if we have runs
        if self._runs:
            # Group by adapter
            adapters = set(r.adapter_name for r in self._runs)
            report['aggregated'] = {
                adapter: self.get_aggregated(adapter).to_dict()
                for adapter in adapters
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of metrics to console."""
        if not self._runs:
            print("No runs recorded")
            return
        
        adapters = set(r.adapter_name for r in self._runs)
        
        print(f"\n{'='*60}")
        print(f"Test: {self.test_name}")
        print(f"{'='*60}")
        
        for adapter in adapters:
            agg = self.get_aggregated(adapter)
            print(f"\nAdapter: {adapter}")
            print(f"  Runs: {agg.run_count}")
            print(f"  Duration: {agg.duration_mean_ms:.2f}ms (±{agg.duration_std_ms:.2f}ms)")
            print(f"  Write latency p50/p95/p99: {agg.write_latency_p50_ms:.2f}/{agg.write_latency_p95_ms:.2f}/{agg.write_latency_p99_ms:.2f}ms")
            print(f"  Read latency p50/p95/p99: {agg.read_latency_p50_ms:.2f}/{agg.read_latency_p95_ms:.2f}/{agg.read_latency_p99_ms:.2f}ms")
            print(f"  Throughput: {agg.events_per_second:.0f} events/sec")
            print(f"  Errors: {agg.total_errors} ({agg.error_rate:.2%})")
