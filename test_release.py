#!/usr/bin/env python3
"""
krnx Release Test Suite

Comprehensive automated testing for pre-release validation.
Run this before publishing to PyPI or posting to r/llmdevs.

Usage:
    python test_release.py           # Run all tests
    python test_release.py --quick   # Skip slow tests
    python test_release.py --live    # Include live API tests (needs ANTHROPIC_API_KEY)

Exit codes:
    0 = All tests passed
    1 = Some tests failed
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# TEST FRAMEWORK
# =============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None

class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []
        self.temp_dirs: List[str] = []
    
    def run_test(self, name: str, test_fn):
        """Run a single test and record result."""
        print(f"  {name}...", end=" ", flush=True)
        start = time.time()
        try:
            test_fn()
            duration = time.time() - start
            print(f"✓ ({duration:.2f}s)")
            self.results.append(TestResult(name, True, duration))
        except Exception as e:
            duration = time.time() - start
            print(f"✗")
            print(f"    Error: {e}")
            self.results.append(TestResult(name, False, duration, str(e)))
    
    def make_temp_dir(self) -> str:
        """Create a temp directory that will be cleaned up."""
        d = tempfile.mkdtemp(prefix="krnx_test_")
        self.temp_dirs.append(d)
        return d
    
    def cleanup(self):
        """Clean up temp directories."""
        for d in self.temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
    
    def summary(self) -> Tuple[int, int]:
        """Print summary and return (passed, failed) counts."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration for r in self.results)
        
        print()
        print("=" * 60)
        print(f"RESULTS: {passed} passed, {failed} failed ({total_time:.2f}s)")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}: {r.error}")
        
        return passed, failed


runner = TestRunner()


# =============================================================================
# IMPORT TESTS
# =============================================================================

def test_import_krnx():
    """Test that krnx can be imported."""
    import krnx
    assert hasattr(krnx, 'init')
    assert hasattr(krnx, 'Substrate')
    assert hasattr(krnx, 'Event')
    assert hasattr(krnx, '__version__')

def test_import_cli():
    """Test that CLI module can be imported."""
    from krnx import cli
    assert hasattr(cli, 'app')
    assert hasattr(cli, 'main')

def test_import_studio():
    """Test that Studio module can be imported."""
    from krnx.studio import run_studio
    assert callable(run_studio)

def test_import_cs_agent():
    """Test that CS agent module can be imported."""
    from krnx import cs_agent
    assert hasattr(cs_agent, 'run_demo')
    assert hasattr(cs_agent, 'run_agent')
    assert hasattr(cs_agent, 'SCENARIO_MAIN')
    assert hasattr(cs_agent, 'SCENARIO_FIX')


# =============================================================================
# CORE FUNCTIONALITY TESTS
# =============================================================================

def test_init_workspace():
    """Test workspace initialization."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-workspace", path=temp_dir)
    assert s is not None
    assert s.name == "test-workspace"

def test_record_event():
    """Test recording events."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-record", path=temp_dir)
    
    event_id = s.record("test", {"message": "hello"})
    assert event_id.startswith("evt_")
    
    events = s.log()
    assert len(events) == 1
    assert events[0].type == "test"
    assert events[0].content["message"] == "hello"

def test_hash_chain():
    """Test hash chain integrity."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-chain", path=temp_dir)
    
    # Record multiple events
    for i in range(10):
        s.record("event", {"index": i})
    
    # Verify chain
    assert s.verify() == True
    
    events = s.log()
    assert len(events) == 10
    
    # Events come back in reverse chronological order
    # Reverse to get chronological order for chain checking
    events_chrono = list(reversed(events))
    
    # Check chain links (each event's parent should be previous event's hash)
    for i in range(1, len(events_chrono)):
        assert events_chrono[i].parent == events_chrono[i-1].hash, \
            f"Chain broken at index {i}"

def test_branching():
    """Test branch creation and isolation."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-branch", path=temp_dir)
    
    # Record on main
    e1 = s.record("main-event", {"branch": "main"})
    
    # Create branch
    s.branch("feature", from_event=e1)
    
    # Record on branch
    s.record("branch-event", {"branch": "feature"}, branch="feature")
    
    # Verify isolation
    main_events = s.log(branch="main")
    feature_events = s.log(branch="feature")
    
    assert len(main_events) == 1
    assert len(feature_events) == 2  # Includes fork point + new event
    
    # Verify both chains
    assert s.verify(branch="main") == True
    assert s.verify(branch="feature") == True

def test_search():
    """Test event search."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-search", path=temp_dir)
    
    s.record("alpha", {"data": "findme"})
    s.record("beta", {"data": "other"})
    s.record("gamma", {"data": "findme too"})
    
    results = s.search("findme")
    assert len(results) >= 2

def test_export_import():
    """Test JSONL export and import."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-export", path=temp_dir)
    
    # Record events
    for i in range(5):
        s.record("event", {"index": i})
    
    # Export (use named param since signature is export(branch, path))
    export_path = Path(temp_dir) / "export.jsonl"
    s.export(path=str(export_path))
    assert export_path.exists()
    
    # Verify export content
    with open(export_path) as f:
        lines = f.readlines()
    assert len(lines) == 5


# =============================================================================
# CS AGENT TESTS
# =============================================================================

def test_cs_agent_mock():
    """Test CS agent with mock responses."""
    import krnx
    from krnx.cs_agent import run_demo, SCENARIO_MAIN
    
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-cs-agent", path=temp_dir)
    
    results = run_demo(s, verbose=False, mock=True)
    
    # Check main branch result
    assert results["main"]["decision"] == "APPROVE"
    assert results["main"]["result"]["outcome"] == "LOSS"
    
    # Check fix branch result
    assert results["fix"]["decision"] == "DENY"
    assert results["fix"]["result"]["outcome"] == "FRAUD_PREVENTED"
    
    # Check verification
    assert results["main_valid"] == True
    assert results["fix_valid"] == True

def test_cs_agent_events():
    """Test CS agent records correct events."""
    import krnx
    from krnx.cs_agent import run_agent, SCENARIO_MAIN
    
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-cs-events", path=temp_dir)
    
    result = run_agent(s, SCENARIO_MAIN, branch="main", mock=True)
    
    events = s.log(branch="main")
    assert len(events) == 4
    
    event_types = [e.type for e in events]
    assert "observe" in event_types
    assert "think" in event_types
    assert "act" in event_types
    assert "result" in event_types


# =============================================================================
# CLI TESTS
# =============================================================================

def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run(
        ["krnx", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Git for" in result.stdout

def test_cli_version():
    """Test CLI version command."""
    result = subprocess.run(
        ["krnx", "version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "krnx" in result.stdout

def test_cli_try_no_key():
    """Test CLI try command without API key."""
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    
    result = subprocess.run(
        ["krnx", "try"],
        capture_output=True,
        text=True,
        env=env
    )
    assert result.returncode == 1
    assert "ANTHROPIC_API_KEY" in result.stdout

def test_cli_init_record_log():
    """Test CLI init, record, log workflow."""
    temp_dir = runner.make_temp_dir()
    
    # Change to temp dir
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        # Init
        result = subprocess.run(
            ["krnx", "init", "cli-test"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Record
        result = subprocess.run(
            ["krnx", "record", "test", '{"message": "hello"}'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        
        # Log
        result = subprocess.run(
            ["krnx", "log"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "test" in result.stdout
        
    finally:
        os.chdir(original_dir)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

def test_write_throughput():
    """Test write performance (should be >1000 events/sec)."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-perf", path=temp_dir)
    
    n = 1000
    start = time.time()
    for i in range(n):
        s.record("perf", {"index": i})
    duration = time.time() - start
    
    throughput = n / duration
    assert throughput > 1000, f"Throughput {throughput:.0f} events/sec is too low"

def test_verify_performance():
    """Test verify performance (should be >10000 events/sec)."""
    import krnx
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-verify-perf", path=temp_dir)
    
    # Record events
    for i in range(1000):
        s.record("event", {"index": i})
    
    # Time verify
    start = time.time()
    assert s.verify() == True
    duration = time.time() - start
    
    throughput = 1000 / duration
    assert throughput > 10000, f"Verify throughput {throughput:.0f} events/sec is too low"


# =============================================================================
# LIVE API TESTS (requires ANTHROPIC_API_KEY)
# =============================================================================

def test_live_api():
    """Test live API call (requires ANTHROPIC_API_KEY)."""
    import krnx
    from krnx.cs_agent import run_agent, SCENARIO_MAIN
    
    temp_dir = runner.make_temp_dir()
    s = krnx.init("test-live", path=temp_dir)
    
    result = run_agent(s, SCENARIO_MAIN, branch="main", mock=False)
    
    assert result["decision"] in ["APPROVE", "DENY"]
    assert result["events_recorded"] == 4


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = sys.argv[1:]
    quick_mode = "--quick" in args
    live_mode = "--live" in args
    
    print("=" * 60)
    print("krnx Release Test Suite")
    print("=" * 60)
    print()
    
    # Import tests
    print("[1/6] Import Tests")
    runner.run_test("import krnx", test_import_krnx)
    runner.run_test("import cli", test_import_cli)
    runner.run_test("import studio", test_import_studio)
    runner.run_test("import cs_agent", test_import_cs_agent)
    print()
    
    # Core tests
    print("[2/6] Core Functionality")
    runner.run_test("init workspace", test_init_workspace)
    runner.run_test("record event", test_record_event)
    runner.run_test("hash chain", test_hash_chain)
    runner.run_test("branching", test_branching)
    runner.run_test("search", test_search)
    runner.run_test("export/import", test_export_import)
    print()
    
    # CS Agent tests
    print("[3/6] CS Agent (Mock)")
    runner.run_test("mock demo", test_cs_agent_mock)
    runner.run_test("event recording", test_cs_agent_events)
    print()
    
    # CLI tests
    print("[4/6] CLI Commands")
    runner.run_test("help", test_cli_help)
    runner.run_test("version", test_cli_version)
    runner.run_test("try (no key)", test_cli_try_no_key)
    runner.run_test("init/record/log", test_cli_init_record_log)
    print()
    
    # Performance tests
    if not quick_mode:
        print("[5/6] Performance")
        runner.run_test("write throughput", test_write_throughput)
        runner.run_test("verify performance", test_verify_performance)
        print()
    else:
        print("[5/6] Performance (skipped - quick mode)")
        print()
    
    # Live API tests
    if live_mode:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print("[6/6] Live API Tests")
            runner.run_test("live API call", test_live_api)
            print()
        else:
            print("[6/6] Live API Tests (skipped - no ANTHROPIC_API_KEY)")
            print()
    else:
        print("[6/6] Live API Tests (skipped - use --live to enable)")
        print()
    
    # Cleanup and summary
    runner.cleanup()
    passed, failed = runner.summary()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
