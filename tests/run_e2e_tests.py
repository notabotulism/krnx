#!/usr/bin/env python3
"""
KRNX E2E Test Runner

Convenience script to run all E2E test suites.

Usage:
    python3 run_e2e_tests.py               # Run all tests
    python3 run_e2e_tests.py --suite 1     # Run functionality tests only
    python3 run_e2e_tests.py --suite 2     # Run integration tests only
    python3 run_e2e_tests.py --suite 3     # Run stress tests only
    python3 run_e2e_tests.py --suite 4     # Run soak test (1 hour)
    python3 run_e2e_tests.py --quick       # Run functionality + integration only
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Determine paths
SCRIPT_DIR = Path(__file__).parent
TESTS_DIR = SCRIPT_DIR / "tests" if (SCRIPT_DIR / "tests").exists() else SCRIPT_DIR

# Test suite definitions
SUITES = {
    1: {
        "name": "Functionality Tests",
        "file": "test_e2e_functionality.py",
        "description": "Tests each component in isolation",
        "duration": "~2-5 minutes",
    },
    2: {
        "name": "Integration Tests", 
        "file": "test_e2e_integration.py",
        "description": "Tests full pipeline end-to-end",
        "duration": "~5-10 minutes",
    },
    3: {
        "name": "Stress Tests",
        "file": "test_e2e_stress.py",
        "description": "Tests under extreme load",
        "duration": "~10-15 minutes",
    },
    4: {
        "name": "Soak Test",
        "file": "test_e2e_soak.py",
        "description": "Long-running test for memory leaks",
        "duration": "1+ hours (configurable)",
    },
}


def run_suite(suite_num: int, extra_args: list = None) -> int:
    """Run a single test suite."""
    suite = SUITES.get(suite_num)
    if not suite:
        print(f"Unknown suite: {suite_num}")
        return 1
    
    test_file = TESTS_DIR / suite["file"]
    if not test_file.exists():
        # Try current directory
        test_file = SCRIPT_DIR / suite["file"]
    
    if not test_file.exists():
        print(f"Test file not found: {suite['file']}")
        return 1
    
    print(f"\n{'=' * 70}")
    print(f"Running Suite {suite_num}: {suite['name']}")
    print(f"Description: {suite['description']}")
    print(f"Expected duration: {suite['duration']}")
    print(f"{'=' * 70}\n")
    
    cmd = [sys.executable, str(test_file)]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="KRNX E2E Test Runner")
    parser.add_argument("--suite", type=int, choices=[1, 2, 3, 4], help="Run specific suite (1-4)")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (1 & 2)")
    parser.add_argument("--all", action="store_true", help="Run all tests including soak")
    parser.add_argument("--stress-threads", type=int, default=50, help="Threads for stress test")
    parser.add_argument("--stress-events", type=int, default=100, help="Events per thread for stress test")
    parser.add_argument("--soak-duration", type=int, default=3600, help="Soak test duration in seconds")
    args = parser.parse_args()
    
    print("=" * 70)
    print("KRNX E2E TEST RUNNER")
    print("=" * 70)
    print("\nAvailable test suites:")
    for num, suite in SUITES.items():
        print(f"  {num}. {suite['name']}: {suite['description']} ({suite['duration']})")
    
    results = {}
    
    if args.suite:
        # Run single suite
        extra_args = []
        if args.suite == 3:
            extra_args = [f"--threads={args.stress_threads}", f"--events={args.stress_events}"]
        elif args.suite == 4:
            extra_args = [f"--duration={args.soak_duration}"]
        
        return run_suite(args.suite, extra_args)
    
    elif args.quick:
        # Run quick tests
        for suite_num in [1, 2]:
            results[suite_num] = run_suite(suite_num)
    
    elif args.all:
        # Run all tests
        for suite_num in [1, 2, 3]:
            results[suite_num] = run_suite(suite_num)
            if results[suite_num] != 0:
                print(f"\nSuite {suite_num} failed, stopping.")
                break
        
        # Soak test with shorter duration for --all
        if all(r == 0 for r in results.values()):
            results[4] = run_suite(4, [f"--duration={min(args.soak_duration, 300)}"])  # 5 min max
    
    else:
        # Default: run functionality and integration
        for suite_num in [1, 2]:
            results[suite_num] = run_suite(suite_num)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for suite_num, result in results.items():
        status = "✓ PASSED" if result == 0 else "✗ FAILED"
        print(f"  Suite {suite_num} ({SUITES[suite_num]['name']}): {status}")
        if result != 0:
            all_passed = False
    
    print("=" * 70)
    print(f"Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
