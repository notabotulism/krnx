#!/usr/bin/env python3
"""
KRNX Layer 3 Demo Test Runner

Runs all Layer 3 Demo tests (D1-D10) and generates proof output.

Layer 3 tests prove what KRNX ENABLES - real AI applications
doing things they couldn't do without temporal memory infrastructure.

Usage:
    # Run all Layer 3 tests
    python run_layer3_tests.py
    
    # Run specific demo/guarantee
    python run_layer3_tests.py --demo D3
    
    # Run THE DIFFERENTIATOR tests (temporal)
    python run_layer3_tests.py --demo temporal
    
    # Run benchmark tests (requires LLM API key)
    python run_layer3_tests.py --demo benchmarks
    
    # Run multi-agent tests
    python run_layer3_tests.py --demo multiagent
    
    # Run with verbose output
    python run_layer3_tests.py -v
    
    # Generate proof evidence file
    python run_layer3_tests.py --output evidence/proof_demos_$(date +%m.%d.%Y).txt
    
    # Quick mode (reduced sample sizes)
    python run_layer3_tests.py --quick

Environment Variables:
    OPENAI_API_KEY      - Required for D1/D2 benchmarks
    ANTHROPIC_API_KEY   - Alternative to OpenAI for benchmarks
    KRNX_LLM_PROVIDER   - 'openai' or 'anthropic' (default: openai)
    KRNX_LLM_MODEL      - Model name (default: gpt-4o-mini)
    REDIS_URL           - Redis connection (default: redis://localhost:6379)
    QDRANT_URL          - Qdrant connection (default: http://localhost:6333)
    KRNX_QUICK_MODE     - 'true' for reduced sample sizes
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path
from datetime import datetime


# Demo/Guarantee mappings
DEMO_MAP = {
    # Individual demos
    "D1": "test_d1_locomo.py",
    "D2": "test_d2_longmemeval.py",
    "D3": "test_d3_d5_temporal.py::TestD3TemporalReplayAccuracy",
    "D4": "test_d3_d5_temporal.py::TestD4FactSupersession",
    "D5": "test_d3_d5_temporal.py::TestD5HashChainProvenance",
    "D6": "test_d6_d10_multiagent.py::TestD6SharedSubstrateRead",
    "D7": "test_d6_d10_multiagent.py::TestD7SharedSubstrateWrite",
    "D8": "test_d6_d10_multiagent.py::TestD8TemporalConsistencyAcrossAgents",
    "D9": "test_d6_d10_multiagent.py::TestD9AgentStateIsolation",
    "D10": "test_d6_d10_multiagent.py::TestD10CoordinationPrimitives",
    
    # Grouped demos
    "temporal": "test_d3_d5_temporal.py",           # THE DIFFERENTIATOR
    "multiagent": "test_d6_d10_multiagent.py",      # Multi-agent coordination
    "benchmarks": ["test_d1_locomo.py", "test_d2_longmemeval.py"],  # Benchmark tests
    "hashchain": "test_d3_d5_temporal.py::TestD5HashChainProvenance",
    
    # Categories
    "fast": [                                        # Tests that don't need LLM
        "test_d3_d5_temporal.py",
        "test_d6_d10_multiagent.py",
    ],
    "all": None,  # Run everything
}

# Marker mappings for filtering
MARKER_MAP = {
    "slow": "-m slow",
    "requires_llm": "-m requires_llm",
    "failure_mode": "-m failure_mode",
    "no_llm": "-m 'not requires_llm'",
}


def check_environment():
    """Check and report environment status."""
    issues = []
    warnings = []
    
    # Check for LLM API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    if not has_openai and not has_anthropic:
        warnings.append("No LLM API key found (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
        warnings.append("  → D1/D2 benchmark tests will be skipped")
    
    # Check Redis
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Check Qdrant
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    
    return issues, warnings, {
        "has_openai": has_openai,
        "has_anthropic": has_anthropic,
        "redis_url": redis_url,
        "qdrant_url": qdrant_url,
    }


def print_header(demo: str, env_info: dict):
    """Print informative header."""
    print("=" * 70)
    print("KRNX LAYER 3 DEMO TESTS - THE DIFFERENTIATOR")
    print("=" * 70)
    print(f"Demo/Guarantee: {demo}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    print("Environment:")
    print(f"  LLM: {'OpenAI ✓' if env_info['has_openai'] else 'Anthropic ✓' if env_info['has_anthropic'] else '✗ None'}")
    print(f"  Redis: {env_info['redis_url']}")
    print(f"  Qdrant: {env_info['qdrant_url']}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run KRNX Layer 3 Demo Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run temporal tests (THE DIFFERENTIATOR) - no LLM needed
    python run_layer3_tests.py --demo temporal
    
    # Run all fast tests (no LLM required)
    python run_layer3_tests.py --demo fast
    
    # Run D3 temporal replay tests only
    python run_layer3_tests.py --demo D3
    
    # Run benchmarks with proof output
    python run_layer3_tests.py --demo benchmarks -o evidence/benchmarks.txt

Demo Groups:
    temporal   - D3-D5: Temporal replay, fact supersession, hash chain
    multiagent - D6-D10: Multi-agent coordination
    benchmarks - D1-D2: LOCOMO and LongMemEval (requires LLM)
    fast       - All tests that don't require LLM API
    all        - Everything
        """
    )
    
    parser.add_argument(
        "--demo", "-d",
        choices=list(DEMO_MAP.keys()),
        default="all",
        help="Which demo/guarantee to test (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for proof evidence"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast", "-x",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode (reduced sample sizes)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip tests that require LLM API"
    )
    parser.add_argument(
        "--markers", "-m",
        type=str,
        default=None,
        help="Pytest marker expression (e.g., 'temporal and not slow')"
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect tests, don't run them"
    )
    
    args = parser.parse_args()
    
    # Check environment
    issues, warnings, env_info = check_environment()
    
    if issues:
        print("ERROR: Cannot run tests:")
        for issue in issues:
            print(f"  ✗ {issue}")
        return 1
    
    if warnings:
        print("WARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
        print()
    
    # Set quick mode environment variable
    if args.quick:
        os.environ["KRNX_QUICK_MODE"] = "true"
        os.environ["KRNX_LATENCY_SAMPLES"] = "10"
        os.environ["KRNX_LOCOMO_SAMPLES"] = "10"
        os.environ["KRNX_LONGMEMEVAL_SAMPLES"] = "10"
    
    # Determine test path - tests are in the same directory as this script
    script_dir = Path(__file__).parent
    test_dir = script_dir  # Tests are in the same directory
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Determine what to run
    demo_value = DEMO_MAP.get(args.demo)
    
    if demo_value is None:
        # Run all tests
        cmd.append(str(test_dir))
    elif isinstance(demo_value, list):
        # Multiple test files
        for test_file in demo_value:
            cmd.append(str(test_dir / test_file))
    else:
        # Single test file or class
        if "::" in demo_value:
            # Test file with class
            file_part, class_part = demo_value.split("::", 1)
            cmd.append(f"{test_dir / file_part}::{class_part}")
        else:
            cmd.append(str(test_dir / demo_value))
    
    # Add options
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    if args.failfast:
        cmd.append("-x")
    
    if args.no_llm:
        cmd.extend(["-m", "not requires_llm"])
    elif args.markers:
        cmd.extend(["-m", args.markers])
    
    if args.collect_only:
        cmd.append("--collect-only")
    
    # Always show print statements
    cmd.append("-s")
    
    # Short traceback
    cmd.append("--tb=short")
    
    # Print header
    print_header(args.demo, env_info)
    
    # Output handling
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Output file: {output_path}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 70)
        print()
        
        with open(output_path, "w") as f:
            # Write header
            f.write("=" * 70 + "\n")
            f.write("KRNX LAYER 3 DEMOS - PROOF OUTPUT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Demo: {args.demo}\n")
            f.write(f"Quick Mode: {args.quick}\n")
            f.write(f"LLM Available: {'Yes' if env_info['has_openai'] or env_info['has_anthropic'] else 'No'}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            
            # Run tests and capture output
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=script_dir.parent  # Run from prod_test parent
            )
            
            f.write(result.stdout)
            
            # Write footer
            f.write("\n" + "=" * 70 + "\n")
            f.write(f"Exit code: {result.returncode}\n")
            status = "PASSED" if result.returncode == 0 else "FAILED"
            f.write(f"Status: {status}\n")
            f.write("=" * 70 + "\n")
        
        # Also print to console
        print(result.stdout)
        
        print()
        print("=" * 70)
        print(f"Proof output saved to: {output_path}")
        print("=" * 70)
        
        return result.returncode
    else:
        print(f"Command: {' '.join(cmd)}")
        print("=" * 70)
        print()
        
        result = subprocess.run(cmd, cwd=script_dir.parent)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
