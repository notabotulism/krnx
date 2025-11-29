#!/usr/bin/env python3
"""
KRNX Layer 2 Fabric Test Runner

Runs all Layer 2 Fabric tests (F1-F10) and generates proof output.

Usage:
    # Run all Layer 2 tests
    python run_layer2_tests.py
    
    # Run specific guarantee
    python run_layer2_tests.py --guarantee F8
    
    # Run with verbose output
    python run_layer2_tests.py -v
    
    # Generate proof evidence file
    python run_layer2_tests.py --output evidence/proof_fabric_$(date +%m.%d.%Y).txt
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run KRNX Layer 2 Fabric Tests")
    parser.add_argument(
        "--guarantee", "-g",
        choices=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "all"],
        default="all",
        help="Which guarantee to test (default: all)"
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
    
    args = parser.parse_args()
    
    # Determine test path
    script_dir = Path(__file__).parent
    test_dir = script_dir / "layer2_fabric"
    
    # Map guarantees to test files
    guarantee_map = {
        "F1": "test_f1_f2_f3_orchestration.py::TestF1RememberOrchestration",
        "F2": "test_f1_f2_f3_orchestration.py::TestF2RecallOrchestration",
        "F3": "test_f1_f2_f3_orchestration.py::TestF3ContextBuilding",
        "F4": "test_f4_f5_embedding_vectors.py::TestF4EmbeddingGeneration",
        "F5": "test_f4_f5_embedding_vectors.py::TestF5VectorSearch",
        "F6": "test_f6_f7_queue_health.py::TestF6JobEnqueue",
        "F7": "test_f6_f7_queue_health.py::TestF7HealthChecks",
        "F8": "test_f8_enrichment.py",
        "F9": "test_f9_retention.py",
        "F10": "test_f10_identity.py",
    }
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.guarantee == "all":
        cmd.append(str(test_dir))
    else:
        test_path = test_dir / guarantee_map[args.guarantee]
        cmd.append(str(test_path))
    
    # Add options
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-v")  # Always show test names
    
    if args.failfast:
        cmd.append("-x")
    
    # Always show print statements
    cmd.append("-s")
    
    # Add timing
    cmd.append("--tb=short")
    
    # Output handling
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Running Layer 2 tests, output to: {output_path}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 70)
        
        with open(output_path, "w") as f:
            # Write header
            f.write("=" * 70 + "\n")
            f.write("KRNX LAYER 2 FABRIC - PROOF OUTPUT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Guarantee: {args.guarantee}\n")
            f.write("=" * 70 + "\n\n")
            
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
            f.write("=" * 70 + "\n")
        
        print(result.stdout)
        return result.returncode
    else:
        print(f"Running Layer 2 tests...")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 70)
        
        result = subprocess.run(cmd, cwd=script_dir.parent)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
