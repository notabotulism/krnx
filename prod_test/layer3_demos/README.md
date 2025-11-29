# KRNX Layer 3 Tests: Academic Rigor Edition

## Overview

Layer 3 tests prove what KRNX **enables** - real AI applications doing things they couldn't do without temporal memory infrastructure.

**Academic Rigor Features:**
- P1-P2-P3-P4 proof methodology
- Statistical analysis (n=30, CI, effect size, outlier removal)
- Strong assertions with exact expected values
- LLM-as-judge evaluation for benchmarks
- Failure mode testing

## Test Structure

```
layer3_tests_fixed/
├── conftest.py              # Shared fixtures, statistical helpers, dataset downloaders
├── test_d1_locomo.py        # D1: LOCOMO benchmark
├── test_d2_longmemeval.py   # D2: LongMemEval benchmark
├── test_d3_d5_temporal.py   # D3-D5: Temporal proofs (THE DIFFERENTIATOR)
├── test_d6_d10_multiagent.py # D6-D10: Multi-agent coordination
├── pytest.ini               # Test configuration
├── README.md                # This file
└── data/                    # Downloaded datasets (auto-created)
```

## Test Categories

### D1-D2: Benchmark Comparability
- **D1 LOCOMO**: Long-term conversational memory benchmark (10 conversations, ~9K tokens each)
- **D2 LongMemEval**: Large-scale memory benchmark (115K-355K tokens per question)

### D3-D5: Temporal Memory (THE DIFFERENTIATOR)
- **D3 Temporal Replay**: Reconstruct exact state at any timestamp T
- **D4 Fact Supersession**: Answers change correctly based on as_of parameter
- **D5 Hash-Chain Provenance**: Cryptographic proof history wasn't tampered

### D6-D10: Multi-Agent Coordination
- **D6 Read Consistency**: Concurrent agents see identical event sets
- **D7 Write Coordination**: All concurrent writes persisted without loss
- **D8 Temporal Consistency**: All agents replaying to T see identical state
- **D9 State Isolation**: Agent-specific user_id creates isolated streams
- **D10 Coordination Primitives**: Leader election, distributed counters

## Running Tests

### Quick Mode (No LLM Required)
```bash
# Temporal tests only - THE DIFFERENTIATOR
pytest test_d3_d5_temporal.py -v

# Multi-agent tests
pytest test_d6_d10_multiagent.py -v

# All non-LLM tests
pytest -v -m "not requires_llm"
```

### Full Benchmark Mode
```bash
# Set LLM API key
export OPENAI_API_KEY="sk-..."  # or ANTHROPIC_API_KEY

# Run LOCOMO benchmark
pytest test_d1_locomo.py -v

# Run LongMemEval benchmark
pytest test_d2_longmemeval.py -v

# Run all tests
pytest -v
```

### Configuration Environment Variables
```bash
# LLM settings
export KRNX_LLM_PROVIDER="openai"        # or "anthropic"
export KRNX_LLM_MODEL="gpt-4o-mini"      # or "claude-sonnet-4-20250514"

# Sample sizes (for speed/accuracy tradeoff)
export KRNX_LOCOMO_SAMPLES="50"          # Questions to test
export KRNX_LONGMEMEVAL_SAMPLES="50"     # Questions to test
export KRNX_LATENCY_SAMPLES="30"         # Measurements for statistical analysis

# Infrastructure
export REDIS_URL="redis://localhost:6379"
export QDRANT_URL="http://localhost:6333"
export KRNX_VECTOR_MEMORY="false"        # Use in-memory vectors

# Quick mode
export KRNX_QUICK_MODE="true"            # Reduce sample sizes
```

## Statistical Methodology

### Latency Measurements (n=30)
- Multiple runs for parametric validity
- Z-score outlier removal (threshold=3.0)
- 95% confidence intervals using t-distribution
- Percentiles: p50, p95, p99

### Effect Size (Cohen's d)
```
Interpretation:
- |d| < 0.2: negligible
- 0.2 ≤ |d| < 0.5: small
- 0.5 ≤ |d| < 0.8: medium
- |d| ≥ 0.8: large
```

### Proof Methodology (P1-P2-P3-P4)
Each test follows a formal proof structure:
- **P1**: Preconditions established
- **P2**: Operation performed
- **P3**: Postconditions verified
- **P4**: Strong assertions with exact values

## Why This Matters

### RAG Systems CANNOT:
- Reconstruct exact state at time T
- Answer "what did we know on Tuesday?"
- Provide cryptographic audit trails
- Guarantee temporal consistency across agents

### KRNX CAN:
- Time-travel to any moment in history
- Answer temporal queries precisely
- Prove history wasn't tampered
- Coordinate multiple agents through shared memory

## Paper-Ready Metrics

Test output is formatted for academic paper inclusion:

```
======================================================================
D3.1 PROOF SUMMARY: Temporal Replay Returns Exact Historical State
======================================================================

PROOF POINTS:
  P1_migration_complete: ✓ PASS
  P2_T1_has_exactly_phase1: ✓ PASS
  P2_T1_no_phase2: ✓ PASS
  ...

METRICS:
  total_events_written: 30
  T1_replay_count: 10
  T2_replay_count: 20

STATISTICAL SUMMARY (n=30, raw_n=30):
  Mean: 12.3456 ± 1.2345 (95% CI: [11.0000, 13.6912], n=30)
  Percentiles: p50=12.1234, p95=14.5678, p99=15.8901

RESULT: ✓ PASSED
======================================================================
```

## Verified API Signatures

All tests use verified API signatures from source code:

```python
# Kernel
from chillbot.kernel.models import Event
Event.compute_hash() -> str
Event.verify_hash_chain(previous_event) -> bool

# Compute
from chillbot.compute.embeddings import EmbeddingEngine
EmbeddingEngine(model_name="all-MiniLM-L6-v2")

from chillbot.compute.vectors import VectorStore, VectorStoreBackend
VectorStore(url="...", backend=VectorStoreBackend.MEMORY)
```

## License

Part of the KRNX project. See main repository for license.
