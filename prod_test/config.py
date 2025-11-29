"""
KRNX Test Harness Configuration

Central configuration for all test layers.
"""

import os
from pathlib import Path

# ==============================================
# PATHS
# ==============================================

# Test harness root
HARNESS_ROOT = Path(__file__).parent

# Reports output directory
REPORTS_DIR = HARNESS_ROOT / "reports"

# Test data directory (ephemeral, created per test run)
TEST_DATA_DIR = HARNESS_ROOT / "test_data"

# ==============================================
# REDIS CONFIGURATION
# ==============================================

REDIS_HOST = os.getenv("KRNX_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("KRNX_REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("KRNX_REDIS_PASSWORD", None)

# ==============================================
# TEST PARAMETERS
# ==============================================

# Layer 1: Kernel Proofs
KERNEL_TEST_CONFIG = {
    # K1: Append-only
    "k1_event_count": 100,
    
    # K2: Hash chain
    "k2_chain_length": 50,
    
    # K3: Replay determinism
    "k3_event_count": 100,
    "k3_replay_iterations": 5,
    
    # K4: Timestamp monotonicity
    "k4_event_count": 200,
    "k4_concurrent_writers": 4,
    
    # K5: STM→LTM migration
    "k5_event_count": 100,
    "k5_max_migration_wait_seconds": 30,
    
    # K6: Crash recovery
    "k6_event_count": 50,
    "k6_simulate_crash": True,
    
    # K7: TTL expiry
    "k7_event_count": 10,
    "k7_ttl_seconds": 2,
    "k7_wait_for_expiry_seconds": 5,
}

# Layer 2: Fabric Proofs (future)
FABRIC_TEST_CONFIG = {
    "f1_stream_event_count": 100,
    "f2_cross_stream_count": 50,
}

# Layer 3: Application Demos (future)
DEMO_TEST_CONFIG = {
    # D1-D2: Needle in haystack
    "haystack_sizes": [100, 500, 1000, 5000],
    "needle_positions": ["early", "middle", "late"],
    
    # D3: Fact versioning
    "fact_versions": 5,
    
    # Statistical validation
    "min_runs_for_significance": 30,
    "confidence_level": 0.95,
}

# ==============================================
# TIMEOUTS
# ==============================================

DEFAULT_TIMEOUT_SECONDS = 30
LTM_MIGRATION_TIMEOUT_SECONDS = 60
WORKER_DRAIN_TIMEOUT_SECONDS = 10

# ==============================================
# STATISTICAL VALIDATION
# ==============================================

# Minimum effect size (Cohen's d) to consider meaningful
MIN_EFFECT_SIZE = 0.5

# P-value threshold for significance
SIGNIFICANCE_THRESHOLD = 0.05

# Number of runs for statistical power
MIN_STATISTICAL_RUNS = 30
