"""
KRNX Layer 2 Fabric Test Fixtures - Production Grade (CORRECTED)

Aligned with actual KRNX implementation APIs.

CORRECTED API SIGNATURES:
- KRNXController.query_events(): NO 'order' param, returns chronological by default
- KRNXController.get_event(event_id): Takes ONLY event_id, not workspace/user
- MemoryFabric.remember(): NO 'timestamp' param
- SalienceEngine.compute(): Uses event_id, timestamp, access_count, avg_similarity
- TemporalEnricher.enrich(): Uses timestamp, previous_event, now (not event object)
"""

import pytest
import time
import uuid
import tempfile
import shutil
import statistics
import math
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import scipy.stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class FabricTestConfig:
    """Configuration aligned with Playbook."""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    vector_backend: str = "memory"
    
    operation_timeout: float = 10.0
    embedding_timeout: float = 30.0
    migration_timeout: float = 30.0
    
    concurrent_configs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(10, 100), (50, 100), (10, 1000)]
    )
    
    min_runs: int = 30
    confidence_level: float = 0.95
    significance_threshold: float = 0.05


@pytest.fixture(scope="session")
def fabric_config() -> FabricTestConfig:
    return FabricTestConfig()


# ==============================================
# STATISTICAL UTILITIES
# ==============================================

@dataclass
class StatisticalResult:
    mean: float
    std_dev: float
    ci_lower: float
    ci_upper: float
    n: int
    p50: float
    p95: float
    p99: float
    values: List[float]
    
    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.std_dev:.4f} (95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}], n={self.n})"


@dataclass
class ComparisonResult:
    group_a: StatisticalResult
    group_b: StatisticalResult
    t_statistic: float
    p_value: float
    effect_size: float
    significant: bool
    
    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"
        return f"t={self.t_statistic:.4f}, p={self.p_value:.6f}, d={self.effect_size:.4f} ({sig})"


@pytest.fixture(scope="session")
def statistical_analyzer(fabric_config):
    class StatisticalAnalyzer:
        def __init__(self, config):
            self.confidence_level = config.confidence_level
            self.significance_threshold = config.significance_threshold
        
        def analyze(self, values: List[float]) -> StatisticalResult:
            n = len(values)
            if n < 2:
                raise ValueError(f"Need at least 2 values, got {n}")
            
            mean = statistics.mean(values)
            std_dev = statistics.stdev(values)
            sorted_vals = sorted(values)
            
            t_crit = scipy_stats.t.ppf((1 + self.confidence_level) / 2, n - 1) if HAS_SCIPY else 2.0
            margin = t_crit * (std_dev / math.sqrt(n))
            
            return StatisticalResult(
                mean=mean,
                std_dev=std_dev,
                ci_lower=mean - margin,
                ci_upper=mean + margin,
                n=n,
                p50=sorted_vals[n // 2],
                p95=sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
                p99=sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
                values=values
            )
        
        def compare(self, group_a: List[float], group_b: List[float]) -> ComparisonResult:
            result_a = self.analyze(group_a)
            result_b = self.analyze(group_b)
            
            if HAS_SCIPY:
                t_stat, p_value = scipy_stats.ttest_ind(group_a, group_b)
            else:
                t_stat, p_value = 0.0, 0.5
            
            pooled_std = math.sqrt(
                ((len(group_a) - 1) * result_a.std_dev**2 + 
                 (len(group_b) - 1) * result_b.std_dev**2) /
                (len(group_a) + len(group_b) - 2)
            ) if (len(group_a) + len(group_b)) > 2 else 1.0
            effect_size = abs(result_a.mean - result_b.mean) / pooled_std if pooled_std > 0 else 0
            
            return ComparisonResult(
                group_a=result_a,
                group_b=result_b,
                t_statistic=t_stat,
                p_value=p_value,
                effect_size=effect_size,
                significant=p_value < self.significance_threshold
            )
    
    return StatisticalAnalyzer(fabric_config)


# ==============================================
# ISOLATION FIXTURES
# ==============================================

@pytest.fixture
def unique_workspace() -> str:
    return f"ws_{uuid.uuid4().hex[:12]}"

@pytest.fixture
def unique_user() -> str:
    return f"user_{uuid.uuid4().hex[:12]}"

@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp(prefix="krnx_l2_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================
# KERNEL & FABRIC FIXTURES
# ==============================================

@pytest.fixture
def fabric_no_embed(temp_data_dir, unique_workspace):
    """Fabric without embedding - for stream/concurrency tests."""
    try:
        from chillbot.fabric.orchestrator import MemoryFabric
        from chillbot.kernel.controller import KRNXController
        from chillbot.kernel.connection_pool import configure_pool
        
        try:
            configure_pool(host="localhost", port=6379, max_connections=200)
        except Exception:
            pass
        
        kernel = KRNXController(
            data_path=temp_data_dir,
            redis_host="localhost",
            redis_port=6379,
            enable_async_worker=True,
            enable_backpressure=True,
            redis_max_connections=200,
        )
        
        fabric = MemoryFabric(
            kernel=kernel,
            auto_embed=False,
            auto_enrich=True,
            default_workspace=unique_workspace,
        )
        
        yield fabric
        
        fabric.close()
        kernel.close()
        
    except ImportError as e:
        pytest.skip(f"MemoryFabric not available: {e}")


# ==============================================
# ENRICHMENT FIXTURES (CORRECTED APIs)
# ==============================================

@pytest.fixture
def feature_extractor():
    try:
        from chillbot.fabric.enrichment.features import FeatureExtractor
        return FeatureExtractor()
    except ImportError as e:
        pytest.skip(f"FeatureExtractor not available: {e}")


@pytest.fixture
def relation_scorer():
    try:
        from chillbot.fabric.enrichment.relations import RelationScorer, RelationScoringConfig
        config = RelationScoringConfig(
            duplicate_threshold=0.95,
            supersede_threshold=0.70,
            expand_threshold=0.50,
            contradict_threshold=0.70,
        )
        return RelationScorer(config=config)
    except ImportError as e:
        pytest.skip(f"RelationScorer not available: {e}")


@pytest.fixture
def salience_engine():
    """CORRECTED: SalienceEngine fixture with proper config."""
    try:
        from chillbot.fabric.enrichment.salience import SalienceEngine, SalienceConfig
        config = SalienceConfig(
            recency_halflife=86400.0,
            recency_min=0.1,
            recency_weight=0.30,
            frequency_weight=0.15,
            semantic_weight=0.40,
            structural_weight=0.15,
        )
        return SalienceEngine(config=config)
    except ImportError as e:
        pytest.skip(f"SalienceEngine not available: {e}")


@pytest.fixture
def retention_classifier():
    try:
        from chillbot.fabric.enrichment.retention_v2 import RetentionClassifier, RetentionConfig
        config = RetentionConfig(
            salience_high_threshold=0.5,
            salience_low_threshold=0.3,
            drift_high_threshold=0.55,
            drift_low_threshold=0.3,
        )
        return RetentionClassifier(config=config)
    except ImportError as e:
        pytest.skip(f"RetentionClassifier not available: {e}")


@pytest.fixture
def temporal_enricher():
    """CORRECTED: TemporalEnricher from temporal module."""
    try:
        from chillbot.fabric.enrichment.temporal import TemporalEnricher
        return TemporalEnricher(episode_gap_threshold=300.0)
    except ImportError as e:
        pytest.skip(f"TemporalEnricher not available: {e}")


@pytest.fixture
def entity_extractor():
    try:
        from chillbot.fabric.enrichment.entities import EntityExtractor
        return EntityExtractor()
    except ImportError as e:
        pytest.skip(f"EntityExtractor not available: {e}")


@pytest.fixture
def identity_resolver():
    try:
        from chillbot.fabric.identity import IdentityResolver
        return IdentityResolver()
    except ImportError as e:
        pytest.skip(f"IdentityResolver not available: {e}")


# ==============================================
# WAIT UTILITIES
# ==============================================

@pytest.fixture
def wait_for_migration(fabric_config):
    def waiter(kernel, expected_count: int, timeout: float = None) -> bool:
        timeout = timeout or fabric_config.migration_timeout
        start = time.time()
        while time.time() - start < timeout:
            metrics = kernel.get_worker_metrics()
            if metrics.messages_processed >= expected_count:
                return True
            time.sleep(0.1)
        return False
    return waiter


# ==============================================
# PROOF SUMMARY
# ==============================================

@pytest.fixture
def print_proof_summary():
    def printer(
        test_id: str,
        guarantee: str,
        metrics: Dict[str, Any],
        result: str,
        statistical: Optional[StatisticalResult] = None,
        comparison: Optional[ComparisonResult] = None,
    ):
        print(f"\n{'='*70}")
        print(f"{test_id} PROOF SUMMARY: {guarantee}")
        print(f"{'='*70}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        if statistical:
            print(f"  STATISTICAL: {statistical}")
        if comparison:
            print(f"  COMPARISON: {comparison}")
        print(f"  RESULT: {result}")
        print(f"{'='*70}\n")
    return printer


# ==============================================
# PARAMETRIZE CONSTANTS
# ==============================================

CONCURRENT_CONFIGS = [(10, 100), (50, 100), (10, 1000)]
SCALE_LEVELS = [100, 1000, 5000]
