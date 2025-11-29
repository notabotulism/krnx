"""
KRNX Layer 3 Demo Tests - Shared Fixtures (Academic Rigor v3)

Layer 3 tests prove what KRNX ENABLES - real AI applications
doing things they couldn't do without temporal memory infrastructure.

ACADEMIC RIGOR FEATURES:
- Statistical analysis: n=30 for latency, CI, effect size, outlier removal
- P1-P2-P3-P4 proof methodology throughout
- Strong assertions with exact expected values
- Dataset auto-download (LOCOMO, LongMemEval)
- Verified API imports against actual source code
- Failure mode testing

VERIFIED API SIGNATURES (from source):
- EmbeddingEngine(model_name=) with .embed(), .embed_batch(), .dimension
- VectorStore(url=, backend=VectorStoreBackend.MEMORY) 
- JobQueue(stream_name=) with .dequeue_batch(consumer_name, count, block_ms)
- Event.compute_hash(), Event.verify_hash_chain(previous_event)
- KRNXController.replay_to_timestamp(workspace_id, user_id, timestamp)
"""

import os
import sys
import pytest
import time
import uuid
import tempfile
import threading
import statistics
import math
import json
import hashlib
import logging
import urllib.request
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# =============================================================================
# PATH SETUP
# =============================================================================

_current_dir = Path(__file__).parent
_prod_test_dir = _current_dir.parent
_project_root = _prod_test_dir.parent

for path in [_project_root, _prod_test_dir]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Data directory for downloaded datasets
DATA_DIR = _current_dir / "data"
DATA_DIR.mkdir(exist_ok=True)


# =============================================================================
# KRNX IMPORTS (verified against actual source)
# =============================================================================

# Kernel imports
try:
    from chillbot.kernel.controller import KRNXController
    from chillbot.kernel.models import Event, create_event
    from chillbot.kernel.exceptions import KRNXError, NotFoundError
    from chillbot.kernel.connection_pool import configure_pool, close_pool, clear_thread_local
    HAS_KERNEL = True
except ImportError as e:
    logger.warning(f"Could not import kernel: {e}")
    HAS_KERNEL = False
    KRNXController = None
    Event = None
    create_event = None
    configure_pool = None
    close_pool = None
    clear_thread_local = None

# Fabric imports
try:
    from chillbot.fabric.orchestrator import MemoryFabric, RecallResult, MemoryItem
    HAS_FABRIC = True
except ImportError as e:
    logger.warning(f"Could not import fabric: {e}")
    HAS_FABRIC = False
    MemoryFabric = None
    RecallResult = None
    MemoryItem = None

# Compute imports (VERIFIED: correct class names from source)
try:
    from chillbot.compute.embeddings import EmbeddingEngine
    from chillbot.compute.vectors import VectorStore, VectorStoreBackend
    from chillbot.compute.queue import JobQueue, JobType
    from chillbot.compute.salience import SalienceEngine, SalienceMethod
    HAS_COMPUTE = True
except ImportError as e:
    logger.warning(f"Could not import compute: {e}")
    HAS_COMPUTE = False
    EmbeddingEngine = None
    VectorStore = None
    VectorStoreBackend = None
    JobQueue = None
    SalienceEngine = None

# Enrichment imports
try:
    from chillbot.fabric.enrichment.retention_v2 import RetentionClassifier
    from chillbot.fabric.enrichment.relations import RelationType, RelationResult
    HAS_ENRICHMENT = True
except ImportError as e:
    logger.warning(f"Could not import enrichment: {e}")
    HAS_ENRICHMENT = False
    RetentionClassifier = None

# Full stack check
HAS_KRNX = HAS_KERNEL and HAS_FABRIC

# LLM client imports (optional)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# =============================================================================
# CONFIGURATION
# =============================================================================

LAYER3_CONFIG = {
    # LLM settings
    "default_model": os.environ.get("KRNX_LLM_MODEL", "gpt-4o-mini"),
    "default_provider": os.environ.get("KRNX_LLM_PROVIDER", "openai"),
    "llm_temperature": 0.0,
    "llm_max_tokens": 1024,
    
    # Benchmark settings
    "locomo_sample_size": int(os.environ.get("KRNX_LOCOMO_SAMPLES", "50")),
    "longmemeval_sample_size": int(os.environ.get("KRNX_LONGMEMEVAL_SAMPLES", "50")),
    
    # Multi-agent settings
    "default_agent_count": 3,
    "agent_coordination_timeout": 30,
    
    # Statistical rigor settings (ACADEMIC GRADE)
    "latency_sample_size": int(os.environ.get("KRNX_LATENCY_SAMPLES", "30")),
    "functional_runs": int(os.environ.get("KRNX_FUNCTIONAL_RUNS", "5")),
    "outlier_z_threshold": 3.0,
    "confidence_level": 0.95,
    "min_samples_parametric": 20,
    
    # Infrastructure
    "redis_url": os.environ.get("REDIS_URL", "redis://localhost:6379"),
    "qdrant_url": os.environ.get("QDRANT_URL", "http://localhost:6333"),
    "use_memory_vectors": os.environ.get("KRNX_VECTOR_MEMORY", "false").lower() == "true",
    
    # Test behavior
    "quick_mode": os.environ.get("KRNX_QUICK_MODE", "false").lower() == "true",
}

# Multi-agent test configurations
MULTI_AGENT_CONFIGS = [
    (2, 50),   # 2 agents, 50 events each
    (3, 100),  # 3 agents, 100 events each  
    (5, 50),   # 5 agents, 50 events each
]


# =============================================================================
# DATASET URLS
# =============================================================================

DATASET_URLS = {
    "locomo": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "longmemeval_s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "longmemeval_m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
    "longmemeval_oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    raw_response: Any = None


@dataclass
class BenchmarkQuestion:
    """A question from LOCOMO or LongMemEval."""
    question_id: str
    question: str
    expected_answer: str
    category: str
    evidence_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of answering a benchmark question."""
    question_id: str
    question: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    confidence: float
    latency_ms: float
    tokens_used: int
    retrieved_events: int
    category: str


@dataclass
class AgentState:
    """State of an agent in multi-agent tests."""
    agent_id: str
    workspace_id: str
    events_written: int = 0
    events_read: int = 0
    queries_made: int = 0
    last_activity: float = 0.0
    errors: List[str] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)


@dataclass
class StatisticalResult:
    """
    Statistical analysis result with academic rigor.
    
    Includes:
    - Central tendency (mean, median)
    - Dispersion (std, variance)
    - Percentiles (p50, p95, p99)
    - Confidence intervals (95% CI)
    - Outlier information
    """
    mean: float
    std: float
    variance: float
    median: float
    p50: float
    p95: float
    p99: float
    min_val: float
    max_val: float
    n: int
    ci_low: float
    ci_high: float
    outliers_removed: int = 0
    raw_n: int = 0  # Original sample size before outlier removal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "variance": round(self.variance, 6),
            "median": round(self.median, 6),
            "p50": round(self.p50, 6),
            "p95": round(self.p95, 6),
            "p99": round(self.p99, 6),
            "min": round(self.min_val, 6),
            "max": round(self.max_val, 6),
            "n": self.n,
            "raw_n": self.raw_n,
            "ci_95": f"[{self.ci_low:.6f}, {self.ci_high:.6f}]",
            "outliers_removed": self.outliers_removed,
        }
    
    def __str__(self) -> str:
        return (
            f"Mean: {self.mean:.4f} ± {self.std:.4f} "
            f"(95% CI: [{self.ci_low:.4f}, {self.ci_high:.4f}], n={self.n})"
        )


@dataclass
class ProofResult:
    """Result of a formal proof with P1-P2-P3-P4 methodology."""
    test_id: str
    guarantee: str
    proofs: Dict[str, bool]  # P1, P2, P3, P4 -> pass/fail
    metrics: Dict[str, Any]
    statistical: Optional[StatisticalResult] = None
    passed: bool = True
    details: str = ""
    
    def __post_init__(self):
        # Auto-compute passed from proofs
        self.passed = all(self.proofs.values())


# =============================================================================
# DATASET DOWNLOADER
# =============================================================================

class DatasetDownloader:
    """
    Download and cache benchmark datasets.
    
    Datasets are downloaded once and cached in data/ directory.
    """
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
    
    def download(self, dataset_name: str, force: bool = False) -> Path:
        """
        Download dataset if not already cached.
        
        Args:
            dataset_name: One of 'locomo', 'longmemeval_s', 'longmemeval_m', 'longmemeval_oracle'
            force: Force re-download even if cached
            
        Returns:
            Path to downloaded file
        """
        if dataset_name not in DATASET_URLS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_URLS.keys())}")
        
        url = DATASET_URLS[dataset_name]
        filename = f"{dataset_name}.json"
        filepath = self.data_dir / filename
        
        if filepath.exists() and not force:
            logger.info(f"Using cached dataset: {filepath}")
            return filepath
        
        logger.info(f"Downloading {dataset_name} from {url}...")
        
        try:
            urllib.request.urlretrieve(url, filepath)
            logger.info(f"Downloaded {dataset_name} to {filepath}")
            return filepath
        except Exception as e:
            raise RuntimeError(f"Failed to download {dataset_name}: {e}")
    
    def get_locomo(self, force: bool = False) -> Path:
        """Download LOCOMO dataset."""
        return self.download("locomo", force)
    
    def get_longmemeval(self, variant: str = "s", force: bool = False) -> Path:
        """
        Download LongMemEval dataset.
        
        Args:
            variant: 's' for short (~115K tokens), 'm' for medium (~355K tokens), 'oracle' for ground truth
        """
        dataset_name = f"longmemeval_{variant}"
        return self.download(dataset_name, force)
    
    def is_cached(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded."""
        filename = f"{dataset_name}.json"
        return (self.data_dir / filename).exists()


# =============================================================================
# STATISTICAL ANALYZER (Academic Rigor)
# =============================================================================

class StatisticalAnalyzer:
    """
    Statistical analysis with academic rigor.
    
    Features:
    - Z-score outlier removal (configurable threshold)
    - 95% confidence intervals (t-distribution for small samples)
    - Effect size calculation (Cohen's d)
    - Normality indicators
    - Multiple run aggregation
    """
    
    def __init__(
        self,
        outlier_threshold: float = LAYER3_CONFIG["outlier_z_threshold"],
        confidence_level: float = LAYER3_CONFIG["confidence_level"],
        min_samples: int = 5,
    ):
        self.outlier_threshold = outlier_threshold
        self.confidence_level = confidence_level
        self.min_samples = min_samples
        
        # t-distribution critical values for 95% CI
        # Approximation for df >= 30: use 1.96, for smaller use table
        self._t_values = {
            5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
            10: 2.228, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        }
    
    def _get_t_value(self, n: int) -> float:
        """Get t-value for confidence interval calculation."""
        df = n - 1
        if df >= 30:
            return 1.96  # Normal approximation
        
        # Find closest value
        for key in sorted(self._t_values.keys()):
            if df <= key:
                return self._t_values[key]
        return 1.96
    
    def remove_outliers(self, values: List[float]) -> Tuple[List[float], int]:
        """
        Remove outliers using Z-score method.
        
        Args:
            values: List of values
            
        Returns:
            (cleaned_values, num_removed)
        """
        if len(values) < self.min_samples:
            return values, 0
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        
        if std == 0:
            return values, 0
        
        cleaned = [
            v for v in values
            if abs(v - mean) / std <= self.outlier_threshold
        ]
        
        return cleaned, len(values) - len(cleaned)
    
    def analyze(
        self,
        values: List[float],
        remove_outliers: bool = True,
    ) -> StatisticalResult:
        """
        Perform full statistical analysis.
        
        Args:
            values: List of measurements
            remove_outliers: Whether to remove outliers via Z-score
            
        Returns:
            StatisticalResult with all metrics
        """
        raw_n = len(values)
        
        if not values:
            return StatisticalResult(
                mean=0, std=0, variance=0, median=0,
                p50=0, p95=0, p99=0,
                min_val=0, max_val=0, n=0, raw_n=0,
                ci_low=0, ci_high=0, outliers_removed=0,
            )
        
        outliers_removed = 0
        if remove_outliers and len(values) >= self.min_samples:
            values, outliers_removed = self.remove_outliers(values)
        
        n = len(values)
        if n == 0:
            return StatisticalResult(
                mean=0, std=0, variance=0, median=0,
                p50=0, p95=0, p99=0,
                min_val=0, max_val=0, n=0, raw_n=raw_n,
                ci_low=0, ci_high=0, outliers_removed=outliers_removed,
            )
        
        mean = statistics.mean(values)
        std = statistics.stdev(values) if n > 1 else 0
        variance = std ** 2
        median = statistics.median(values)
        
        sorted_vals = sorted(values)
        p50 = sorted_vals[int(n * 0.50)] if n > 0 else 0
        p95 = sorted_vals[min(int(n * 0.95), n - 1)] if n > 0 else 0
        p99 = sorted_vals[min(int(n * 0.99), n - 1)] if n > 0 else 0
        
        # Confidence interval using t-distribution
        t_val = self._get_t_value(n)
        se = std / math.sqrt(n) if n > 1 else 0
        ci_low = mean - t_val * se
        ci_high = mean + t_val * se
        
        return StatisticalResult(
            mean=mean,
            std=std,
            variance=variance,
            median=median,
            p50=p50,
            p95=p95,
            p99=p99,
            min_val=min(sorted_vals) if sorted_vals else 0,
            max_val=max(sorted_vals) if sorted_vals else 0,
            n=n,
            raw_n=raw_n,
            ci_low=ci_low,
            ci_high=ci_high,
            outliers_removed=outliers_removed,
        )
    
    def effect_size(
        self,
        group1: List[float],
        group2: List[float],
    ) -> float:
        """
        Calculate Cohen's d effect size.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        
        Args:
            group1: First group measurements
            group2: Second group measurements
            
        Returns:
            Cohen's d (positive means group1 > group2)
        """
        if not group1 or not group2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        
        var1 = statistics.variance(group1) if n1 > 1 else 0
        var2 = statistics.variance(group2) if n2 > 1 else 0
        
        # Pooled standard deviation
        if n1 + n2 <= 2:
            return 0.0
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1
        
        return (mean1 - mean2) / pooled_std
    
    def is_significant(
        self,
        group1: List[float],
        group2: List[float],
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Check if difference between groups is significant.
        
        Uses effect size (Cohen's d) rather than p-value for practical significance.
        
        Returns:
            (is_significant, effect_size)
        """
        d = self.effect_size(group1, group2)
        return abs(d) >= threshold, d


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    """
    
    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider
        self.model = model or LAYER3_CONFIG["default_model"]
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        if self.provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("openai package not installed")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic":
            if not HAS_ANTHROPIC:
                raise ImportError("anthropic package not installed")
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = Anthropic(api_key=api_key)
            
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def complete(
        self,
        prompt: str,
        context: str = "",
        system_prompt: str = None,
        max_tokens: int = None,
        temperature: float = None,
    ) -> LLMResponse:
        """Generate completion from LLM."""
        max_tokens = max_tokens or LAYER3_CONFIG["llm_max_tokens"]
        temperature = temperature if temperature is not None else LAYER3_CONFIG["llm_temperature"]
        
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant with access to the user's memory. "
                "Use the provided context to answer questions accurately. "
                "If the answer isn't in the context, say so clearly."
            )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Context from memory:\n{context}\n\nQuestion: {prompt}"
            })
        else:
            messages.append({"role": "user", "content": prompt})
        
        start_time = time.perf_counter()
        
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            raw = response
            
        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                messages=[m for m in messages if m["role"] != "system"],
                system=system_prompt,
                max_tokens=max_tokens,
            )
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            raw = response
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            raw_response=raw,
        )
    
    def judge_answer(
        self,
        question: str,
        expected: str,
        actual: str,
    ) -> Tuple[bool, float, str]:
        """
        Use LLM to judge if answer is semantically correct.
        
        Returns:
            (is_correct, confidence, explanation)
        """
        judge_prompt = f"""Judge if the actual answer is semantically equivalent to the expected answer.
Be lenient with formatting differences but strict on factual correctness.

QUESTION: {question}
EXPECTED ANSWER: {expected}
ACTUAL ANSWER: {actual}

Respond ONLY in this exact format:
CORRECT: [yes/no]
CONFIDENCE: [0.0-1.0]
EXPLANATION: [one sentence explanation]"""

        response = self.complete(prompt=judge_prompt, temperature=0.0, max_tokens=200)
        
        lines = response.content.strip().split("\n")
        is_correct = False
        confidence = 0.5
        explanation = ""
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith("CORRECT:"):
                is_correct = "yes" in line.lower()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.upper().startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
        
        return is_correct, confidence, explanation


# =============================================================================
# MULTI-AGENT HARNESS (FIXED)
# =============================================================================

@dataclass
class MultiAgentHarness:
    """
    Harness for running multi-agent coordination tests.
    
    FIXES:
    - Uses shared_user_id for cross-agent event visibility
    - Removed duplicate return statement
    - Added event tracking for verification
    - Added query_workspace() for all-events query
    """
    fabric: Any
    workspace_id: str
    agents: Dict[str, AgentState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _barrier: Optional[threading.Barrier] = None
    _event_log: List[Dict] = field(default_factory=list)
    _shared_user_id: str = "shared_workspace_user"
    
    def register_agent(self, agent_id: str) -> AgentState:
        """Register a new agent."""
        with self._lock:
            state = AgentState(
                agent_id=agent_id,
                workspace_id=self.workspace_id,
            )
            self.agents[agent_id] = state
            return state
    
    def agent_write(
        self,
        agent_id: str,
        content: Dict,
        use_shared_user: bool = True,
    ) -> str:
        """
        Agent writes an event to shared workspace.
        
        Args:
            agent_id: The writing agent's ID
            content: Event content
            use_shared_user: If True, use shared_user_id for cross-agent visibility
        """
        user_id = self._shared_user_id if use_shared_user else agent_id
        
        event_id = self.fabric.remember(
            content={**content, "_agent": agent_id, "_timestamp": time.time()},
            workspace_id=self.workspace_id,
            user_id=user_id,
        )
        
        with self._lock:
            self.agents[agent_id].events_written += 1
            self.agents[agent_id].last_activity = time.time()
            self.agents[agent_id].event_ids.append(event_id)
            self._event_log.append({
                "time": time.time(),
                "agent": agent_id,
                "action": "WRITE",
                "event_id": event_id,
                "user_id": user_id,
            })
        
        return event_id
    
    def agent_query(
        self,
        agent_id: str,
        limit: int = 100,
        as_of: float = None,
        user_id: str = None,
    ) -> List:
        """
        Agent queries the shared workspace.
        
        Args:
            agent_id: The querying agent
            limit: Max events to return
            as_of: Timestamp for temporal replay (optional)
            user_id: Filter by user (default: shared_user_id for cross-agent visibility)
        """
        with self._lock:
            self.agents[agent_id].queries_made += 1
        
        query_user = user_id if user_id is not None else self._shared_user_id
        
        if as_of:
            events = self.fabric.kernel.replay_to_timestamp(
                workspace_id=self.workspace_id,
                user_id=query_user,
                timestamp=as_of,
            )
        else:
            events = self.fabric.kernel.query_events(
                workspace_id=self.workspace_id,
                user_id=query_user,
                limit=limit,
            )
        
        # Convert to list if generator
        events = list(events) if not isinstance(events, list) else events
        
        with self._lock:
            self.agents[agent_id].events_read += len(events)
            self.agents[agent_id].last_activity = time.time()
            self._event_log.append({
                "time": time.time(),
                "agent": agent_id,
                "action": "QUERY",
                "user_id": query_user,
                "events_returned": len(events),
                "as_of": as_of,
            })
        
        return events
    
    def query_workspace(self, limit: int = 1000) -> List:
        """Query all events in workspace using shared_user_id."""
        events = self.fabric.kernel.query_events(
            workspace_id=self.workspace_id,
            user_id=self._shared_user_id,
            limit=limit,
        )
        return list(events) if not isinstance(events, list) else events
    
    def set_barrier(self, n_agents: int):
        """Set up barrier for synchronized agent starts."""
        self._barrier = threading.Barrier(n_agents)
    
    def wait_at_barrier(self):
        """Wait at the barrier."""
        if self._barrier:
            self._barrier.wait()
    
    def get_event_log(self) -> List[Dict]:
        """Get the full event log."""
        with self._lock:
            return list(self._event_log)
    
    def get_agent_stats(self) -> Dict[str, Dict]:
        """Get statistics for all agents."""
        with self._lock:
            return {
                agent_id: {
                    "events_written": state.events_written,
                    "events_read": state.events_read,
                    "queries_made": state.queries_made,
                    "errors": len(state.errors),
                    "event_ids": list(state.event_ids),
                }
                for agent_id, state in self.agents.items()
            }
    
    def verify_event_count(self, expected: int) -> Tuple[bool, int]:
        """
        Verify total events in workspace matches expected.
        
        Returns:
            (passed, actual_count)
        """
        events = self.query_workspace(limit=expected + 100)
        actual = len(events)
        return actual == expected, actual
    
    def get_total_events_written(self) -> int:
        """Get total events written by all agents."""
        with self._lock:
            return sum(s.events_written for s in self.agents.values())


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def dataset_downloader():
    """Session-scoped dataset downloader."""
    return DatasetDownloader()


@pytest.fixture(scope="session")
def statistical_analyzer():
    """Session-scoped statistical analyzer."""
    return StatisticalAnalyzer()


# =============================================================================
# SESSION-SCOPED REDIS POOL CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def configure_redis_pool():
    """
    Configure Redis connection pool once for entire test session.
    
    This prevents issues with the singleton pattern where multiple
    KRNXController instances would try to reconfigure the pool.
    """
    if not HAS_KERNEL or configure_pool is None:
        yield
        return
    
    # Parse Redis URL
    redis_url = LAYER3_CONFIG["redis_url"]
    redis_host = "localhost"
    redis_port = 6379
    
    if redis_url.startswith("redis://"):
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        if len(parts) > 1:
            redis_port = int(parts[1].split("/")[0])
    
    # Configure pool once at session start
    try:
        configure_pool(host=redis_host, port=redis_port, max_connections=200)
        logger.info(f"[POOL] Configured Redis pool: {redis_host}:{redis_port}")
    except Exception as e:
        logger.warning(f"[POOL] Pool already configured or error: {e}")
    
    yield
    
    # Cleanup at session end
    try:
        if close_pool:
            close_pool()
            logger.info("[POOL] Closed Redis pool")
    except Exception as e:
        logger.warning(f"[POOL] Error closing pool: {e}")


@pytest.fixture
def unique_workspace() -> str:
    """Generate unique workspace ID."""
    return f"l3-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def unique_user() -> str:
    """Generate unique user ID."""
    return f"l3-user-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def kernel(configure_redis_pool):
    """Fresh KRNXController instance (uses session-scoped Redis pool)."""
    if not HAS_KERNEL:
        pytest.skip("KRNX kernel not available")
    
    test_dir = tempfile.mkdtemp(prefix="krnx_l3_kernel_")
    
    # Parse Redis URL for KRNXController (it will use existing pool)
    redis_url = LAYER3_CONFIG["redis_url"]
    redis_host = "localhost"
    redis_port = 6379
    
    if redis_url.startswith("redis://"):
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        if len(parts) > 1:
            redis_port = int(parts[1].split("/")[0])
    
    controller = KRNXController(
        data_path=test_dir,
        redis_host=redis_host,
        redis_port=redis_port,
        enable_hash_chain=True,
        enable_backpressure=False,
    )
    
    yield controller
    
    # Don't close pool - managed by session fixture
    controller.shutdown(close_connection_pool=False)
    
    # Clear thread-local cache for next test
    if clear_thread_local:
        clear_thread_local()
    
    import shutil
    try:
        shutil.rmtree(test_dir, ignore_errors=True)
    except:
        pass


@pytest.fixture
def fabric_no_embed(configure_redis_pool):
    """Fabric without embeddings (faster for non-semantic tests)."""
    if not HAS_KRNX:
        pytest.skip("KRNX components not available")
    
    test_dir = tempfile.mkdtemp(prefix="krnx_l3_fabric_")
    
    # Parse Redis URL
    redis_url = LAYER3_CONFIG["redis_url"]
    redis_host = "localhost"
    redis_port = 6379
    
    if redis_url.startswith("redis://"):
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        if len(parts) > 1:
            redis_port = int(parts[1].split("/")[0])
    
    kernel = KRNXController(
        data_path=test_dir,
        redis_host=redis_host,
        redis_port=redis_port,
        enable_hash_chain=True,
        enable_backpressure=False,
    )
    
    fabric = MemoryFabric(
        kernel=kernel,
        auto_embed=False,
        auto_enrich=False,
    )
    
    yield fabric
    
    fabric.close()
    kernel.shutdown(close_connection_pool=False)
    
    # Clear thread-local cache for next test
    if clear_thread_local:
        clear_thread_local()
    
    import shutil
    try:
        shutil.rmtree(test_dir, ignore_errors=True)
    except:
        pass


@pytest.fixture
def fabric_with_vectors(configure_redis_pool):
    """Fabric WITH embeddings and vector store."""
    if not HAS_KRNX:
        pytest.skip("KRNX components not available")
    if not HAS_COMPUTE:
        pytest.skip("Compute modules not available")
    
    test_dir = tempfile.mkdtemp(prefix="krnx_l3_vectors_")
    
    # Parse Redis URL
    redis_url = LAYER3_CONFIG["redis_url"]
    redis_host = "localhost"
    redis_port = 6379
    
    if redis_url.startswith("redis://"):
        parts = redis_url.replace("redis://", "").split(":")
        redis_host = parts[0]
        if len(parts) > 1:
            redis_port = int(parts[1].split("/")[0])
    
    kernel = KRNXController(
        data_path=test_dir,
        redis_host=redis_host,
        redis_port=redis_port,
        enable_hash_chain=True,
        enable_backpressure=False,
    )
    
    try:
        embeddings = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        kernel.shutdown(close_connection_pool=False)
        pytest.skip(f"Could not initialize embeddings: {e}")
    
    qdrant_url = LAYER3_CONFIG["qdrant_url"]
    use_memory = LAYER3_CONFIG["use_memory_vectors"]
    
    try:
        if use_memory:
            vectors = VectorStore(
                url=qdrant_url,
                backend=VectorStoreBackend.MEMORY,
            )
        else:
            vectors = VectorStore(
                url=qdrant_url,
                backend=VectorStoreBackend.QDRANT,
            )
    except Exception as e:
        kernel.shutdown(close_connection_pool=False)
        pytest.skip(f"Could not initialize vector store: {e}")
    
    fabric = MemoryFabric(
        kernel=kernel,
        embeddings=embeddings,
        vectors=vectors,
        auto_embed=True,
        auto_enrich=False,
    )
    
    yield fabric
    
    fabric.close()
    kernel.shutdown(close_connection_pool=False)
    
    # Clear thread-local cache for next test
    if clear_thread_local:
        clear_thread_local()
    
    import shutil
    try:
        shutil.rmtree(test_dir, ignore_errors=True)
    except:
        pass


@pytest.fixture
def multi_agent_harness(fabric_no_embed, unique_workspace):
    """Multi-agent test harness."""
    return MultiAgentHarness(
        fabric=fabric_no_embed,
        workspace_id=unique_workspace,
    )


@pytest.fixture(scope="session")
def llm_client():
    """Session-scoped LLM client."""
    provider = LAYER3_CONFIG["default_provider"]
    model = LAYER3_CONFIG["default_model"]
    
    try:
        return LLMClient(provider=provider, model=model)
    except (ImportError, ValueError) as e:
        pytest.skip(f"LLM client not available: {e}")


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def wait_for_migration():
    """
    Wait for STM→LTM migration to complete.
    
    Usage:
        # Simple - uses fabric's default workspace
        wait_for_migration(fabric, 10)
        
        # With explicit workspace/user
        wait_for_migration(fabric, 10, workspace_id="ws", user_id="user")
    """
    def _wait(
        kernel_or_fabric, 
        expected_count: int, 
        timeout: int = 30, 
        workspace_id: str = None, 
        user_id: str = None
    ) -> bool:
        kernel = kernel_or_fabric
        if hasattr(kernel_or_fabric, 'kernel'):
            kernel = kernel_or_fabric.kernel
        
        # Default workspace/user if not provided
        workspace_id = workspace_id or getattr(kernel_or_fabric, 'default_workspace', 'default')
        user_id = user_id or 'default'
        
        start = time.time()
        while time.time() - start < timeout:
            try:
                # Method 1: Use LTM stats (global count)
                if hasattr(kernel, 'ltm') and hasattr(kernel.ltm, 'get_stats'):
                    stats = kernel.ltm.get_stats()
                    count = stats.get("warm_events", 0) + stats.get("cold_events", 0)
                    if count >= expected_count:
                        return True
                
                # Method 2: Query specific workspace/user
                events = kernel.query_events(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    limit=expected_count + 100
                )
                events = list(events) if not isinstance(events, list) else events
                count = len(events)
                
                if count >= expected_count:
                    return True
                    
            except Exception as e:
                logger.debug(f"wait_for_migration check failed: {e}")
            
            time.sleep(0.1)
        
        return False
    
    return _wait


@pytest.fixture
def verify_hash_chain():
    """
    Verify hash chain integrity using Event model's built-in methods.
    
    Returns callable that verifies chain and returns (is_valid, broken_at, message).
    """
    def _verify(events: List) -> Tuple[bool, int, str]:
        if not events:
            return True, -1, "Empty chain is valid"
        
        if len(events) == 1:
            # Single event - check it has no previous_hash (or previous_hash is None)
            first = events[0]
            if hasattr(first, 'previous_hash') and first.previous_hash is not None:
                return False, 0, "First event should have no previous_hash"
            return True, -1, "Single event chain is valid"
        
        # Sort by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Verify chain
        for i in range(1, len(sorted_events)):
            current = sorted_events[i]
            previous = sorted_events[i - 1]
            
            # Use built-in verification if available
            if hasattr(current, 'verify_hash_chain'):
                if not current.verify_hash_chain(previous):
                    return False, i, f"Chain broken at index {i}: verify_hash_chain() failed"
            elif hasattr(current, 'previous_hash') and hasattr(previous, 'compute_hash'):
                expected = previous.compute_hash()
                if current.previous_hash != expected:
                    return False, i, f"Chain broken at index {i}: hash mismatch"
        
        return True, -1, f"Chain valid ({len(sorted_events)} events)"
    
    return _verify


@pytest.fixture
def print_proof_summary():
    """Print formatted proof summary for academic paper inclusion."""
    def _print(proof: ProofResult):
        print(f"\n{'='*70}")
        print(f"{proof.test_id} PROOF SUMMARY: {proof.guarantee}")
        print(f"{'='*70}")
        
        # Print proof points
        print("\nPROOF POINTS:")
        for proof_id, passed in proof.proofs.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {proof_id}: {status}")
        
        # Print metrics
        print("\nMETRICS:")
        for key, value in proof.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        
        # Print statistical summary
        if proof.statistical and proof.statistical.n > 0:
            print(f"\nSTATISTICAL SUMMARY (n={proof.statistical.n}, raw_n={proof.statistical.raw_n}):")
            print(f"  {proof.statistical}")
            print(f"  Percentiles: p50={proof.statistical.p50:.4f}, p95={proof.statistical.p95:.4f}, p99={proof.statistical.p99:.4f}")
            if proof.statistical.outliers_removed > 0:
                print(f"  Outliers removed: {proof.statistical.outliers_removed}")
        
        # Print result
        result_status = "✓ PASSED" if proof.passed else "✗ FAILED"
        print(f"\nRESULT: {result_status}")
        if proof.details:
            print(f"DETAILS: {proof.details}")
        print(f"{'='*70}\n")
    
    return _print


# =============================================================================
# MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "layer3: Layer 3 demo tests")
    config.addinivalue_line("markers", "requires_llm: Requires LLM API key")
    config.addinivalue_line("markers", "requires_redis: Requires Redis")
    config.addinivalue_line("markers", "requires_qdrant: Requires Qdrant")
    config.addinivalue_line("markers", "benchmark: Benchmark test (LOCOMO, LongMemEval)")
    config.addinivalue_line("markers", "multiagent: Multi-agent coordination test")
    config.addinivalue_line("markers", "temporal: Temporal replay test")
    config.addinivalue_line("markers", "hashchain: Hash chain integrity test")
    config.addinivalue_line("markers", "failure_mode: Failure mode test")
    config.addinivalue_line("markers", "slow: Slow test (>30s)")
