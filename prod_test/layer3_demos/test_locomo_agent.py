"""
KRNX + ChillBotAgent - LOCOMO Benchmark Test

OBJECTIVE: Evaluate KRNX's memory capabilities against the LOCOMO benchmark
using the same methodology as Mem0's published paper (arXiv:2504.19413).

METHODOLOGY:
- Dataset: LOCOMO (ACL 2024) - 10 conversations, ~400 turns each, 1986 questions
- Evaluation: LLM-as-judge (binary correct/incorrect)
- Comparison: Mem0 published 66.9% (graph variant), OpenAI baseline 52.9%
- Our modes: raw (fabric only), agent_rules, agent_llm, agent_hybrid

TRANSPARENCY:
- All results exported to JSON with full provenance
- Per-question results available for inspection
- Latency and token usage tracked
- Failures logged, not hidden

REPRODUCIBILITY:
- Random seed fixed
- Temperature=0 for all LLM calls
- Sample selection is deterministic (first N or stratified)
- Full configuration exported

FAIRNESS:
- No benchmark-specific tuning in the agent
- Same LLM (Claude) for generation and judging
- Baseline (no memory) included for reference
- Multiple runs for statistical validity

Author: KRNX Team
Date: 2024
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

# ==============================================
# PATH SETUP
# ==============================================
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent  # D:\chillbot

# Add project root to path for chillbot and chillbot_agent imports
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration - all settings in one place."""
    
    # Dataset
    dataset_path: str = "./data/locomo.json"
    sample_size: int = 100              # Questions per run (use -1 for all)
    stratified_sampling: bool = True    # Equal samples per category
    random_seed: int = 42               # For reproducibility
    
    # Runs
    num_runs: int = 3                   # Runs per mode for statistical validity
    
    # Modes to test
    modes: List[str] = field(default_factory=lambda: [
        "baseline",         # No memory (LLM only)
        "raw",              # fabric.recall() only
        "agent_rules",      # ChillBotAgent with rules planner
        "agent_hybrid",     # ChillBotAgent with hybrid planner
        # "agent_llm",      # Uncomment for full LLM planning (expensive)
    ])
    
    # LLM settings
    llm_temperature: float = 0.0        # Deterministic
    llm_max_tokens: int = 500
    judge_temperature: float = 0.0      # Deterministic judging
    
    # Output
    results_dir: str = "./results"
    export_per_question: bool = True    # Export individual Q&A results
    verbose: bool = True
    
    # Timeouts
    question_timeout: float = 60.0      # Seconds per question
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================
# RESULT STRUCTURES
# ==============================================

@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question: str
    expected_answer: str
    actual_answer: str
    
    # Scoring
    score: float                    # 0.0 or 1.0 (binary)
    judge_reasoning: str = ""       # LLM judge explanation
    
    # Metadata
    category: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    conversation_id: str = ""
    
    # Performance
    latency_ms: float = 0.0
    events_retrieved: int = 0
    events_used: int = 0
    
    # Provenance
    mode: str = ""
    query_intent: str = ""
    run_id: int = 0
    
    # Error tracking
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    """Result for a single benchmark run."""
    run_id: int
    mode: str
    
    # Scores
    accuracy: float = 0.0
    total_questions: int = 0
    correct_count: int = 0
    error_count: int = 0
    
    # Per-category breakdown
    category_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Performance
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Individual results
    questions: List[QuestionResult] = field(default_factory=list)
    
    # Metadata
    started_at: str = ""
    completed_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["questions"] = [q.to_dict() for q in self.questions]
        return result


@dataclass 
class BenchmarkReport:
    """Complete benchmark report."""
    
    # Summary
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Results by mode
    results_by_mode: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # All runs
    all_runs: List[RunResult] = field(default_factory=list)
    
    # Dataset info
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    
    # Comparison to published results
    comparison: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["all_runs"] = [r.to_dict() for r in self.all_runs]
        return result


# ==============================================
# LLM CLIENT WRAPPER
# ==============================================

class LLMClient:
    """
    Wrapper for LLM API calls.
    
    Supports OpenAI API.
    Tracks token usage for cost estimation.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self._client = None
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI()
            logging.info(f"[LLM] Initialized OpenAI client (model={self.model})")
        except ImportError:
            logging.error("[LLM] openai package not installed")
            raise
    
    def complete(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate completion."""
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=tokens,
            temperature=temp,
        )
        
        # Track usage
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens
        
        return response.choices[0].message.content
    
    def get_usage(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }


# ==============================================
# LLM JUDGE
# ==============================================

class LLMJudge:
    """
    LLM-as-judge evaluator.
    
    Same methodology as Mem0 paper: binary correct/incorrect scoring
    based on semantic equivalence, not exact match.
    """
    
    JUDGE_PROMPT = """You are evaluating if an answer is correct based on the expected answer.

IMPORTANT: Judge semantic correctness, not exact wording.
- The answer is CORRECT if it conveys the same information as expected
- Minor wording differences are acceptable
- The answer is INCORRECT if it contradicts, is missing key information, or says "I don't know" when information was available

Expected Answer: {expected}

Actual Answer: {actual}

Respond with ONLY one of these two words:
CORRECT
INCORRECT"""

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._cache: Dict[str, Tuple[float, str]] = {}
    
    def judge(
        self,
        expected: str,
        actual: str,
    ) -> Tuple[float, str]:
        """
        Judge if answer is correct.
        
        Returns:
            (score, reasoning) - score is 0.0 or 1.0
        """
        # Check cache
        cache_key = f"{expected}|||{actual}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        prompt = self.JUDGE_PROMPT.format(expected=expected, actual=actual)
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                temperature=0.0,
                max_tokens=50,
            )
            
            response_upper = response.strip().upper()
            
            if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
                result = (1.0, "correct")
            else:
                result = (0.0, "incorrect")
            
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logging.error(f"[JUDGE] Evaluation failed: {e}")
            return (0.0, f"error: {str(e)}")


# ==============================================
# DATASET LOADER
# ==============================================

class LOCOMODataset:
    """
    LOCOMO dataset loader.
    
    Dataset structure (actual format):
    [
      {
        "qa": [
          {question, answer, evidence: ["D1:3"], category: 1-4},
          ...
        ],
        "conversation": {
          "session_1": [...],
          "session_1_time": "...",
          ...
        },
        ...
      },
      ...
    ]
    
    Categories:
    1 = single_hop
    2 = temporal  
    3 = multi_hop
    4 = open_domain
    5 = adversarial (if present)
    """
    
    CATEGORY_MAP = {
        1: "single_hop",
        2: "temporal",
        3: "multi_hop", 
        4: "open_domain",
        5: "adversarial",
    }
    
    CATEGORIES = ["single_hop", "temporal", "multi_hop", "open_domain", "adversarial"]
    
    def __init__(self, path: str):
        self.path = path
        self.data = None
        self.conversations = {}
        self.questions = []
        
        self._load()
    
    def _load(self):
        """Load dataset from JSON."""
        with open(self.path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Handle list format (actual LOCOMO format)
        if isinstance(raw_data, list):
            for i, item in enumerate(raw_data):
                conv_id = f"D{i+1}"
                
                # Parse conversation
                conversation = item.get("conversation", {})
                turns = []
                
                # Extract sessions (session_1, session_2, etc.)
                session_idx = 1
                while f"session_{session_idx}" in conversation:
                    session_key = f"session_{session_idx}"
                    time_key = f"session_{session_idx}_date_time"
                    session_turns = conversation.get(session_key, [])
                    session_time = conversation.get(time_key, "")
                    
                    for turn_idx, turn in enumerate(session_turns):
                        if isinstance(turn, dict):
                            turns.append({
                                "speaker": turn.get("speaker", turn.get("role", "unknown")),
                                "text": turn.get("text", turn.get("content", "")),
                                "dia_id": turn.get("dia_id", f"{conv_id}:{session_idx}:{turn_idx}"),
                                "session": session_idx,
                                "session_time": session_time,
                            })
                        elif isinstance(turn, str):
                            turns.append({
                                "speaker": "unknown",
                                "text": turn,
                                "dia_id": f"{conv_id}:{session_idx}:{turn_idx}",
                                "session": session_idx,
                                "session_time": session_time,
                            })
                    
                    session_idx += 1
                
                self.conversations[conv_id] = {
                    "turns": turns,
                    "speaker_a": conversation.get("speaker_a", "A"),
                    "speaker_b": conversation.get("speaker_b", "B"),
                    "raw": item,
                }
                
                # Parse questions from "qa" field
                qa_list = item.get("qa", item.get("question_answer", []))
                for qa_idx, qa in enumerate(qa_list):
                    category_raw = qa.get("category", 0)
                    # Convert integer category to string
                    if isinstance(category_raw, int):
                        category = self.CATEGORY_MAP.get(category_raw, f"category_{category_raw}")
                    else:
                        category = str(category_raw)
                    
                    self.questions.append({
                        "conversation_id": conv_id,
                        "question": qa.get("question", ""),
                        "answer": str(qa.get("answer", "")),  # Ensure string
                        "evidence": qa.get("evidence", []),
                        "category": category,
                        "question_id": f"{conv_id}_q{qa_idx}",
                    })
        
        # Handle dict format (alternative)
        elif isinstance(raw_data, dict):
            for conv_id, conv_data in raw_data.items():
                if not isinstance(conv_data, dict):
                    continue
                self._parse_conversation_dict(conv_id, conv_data)
        
        logging.info(
            f"[DATASET] Loaded {len(self.conversations)} conversations, "
            f"{len(self.questions)} questions"
        )
    
    def _parse_conversation_dict(self, conv_id: str, conv_data: dict):
        """Parse conversation from dict format."""
        conversation = conv_data.get("conversation", {})
        turns = []
        
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_key = f"session_{session_idx}"
            session_turns = conversation[session_key]
            
            for turn in session_turns:
                turns.append({
                    "speaker": turn.get("speaker", "unknown"),
                    "text": turn.get("text", ""),
                    "dia_id": turn.get("dia_id", ""),
                    "session": session_idx,
                })
            
            session_idx += 1
        
        self.conversations[conv_id] = {
            "turns": turns,
        }
        
        qa_list = conv_data.get("qa", conv_data.get("question_answer", []))
        for qa in qa_list:
            category_raw = qa.get("category", 0)
            if isinstance(category_raw, int):
                category = self.CATEGORY_MAP.get(category_raw, f"category_{category_raw}")
            else:
                category = str(category_raw)
            
            self.questions.append({
                "conversation_id": conv_id,
                "question": qa.get("question", ""),
                "answer": str(qa.get("answer", "")),
                "evidence": qa.get("evidence", []),
                "category": category,
            })
        
        logging.info(
            f"[DATASET] Loaded {len(self.conversations)} conversations, "
            f"{len(self.questions)} questions"
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        category_counts = defaultdict(int)
        for q in self.questions:
            category_counts[q["category"]] += 1
        
        total_turns = sum(
            len(c["turns"]) for c in self.conversations.values()
        )
        
        return {
            "conversations": len(self.conversations),
            "total_turns": total_turns,
            "total_questions": len(self.questions),
            "questions_by_category": dict(category_counts),
        }
    
    def sample_questions(
        self,
        n: int,
        stratified: bool = True,
        seed: int = 42,
    ) -> List[Dict]:
        """
        Sample questions for benchmark.
        
        Args:
            n: Number of questions (-1 for all)
            stratified: Equal samples per category
            seed: Random seed
        
        Returns:
            List of question dicts
        """
        if n == -1 or n >= len(self.questions):
            return self.questions.copy()
        
        random.seed(seed)
        
        if not stratified:
            return random.sample(self.questions, n)
        
        # Stratified sampling
        by_category = defaultdict(list)
        for q in self.questions:
            by_category[q["category"]].append(q)
        
        per_category = n // len(by_category)
        remainder = n % len(by_category)
        
        sampled = []
        for i, (cat, questions) in enumerate(sorted(by_category.items())):
            count = per_category + (1 if i < remainder else 0)
            count = min(count, len(questions))
            sampled.extend(random.sample(questions, count))
        
        random.shuffle(sampled)
        return sampled[:n]


# ==============================================
# BENCHMARK RUNNER
# ==============================================

class LOCOMOBenchmark:
    """
    LOCOMO benchmark runner.
    
    Tests KRNX + ChillBotAgent against published baselines.
    """
    
    def __init__(
        self,
        config: BenchmarkConfig,
        fabric=None,
        kernel=None,
        llm: Optional[LLMClient] = None,
    ):
        self.config = config
        self.fabric = fabric
        self.kernel = kernel
        self.llm = llm or LLMClient()
        
        self.judge = LLMJudge(self.llm)
        self.dataset = None
        self._adapter = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging."""
        level = logging.DEBUG if self.config.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
        )
    
    def _get_adapter(self, mode: str):
        """Get or create benchmark adapter for mode."""
        from chillbot_agent import BenchmarkAdapter
        
        return BenchmarkAdapter(
            fabric=self.fabric,
            kernel=self.kernel,
            llm=self.llm,
            mode=mode if mode != "baseline" else "raw",
        )
    
    def run(self) -> BenchmarkReport:
        """
        Run complete benchmark.
        
        Returns:
            BenchmarkReport with all results
        """
        start_time = datetime.now()
        logging.info("=" * 60)
        logging.info("KRNX + ChillBotAgent LOCOMO Benchmark")
        logging.info("=" * 60)
        logging.info(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")
        
        # Load dataset
        self.dataset = LOCOMODataset(self.config.dataset_path)
        dataset_info = self.dataset.get_info()
        logging.info(f"Dataset: {dataset_info}")
        
        # Sample questions
        questions = self.dataset.sample_questions(
            n=self.config.sample_size,
            stratified=self.config.stratified_sampling,
            seed=self.config.random_seed,
        )
        logging.info(f"Sampled {len(questions)} questions")
        
        # Run benchmark for each mode
        all_runs = []
        results_by_mode = {}
        
        for mode in self.config.modes:
            logging.info(f"\n{'='*60}")
            logging.info(f"MODE: {mode}")
            logging.info(f"{'='*60}")
            
            mode_runs = []
            
            for run_id in range(self.config.num_runs):
                logging.info(f"\n--- Run {run_id + 1}/{self.config.num_runs} ---")
                
                run_result = self._run_single(
                    mode=mode,
                    run_id=run_id,
                    questions=questions,
                )
                
                mode_runs.append(run_result)
                all_runs.append(run_result)
                
                logging.info(
                    f"Run {run_id + 1} complete: "
                    f"accuracy={run_result.accuracy:.1%}, "
                    f"latency={run_result.avg_latency_ms:.1f}ms"
                )
            
            # Aggregate mode results
            results_by_mode[mode] = self._aggregate_runs(mode_runs)
        
        # Build report
        report = BenchmarkReport(
            timestamp=start_time.isoformat(),
            config=self.config.to_dict(),
            results_by_mode=results_by_mode,
            all_runs=all_runs,
            dataset_info=dataset_info,
            comparison=self._build_comparison(results_by_mode),
        )
        
        # Export results
        self._export_report(report)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _run_single(
        self,
        mode: str,
        run_id: int,
        questions: List[Dict],
    ) -> RunResult:
        """Run a single benchmark iteration."""
        started_at = datetime.now().isoformat()
        
        result = RunResult(
            run_id=run_id,
            mode=mode,
            started_at=started_at,
        )
        
        # Get unique conversations needed
        conv_ids = set(q["conversation_id"] for q in questions)
        
        # Ingest conversations (skip for baseline)
        if mode != "baseline":
            adapter = self._get_adapter(mode)
            
            for conv_id in conv_ids:
                conv = self.dataset.conversations[conv_id]
                workspace_id = f"locomo_{conv_id}_run{run_id}"
                
                # Clear previous data
                adapter.clear_workspace(workspace_id)
                
                # Ingest turns
                adapter.ingest_conversation(
                    turns=conv["turns"],
                    workspace_id=workspace_id,
                    user_id="benchmark",
                )
        
        # Answer questions
        category_results = defaultdict(lambda: {"correct": 0, "total": 0})
        
        for i, q in enumerate(questions):
            if (i + 1) % 10 == 0:
                logging.info(f"  Progress: {i + 1}/{len(questions)}")
            
            q_result = self._answer_question(
                mode=mode,
                run_id=run_id,
                question=q,
            )
            
            result.questions.append(q_result)
            result.total_latency_ms += q_result.latency_ms
            
            # Track category
            cat = q_result.category
            category_results[cat]["total"] += 1
            if q_result.score > 0:
                category_results[cat]["correct"] += 1
                result.correct_count += 1
            
            if q_result.error:
                result.error_count += 1
        
        # Compute final stats
        result.total_questions = len(questions)
        result.accuracy = result.correct_count / result.total_questions if result.total_questions > 0 else 0
        result.avg_latency_ms = result.total_latency_ms / result.total_questions if result.total_questions > 0 else 0
        
        # Category breakdown
        for cat, stats in category_results.items():
            result.category_scores[cat] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
        
        result.completed_at = datetime.now().isoformat()
        return result
    
    def _answer_question(
        self,
        mode: str,
        run_id: int,
        question: Dict,
    ) -> QuestionResult:
        """Answer a single benchmark question."""
        q_result = QuestionResult(
            question_id=f"{question['conversation_id']}_{question['evidence'][0] if question['evidence'] else 'unknown'}",
            question=question["question"],
            expected_answer=question["answer"],
            actual_answer="",
            score=0.0,
            category=question["category"],
            evidence_ids=question.get("evidence", []),
            conversation_id=question["conversation_id"],
            mode=mode,
            run_id=run_id,
        )
        
        start_time = time.time()
        workspace_id = f"locomo_{question['conversation_id']}_run{run_id}"
        
        try:
            if mode == "baseline":
                # No memory - just ask LLM directly
                answer = self._answer_baseline(question["question"])
            else:
                adapter = self._get_adapter(mode)
                result = adapter.answer_question(
                    question=question["question"],
                    workspace_id=workspace_id,
                    user_id="benchmark",
                )
                answer = result.actual_answer
                q_result.events_retrieved = result.events_retrieved
                q_result.events_used = result.events_used
                q_result.query_intent = result.query_intent
            
            q_result.actual_answer = answer
            q_result.latency_ms = (time.time() - start_time) * 1000
            
            # Judge the answer
            score, reasoning = self.judge.judge(
                expected=question["answer"],
                actual=answer,
            )
            q_result.score = score
            q_result.judge_reasoning = reasoning
            
        except Exception as e:
            logging.error(f"Error on question {q_result.question_id}: {e}")
            q_result.error = str(e)
            q_result.latency_ms = (time.time() - start_time) * 1000
        
        return q_result
    
    def _answer_baseline(self, question: str) -> str:
        """Answer without any memory (baseline)."""
        system_prompt = """You are answering questions about conversations.
You do NOT have access to any conversation history.
If you don't know the answer, say "I don't have that information."
Do not make up answers."""
        
        return self.llm.complete(
            prompt=f"Question: {question}\n\nAnswer:",
            system_prompt=system_prompt,
            max_tokens=self.config.llm_max_tokens,
        )
    
    def _aggregate_runs(self, runs: List[RunResult]) -> Dict[str, Any]:
        """Aggregate multiple runs into summary statistics."""
        accuracies = [r.accuracy for r in runs]
        latencies = [r.avg_latency_ms for r in runs]
        
        # Aggregate category scores
        category_agg = defaultdict(list)
        for run in runs:
            for cat, scores in run.category_scores.items():
                category_agg[cat].append(scores["accuracy"])
        
        return {
            "accuracy_mean": sum(accuracies) / len(accuracies),
            "accuracy_std": self._std(accuracies),
            "accuracy_min": min(accuracies),
            "accuracy_max": max(accuracies),
            "latency_mean": sum(latencies) / len(latencies),
            "latency_std": self._std(latencies),
            "num_runs": len(runs),
            "category_breakdown": {
                cat: {
                    "mean": sum(scores) / len(scores),
                    "std": self._std(scores),
                }
                for cat, scores in category_agg.items()
            },
        }
    
    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _build_comparison(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Build comparison against published baselines."""
        comparison = {
            "published_baselines": {
                "mem0_graph": 0.669,        # Mem0 paper (graph variant)
                "mem0_vector": 0.635,       # Mem0 paper (vector only)
                "openai_assistants": 0.529, # OpenAI Assistants API
                "full_context": 0.524,      # Full context baseline
                "rag_basic": 0.456,         # Basic RAG
            },
            "our_results": {},
            "delta_vs_mem0": {},
        }
        
        for mode, stats in results.items():
            our_acc = stats["accuracy_mean"]
            comparison["our_results"][mode] = {
                "accuracy": our_acc,
                "std": stats["accuracy_std"],
            }
            comparison["delta_vs_mem0"][mode] = our_acc - 0.669
        
        return comparison
    
    def _export_report(self, report: BenchmarkReport):
        """Export report to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(self.config.results_dir)
        
        # Main report
        report_path = base_path / f"locomo_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logging.info(f"Report saved to: {report_path}")
        
        # Per-question results (if enabled)
        if self.config.export_per_question:
            questions_path = base_path / f"locomo_questions_{timestamp}.jsonl"
            with open(questions_path, 'w') as f:
                for run in report.all_runs:
                    for q in run.questions:
                        f.write(json.dumps(q.to_dict()) + "\n")
            logging.info(f"Question details saved to: {questions_path}")
        
        # Summary CSV
        csv_path = base_path / f"locomo_summary_{timestamp}.csv"
        with open(csv_path, 'w') as f:
            f.write("mode,accuracy_mean,accuracy_std,latency_ms,vs_mem0\n")
            for mode, stats in report.results_by_mode.items():
                delta = stats["accuracy_mean"] - 0.669
                f.write(f"{mode},{stats['accuracy_mean']:.4f},{stats['accuracy_std']:.4f},{stats['latency_mean']:.1f},{delta:+.4f}\n")
        logging.info(f"Summary CSV saved to: {csv_path}")
    
    def _print_summary(self, report: BenchmarkReport):
        """Print formatted summary to console."""
        print("\n")
        print("=" * 70)
        print("                    LOCOMO BENCHMARK RESULTS")
        print("=" * 70)
        print(f"  Dataset: {report.dataset_info['total_questions']} questions, {report.dataset_info['conversations']} conversations")
        print(f"  Sample size: {self.config.sample_size} questions, {self.config.num_runs} runs per mode")
        print(f"  Random seed: {self.config.random_seed}")
        print("-" * 70)
        
        # Results table
        print(f"{'Mode':<20} {'Accuracy':<15} {'Std':<10} {'Latency':<12} {'vs Mem0':<10}")
        print("-" * 70)
        
        for mode, stats in report.results_by_mode.items():
            acc = f"{stats['accuracy_mean']*100:.1f}%"
            std = f"±{stats['accuracy_std']*100:.1f}%"
            latency = f"{stats['latency_mean']:.0f}ms"
            delta = stats['accuracy_mean'] - 0.669
            delta_str = f"{delta*100:+.1f}%"
            
            print(f"{mode:<20} {acc:<15} {std:<10} {latency:<12} {delta_str:<10}")
        
        print("-" * 70)
        print("  Published baselines:")
        print("    Mem0 (graph):     66.9%")
        print("    Mem0 (vector):    63.5%")
        print("    OpenAI Assist:    52.9%")
        print("=" * 70)
        
        # Category breakdown for best mode
        best_mode = max(report.results_by_mode.items(), key=lambda x: x[1]["accuracy_mean"])
        print(f"\n  Category breakdown ({best_mode[0]}):")
        for cat, scores in best_mode[1]["category_breakdown"].items():
            print(f"    {cat:<15}: {scores['mean']*100:.1f}% ±{scores['std']*100:.1f}%")
        
        # LLM usage
        usage = self.llm.get_usage()
        print(f"\n  LLM tokens used: {usage['total_tokens']:,}")
        print("=" * 70)


# ==============================================
# MAIN
# ==============================================

def main():
    """Run LOCOMO benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KRNX LOCOMO Benchmark")
    parser.add_argument("--dataset", default="./data/locomo.json", help="Path to LOCOMO dataset")
    parser.add_argument("--samples", type=int, default=100, help="Number of questions (-1 for all)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results", default="./results", help="Results directory")
    parser.add_argument("--modes", nargs="+", default=["baseline", "raw", "agent_rules", "agent_hybrid"])
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Create config
    config = BenchmarkConfig(
        dataset_path=args.dataset,
        sample_size=args.samples,
        num_runs=args.runs,
        random_seed=args.seed,
        results_dir=args.results,
        modes=args.modes,
        verbose=args.verbose,
    )
    
    # Initialize KRNX components
    # Matching exact patterns from conftest.py
    fabric = None
    kernel = None
    
    try:
        # Kernel imports
        from chillbot.kernel.controller import KRNXController
        
        # Fabric imports
        from chillbot.fabric.orchestrator import MemoryFabric
        
        # Compute imports
        from chillbot.compute.embeddings import EmbeddingEngine
        from chillbot.compute.vectors import VectorStore, VectorStoreBackend
        
        # Parse Redis URL
        redis_url = os.environ.get("KRNX_REDIS_URL", "redis://localhost:6379")
        redis_host = "localhost"
        redis_port = 6379
        
        if redis_url.startswith("redis://"):
            parts = redis_url.replace("redis://", "").split(":")
            redis_host = parts[0]
            if len(parts) > 1:
                redis_port = int(parts[1].split("/")[0])
        
        # Initialize kernel
        kernel = KRNXController(
            data_path="./benchmark_data",
            redis_host=redis_host,
            redis_port=redis_port,
            enable_hash_chain=True,
            enable_backpressure=False,
        )
        
        # Initialize embeddings
        embeddings = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
        
        # Initialize vectors
        qdrant_url = os.environ.get("KRNX_QDRANT_URL", "http://localhost:6333")
        use_memory = os.environ.get("KRNX_USE_MEMORY_VECTORS", "false").lower() == "true"
        
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
        
        # Initialize fabric
        fabric = MemoryFabric(
            kernel=kernel,
            embeddings=embeddings,
            vectors=vectors,
            auto_embed=True,
            auto_enrich=False,
        )
        
        logging.info(f"[INIT] KRNX initialized successfully")
        
    except ImportError as e:
        logging.error(f"Failed to import KRNX components: {e}")
        logging.error("Running with fabric=None (baseline only)")
    except Exception as e:
        logging.error(f"Failed to initialize KRNX: {e}")
        logging.error("Running with fabric=None (baseline only)")
    
    # Run benchmark
    benchmark = LOCOMOBenchmark(
        config=config,
        fabric=fabric,
        kernel=kernel,
    )
    
    report = benchmark.run()
    
    return report


if __name__ == "__main__":
    main()
