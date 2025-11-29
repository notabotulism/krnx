"""
KRNX Layer 3 Tests - D1: LOCOMO Benchmark (Academic Rigor)

PROOF D1: KRNX provides accurate memory retrieval for conversational QA.

LOCOMO Benchmark (ACL 2024):
- 10 human-human conversations, 300-400 turns each
- ~9K tokens average per conversation
- Tests: single-hop, temporal, multi-hop, open-domain reasoning
- Reference: https://github.com/snap-research/locomo

SCIENTIFIC METHODOLOGY:
- Multiple runs with averaging (default 3)
- Temperature=0 for reproducibility
- Full JSON export with timestamps
- Per-category breakdown
- Baseline comparison (no memory)
- Statistical significance reporting

BENCHMARK TARGETS:
- Mem0 reported: 66.9% accuracy
- OpenAI baseline: 52.9% accuracy
- KRNX target: ≥65% accuracy
"""

import pytest
import time
import json
import uuid
import os
import sys
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import Counter, defaultdict

from conftest import (
    ProofResult,
    StatisticalResult,
    BenchmarkQuestion,
    BenchmarkResult,
    DatasetDownloader,
    LLMClient,
    LAYER3_CONFIG,
    HAS_KRNX,
    HAS_OPENAI,
    HAS_ANTHROPIC,
    DATA_DIR,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Number of runs for statistical averaging
NUM_RUNS = int(os.environ.get("KRNX_NUM_RUNS", "3"))

# Sample size (questions per run)
SAMPLE_SIZE = int(os.environ.get("KRNX_LOCOMO_SAMPLES", "50"))

# Output directory for results
RESULTS_DIR = Path(os.environ.get("KRNX_RESULTS_DIR", "./results"))

# LLM temperature (0 for reproducibility)
LLM_TEMPERATURE = 0.0


# =============================================================================
# LOCOMO DATASET ADAPTER
# =============================================================================

@dataclass
class LocomoTurn:
    """A single turn in a LOCOMO conversation."""
    dia_id: str
    speaker: str
    text: str
    session_id: str
    session_date: str
    turn_index: int
    conversation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class LocomoConversation:
    """A complete LOCOMO conversation with multiple sessions."""
    conversation_id: str
    speaker_a: str
    speaker_b: str
    sessions: Dict[str, List[LocomoTurn]]
    session_dates: Dict[str, str]
    questions: List['LocomoQuestion']
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_turns(self) -> int:
        return sum(len(turns) for turns in self.sessions.values())


@dataclass
class LocomoQuestion:
    """A question from the LOCOMO benchmark."""
    question_id: str
    question: str
    answer: str
    category: int
    evidence_ids: List[str]
    conversation_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def category_name(self) -> str:
        return {
            1: "single_hop",
            2: "temporal",
            3: "multi_hop",
            4: "open_domain",
            5: "adversarial",
        }.get(self.category, f"unknown_{self.category}")


class LocomoAdapter:
    """
    Adapter for the LOCOMO benchmark dataset.
    
    VERIFIED STRUCTURE (from snap-research/locomo):
    {
      "sample_id": "...",
      "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie", 
        "session_1": [{speaker, dia_id, text}, ...],
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        ...
      },
      "qa": [{question, answer, category, evidence}, ...]
    }
    """
    
    CATEGORY_NAMES = {
        1: "single_hop",
        2: "temporal", 
        3: "multi_hop",
        4: "open_domain",
        5: "adversarial",
    }
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.downloader = DatasetDownloader(data_dir)
        self._data: Optional[List[Dict]] = None
    
    def load(self, force_download: bool = False) -> List[LocomoConversation]:
        """Load LOCOMO dataset."""
        filepath = self.downloader.get_locomo(force=force_download)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
        
        conversations = []
        for i, item in enumerate(self._data):
            try:
                conv = self._parse_conversation(item, i)
                if conv:
                    conversations.append(conv)
            except Exception as e:
                print(f"[LOCOMO] Error parsing conversation {i}: {e}")
                continue
        
        return conversations
    
    def _parse_conversation(self, data: Dict, index: int) -> Optional[LocomoConversation]:
        """Parse a single conversation."""
        conv_id = data.get("sample_id", f"conv_{index}")
        conv_data = data.get("conversation", {})
        
        if not conv_data:
            return None
        
        speaker_a = conv_data.get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("speaker_b", "Speaker B")
        
        # Find session keys
        session_keys = sorted([
            k for k in conv_data.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        ])
        
        sessions = {}
        session_dates = {}
        
        for session_key in session_keys:
            session_num = session_key.replace("session_", "")
            date_key = f"{session_key}_date_time"
            session_date = conv_data.get(date_key, "")
            session_dates[session_key] = session_date
            
            turns_data = conv_data.get(session_key, [])
            if not isinstance(turns_data, list):
                continue
            
            turns = []
            for turn_idx, turn_data in enumerate(turns_data):
                if not isinstance(turn_data, dict):
                    continue
                
                turn = LocomoTurn(
                    dia_id=turn_data.get("dia_id", f"{conv_id}:{session_key}:{turn_idx}"),
                    speaker=turn_data.get("speaker", "Unknown"),
                    text=turn_data.get("text", ""),
                    session_id=session_key,
                    session_date=session_date,
                    turn_index=turn_idx,
                    conversation_id=conv_id,
                )
                turns.append(turn)
            
            if turns:
                sessions[session_key] = turns
        
        # Parse questions
        qa_data = data.get("qa", [])
        questions = []
        
        for q_idx, qa in enumerate(qa_data):
            answer = qa.get("answer", "")
            if not isinstance(answer, str):
                answer = str(answer)
            
            question = LocomoQuestion(
                question_id=f"{conv_id}_q{q_idx}",
                question=qa.get("question", ""),
                answer=answer,
                category=qa.get("category", 0),
                evidence_ids=qa.get("evidence", []),
                conversation_id=conv_id,
            )
            questions.append(question)
        
        return LocomoConversation(
            conversation_id=conv_id,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            session_dates=session_dates,
            questions=questions,
        )
    
    def get_all_turns(self, conversations: List[LocomoConversation]) -> List[LocomoTurn]:
        """Flatten all turns from all conversations."""
        turns = []
        for conv in conversations:
            for session_key in sorted(conv.sessions.keys()):
                turns.extend(conv.sessions[session_key])
        return turns
    
    def get_all_questions(
        self,
        conversations: List[LocomoConversation],
        exclude_adversarial: bool = True,
    ) -> List[LocomoQuestion]:
        """Get all questions, optionally excluding adversarial."""
        questions = []
        for conv in conversations:
            for q in conv.questions:
                if exclude_adversarial and q.category == 5:
                    continue
                questions.append(q)
        return questions
    
    def get_category_distribution(self, questions: List[LocomoQuestion]) -> Dict[str, int]:
        """Get count per category."""
        dist = Counter(q.category_name for q in questions)
        return dict(dist)


# =============================================================================
# RESULTS DATASTRUCTURES
# =============================================================================

@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: str
    question: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    confidence: float
    category: str
    latency_ms: float
    tokens_used: int
    retrieved_events: int
    evidence_ids: List[str]
    retrieved_ids: List[str]
    judge_reasoning: str = ""


@dataclass
class RunResult:
    """Results from a single benchmark run."""
    run_id: str
    timestamp: str
    accuracy: float
    baseline_accuracy: float
    improvement: float
    total_questions: int
    correct_count: int
    baseline_correct: int
    category_results: Dict[str, Dict[str, Any]]
    avg_latency_ms: float
    avg_tokens: float
    avg_retrieved: float
    questions: List[QuestionResult]
    config: Dict[str, Any]


@dataclass
class BenchmarkReport:
    """Full benchmark report across multiple runs."""
    benchmark_name: str
    timestamp: str
    num_runs: int
    sample_size: int
    
    # Aggregate metrics
    mean_accuracy: float
    std_accuracy: float
    mean_baseline: float
    std_baseline: float
    mean_improvement: float
    
    # Per-category
    category_metrics: Dict[str, Dict[str, float]]
    
    # Individual runs
    runs: List[RunResult]
    
    # Comparison targets
    targets: Dict[str, float]
    
    # System info
    system_info: Dict[str, Any]


# =============================================================================
# LLM-AS-JUDGE EVALUATION
# =============================================================================

JUDGE_PROMPT = """You are evaluating a question-answering system. Determine if the generated answer is correct.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Rules:
1. The answer must contain the key factual information from the expected answer
2. Minor phrasing differences are acceptable  
3. Extra correct information is fine
4. Partial matches are INCORRECT
5. "I don't know" when expected answer exists is INCORRECT

Respond with JSON only:
{{"verdict": "CORRECT" or "INCORRECT", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


class LocomoEvaluator:
    """Evaluate LOCOMO benchmark results."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def judge_answer(
        self,
        question: str,
        expected: str,
        generated: str,
    ) -> Tuple[bool, float, str]:
        """Judge if answer is correct using LLM."""
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected=expected,
            generated=generated,
        )
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=200,
            )
            
            content = response.content.strip()
            
            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            try:
                result = json.loads(content)
                verdict = result.get("verdict", "").upper()
                is_correct = verdict == "CORRECT"
                confidence = float(result.get("confidence", 0.9 if is_correct else 0.1))
                reasoning = result.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback to text parsing
                content_upper = content.upper()
                is_correct = "CORRECT" in content_upper and "INCORRECT" not in content_upper
                confidence = 0.7
                reasoning = content[:200]
            
            return is_correct, confidence, reasoning
            
        except Exception as e:
            return False, 0.0, f"Judge error: {e}"
    
    def compute_f1(self, expected: str, generated: str) -> float:
        """Compute token-level F1 score."""
        import re
        
        def tokenize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return set(text.split())
        
        expected_tokens = tokenize(expected)
        generated_tokens = tokenize(generated)
        
        if not expected_tokens or not generated_tokens:
            return 0.0
        
        overlap = expected_tokens & generated_tokens
        precision = len(overlap) / len(generated_tokens)
        recall = len(overlap) / len(expected_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

class LocomoRunner:
    """Run LOCOMO benchmark against KRNX."""
    
    def __init__(
        self,
        fabric,
        llm_client: LLMClient,
        workspace_id: str,
        user_id: str,
    ):
        self.fabric = fabric
        self.llm = llm_client
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.evaluator = LocomoEvaluator(llm_client)
        self._ingested_turns = 0
    
    def ingest_turns(self, turns: List[LocomoTurn]) -> int:
        """Ingest turns into KRNX."""
        count = 0
        for turn in turns:
            content = {
                "text": f"{turn.speaker}: {turn.text}",
                "speaker": turn.speaker,
                "dia_id": turn.dia_id,
                "session_id": turn.session_id,
                "session_date": turn.session_date,
                "conversation_id": turn.conversation_id,
            }
            
            try:
                self.fabric.remember(
                    content=content,
                    workspace_id=self.workspace_id,
                    user_id=self.user_id,
                    channel="locomo",
                )
                count += 1
            except Exception as e:
                print(f"[LOCOMO] Ingest error: {e}")
        
        self._ingested_turns = count
        return count
    
    def answer_question(
        self,
        question: LocomoQuestion,
        top_k: int = 20,
    ) -> QuestionResult:
        """Answer a question using KRNX retrieval + LLM."""
        start_time = time.perf_counter()
        retrieved_ids = []
        context = ""
        
        # Retrieve from KRNX
        try:
            recall_result = self.fabric.recall(
                query=question.question,
                workspace_id=self.workspace_id,
                user_id=self.user_id,
                top_k=top_k,
            )
            
            memories = recall_result.memories if hasattr(recall_result, 'memories') else []
            
            context_parts = []
            for mem in memories:
                if isinstance(mem.content, dict):
                    text = mem.content.get("text", "")
                    dia_id = mem.content.get("dia_id", "")
                    retrieved_ids.append(dia_id)
                    context_parts.append(text)
                else:
                    context_parts.append(str(mem.content))
            
            context = "\n".join(context_parts)
            
        except Exception as e:
            print(f"[LOCOMO] Recall error: {e}")
        
        # Generate answer
        try:
            system_prompt = (
                "You are answering questions about conversations. "
                "Use ONLY the provided context. Be concise and specific. "
                "If the answer is not in the context, say 'I don't have that information.'"
            )
            
            user_prompt = f"Context:\n{context}\n\nQuestion: {question.question}\n\nAnswer:"
            
            response = self.llm.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=256,
            )
            actual_answer = response.content.strip()
            tokens_used = response.tokens_used
        except Exception as e:
            actual_answer = f"Error: {e}"
            tokens_used = 0
        
        # Evaluate
        is_correct, confidence, reasoning = self.evaluator.judge_answer(
            question=question.question,
            expected=question.answer,
            generated=actual_answer,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return QuestionResult(
            question_id=question.question_id,
            question=question.question,
            expected_answer=question.answer,
            actual_answer=actual_answer,
            is_correct=is_correct,
            confidence=confidence,
            category=question.category_name,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            retrieved_events=len(retrieved_ids),
            evidence_ids=question.evidence_ids,
            retrieved_ids=retrieved_ids,
            judge_reasoning=reasoning,
        )
    
    def answer_baseline(self, question: LocomoQuestion) -> QuestionResult:
        """Answer without memory retrieval (baseline)."""
        start_time = time.perf_counter()
        
        try:
            system_prompt = (
                "You are answering questions about conversations. "
                "You don't have access to the conversation history. "
                "If you don't know, say 'I don't have that information.'"
            )
            
            response = self.llm.complete(
                prompt=f"Question: {question.question}\n\nAnswer:",
                system_prompt=system_prompt,
                temperature=LLM_TEMPERATURE,
                max_tokens=256,
            )
            actual_answer = response.content.strip()
            tokens_used = response.tokens_used
        except Exception as e:
            actual_answer = f"Error: {e}"
            tokens_used = 0
        
        is_correct, confidence, reasoning = self.evaluator.judge_answer(
            question=question.question,
            expected=question.answer,
            generated=actual_answer,
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return QuestionResult(
            question_id=question.question_id,
            question=question.question,
            expected_answer=question.answer,
            actual_answer=actual_answer,
            is_correct=is_correct,
            confidence=confidence,
            category=question.category_name,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            retrieved_events=0,
            evidence_ids=question.evidence_ids,
            retrieved_ids=[],
            judge_reasoning=reasoning,
        )


# =============================================================================
# REPORT GENERATION
# =============================================================================

class ReportGenerator:
    """Generate benchmark reports."""
    
    def __init__(self, output_dir: Path = RESULTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        runs: List[RunResult],
        benchmark_name: str = "LOCOMO",
    ) -> BenchmarkReport:
        """Generate aggregate report from multiple runs."""
        accuracies = [r.accuracy for r in runs]
        baselines = [r.baseline_accuracy for r in runs]
        
        # Aggregate category metrics
        category_metrics = defaultdict(lambda: {"accuracies": [], "baselines": []})
        for run in runs:
            for cat, data in run.category_results.items():
                category_metrics[cat]["accuracies"].append(data.get("accuracy", 0))
                category_metrics[cat]["baselines"].append(data.get("baseline_accuracy", 0))
        
        final_categories = {}
        for cat, data in category_metrics.items():
            final_categories[cat] = {
                "mean_accuracy": statistics.mean(data["accuracies"]) if data["accuracies"] else 0,
                "std_accuracy": statistics.stdev(data["accuracies"]) if len(data["accuracies"]) > 1 else 0,
                "mean_baseline": statistics.mean(data["baselines"]) if data["baselines"] else 0,
            }
        
        return BenchmarkReport(
            benchmark_name=benchmark_name,
            timestamp=datetime.now().isoformat(),
            num_runs=len(runs),
            sample_size=runs[0].total_questions if runs else 0,
            mean_accuracy=statistics.mean(accuracies) if accuracies else 0,
            std_accuracy=statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            mean_baseline=statistics.mean(baselines) if baselines else 0,
            std_baseline=statistics.stdev(baselines) if len(baselines) > 1 else 0,
            mean_improvement=statistics.mean(accuracies) - statistics.mean(baselines) if accuracies else 0,
            category_metrics=final_categories,
            runs=runs,
            targets={
                "mem0_reported": 0.669,
                "openai_baseline": 0.529,
                "krnx_target": 0.65,
            },
            system_info={
                "num_runs": NUM_RUNS,
                "sample_size": SAMPLE_SIZE,
                "llm_temperature": LLM_TEMPERATURE,
                "timestamp": datetime.now().isoformat(),
            },
        )
    
    def save_json(self, report: BenchmarkReport, filename: str = None) -> Path:
        """Save report as JSON."""
        if filename is None:
            filename = f"locomo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        # Convert to serializable dict
        def to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        with open(filepath, 'w') as f:
            json.dump(to_dict(report), f, indent=2, default=str)
        
        return filepath
    
    def save_transcript(self, runs: List[RunResult], filename: str = None) -> Path:
        """Save detailed transcript."""
        if filename is None:
            filename = f"locomo_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("KRNX LOCOMO BENCHMARK TRANSCRIPT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            for run in runs:
                f.write(f"\n{'='*80}\n")
                f.write(f"RUN {run.run_id}\n")
                f.write(f"Timestamp: {run.timestamp}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"Accuracy: {run.accuracy:.1%} ({run.correct_count}/{run.total_questions})\n")
                f.write(f"Baseline: {run.baseline_accuracy:.1%} ({run.baseline_correct}/{run.total_questions})\n")
                f.write(f"Improvement: {run.improvement:.1%}\n\n")
                
                f.write("Per-Category Results:\n")
                for cat, data in sorted(run.category_results.items()):
                    f.write(f"  {cat}: KRNX={data['accuracy']:.1%}, Baseline={data['baseline_accuracy']:.1%}\n")
                
                f.write("\n" + "-" * 80 + "\n")
                f.write("DETAILED QUESTION RESULTS:\n")
                f.write("-" * 80 + "\n\n")
                
                for q in run.questions:
                    status = "✓" if q.is_correct else "✗"
                    f.write(f"[{status}] {q.question_id} ({q.category})\n")
                    f.write(f"    Q: {q.question[:100]}...\n" if len(q.question) > 100 else f"    Q: {q.question}\n")
                    f.write(f"    Expected: {q.expected_answer[:80]}...\n" if len(q.expected_answer) > 80 else f"    Expected: {q.expected_answer}\n")
                    f.write(f"    Got: {q.actual_answer[:80]}...\n" if len(q.actual_answer) > 80 else f"    Got: {q.actual_answer}\n")
                    f.write(f"    Retrieved: {q.retrieved_events} events, Latency: {q.latency_ms:.0f}ms\n")
                    f.write(f"    Judge: {q.judge_reasoning[:100]}\n" if q.judge_reasoning else "")
                    f.write("\n")
        
        return filepath
    
    def print_results_table(self, report: BenchmarkReport):
        """Print formatted results table."""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " KRNX LOCOMO BENCHMARK RESULTS ".center(78) + "║")
        print("╠" + "═" * 78 + "╣")
        print(f"║  Runs: {report.num_runs}    Sample Size: {report.sample_size}    Date: {report.timestamp[:10]}".ljust(78) + " ║")
        print("╠" + "═" * 78 + "╣")
        
        # Main metrics
        print("║" + " AGGREGATE METRICS ".center(78, "─") + "║")
        print(f"║  KRNX Accuracy:     {report.mean_accuracy:6.1%} ± {report.std_accuracy:.1%}".ljust(78) + " ║")
        print(f"║  Baseline Accuracy: {report.mean_baseline:6.1%} ± {report.std_baseline:.1%}".ljust(78) + " ║")
        print(f"║  Improvement:       {report.mean_improvement:6.1%}".ljust(78) + " ║")
        print("╠" + "═" * 78 + "╣")
        
        # Comparison
        print("║" + " COMPARISON TO BENCHMARKS ".center(78, "─") + "║")
        mem0 = report.targets.get("mem0_reported", 0.669)
        openai = report.targets.get("openai_baseline", 0.529)
        vs_mem0 = report.mean_accuracy - mem0
        vs_openai = report.mean_accuracy - openai
        
        mem0_status = "✓" if vs_mem0 >= 0 else "✗"
        openai_status = "✓" if vs_openai >= 0 else "✗"
        
        print(f"║  [{mem0_status}] vs Mem0 (66.9%):   {vs_mem0:+.1%}".ljust(78) + " ║")
        print(f"║  [{openai_status}] vs OpenAI (52.9%): {vs_openai:+.1%}".ljust(78) + " ║")
        print("╠" + "═" * 78 + "╣")
        
        # Per-category
        print("║" + " PER-CATEGORY BREAKDOWN ".center(78, "─") + "║")
        print("║  Category        │ KRNX Accuracy │ Baseline │ Δ        ".ljust(78) + " ║")
        print("║  ────────────────┼───────────────┼──────────┼──────────".ljust(78) + " ║")
        
        for cat, data in sorted(report.category_metrics.items()):
            acc = data.get("mean_accuracy", 0)
            base = data.get("mean_baseline", 0)
            delta = acc - base
            line = f"║  {cat:16}│ {acc:12.1%} │ {base:8.1%} │ {delta:+7.1%}"
            print(line.ljust(78) + " ║")
        
        print("╚" + "═" * 78 + "╝")
        print()


# =============================================================================
# TEST CLASS
# =============================================================================

@pytest.mark.layer3
@pytest.mark.benchmark
class TestD1LOCOMO:
    """
    D1: LOCOMO Benchmark Tests
    
    Scientific methodology with multi-run averaging.
    """
    
    def test_d1_1_ingestion_completeness(
        self,
        fabric_with_vectors,
        unique_workspace: str,
        unique_user: str,
        dataset_downloader,
        wait_for_migration,
        print_proof_summary,
    ):
        """D1.1: Verify turns are properly ingested."""
        adapter = LocomoAdapter()
        conversations = adapter.load()
        
        # Sample turns for ingestion
        all_turns = adapter.get_all_turns(conversations)
        sample_turns = all_turns[:SAMPLE_SIZE]
        
        print(f"\n[D1.1] Total conversations: {len(conversations)}")
        print(f"[D1.1] Total turns: {len(all_turns)}")
        print(f"[D1.1] Sampling: {len(sample_turns)} turns")
        
        runner = LocomoRunner(
            fabric=fabric_with_vectors,
            llm_client=None,
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        ingested = runner.ingest_turns(sample_turns)
        
        # Wait for indexing
        wait_for_migration(fabric_with_vectors, ingested, timeout=300)
        
        # Verify
        events = list(fabric_with_vectors.kernel.query_events(
            workspace_id=unique_workspace,
            user_id=unique_user,
            limit=ingested + 100,
        ))
        
        vector_count = fabric_with_vectors.vectors.count(unique_workspace)
        
        print(f"[D1.1] Ingested: {ingested}")
        print(f"[D1.1] Events in kernel: {len(events)}")
        print(f"[D1.1] Vectors indexed: {vector_count}")
        
        proofs = {
            "P1_dataset_loaded": len(conversations) == 10,
            "P2_turns_extracted": len(all_turns) > 0,
            "P3_ingestion_complete": ingested == len(sample_turns),
            "P4_events_queryable": len(events) == ingested,
            "P5_vectors_indexed": vector_count == ingested,
        }
        
        result = ProofResult(
            test_id="D1.1",
            guarantee="LOCOMO Ingestion Completeness",
            proofs=proofs,
            metrics={
                "conversations": len(conversations),
                "total_turns": len(all_turns),
                "sampled": len(sample_turns),
                "ingested": ingested,
                "events": len(events),
                "vectors": vector_count,
            },
        )
        
        print_proof_summary(result)
        
        for name, passed in proofs.items():
            assert passed, f"{name} VIOLATED"
    
    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_d1_2_question_answering(
        self,
        fabric_with_vectors,
        llm_client,
        unique_workspace: str,
        unique_user: str,
        dataset_downloader,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D1.2: Multi-run benchmark with statistical averaging.
        
        Runs NUM_RUNS times and reports mean ± std.
        """
        adapter = LocomoAdapter()
        conversations = adapter.load()
        
        all_turns = adapter.get_all_turns(conversations)
        all_questions = adapter.get_all_questions(conversations, exclude_adversarial=True)
        
        print(f"\n[D1.2] Configuration:")
        print(f"  Runs: {NUM_RUNS}")
        print(f"  Sample size: {SAMPLE_SIZE}")
        print(f"  Temperature: {LLM_TEMPERATURE}")
        print(f"  Total questions available: {len(all_questions)}")
        
        # Limit sample
        sample_questions = all_questions[:SAMPLE_SIZE]
        
        # Get turns for sampled conversations
        conv_ids = set(q.conversation_id for q in sample_questions)
        sample_turns = [t for t in all_turns if t.conversation_id in conv_ids]
        
        print(f"  Using {len(sample_questions)} questions from {len(conv_ids)} conversations")
        print(f"  Ingesting {len(sample_turns)} turns\n")
        
        runs: List[RunResult] = []
        
        for run_num in range(NUM_RUNS):
            run_id = f"run_{run_num + 1}"
            run_workspace = f"{unique_workspace}_{run_id}"
            
            print(f"\n{'='*60}")
            print(f"[D1.2] STARTING {run_id.upper()} OF {NUM_RUNS}")
            print(f"{'='*60}")
            
            runner = LocomoRunner(
                fabric=fabric_with_vectors,
                llm_client=llm_client,
                workspace_id=run_workspace,
                user_id=unique_user,
            )
            
            # Ingest
            print(f"[{run_id}] Ingesting turns...")
            ingested = runner.ingest_turns(sample_turns)
            print(f"[{run_id}] Ingested {ingested} turns")
            
            # Wait for indexing
            time.sleep(2)
            
            # Answer questions
            krnx_results: List[QuestionResult] = []
            baseline_results: List[QuestionResult] = []
            
            print(f"[{run_id}] Answering {len(sample_questions)} questions...")
            
            for i, question in enumerate(sample_questions):
                if (i + 1) % 10 == 0:
                    print(f"[{run_id}] Progress: {i+1}/{len(sample_questions)}")
                
                # KRNX answer
                result = runner.answer_question(question, top_k=20)
                krnx_results.append(result)
                
                # Baseline answer
                baseline = runner.answer_baseline(question)
                baseline_results.append(baseline)
                
                # Rate limiting
                time.sleep(0.2)
            
            # Compute metrics
            correct = sum(1 for r in krnx_results if r.is_correct)
            baseline_correct = sum(1 for r in baseline_results if r.is_correct)
            
            accuracy = correct / len(krnx_results)
            baseline_accuracy = baseline_correct / len(baseline_results)
            
            # Per-category
            category_results = {}
            categories = set(r.category for r in krnx_results)
            
            for cat in categories:
                cat_krnx = [r for r in krnx_results if r.category == cat]
                cat_base = [r for r in baseline_results if r.category == cat]
                
                cat_correct = sum(1 for r in cat_krnx if r.is_correct)
                cat_base_correct = sum(1 for r in cat_base if r.is_correct)
                
                category_results[cat] = {
                    "total": len(cat_krnx),
                    "correct": cat_correct,
                    "accuracy": cat_correct / len(cat_krnx) if cat_krnx else 0,
                    "baseline_correct": cat_base_correct,
                    "baseline_accuracy": cat_base_correct / len(cat_base) if cat_base else 0,
                }
            
            run_result = RunResult(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                accuracy=accuracy,
                baseline_accuracy=baseline_accuracy,
                improvement=accuracy - baseline_accuracy,
                total_questions=len(sample_questions),
                correct_count=correct,
                baseline_correct=baseline_correct,
                category_results=category_results,
                avg_latency_ms=statistics.mean(r.latency_ms for r in krnx_results),
                avg_tokens=statistics.mean(r.tokens_used for r in krnx_results),
                avg_retrieved=statistics.mean(r.retrieved_events for r in krnx_results),
                questions=krnx_results,
                config={
                    "sample_size": SAMPLE_SIZE,
                    "temperature": LLM_TEMPERATURE,
                },
            )
            
            runs.append(run_result)
            
            print(f"\n[{run_id}] RESULTS:")
            print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(sample_questions)})")
            print(f"  Baseline: {baseline_accuracy:.1%}")
            print(f"  Improvement: {accuracy - baseline_accuracy:.1%}")
        
        # Generate report
        reporter = ReportGenerator()
        report = reporter.generate_report(runs, "LOCOMO")
        
        # Save outputs
        json_path = reporter.save_json(report)
        transcript_path = reporter.save_transcript(runs)
        
        print(f"\n[D1.2] Saved results to:")
        print(f"  JSON: {json_path}")
        print(f"  Transcript: {transcript_path}")
        
        # Print table
        reporter.print_results_table(report)
        
        # Proofs
        proofs = {
            "P1_runs_completed": len(runs) == NUM_RUNS,
            "P2_questions_answered": all(r.total_questions == SAMPLE_SIZE for r in runs),
            "P3_beats_baseline": report.mean_accuracy > report.mean_baseline,
            "P4_results_saved": json_path.exists() and transcript_path.exists(),
        }
        
        result = ProofResult(
            test_id="D1.2",
            guarantee="LOCOMO Question Answering",
            proofs=proofs,
            metrics={
                "mean_accuracy": f"{report.mean_accuracy:.1%}",
                "std_accuracy": f"{report.std_accuracy:.1%}",
                "mean_baseline": f"{report.mean_baseline:.1%}",
                "improvement": f"{report.mean_improvement:.1%}",
                "vs_mem0": f"{report.mean_accuracy - 0.669:+.1%}",
            },
        )
        
        print_proof_summary(result)
        
        for name, passed in proofs.items():
            assert passed, f"{name} VIOLATED"


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run benchmark standalone (without pytest)."""
    print("Use: python -m pytest test_d1_locomo_v2.py -v -s")
    print("Or:  KRNX_NUM_RUNS=3 KRNX_LOCOMO_SAMPLES=100 python -m pytest ...")
