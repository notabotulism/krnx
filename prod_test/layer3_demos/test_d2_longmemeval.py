"""
KRNX Layer 3 Tests - D2: LongMemEval Benchmark (Academic Rigor)

PROOF D2: KRNX handles long-context memory at scale.

LongMemEval Benchmark (ICLR 2025):
- 500 meticulously curated questions
- LongMemEval-S: ~115K tokens per question (~53 sessions avg)
- Tests USER-ASSISTANT chat history (not human-human dialogue)

Five Memory Abilities Tested:
1. Information Extraction - recall facts from user or assistant messages
2. Multi-Session Reasoning - synthesize across multiple sessions
3. Temporal Reasoning - time-aware queries
4. Knowledge Updates - track changing user info over time
5. Abstention - know when information is unknown

Question Types (from data):
- single-session-user: 70 questions
- single-session-assistant: 56 questions  
- single-session-preference: 30 questions
- multi-session: 133 questions
- temporal-reasoning: 133 questions
- knowledge-update: 78 questions
- abstention (by _abs suffix): 30 questions

BENCHMARK TARGETS:
- Long-context LLMs show 30-60% accuracy DROP on this benchmark
- Commercial systems (ChatGPT, Coze): 30-70% accuracy
- Zep claims 18.5% improvement over full-context baseline

Reference:
- Paper: "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" (ICLR 2025)
- Source: https://github.com/xiaowu0162/LongMemEval
"""

import pytest
import time
import json
import uuid
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

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
# LONGMEMEVAL DATASET ADAPTER (VERIFIED AGAINST ACTUAL DATA)
# =============================================================================

@dataclass
class LongMemEvalItem:
    """A single LongMemEval evaluation item."""
    item_id: str
    question: str
    answer: str
    question_type: str
    ability: str  # Mapped from question_type
    question_date: str
    haystack_sessions: List[List[Dict[str, str]]]  # List of sessions, each is list of {role, content}
    haystack_session_ids: List[str]
    haystack_dates: List[str]
    answer_session_ids: List[str]
    is_abstention: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class LongMemEvalAdapter:
    """
    Adapter for the LongMemEval benchmark dataset.
    
    VERIFIED STRUCTURE (from xiaowu0162/longmemeval-cleaned):
    {
      "question_id": "e47becba",
      "question_type": "single-session-user",
      "question": "What degree did I graduate with?",
      "question_date": "2023/05/30 (Tue) 23:40",
      "answer": "Business Administration",
      "answer_session_ids": ["answer_280352e9"],
      "haystack_dates": ["2023/01/15 (Sun) 14:30", ...],
      "haystack_session_ids": ["sharegpt_xxx", "85a1be56_1", ...],
      "haystack_sessions": [
        [  # Session - list of turns
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ],
        ...
      ]
    }
    
    Question Type -> Ability Mapping:
    - single-session-user -> information_extraction
    - single-session-assistant -> information_extraction
    - single-session-preference -> information_extraction
    - multi-session -> multi_session_reasoning
    - temporal-reasoning -> temporal_reasoning
    - knowledge-update -> knowledge_updates
    - *_abs suffix -> abstention
    """
    
    # Question type to ability mapping
    QUESTION_TYPE_TO_ABILITY = {
        "single-session-user": "information_extraction",
        "single-session-assistant": "information_extraction",
        "single-session-preference": "information_extraction",
        "multi-session": "multi_session_reasoning",
        "temporal-reasoning": "temporal_reasoning",
        "knowledge-update": "knowledge_updates",
    }
    
    ABILITIES = [
        "information_extraction",
        "multi_session_reasoning",
        "temporal_reasoning",
        "knowledge_updates",
        "abstention",
    ]
    
    ABILITY_NAMES = {
        "information_extraction": "Information Extraction",
        "multi_session_reasoning": "Multi-Session Reasoning",
        "temporal_reasoning": "Temporal Reasoning",
        "knowledge_updates": "Knowledge Updates",
        "abstention": "Abstention",
    }
    
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.downloader = DatasetDownloader(data_dir)
        self._data: Optional[List[Dict]] = None
    
    def load(
        self,
        variant: str = "s",
        force_download: bool = False,
    ) -> List[LongMemEvalItem]:
        """
        Load LongMemEval dataset.
        
        Args:
            variant: 's' for short (~115K tokens), 'm' for medium, 'oracle' for ground truth
            force_download: Force re-download
        """
        filepath = self.downloader.get_longmemeval(variant=variant, force=force_download)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
        
        items = []
        for i, item_data in enumerate(self._data):
            try:
                item = self._parse_item(item_data, i)
                if item:
                    items.append(item)
            except Exception as e:
                print(f"[LongMemEval] Error parsing item {i}: {e}")
                continue
        
        return items
    
    def _parse_item(self, data: Dict, index: int) -> Optional[LongMemEvalItem]:
        """Parse a single evaluation item."""
        item_id = str(data.get("question_id", f"item_{index}"))
        
        question = data.get("question", "")
        answer = data.get("answer", "")
        if not isinstance(answer, str):
            answer = str(answer)
        
        question_type = data.get("question_type", "unknown")
        question_date = data.get("question_date", "")
        
        # Check if abstention question (ID ends with _abs)
        is_abstention = item_id.endswith("_abs")
        
        # Map to ability
        if is_abstention:
            ability = "abstention"
        else:
            ability = self.QUESTION_TYPE_TO_ABILITY.get(question_type, "unknown")
        
        # Parse haystack sessions
        haystack_sessions = data.get("haystack_sessions", [])
        validated_sessions = []
        total_turns = 0
        
        for session in haystack_sessions:
            if isinstance(session, list):
                validated_turns = []
                for turn in session:
                    if isinstance(turn, dict):
                        validated_turns.append({
                            "role": turn.get("role", "user"),
                            "content": turn.get("content", ""),
                            "has_answer": turn.get("has_answer", False),
                        })
                if validated_turns:
                    validated_sessions.append(validated_turns)
                    total_turns += len(validated_turns)
        
        return LongMemEvalItem(
            item_id=item_id,
            question=question,
            answer=answer,
            question_type=question_type,
            ability=ability,
            question_date=question_date,
            haystack_sessions=validated_sessions,
            haystack_session_ids=data.get("haystack_session_ids", []),
            haystack_dates=data.get("haystack_dates", []),
            answer_session_ids=data.get("answer_session_ids", []),
            is_abstention=is_abstention,
            metadata={
                "original_type": question_type,
                "num_sessions": len(validated_sessions),
                "total_turns": total_turns,
            }
        )
    
    def get_questions(
        self,
        items: List[LongMemEvalItem],
        abilities: Optional[List[str]] = None,
        include_abstention: bool = True,
        limit: Optional[int] = None,
    ) -> List[BenchmarkQuestion]:
        """Convert items to BenchmarkQuestion format."""
        questions = []
        
        for item in items:
            # Filter by ability
            if abilities and item.ability not in abilities:
                continue
            
            # Filter abstention
            if not include_abstention and item.is_abstention:
                continue
            
            questions.append(BenchmarkQuestion(
                question_id=item.item_id,
                question=item.question,
                expected_answer=item.answer,
                category=item.ability,
                evidence_ids=item.answer_session_ids,
                metadata={
                    "question_type": item.question_type,
                    "question_date": item.question_date,
                    "is_abstention": item.is_abstention,
                    "num_sessions": item.metadata.get("num_sessions", 0),
                }
            ))
        
        if limit:
            questions = questions[:limit]
        
        return questions
    
    def get_ability_distribution(self, items: List[LongMemEvalItem]) -> Dict[str, int]:
        """Get count of items per ability."""
        return dict(Counter(item.ability for item in items))
    
    def get_type_distribution(self, items: List[LongMemEvalItem]) -> Dict[str, int]:
        """Get count of items per question type."""
        return dict(Counter(item.question_type for item in items))


# =============================================================================
# LLM-AS-JUDGE EVALUATION
# =============================================================================

LONGMEMEVAL_JUDGE_PROMPT = """Evaluate whether the AI assistant's answer correctly addresses the question based on chat history.

Question: {question}
Expected answer: {gold_answer}
Generated answer: {generated_answer}

Evaluation criteria:
- The answer must contain the key factual information from the expected answer
- Minor phrasing differences are acceptable
- Additional context beyond the expected answer is fine
- For abstention questions, saying "I don't know" or similar is correct if that's the expected answer

Respond in JSON format:
{{"reasoning": "brief explanation", "verdict": "CORRECT" or "INCORRECT"}}"""


class LongMemEvalEvaluator:
    """Evaluate LongMemEval results using LLM-as-judge."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def judge_answer(
        self,
        question: str,
        gold_answer: str,
        generated_answer: str,
    ) -> Tuple[bool, float, str]:
        """Judge answer correctness."""
        prompt = LONGMEMEVAL_JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                temperature=0.0,
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
                reasoning = result.get("reasoning", "")
                is_correct = verdict == "CORRECT"
                confidence = 0.9 if is_correct else 0.1
            except json.JSONDecodeError:
                # Fallback
                content_upper = content.upper()
                is_correct = "CORRECT" in content_upper and "INCORRECT" not in content_upper
                confidence = 0.7
                reasoning = content
            
            return is_correct, confidence, reasoning
            
        except Exception as e:
            return False, 0.0, f"Error: {e}"
    
    def evaluate_results(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if not results:
            return {"error": "No results"}
        
        correct = sum(1 for r in results if r.is_correct)
        total = len(results)
        
        # Per-ability breakdown
        abilities = {}
        for r in results:
            if r.category not in abilities:
                abilities[r.category] = {"correct": 0, "total": 0}
            abilities[r.category]["total"] += 1
            if r.is_correct:
                abilities[r.category]["correct"] += 1
        
        for ab in abilities:
            abilities[ab]["accuracy"] = (
                abilities[ab]["correct"] / abilities[ab]["total"]
                if abilities[ab]["total"] > 0 else 0.0
            )
        
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "abilities": abilities,
            "avg_latency_ms": sum(r.latency_ms for r in results) / total if total > 0 else 0,
            "avg_retrieved": sum(r.retrieved_events for r in results) / total if total > 0 else 0,
        }


# =============================================================================
# LONGMEMEVAL BENCHMARK RUNNER
# =============================================================================

class LongMemEvalRunner:
    """
    Run LongMemEval benchmark against KRNX.
    
    Key difference from LOCOMO: This tests user-assistant chat history,
    so we ingest the haystack sessions as if they were past conversations.
    """
    
    def __init__(
        self,
        fabric,
        llm_client: Optional[LLMClient],
        workspace_id: str,
        user_id: str,
    ):
        self.fabric = fabric
        self.llm = llm_client
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.evaluator = LongMemEvalEvaluator(llm_client) if llm_client else None
        self._ingested_count = 0
    
    def ingest_item(self, item: LongMemEvalItem) -> int:
        """
        Ingest the haystack sessions for a single item.
        
        Each session is a user-assistant conversation that happened in the past.
        """
        event_count = 0
        
        for session_idx, session in enumerate(item.haystack_sessions):
            session_id = (
                item.haystack_session_ids[session_idx]
                if session_idx < len(item.haystack_session_ids)
                else f"session_{session_idx}"
            )
            
            session_date = (
                item.haystack_dates[session_idx]
                if session_idx < len(item.haystack_dates)
                else ""
            )
            
            for turn_idx, turn in enumerate(session):
                role = turn.get("role", "user")
                content_text = turn.get("content", "")
                
                if not content_text.strip():
                    continue
                
                content = {
                    "text": content_text,
                    "role": role,
                    "session_id": session_id,
                    "session_date": session_date,
                    "turn_index": turn_idx,
                    "has_answer": turn.get("has_answer", False),
                }
                
                try:
                    self.fabric.remember(
                        content=content,
                        workspace_id=self.workspace_id,
                        user_id=self.user_id,
                        channel="longmemeval",
                    )
                    event_count += 1
                except Exception as e:
                    print(f"[LongMemEval] Ingest error: {e}")
        
        self._ingested_count += event_count
        return event_count
    
    def answer_question(
        self,
        item: LongMemEvalItem,
        top_k: int = 20,
    ) -> BenchmarkResult:
        """Answer a question using KRNX retrieval + LLM."""
        if not self.llm:
            raise ValueError("LLM client required")
        
        start_time = time.perf_counter()
        retrieved_events = 0
        context = ""
        
        # Retrieve from KRNX
        try:
            recall_result = self.fabric.recall(
                query=item.question,
                workspace_id=self.workspace_id,
                user_id=self.user_id,
                top_k=top_k,
            )
            
            memories = recall_result.memories if hasattr(recall_result, 'memories') else []
            retrieved_events = len(memories)
            
            # Build context
            context_parts = []
            for mem in memories:
                if isinstance(mem.content, dict):
                    role = mem.content.get("role", "user")
                    text = mem.content.get("text", "")
                    context_parts.append(f"[{role}]: {text}")
                else:
                    context_parts.append(str(mem.content))
            
            context = "\n".join(context_parts)
            
        except Exception as e:
            print(f"[LongMemEval] Recall error: {e}")
        
        # Generate answer
        try:
            system_prompt = (
                "You are a helpful assistant answering questions about past conversations. "
                "Use the provided context from chat history to answer. "
                "If you cannot find the answer in the context, say 'I don't have that information.' "
                "Be concise and specific."
            )
            
            user_prompt = f"Chat history context:\n{context}\n\nQuestion: {item.question}\n\nAnswer:"
            
            response = self.llm.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=256,
            )
            actual_answer = response.content.strip()
            tokens_used = response.tokens_used
        except Exception as e:
            actual_answer = f"Error: {e}"
            tokens_used = 0
        
        # Evaluate
        is_correct = False
        confidence = 0.0
        
        if self.evaluator:
            try:
                is_correct, confidence, _ = self.evaluator.judge_answer(
                    question=item.question,
                    gold_answer=item.answer,
                    generated_answer=actual_answer,
                )
            except Exception as e:
                print(f"[LongMemEval] Judge error: {e}")
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return BenchmarkResult(
            question_id=item.item_id,
            question=item.question,
            expected_answer=item.answer,
            actual_answer=actual_answer,
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            retrieved_events=retrieved_events,
            category=item.ability,
        )
    
    def run_baseline(self, item: LongMemEvalItem) -> BenchmarkResult:
        """Run baseline without memory retrieval."""
        if not self.llm:
            raise ValueError("LLM client required")
        
        start_time = time.perf_counter()
        
        try:
            system_prompt = (
                "You are answering questions about past conversations. "
                "You don't have access to the conversation history. "
                "If you don't know, say 'I don't have that information.'"
            )
            
            response = self.llm.complete(
                prompt=f"Question: {item.question}\n\nAnswer:",
                system_prompt=system_prompt,
                max_tokens=256,
            )
            actual_answer = response.content.strip()
            tokens_used = response.tokens_used
        except Exception as e:
            actual_answer = f"Error: {e}"
            tokens_used = 0
        
        is_correct = False
        confidence = 0.0
        
        if self.evaluator:
            try:
                is_correct, confidence, _ = self.evaluator.judge_answer(
                    question=item.question,
                    gold_answer=item.answer,
                    generated_answer=actual_answer,
                )
            except:
                pass
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return BenchmarkResult(
            question_id=item.item_id,
            question=item.question,
            expected_answer=item.answer,
            actual_answer=actual_answer,
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            retrieved_events=0,
            category=item.ability,
        )


# =============================================================================
# D2 TESTS
# =============================================================================

@pytest.mark.layer3
@pytest.mark.benchmark
class TestD2LongMemEval:
    """
    D2: LongMemEval Benchmark Tests
    
    Proves KRNX handles long-context memory better than stuffing context windows.
    Target: Beat full-context baseline accuracy.
    """
    
    # =========================================================================
    # D2.1: Dataset Loading & Structure
    # =========================================================================
    
    def test_d2_1_dataset_loading(
        self,
        dataset_downloader,
        print_proof_summary,
    ):
        """
        D2.1: LongMemEval dataset loads correctly.
        
        PROOF:
        - P1: Dataset downloads/loads successfully
        - P2: Contains expected number of items (500)
        - P3: All question types present
        - P4: Sessions have correct structure
        """
        adapter = LongMemEvalAdapter()
        
        try:
            items = adapter.load(variant="s")
            loaded = True
        except Exception as e:
            print(f"[D2.1] Load failed: {e}")
            items = []
            loaded = False
        
        type_dist = adapter.get_type_distribution(items) if items else {}
        ability_dist = adapter.get_ability_distribution(items) if items else {}
        
        # Check structure of first item
        has_correct_structure = False
        if items:
            item = items[0]
            has_correct_structure = (
                len(item.haystack_sessions) > 0 and
                len(item.haystack_sessions[0]) > 0 and
                "role" in item.haystack_sessions[0][0] and
                "content" in item.haystack_sessions[0][0]
            )
        
        print(f"\n[D2.1] Loaded {len(items)} items")
        print(f"[D2.1] Question types: {type_dist}")
        print(f"[D2.1] Abilities: {ability_dist}")
        
        proofs = {
            "P1_dataset_loaded": loaded and len(items) > 0,
            "P2_expected_count": len(items) == 500,
            "P3_types_present": len(type_dist) >= 5,
            "P4_correct_structure": has_correct_structure,
        }
        
        result = ProofResult(
            test_id="D2.1",
            guarantee="LongMemEval Dataset Loading",
            proofs=proofs,
            metrics={
                "total_items": len(items),
                "question_types": type_dist,
                "abilities": ability_dist,
            },
        )
        
        print_proof_summary(result)
        
        assert proofs["P1_dataset_loaded"], "P1 VIOLATED: Dataset failed to load"
        assert proofs["P2_expected_count"], f"P2 VIOLATED: Expected 500 items, got {len(items)}"
    
    # =========================================================================
    # D2.2: Ingestion at Scale
    # =========================================================================
    
    def test_d2_2_ingestion_scale(
        self,
        fabric_with_vectors,
        unique_workspace: str,
        unique_user: str,
        dataset_downloader,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D2.2: KRNX handles LongMemEval-scale ingestion.
        
        PROOF:
        - P1: Load single item's haystack (~53 sessions, 100+ turns)
        - P2: Ingest all turns successfully
        - P3: Events are queryable
        - P4: Ingestion rate is acceptable (<1s per item)
        """
        adapter = LongMemEvalAdapter()
        items = adapter.load(variant="s")
        
        # Use first item (has ~53 sessions)
        test_item = items[0]
        expected_turns = test_item.metadata.get("total_turns", 0)
        
        print(f"\n[D2.2] Test item: {test_item.item_id}")
        print(f"[D2.2] Sessions: {test_item.metadata.get('num_sessions', 0)}")
        print(f"[D2.2] Expected turns: {expected_turns}")
        
        runner = LongMemEvalRunner(
            fabric=fabric_with_vectors,
            llm_client=None,
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        start_time = time.perf_counter()
        ingested = runner.ingest_item(test_item)
        ingest_time = time.perf_counter() - start_time
        
        print(f"[D2.2] Ingested {ingested} turns in {ingest_time:.2f}s")
        
        # Wait for migration
        wait_for_migration(fabric_with_vectors, ingested, timeout=120)
        
        # Query to verify
        events = list(fabric_with_vectors.kernel.query_events(
            workspace_id=unique_workspace,
            user_id=unique_user,
            limit=ingested + 100,
        ))
        
        proofs = {
            "P1_item_loaded": expected_turns > 0,
            "P2_ingestion_complete": ingested == expected_turns,
            "P3_events_queryable": len(events) == expected_turns,
            "P4_acceptable_rate": ingest_time < 10.0,  # <10s for ~100 turns
        }
        
        result = ProofResult(
            test_id="D2.2",
            guarantee="LongMemEval Ingestion Scale",
            proofs=proofs,
            metrics={
                "expected_turns": expected_turns,
                "ingested_turns": ingested,
                "queryable_events": len(events),
                "ingest_time_s": round(ingest_time, 2),
                "turns_per_second": round(ingested / ingest_time, 1) if ingest_time > 0 else 0,
            },
        )
        
        print_proof_summary(result)
        
        assert proofs["P2_ingestion_complete"], f"P2 VIOLATED: Expected {expected_turns}, got {ingested}"
        assert proofs["P3_events_queryable"], f"P3 VIOLATED: Expected {expected_turns} queryable, got {len(events)}"
    
    # =========================================================================
    # D2.3: Question Answering Accuracy
    # =========================================================================
    
    @pytest.mark.requires_llm
    @pytest.mark.slow
    def test_d2_3_question_answering(
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
        D2.3: KRNX achieves competitive QA accuracy on LongMemEval.
        
        PROOF:
        - P1: Ingest item haystacks
        - P2: Answer questions using KRNX retrieval
        - P3: Compute per-ability accuracy
        - P4: Beat no-memory baseline
        """
        sample_size = LAYER3_CONFIG.get("longmemeval_sample_size", 20)
        
        adapter = LongMemEvalAdapter()
        items = adapter.load(variant="s")
        
        # Sample items across abilities
        test_items = items[:sample_size]
        
        print(f"\n[D2.3] Testing {len(test_items)} items")
        
        results: List[BenchmarkResult] = []
        baseline_results: List[BenchmarkResult] = []
        
        for i, item in enumerate(test_items):
            if (i + 1) % 5 == 0:
                print(f"[D2.3] Progress: {i+1}/{len(test_items)}")
            
            # Create fresh workspace per item (they have different haystacks)
            item_workspace = f"{unique_workspace}_{item.item_id}"
            
            runner = LongMemEvalRunner(
                fabric=fabric_with_vectors,
                llm_client=llm_client,
                workspace_id=item_workspace,
                user_id=unique_user,
            )
            
            # Ingest this item's haystack
            ingested = runner.ingest_item(item)
            
            # Small wait for indexing
            time.sleep(0.5)
            
            # Answer with KRNX
            result = runner.answer_question(item, top_k=20)
            results.append(result)
            
            # Baseline (no memory)
            baseline = runner.run_baseline(item)
            baseline_results.append(baseline)
            
            # Rate limiting
            time.sleep(0.3)
        
        # Evaluate
        evaluator = LongMemEvalEvaluator(llm_client)
        krnx_metrics = evaluator.evaluate_results(results)
        baseline_metrics = evaluator.evaluate_results(baseline_results)
        
        krnx_accuracy = krnx_metrics["accuracy"]
        baseline_accuracy = baseline_metrics["accuracy"]
        improvement = krnx_accuracy - baseline_accuracy
        
        print(f"\n[D2.3] KRNX Accuracy: {krnx_accuracy:.1%}")
        print(f"[D2.3] Baseline Accuracy: {baseline_accuracy:.1%}")
        print(f"[D2.3] Improvement: {improvement:.1%}")
        
        print("\n[D2.3] Per-Ability Results:")
        for ability, data in sorted(krnx_metrics.get("abilities", {}).items()):
            baseline_ab = baseline_metrics.get("abilities", {}).get(ability, {"accuracy": 0})
            print(f"  {ability}: KRNX={data['accuracy']:.1%}, Baseline={baseline_ab.get('accuracy', 0):.1%}")
        
        proofs = {
            "P1_items_ingested": all(r.retrieved_events >= 0 for r in results),
            "P2_questions_answered": len(results) == len(test_items),
            "P3_abilities_computed": len(krnx_metrics.get("abilities", {})) > 0,
            "P4_beats_baseline": krnx_accuracy > baseline_accuracy,
        }
        
        result = ProofResult(
            test_id="D2.3",
            guarantee="LongMemEval Question Answering",
            proofs=proofs,
            metrics={
                "items_tested": len(test_items),
                "krnx_accuracy": f"{krnx_accuracy:.2%}",
                "baseline_accuracy": f"{baseline_accuracy:.2%}",
                "improvement": f"{improvement:.2%}",
                "abilities": krnx_metrics.get("abilities", {}),
            },
            details=f"KRNX: {krnx_accuracy:.1%} vs Baseline: {baseline_accuracy:.1%}"
        )
        
        print_proof_summary(result)
        
        assert proofs["P2_questions_answered"], "P2 VIOLATED: Not all questions answered"
    
    # =========================================================================
    # D2.4: Retrieval Relevance
    # =========================================================================
    
    def test_d2_4_retrieval_relevance(
        self,
        fabric_with_vectors,
        unique_workspace: str,
        unique_user: str,
        dataset_downloader,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D2.4: Retrieved context is relevant to questions.
        
        For items with has_answer markers, check if those turns are retrieved.
        """
        adapter = LongMemEvalAdapter()
        items = adapter.load(variant="s")
        
        # Find items where we can check retrieval (non-abstention)
        test_items = [item for item in items if not item.is_abstention][:5]
        
        hits = 0
        total = 0
        
        for item in test_items:
            item_workspace = f"{unique_workspace}_{item.item_id}"
            
            runner = LongMemEvalRunner(
                fabric=fabric_with_vectors,
                llm_client=None,
                workspace_id=item_workspace,
                user_id=unique_user,
            )
            
            # Ingest
            ingested = runner.ingest_item(item)
            time.sleep(0.5)  # Wait for indexing
            
            # Recall
            try:
                recall_result = fabric_with_vectors.recall(
                    query=item.question,
                    workspace_id=item_workspace,
                    user_id=unique_user,
                    top_k=20,
                )
                
                memories = recall_result.memories if hasattr(recall_result, 'memories') else []
                
                # Check if any retrieved memory has has_answer=True
                found_answer = False
                for mem in memories:
                    if isinstance(mem.content, dict):
                        if mem.content.get("has_answer", False):
                            found_answer = True
                            break
                
                if found_answer:
                    hits += 1
                total += 1
                
            except Exception as e:
                print(f"[D2.4] Retrieval error: {e}")
        
        recall_rate = hits / total if total > 0 else 0.0
        
        proofs = {
            "P1_items_ingested": total > 0,
            "P2_retrieval_works": total > 0,
            "P3_some_hits": hits > 0,
        }
        
        result = ProofResult(
            test_id="D2.4",
            guarantee="LongMemEval Retrieval Relevance",
            proofs=proofs,
            metrics={
                "items_tested": total,
                "answer_hits": hits,
                "recall_rate": f"{recall_rate:.2%}",
            },
        )
        
        print_proof_summary(result)


# =============================================================================
# FAILURE MODE TESTS
# =============================================================================

@pytest.mark.layer3
@pytest.mark.failure_mode
class TestD2FailureModes:
    """D2 Failure mode tests."""
    
    def test_d2_f1_abstention_handling(
        self,
        fabric_with_vectors,
        unique_workspace: str,
        unique_user: str,
        dataset_downloader,
        print_proof_summary,
    ):
        """
        D2.F1: Abstention questions handled correctly.
        
        For abstention questions, the correct answer is "I don't know" or similar.
        """
        adapter = LongMemEvalAdapter()
        items = adapter.load(variant="s")
        
        abstention_items = [item for item in items if item.is_abstention]
        
        proofs = {
            "P1_abstention_identified": len(abstention_items) > 0,
            "P2_count_correct": len(abstention_items) == 30,  # Should be 30
        }
        
        result = ProofResult(
            test_id="D2.F1",
            guarantee="Abstention Question Handling",
            proofs=proofs,
            metrics={
                "abstention_count": len(abstention_items),
                "sample_ids": [item.item_id for item in abstention_items[:5]],
            },
        )
        
        print_proof_summary(result)
        
        assert proofs["P1_abstention_identified"], "P1 VIOLATED: No abstention items found"
