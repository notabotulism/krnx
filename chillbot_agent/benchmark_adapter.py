"""
ChillBot Agent - Benchmark Adapter

Integrates ChillBotAgent with LOCOMO/LongMemEval benchmarks
for fair comparison against Mem0 and Letta.

This adapter replaces raw fabric.recall() with the full agent pipeline:
    Query → Planner → Router → Processor → Assembler → LLM → Answer

Test modes:
    - rules: Fast, deterministic (for baseline comparisons)
    - llm: Full intelligence (for best accuracy)
    - hybrid: Production mode (default)

Usage:
    adapter = BenchmarkAdapter(
        fabric=fabric,
        kernel=kernel,
        llm=llm_client,
        mode="hybrid",
    )
    
    # Ingest conversation
    adapter.ingest_conversation(turns, workspace_id, user_id)
    
    # Answer questions
    for question in questions:
        result = adapter.answer_question(question, workspace_id, user_id)
        print(result["answer"])
        print(result["latency_ms"])

Comparison modes:
    - "raw": Use fabric.recall() directly (baseline)
    - "agent_rules": Use ChillBotAgent in rules mode
    - "agent_llm": Use ChillBotAgent in llm mode
    - "agent_hybrid": Use ChillBotAgent in hybrid mode (default)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# ==============================================
# BENCHMARK RESULT
# ==============================================

@dataclass
class BenchmarkResult:
    """Result from a benchmark question."""
    question: str = ""
    expected_answer: str = ""
    actual_answer: str = ""
    
    # Scoring
    score: float = 0.0              # 0 or 1 for binary, 0-1 for graded
    is_correct: bool = False
    
    # Metadata
    question_id: str = ""
    category: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    
    # Performance
    latency_ms: float = 0.0
    events_retrieved: int = 0
    events_used: int = 0
    
    # Provenance
    mode: str = "hybrid"
    query_intent: str = ""
    planner_mode: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "score": self.score,
            "is_correct": self.is_correct,
            "latency_ms": round(self.latency_ms, 2),
            "events_retrieved": self.events_retrieved,
            "events_used": self.events_used,
            "mode": self.mode,
            "query_intent": self.query_intent,
        }


# ==============================================
# BENCHMARK ADAPTER
# ==============================================

class BenchmarkAdapter:
    """
    Adapter for running KRNX/ChillBot against memory benchmarks.
    
    Supports multiple modes for comparison:
    - raw: Just fabric.recall() (baseline)
    - agent_*: Full ChillBotAgent pipeline
    """
    
    def __init__(
        self,
        fabric=None,
        kernel=None,
        llm=None,
        embeddings=None,
        vectors=None,
        cross_encoder=None,
        mode: str = "agent_hybrid",
    ):
        """
        Initialize benchmark adapter.
        
        Args:
            fabric: MemoryFabric instance
            kernel: KRNXController instance
            llm: LLM client
            embeddings: Embedding model
            vectors: VectorStore
            cross_encoder: CrossEncoderReranker
            mode: One of "raw", "agent_rules", "agent_llm", "agent_hybrid"
        """
        self.fabric = fabric
        self.kernel = kernel
        self.llm = llm
        self.embeddings = embeddings
        self.vectors = vectors
        self.cross_encoder = cross_encoder
        self.mode = mode
        
        # Initialize agent if using agent modes
        self._agent = None
        if mode.startswith("agent_"):
            self._init_agent(mode)
        
        logger.info(f"[ADAPTER] Initialized in {mode} mode")
    
    def _init_agent(self, mode: str):
        """Initialize ChillBotAgent for agent modes."""
        from chillbot_agent import ChillBotAgent
        
        # Extract planner mode from adapter mode
        if mode == "agent_rules":
            planner_mode = "rules"
        elif mode == "agent_llm":
            planner_mode = "llm"
        else:  # agent_hybrid
            planner_mode = "hybrid"
        
        self._agent = ChillBotAgent(
            fabric=self.fabric,
            kernel=self.kernel,
            llm=self.llm,
            embeddings=self.embeddings,
            vectors=self.vectors,
            cross_encoder=self.cross_encoder,
            mode=planner_mode,
        )
    
    # ==============================================
    # INGESTION
    # ==============================================
    
    def ingest_conversation(
        self,
        turns: List[Dict[str, Any]],
        workspace_id: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
    ) -> int:
        """
        Ingest conversation turns into KRNX.
        
        Args:
            turns: List of turn dicts with 'speaker', 'text', etc.
            workspace_id: Workspace to store in
            user_id: User ID
            session_id: Optional session ID
        
        Returns:
            Number of turns ingested
        """
        if self.fabric is None:
            raise RuntimeError("No fabric configured")
        
        count = 0
        for turn in turns:
            # Extract text
            text = turn.get("text") or turn.get("content") or turn.get("message", "")
            if not text:
                continue
            
            # Build content dict
            content = {
                "text": text,
                "speaker": turn.get("speaker", "unknown"),
                "role": turn.get("role", "user"),
            }
            
            # Optional fields
            if "dia_id" in turn:
                content["dia_id"] = turn["dia_id"]
            if "turn_id" in turn:
                content["turn_id"] = turn["turn_id"]
            
            # Build metadata
            metadata = {}
            if session_id:
                metadata["session_id"] = session_id
            if "timestamp" in turn:
                metadata["original_timestamp"] = turn["timestamp"]
            
            # Store
            self.fabric.remember(
                content=content,
                workspace_id=workspace_id,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
            )
            count += 1
        
        logger.info(f"[ADAPTER] Ingested {count} turns to {workspace_id}")
        return count
    
    def ingest_messages(
        self,
        messages: List[Dict[str, Any]],
        workspace_id: str,
        user_id: str = "default",
    ) -> int:
        """
        Ingest chat messages (OpenAI format).
        
        Args:
            messages: List of {role, content} dicts
            workspace_id: Workspace to store in
            user_id: User ID
        
        Returns:
            Number of messages ingested
        """
        turns = []
        for msg in messages:
            turns.append({
                "text": msg.get("content", ""),
                "role": msg.get("role", "user"),
                "speaker": msg.get("role", "user"),
            })
        
        return self.ingest_conversation(turns, workspace_id, user_id)
    
    # ==============================================
    # QUESTION ANSWERING
    # ==============================================
    
    def answer_question(
        self,
        question: str,
        workspace_id: str,
        user_id: str = "default",
        expected_answer: str = "",
        question_id: str = "",
        category: str = "",
    ) -> BenchmarkResult:
        """
        Answer a benchmark question.
        
        Args:
            question: The question to answer
            workspace_id: Workspace to search
            user_id: User filter
            expected_answer: Expected answer (for scoring)
            question_id: Question ID (for tracking)
            category: Question category (for breakdown)
        
        Returns:
            BenchmarkResult with answer and metadata
        """
        start_time = time.time()
        
        if self.mode == "raw":
            result = self._answer_raw(question, workspace_id, user_id)
        else:
            result = self._answer_agent(question, workspace_id, user_id)
        
        # Set metadata
        result.question = question
        result.expected_answer = expected_answer
        result.question_id = question_id
        result.category = category
        result.latency_ms = (time.time() - start_time) * 1000
        result.mode = self.mode
        
        return result
    
    def _answer_raw(
        self,
        question: str,
        workspace_id: str,
        user_id: str,
    ) -> BenchmarkResult:
        """
        Answer using raw fabric.recall() - baseline mode.
        """
        # Recall memories
        recall_result = self.fabric.recall(
            query=question,
            workspace_id=workspace_id,
            user_id=user_id,
            top_k=10,
        )
        
        events_retrieved = len(recall_result.memories)
        
        # Build context
        context_parts = []
        for i, memory in enumerate(recall_result.memories, 1):
            content = memory.content
            if isinstance(content, dict):
                text = content.get("text", str(content))
            else:
                text = str(content)
            context_parts.append(f"[{i}] {text}")
        
        context = "\n".join(context_parts)
        
        # Generate answer with LLM
        if self.llm:
            answer = self._generate_with_llm(question, context)
        else:
            answer = f"Found {events_retrieved} relevant memories."
        
        return BenchmarkResult(
            actual_answer=answer,
            events_retrieved=events_retrieved,
            events_used=events_retrieved,
            planner_mode="raw",
            query_intent="semantic",
        )
    
    def _answer_agent(
        self,
        question: str,
        workspace_id: str,
        user_id: str,
    ) -> BenchmarkResult:
        """
        Answer using ChillBotAgent - full intelligence mode.
        """
        if self._agent is None:
            raise RuntimeError("Agent not initialized")
        
        # Get full result from agent
        agent_result = self._agent.answer_with_details(
            question=question,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        
        return BenchmarkResult(
            actual_answer=agent_result.answer,
            events_retrieved=len(agent_result.retrieval_result.events) if agent_result.retrieval_result else 0,
            events_used=agent_result.assembled_context.events_included if agent_result.assembled_context else 0,
            planner_mode=agent_result.query_plan.planner_mode if agent_result.query_plan else "",
            query_intent=agent_result.query_plan.intent.value if agent_result.query_plan else "",
        )
    
    def _generate_with_llm(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate answer using LLM."""
        system_prompt = """You are answering questions based on conversation history.
Use the provided context to answer accurately. If the context doesn't contain
the answer, say "I don't have that information in the conversation history."
Keep answers concise and factual."""
        
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=500,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"[ADAPTER] LLM generation failed: {e}")
            return "Error generating answer"
    
    # ==============================================
    # BASELINE (NO MEMORY)
    # ==============================================
    
    def answer_baseline(
        self,
        question: str,
        expected_answer: str = "",
        question_id: str = "",
        category: str = "",
    ) -> BenchmarkResult:
        """
        Answer without any memory context (baseline).
        
        Used to establish the "no memory" baseline for comparison.
        """
        start_time = time.time()
        
        if self.llm:
            system_prompt = """You are answering questions about conversations.
You don't have access to any conversation history.
If you don't know the answer, say "I don't have that information."
Do not make up answers."""
            
            try:
                answer = self.llm.complete(
                    prompt=f"Question: {question}\n\nAnswer:",
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=500,
                )
            except Exception:
                answer = "I don't have that information."
        else:
            answer = "I don't have access to conversation history."
        
        return BenchmarkResult(
            question=question,
            expected_answer=expected_answer,
            actual_answer=answer,
            question_id=question_id,
            category=category,
            latency_ms=(time.time() - start_time) * 1000,
            events_retrieved=0,
            events_used=0,
            mode="baseline",
            planner_mode="none",
            query_intent="none",
        )
    
    # ==============================================
    # SCORING
    # ==============================================
    
    def score_answer(
        self,
        result: BenchmarkResult,
        scoring_method: str = "llm",
    ) -> BenchmarkResult:
        """
        Score an answer against expected.
        
        Args:
            result: BenchmarkResult to score
            scoring_method: "llm" or "exact"
        
        Returns:
            Updated BenchmarkResult with score
        """
        if not result.expected_answer:
            return result
        
        if scoring_method == "exact":
            result.is_correct = (
                result.actual_answer.lower().strip() == 
                result.expected_answer.lower().strip()
            )
            result.score = 1.0 if result.is_correct else 0.0
        
        elif scoring_method == "llm" and self.llm:
            result.score = self._score_with_llm(
                result.actual_answer,
                result.expected_answer,
            )
            result.is_correct = result.score >= 0.5
        
        return result
    
    def _score_with_llm(
        self,
        actual: str,
        expected: str,
    ) -> float:
        """Score answer using LLM-as-judge."""
        system_prompt = """You are evaluating if an answer is correct.
Compare the actual answer to the expected answer.
Consider semantic similarity - exact wording is not required.
Return ONLY a number from 0 to 1:
- 1.0: Completely correct
- 0.5: Partially correct
- 0.0: Incorrect or contradicts expected"""
        
        prompt = f"""Expected answer: {expected}
Actual answer: {actual}

Score (0-1):"""
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=10,
            )
            
            # Parse score
            import re
            match = re.search(r'([01]\.?\d*)', response)
            if match:
                return min(1.0, max(0.0, float(match.group(1))))
            return 0.0
        except Exception as e:
            logger.error(f"[ADAPTER] Scoring failed: {e}")
            return 0.0
    
    # ==============================================
    # UTILITY
    # ==============================================
    
    def set_mode(self, mode: str):
        """Change adapter mode."""
        self.mode = mode
        if mode.startswith("agent_"):
            self._init_agent(mode)
        logger.info(f"[ADAPTER] Mode changed to {mode}")
    
    def clear_workspace(self, workspace_id: str):
        """Clear a workspace (for test isolation)."""
        # Clear vectors if available
        if self.vectors:
            try:
                self.vectors.delete_collection(workspace_id)
            except Exception:
                pass
        
        # Note: STM/LTM cleanup would require kernel support
        logger.info(f"[ADAPTER] Cleared workspace {workspace_id}")


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'BenchmarkResult',
    'BenchmarkAdapter',
]
