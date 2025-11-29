"""
ChillBot Agent - Main Orchestration

The intelligence layer on top of KRNX kernel and fabric.
Provides intelligent memory retrieval and question answering.

Architecture:
    Query → Planner → Router → Processor → Assembler → LLM → Answer

Features:
- Three planner modes: rules (fast), llm (accurate), hybrid (default)
- Multi-strategy retrieval: temporal, entity, semantic, multi-hop
- Cross-encoder reranking for precision
- Relation-aware filtering (no stale data)
- Episode-grouped context assembly
- Source citations in answers

Usage:
    from chillbot_agent import ChillBotAgent
    
    agent = ChillBotAgent(
        fabric=fabric,
        kernel=kernel,
        llm=llm_client,
        mode="hybrid",
    )
    
    # Answer a question
    answer = agent.answer(
        question="What did we discuss about the project timeline?",
        workspace_id="workspace_123",
        user_id="user_456",
    )
    
    # With full result details
    result = agent.answer_with_details(question, workspace_id, user_id)
    print(result.answer)
    print(result.sources_used)
    print(result.latency_breakdown)

Constitution compliance:
- Uses kernel/fabric APIs (no direct DB access)
- Deterministic enrichment (no hidden LLM calls in processing)
- Transparent operation (full provenance in results)
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .planner import QueryPlanner, QueryPlan, QueryIntent
from .router import RetrievalRouter, RetrievalResult
from .processor import ResultProcessor, ProcessedResult
from .assembler import ContextAssembler, ContextConfig, AssembledContext

logger = logging.getLogger(__name__)


# ==============================================
# ANSWER RESULT
# ==============================================

@dataclass
class AnswerResult:
    """
    Full result from agent answer operation.
    
    Includes answer, sources, and detailed provenance.
    """
    # The answer
    answer: str = ""
    
    # Sources used
    sources: List[str] = field(default_factory=list)  # event_ids
    source_map: Dict[int, str] = field(default_factory=dict)  # citation index -> event_id
    
    # Query analysis
    query_plan: Optional[QueryPlan] = None
    
    # Pipeline results
    retrieval_result: Optional[RetrievalResult] = None
    processed_result: Optional[ProcessedResult] = None
    assembled_context: Optional[AssembledContext] = None
    
    # Latency breakdown
    latency_ms: float = 0.0
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    workspace_id: str = ""
    user_id: str = ""
    mode: str = "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "latency_ms": round(self.latency_ms, 2),
            "latency_breakdown": {
                k: round(v, 2) for k, v in self.latency_breakdown.items()
            },
            "mode": self.mode,
            "query_intent": self.query_plan.intent.value if self.query_plan else None,
            "events_retrieved": len(self.retrieval_result.events) if self.retrieval_result else 0,
            "events_used": self.assembled_context.events_included if self.assembled_context else 0,
        }


# ==============================================
# CHILLBOT AGENT
# ==============================================

class ChillBotAgent:
    """
    ChillBot Agent - Intelligence layer for KRNX.
    
    Provides intelligent memory retrieval and question answering
    by orchestrating planner, router, processor, and assembler.
    """
    
    def __init__(
        self,
        fabric=None,
        kernel=None,
        llm=None,
        embeddings=None,
        vectors=None,
        cross_encoder=None,
        mode: str = "hybrid",
        confidence_threshold: float = 0.7,
        context_config: Optional[ContextConfig] = None,
    ):
        """
        Initialize ChillBot agent.
        
        Args:
            fabric: MemoryFabric instance
            kernel: KRNXController instance
            llm: LLM client with complete(prompt, system_prompt, **kwargs) method
            embeddings: Embedding model (optional, uses fabric's if not provided)
            vectors: VectorStore (optional, uses fabric's if not provided)
            cross_encoder: CrossEncoderReranker (optional)
            mode: Planner mode - "rules", "llm", or "hybrid"
            confidence_threshold: Confidence needed to skip LLM in hybrid mode
            context_config: Context assembly configuration
        """
        self.fabric = fabric
        self.kernel = kernel
        self.llm = llm
        self.mode = mode
        
        # Use fabric's embeddings/vectors if not provided
        self.embeddings = embeddings or (fabric.embeddings if fabric else None)
        self.vectors = vectors or (fabric.vectors if fabric else None)
        
        # Initialize components
        self._planner = QueryPlanner(
            llm_client=llm,
            mode=mode,
            confidence_threshold=confidence_threshold,
        )
        
        self._router = RetrievalRouter(
            fabric=fabric,
            kernel=kernel,
            embeddings=self.embeddings,
            vectors=self.vectors,
        )
        
        self._processor = ResultProcessor(
            cross_encoder=cross_encoder,
            enable_reranking=(cross_encoder is not None),
        )
        
        self._assembler = ContextAssembler(
            config=context_config or ContextConfig(),
        )
        
        logger.info(f"[AGENT] ChillBot initialized in {mode} mode")
    
    # ==============================================
    # MAIN API
    # ==============================================
    
    def answer(
        self,
        question: str,
        workspace_id: str,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Answer a question using memory context.
        
        Simple API that returns just the answer string.
        
        Args:
            question: The question to answer
            workspace_id: Workspace to search
            user_id: Optional user filter
        
        Returns:
            Answer string
        """
        result = self.answer_with_details(question, workspace_id, user_id)
        return result.answer
    
    def answer_with_details(
        self,
        question: str,
        workspace_id: str,
        user_id: Optional[str] = None,
    ) -> AnswerResult:
        """
        Answer a question with full result details.
        
        Returns complete provenance including sources,
        latency breakdown, and intermediate results.
        
        Args:
            question: The question to answer
            workspace_id: Workspace to search
            user_id: Optional user filter
        
        Returns:
            AnswerResult with full details
        """
        total_start = time.time()
        latency = {}
        
        # Step 1: Plan query
        plan_start = time.time()
        plan = self._planner.plan(question)
        latency["planning"] = (time.time() - plan_start) * 1000
        
        logger.info(
            f"[AGENT] Query planned: intent={plan.intent.value}, "
            f"confidence={plan.confidence:.2f}, mode={plan.planner_mode}"
        )
        
        # Step 2: Retrieve
        retrieve_start = time.time()
        retrieval = self._router.retrieve(
            plan=plan,
            query=question,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        latency["retrieval"] = (time.time() - retrieve_start) * 1000
        
        # Step 3: Process
        process_start = time.time()
        processed = self._processor.process(
            retrieval_result=retrieval,
            query=question,
            plan=plan,
        )
        latency["processing"] = (time.time() - process_start) * 1000
        
        # Step 4: Assemble context
        assemble_start = time.time()
        assembled = self._assembler.assemble(
            processed=processed,
            query=question,
        )
        latency["assembly"] = (time.time() - assemble_start) * 1000
        
        # Step 5: Generate answer
        generate_start = time.time()
        answer = self._generate_answer(question, assembled)
        latency["generation"] = (time.time() - generate_start) * 1000
        
        # Build result
        total_latency = (time.time() - total_start) * 1000
        
        # Extract source event_ids
        sources = list(assembled.source_map.values())
        
        return AnswerResult(
            answer=answer,
            sources=sources,
            source_map=assembled.source_map,
            query_plan=plan,
            retrieval_result=retrieval,
            processed_result=processed,
            assembled_context=assembled,
            latency_ms=total_latency,
            latency_breakdown=latency,
            workspace_id=workspace_id,
            user_id=user_id or "",
            mode=self.mode,
        )
    
    # ==============================================
    # ANSWER GENERATION
    # ==============================================
    
    def _generate_answer(
        self,
        question: str,
        assembled: AssembledContext,
    ) -> str:
        """
        Generate answer using LLM.
        
        Args:
            question: The question
            assembled: Assembled context
        
        Returns:
            Answer string
        """
        if self.llm is None:
            return self._generate_answer_no_llm(question, assembled)
        
        # Build prompt
        context = assembled.context
        
        if isinstance(context, list):
            # Messages format - use directly
            messages = context + [
                {"role": "user", "content": question}
            ]
            
            try:
                response = self.llm.chat(messages=messages, temperature=0.0)
                return response
            except AttributeError:
                # Fallback to complete if no chat method
                context_text = context[0]["content"] if context else ""
        elif isinstance(context, dict):
            # JSON format
            import json
            context_text = json.dumps(context.get("memories", []), indent=2)
        else:
            # Text format
            context_text = context
        
        # Build prompt for complete API
        system_prompt = """You are a helpful assistant with access to conversation history.
Use the provided context to answer questions accurately.
If you cite information from the context, reference it by number [N].
If the context doesn't contain relevant information, say so honestly."""
        
        prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=1000,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"[AGENT] LLM generation failed: {e}")
            return self._generate_answer_no_llm(question, assembled)
    
    def _generate_answer_no_llm(
        self,
        question: str,
        assembled: AssembledContext,
    ) -> str:
        """
        Generate answer without LLM (return context summary).
        
        Used when LLM is not available.
        """
        if assembled.events_included == 0:
            return "No relevant context found to answer this question."
        
        context = assembled.context
        if isinstance(context, str):
            return f"Found {assembled.events_included} relevant memories:\n\n{context}"
        elif isinstance(context, list):
            return context[0].get("content", "Context found but could not format.")
        else:
            return f"Found {assembled.events_included} relevant memories."
    
    # ==============================================
    # REMEMBER (Write Path)
    # ==============================================
    
    def remember(
        self,
        content: str,
        workspace_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory through the fabric.
        
        Convenience method that delegates to fabric.remember().
        
        Args:
            content: Content to remember
            workspace_id: Workspace to store in
            user_id: Optional user association
            metadata: Optional metadata
        
        Returns:
            Event ID
        """
        if self.fabric is None:
            raise RuntimeError("No fabric configured for remember()")
        
        return self.fabric.remember(
            content=content,
            workspace_id=workspace_id,
            user_id=user_id,
            metadata=metadata,
        )
    
    # ==============================================
    # CONFIGURATION
    # ==============================================
    
    def set_mode(self, mode: str):
        """
        Change planner mode.
        
        Args:
            mode: "rules", "llm", or "hybrid"
        """
        self._planner.set_mode(mode)
        self.mode = mode
        logger.info(f"[AGENT] Mode changed to {mode}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "mode": self.mode,
            "has_fabric": self.fabric is not None,
            "has_kernel": self.kernel is not None,
            "has_llm": self.llm is not None,
            "has_cross_encoder": self._processor.cross_encoder is not None,
        }


# ==============================================
# CONVENIENCE FACTORY
# ==============================================

def create_agent(
    fabric=None,
    kernel=None,
    llm=None,
    mode: str = "hybrid",
    **kwargs,
) -> ChillBotAgent:
    """
    Create a ChillBot agent with sensible defaults.
    
    Args:
        fabric: MemoryFabric instance
        kernel: KRNXController instance
        llm: LLM client
        mode: Planner mode
        **kwargs: Additional arguments for ChillBotAgent
    
    Returns:
        Configured ChillBotAgent
    """
    return ChillBotAgent(
        fabric=fabric,
        kernel=kernel,
        llm=llm,
        mode=mode,
        **kwargs,
    )


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'AnswerResult',
    'ChillBotAgent',
    'create_agent',
]
