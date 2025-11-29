"""
ChillBot Agent - Intelligence Layer for KRNX

The application layer that demonstrates KRNX kernel and fabric capabilities.
Provides intelligent memory retrieval and question answering.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                     ChillBotAgent                            │
    │  ┌───────────┐  ┌────────┐  ┌───────────┐  ┌───────────┐   │
    │  │  Planner  │→ │ Router │→ │ Processor │→ │ Assembler │   │
    │  │           │  │        │  │           │  │           │   │
    │  │ rules     │  │temporal│  │ rerank    │  │ salience  │   │
    │  │ llm       │  │entity  │  │ dedupe    │  │ episode   │   │
    │  │ hybrid    │  │semantic│  │ filter    │  │ tokens    │   │
    │  └───────────┘  └────────┘  └───────────┘  └───────────┘   │
    └──────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
         ┌─────────┐  ┌─────────┐  ┌─────────┐
         │ Kernel  │  │ Fabric  │  │   LLM   │
         │ (KRNX)  │  │ (KRNX)  │  │         │
         └─────────┘  └─────────┘  └─────────┘

Usage:
    from chillbot_agent import ChillBotAgent, create_agent
    
    # Create agent with KRNX components
    agent = ChillBotAgent(
        fabric=fabric,
        kernel=kernel,
        llm=llm_client,
        mode="hybrid",  # or "rules", "llm"
    )
    
    # Simple answer
    answer = agent.answer(
        question="What was the budget we discussed?",
        workspace_id="workspace_123",
    )
    
    # Full details
    result = agent.answer_with_details(question, workspace_id)
    print(result.answer)
    print(result.sources)
    print(result.latency_breakdown)

Modes:
    - rules: Fast, deterministic, zero API cost
    - llm: Accurate, handles ambiguity, costs per query
    - hybrid: Rules first, LLM fallback (default)

Constitution compliance:
    - Uses kernel/fabric APIs (no direct DB access)
    - Deterministic enrichment in processing
    - Transparent operation (full provenance)
"""

__version__ = "0.1.0"

# ==============================================
# MAIN EXPORTS
# ==============================================

from .agent import (
    ChillBotAgent,
    AnswerResult,
    create_agent,
)

from .planner import (
    QueryPlanner,
    QueryPlan,
    QueryIntent,
    TemporalRange,
    RuleBasedPlanner,
    LLMPlanner,
)

from .router import (
    RetrievalRouter,
    RetrievalResult,
)

from .processor import (
    ResultProcessor,
    ProcessedResult,
)

from .assembler import (
    ContextAssembler,
    ContextConfig,
    AssembledContext,
)

# Benchmark adapter (optional import)
try:
    from .benchmark_adapter import (
        BenchmarkAdapter,
        BenchmarkResult,
    )
    _HAS_BENCHMARK = True
except ImportError:
    _HAS_BENCHMARK = False


# ==============================================
# ALL EXPORTS
# ==============================================

__all__ = [
    # Version
    "__version__",
    
    # Main agent
    "ChillBotAgent",
    "AnswerResult",
    "create_agent",
    
    # Planner
    "QueryPlanner",
    "QueryPlan",
    "QueryIntent",
    "TemporalRange",
    "RuleBasedPlanner",
    "LLMPlanner",
    
    # Router
    "RetrievalRouter",
    "RetrievalResult",
    
    # Processor
    "ResultProcessor",
    "ProcessedResult",
    
    # Assembler
    "ContextAssembler",
    "ContextConfig",
    "AssembledContext",
    
    # Benchmark (if available)
    "BenchmarkAdapter",
    "BenchmarkResult",
]
