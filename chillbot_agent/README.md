ChillBot Agent Package
View your package
File Structure
chillbot_agent/
├── __init__.py          (4.5K)  - Package exports
├── planner.py           (21K)   - Query classification (rules/llm/hybrid)
├── router.py            (15K)   - Retrieval dispatch (temporal/entity/semantic/multi-hop)
├── processor.py         (15K)   - Reranking, filtering, deduplication
├── assembler.py         (18K)   - Context building (salience, episodes, tokens)
├── agent.py             (15K)   - Main orchestrator (ChillBotAgent)
└── benchmark_adapter.py (17K)   - LOCOMO/LongMemEval integration
Architecture
Query
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        QueryPlanner                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ RuleBased   │  │ LLMBased    │  │ Hybrid (default)        │  │
│  │ - temporal  │  │ - ambiguity │  │ - rules first           │  │
│  │ - entity    │  │ - complex   │  │ - LLM if uncertain      │  │
│  │ - multi_hop │  │ - nuance    │  │ - confidence threshold  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ QueryPlan
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RetrievalRouter                              │
│  TEMPORAL  → kernel.replay_to_timestamp() / get_events_in_range │
│  ENTITY    → filter by entities (@ # "quoted")                  │
│  SEMANTIC  → fabric.recall() with vector search                 │
│  MULTI_HOP → iterative retrieval with expansion                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │ RetrievalResult
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ResultProcessor                               │
│  1. Cross-encoder rerank (higher precision)                     │
│  2. Deduplicate (content hash)                                  │
│  3. Filter superseded (relation chains)                         │
│  4. Filter ephemeral (retention class)                          │
│  5. Salience scoring                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ ProcessedResult
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ContextAssembler                               │
│  - Group by episode (chronological within)                      │
│  - Sort by salience (most important first)                      │
│  - Token budget (respects LLM limits)                           │
│  - Output formats: text / messages / json                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ AssembledContext
                            ▼
                        LLM Answer
Usage
pythonfrom chillbot_agent import ChillBotAgent, BenchmarkAdapter

# For direct use
agent = ChillBotAgent(
    fabric=fabric,
    kernel=kernel,
    llm=llm_client,
    mode="hybrid",  # or "rules", "llm"
)

answer = agent.answer(
    question="What did we discuss about the timeline?",
    workspace_id="ws_123",
)

# For benchmarking
adapter = BenchmarkAdapter(
    fabric=fabric,
    kernel=kernel,
    llm=llm_client,
    mode="agent_hybrid",  # or "agent_rules", "agent_llm", "raw"
)

# Ingest data
adapter.ingest_conversation(turns, workspace_id, user_id)

# Answer + score
result = adapter.answer_question(question, workspace_id, user_id)
result = adapter.score_answer(result, scoring_method="llm")