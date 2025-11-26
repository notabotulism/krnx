# Chillbot - AI Memory Infrastructure

> The substrate for AI agents that remember.

**Durable. Replayable. Yours.**

## What is Chillbot?

Chillbot is pure AI memory infrastructure. We're the layer below Mem0, Letta, and Zep — they build frameworks, we build substrate. Think of us as Redis for AI memory.

```python
from chillbot import Memory

memory = Memory("my-agent")
memory.remember("user loves hiking in the Alps")
memories = memory.recall("outdoor activities")
context = memory.context("plan a weekend trip", max_tokens=4000)
```

Three lines to durable AI memory. No config. No infra. No ops.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CHILLBOT SDK                                   │
│                                                                             │
│   from chillbot import Memory                                               │
│   memory = Memory("my-agent")                                               │
│   memory.remember(...) / memory.recall(...) / memory.context(...)          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              MEMORY FABRIC                                  │
│                                                                             │
│   • Identity (agent/scope resolution)                                       │
│   • Context Builder (assemble memories for LLM)                             │
│   • Retention Policies (trigger, not compute)                               │
│   • Read Orchestration (Kernel + Compute → unified response)                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              KRNX COMPUTE                                   │
│                                                                             │
│   • Job Queue (SQLite-backed)                                               │
│   • Workers (async Python)                                                  │
│   • Embeddings (sentence-transformers)                                      │
│   • Vector Store (Qdrant)                                                   │
│   • Salience Scoring (recency, frequency, semantic)                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              KRNX KERNEL                                    │
│                                                                             │
│   • Event Store (append-only, hash-chained)                                 │
│   • STM (Redis/KeyDB)                                                       │
│   • LTM (SQLite WAL)                                                        │
│   • Replay Engine                                                           │
│   • Provenance                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Kernel** | ✅ Complete | Event store, STM/LTM, replay, hash-chain |
| **Kernel SDK** | ✅ Complete | KRNXClient, Agent, async support |
| **Compute** | ✅ Complete | Job queue, embeddings, vectors, salience |
| **Fabric** | 🔄 Next | Orchestration, context builder |
| **Memory Class** | 🔄 Next | Simple "it just works" interface |
| **Hosting** | 📋 Planned | Managed infrastructure |

---

## Quick Start

### Installation

```bash
pip install chillbot
```

### Basic Usage

```python
from chillbot import Memory

# Initialize memory for your agent
memory = Memory("my-agent")

# Store memories
memory.remember("User's name is Alice")
memory.remember("User prefers dark mode")
memory.remember({"type": "preference", "key": "language", "value": "Python"})

# Semantic recall
memories = memory.recall("what are their preferences?")
for m in memories:
    print(f"{m['score']:.2f}: {m['content']}")

# Build LLM context
context = memory.context(
    query="help them with a coding question",
    max_tokens=4000
)

# Time travel
history = memory.replay(since="1h")
```

### Infrastructure Access

For power users who need direct control:

```python
from chillbot.kernel import KRNXClient
from chillbot.compute import JobQueue, VectorStore, EmbeddingEngine

# Direct kernel access
krnx = KRNXClient(base_url="http://localhost:6380")
krnx.remember(workspace="app", user="user1", content={"msg": "hello"})

# Direct compute access
queue = JobQueue("jobs.db")
vectors = VectorStore(url="http://localhost:6333")
embeddings = EmbeddingEngine()
```

---

## Package Structure

```
chillbot/
├── __init__.py                 # Main exports: Memory, Agent
├── memory.py                   # Memory class (simple interface)
│
├── kernel/                     # KRNX Kernel (your existing code)
│   ├── __init__.py
│   ├── client.py               # KRNXClient, AsyncKRNXClient
│   ├── agent.py                # Agent base class
│   ├── exceptions.py           # Error types
│   ├── controller.py           # Core kernel logic
│   ├── stm.py                  # Redis STM
│   ├── ltm.py                  # SQLite LTM
│   └── models.py               # Data models
│
├── compute/                    # KRNX Compute (NEW)
│   ├── __init__.py             # Exports
│   ├── queue.py                # SQLite job queue
│   ├── embeddings.py           # Embedding generation
│   ├── vectors.py              # Qdrant vector store
│   ├── salience.py             # Importance scoring
│   └── worker.py               # Background worker
│
├── fabric/                     # Memory Fabric (NEXT)
│   ├── __init__.py
│   ├── orchestrator.py         # Main fabric class
│   ├── context.py              # Context builder
│   ├── identity.py             # Agent/scope resolution
│   └── retention.py            # Retention policies
│
└── server/                     # API Server
    ├── __init__.py
    ├── api.py                  # FastAPI routes
    └── config.py               # Configuration
```

---

## Component Details

### Kernel (Complete)

The foundation. Append-only event store with temporal replay.

```python
from chillbot.kernel import KRNXClient

krnx = KRNXClient()

# Store
krnx.remember(
    workspace="chatbot",
    user="user_123",
    content={"role": "user", "message": "Hello!"},
    channel="chat"
)

# Query
events = krnx.recall(workspace="chatbot", user="user_123", limit=10)

# Time travel
past_state = krnx.replay(
    workspace="chatbot",
    user="user_123",
    timestamp=yesterday
)

# GDPR
krnx.erase_user(workspace="chatbot", user="user_123")
```

### Compute (Complete)

Background processing for embeddings, vectors, and scoring.

```python
from chillbot.compute import JobQueue, JobType, EmbeddingEngine, VectorStore

# Job Queue
queue = JobQueue("jobs.db")
job_id = queue.enqueue(
    job_type=JobType.EMBED,
    workspace_id="my-app",
    payload={"event_id": "evt_123", "text": "Hello world"}
)

# Embeddings
embeddings = EmbeddingEngine()  # Uses all-MiniLM-L6-v2 by default
vector = embeddings.embed("Hello world")  # Returns 384-dim vector

# Vector Store
vectors = VectorStore(url="http://localhost:6333")
vectors.ensure_collection("my-app", dimension=384)
vectors.index("my-app", "evt_123", vector, {"text": "Hello"})
results = vectors.search("my-app", query_vector, top_k=10)

# Salience
from chillbot.compute import SalienceEngine, SalienceMethod
salience = SalienceEngine()
score = salience.compute(
    event_id="evt_123",
    timestamp=time.time(),
    access_count=5,
    method=SalienceMethod.COMPOSITE
)
```

### Fabric (Next)

Orchestration layer that ties Kernel and Compute together.

```python
from chillbot.fabric import MemoryFabric

fabric = MemoryFabric(kernel, compute)

# Remember (writes to kernel + enqueues embedding job)
fabric.remember(workspace, user, "user likes hiking")

# Recall (searches vectors, enriches from kernel)
memories = fabric.recall(workspace, "outdoor activities")

# Context (builds LLM-ready context)
context = fabric.context(workspace, user, "plan a trip", max_tokens=4000)
```

---

## Deployment

### Local Development

```bash
# Start dependencies
docker-compose up -d redis qdrant

# Run kernel server
python -m chillbot.server

# Run compute worker
python -m chillbot.compute.worker
```

### Docker Compose

```yaml
version: '3.8'
services:
  kernel:
    build: .
    command: python -m chillbot.server
    ports:
      - "6380:6380"
    depends_on:
      - redis
      - qdrant
    volumes:
      - ./data:/data

  worker:
    build: .
    command: python -m chillbot.compute.worker
    depends_on:
      - kernel
      - qdrant
    volumes:
      - ./data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

### Hosted (Coming Soon)

```python
from chillbot import Memory

# Just works — connects to chillbot.io
memory = Memory("my-agent", api_key="chillbot_xxx")
```

---

## File Placement Guide

When integrating your existing KRNX kernel code:

```
YOUR EXISTING FILES              →  CHILLBOT LOCATION
─────────────────────────────────────────────────────────
krnx_sdk/client.py               →  chillbot/kernel/client.py
krnx_sdk/agent.py                →  chillbot/kernel/agent.py
krnx_sdk/exceptions.py           →  chillbot/kernel/exceptions.py
krnx/controller.py               →  chillbot/kernel/controller.py
krnx/stm.py                      →  chillbot/kernel/stm.py
krnx/ltm.py                      →  chillbot/kernel/ltm.py
krnx/models.py                   →  chillbot/kernel/models.py
krnx/api_server.py               →  chillbot/server/api.py
```

---

## Roadmap

### Phase 1: Substrate (Current)
- [x] Kernel (event store, STM/LTM, replay)
- [x] Kernel SDK (KRNXClient, Agent)
- [x] Compute (queue, embeddings, vectors, salience)
- [ ] Fabric (orchestrator, context builder)
- [ ] Memory class (simple interface)

### Phase 2: Hosting
- [ ] Control plane (auth, provisioning, billing)
- [ ] Primitive hosting (SQLite, Redis, Qdrant)
- [ ] Full stack hosting (one-click deploy)
- [ ] Backup/restore service

### Phase 3: Scale
- [ ] Multi-tenant optimization
- [ ] Enterprise features
- [ ] Self-host support
- [ ] Open source core

---

## Philosophy

From the KRNX Constitution:

> **Purity**: The kernel remains simple, durable, and minimal. No magic. No guessing. No hidden state.
>
> **Modularity**: Every component must stand alone. The stack must compose. Nothing is mandatory.
>
> **Continuity**: Agents must be able to remember, replay, and evolve — reliably — across time.
>
> **Transparency**: Debuggability is sacred. Observability is necessary. Predictability is non-negotiable.

---

## License

MIT

---

## Links

- Website: https://chillbot.io
- Docs: https://docs.chillbot.io
- GitHub: https://github.com/chillbot/chillbot
- Discord: https://discord.gg/chillbot
