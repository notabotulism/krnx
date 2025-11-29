# KRNX

Temporal memory infrastructure for AI agents.

Reconstruct any moment. Debug any decision. Durable memory in one import.
```bash
pip install krnx
```

## What it does

KRNX is an append-only event store with temporal replay. Think of it as the memory layer beneath your AI application.

- **Temporal replay** — Reconstruct state at any point in history
- **Hash-chain provenance** — Cryptographic proof of what happened and when
- **Multi-agent coordination** — Agents share state through a durable substrate
- **Crash recovery** — Process dies, restarts, picks up exactly where it left off

## Quickstart
```python
from krnx import KRNXClient

# Local mode (embedded kernel)
client = KRNXClient(data_path="./krnx-data", redis_host="localhost")

# Store memories
client.remember(
    content={"query": "What's our Q4 budget?", "response": "$2.5M"},
    workspace="acme-corp",
    user="alice"
)

# Recall recent memories
memories = client.recall(workspace="acme-corp", user="alice", limit=10)

# Time-travel: reconstruct state from 1 hour ago
import time
one_hour_ago = time.time() - 3600
history = client.replay(workspace="acme-corp", timestamp=one_hour_ago)
```

## Multi-Agent Coordination

Agents coordinate through a shared substrate, not message passing.
```python
# Create a consumer group
client.create_agent_group(workspace="project", group="workers")

# Agent consumes events from shared substrate
events = client.consume(
    workspace="project",
    group="workers",
    agent_id="agent_001",
    count=10
)

# Process and acknowledge
for event in events:
    process(event)
    client.ack(workspace="project", group="workers", message_id=event["message_id"])
```

Each event is delivered to exactly one agent in the group. If an agent dies before acking, the event is redelivered.

## Temporal Replay

Reconstruct complete state at any point in history.
```python
# What did the system know at midnight last night?
midnight = datetime(2024, 1, 15, 0, 0).timestamp()
state = client.state(workspace="acme-corp", as_of=midnight)

# Replay all events in a time window
events = client.replay(
    workspace="acme-corp",
    start=midnight,
    end=midnight + 3600  # Next hour
)

# Get event distribution over time
timeline = client.timeline(workspace="acme-corp", bucket_size="hour")
```

## Provenance

Cryptographic hash-chain for audit trails.
```python
# Get the provenance chain for an event
chain = client.get_provenance_chain(
    workspace="acme-corp",
    event_id="evt_abc123",
    max_depth=100
)

# Verify integrity (detect tampering)
result = client.verify_event(workspace="acme-corp", event_id="evt_abc123")
assert result["verified"] == True
```

## Semantic Recall

Find relevant memories using embeddings.
```python
# Semantic search
results = client.semantic_recall(
    query="budget discussions Q4",
    workspace="acme-corp",
    top_k=10
)

# Build LLM-ready context
context = client.build_context(
    query="What was decided about the Q4 budget?",
    workspace="acme-corp",
    max_tokens=4000,
    format="text"  # or "json", "messages"
)
```

## Fact Versioning (Supersession)

When facts change, old versions remain in history.
```python
# Original fact
event_id = client.remember(
    content={"ceo": "Alice"},
    workspace="acme-corp"
)

# Fact changes — supersede, don't delete
client.supersede(
    workspace="acme-corp",
    old_event_id=event_id,
    new_content={"ceo": "Bob"},
    reason="Leadership change March 2024"
)

# Get version history
versions = client.get_version_chain(workspace="acme-corp", event_id=event_id)
```

## Branches

Explore alternative memory states without affecting the main timeline.
```python
# Create a branch to try something
client.create_branch(
    workspace="acme-corp",
    branch_id="experiment-1",
    parent_branch="main"
)

# Do work on the branch...

# Merge back if it worked
client.merge_branch(
    workspace="acme-corp",
    source_branch="experiment-1",
    target_branch="main"
)
```

## Remote Mode

Run KRNX as a service.
```python
# Connect to remote KRNX server
client = KRNXClient(
    base_url="http://localhost:6380",
    api_key="your-api-key"
)

# Same API, network-backed
client.remember(content="Hello", workspace="demo")
```

## Async Support

For async applications (FastAPI, etc.):
```python
from krnx import AsyncKRNXClient

async with AsyncKRNXClient(base_url="http://localhost:6380") as client:
    await client.remember(content="Hello", workspace="demo")
    memories = await client.recall(workspace="demo")
```

## Requirements

- Python 3.9+
- Redis (for local mode)

## Architecture
```
┌─────────────────────────────────────────────────┐
│                   Your App                       │
├─────────────────────────────────────────────────┤
│                 KRNXClient                       │
├──────────┬──────────┬──────────┬────────────────┤
│   STM    │   LTM    │ Compute  │   Provenance   │
│ (Redis)  │ (SQLite) │  Layer   │  (Hash-chain)  │
│  0-24h   │  0-30d   │ Optional │   Integrity    │
└──────────┴──────────┴──────────┴────────────────┘
```

- **STM (Short-Term Memory)**: Redis — hot storage, sub-millisecond access
- **LTM (Long-Term Memory)**: SQLite WAL — warm storage, durable
- **Compute Layer**: Embeddings, salience scoring — optional, layered on top
- **Provenance**: Hash-chain linking events — cryptographic audit trail

## Links

- [Documentation](https://krnx.dev/docs)
- [API Reference](https://krnx.dev/api)
- [GitHub](https://github.com/chillbot/krnx)
- [Examples](https://github.com/chillbot/krnx/tree/main/examples)

## License

MIT