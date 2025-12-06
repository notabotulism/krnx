# KRNX API Guide

**Temporal memory for AI agents. Zero config. Just works.**

```python
pip install krnxlite
```

---

## 30 Seconds to Memory

```python
from krnxlite import mem

# Store anything
mem.store("user said hello")
mem.store({"action": "approve", "confidence": 0.95})

# Get it back
events = mem.all()
events = mem.last(10)
events = mem.search("approve")

# Time travel
state = mem.at(timestamp)
events = mem.since(event_id)

# Verify nothing was tampered
assert mem.verify()
```

That's it. SQLite under the hood. Hash chain for integrity. Works offline.

---

## Agent Tracking

Track what your AI agent thinks, says, and does:

```python
from krnxlite import Agent

agent = Agent("customer-service")

# Record behavior
agent.think("Checking refund policy...")
agent.hear("I want a refund!", from_="customer")
agent.say("I'd be happy to help with that.")
agent.act("lookup_order", order_id="12345")
agent.observe({"customer_sentiment": "frustrated"})
agent.result("refund_processed", success=True, amount=50.00)

# Query history
timeline = agent.timeline()              # Everything
thoughts = agent.timeline(type="think")  # Just thinking
recent = agent.last(5)                   # Last 5 events
```

### Point-in-Time Queries

What did the agent know at a specific moment?

```python
# Get state at timestamp
state = agent.at(1701705600)

# Get events since a decision
events = agent.since("evt_abc123")

# Search for something specific  
results = agent.search("policy bypass")
```

### Branching (Time Travel Debugging)

Found a bug? Branch from before it happened:

```python
# Create branch from specific event
agent.branch("fix-attempt", from_event="evt_bad_decision")

# Now you're on the new branch
agent.think("Let me try a different approach...")
agent.act("alternative_action")

# Switch between branches
agent.checkout("main")      # Back to original
agent.checkout("fix-attempt")  # Back to experiment

# Compare outcomes
main_events = agent.checkout("main").timeline()
fixed_events = agent.checkout("fix-attempt").timeline()
```

### Hash Chain Integrity

Every event is cryptographically chained:

```python
# Verify nothing was tampered
assert agent.verify()

# Each event has:
event = agent.get("evt_abc123")
print(event.hash)         # This event's hash
print(event.parent_hash)  # Previous event's hash

# Chain: evt_0 → evt_1 → evt_2 → ...
#        hash_0 ← hash_1 ← hash_2
```

---

## Multi-Agent Coordination

Multiple agents, one shared memory:

```python
from krnxlite import Swarm

swarm = Swarm("code-review")

# Create agents
security = swarm.agent("security")
performance = swarm.agent("performance")
style = swarm.agent("style")

# Each agent works independently
security.think("Checking for SQL injection...")
security.act("flag_issue", file="db.py", line=42, severity="high")

performance.think("Analyzing hot paths...")
performance.act("flag_issue", file="api.py", line=100, severity="medium")

# See everyone's work
all_events = swarm.timeline()
security_only = swarm.timeline(agent="security")

# Verify the whole swarm's history
assert swarm.verify()
```

### Consumer Groups

Coordinate work without duplicating effort:

```python
# Agent joins a consumer group
events = swarm.consume("reviewers", "security-agent-1", count=10)

for event in events:
    # Process the event
    process(event)
    
    # Mark as processed (won't be returned again)
    swarm.ack("reviewers", "security-agent-1", event.event_id)
```

---

## Decorator API

Auto-track function calls:

```python
from krnxlite import tracked

@tracked("order-processor")
def process_order(order_id: str, items: list):
    # ... your logic ...
    return {"status": "confirmed", "total": 99.99}

# Every call is automatically recorded:
# - Function name
# - Arguments
# - Return value type
# - Duration
# - Errors (if any)

process_order("ORD-123", ["item1", "item2"])
```

---

## Configuration

### Custom Storage Path

```python
from krnxlite import Agent, mem

# Global memory
mem.configure(workspace="my-app", path="./data")

# Per-agent
agent = Agent("my-agent", path="./agent-data")
```

### Workspace Isolation

Each workspace is a separate database:

```python
# These are completely isolated
agent1 = Agent("bot", workspace="production")
agent2 = Agent("bot", workspace="staging")
```

---

## Full API Reference

### Memory (Simple API)

| Method | Description |
|--------|-------------|
| `mem.store(content, type, agent)` | Store event |
| `mem.all(limit)` | Get all events |
| `mem.last(n)` | Get last N events |
| `mem.get(event_id)` | Get single event |
| `mem.at(timestamp)` | State at point in time |
| `mem.since(event_id)` | Events since event |
| `mem.search(query)` | Search by content |
| `mem.replay(start, end)` | Replay time range |
| `mem.branch(name, from_event)` | Create branch |
| `mem.branches()` | List branches |
| `mem.verify()` | Verify hash chain |
| `mem.count()` | Count events |
| `mem.clear()` | Clear all events |

### Agent

| Method | Description |
|--------|-------------|
| `agent.think(thought, confidence)` | Record thinking |
| `agent.say(message, to)` | Record outgoing message |
| `agent.hear(message, from_)` | Record incoming message |
| `agent.act(action, **params)` | Record action |
| `agent.observe(observation)` | Record observation |
| `agent.result(outcome, success)` | Record result |
| `agent.error(error)` | Record error |
| `agent.log(message, level)` | Record log |
| `agent.timeline(type, limit)` | Get timeline |
| `agent.last(n, type)` | Get last N events |
| `agent.at(timestamp)` | State at time |
| `agent.since(event_id)` | Events since |
| `agent.search(query)` | Search events |
| `agent.branch(name, from_event)` | Create & switch branch |
| `agent.checkout(branch)` | Switch branch |
| `agent.branches()` | List branches |
| `agent.replay(start, end)` | Replay range |
| `agent.verify()` | Verify integrity |

### Swarm

| Method | Description |
|--------|-------------|
| `swarm.agent(name)` | Get/create agent |
| `swarm.spawn(*names)` | Create multiple agents |
| `swarm.timeline(agent, type)` | Get swarm timeline |
| `swarm.consume(group, consumer_id)` | Consume from group |
| `swarm.ack(group, consumer_id, event_id)` | Acknowledge event |
| `swarm.broadcast(type, content)` | Broadcast to all |
| `swarm.verify()` | Verify integrity |

### Event Object

```python
@dataclass
class Event:
    event_id: str           # Unique identifier
    event_type: str         # think, say, act, etc.
    from_agent: str         # Who created it
    content: Dict[str, Any] # The actual data
    timestamp: float        # Unix timestamp
    branch: str             # Branch name
    parent_hash: str        # Previous event's hash
    hash: str               # This event's hash
    metadata: Dict          # Extra metadata
```

---

## Common Patterns

### Audit Trail

```python
agent = Agent("compliance-bot")

# Every action is recorded with hash chain
agent.act("approve_transaction", 
    transaction_id="TXN-123",
    amount=10000,
    approved_by="system"
)

# Later: prove the action happened and wasn't modified
assert agent.verify()
event = agent.search("TXN-123")[0]
print(f"Hash: {event.hash}")
print(f"Timestamp: {event.timestamp}")
```

### Debugging Agent Behavior

```python
agent = Agent("chatbot")

# Run your agent...
agent.hear("What's the weather?", from_="user")
agent.think("User wants weather info, checking location...")
agent.act("call_weather_api", location="NYC")
agent.say("It's 72°F in New York!")

# Later: why did it do that?
thoughts = agent.timeline(type="think")
for t in thoughts:
    print(f"{t.timestamp}: {t.content['thought']}")
```

### A/B Testing Agent Strategies

```python
agent = Agent("recommendation-engine")

# Record baseline
agent.think("Using collaborative filtering...")
agent.act("recommend", items=["A", "B", "C"])

# Branch for experiment
agent.branch("content-based-test")
agent.think("Trying content-based approach...")
agent.act("recommend", items=["X", "Y", "Z"])

# Compare
main_recs = agent.checkout("main").search("recommend")
test_recs = agent.checkout("content-based-test").search("recommend")
```

### Error Recovery

```python
agent = Agent("task-runner")

try:
    agent.think("Starting task...")
    agent.act("risky_operation")
except Exception as e:
    agent.error(str(e), traceback=traceback.format_exc())
    
    # Branch from before the error
    last_good = agent.timeline()[-2].event_id  # Before risky_operation
    agent.branch("recovery", from_event=last_good)
    agent.think("Trying alternative approach...")
```

---

## Why KRNX?

| Problem | KRNX Solution |
|---------|---------------|
| Agents forget between sessions | Persistent memory |
| "Why did the agent do that?" | Full timeline + search |
| Can't reproduce bugs | Replay any time range |
| No audit trail | Hash chain integrity |
| Multi-agent chaos | Consumer groups |
| "What if it had done X?" | Branching |

---

## Installation

```bash
pip install krnxlite

# Optional: for live demo
pip install anthropic textual
```

## Quick Links

- [GitHub](https://github.com/krnx/krnxlite)
- [Studio Demo](./studio/README.md)
- [Architecture](./ARCHITECTURE.md)

---

*Deterministic substrate for probabilistic AI.*
