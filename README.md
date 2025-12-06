# krnx

**Git for ML agent state.**

Record. Branch. Replay. Verify.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-24%2F24-brightgreen.svg)](#benchmarks)

---

<p align="center">
  <img src="https://raw.githubusercontent.com/your-org/krnx/main/assets/demo.gif" alt="krnx demo" width="700">
</p>

---

## The Problem

You built an ML agent. It works. But you can't put it in production because:

- **No audit trail** — What did it actually do?
- **No replay** — Why did it make that decision?
- **No proof** — Can you demonstrate what happened?
- **No SLA** — Can you guarantee reliability?

Your clients need accountability. You have nothing to give them.

## The Solution

```python
import krnx

s = krnx.init("myagent")

s.record("observe", {"input": "refund $8,241"})
s.record("think", {"reasoning": "checking tier..."})
s.record("act", {"action": "approve"})
s.record("result", {"outcome": "LOSS: $8,141"})

# Something went wrong. Find it.
s.branch("fix", from_event="evt_abc123")
s.record("observe", {"tier": "BASIC"}, branch="fix")  # correct data
s.record("act", {"action": "deny"}, branch="fix")
s.record("result", {"outcome": "OK: $0"}, branch="fix")

# Prove both timelines are intact
assert s.verify("main")
assert s.verify("fix")
```

Same agent. Same code. Different context. Different outcome.

**4 lines to instrument. Full time-travel debugging forever.**

## Install

```bash
pip install krnx
```

## Quickstart

### Record

```python
import krnx

s = krnx.init("myagent")

# Record agent decisions
s.record("observe", {"user_input": "refund $8000"})
s.record("think", {"reasoning": "checking customer tier..."})
s.record("act", {"action": "approve", "amount": 8000})
s.record("result", {"outcome": "processed"})
```

### Query

```python
# Recent events
events = s.log(limit=10)

# State at any point
state = s.at("evt_abc123")
state = s.at(1701234567.0)  # timestamp

# Search
matches = s.search("approve")
```

### Branch

```python
# Fork timeline
s.branch("fix", from_event="evt_abc123")

# Work on branch
s.record("observe", {"corrected": True}, branch="fix")

# Compare
diff = s.diff("main", "fix")
```

### Verify

```python
# Check hash chain integrity
assert s.verify()

# Export for audit
s.export(path="audit.jsonl")
```

### Instrument

```python
# Decorator — auto-records function calls
@s.trace("think")
def reason(prompt):
    return llm.call(prompt)

# Context manager — captures duration + result
with s.span("act") as span:
    result = execute_action()
    span.content = {"result": result}
```

## CLI

```bash
krnx init myagent              # Create workspace
krnx record think '{"x": 1}'   # Record event
krnx log                       # View timeline
krnx log --limit 5 --type act  # Filter

krnx branch fix --from evt_abc # Fork
krnx checkout fix              # Switch
krnx diff main..fix            # Compare

krnx checkpoint v1.0 -d "Release"  # Named save point
krnx checkpoints               # List checkpoints
krnx branch-from-checkpoint hotfix v1.0  # Branch from checkpoint

krnx stats                     # Show statistics
krnx stats --json              # Output as JSON

krnx verify                    # Check integrity
krnx export -o audit.jsonl     # Export

krnx studio                    # Launch TUI
krnx demo                      # Run demo
```

## Studio

Visual TUI for exploring timelines.

```bash
krnx studio
```

```
┌─ Timeline ─────────────────────────────────────────┐
│  ● main                                            │
│  │                                                 │
│  ├─○ evt_a1b2  observe  "refund $8,241"           │
│  ├─○ evt_c3d4  think    "checking tier..."        │
│  ├─○ evt_e5f6  act      "approve"                 │
│  └─○ evt_g7h8  result   "LOSS: $8,141"            │
│                                                    │
│  ● fix (from evt_a1b2)                            │
│  │                                                 │
│  ├─○ evt_i9j0  observe  "tier: BASIC"             │
│  ├─○ evt_k1l2  act      "deny"                    │
│  └─○ evt_m3n4  result   "OK: $0"                  │
└────────────────────────────────────────────────────┘
```

**Controls:** `↑↓` navigate, `←→` branches, `Enter` inspect, `b` branch, `d` diff, `v` verify

## Demo

47-second self-running showcase:

```bash
krnx demo
```

Shows an agent making a bad decision due to stale cache data, then branching to fix it.

**Controls:** `SPACE` pause, `R` restart, `Q` quit

## Benchmarks

Tested on WSL2 / Linux / Python 3.10+

| Metric | Result |
|--------|--------|
| Write throughput | **5,500+ events/sec** |
| Read latency (p50) | **< 1ms** |
| Read latency (p99) | **< 4ms** |
| Sustained (10min soak) | **3,800 events/sec, no degradation** |
| Memory growth | **0.1 MB** (no leaks) |
| Disk usage | **~280 bytes/event** |

| Test Suite | Result |
|------------|--------|
| Core | 13/13 |
| Checkpoints | 6/6 |
| Observability | 5/5 |
| **Total** | **24/24** |

Run benchmarks yourself:

```bash
python bench.py
python bench.py --soak-duration 600  # 10-minute soak
```

## How It Works

**Storage:** SQLite with WAL mode. No external dependencies.

**Hash Chain:** Every event includes a SHA-256 hash of its contents + parent hash. Tampering breaks the chain. `verify()` catches it.

**Branching:** Copy-on-fork. Branch from any event to create alternate timelines.

**Replay:** Reconstruct exact state at any point. Debug decisions. Test counterfactuals.

```
main ──●──────●──────●──────●──────●
       │      │      │      │      │
    observe think   act  result  [LOSS]
                     │
                     └─── fix ───●──────●──────●
                                 │      │      │
                              observe  act  result
                                            [OK]
```

## API Reference

### Core

| Method | Description |
|--------|-------------|
| `krnx.init(name, path, agent)` | Create/open workspace |
| `s.record(type, content, ...)` | Write event |
| `s.log(limit, branch, agent, type, before, after)` | Query events |
| `s.show(event_id)` | Get single event |
| `s.at(ref)` | State at point (event ID or timestamp) |
| `s.search(query, limit, branch)` | Search content |

### Branching

| Method | Description |
|--------|-------------|
| `s.branch(name, from_event)` | Fork timeline |
| `s.branches(deleted)` | List branches |
| `s.branch_delete(name)` | Soft delete |
| `s.diff(branch_a, branch_b)` | Compare branches |

### Checkpoints

| Method | Description |
|--------|-------------|
| `s.checkpoint(name, description, event_id)` | Create named save point |
| `s.checkpoints(branch)` | List checkpoints |
| `s.get_checkpoint(name)` | Get checkpoint details |
| `s.checkpoint_delete(name)` | Delete checkpoint |
| `s.branch_from_checkpoint(branch, checkpoint)` | Branch from checkpoint |

### Observability

| Method | Description |
|--------|-------------|
| `s.stats(branch, since_hours)` | Get statistics |
| `s.count(branch)` | Count events |

### Integrity

| Method | Description |
|--------|-------------|
| `s.verify(branch)` | Check hash chain |
| `s.replay(callback, branch, start, end)` | Replay with context |
| `s.export(branch, path)` | Export JSONL |
| `s.import_events(path, branch)` | Import JSONL |

## Security

- **SQL injection** — Parameterized queries, tested with 6 attack payloads
- **Path traversal** — Workspace names sanitized
- **Hash forgery** — SHA-256 chain, tampering detected
- **Replay attacks** — Duplicate events break chain verification
- **Data isolation** — Workspaces and branches fully isolated

See [bench.py](bench.py) for full security test suite.

## Use Cases

**Debugging** — Why did the agent do that? Replay the exact sequence.

**Audit** — Prove what happened. Export verifiable logs.

**A/B Testing** — Branch from decision point, try different approaches.

**Compliance** — Tamper-evident logs for regulated industries.

**Multi-Agent** — One timeline, multiple agents. Full coordination history.

## Roadmap

- [ ] Semantic search (embeddings)
- [ ] Distributed sync
- [ ] Tiered storage (hot/warm/cold)
- [ ] Streaming ingestion
- [ ] Web UI

## Contributing

Issues and PRs welcome.

```bash
git clone https://github.com/your-org/krnx
cd krnx
pip install -e ".[dev]"
pytest
python bench.py
```

## License

MIT

---

<p align="center">
  <b>krnx</b> — Put your agents in production with confidence.
</p>
