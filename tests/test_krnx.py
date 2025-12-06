"""Tests for krnx."""

import tempfile
import pytest
import krnx
from krnx import Substrate, Event, IntegrityError


class TestSubstrate:
    """Core substrate tests."""

    def test_record_and_log(self):
        """Basic write and read."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            event_id = s.record("think", {"thought": "hello"})
            assert event_id.startswith("evt_")
            
            events = s.log(limit=10)
            assert len(events) == 1
            assert events[0].content["thought"] == "hello"

    def test_hash_chain(self):
        """Events form a hash chain."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("a", {"n": 1})
            s.record("b", {"n": 2})
            s.record("c", {"n": 3})
            
            events = s.log(limit=10)
            events.reverse()  # Oldest first
            
            # Each event's parent should match previous event's hash
            for i in range(1, len(events)):
                assert events[i].parent == events[i-1].hash

    def test_verify(self):
        """Hash chain verification."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"thought": "one"})
            s.record("act", {"action": "two"})
            s.record("result", {"outcome": "three"})
            
            assert s.verify() is True

    def test_branch(self):
        """Timeline branching."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            # Create main timeline
            e1 = s.record("start", {"n": 1})
            s.record("middle", {"n": 2})
            s.record("end", {"n": 3})
            
            # Branch from first event
            s.branch("alt", from_event=e1)
            
            # Add to branch
            s.record("different", {"n": 99}, branch="alt")
            
            # Verify both branches
            assert s.verify("main") is True
            assert s.verify("alt") is True

    def test_diff(self):
        """Branch comparison."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            e1 = s.record("start", {"n": 1})
            s.record("main_only", {"n": 2})
            
            s.branch("alt", from_event=e1)
            s.record("alt_only", {"n": 99}, branch="alt")
            
            diff = s.diff("main", "alt")
            
            assert len(diff["common"]) == 1
            assert len(diff["only_a"]) == 1
            assert len(diff["only_b"]) == 1

    def test_search(self):
        """Content search."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"thought": "checking inventory"})
            s.record("act", {"action": "approve", "amount": 500})
            s.record("think", {"thought": "user satisfied"})
            
            results = s.search("approve")
            assert len(results) == 1
            assert results[0].content["action"] == "approve"

    def test_show(self):
        """Get single event."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            event_id = s.record("test", {"data": "hello"})
            event = s.show(event_id)
            
            assert event is not None
            assert event.id == event_id
            assert event.content["data"] == "hello"

    def test_at(self):
        """Point-in-time state."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            import time
            s.record("a", {"n": 1})
            time.sleep(0.01)
            t = time.time()
            time.sleep(0.01)
            s.record("b", {"n": 2})
            s.record("c", {"n": 3})
            
            # Get state at time t (should only have first event)
            state = s.at(t)
            assert len(state) == 1
            assert state[0].content["n"] == 1

    def test_branch_delete(self):
        """Soft delete branch."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            e1 = s.record("start", {})
            s.branch("temp", from_event=e1)
            
            branches = s.branches()
            assert len(branches) == 2
            
            s.branch_delete("temp")
            
            branches = s.branches()
            assert len(branches) == 1
            
            # Can still see with deleted=True
            all_branches = s.branches(deleted=True)
            assert len(all_branches) == 2

    def test_export_import(self):
        """Export and import."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("a", {"n": 1})
            s.record("b", {"n": 2})
            
            # Export
            jsonl = s.export()
            assert "n" in jsonl
            
            # Import to new branch
            s.branch("imported")
            import_path = f"{tmp}/export.jsonl"
            with open(import_path, "w") as f:
                f.write(jsonl)
            
            count = s.import_events(import_path, branch="imported")
            assert count == 2


class TestEvent:
    """Event dataclass tests."""

    def test_event_fields(self):
        """Event has expected fields."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp, agent="AGENT")
            s.record("think", {"thought": "test"})
            
            event = s.log()[0]
            
            assert hasattr(event, 'id')
            assert hasattr(event, 'type')
            assert hasattr(event, 'content')
            assert hasattr(event, 'ts')
            assert hasattr(event, 'hash')
            assert hasattr(event, 'parent')
            assert hasattr(event, 'agent')
            assert hasattr(event, 'branch')
            
            assert event.type == "think"
            assert event.agent == "AGENT"
            assert event.branch == "main"


class TestInstrumentation:
    """Instrumentation helper tests."""

    def test_trace_decorator(self):
        """Trace decorator records events."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            @s.trace("compute")
            def add(a, b):
                return a + b
            
            result = add(2, 3)
            assert result == 5
            
            events = s.log(limit=10)
            assert len(events) == 2  # start and end
            
    def test_span_context_manager(self):
        """Span context manager records events."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            with s.span("work") as span:
                span.content = {"result": 42}
            
            events = s.log(limit=10)
            assert len(events) == 2  # start and end
            assert events[0].content.get("result") == 42


class TestCheckpoints:
    """Checkpoint feature tests."""

    def test_create_checkpoint(self):
        """Create and retrieve a checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            # Record some events
            s.record("think", {"n": 1})
            s.record("act", {"n": 2})
            last_id = s.record("result", {"n": 3})
            
            # Create checkpoint
            cp_event = s.checkpoint("v1.0", description="First release")
            
            # Should point to latest event
            assert cp_event == last_id
            
            # Retrieve checkpoint
            cp = s.get_checkpoint("v1.0")
            assert cp is not None
            assert cp["name"] == "v1.0"
            assert cp["description"] == "First release"
            assert cp["branch"] == "main"

    def test_checkpoint_at_specific_event(self):
        """Create checkpoint at specific event."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("a", {"n": 1})
            target = s.record("b", {"n": 2})
            s.record("c", {"n": 3})
            
            cp_event = s.checkpoint("at-b", event_id=target)
            assert cp_event == target

    def test_list_checkpoints(self):
        """List all checkpoints."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"n": 1})
            s.checkpoint("alpha")
            
            s.record("act", {"n": 2})
            s.checkpoint("beta", description="Second checkpoint")
            
            cps = s.checkpoints()
            assert len(cps) == 2
            names = {cp["name"] for cp in cps}
            assert names == {"alpha", "beta"}

    def test_delete_checkpoint(self):
        """Delete a checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"n": 1})
            s.checkpoint("temp")
            
            assert len(s.checkpoints()) == 1
            
            s.checkpoint_delete("temp")
            assert len(s.checkpoints()) == 0

    def test_branch_from_checkpoint(self):
        """Create branch from checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("a", {"n": 1})
            s.record("b", {"n": 2})
            s.checkpoint("stable")
            s.record("c", {"n": 3})  # After checkpoint
            
            s.branch_from_checkpoint("hotfix", "stable")
            
            branches = [b["name"] for b in s.branches()]
            assert "hotfix" in branches
            
            # Hotfix branch should have events up to checkpoint
            # (not including 'c' which was after checkpoint)
            hotfix_events = s.log(limit=100, branch="hotfix")
            assert len(hotfix_events) == 2  # a and b only

    def test_checkpoint_uniqueness(self):
        """Checkpoint names must be unique."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"n": 1})
            s.checkpoint("unique")
            
            with pytest.raises(ValueError, match="already exists"):
                s.checkpoint("unique")


class TestStats:
    """Observability/stats tests."""

    def test_basic_stats(self):
        """Get basic stats."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"n": 1})
            s.record("think", {"n": 2})
            s.record("act", {"n": 3})
            
            data = s.stats()
            
            assert data["total_events"] == 3
            assert data["by_type"]["think"] == 2
            assert data["by_type"]["act"] == 1

    def test_stats_by_branch(self):
        """Stats filtered by branch."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"n": 1})
            s.record("think", {"n": 2})
            
            s.branch("feature")
            s.record("act", {"n": 3}, branch="feature")
            s.record("act", {"n": 4}, branch="feature")
            s.record("act", {"n": 5}, branch="feature")
            
            # Stats for main only
            main_stats = s.stats(branch="main")
            assert main_stats["total_events"] == 2
            
            # Stats for feature only
            feature_stats = s.stats(branch="feature")
            assert feature_stats["total_events"] == 5  # includes copied events

    def test_stats_by_agent(self):
        """Stats show agent breakdown."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp, agent="agent-a")
            
            s.record("think", {"n": 1})
            s.record("think", {"n": 2}, agent="agent-b")
            s.record("act", {"n": 3}, agent="agent-b")
            
            data = s.stats()
            
            assert data["by_agent"]["agent-a"] == 1
            assert data["by_agent"]["agent-b"] == 2

    def test_stats_with_checkpoints(self):
        """Stats include checkpoint count."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            s.record("think", {"n": 1})
            s.checkpoint("v1")
            s.record("act", {"n": 2})
            s.checkpoint("v2")
            
            data = s.stats()
            assert data["checkpoints"] == 2

    def test_stats_tokens(self):
        """Stats aggregate tokens from think events."""
        with tempfile.TemporaryDirectory() as tmp:
            s = krnx.init("test", path=tmp)
            
            # Think events with tokens_used
            s.record("think", {"thought": "a", "tokens_used": 100})
            s.record("think", {"thought": "b", "tokens_used": 200})
            s.record("act", {"action": "c"})  # No tokens
            
            data = s.stats()
            assert data["tokens_total"] == 300
