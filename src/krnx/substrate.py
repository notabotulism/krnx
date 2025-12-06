"""
krnx - Git for ML Agent State

Temporal memory substrate with:
- Hash-chain integrity
- Timeline branching
- Point-in-time replay
- Multi-agent support

Usage:
    import krnx
    
    s = krnx.init("myagent")
    
    # Record
    s.record("think", {"thought": "checking tier..."})
    s.record("act", {"action": "approve", "amount": 500})
    
    # Query
    events = s.log(limit=10)
    state = s.at(timestamp)
    
    # Branch & replay
    s.branch("fix", from_event=events[0].id)
    s.replay(callback, branch="fix")
    
    # Verify
    assert s.verify()
"""

import os
import json
import time
import hashlib
import sqlite3
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union, Callable
from contextlib import contextmanager


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Event:
    """A single event in the timeline."""
    id: str
    type: str
    agent: str
    content: Dict[str, Any]
    ts: float
    branch: str = "main"
    parent: Optional[str] = None
    hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Event":
        return cls(
            id=row["event_id"],
            type=row["event_type"],
            agent=row["agent"],
            content=json.loads(row["content"]),
            ts=row["ts"],
            branch=row["branch"],
            parent=row["parent_hash"],
            hash=row["hash"],
        )


# =============================================================================
# SCHEMA
# =============================================================================

SCHEMA = """
-- Events: The core append-only log
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    event_type TEXT NOT NULL,
    agent TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL,
    branch TEXT DEFAULT 'main',
    parent_hash TEXT,
    hash TEXT NOT NULL
);

-- Branches metadata
CREATE TABLE IF NOT EXISTS branches (
    name TEXT PRIMARY KEY,
    parent_branch TEXT,
    fork_event_id TEXT,
    fork_ts REAL,
    created_at REAL DEFAULT (strftime('%s', 'now') + 0.0),
    deleted_at REAL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_branch ON events(branch);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent ON events(agent);

-- Checkpoints: Named save points
CREATE TABLE IF NOT EXISTS checkpoints (
    name TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    description TEXT,
    created_at REAL DEFAULT (strftime('%s', 'now') + 0.0),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_branch ON checkpoints(branch);

-- Initialize main branch
INSERT OR IGNORE INTO branches (name) VALUES ('main');
"""


# =============================================================================
# SUBSTRATE
# =============================================================================

class Substrate:
    """
    Git for ML agent state.
    
    Record everything. Branch timelines. Replay with context.
    Verify integrity. Put agents in production with confidence.
    """
    
    def __init__(
        self,
        name: str,
        path: Optional[Union[str, Path]] = None,
        agent: str = "default",
    ):
        """
        Initialize substrate.
        
        Args:
            name: Workspace name (becomes database filename)
            path: Directory for data (default: ./.krnx/)
            agent: Default agent name for records
        """
        # Sanitize workspace name to prevent path traversal
        safe_name = self._sanitize_name(name)
        self.name = safe_name
        self.default_agent = agent
        
        # Storage path
        self.path = Path(path) if path else Path(".krnx")
        self.path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.path / f"{safe_name}.db"
        
        # Verify db_path is within storage path (defense in depth)
        if not str(self.db_path.resolve()).startswith(str(self.path.resolve())):
            raise ValueError(f"Invalid workspace name: {name}")
        
        # Thread safety
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Initialize
        self._init_db()
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize workspace name to prevent path traversal and injection."""
        import re
        # Remove path separators and parent directory references
        safe = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        # Only allow alphanumeric, underscore, hyphen
        safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', safe)
        # Ensure not empty
        if not safe:
            safe = "workspace"
        # Limit length
        return safe[:64]
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return self._local.conn
    
    @contextmanager
    def _transaction(self):
        """Transaction context manager."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.executescript(SCHEMA)
    
    def _compute_hash(self, type: str, agent: str, content: Dict,
                      ts: float, parent: Optional[str]) -> str:
        """Compute event hash."""
        # Normalize timestamp to float for consistent hashing
        ts = float(ts)
        payload = json.dumps({
            "type": type,
            "agent": agent,
            "content": content,
            "ts": ts,
            "parent": parent,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    
    def _get_last_hash(self, branch: str = "main") -> Optional[str]:
        """Get hash of last event in branch."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT hash FROM events WHERE branch = ? ORDER BY id DESC LIMIT 1",
            (branch,)
        ).fetchone()
        return row["hash"] if row else None
    
    def _generate_id(self) -> str:
        """Generate event ID."""
        return f"evt_{hashlib.sha256(f'{time.time()}{id(self)}'.encode()).hexdigest()[:16]}"
    
    # =========================================================================
    # CORE API
    # =========================================================================
    
    def record(
        self,
        type: str,
        content: Dict[str, Any],
        agent: Optional[str] = None,
        ts: Optional[float] = None,
        branch: str = "main",
    ) -> str:
        """
        Record an event to the timeline.
        
        Args:
            type: Event type (think, observe, act, result, or custom)
            content: Event payload (arbitrary dict)
            agent: Agent name (default: instance default)
            ts: Timestamp (default: now)
            branch: Branch name (default: main)
        
        Returns:
            Event ID
        """
        agent = agent or self.default_agent
        ts = float(ts) if ts is not None else time.time()
        event_id = self._generate_id()
        
        with self._lock:
            parent = self._get_last_hash(branch)
            hash = self._compute_hash(type, agent, content, ts, parent)
            
            with self._transaction() as conn:
                conn.execute(
                    """INSERT INTO events 
                       (event_id, event_type, agent, content, ts, branch, parent_hash, hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (event_id, type, agent, json.dumps(content), ts, branch, parent, hash)
                )
        
        return event_id
    
    def log(
        self,
        limit: int = 100,
        branch: str = "main",
        agent: Optional[str] = None,
        type: Optional[str] = None,
        before: Optional[float] = None,
        after: Optional[float] = None,
    ) -> List[Event]:
        """
        Read events from timeline.
        
        Args:
            limit: Maximum events to return
            branch: Branch to query
            agent: Filter by agent
            type: Filter by event type
            before: Events before this timestamp
            after: Events after this timestamp
        
        Returns:
            List of events (newest first)
        """
        conn = self._get_conn()
        
        query = "SELECT * FROM events WHERE branch = ?"
        params: List[Any] = [branch]
        
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        
        if type:
            query += " AND event_type = ?"
            params.append(type)
        
        if before:
            query += " AND ts < ?"
            params.append(before)
        
        if after:
            query += " AND ts > ?"
            params.append(after)
        
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        return [Event.from_row(row) for row in rows]
    
    def show(self, event_id: str) -> Optional[Event]:
        """
        Get a single event by ID.
        
        Args:
            event_id: Event ID
        
        Returns:
            Event or None if not found
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,)
        ).fetchone()
        return Event.from_row(row) if row else None
    
    def at(
        self,
        ref: Union[float, str],
        branch: str = "main",
    ) -> List[Event]:
        """
        Get state at a point in time.
        
        Args:
            ref: Timestamp (float) or event ID (str)
            branch: Branch to query
        
        Returns:
            All events up to and including that point
        """
        conn = self._get_conn()
        
        if isinstance(ref, str):
            # Event ID reference
            rows = conn.execute(
                """SELECT * FROM events 
                   WHERE branch = ? AND id <= (SELECT id FROM events WHERE event_id = ?)
                   ORDER BY id ASC""",
                (branch, ref)
            ).fetchall()
        else:
            # Timestamp reference
            rows = conn.execute(
                """SELECT * FROM events 
                   WHERE branch = ? AND ts <= ?
                   ORDER BY id ASC""",
                (branch, ref)
            ).fetchall()
        
        return [Event.from_row(row) for row in rows]
    
    def search(
        self,
        query: str,
        limit: int = 100,
        branch: str = "main",
    ) -> List[Event]:
        """
        Search events by content.
        
        Args:
            query: Search string (substring match)
            limit: Maximum results
            branch: Branch to search
        
        Returns:
            Matching events (newest first)
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM events 
               WHERE branch = ? AND content LIKE ?
               ORDER BY id DESC LIMIT ?""",
            (branch, f"%{query}%", limit)
        ).fetchall()
        return [Event.from_row(row) for row in rows]
    
    # =========================================================================
    # BRANCHING
    # =========================================================================
    
    def branch(
        self,
        name: str,
        from_event: Optional[str] = None,
    ) -> str:
        """
        Fork a new timeline.
        
        Args:
            name: Branch name
            from_event: Fork from this event (default: latest)
        
        Returns:
            Branch name
        """
        conn = self._get_conn()
        
        # Check branch doesn't exist
        existing = conn.execute(
            "SELECT name FROM branches WHERE name = ? AND deleted_at IS NULL",
            (name,)
        ).fetchone()
        if existing:
            raise ValueError(f"Branch '{name}' already exists")
        
        # Get fork point
        if from_event:
            fork_row = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (from_event,)
            ).fetchone()
            if not fork_row:
                raise ValueError(f"Event '{from_event}' not found")
            fork_ts = fork_row["ts"]
            parent_branch = fork_row["branch"]
        else:
            # Fork from latest on main
            fork_row = conn.execute(
                "SELECT * FROM events WHERE branch = 'main' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            from_event = fork_row["event_id"] if fork_row else None
            fork_ts = fork_row["ts"] if fork_row else time.time()
            parent_branch = "main"
        
        with self._transaction() as conn:
            # Create branch record
            conn.execute(
                """INSERT INTO branches (name, parent_branch, fork_event_id, fork_ts)
                   VALUES (?, ?, ?, ?)""",
                (name, parent_branch, from_event, fork_ts)
            )
            
            # Copy events up to fork point
            if from_event:
                conn.execute(
                    """INSERT INTO events (event_id, event_type, agent, content, ts, branch, parent_hash, hash)
                       SELECT 
                           'evt_' || hex(randomblob(8)) as event_id,
                           event_type, agent, content, ts, ? as branch, parent_hash, hash
                       FROM events 
                       WHERE branch = ? AND id <= (SELECT id FROM events WHERE event_id = ?)
                       ORDER BY id ASC""",
                    (name, parent_branch, from_event)
                )
        
        return name
    
    def branch_delete(self, name: str) -> None:
        """
        Soft delete a branch.
        
        Args:
            name: Branch name
        """
        if name == "main":
            raise ValueError("Cannot delete main branch")
        
        with self._transaction() as conn:
            conn.execute(
                "UPDATE branches SET deleted_at = ? WHERE name = ?",
                (time.time(), name)
            )
    
    def branches(self, deleted: bool = False) -> List[Dict[str, Any]]:
        """
        List all branches.
        
        Args:
            deleted: Include deleted branches
        
        Returns:
            List of branch info dicts
        """
        conn = self._get_conn()
        
        if deleted:
            rows = conn.execute(
                "SELECT name, parent_branch, fork_event_id, fork_ts, created_at, deleted_at FROM branches"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT name, parent_branch, fork_event_id, fork_ts, created_at, deleted_at FROM branches WHERE deleted_at IS NULL"
            ).fetchall()
        
        return [dict(row) for row in rows]
    
    def diff(
        self,
        branch_a: str,
        branch_b: str,
    ) -> Dict[str, List[Event]]:
        """
        Compare two branches.
        
        Args:
            branch_a: First branch
            branch_b: Second branch
        
        Returns:
            Dict with 'only_a', 'only_b', 'common' event lists
        """
        conn = self._get_conn()
        
        # Get events from both branches
        rows_a = conn.execute(
            "SELECT * FROM events WHERE branch = ? ORDER BY id ASC",
            (branch_a,)
        ).fetchall()
        rows_b = conn.execute(
            "SELECT * FROM events WHERE branch = ? ORDER BY id ASC",
            (branch_b,)
        ).fetchall()
        
        events_a = [Event.from_row(r) for r in rows_a]
        events_b = [Event.from_row(r) for r in rows_b]
        
        # Compare by hash (content-based)
        hashes_a = {e.hash for e in events_a}
        hashes_b = {e.hash for e in events_b}
        
        common_hashes = hashes_a & hashes_b
        only_a_hashes = hashes_a - hashes_b
        only_b_hashes = hashes_b - hashes_a
        
        return {
            "only_a": [e for e in events_a if e.hash in only_a_hashes],
            "only_b": [e for e in events_b if e.hash in only_b_hashes],
            "common": [e for e in events_a if e.hash in common_hashes],
        }
    
    # =========================================================================
    # CHECKPOINTS
    # =========================================================================
    
    def checkpoint(
        self,
        name: str,
        description: Optional[str] = None,
        branch: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        """
        Create a named checkpoint at the current position.
        
        Like `git tag` - a named reference to a specific event.
        Use checkpoints to mark important moments in the timeline.
        
        Args:
            name: Checkpoint name (must be unique)
            description: Optional description
            branch: Branch to checkpoint (default: current/main)
            event_id: Specific event ID (default: latest on branch)
        
        Returns:
            Event ID at checkpoint
        
        Raises:
            ValueError: If checkpoint name exists or event not found
        """
        conn = self._get_conn()
        branch = branch or "main"
        
        # Check if checkpoint name already exists
        existing = conn.execute(
            "SELECT name FROM checkpoints WHERE name = ?",
            (name,)
        ).fetchone()
        if existing:
            raise ValueError(f"Checkpoint '{name}' already exists")
        
        # Get target event
        if event_id:
            row = conn.execute(
                "SELECT event_id, branch FROM events WHERE event_id = ?",
                (event_id,)
            ).fetchone()
            if not row:
                raise ValueError(f"Event '{event_id}' not found")
            target_event_id = row["event_id"]
            target_branch = row["branch"]
        else:
            # Get latest event on branch
            row = conn.execute(
                "SELECT event_id FROM events WHERE branch = ? ORDER BY id DESC LIMIT 1",
                (branch,)
            ).fetchone()
            if not row:
                raise ValueError(f"No events on branch '{branch}'")
            target_event_id = row["event_id"]
            target_branch = branch
        
        # Create checkpoint
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO checkpoints (name, event_id, branch, description)
                   VALUES (?, ?, ?, ?)""",
                (name, target_event_id, target_branch, description)
            )
        
        return target_event_id
    
    def checkpoint_delete(self, name: str) -> None:
        """
        Delete a checkpoint.
        
        Args:
            name: Checkpoint name to delete
        
        Raises:
            ValueError: If checkpoint not found
        """
        conn = self._get_conn()
        
        existing = conn.execute(
            "SELECT name FROM checkpoints WHERE name = ?",
            (name,)
        ).fetchone()
        if not existing:
            raise ValueError(f"Checkpoint '{name}' not found")
        
        with self._transaction() as conn:
            conn.execute("DELETE FROM checkpoints WHERE name = ?", (name,))
    
    def checkpoints(self, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all checkpoints.
        
        Args:
            branch: Filter by branch (optional)
        
        Returns:
            List of checkpoint info dicts
        """
        conn = self._get_conn()
        
        if branch:
            rows = conn.execute(
                """SELECT c.name, c.event_id, c.branch, c.description, c.created_at,
                          e.event_type, e.ts as event_ts
                   FROM checkpoints c
                   LEFT JOIN events e ON c.event_id = e.event_id
                   WHERE c.branch = ?
                   ORDER BY c.created_at DESC""",
                (branch,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT c.name, c.event_id, c.branch, c.description, c.created_at,
                          e.event_type, e.ts as event_ts
                   FROM checkpoints c
                   LEFT JOIN events e ON c.event_id = e.event_id
                   ORDER BY c.created_at DESC"""
            ).fetchall()
        
        return [dict(row) for row in rows]
    
    def get_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific checkpoint.
        
        Args:
            name: Checkpoint name
        
        Returns:
            Checkpoint info dict or None
        """
        conn = self._get_conn()
        row = conn.execute(
            """SELECT c.name, c.event_id, c.branch, c.description, c.created_at,
                      e.event_type, e.ts as event_ts
               FROM checkpoints c
               LEFT JOIN events e ON c.event_id = e.event_id
               WHERE c.name = ?""",
            (name,)
        ).fetchone()
        return dict(row) if row else None
    
    def branch_from_checkpoint(
        self,
        branch_name: str,
        checkpoint_name: str,
    ) -> str:
        """
        Create a branch from a checkpoint.
        
        Convenience method combining get_checkpoint + branch.
        
        Args:
            branch_name: Name for new branch
            checkpoint_name: Checkpoint to branch from
        
        Returns:
            Branch name
        
        Raises:
            ValueError: If checkpoint not found
        """
        cp = self.get_checkpoint(checkpoint_name)
        if not cp:
            raise ValueError(f"Checkpoint '{checkpoint_name}' not found")
        
        return self.branch(branch_name, from_event=cp["event_id"])
    
    # =========================================================================
    # REPLAY
    # =========================================================================
    
    def replay(
        self,
        callback: Callable[["Substrate", List[Event]], None],
        branch: str = "main",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> None:
        """
        Replay events through a callback.
        
        Callback receives (substrate, context) where context is
        all events up to the current point.
        
        Args:
            callback: Function to call with (substrate, context)
            branch: Branch to replay
            start: Start from this event ID
            end: End at this event ID
        """
        events = self.log(limit=10000, branch=branch)
        events.reverse()  # Oldest first
        
        # Filter by start/end
        if start:
            start_idx = next((i for i, e in enumerate(events) if e.id == start), 0)
            events = events[start_idx:]
        if end:
            end_idx = next((i for i, e in enumerate(events) if e.id == end), len(events))
            events = events[:end_idx + 1]
        
        # Replay
        for i, event in enumerate(events):
            context = events[:i + 1]
            callback(self, context)
    
    # =========================================================================
    # INTEGRITY
    # =========================================================================
    
    def verify(self, branch: str = "main") -> bool:
        """
        Verify hash chain integrity.
        
        Args:
            branch: Branch to verify
        
        Returns:
            True if intact, raises on corruption
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM events WHERE branch = ? ORDER BY id ASC",
            (branch,)
        ).fetchall()
        
        if not rows:
            return True
        
        prev_hash = None
        for row in rows:
            event = Event.from_row(row)
            
            # Check parent hash matches
            if event.parent != prev_hash:
                raise IntegrityError(
                    f"Hash chain broken at {event.id}: "
                    f"expected parent {prev_hash}, got {event.parent}"
                )
            
            # Recompute and verify hash
            computed = self._compute_hash(
                event.type, event.agent, event.content, event.ts, event.parent
            )
            if computed != event.hash:
                raise IntegrityError(
                    f"Hash mismatch at {event.id}: "
                    f"expected {computed}, got {event.hash}"
                )
            
            prev_hash = event.hash
        
        return True
    
    # =========================================================================
    # EXPORT / IMPORT
    # =========================================================================
    
    def export(
        self,
        branch: str = "main",
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Export branch to JSONL.
        
        Args:
            branch: Branch to export
            path: Output file path (default: stdout)
        
        Returns:
            JSONL string if no path, else path written
        """
        events = self.log(limit=100000, branch=branch)
        events.reverse()  # Oldest first
        
        lines = [json.dumps(e.to_dict()) for e in events]
        output = "\n".join(lines)
        
        if path:
            Path(path).write_text(output)
            return str(path)
        return output
    
    def import_events(
        self,
        path: Union[str, Path],
        branch: str = "main",
    ) -> int:
        """
        Import events from JSONL.
        
        Args:
            path: Input file path
            branch: Target branch
        
        Returns:
            Number of events imported
        """
        content = Path(path).read_text()
        count = 0
        
        for line in content.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            self.record(
                type=data["type"],
                content=data["content"],
                agent=data.get("agent", "imported"),
                ts=data.get("ts"),
                branch=branch,
            )
            count += 1
        
        return count
    
    # =========================================================================
    # INSTRUMENTATION HELPERS
    # =========================================================================
    
    def trace(self, type: str):
        """
        Decorator to trace function calls.
        
        Usage:
            @s.trace("think")
            def reason(input):
                return llm.call(input)
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                self.record(type, {"fn": fn.__name__, "args": str(args)[:200], "status": "start"})
                try:
                    result = fn(*args, **kwargs)
                    self.record(type, {"fn": fn.__name__, "result": str(result)[:200], "status": "end"})
                    return result
                except Exception as e:
                    self.record(type, {"fn": fn.__name__, "error": str(e), "status": "error"})
                    raise
            return wrapper
        return decorator
    
    @contextmanager
    def span(self, type: str):
        """
        Context manager to trace code blocks.
        
        Usage:
            with s.span("act") as span:
                result = do_something()
                span.content = {"result": result}
        """
        class Span:
            def __init__(self):
                self.content = {}
        
        span = Span()
        self.record(type, {"status": "start"})
        try:
            yield span
            self.record(type, {**span.content, "status": "end"})
        except Exception as e:
            self.record(type, {"error": str(e), "status": "error"})
            raise
    
    # =========================================================================
    # UTILS
    # =========================================================================
    
    def count(self, branch: str = "main") -> int:
        """Count events in branch."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as c FROM events WHERE branch = ?",
            (branch,)
        ).fetchone()
        return row["c"]
    
    def stats(
        self,
        branch: Optional[str] = None,
        since_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics.
        
        Aggregates events by type, agent, branch, and time.
        Includes token/cost estimation if content has tokens_used.
        
        Args:
            branch: Filter by branch (optional, None = all branches)
            since_hours: Only count events from last N hours
        
        Returns:
            Dict with:
            - total_events: int
            - branches: List[str]
            - by_type: Dict[str, int]
            - by_agent: Dict[str, int]
            - by_branch: Dict[str, int]
            - tokens_total: int (if available)
            - time_range: {start: float, end: float}
            - recent_24h: int
            - checkpoints: int
        """
        conn = self._get_conn()
        
        # Build WHERE clause
        conditions = []
        params = []
        
        if branch:
            conditions.append("branch = ?")
            params.append(branch)
        
        if since_hours:
            cutoff = time.time() - (since_hours * 3600)
            conditions.append("ts >= ?")
            params.append(cutoff)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # Total events
        total = conn.execute(
            f"SELECT COUNT(*) as c FROM events {where_clause}",
            params
        ).fetchone()["c"]
        
        # By type
        by_type_rows = conn.execute(
            f"SELECT event_type, COUNT(*) as c FROM events {where_clause} GROUP BY event_type ORDER BY c DESC",
            params
        ).fetchall()
        by_type = {row["event_type"]: row["c"] for row in by_type_rows}
        
        # By agent
        by_agent_rows = conn.execute(
            f"SELECT agent, COUNT(*) as c FROM events {where_clause} GROUP BY agent ORDER BY c DESC",
            params
        ).fetchall()
        by_agent = {row["agent"]: row["c"] for row in by_agent_rows}
        
        # By branch
        by_branch_rows = conn.execute(
            f"SELECT branch, COUNT(*) as c FROM events {where_clause} GROUP BY branch ORDER BY c DESC",
            params
        ).fetchall()
        by_branch = {row["branch"]: row["c"] for row in by_branch_rows}
        
        # Time range
        time_range_row = conn.execute(
            f"SELECT MIN(ts) as start_ts, MAX(ts) as end_ts FROM events {where_clause}",
            params
        ).fetchone()
        
        # Recent 24h
        cutoff_24h = time.time() - 86400
        recent_params = params + [cutoff_24h] if params else [cutoff_24h]
        if where_clause:
            recent_24h = conn.execute(
                f"SELECT COUNT(*) as c FROM events {where_clause} AND ts >= ?",
                recent_params
            ).fetchone()["c"]
        else:
            recent_24h = conn.execute(
                "SELECT COUNT(*) as c FROM events WHERE ts >= ?",
                [cutoff_24h]
            ).fetchone()["c"]
        
        # Token aggregation (from think events with tokens_used)
        tokens_total = 0
        try:
            # Get think events and sum tokens_used from content
            think_rows = conn.execute(
                f"SELECT content FROM events {where_clause} {'AND' if where_clause else 'WHERE'} event_type = 'think'",
                params
            ).fetchall()
            for row in think_rows:
                content = json.loads(row["content"])
                if "tokens_used" in content:
                    tokens_total += content["tokens_used"]
        except:
            pass  # No token data available
        
        # Checkpoints count
        checkpoints_count = conn.execute(
            "SELECT COUNT(*) as c FROM checkpoints"
        ).fetchone()["c"]
        
        # Branches list
        branches_list = [b["name"] for b in self.branches()]
        
        return {
            "total_events": total,
            "branches": branches_list,
            "by_type": by_type,
            "by_agent": by_agent,
            "by_branch": by_branch,
            "tokens_total": tokens_total,
            "time_range": {
                "start": time_range_row["start_ts"],
                "end": time_range_row["end_ts"],
            } if time_range_row["start_ts"] else None,
            "recent_24h": recent_24h,
            "checkpoints": checkpoints_count,
        }
    
    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __repr__(self):
        return f"Substrate(name='{self.name}', path='{self.path}')"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class IntegrityError(Exception):
    """Hash chain integrity violation."""
    pass


# =============================================================================
# MODULE-LEVEL INIT
# =============================================================================

def init(
    name: str,
    path: Optional[Union[str, Path]] = None,
    agent: str = "default",
) -> Substrate:
    """
    Create or open a krnx workspace.
    
    Args:
        name: Workspace name
        path: Storage directory (default: ./.krnx/)
        agent: Default agent name
    
    Returns:
        Substrate instance
    """
    return Substrate(name=name, path=path, agent=agent)
