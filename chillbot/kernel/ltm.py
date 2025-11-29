"""
KRNX LTM - Long-Term Memory (SQLite) - v0.4.0 (Hash Chain Support)

NEW IN v0.4.0:
- Added 'hash' column to events table for O(1) hash lookups
- get_event_by_hash() - retrieve event by computed hash
- verify_hash_chain() - verify entire hash chain integrity
- replay_to_timestamp() - temporal replay for time-travel queries
- Auto-migration for existing databases (adds hash column if missing)

PERFORMANCE FIXES (from v0.3.x):
1. Use executemany() instead of execute() loop - 10-50x faster
2. Disable WAL autocheckpoint during batch writes
3. Manual checkpoint on close/drain

This eliminates the "10 second pause every 5000 events" issue.
"""

import sqlite3
import zlib
import json
import time
import tarfile
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from chillbot.kernel.models import Event

import logging
logger = logging.getLogger(__name__)


# ==============================================
# EXCEPTIONS
# ==============================================

class LTMStorageError(Exception):
    """Base exception for LTM storage errors"""
    pass


class LTMArchivalError(LTMStorageError):
    """Archival operation failed"""
    pass


class LTMSnapshotError(LTMStorageError):
    """Snapshot operation failed"""
    pass


class LTMIntegrityError(LTMStorageError):
    """Database integrity check failed"""
    pass


# ==============================================
# TRANSACTION CONTEXT MANAGER
# ==============================================

class TransactionContext:
    """Context manager for SQLite transactions with automatic rollback."""
    
    def __init__(self, db: sqlite3.Connection, operation: str):
        self.db = db
        self.operation = operation
    
    def __enter__(self):
        logger.debug(f"[TXN_START] {self.operation}")
        self.db.execute("BEGIN TRANSACTION")
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.db.commit()
            logger.debug(f"[TXN_COMMIT] {self.operation}")
        else:
            self.db.rollback()
            logger.error(f"[TXN_ROLLBACK] {self.operation} failed: {exc_val}")
        return False


class LTM:
    """
    Long-Term Memory using SQLite dual-tier storage.
    
    v0.4.0: Added hash column for hash-chain verification support.
    
    OPTIMIZED for high-throughput batch writes:
    - executemany() for batch inserts (10-50x faster)
    - Disabled autocheckpoint during batch mode
    - Manual checkpoint on drain
    """
    
    def __init__(
        self,
        data_path: str,
        warm_retention_days: int = 30,
        compression_level: int = 6,
        enable_auto_archival: bool = False,
        archival_interval_hours: int = 24,
        enable_auto_snapshots: bool = False,
        snapshot_interval_hours: int = 168,
        high_throughput_mode: bool = True,
    ):
        """
        Initialize LTM with dual-tier storage.
        
        Args:
            data_path: Directory for database files
            warm_retention_days: Days to keep in warm tier (default 30)
            compression_level: zlib compression level 1-9 (default 6)
            enable_auto_archival: Start background archival worker
            archival_interval_hours: Hours between archival runs
            enable_auto_snapshots: Start background snapshot worker
            snapshot_interval_hours: Hours between snapshots
            high_throughput_mode: Enable batch write optimizations
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.warm_retention_days = warm_retention_days
        self.compression_level = compression_level
        self.high_throughput_mode = high_throughput_mode
        
        # Database paths
        self.db_path = self.data_path / "events.db"
        self.archive_db_path = self.data_path / "events_archive.db"
        
        # Connections
        self.db: Optional[sqlite3.Connection] = None
        self.archive_db: Optional[sqlite3.Connection] = None
        
        # Auto-workers
        self._archival_running = False
        self._archival_worker_thread: Optional[threading.Thread] = None
        self._snapshot_running = False
        self._snapshot_worker_thread: Optional[threading.Thread] = None
        
        # Initialize
        self._connect()
        self._init_schemas()
        self._migrate_schema()  # v0.4.0: Add hash column if missing
        
        # Write lock for thread safety
        self._write_lock = threading.Lock()
        
        # Batch write counter (for periodic checkpoint)
        self._batch_write_count = 0
        self._checkpoint_interval = 100  # Checkpoint every N batches
        
        # Start workers if enabled
        if enable_auto_archival:
            self.start_archival_worker(archival_interval_hours)
        
        if enable_auto_snapshots:
            self.start_snapshot_worker(snapshot_interval_hours)
        
        print(f"[OK] LTM v0.4.0 initialized at {data_path}")
        print(f"   Warm tier: {self.db_path}")
        print(f"   Cold tier: {self.archive_db_path}")
        if high_throughput_mode:
            print(f"   High-throughput mode: ENABLED")
    
    # ==============================================
    # CONNECTION & SCHEMA
    # ==============================================
    
    def _connect(self):
        """Connect to both warm and cold databases"""
        # Warm tier (read/write, optimized for concurrent access)
        self.db = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None  # Autocommit mode for explicit transactions
        )
        self.db.row_factory = sqlite3.Row
        
        # PERFORMANCE PRAGMAS
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA cache_size=-64000")  # 64MB cache (negative = KB)
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.execute("PRAGMA mmap_size=268435456")  # 256MB memory-mapped I/O
        self.db.execute("PRAGMA busy_timeout=30000")   # 30 second busy timeout
        
        # HIGH THROUGHPUT: Disable autocheckpoint (we'll checkpoint manually)
        if self.high_throughput_mode:
            self.db.execute("PRAGMA wal_autocheckpoint=0")  # DISABLED
        else:
            self.db.execute("PRAGMA wal_autocheckpoint=1000")  # Default-ish
        
        # Cold tier
        self.archive_db = sqlite3.connect(
            self.archive_db_path,
            check_same_thread=False,
            timeout=30.0
        )
        self.archive_db.row_factory = sqlite3.Row
        self.archive_db.execute("PRAGMA journal_mode=WAL")
        self.archive_db.execute("PRAGMA query_only=OFF")
        self.archive_db.execute("PRAGMA cache_size=-16000")  # 16MB
        self.archive_db.execute("PRAGMA busy_timeout=30000")
    
    def _init_schemas(self):
        """Initialize database schemas"""
        # === WARM TIER SCHEMA (v0.4.0: includes hash column) ===
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                created_at REAL NOT NULL,
                previous_hash TEXT,
                hash TEXT,
                channel TEXT,
                ttl_seconds INTEGER,
                retention_class TEXT,
                metadata TEXT,
                CHECK (timestamp > 0)
            );
            
            CREATE INDEX IF NOT EXISTS idx_events_workspace_time 
                ON events(workspace_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_events_user_time 
                ON events(user_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
                ON events(timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_events_channel
                ON events(channel);
            
            CREATE INDEX IF NOT EXISTS idx_events_hash
                ON events(hash);
            
            CREATE INDEX IF NOT EXISTS idx_events_previous_hash
                ON events(previous_hash);
            
            CREATE INDEX IF NOT EXISTS idx_events_workspace_user_time
                ON events(workspace_id, user_id, timestamp DESC);
            
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                state_compressed BLOB NOT NULL,
                event_count INTEGER NOT NULL,
                storage_bytes INTEGER,
                created_at REAL NOT NULL,
                CHECK (timestamp > 0)
            );
            
            CREATE INDEX IF NOT EXISTS idx_snapshots_workspace_time 
                ON snapshots(workspace_id, timestamp DESC);
        """)
        
        # === COLD TIER SCHEMA (v0.4.0: includes hash column) ===
        self.archive_db.executescript("""
            CREATE TABLE IF NOT EXISTS events_archive (
                event_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content_compressed BLOB NOT NULL,
                metadata_compressed BLOB,
                timestamp REAL NOT NULL,
                previous_hash TEXT,
                hash TEXT,
                archived_at REAL NOT NULL,
                compression_ratio REAL,
                CHECK (timestamp > 0)
            );
            
            CREATE INDEX IF NOT EXISTS idx_archive_workspace_time 
                ON events_archive(workspace_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_archive_user_time 
                ON events_archive(user_id, timestamp DESC);
            
            CREATE INDEX IF NOT EXISTS idx_archive_timestamp 
                ON events_archive(timestamp);
            
            CREATE INDEX IF NOT EXISTS idx_archive_hash
                ON events_archive(hash);
            
            CREATE INDEX IF NOT EXISTS idx_archive_previous_hash
                ON events_archive(previous_hash);
        """)
        
        self.db.commit()
        self.archive_db.commit()
    
    def _migrate_schema(self):
        """
        v0.4.0: Migrate existing databases to add hash column.
        
        Safe to run multiple times - only adds column if missing.
        """
        try:
            # Check if hash column exists in warm tier
            cursor = self.db.execute("PRAGMA table_info(events)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'hash' not in columns:
                logger.info("[MIGRATE] Adding 'hash' column to events table...")
                with self._write_lock:
                    self.db.execute("ALTER TABLE events ADD COLUMN hash TEXT")
                    self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_hash ON events(hash)")
                    self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_previous_hash ON events(previous_hash)")
                    self.db.execute("CREATE INDEX IF NOT EXISTS idx_events_workspace_user_time ON events(workspace_id, user_id, timestamp DESC)")
                    self.db.commit()
                logger.info("[MIGRATE] Warm tier migration complete")
            
            # Check cold tier
            cursor = self.archive_db.execute("PRAGMA table_info(events_archive)")
            archive_columns = [row[1] for row in cursor.fetchall()]
            
            if 'hash' not in archive_columns:
                logger.info("[MIGRATE] Adding 'hash' column to events_archive table...")
                self.archive_db.execute("ALTER TABLE events_archive ADD COLUMN hash TEXT")
                self.archive_db.execute("CREATE INDEX IF NOT EXISTS idx_archive_hash ON events_archive(hash)")
                self.archive_db.execute("CREATE INDEX IF NOT EXISTS idx_archive_previous_hash ON events_archive(previous_hash)")
                self.archive_db.commit()
                logger.info("[MIGRATE] Cold tier migration complete")
                
        except Exception as e:
            logger.warning(f"[MIGRATE] Schema migration check: {e}")
    
    # ==============================================
    # EVENT STORAGE - OPTIMIZED (v0.4.0: stores hash)
    # ==============================================
    
    def store_event(self, event: Event) -> bool:
        """Store single event in warm tier."""
        try:
            # Compute hash for storage
            event_hash = event.compute_hash()
            
            with self._write_lock:
                self.db.execute("BEGIN TRANSACTION")
                self.db.execute("""
                    INSERT OR REPLACE INTO events (
                        event_id, workspace_id, user_id, session_id,
                        content, timestamp, created_at,
                        previous_hash, hash, channel, ttl_seconds, retention_class,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.workspace_id,
                    event.user_id,
                    event.session_id,
                    json.dumps(event.content),
                    event.timestamp,
                    event.created_at,
                    event.previous_hash,
                    event_hash,
                    event.channel,
                    event.ttl_seconds,
                    event.retention_class,
                    json.dumps(event.metadata) if event.metadata else None
                ))
                self.db.commit()
            return True
        
        except sqlite3.Error as e:
            logger.error(f"[FAIL] LTM store_event failed: {e}")
            try:
                self.db.rollback()
            except:
                pass
            raise LTMStorageError(f"store_event failed: {e}") from e
    
    def store_events_batch(self, events: List[Event]) -> int:
        """
        Store multiple events using executemany() - MUCH FASTER.
        
        v0.4.0: Now computes and stores hash for each event.
        """
        if not events:
            return 0
        
        try:
            # Prepare data tuples (with hash computation)
            data = [
                (
                    event.event_id,
                    event.workspace_id,
                    event.user_id,
                    event.session_id,
                    json.dumps(event.content),
                    event.timestamp,
                    event.created_at,
                    event.previous_hash,
                    event.compute_hash(),  # v0.4.0: Compute hash
                    event.channel,
                    event.ttl_seconds,
                    event.retention_class,
                    json.dumps(event.metadata) if event.metadata else None
                )
                for event in events
            ]
            
            with self._write_lock:
                self.db.execute("BEGIN TRANSACTION")
                
                # EXECUTEMANY - single prepared statement, multiple executions
                self.db.executemany("""
                    INSERT OR REPLACE INTO events (
                        event_id, workspace_id, user_id, session_id,
                        content, timestamp, created_at,
                        previous_hash, hash, channel, ttl_seconds, retention_class,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, data)
                
                self.db.commit()
                
                # Track batches for periodic checkpoint
                self._batch_write_count += 1
                
                # Periodic checkpoint in high-throughput mode
                if self.high_throughput_mode and self._batch_write_count >= self._checkpoint_interval:
                    self._do_checkpoint()
                    self._batch_write_count = 0
            
            logger.debug(f"[BATCH] Stored {len(events)} events via executemany")
            return len(events)
        
        except sqlite3.Error as e:
            logger.error(f"[FAIL] LTM batch store failed: {e}")
            try:
                self.db.rollback()
            except:
                pass
            raise LTMStorageError(f"batch store failed: {e}") from e
    
    def store_events_batch_raw(self, event_jsons: List[str]) -> int:
        """
        Store pre-serialized events using executemany().
        
        v0.4.0: Computes hash from deserialized event data.
        """
        if not event_jsons:
            return 0
        
        try:
            # Parse and prepare data
            data = []
            for event_json in event_jsons:
                event_dict = json.loads(event_json)
                
                content_json = event_dict['content']
                if isinstance(content_json, dict):
                    content_json = json.dumps(content_json)
                
                metadata_json = event_dict.get('metadata')
                if isinstance(metadata_json, dict):
                    metadata_json = json.dumps(metadata_json)
                
                # Reconstruct Event to compute hash
                from chillbot.kernel.models import Event
                temp_event = Event(
                    event_id=event_dict['event_id'],
                    workspace_id=event_dict['workspace_id'],
                    user_id=event_dict['user_id'],
                    session_id=event_dict['session_id'],
                    content=event_dict['content'] if isinstance(event_dict['content'], dict) else json.loads(event_dict['content']),
                    timestamp=event_dict['timestamp'],
                    created_at=event_dict['created_at'],
                    previous_hash=event_dict.get('previous_hash'),
                    channel=event_dict.get('channel'),
                    ttl_seconds=event_dict.get('ttl_seconds'),
                    retention_class=event_dict.get('retention_class'),
                    metadata=event_dict.get('metadata') or {},
                )
                event_hash = temp_event.compute_hash()
                
                data.append((
                    event_dict['event_id'],
                    event_dict['workspace_id'],
                    event_dict['user_id'],
                    event_dict['session_id'],
                    content_json,
                    event_dict['timestamp'],
                    event_dict['created_at'],
                    event_dict.get('previous_hash'),
                    event_hash,
                    event_dict.get('channel'),
                    event_dict.get('ttl_seconds'),
                    event_dict.get('retention_class'),
                    metadata_json
                ))
            
            with self._write_lock:
                self.db.execute("BEGIN TRANSACTION")
                
                self.db.executemany("""
                    INSERT OR REPLACE INTO events (
                        event_id, workspace_id, user_id, session_id,
                        content, timestamp, created_at,
                        previous_hash, hash, channel, ttl_seconds, retention_class,
                        metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, data)
                
                self.db.commit()
                
                self._batch_write_count += 1
                if self.high_throughput_mode and self._batch_write_count >= self._checkpoint_interval:
                    self._do_checkpoint()
                    self._batch_write_count = 0
            
            logger.debug(f"[BATCH_RAW] Stored {len(data)} events via executemany")
            return len(data)
        
        except sqlite3.Error as e:
            logger.error(f"[FAIL] LTM batch store (raw) failed: {e}")
            try:
                self.db.rollback()
            except:
                pass
            raise LTMStorageError(f"batch store (raw) failed: {e}") from e
    
    def _do_checkpoint(self):
        """Perform WAL checkpoint (non-blocking)."""
        try:
            # PASSIVE checkpoint - doesn't block readers/writers
            result = self.db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            logger.debug(f"[CHECKPOINT] PASSIVE: {result}")
        except Exception as e:
            logger.warning(f"[CHECKPOINT] Failed: {e}")
    
    def force_checkpoint(self):
        """Force full WAL checkpoint (may block briefly)."""
        try:
            with self._write_lock:
                result = self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                logger.info(f"[CHECKPOINT] TRUNCATE: {result}")
        except Exception as e:
            logger.error(f"[CHECKPOINT] Force failed: {e}")
    
    # ==============================================
    # QUERY OPERATIONS
    # ==============================================
    
    def query_events(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        channel: Optional[str] = None,
    ) -> List[Event]:
        """Query events from warm tier."""
        conditions = ["workspace_id = ?"]
        params = [workspace_id]
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        
        if channel:
            conditions.append("channel = ?")
            params.append(channel)
        
        where_clause = " AND ".join(conditions)
        params.append(limit)
        
        query = f"""
            SELECT * FROM events
            WHERE {where_clause}
            ORDER BY timestamp ASC
            LIMIT ?
        """
        
        rows = self.db.execute(query, params).fetchall()
        return [self._row_to_event(row) for row in rows]
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get single event by ID."""
        row = self.db.execute(
            "SELECT * FROM events WHERE event_id = ?",
            (event_id,)
        ).fetchone()
        
        if row:
            return self._row_to_event(row)
        return None
    
    def get_event_by_hash(self, hash: str) -> Optional[Event]:
        """
        v0.4.0: Get event by its computed hash.
        
        O(1) lookup via hash index.
        """
        row = self.db.execute(
            "SELECT * FROM events WHERE hash = ?",
            (hash,)
        ).fetchone()
        
        if row:
            return self._row_to_event(row)
        
        # Also check archive
        archive_row = self.archive_db.execute(
            "SELECT * FROM events_archive WHERE hash = ?",
            (hash,)
        ).fetchone()
        
        if archive_row:
            return self._archive_row_to_event(archive_row)
        
        return None
    
    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert database row to Event object."""
        content = row['content']
        if isinstance(content, str):
            content = json.loads(content)
        
        metadata = row['metadata']
        if metadata and isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        return Event(
            event_id=row['event_id'],
            workspace_id=row['workspace_id'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            content=content,
            timestamp=row['timestamp'],
            created_at=row['created_at'],
            previous_hash=row['previous_hash'],
            channel=row['channel'] if 'channel' in row.keys() else None,
            ttl_seconds=row['ttl_seconds'] if 'ttl_seconds' in row.keys() else None,
            retention_class=row['retention_class'] if 'retention_class' in row.keys() else None,
            metadata=metadata or {},
        )
    
    def _archive_row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert archive row to Event object (decompresses content)."""
        content_compressed = row['content_compressed']
        content = json.loads(zlib.decompress(content_compressed).decode('utf-8'))
        
        metadata = {}
        if row['metadata_compressed']:
            metadata = json.loads(zlib.decompress(row['metadata_compressed']).decode('utf-8'))
        
        return Event(
            event_id=row['event_id'],
            workspace_id=row['workspace_id'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            content=content,
            timestamp=row['timestamp'],
            created_at=row['timestamp'],  # archived_at used as proxy
            previous_hash=row['previous_hash'],
            metadata=metadata,
        )
    
    # ==============================================
    # HASH CHAIN VERIFICATION (v0.4.0)
    # ==============================================
    
    def verify_hash_chain(
        self,
        workspace_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        v0.4.0: Verify entire hash chain for a workspace:user stream.
        
        Walks through all events in chronological order and verifies:
        1. Each event's stored hash matches computed hash
        2. Each event's previous_hash matches predecessor's hash
        
        Returns:
            {
                'valid': bool,
                'events_verified': int,
                'gaps': int,
                'corrupted': int,
                'first_event_id': str,
                'last_event_id': str,
                'issues': List[Dict]
            }
        """
        # Get all events in chronological order (oldest first)
        events = self.db.execute("""
            SELECT * FROM events
            WHERE workspace_id = ? AND user_id = ?
            ORDER BY timestamp ASC
        """, (workspace_id, user_id)).fetchall()
        
        if not events:
            return {
                'valid': True,
                'events_verified': 0,
                'gaps': 0,
                'corrupted': 0,
                'first_event_id': None,
                'last_event_id': None,
                'issues': []
            }
        
        issues = []
        gaps = 0
        corrupted = 0
        
        previous_hash = None
        
        for i, row in enumerate(events):
            event = self._row_to_event(row)
            stored_hash = row['hash'] if 'hash' in row.keys() else None
            
            # Check 1: Verify stored hash matches computed hash
            computed_hash = event.compute_hash()
            if stored_hash and stored_hash != computed_hash:
                corrupted += 1
                issues.append({
                    'type': 'hash_mismatch',
                    'event_id': event.event_id,
                    'position': i,
                    'stored_hash': stored_hash,
                    'computed_hash': computed_hash
                })
            
            # Check 2: Verify chain link
            if i == 0:
                # First event should have no previous_hash (genesis)
                if event.previous_hash is not None:
                    issues.append({
                        'type': 'genesis_has_previous',
                        'event_id': event.event_id,
                        'position': i,
                        'previous_hash': event.previous_hash
                    })
            else:
                # Subsequent events should link to predecessor
                if event.previous_hash != previous_hash:
                    gaps += 1
                    issues.append({
                        'type': 'chain_break',
                        'event_id': event.event_id,
                        'position': i,
                        'expected_previous': previous_hash,
                        'actual_previous': event.previous_hash
                    })
            
            # Update for next iteration
            previous_hash = computed_hash
        
        first_event = self._row_to_event(events[0])
        last_event = self._row_to_event(events[-1])
        
        return {
            'valid': len(issues) == 0,
            'events_verified': len(events),
            'gaps': gaps,
            'corrupted': corrupted,
            'first_event_id': first_event.event_id,
            'last_event_id': last_event.event_id,
            'issues': issues
        }
    
    # ==============================================
    # TEMPORAL REPLAY (v0.4.0)
    # ==============================================
    
    def replay_to_timestamp(
        self,
        workspace_id: str,
        user_id: str,
        timestamp: float,
    ) -> List[Event]:
        """
        v0.4.0: Replay all events up to a given timestamp.
        
        Returns events in chronological order (oldest first).
        This is THE DIFFERENTIATOR - RAG cannot do this.
        """
        rows = self.db.execute("""
            SELECT * FROM events
            WHERE workspace_id = ? AND user_id = ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (workspace_id, user_id, timestamp)).fetchall()
        
        return [self._row_to_event(row) for row in rows]
    
    def get_events_in_range(
        self,
        workspace_id: str,
        user_id: str,
        start_time: float,
        end_time: float,
    ) -> List[Event]:
        """
        Get events in a time range, chronological order.
        """
        rows = self.db.execute("""
            SELECT * FROM events
            WHERE workspace_id = ? AND user_id = ? 
                AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
        """, (workspace_id, user_id, start_time, end_time)).fetchall()
        
        return [self._row_to_event(row) for row in rows]
    
    # ==============================================
    # DELETE OPERATIONS (GDPR)
    # ==============================================
    
    def delete_user_events(self, workspace_id: str, user_id: str) -> Dict[str, int]:
        """Delete all events for a user."""
        try:
            with self._write_lock:
                warm_deleted = self.db.execute(
                    "DELETE FROM events WHERE workspace_id = ? AND user_id = ?",
                    (workspace_id, user_id)
                ).rowcount
                self.db.commit()
                
                cold_deleted = self.archive_db.execute(
                    "DELETE FROM events_archive WHERE workspace_id = ? AND user_id = ?",
                    (workspace_id, user_id)
                ).rowcount
                self.archive_db.commit()
            
            return {'warm_deleted': warm_deleted, 'cold_deleted': cold_deleted}
        
        except sqlite3.Error as e:
            logger.error(f"[FAIL] Delete user events failed: {e}")
            raise LTMStorageError(f"delete_user_events failed: {e}") from e
    
    def delete_workspace_events(self, workspace_id: str) -> Dict[str, int]:
        """Delete all events for a workspace."""
        try:
            with self._write_lock:
                warm_deleted = self.db.execute(
                    "DELETE FROM events WHERE workspace_id = ?",
                    (workspace_id,)
                ).rowcount
                self.db.commit()
                
                cold_deleted = self.archive_db.execute(
                    "DELETE FROM events_archive WHERE workspace_id = ?",
                    (workspace_id,)
                ).rowcount
                self.archive_db.commit()
            
            return {'warm_deleted': warm_deleted, 'cold_deleted': cold_deleted}
        
        except sqlite3.Error as e:
            logger.error(f"[FAIL] Delete workspace events failed: {e}")
            raise LTMStorageError(f"delete_workspace_events failed: {e}") from e
    
    # ==============================================
    # STATS & LIFECYCLE
    # ==============================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LTM statistics."""
        try:
            warm_count = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            cold_count = self.archive_db.execute("SELECT COUNT(*) FROM events_archive").fetchone()[0]
            
            warm_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            cold_size = self.archive_db_path.stat().st_size if self.archive_db_path.exists() else 0
            
            return {
                'warm_events': warm_count,
                'cold_events': cold_count,
                'total_events': warm_count + cold_count,
                'warm_size_mb': round(warm_size / (1024 * 1024), 2),
                'cold_size_mb': round(cold_size / (1024 * 1024), 2),
                'total_size_mb': round((warm_size + cold_size) / (1024 * 1024), 2),
                'batch_writes': self._batch_write_count,
            }
        except Exception as e:
            logger.error(f"[FAIL] Get stats error: {e}")
            return {'error': str(e)}
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify database integrity."""
        warm_ok = self.db.execute("PRAGMA integrity_check").fetchone()[0] == 'ok'
        cold_ok = self.archive_db.execute("PRAGMA integrity_check").fetchone()[0] == 'ok'
        
        return {
            'healthy': warm_ok and cold_ok,
            'warm_tier': {'integrity_ok': warm_ok},
            'cold_tier': {'integrity_ok': cold_ok},
        }
    
    def close(self):
        """Close connections with final checkpoint."""
        # Final checkpoint before close
        if self.high_throughput_mode and self.db:
            logger.info("[LTM] Final checkpoint before close...")
            try:
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as e:
                logger.warning(f"[LTM] Final checkpoint failed: {e}")
        
        if self.db:
            self.db.close()
            self.db = None
        
        if self.archive_db:
            self.archive_db.close()
            self.archive_db = None
        
        print("[OK] LTM connections closed")
    
    # ==============================================
    # STUBS FOR COMPATIBILITY
    # ==============================================
    
    def start_archival_worker(self, interval_hours: int = 24):
        """Start archival worker (stub)."""
        pass
    
    def stop_archival_worker(self):
        """Stop archival worker (stub)."""
        pass
    
    def start_snapshot_worker(self, interval_hours: int = 168):
        """Start snapshot worker (stub)."""
        pass
    
    def stop_snapshot_worker(self):
        """Stop snapshot worker (stub)."""
        pass
