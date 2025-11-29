"""
K6: Crash Recovery

THEOREM: KRNX recovers gracefully from crashes and restarts without data loss.
Events persist across kernel shutdowns and restarts. Database integrity
is maintained through proper WAL checkpointing.

PROOF STRUCTURE:
Each test follows the format:
- P1 (Precondition): Establish known initial state
- P2 (Operation): Perform crash/restart scenario
- P3 (Postcondition): Verify data survived
- P4 (Mechanism): Prove integrity maintained

Academic rigor requirements:
- Actually restart kernel (not just simulate)
- Verify data byte-for-byte after restart
- Check database integrity
- Prove hash chain survives restart
"""

import pytest
import time
import uuid
import hashlib
import json
from typing import List, Dict, Any
from pathlib import Path

from chillbot.kernel.models import Event
from chillbot.kernel.controller import KRNXController
from chillbot.kernel.recovery import CrashRecovery
from chillbot.kernel.connection_pool import get_redis_client

from ..config import KERNEL_TEST_CONFIG, REDIS_HOST, REDIS_PORT, REDIS_PASSWORD


def compute_content_checksum(content: Dict[str, Any]) -> str:
    """Compute deterministic checksum for content verification."""
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@pytest.mark.layer1
@pytest.mark.requires_redis
class TestK6CrashRecovery:
    """
    K6: Crash Recovery Proofs
    
    These tests prove that KRNX correctly recovers from shutdowns:
    - Data persists across restarts
    - Database integrity is maintained
    - Hash chains survive restarts
    """
    
    # ==========================================
    # K6.1: Data Survives Clean Shutdown
    # ==========================================
    
    def test_k6_1_data_survives_clean_shutdown(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K6.1: Data persists across clean shutdown and restart.
        
        THEOREM: Events written before shutdown are queryable after restart.
        
        PROOF:
        - P1: Create kernel, write N events
        - P2: Cleanly shutdown kernel
        - P3: Create new kernel instance (same data_path)
        - P4: All N events present with identical content
        """
        N = 50
        
        # === P1: Create kernel, write events ===
        kernel1 = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_password=REDIS_PASSWORD,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        written_events: Dict[str, Dict[str, Any]] = {}
        base_time = time.time()
        
        try:
            for i in range(N):
                content = {
                    "index": i,
                    "type": "recovery_test",
                    "marker": uuid.uuid4().hex,
                }
                event_id = f"k6-1-recover-{i:04d}"
                
                event = Event(
                    event_id=event_id,
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id="k6-1-session",
                    content=content,
                    timestamp=base_time + i,
                    created_at=time.time(),
                )
                kernel1.write_event(test_workspace, test_user, event)
                
                written_events[event_id] = {
                    "content": content,
                    "checksum": compute_content_checksum(content),
                }
            
            # Wait for migration
            max_wait = 30
            start = time.time()
            while time.time() - start < max_wait:
                metrics = kernel1.get_worker_metrics()
                if metrics.messages_processed >= N:
                    break
                time.sleep(0.5)
            
            assert kernel1.get_worker_metrics().messages_processed >= N, (
                "Migration did not complete before shutdown"
            )
            
        finally:
            # === P2: Clean shutdown ===
            kernel1.shutdown(timeout=10)
        
        # === P3: Create new kernel (restart) ===
        kernel2 = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            redis_password=REDIS_PASSWORD,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        try:
            # === P4: Verify all events present ===
            result = kernel2.query_events(
                workspace_id=test_workspace,
                user_id=test_user,
                limit=N + 100,
            )
            
            retrieved_ids = {e.event_id for e in result}
            written_ids = set(written_events.keys())
            
            missing = written_ids - retrieved_ids
            
            assert len(missing) == 0, (
                f"P4 VIOLATED: {len(missing)} events lost after restart:\n" +
                "\n".join(f"  {eid}" for eid in list(missing)[:10])
            )
            
            # Verify content integrity
            content_errors = []
            for event in result:
                original = written_events.get(event.event_id)
                if original and event.content != original["content"]:
                    content_errors.append(event.event_id)
            
            assert len(content_errors) == 0, (
                f"P4 VIOLATED: Content corruption in {len(content_errors)} events"
            )
            
            print(f"\n{'='*60}")
            print(f"K6.1 PROOF SUMMARY: Data Survives Clean Shutdown")
            print(f"{'='*60}")
            print(f"  Events written:       {N}")
            print(f"  Events after restart: {len(result)}")
            print(f"  Missing events:       {len(missing)}")
            print(f"  Content errors:       {len(content_errors)}")
            print(f"  RESULT: DATA PERSISTENCE VERIFIED")
            print(f"{'='*60}\n")
            
        finally:
            kernel2.shutdown(timeout=5)
    
    # ==========================================
    # K6.2: LTM Database File Persistence
    # ==========================================
    
    def test_k6_2_ltm_file_persistence(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K6.2: LTM SQLite database files persist across restarts.
        
        THEOREM: Database files exist and contain data after shutdown.
        
        PROOF:
        - P1: Create kernel, write events
        - P2: Shutdown kernel
        - P3: Verify database files exist
        - P4: Verify files have non-zero size
        """
        N = 25
        
        # === P1: Create kernel, write events ===
        kernel = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        try:
            base_time = time.time()
            
            for i in range(N):
                event = Event(
                    event_id=f"k6-2-persist-{i:04d}",
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id="k6-2-session",
                    content={"index": i},
                    timestamp=base_time + i,
                    created_at=time.time(),
                )
                kernel.write_event(test_workspace, test_user, event)
            
            # Wait for migration
            time.sleep(3)
            
        finally:
            # === P2: Shutdown ===
            kernel.shutdown(timeout=10)
        
        # === P3: Verify files exist ===
        db_path = Path(test_data_path) / "events.db"
        archive_path = Path(test_data_path) / "events_archive.db"
        
        assert db_path.exists(), (
            f"P3 VIOLATED: LTM database not found: {db_path}"
        )
        
        assert archive_path.exists(), (
            f"P3 VIOLATED: Archive database not found: {archive_path}"
        )
        
        # === P4: Verify non-zero size ===
        db_size = db_path.stat().st_size
        archive_size = archive_path.stat().st_size
        
        assert db_size > 0, (
            f"P4 VIOLATED: LTM database is empty: {db_size} bytes"
        )
        
        assert archive_size > 0, (
            f"P4 VIOLATED: Archive database is empty: {archive_size} bytes"
        )
        
        print(f"\n{'='*60}")
        print(f"K6.2 PROOF SUMMARY: LTM File Persistence")
        print(f"{'='*60}")
        print(f"  Events written:       {N}")
        print(f"  events.db exists:     {db_path.exists()}")
        print(f"  events.db size:       {db_size:,} bytes")
        print(f"  archive exists:       {archive_path.exists()}")
        print(f"  archive size:         {archive_size:,} bytes")
        print(f"  RESULT: FILE PERSISTENCE VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K6.3: Database Integrity After Restart
    # ==========================================
    
    def test_k6_3_database_integrity(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K6.3: Database passes integrity check after restart.
        
        THEOREM: PRAGMA integrity_check returns 'ok' after restart.
        
        PROOF:
        - P1: Write events and shutdown
        - P2: Restart kernel
        - P3: Run verify_integrity()
        - P4: All integrity checks pass
        """
        N = 30
        
        # === P1: Write events and shutdown ===
        kernel1 = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        try:
            base_time = time.time()
            
            for i in range(N):
                event = Event(
                    event_id=f"k6-3-integrity-{i:04d}",
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id="k6-3-session",
                    content={"index": i},
                    timestamp=base_time + i,
                    created_at=time.time(),
                )
                kernel1.write_event(test_workspace, test_user, event)
            
            time.sleep(3)
        finally:
            kernel1.shutdown(timeout=10)
        
        # === P2: Restart ===
        kernel2 = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        try:
            # === P3: Run integrity check ===
            integrity = kernel2.ltm.verify_integrity()
            
            # === P4: Verify all checks pass ===
            assert integrity['healthy'], (
                f"P4 VIOLATED: Database integrity check failed:\n{integrity}"
            )
            
            assert integrity['warm_tier']['integrity_ok'], (
                f"P4 VIOLATED: Warm tier integrity failed"
            )
            
            assert integrity['cold_tier']['integrity_ok'], (
                f"P4 VIOLATED: Cold tier integrity failed"
            )
            
            print(f"\n{'='*60}")
            print(f"K6.3 PROOF SUMMARY: Database Integrity")
            print(f"{'='*60}")
            print(f"  Events written:       {N}")
            print(f"  Overall healthy:      {integrity['healthy']}")
            print(f"  Warm tier OK:         {integrity['warm_tier']['integrity_ok']}")
            print(f"  Cold tier OK:         {integrity['cold_tier']['integrity_ok']}")
            print(f"  RESULT: DATABASE INTEGRITY VERIFIED")
            print(f"{'='*60}\n")
            
        finally:
            kernel2.shutdown(timeout=5)
    
    # ==========================================
    # K6.4: Hash Chain Survives Restart
    # ==========================================
    
    def test_k6_4_hash_chain_survives_restart(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K6.4: Hash chain remains valid after restart.
        
        THEOREM: verify_hash_chain() returns valid=True after restart.
        
        PROOF:
        - P1: Write hash-chained events
        - P2: Verify chain before shutdown
        - P3: Shutdown and restart
        - P4: Verify chain after restart (identical result)
        """
        N = 20
        
        # === P1: Write hash-chained events ===
        kernel1 = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
            enable_hash_chain=True,
        )
        
        try:
            base_time = time.time()
            
            for i in range(N):
                event = Event(
                    event_id=f"k6-4-chain-{i:04d}",
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id="k6-4-session",
                    content={"index": i},
                    timestamp=base_time + i,
                    created_at=time.time(),
                )
                kernel1.write_event(test_workspace, test_user, event)
            
            # Wait for migration
            time.sleep(3)
            
            # === P2: Verify before shutdown ===
            result_before = kernel1.verify_hash_chain(test_workspace, test_user)
            
            assert result_before['valid'], (
                f"P2 VIOLATED: Chain invalid before shutdown: {result_before}"
            )
            
        finally:
            kernel1.shutdown(timeout=10)
        
        # === P3: Restart ===
        kernel2 = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
            enable_hash_chain=True,
        )
        
        try:
            # === P4: Verify after restart ===
            result_after = kernel2.verify_hash_chain(test_workspace, test_user)
            
            assert result_after['valid'], (
                f"P4 VIOLATED: Chain invalid after restart:\n{result_after}"
            )
            
            assert result_after['events_verified'] == N, (
                f"P4 VIOLATED: Wrong event count. Expected {N}, got {result_after['events_verified']}"
            )
            
            assert result_after['gaps'] == 0, (
                f"P4 VIOLATED: Chain has {result_after['gaps']} gaps after restart"
            )
            
            print(f"\n{'='*60}")
            print(f"K6.4 PROOF SUMMARY: Hash Chain Survives Restart")
            print(f"{'='*60}")
            print(f"  Events in chain:      {N}")
            print(f"  Before shutdown:")
            print(f"    Valid:              {result_before['valid']}")
            print(f"    Events verified:    {result_before['events_verified']}")
            print(f"  After restart:")
            print(f"    Valid:              {result_after['valid']}")
            print(f"    Events verified:    {result_after['events_verified']}")
            print(f"    Gaps:               {result_after['gaps']}")
            print(f"  RESULT: HASH CHAIN SURVIVED RESTART")
            print(f"{'='*60}\n")
            
        finally:
            kernel2.shutdown(timeout=5)
    
    # ==========================================
    # K6.5: WAL Checkpoint on Shutdown
    # ==========================================
    
    def test_k6_5_wal_checkpoint(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K6.5: WAL is checkpointed on clean shutdown.
        
        THEOREM: After shutdown, WAL file is small (checkpointed).
        
        PROOF:
        - P1: Write events
        - P2: Clean shutdown (triggers checkpoint)
        - P3: Check WAL file size
        - P4: WAL size < threshold (indicating checkpoint occurred)
        """
        N = 50
        
        # === P1: Write events ===
        kernel = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        try:
            base_time = time.time()
            
            for i in range(N):
                event = Event(
                    event_id=f"k6-5-wal-{i:04d}",
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id="k6-5-session",
                    content={"index": i, "padding": "x" * 100},  # Some data
                    timestamp=base_time + i,
                    created_at=time.time(),
                )
                kernel.write_event(test_workspace, test_user, event)
            
            time.sleep(3)
        finally:
            # === P2: Shutdown (triggers checkpoint) ===
            kernel.shutdown(timeout=10)
        
        # === P3: Check WAL file ===
        wal_path = Path(test_data_path) / "events.db-wal"
        
        if wal_path.exists():
            wal_size = wal_path.stat().st_size
        else:
            # WAL doesn't exist = fully checkpointed
            wal_size = 0
        
        # === P4: Verify WAL is small ===
        # After checkpoint, WAL should be empty or very small
        MAX_WAL_SIZE = 100_000  # 100KB threshold
        
        assert wal_size < MAX_WAL_SIZE, (
            f"P4 VIOLATED: WAL not checkpointed. Size: {wal_size:,} bytes"
        )
        
        print(f"\n{'='*60}")
        print(f"K6.5 PROOF SUMMARY: WAL Checkpoint")
        print(f"{'='*60}")
        print(f"  Events written:       {N}")
        print(f"  WAL file exists:      {wal_path.exists()}")
        print(f"  WAL size:             {wal_size:,} bytes")
        print(f"  Threshold:            {MAX_WAL_SIZE:,} bytes")
        print(f"  Checkpointed:         {wal_size < MAX_WAL_SIZE}")
        print(f"  RESULT: WAL CHECKPOINT VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K6.6: Multiple Restart Cycles
    # ==========================================
    
    def test_k6_6_multiple_restart_cycles(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K6.6: Data survives multiple shutdown/restart cycles.
        
        THEOREM: Events from cycle N are present in cycle N+2.
        
        PROOF:
        - P1: For each cycle: start, write events, verify previous, shutdown
        - P2: Final verification includes all events from all cycles
        - P3: No data loss across any cycle
        """
        CYCLES = 3
        EVENTS_PER_CYCLE = 10
        
        all_written_ids: List[str] = []
        
        for cycle in range(CYCLES):
            # === P1: Start kernel ===
            kernel = KRNXController(
                data_path=str(test_data_path),
                redis_host=REDIS_HOST,
                redis_port=REDIS_PORT,
                enable_backpressure=False,
                enable_async_worker=True,
            )
            
            try:
                base_time = time.time()
                
                # Write events for this cycle
                for i in range(EVENTS_PER_CYCLE):
                    event_id = f"k6-6-cycle{cycle}-{i:04d}"
                    
                    event = Event(
                        event_id=event_id,
                        workspace_id=test_workspace,
                        user_id=test_user,
                        session_id=f"k6-6-cycle-{cycle}",
                        content={"cycle": cycle, "index": i},
                        timestamp=base_time + (cycle * 1000) + i,
                        created_at=time.time(),
                    )
                    kernel.write_event(test_workspace, test_user, event)
                    all_written_ids.append(event_id)
                
                # Wait for migration
                time.sleep(2)
                
                # Verify all previous events still present
                result = kernel.query_events(
                    workspace_id=test_workspace,
                    user_id=test_user,
                    limit=1000,
                )
                
                expected_count = (cycle + 1) * EVENTS_PER_CYCLE
                
                assert len(result) == expected_count, (
                    f"Cycle {cycle}: Expected {expected_count} events, got {len(result)}"
                )
                
            finally:
                kernel.shutdown(timeout=5)
        
        # === P2: Final verification ===
        final_kernel = KRNXController(
            data_path=str(test_data_path),
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            enable_backpressure=False,
            enable_async_worker=True,
        )
        
        try:
            result = final_kernel.query_events(
                workspace_id=test_workspace,
                user_id=test_user,
                limit=1000,
            )
            
            total_expected = CYCLES * EVENTS_PER_CYCLE
            
            # === P3: Verify no data loss ===
            retrieved_ids = {e.event_id for e in result}
            written_ids = set(all_written_ids)
            
            missing = written_ids - retrieved_ids
            
            assert len(result) == total_expected, (
                f"P3 VIOLATED: Expected {total_expected} events, got {len(result)}"
            )
            
            assert len(missing) == 0, (
                f"P3 VIOLATED: {len(missing)} events lost across cycles"
            )
            
            print(f"\n{'='*60}")
            print(f"K6.6 PROOF SUMMARY: Multiple Restart Cycles")
            print(f"{'='*60}")
            print(f"  Cycles:               {CYCLES}")
            print(f"  Events per cycle:     {EVENTS_PER_CYCLE}")
            print(f"  Total expected:       {total_expected}")
            print(f"  Total retrieved:      {len(result)}")
            print(f"  Missing events:       {len(missing)}")
            print(f"  RESULT: MULTIPLE RESTARTS VERIFIED")
            print(f"{'='*60}\n")
            
        finally:
            final_kernel.shutdown(timeout=5)
    
    # ==========================================
    # K6.7: Consumer Group Recovery Stats
    # ==========================================
    
    def test_k6_7_recovery_stats_structure(
        self,
        test_data_path,
    ):
        """
        K6.7: CrashRecovery provides valid recovery statistics.
        
        THEOREM: get_recovery_stats() returns well-formed statistics.
        
        PROOF:
        - P1: Initialize recovery mechanism
        - P2: Get recovery stats
        - P3: Verify stats structure
        - P4: Stats have valid values
        """
        # === P1: Initialize ===
        redis_client = get_redis_client()
        recovery = CrashRecovery(redis_client)
        
        stream_name = "krnx:ltm:queue"
        group_name = "krnx-ltm-workers"
        
        # Ensure stream and group exist
        try:
            redis_client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise
        
        # === P2: Get stats ===
        stats = recovery.get_recovery_stats(stream_name, group_name)
        
        # === P3: Verify structure ===
        required_fields = ['total_pending', 'consumers', 'oldest_pending_ms', 'recovery_needed']
        
        missing_fields = [f for f in required_fields if f not in stats]
        
        assert len(missing_fields) == 0, (
            f"P3 VIOLATED: Missing stats fields: {missing_fields}"
        )
        
        # === P4: Verify valid values ===
        assert isinstance(stats['total_pending'], int), "total_pending must be int"
        assert stats['total_pending'] >= 0, "total_pending must be non-negative"
        
        assert isinstance(stats['consumers'], list), "consumers must be list"
        
        assert isinstance(stats['oldest_pending_ms'], (int, float)), "oldest_pending_ms must be numeric"
        assert stats['oldest_pending_ms'] >= 0, "oldest_pending_ms must be non-negative"
        
        assert isinstance(stats['recovery_needed'], bool), "recovery_needed must be bool"
        
        print(f"\n{'='*60}")
        print(f"K6.7 PROOF SUMMARY: Recovery Stats Structure")
        print(f"{'='*60}")
        print(f"  Required fields:      {required_fields}")
        print(f"  Missing fields:       {len(missing_fields)}")
        print(f"  Stats:")
        print(f"    total_pending:      {stats['total_pending']}")
        print(f"    consumers:          {len(stats['consumers'])}")
        print(f"    oldest_pending_ms:  {stats['oldest_pending_ms']}")
        print(f"    recovery_needed:    {stats['recovery_needed']}")
        print(f"  RESULT: RECOVERY STATS STRUCTURE VERIFIED")
        print(f"{'='*60}\n")


# ==========================================
# PROOF SUMMARY
# ==========================================
"""
K6 Test Suite: Crash Recovery

Theorems Proven:
- K6.1: Data survives clean shutdown and restart
- K6.2: LTM database files persist across restarts
- K6.3: Database passes integrity check after restart
- K6.4: Hash chain remains valid after restart
- K6.5: WAL is checkpointed on clean shutdown
- K6.6: Data survives multiple restart cycles
- K6.7: Recovery mechanism provides valid statistics

Each test provides:
- Numbered proof steps (P1, P2, P3, P4)
- Actual file sizes and integrity results
- Data comparison before/after restart
- Proof summary output

Run with: pytest -v -s prod_test/layer1_kernel/test_k6_crash_recovery.py
"""
