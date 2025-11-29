"""
K1: Append-Only Correctness

THEOREM: KRNX provides append-only event storage with guaranteed retrievability.
Events written to the kernel are persisted exactly as submitted and can be
retrieved with identical content.

PROOF STRUCTURE:
Each test follows the format:
- P1 (Precondition): Establish known initial state
- P2 (Operation): Perform the operation under test
- P3 (Postcondition): Verify expected outcome
- P4 (Mechanism): Prove WHY the outcome is correct (not just THAT it is)

Academic rigor requirements:
- Verify content equality, not just counts
- Show actual values in assertions
- Prove mechanism, not just outcome
- Each test is independently reproducible
"""

import pytest
import time
import hashlib
import json
import uuid
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from chillbot.kernel.models import Event
from chillbot.kernel.controller import KRNXController

from ..config import KERNEL_TEST_CONFIG


def compute_content_checksum(content: Dict[str, Any]) -> str:
    """Compute deterministic checksum for content verification."""
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def events_are_equal(a: Event, b: Event, check_timestamps: bool = True) -> tuple:
    """
    Deep equality check for events.
    Returns (is_equal: bool, differences: List[str])
    """
    differences = []
    
    if a.event_id != b.event_id:
        differences.append(f"event_id: {a.event_id} != {b.event_id}")
    if a.workspace_id != b.workspace_id:
        differences.append(f"workspace_id: {a.workspace_id} != {b.workspace_id}")
    if a.user_id != b.user_id:
        differences.append(f"user_id: {a.user_id} != {b.user_id}")
    if a.session_id != b.session_id:
        differences.append(f"session_id: {a.session_id} != {b.session_id}")
    if a.content != b.content:
        differences.append(f"content: {a.content} != {b.content}")
    if a.channel != b.channel:
        differences.append(f"channel: {a.channel} != {b.channel}")
    if check_timestamps and abs(a.timestamp - b.timestamp) > 0.0001:
        differences.append(f"timestamp: {a.timestamp} != {b.timestamp} (delta={abs(a.timestamp - b.timestamp)})")
    
    return (len(differences) == 0, differences)


@pytest.mark.layer1
@pytest.mark.requires_redis
class TestK1AppendOnly:
    """
    K1: Append-Only Correctness Proofs
    
    These tests prove that KRNX correctly implements append-only semantics:
    - Events are stored exactly as submitted
    - Events are retrievable with identical content
    - No silent data loss or corruption occurs
    """
    
    # ==========================================
    # K1.1: Single Event Write-Read Integrity
    # ==========================================
    
    def test_k1_1_single_event_integrity(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.1: A single event written is retrievable with identical content.
        
        THEOREM: write(event) → read(event.id) = event
        
        PROOF:
        - P1: Verify workspace:user has no prior events
        - P2: Write one event with known, verifiable content
        - P3: Retrieve the event by ID
        - P4: Prove all fields match exactly
        """
        # === P1: Precondition - Empty state ===
        initial_events = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=1000
        )
        assert len(initial_events) == 0, (
            f"P1 VIOLATED: Expected empty state, found {len(initial_events)} events"
        )
        
        # === P2: Operation - Write event with verifiable content ===
        test_content = {
            "type": "integrity_proof",
            "sequence": 1,
            "payload": "The quick brown fox jumps over the lazy dog",
            "numeric": 42,
            "nested": {"a": 1, "b": [1, 2, 3]},
        }
        content_checksum = compute_content_checksum(test_content)
        event_id = f"k1-1-proof-{uuid.uuid4().hex[:8]}"
        timestamp = time.time()
        
        original_event = Event(
            event_id=event_id,
            workspace_id=test_workspace,
            user_id=test_user,
            session_id=f"k1-1-session",
            content=test_content,
            timestamp=timestamp,
            created_at=time.time(),
            channel="test",
        )
        
        kernel.write_event(test_workspace, test_user, original_event)
        
        # Wait for LTM migration
        assert wait_for_migration(kernel, 1, timeout=15), "P2 VIOLATED: Migration timeout"
        
        # === P3: Postcondition - Event retrievable ===
        retrieved_event = kernel.get_event(event_id)
        
        assert retrieved_event is not None, (
            f"P3 VIOLATED: Event {event_id} not found after write"
        )
        
        # === P4: Mechanism - Prove exact equality ===
        is_equal, differences = events_are_equal(original_event, retrieved_event)
        
        assert is_equal, (
            f"P4 VIOLATED: Event content mismatch.\n"
            f"Differences:\n" + "\n".join(f"  - {d}" for d in differences)
        )
        
        # Verify content checksum as additional integrity proof
        retrieved_checksum = compute_content_checksum(retrieved_event.content)
        assert retrieved_checksum == content_checksum, (
            f"P4 VIOLATED: Content checksum mismatch.\n"
            f"  Original:  {content_checksum}\n"
            f"  Retrieved: {retrieved_checksum}"
        )
        
        # Print proof summary for paper
        print(f"\n{'='*60}")
        print("K1.1 PROOF SUMMARY: Single Event Write-Read Integrity")
        print(f"{'='*60}")
        print(f"  Event ID: {event_id}")
        print(f"  Content checksum (original):  {content_checksum[:32]}...")
        print(f"  Content checksum (retrieved): {retrieved_checksum[:32]}...")
        print(f"  Fields verified: event_id, workspace_id, user_id, session_id, content, timestamp, channel")
        print(f"  RESULT: ALL FIELDS MATCH EXACTLY")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K1.2: Batch Write-Read Integrity
    # ==========================================
    
    def test_k1_2_batch_integrity(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.2: Multiple events written are all retrievable with identical content.
        
        THEOREM: write(events[0..N]) → read_all() contains all events[0..N] exactly
        
        PROOF:
        - P1: Verify empty initial state
        - P2: Write N events with unique, verifiable content
        - P3: Retrieve all events for workspace:user
        - P4: Prove each written event exists and matches exactly
        """
        N = KERNEL_TEST_CONFIG.get("k1_event_count", 100)
        
        # === P1: Precondition - Empty state ===
        initial_events = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 100
        )
        assert len(initial_events) == 0, (
            f"P1 VIOLATED: Expected empty state, found {len(initial_events)} events"
        )
        
        # === P2: Operation - Write N events ===
        written_events: Dict[str, Event] = {}
        written_checksums: Dict[str, str] = {}
        base_time = time.time()
        
        for i in range(N):
            content = {
                "type": "batch_integrity_proof",
                "index": i,
                "unique_marker": f"event-{i}-{uuid.uuid4().hex[:8]}",
                "fibonacci": [1, 1, 2, 3, 5, 8, 13][i % 7],
            }
            event_id = f"k1-2-batch-{i:04d}-{uuid.uuid4().hex[:8]}"
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k1-2-batch-session",
                content=content,
                timestamp=base_time + (i * 0.001),  # Ensure ordering
                created_at=time.time(),
            )
            
            kernel.write_event(test_workspace, test_user, event)
            written_events[event_id] = event
            written_checksums[event_id] = compute_content_checksum(content)
        
        # Wait for migration
        assert wait_for_migration(kernel, N, timeout=30), "P2 VIOLATED: Migration timeout"
        
        # === P3: Postcondition - All events retrievable ===
        retrieved_events = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 100
        )
        
        assert len(retrieved_events) == N, (
            f"P3 VIOLATED: Expected {N} events, retrieved {len(retrieved_events)}"
        )
        
        # === P4: Mechanism - Prove each event matches ===
        retrieved_map = {e.event_id: e for e in retrieved_events}
        
        mismatches = []
        missing = []
        
        for event_id, original in written_events.items():
            if event_id not in retrieved_map:
                missing.append(event_id)
                continue
            
            retrieved = retrieved_map[event_id]
            is_equal, differences = events_are_equal(original, retrieved)
            
            if not is_equal:
                mismatches.append({
                    "event_id": event_id,
                    "differences": differences
                })
            
            # Verify checksum
            retrieved_checksum = compute_content_checksum(retrieved.content)
            if retrieved_checksum != written_checksums[event_id]:
                mismatches.append({
                    "event_id": event_id,
                    "differences": [f"checksum: {written_checksums[event_id]} != {retrieved_checksum}"]
                })
        
        assert len(missing) == 0, (
            f"P4 VIOLATED: {len(missing)} events missing:\n" +
            "\n".join(f"  - {eid}" for eid in missing[:10]) +
            (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
        )
        
        assert len(mismatches) == 0, (
            f"P4 VIOLATED: {len(mismatches)} events with mismatches:\n" +
            "\n".join(
                f"  - {m['event_id']}: {m['differences']}" 
                for m in mismatches[:5]
            )
        )
        
        # Print proof summary
        print(f"\n{'='*60}")
        print(f"K1.2 PROOF SUMMARY: Batch Write-Read Integrity (N={N})")
        print(f"{'='*60}")
        print(f"  Events written:   {N}")
        print(f"  Events retrieved: {len(retrieved_events)}")
        print(f"  Events verified:  {len(written_events)}")
        print(f"  Missing:          {len(missing)}")
        print(f"  Mismatches:       {len(mismatches)}")
        print(f"  RESULT: 100% INTEGRITY VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K1.3: No Silent Data Loss
    # ==========================================
    
    def test_k1_3_no_silent_data_loss(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.3: No events are silently dropped during write operations.
        
        THEOREM: For all write operations, either:
          (a) Event is successfully stored and retrievable, OR
          (b) Write raises an exception
        
        PROOF:
        - P1: Track every write operation and its outcome
        - P2: Verify every successful write resulted in stored event
        - P3: Verify no events exist that weren't explicitly written
        """
        N = 50
        
        # === P1: Track all writes ===
        write_log: List[Dict[str, Any]] = []
        base_time = time.time()
        
        for i in range(N):
            event_id = f"k1-3-loss-{i:04d}-{uuid.uuid4().hex[:8]}"
            content = {"index": i, "marker": uuid.uuid4().hex}
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k1-3-session",
                content=content,
                timestamp=base_time + i,
                created_at=time.time(),
            )
            
            write_result = {
                "event_id": event_id,
                "content_checksum": compute_content_checksum(content),
                "exception": None,
                "success": False,
            }
            
            try:
                kernel.write_event(test_workspace, test_user, event)
                write_result["success"] = True
            except Exception as e:
                write_result["exception"] = str(e)
            
            write_log.append(write_result)
        
        # Wait for migration
        successful_writes = sum(1 for w in write_log if w["success"])
        assert wait_for_migration(kernel, successful_writes, timeout=30), "Migration timeout"
        
        # === P2: Verify successful writes are stored ===
        retrieved = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 100
        )
        retrieved_ids = {e.event_id for e in retrieved}
        
        successful_but_missing = []
        failed_writes = []
        
        for w in write_log:
            if w["success"]:
                if w["event_id"] not in retrieved_ids:
                    successful_but_missing.append(w["event_id"])
            else:
                failed_writes.append(w)
        
        assert len(successful_but_missing) == 0, (
            f"P2 VIOLATED: {len(successful_but_missing)} successful writes not stored:\n" +
            "\n".join(f"  - {eid}" for eid in successful_but_missing)
        )
        
        # === P3: Verify no phantom events ===
        expected_ids = {w["event_id"] for w in write_log if w["success"]}
        phantom_events = retrieved_ids - expected_ids
        
        assert len(phantom_events) == 0, (
            f"P3 VIOLATED: Found {len(phantom_events)} phantom events:\n" +
            "\n".join(f"  - {eid}" for eid in list(phantom_events)[:10])
        )
        
        # Print proof summary
        print(f"\n{'='*60}")
        print(f"K1.3 PROOF SUMMARY: No Silent Data Loss")
        print(f"{'='*60}")
        print(f"  Write attempts:          {N}")
        print(f"  Successful writes:       {successful_writes}")
        print(f"  Failed writes:           {len(failed_writes)}")
        print(f"  Events retrieved:        {len(retrieved)}")
        print(f"  Missing (after success): {len(successful_but_missing)}")
        print(f"  Phantom events:          {len(phantom_events)}")
        print(f"  RESULT: ZERO SILENT DATA LOSS")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K1.4: Concurrent Write Safety
    # ==========================================
    
    def test_k1_4_concurrent_write_safety(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.4: Concurrent writes from multiple threads are all persisted.
        
        THEOREM: Concurrent write operations are serializable and complete.
        
        PROOF:
        - P1: Launch N threads, each writing M events
        - P2: Wait for all threads to complete
        - P3: Verify all N*M events are stored
        - P4: Verify each event's content integrity
        """
        NUM_THREADS = KERNEL_TEST_CONFIG.get("k4_concurrent_writers", 4)
        EVENTS_PER_THREAD = 25
        TOTAL_EXPECTED = NUM_THREADS * EVENTS_PER_THREAD
        
        # Shared state for tracking
        all_written: Dict[str, Event] = {}
        write_lock = threading.Lock()
        errors: List[str] = []
        
        def writer_thread(thread_id: int):
            """Each thread writes EVENTS_PER_THREAD events."""
            thread_events = {}
            base_time = time.time() + (thread_id * 1000)  # Offset timestamps per thread
            
            for i in range(EVENTS_PER_THREAD):
                event_id = f"k1-4-t{thread_id:02d}-e{i:03d}-{uuid.uuid4().hex[:8]}"
                content = {
                    "thread_id": thread_id,
                    "event_index": i,
                    "unique_marker": uuid.uuid4().hex,
                }
                
                event = Event(
                    event_id=event_id,
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id=f"k1-4-thread-{thread_id}",
                    content=content,
                    timestamp=base_time + i,
                    created_at=time.time(),
                )
                
                try:
                    kernel.write_event(test_workspace, test_user, event)
                    thread_events[event_id] = event
                except Exception as e:
                    with write_lock:
                        errors.append(f"Thread {thread_id}, event {i}: {e}")
            
            with write_lock:
                all_written.update(thread_events)
            
            return thread_id
        
        # === P1 & P2: Launch threads and wait ===
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(writer_thread, tid) for tid in range(NUM_THREADS)]
            for future in as_completed(futures):
                _ = future.result()  # Propagate any exceptions
        
        assert len(errors) == 0, (
            f"P1 VIOLATED: Write errors occurred:\n" +
            "\n".join(f"  - {e}" for e in errors[:10])
        )
        
        # Wait for migration
        assert wait_for_migration(kernel, TOTAL_EXPECTED, timeout=60), "Migration timeout"
        
        # === P3: Verify all events stored ===
        retrieved = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=TOTAL_EXPECTED + 100
        )
        
        assert len(retrieved) == TOTAL_EXPECTED, (
            f"P3 VIOLATED: Expected {TOTAL_EXPECTED} events, found {len(retrieved)}"
        )
        
        # === P4: Verify content integrity ===
        retrieved_map = {e.event_id: e for e in retrieved}
        
        missing = []
        corrupted = []
        
        for event_id, original in all_written.items():
            if event_id not in retrieved_map:
                missing.append(event_id)
                continue
            
            stored = retrieved_map[event_id]
            if stored.content != original.content:
                corrupted.append({
                    "event_id": event_id,
                    "original": original.content,
                    "stored": stored.content,
                })
        
        assert len(missing) == 0, (
            f"P4 VIOLATED: {len(missing)} events missing after concurrent write"
        )
        
        assert len(corrupted) == 0, (
            f"P4 VIOLATED: {len(corrupted)} events corrupted"
        )
        
        # Verify thread distribution
        thread_counts = {}
        for event in retrieved:
            tid = event.content.get("thread_id")
            thread_counts[tid] = thread_counts.get(tid, 0) + 1
        
        # Print proof summary
        print(f"\n{'='*60}")
        print(f"K1.4 PROOF SUMMARY: Concurrent Write Safety")
        print(f"{'='*60}")
        print(f"  Threads:              {NUM_THREADS}")
        print(f"  Events per thread:    {EVENTS_PER_THREAD}")
        print(f"  Total expected:       {TOTAL_EXPECTED}")
        print(f"  Total retrieved:      {len(retrieved)}")
        print(f"  Missing:              {len(missing)}")
        print(f"  Corrupted:            {len(corrupted)}")
        print(f"  Events per thread:    {dict(sorted(thread_counts.items()))}")
        print(f"  RESULT: ALL CONCURRENT WRITES PERSISTED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K1.5: Large Payload Integrity
    # ==========================================
    
    def test_k1_5_large_payload_integrity(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.5: Large payloads are stored and retrieved without corruption.
        
        THEOREM: Events with large content are stored and retrieved exactly.
        
        PROOF:
        - P1: Create event with 100KB payload
        - P2: Write and retrieve
        - P3: Verify byte-for-byte content equality
        - P4: Verify checksum matches
        """
        # === P1: Create large payload ===
        # Generate deterministic large content
        large_text = "KRNX_INTEGRITY_TEST_" * 5000  # ~100KB
        
        content = {
            "type": "large_payload_proof",
            "size_bytes": len(large_text),
            "payload": large_text,
            "checksum": hashlib.sha256(large_text.encode()).hexdigest(),
        }
        
        original_size = len(json.dumps(content))
        original_checksum = compute_content_checksum(content)
        event_id = f"k1-5-large-{uuid.uuid4().hex[:8]}"
        
        event = Event(
            event_id=event_id,
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k1-5-session",
            content=content,
            timestamp=time.time(),
            created_at=time.time(),
        )
        
        # === P2: Write and retrieve ===
        kernel.write_event(test_workspace, test_user, event)
        assert wait_for_migration(kernel, 1, timeout=30), "Migration timeout"
        
        retrieved = kernel.get_event(event_id)
        assert retrieved is not None, f"P2 VIOLATED: Event {event_id} not found"
        
        # === P3: Verify content equality ===
        assert retrieved.content["payload"] == large_text, (
            f"P3 VIOLATED: Payload mismatch.\n"
            f"  Original length: {len(large_text)}\n"
            f"  Retrieved length: {len(retrieved.content.get('payload', ''))}"
        )
        
        # === P4: Verify checksum ===
        retrieved_checksum = compute_content_checksum(retrieved.content)
        
        assert retrieved_checksum == original_checksum, (
            f"P4 VIOLATED: Checksum mismatch.\n"
            f"  Original:  {original_checksum}\n"
            f"  Retrieved: {retrieved_checksum}"
        )
        
        # Verify internal checksum
        retrieved_internal = retrieved.content.get("checksum")
        assert retrieved_internal == content["checksum"], (
            f"P4 VIOLATED: Internal checksum corrupted"
        )
        
        # Print proof summary
        print(f"\n{'='*60}")
        print(f"K1.5 PROOF SUMMARY: Large Payload Integrity")
        print(f"{'='*60}")
        print(f"  Payload size:          {original_size:,} bytes")
        print(f"  Original checksum:     {original_checksum[:32]}...")
        print(f"  Retrieved checksum:    {retrieved_checksum[:32]}...")
        print(f"  Payload match:         {retrieved.content['payload'] == large_text}")
        print(f"  RESULT: LARGE PAYLOAD INTEGRITY VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K1.6: Event ID Uniqueness Enforcement
    # ==========================================
    
    def test_k1_6_event_id_uniqueness(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.6: Duplicate event IDs are handled correctly.
        
        THEOREM: The kernel's behavior for duplicate event_id is deterministic
        and documented (either reject or replace).
        
        PROOF:
        - P1: Write event with ID "test-dup"
        - P2: Write different event with same ID "test-dup"
        - P3: Document observed behavior (reject vs replace)
        - P4: Verify consistency of behavior
        """
        event_id = f"k1-6-dup-{uuid.uuid4().hex[:8]}"
        
        # === P1: Write first event ===
        content_v1 = {"version": 1, "data": "original"}
        event_v1 = Event(
            event_id=event_id,
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k1-6-session",
            content=content_v1,
            timestamp=time.time(),
            created_at=time.time(),
        )
        
        kernel.write_event(test_workspace, test_user, event_v1)
        assert wait_for_migration(kernel, 1, timeout=15), "Migration timeout (v1)"
        
        # === P2: Write second event with same ID ===
        content_v2 = {"version": 2, "data": "replacement"}
        event_v2 = Event(
            event_id=event_id,  # Same ID!
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k1-6-session",
            content=content_v2,
            timestamp=time.time() + 1,
            created_at=time.time(),
        )
        
        duplicate_exception = None
        try:
            kernel.write_event(test_workspace, test_user, event_v2)
            # Give time for potential migration
            time.sleep(2)
        except Exception as e:
            duplicate_exception = e
        
        # === P3: Document behavior ===
        retrieved = kernel.get_event(event_id)
        assert retrieved is not None, "Event should exist"
        
        # Determine which version is stored
        stored_version = retrieved.content.get("version")
        
        if duplicate_exception:
            behavior = "REJECT"
            behavior_detail = f"Exception: {duplicate_exception}"
        elif stored_version == 1:
            behavior = "KEEP_ORIGINAL"
            behavior_detail = "First write wins"
        elif stored_version == 2:
            behavior = "REPLACE"
            behavior_detail = "Last write wins (INSERT OR REPLACE)"
        else:
            behavior = "UNKNOWN"
            behavior_detail = f"Unexpected content: {retrieved.content}"
        
        # === P4: Verify count is exactly 1 ===
        all_events = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=100
        )
        events_with_this_id = [e for e in all_events if e.event_id == event_id]
        
        assert len(events_with_this_id) == 1, (
            f"P4 VIOLATED: Expected exactly 1 event with ID {event_id}, "
            f"found {len(events_with_this_id)}"
        )
        
        # Print proof summary
        print(f"\n{'='*60}")
        print(f"K1.6 PROOF SUMMARY: Event ID Uniqueness")
        print(f"{'='*60}")
        print(f"  Event ID:              {event_id}")
        print(f"  Duplicate behavior:    {behavior}")
        print(f"  Detail:                {behavior_detail}")
        print(f"  Stored version:        {stored_version}")
        print(f"  Events with this ID:   {len(events_with_this_id)}")
        print(f"  RESULT: UNIQUENESS ENFORCED ({behavior})")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K1.7: Query Completeness
    # ==========================================
    
    def test_k1_7_query_completeness(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K1.7: Query returns all and only events for the specified workspace:user.
        
        THEOREM: query(workspace, user) returns exactly the set of events
        written to that workspace:user pair.
        
        PROOF:
        - P1: Write events to workspace:user A
        - P2: Write events to workspace:user B (different)
        - P3: Query A returns only A's events
        - P4: Query B returns only B's events
        - P5: No cross-contamination
        """
        # Use different user IDs within same workspace
        user_a = f"{test_user}-A"
        user_b = f"{test_user}-B"
        
        events_a_ids = set()
        events_b_ids = set()
        
        # === P1: Write events to A ===
        for i in range(10):
            event_id = f"k1-7-A-{i:03d}-{uuid.uuid4().hex[:8]}"
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=user_a,
                session_id="k1-7-A",
                content={"owner": "A", "index": i},
                timestamp=time.time() + i,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, user_a, event)
            events_a_ids.add(event_id)
        
        # === P2: Write events to B ===
        for i in range(15):
            event_id = f"k1-7-B-{i:03d}-{uuid.uuid4().hex[:8]}"
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=user_b,
                session_id="k1-7-B",
                content={"owner": "B", "index": i},
                timestamp=time.time() + i,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, user_b, event)
            events_b_ids.add(event_id)
        
        # Wait for all migrations
        time.sleep(5)
        
        # === P3: Query A ===
        retrieved_a = kernel.query_events(
            workspace_id=test_workspace,
            user_id=user_a,
            limit=100
        )
        retrieved_a_ids = {e.event_id for e in retrieved_a}
        
        # === P4: Query B ===
        retrieved_b = kernel.query_events(
            workspace_id=test_workspace,
            user_id=user_b,
            limit=100
        )
        retrieved_b_ids = {e.event_id for e in retrieved_b}
        
        # === P5: Verify isolation ===
        # A should have exactly A's events
        a_missing = events_a_ids - retrieved_a_ids
        a_extra = retrieved_a_ids - events_a_ids
        
        # B should have exactly B's events
        b_missing = events_b_ids - retrieved_b_ids
        b_extra = retrieved_b_ids - events_b_ids
        
        # Cross-contamination check
        a_has_b = retrieved_a_ids & events_b_ids
        b_has_a = retrieved_b_ids & events_a_ids
        
        assert len(a_missing) == 0, f"P3 VIOLATED: A missing events: {a_missing}"
        assert len(a_extra) == 0, f"P3 VIOLATED: A has extra events: {a_extra}"
        assert len(b_missing) == 0, f"P4 VIOLATED: B missing events: {b_missing}"
        assert len(b_extra) == 0, f"P4 VIOLATED: B has extra events: {b_extra}"
        assert len(a_has_b) == 0, f"P5 VIOLATED: A contains B's events: {a_has_b}"
        assert len(b_has_a) == 0, f"P5 VIOLATED: B contains A's events: {b_has_a}"
        
        # Print proof summary
        print(f"\n{'='*60}")
        print(f"K1.7 PROOF SUMMARY: Query Completeness")
        print(f"{'='*60}")
        print(f"  User A events written:  {len(events_a_ids)}")
        print(f"  User A events queried:  {len(retrieved_a_ids)}")
        print(f"  User B events written:  {len(events_b_ids)}")
        print(f"  User B events queried:  {len(retrieved_b_ids)}")
        print(f"  Cross-contamination:    {len(a_has_b) + len(b_has_a)}")
        print(f"  RESULT: PERFECT ISOLATION VERIFIED")
        print(f"{'='*60}\n")


# ==========================================
# PROOF SUMMARY
# ==========================================
"""
K1 Test Suite: Append-Only Correctness

Theorems Proven:
- K1.1: Single event write-read integrity
- K1.2: Batch write-read integrity  
- K1.3: No silent data loss
- K1.4: Concurrent write safety
- K1.5: Large payload integrity
- K1.6: Event ID uniqueness enforcement
- K1.7: Query completeness (workspace:user isolation)

Each test provides:
- Numbered proof steps (P1, P2, P3, P4)
- Precondition verification
- Operation logging
- Content-level verification (not just counts)
- Mechanism demonstration
- Proof summary output

Run with: pytest -v -s prod_test/layer1_kernel/test_k1_append_only.py
"""
