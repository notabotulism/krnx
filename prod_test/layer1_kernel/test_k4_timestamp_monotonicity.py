"""
K4: Timestamp Ordering

THEOREM: KRNX preserves timestamp-based event ordering with microsecond precision.
Events can be written with any timestamp and are correctly ordered in queries
and replays, regardless of write order or concurrency.

PROOF STRUCTURE:
Each test follows the format:
- P1 (Precondition): Establish known initial state
- P2 (Operation): Perform the operation under test
- P3 (Postcondition): Verify expected outcome
- P4 (Mechanism): Prove WHY the outcome is correct

Academic rigor requirements:
- Verify ordering at event level, not just counts
- Show actual timestamp values
- Test boundary conditions precisely
- Prove concurrency safety
"""

import pytest
import time
import uuid
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from chillbot.kernel.models import Event
from chillbot.kernel.controller import KRNXController

from ..config import KERNEL_TEST_CONFIG


@pytest.mark.layer1
@pytest.mark.requires_redis
class TestK4TimestampOrdering:
    """
    K4: Timestamp Ordering Proofs
    
    These tests prove that KRNX correctly orders events by timestamp:
    - Sequential writes preserve order
    - Concurrent writes are correctly ordered
    - Microsecond precision is maintained
    - Out-of-order writes are sorted correctly
    """
    
    # ==========================================
    # K4.1: Sequential Writes Preserve Order
    # ==========================================
    
    def test_k4_1_sequential_order_preserved(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.1: Sequential writes with increasing timestamps preserve order.
        
        THEOREM: If write(e1) before write(e2) and ts(e1) < ts(e2),
        then query returns [..., e1, ..., e2, ...] in that relative order.
        
        PROOF:
        - P1: Write N events with strictly increasing timestamps
        - P2: Record write order
        - P3: Retrieve events via replay
        - P4: Verify retrieved order matches write order exactly
        """
        N = 50
        base_time = time.time()
        
        # === P1: Write events with strictly increasing timestamps ===
        written_events: List[Dict[str, Any]] = []
        
        for i in range(N):
            event_id = f"k4-1-seq-{i:04d}"
            timestamp = base_time + i  # Strictly increasing
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-1-session",
                content={"index": i, "write_order": i},
                timestamp=timestamp,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
            
            written_events.append({
                "event_id": event_id,
                "write_order": i,
                "timestamp": timestamp,
            })
        
        # === P2: Record expected order ===
        expected_order = [e["event_id"] for e in written_events]
        
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # === P3: Retrieve via replay ===
        result = kernel.replay_to_timestamp(
            workspace_id=test_workspace,
            user_id=test_user,
            timestamp=base_time + N + 100,
        )
        
        retrieved_order = [e.event_id for e in result]
        
        # === P4: Verify exact order match ===
        assert len(retrieved_order) == len(expected_order), (
            f"P4 VIOLATED: Length mismatch. Expected {len(expected_order)}, got {len(retrieved_order)}"
        )
        
        order_mismatches = []
        for i, (expected, actual) in enumerate(zip(expected_order, retrieved_order)):
            if expected != actual:
                order_mismatches.append({
                    "position": i,
                    "expected": expected,
                    "actual": actual,
                })
        
        assert len(order_mismatches) == 0, (
            f"P4 VIOLATED: Order mismatches at {len(order_mismatches)} positions:\n" +
            "\n".join(
                f"  Position {m['position']}: expected {m['expected']}, got {m['actual']}"
                for m in order_mismatches[:10]
            )
        )
        
        # Verify timestamps are strictly increasing in result
        for i in range(len(result) - 1):
            assert result[i].timestamp < result[i + 1].timestamp, (
                f"P4 VIOLATED: Timestamps not strictly increasing at position {i}"
            )
        
        print(f"\n{'='*60}")
        print(f"K4.1 PROOF SUMMARY: Sequential Order Preserved")
        print(f"{'='*60}")
        print(f"  Events written:       {N}")
        print(f"  Events retrieved:     {len(result)}")
        print(f"  Order mismatches:     {len(order_mismatches)}")
        print(f"  First 5 expected:     {expected_order[:5]}")
        print(f"  First 5 retrieved:    {retrieved_order[:5]}")
        print(f"  RESULT: SEQUENTIAL ORDER PRESERVED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K4.2: Same Timestamp Events All Preserved
    # ==========================================
    
    def test_k4_2_same_timestamp_all_preserved(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.2: Multiple events with identical timestamps are all preserved.
        
        THEOREM: For events e1, e2, ..., en with ts(e1) = ts(e2) = ... = ts(en),
        all n events are stored and retrievable.
        
        PROOF:
        - P1: Write N events with exact same timestamp
        - P2: Each event has unique content
        - P3: Retrieve all events
        - P4: Verify all N events present with correct content
        """
        N = 20
        same_timestamp = time.time() + 1000  # Fixed timestamp for all
        
        # === P1: Write events with identical timestamp ===
        written_events: Dict[str, Dict[str, Any]] = {}
        
        for i in range(N):
            event_id = f"k4-2-same-{i:04d}-{uuid.uuid4().hex[:8]}"
            content = {
                "index": i,
                "unique_marker": uuid.uuid4().hex,
                "timestamp_test": "same_timestamp",
            }
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-2-session",
                content=content,
                timestamp=same_timestamp,  # All identical!
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
            
            written_events[event_id] = {
                "content": content,
                "timestamp": same_timestamp,
            }
        
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # === P3: Retrieve all ===
        result = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 100,
        )
        
        # === P4: Verify all present ===
        retrieved_ids = {e.event_id for e in result}
        written_ids = set(written_events.keys())
        
        missing = written_ids - retrieved_ids
        extra = retrieved_ids - written_ids
        
        assert len(missing) == 0, (
            f"P4 VIOLATED: {len(missing)} events missing:\n" +
            "\n".join(f"  {eid}" for eid in list(missing)[:10])
        )
        
        assert len(extra) == 0, (
            f"P4 VIOLATED: {len(extra)} unexpected events"
        )
        
        # Verify all have the exact same timestamp
        timestamps = {e.timestamp for e in result}
        assert len(timestamps) == 1, (
            f"P4 VIOLATED: Expected 1 unique timestamp, found {len(timestamps)}: {timestamps}"
        )
        assert same_timestamp in timestamps, (
            f"P4 VIOLATED: Expected timestamp {same_timestamp}, found {timestamps}"
        )
        
        # Verify content integrity
        content_errors = []
        for event in result:
            original = written_events.get(event.event_id)
            if original and event.content != original["content"]:
                content_errors.append(event.event_id)
        
        assert len(content_errors) == 0, (
            f"P4 VIOLATED: Content mismatch for {len(content_errors)} events"
        )
        
        print(f"\n{'='*60}")
        print(f"K4.2 PROOF SUMMARY: Same Timestamp Events Preserved")
        print(f"{'='*60}")
        print(f"  Events written:       {N}")
        print(f"  Shared timestamp:     {same_timestamp}")
        print(f"  Events retrieved:     {len(result)}")
        print(f"  Missing events:       {len(missing)}")
        print(f"  Content errors:       {len(content_errors)}")
        print(f"  Unique timestamps:    {len(timestamps)}")
        print(f"  RESULT: ALL SAME-TIMESTAMP EVENTS PRESERVED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K4.3: Concurrent Writes Maintain Ordering
    # ==========================================
    
    def test_k4_3_concurrent_write_ordering(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.3: Concurrent writes from multiple threads are ordered by timestamp.
        
        THEOREM: Regardless of thread interleaving during writes,
        replay returns events strictly ordered by timestamp.
        
        PROOF:
        - P1: Launch N threads, each writing M events with distinct timestamp ranges
        - P2: All writes complete
        - P3: Replay returns all N*M events
        - P4: Events are strictly ordered by timestamp (no ordering violations)
        """
        NUM_THREADS = KERNEL_TEST_CONFIG.get("k4_concurrent_writers", 4)
        EVENTS_PER_THREAD = 25
        TOTAL = NUM_THREADS * EVENTS_PER_THREAD
        base_time = time.time()
        
        # Track all written events
        all_written: Dict[str, float] = {}  # event_id -> timestamp
        write_lock = threading.Lock()
        errors: List[str] = []
        
        def writer_thread(thread_id: int):
            """Each thread writes events with timestamps in its own range."""
            thread_events = {}
            
            for i in range(EVENTS_PER_THREAD):
                # Interleave timestamps across threads for maximum chaos
                # Thread 0: 0, 4, 8, 12...
                # Thread 1: 1, 5, 9, 13...
                # etc.
                timestamp = base_time + (i * NUM_THREADS) + thread_id
                event_id = f"k4-3-t{thread_id:02d}-e{i:03d}"
                
                event = Event(
                    event_id=event_id,
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id=f"k4-3-thread-{thread_id}",
                    content={
                        "thread_id": thread_id,
                        "event_index": i,
                        "timestamp_offset": timestamp - base_time,
                    },
                    timestamp=timestamp,
                    created_at=time.time(),
                )
                
                try:
                    kernel.write_event(test_workspace, test_user, event)
                    thread_events[event_id] = timestamp
                except Exception as e:
                    with write_lock:
                        errors.append(f"Thread {thread_id}, event {i}: {e}")
            
            with write_lock:
                all_written.update(thread_events)
            
            return thread_id
        
        # === P1: Launch concurrent writes ===
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = [executor.submit(writer_thread, tid) for tid in range(NUM_THREADS)]
            for future in as_completed(futures):
                _ = future.result()
        
        assert len(errors) == 0, (
            f"P1 VIOLATED: Write errors:\n" + "\n".join(errors[:10])
        )
        
        # === P2: Verify all writes completed ===
        assert len(all_written) == TOTAL, (
            f"P2 VIOLATED: Expected {TOTAL} writes, got {len(all_written)}"
        )
        
        assert wait_for_migration(kernel, TOTAL, timeout=60), "Migration timeout"
        
        # === P3: Replay ===
        result = kernel.replay_to_timestamp(
            workspace_id=test_workspace,
            user_id=test_user,
            timestamp=base_time + TOTAL * NUM_THREADS + 100,
        )
        
        assert len(result) == TOTAL, (
            f"P3 VIOLATED: Expected {TOTAL} events, got {len(result)}"
        )
        
        # === P4: Verify strict timestamp ordering ===
        ordering_violations: List[Dict[str, Any]] = []
        
        for i in range(len(result) - 1):
            current_ts = result[i].timestamp
            next_ts = result[i + 1].timestamp
            
            if current_ts > next_ts:
                ordering_violations.append({
                    "position": i,
                    "current_event": result[i].event_id,
                    "current_ts": current_ts,
                    "next_event": result[i + 1].event_id,
                    "next_ts": next_ts,
                    "difference": current_ts - next_ts,
                })
        
        assert len(ordering_violations) == 0, (
            f"P4 VIOLATED: {len(ordering_violations)} timestamp ordering violations:\n" +
            "\n".join(
                f"  Position {v['position']}: {v['current_ts']:.6f} > {v['next_ts']:.6f}"
                for v in ordering_violations[:10]
            )
        )
        
        # Verify all events present
        result_ids = {e.event_id for e in result}
        missing = set(all_written.keys()) - result_ids
        
        assert len(missing) == 0, (
            f"P4 VIOLATED: {len(missing)} events missing after concurrent write"
        )
        
        # Show thread distribution
        thread_counts: Dict[int, int] = {}
        for event in result:
            tid = event.content.get("thread_id")
            thread_counts[tid] = thread_counts.get(tid, 0) + 1
        
        print(f"\n{'='*60}")
        print(f"K4.3 PROOF SUMMARY: Concurrent Write Ordering")
        print(f"{'='*60}")
        print(f"  Threads:              {NUM_THREADS}")
        print(f"  Events per thread:    {EVENTS_PER_THREAD}")
        print(f"  Total events:         {TOTAL}")
        print(f"  Events retrieved:     {len(result)}")
        print(f"  Ordering violations:  {len(ordering_violations)}")
        print(f"  Missing events:       {len(missing)}")
        print(f"  Events per thread:    {dict(sorted(thread_counts.items()))}")
        print(f"  RESULT: CONCURRENT ORDERING VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K4.4: Microsecond Precision Preserved
    # ==========================================
    
    def test_k4_4_microsecond_precision(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.4: Timestamps with microsecond precision are stored and retrieved exactly.
        
        THEOREM: For timestamp T with 6 decimal places of precision,
        stored timestamp T' satisfies |T - T'| < 1e-6.
        
        PROOF:
        - P1: Create events with precise timestamps (6 decimal places)
        - P2: Store and retrieve
        - P3: Compare original and retrieved timestamps
        - P4: Verify precision preserved within 1 microsecond
        """
        N = 20
        base_time = time.time()
        
        # === P1: Create events with microsecond precision ===
        written_timestamps: Dict[str, float] = {}
        
        for i in range(N):
            # Create timestamp with 6 decimal places of precision
            precise_timestamp = base_time + i + (i * 0.123456)
            event_id = f"k4-4-precision-{i:04d}"
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-4-session",
                content={"index": i, "expected_ts": precise_timestamp},
                timestamp=precise_timestamp,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
            written_timestamps[event_id] = precise_timestamp
        
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # === P2: Retrieve ===
        result = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 10,
        )
        
        # === P3 & P4: Compare precision ===
        precision_errors: List[Dict[str, Any]] = []
        max_deviation = 0.0
        
        for event in result:
            original_ts = written_timestamps.get(event.event_id)
            if original_ts is None:
                continue
            
            deviation = abs(event.timestamp - original_ts)
            max_deviation = max(max_deviation, deviation)
            
            # Microsecond precision = 1e-6
            if deviation >= 1e-6:
                precision_errors.append({
                    "event_id": event.event_id,
                    "original": original_ts,
                    "retrieved": event.timestamp,
                    "deviation": deviation,
                })
        
        assert len(precision_errors) == 0, (
            f"P4 VIOLATED: {len(precision_errors)} precision errors:\n" +
            "\n".join(
                f"  {e['event_id']}: original={e['original']:.10f}, "
                f"retrieved={e['retrieved']:.10f}, deviation={e['deviation']:.10f}"
                for e in precision_errors[:10]
            )
        )
        
        # Verify exact equality (Python float should preserve this)
        exact_matches = 0
        for event in result:
            original_ts = written_timestamps.get(event.event_id)
            if original_ts is not None and event.timestamp == original_ts:
                exact_matches += 1
        
        print(f"\n{'='*60}")
        print(f"K4.4 PROOF SUMMARY: Microsecond Precision")
        print(f"{'='*60}")
        print(f"  Events tested:        {N}")
        print(f"  Precision errors:     {len(precision_errors)}")
        print(f"  Max deviation:        {max_deviation:.15f}")
        print(f"  Exact matches:        {exact_matches}/{N}")
        print(f"  Sample timestamps:")
        for event in result[:3]:
            original = written_timestamps.get(event.event_id, 0)
            print(f"    {event.event_id}: {original:.10f} -> {event.timestamp:.10f}")
        print(f"  RESULT: MICROSECOND PRECISION VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K4.5: Out-of-Order Writes Sorted Correctly
    # ==========================================
    
    def test_k4_5_out_of_order_writes_sorted(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.5: Events written out of timestamp order are sorted correctly on retrieval.
        
        THEOREM: If writes occur in order [e3, e1, e2] with ts(e1) < ts(e2) < ts(e3),
        replay returns [e1, e2, e3].
        
        PROOF:
        - P1: Write events in deliberately scrambled timestamp order
        - P2: Record write order vs timestamp order
        - P3: Retrieve via replay
        - P4: Verify retrieval order matches timestamp order (not write order)
        """
        N = 30
        base_time = time.time()
        
        # === P1: Create scrambled write order ===
        # Generate indices 0..N-1, then scramble them
        indices = list(range(N))
        # Scramble: reverse in chunks
        scrambled_indices = []
        chunk_size = 5
        for i in range(0, N, chunk_size):
            chunk = indices[i:i + chunk_size]
            scrambled_indices.extend(reversed(chunk))
        
        # Write in scrambled order
        write_log: List[Dict[str, Any]] = []
        
        for write_order, idx in enumerate(scrambled_indices):
            timestamp = base_time + idx  # Timestamp based on original index
            event_id = f"k4-5-scramble-{idx:04d}"
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-5-session",
                content={
                    "original_index": idx,
                    "write_order": write_order,
                },
                timestamp=timestamp,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
            
            write_log.append({
                "event_id": event_id,
                "write_order": write_order,
                "timestamp": timestamp,
                "original_index": idx,
            })
        
        # === P2: Record expected order (by timestamp) ===
        expected_order_by_ts = sorted(write_log, key=lambda x: x["timestamp"])
        expected_ids = [e["event_id"] for e in expected_order_by_ts]
        
        # Write order is different from expected order
        write_order_ids = [e["event_id"] for e in write_log]
        assert write_order_ids != expected_ids, (
            "Test setup error: write order should differ from timestamp order"
        )
        
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # === P3: Retrieve via replay ===
        result = kernel.replay_to_timestamp(
            workspace_id=test_workspace,
            user_id=test_user,
            timestamp=base_time + N + 100,
        )
        
        retrieved_ids = [e.event_id for e in result]
        
        # === P4: Verify matches timestamp order, not write order ===
        assert retrieved_ids == expected_ids, (
            f"P4 VIOLATED: Retrieved order doesn't match timestamp order!\n"
            f"  Write order (first 5):     {write_order_ids[:5]}\n"
            f"  Expected (by ts, first 5): {expected_ids[:5]}\n"
            f"  Retrieved (first 5):       {retrieved_ids[:5]}"
        )
        
        assert retrieved_ids != write_order_ids, (
            "P4: Retrieved order should differ from write order (proves sorting)"
        )
        
        print(f"\n{'='*60}")
        print(f"K4.5 PROOF SUMMARY: Out-of-Order Writes Sorted")
        print(f"{'='*60}")
        print(f"  Events written:       {N}")
        print(f"  Write order (first 5):    {write_order_ids[:5]}")
        print(f"  Expected order (first 5): {expected_ids[:5]}")
        print(f"  Retrieved order (first 5):{retrieved_ids[:5]}")
        print(f"  Write order == Retrieved: {write_order_ids == retrieved_ids}")
        print(f"  Expected == Retrieved:    {expected_ids == retrieved_ids}")
        print(f"  RESULT: OUT-OF-ORDER WRITES CORRECTLY SORTED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K4.6: Timestamp vs Created_At Distinction
    # ==========================================
    
    def test_k4_6_timestamp_vs_created_at(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.6: Event timestamp and created_at are independent.
        
        THEOREM: Events can have timestamp different from created_at.
        Ordering uses timestamp, not created_at.
        
        PROOF:
        - P1: Write events with backdated timestamps (timestamp < created_at)
        - P2: Write events with future timestamps (timestamp > created_at)
        - P3: Verify both timestamp and created_at are stored correctly
        - P4: Verify ordering uses timestamp, not created_at
        """
        now = time.time()
        
        # === P1: Backdated events ===
        backdated_events: List[Dict[str, Any]] = []
        
        for i in range(5):
            timestamp = now - 86400 - i  # 1 day ago
            created_at = now  # Created now
            event_id = f"k4-6-backdated-{i:04d}"
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-6-session",
                content={"type": "backdated", "index": i},
                timestamp=timestamp,
                created_at=created_at,
            )
            kernel.write_event(test_workspace, test_user, event)
            
            backdated_events.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "created_at": created_at,
            })
        
        # === P2: Future-dated events ===
        future_events: List[Dict[str, Any]] = []
        
        for i in range(5):
            timestamp = now + 86400 + i  # 1 day in future
            created_at = now  # Created now
            event_id = f"k4-6-future-{i:04d}"
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-6-session",
                content={"type": "future", "index": i},
                timestamp=timestamp,
                created_at=created_at,
            )
            kernel.write_event(test_workspace, test_user, event)
            
            future_events.append({
                "event_id": event_id,
                "timestamp": timestamp,
                "created_at": created_at,
            })
        
        assert wait_for_migration(kernel, 10, timeout=30), "Migration timeout"
        
        # === P3: Verify storage ===
        result = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=20,
        )
        
        storage_errors = []
        
        all_expected = {e["event_id"]: e for e in backdated_events + future_events}
        
        for event in result:
            expected = all_expected.get(event.event_id)
            if expected is None:
                continue
            
            # Check timestamp stored correctly
            if abs(event.timestamp - expected["timestamp"]) > 0.001:
                storage_errors.append(
                    f"{event.event_id}: timestamp {event.timestamp} != {expected['timestamp']}"
                )
            
            # Check created_at stored correctly
            if abs(event.created_at - expected["created_at"]) > 0.001:
                storage_errors.append(
                    f"{event.event_id}: created_at {event.created_at} != {expected['created_at']}"
                )
        
        assert len(storage_errors) == 0, (
            f"P3 VIOLATED: Storage errors:\n" + "\n".join(storage_errors)
        )
        
        # === P4: Verify ordering by timestamp ===
        # Replay to "now" should only include backdated events (timestamp < now)
        replay_now = kernel.replay_to_timestamp(
            workspace_id=test_workspace,
            user_id=test_user,
            timestamp=now,
        )
        
        replay_ids = {e.event_id for e in replay_now}
        backdated_ids = {e["event_id"] for e in backdated_events}
        future_ids = {e["event_id"] for e in future_events}
        
        # All backdated should be included
        missing_backdated = backdated_ids - replay_ids
        assert len(missing_backdated) == 0, (
            f"P4 VIOLATED: Backdated events missing from replay: {missing_backdated}"
        )
        
        # No future should be included
        included_future = future_ids & replay_ids
        assert len(included_future) == 0, (
            f"P4 VIOLATED: Future events incorrectly included: {included_future}"
        )
        
        print(f"\n{'='*60}")
        print(f"K4.6 PROOF SUMMARY: Timestamp vs Created_At")
        print(f"{'='*60}")
        print(f"  Backdated events:     {len(backdated_events)}")
        print(f"  Future events:        {len(future_events)}")
        print(f"  Storage errors:       {len(storage_errors)}")
        print(f"  Replay to 'now':")
        print(f"    Backdated included: {len(backdated_ids - missing_backdated)}/{len(backdated_ids)}")
        print(f"    Future excluded:    {len(future_ids) - len(included_future)}/{len(future_ids)}")
        print(f"  RESULT: TIMESTAMP VS CREATED_AT DISTINCTION VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K4.7: Replay Respects Time Boundaries
    # ==========================================
    
    def test_k4_7_replay_time_boundaries(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K4.7: replay_to_timestamp(T) respects T as strict upper bound.
        
        THEOREM: For all events e in replay_to_timestamp(T), timestamp(e) ≤ T.
        No event with timestamp > T is included.
        
        PROOF:
        - P1: Create events spanning timestamp range [T0, T0+100]
        - P2: Replay to various timestamps T
        - P3: For each replay, verify all events have timestamp ≤ T
        - P4: Verify events with timestamp > T are excluded
        """
        N = 100
        base_time = time.time()
        
        # === P1: Create events ===
        for i in range(N):
            event = Event(
                event_id=f"k4-7-boundary-{i:04d}",
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k4-7-session",
                content={"index": i},
                timestamp=base_time + i,  # 0 to 99 seconds offset
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
        
        assert wait_for_migration(kernel, N, timeout=60), "Migration timeout"
        
        # === P2 & P3 & P4: Test various boundary points ===
        test_points = [
            (base_time + 0, 1),      # Just first event
            (base_time + 10, 11),    # First 11 events
            (base_time + 50.5, 51),  # First 51 events
            (base_time + 99, 100),   # All events
        ]
        
        boundary_violations = []
        count_errors = []
        
        for replay_ts, expected_count in test_points:
            result = kernel.replay_to_timestamp(
                workspace_id=test_workspace,
                user_id=test_user,
                timestamp=replay_ts,
            )
            
            # Check count
            if len(result) != expected_count:
                count_errors.append({
                    "replay_ts": replay_ts,
                    "expected": expected_count,
                    "actual": len(result),
                })
            
            # Check all events within boundary
            for event in result:
                if event.timestamp > replay_ts:
                    boundary_violations.append({
                        "replay_ts": replay_ts,
                        "event_id": event.event_id,
                        "event_ts": event.timestamp,
                    })
        
        assert len(boundary_violations) == 0, (
            f"P3 VIOLATED: Events outside boundary:\n" +
            "\n".join(
                f"  Replay to {v['replay_ts']:.2f}: {v['event_id']} has ts={v['event_ts']:.2f}"
                for v in boundary_violations
            )
        )
        
        assert len(count_errors) == 0, (
            f"P4 VIOLATED: Count mismatches:\n" +
            "\n".join(
                f"  Replay to {e['replay_ts']:.2f}: expected {e['expected']}, got {e['actual']}"
                for e in count_errors
            )
        )
        
        print(f"\n{'='*60}")
        print(f"K4.7 PROOF SUMMARY: Replay Time Boundaries")
        print(f"{'='*60}")
        print(f"  Total events:          {N}")
        print(f"  Test points:           {len(test_points)}")
        print(f"  Boundary violations:   {len(boundary_violations)}")
        print(f"  Count errors:          {len(count_errors)}")
        print(f"  Test results:")
        for replay_ts, expected_count in test_points:
            print(f"    Replay to +{replay_ts - base_time:.1f}s: expected {expected_count} events ✓")
        print(f"  RESULT: TIME BOUNDARIES VERIFIED")
        print(f"{'='*60}\n")


# ==========================================
# PROOF SUMMARY
# ==========================================
"""
K4 Test Suite: Timestamp Ordering

Theorems Proven:
- K4.1: Sequential writes preserve timestamp order
- K4.2: Same-timestamp events are all preserved
- K4.3: Concurrent writes maintain timestamp ordering
- K4.4: Microsecond precision is preserved
- K4.5: Out-of-order writes are sorted correctly
- K4.6: Timestamp and created_at are independent (ordering uses timestamp)
- K4.7: Replay respects time boundaries strictly

Each test provides:
- Numbered proof steps (P1, P2, P3, P4)
- Actual timestamp values shown
- Ordering verified event-by-event
- Proof summary output

Run with: pytest -v -s prod_test/layer1_kernel/test_k4_timestamp_monotonicity.py
"""
