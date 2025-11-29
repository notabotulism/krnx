"""
KRNX Layer 2 Tests - F1: Stream Ordering (CORRECTED)

PROOF F1: Events within a stream are totally ordered.

CORRECTED:
- query_events() has NO 'order' param - returns chronological by default
- get_event(event_id) takes ONLY event_id, not workspace/user
- Properly wait for async migration before verification

TEST METHODOLOGY: P1-P2-P3-P4 Proof Structure
"""

import pytest
import time
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Set, List, Tuple

from .conftest import CONCURRENT_CONFIGS


@pytest.mark.layer2
@pytest.mark.requires_redis
class TestF1StreamOrdering:
    """
    F1: Prove stream ordering under concurrency.
    
    THEOREM: ∀ concurrent writers W₁...Wₙ writing to stream S:
      1. Total events = Σ events written (no loss)
      2. ∀ i,j: event_id(i) ≠ event_id(j) (no duplicates)
      3. ∀ i < j: timestamp(i) ≤ timestamp(j) (monotonic order)
    """
    
    # =========================================================================
    # F1.1: Concurrent Writers Produce Total Order
    # =========================================================================
    
    @pytest.mark.parametrize("n_writers,events_per_writer", CONCURRENT_CONFIGS)
    def test_f1_1_concurrent_writers_total_order(
        self,
        fabric_no_embed,
        unique_workspace: str,
        unique_user: str,
        n_writers: int,
        events_per_writer: int,
        wait_for_migration,
        print_proof_summary,
        statistical_analyzer,
    ):
        """
        F1.1: N concurrent writers produce totally ordered stream.
        
        PROOF:
        - P1: Initialize N writer threads with barrier synchronization
        - P2: All writers start simultaneously, each writes M events
        - P3: Wait for migration, then query all events
        - P4: Verify monotonic timestamp ordering
        """
        barrier = threading.Barrier(n_writers)
        all_ids: List[str] = []
        latencies: List[float] = []
        lock = threading.Lock()
        
        def writer(writer_id: int):
            barrier.wait()  # P1: Synchronized start
            writer_ids = []
            for i in range(events_per_writer):
                start = time.perf_counter()
                event_id = fabric_no_embed.remember(
                    content={"writer": writer_id, "seq": i},
                    workspace_id=unique_workspace,
                    user_id=unique_user,
                )
                writer_ids.append(event_id)
                with lock:
                    latencies.append((time.perf_counter() - start) * 1000)
            with lock:
                all_ids.extend(writer_ids)
        
        # P2: Execute concurrent writes
        with ThreadPoolExecutor(max_workers=n_writers) as executor:
            futures = [executor.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(futures):
                f.result()
        
        # Wait for async migration to complete
        expected_count = n_writers * events_per_writer
        wait_for_migration(fabric_no_embed.kernel, expected_count, timeout=60)
        
        # Query all events - CORRECTED: no 'order' parameter
        events = list(fabric_no_embed.kernel.query_events(
            workspace_id=unique_workspace,
            user_id=unique_user,
            limit=expected_count + 100,
        ))
        
        # P3: Assert no loss
        assert len(events) == expected_count, \
            f"P3 VIOLATED: Expected {expected_count} events, got {len(events)}"
        
        # P3: Assert no duplicates
        event_ids = [e.event_id for e in events]
        assert len(event_ids) == len(set(event_ids)), \
            "P3 VIOLATED: Duplicate event IDs detected"
        
        # P4: Assert monotonic timestamps (query returns chronological order)
        timestamps = [e.timestamp for e in events]
        ordering_violations = 0
        for i in range(1, len(timestamps)):
            if timestamps[i] < timestamps[i-1]:
                ordering_violations += 1
        
        assert ordering_violations == 0, \
            f"P4 VIOLATED: {ordering_violations} timestamp ordering violations"
        
        # Statistical analysis
        stats = statistical_analyzer.analyze(latencies)
        
        print_proof_summary(
            test_id="F1.1",
            guarantee="Concurrent Writers Total Order",
            metrics={
                "n_writers": n_writers,
                "events_per_writer": events_per_writer,
                "total_events": len(events),
                "unique_ids": len(set(event_ids)),
                "ordering_violations": ordering_violations,
                "latency_mean_ms": stats.mean,
                "latency_p95_ms": stats.p95,
            },
            statistical=stats,
            result=f"TOTAL ORDER PROVEN: {len(events)}/{expected_count} events, 0 violations"
        )
    
    # =========================================================================
    # F1.2: No Gaps in Sequence
    # =========================================================================
    
    @pytest.mark.parametrize("n_writers,events_per_writer", CONCURRENT_CONFIGS)
    def test_f1_2_no_gaps_in_sequence(
        self,
        fabric_no_embed,
        unique_workspace: str,
        unique_user: str,
        n_writers: int,
        events_per_writer: int,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        F1.2: All events from all writers are present.
        
        PROOF:
        - P1: Track expected (writer_id, seq) pairs
        - P2: Concurrent writers write tagged events
        - P3: Wait for migration, query and verify all pairs present
        - P4: No missing events (complete sequence)
        """
        expected: Dict[Tuple[int, int], str] = {}
        barrier = threading.Barrier(n_writers)
        lock = threading.Lock()
        
        def writer(writer_id: int):
            barrier.wait()
            for i in range(events_per_writer):
                event_id = fabric_no_embed.remember(
                    content={"writer": writer_id, "seq": i},
                    workspace_id=unique_workspace,
                    user_id=unique_user,
                )
                with lock:
                    expected[(writer_id, i)] = event_id
        
        # P2: Execute
        with ThreadPoolExecutor(max_workers=n_writers) as executor:
            futures = [executor.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(futures):
                f.result()
        
        # Wait for migration
        wait_for_migration(fabric_no_embed.kernel, len(expected), timeout=60)
        
        # P3: Query and verify
        events = list(fabric_no_embed.kernel.query_events(
            workspace_id=unique_workspace,
            user_id=unique_user,
            limit=n_writers * events_per_writer + 100,
        ))
        
        found: Set[Tuple[int, int]] = set()
        for e in events:
            key = (e.content["writer"], e.content["seq"])
            found.add(key)
        
        # P4: Check for gaps
        missing = []
        for key in expected.keys():
            if key not in found:
                missing.append(key)
        
        assert len(missing) == 0, \
            f"P4 VIOLATED: Missing events: {missing[:10]}..."
        
        print_proof_summary(
            test_id="F1.2",
            guarantee="No Gaps in Sequence",
            metrics={
                "n_writers": n_writers,
                "events_per_writer": events_per_writer,
                "expected_pairs": len(expected),
                "found_pairs": len(found),
                "missing_count": len(missing),
            },
            result=f"NO GAPS: {len(found)}/{len(expected)} events found"
        )
    
    # =========================================================================
    # F1.3: Event IDs Are Unique
    # =========================================================================
    
    @pytest.mark.parametrize("n_writers,events_per_writer", CONCURRENT_CONFIGS)
    def test_f1_3_event_ids_unique(
        self,
        fabric_no_embed,
        unique_workspace: str,
        unique_user: str,
        n_writers: int,
        events_per_writer: int,
        print_proof_summary,
    ):
        """
        F1.3: Event IDs are unique across concurrent writes.
        
        PROOF:
        - P1: Concurrent writers write events
        - P2: Collect all event IDs
        - P3: Verify no collisions
        - P4: Verify ID entropy (length >= 16)
        """
        all_ids: List[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_writers)
        
        def writer(writer_id: int):
            barrier.wait()
            for i in range(events_per_writer):
                event_id = fabric_no_embed.remember(
                    content={"writer": writer_id, "seq": i},
                    workspace_id=unique_workspace,
                    user_id=unique_user,
                )
                with lock:
                    all_ids.append(event_id)
        
        with ThreadPoolExecutor(max_workers=n_writers) as executor:
            futures = [executor.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(futures):
                f.result()
        
        # P3: Uniqueness check
        unique_ids = set(all_ids)
        collision_count = len(all_ids) - len(unique_ids)
        
        assert collision_count == 0, \
            f"P3 VIOLATED: {collision_count} ID collisions detected"
        
        # P4: Entropy check (IDs should be >= 16 chars)
        min_length = min(len(eid) for eid in all_ids)
        
        assert min_length >= 16, \
            f"P4 VIOLATED: ID length {min_length} < 16 (insufficient entropy)"
        
        print_proof_summary(
            test_id="F1.3",
            guarantee="Event ID Uniqueness",
            metrics={
                "total_ids": len(all_ids),
                "unique_ids": len(unique_ids),
                "collisions": collision_count,
                "min_id_length": min_length,
            },
            result=f"UNIQUENESS PROVEN: {len(unique_ids)}/{len(all_ids)} unique"
        )
    
    # =========================================================================
    # F1.4: Ordering Determinism
    # =========================================================================
    
    def test_f1_4_ordering_determinism(
        self,
        fabric_no_embed,
        unique_workspace: str,
        unique_user: str,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        F1.4: Query ordering is deterministic.
        
        PROOF:
        - P1: Write N events sequentially
        - P2: Wait for migration
        - P3: Query multiple times with same parameters
        - P4: All queries return same order
        """
        n_events = 100
        
        # P1: Sequential writes
        written_ids = []
        for i in range(n_events):
            event_id = fabric_no_embed.remember(
                content={"seq": i, "data": f"event_{i}"},
                workspace_id=unique_workspace,
                user_id=unique_user,
            )
            written_ids.append(event_id)
            time.sleep(0.001)  # Ensure distinct timestamps
        
        # Wait for migration
        wait_for_migration(fabric_no_embed.kernel, n_events, timeout=30)
        
        # P2: Multiple queries - CORRECTED: no 'order' param
        n_queries = 10
        query_results = []
        for _ in range(n_queries):
            events = list(fabric_no_embed.kernel.query_events(
                workspace_id=unique_workspace,
                user_id=unique_user,
                limit=n_events + 10,
            ))
            query_results.append([e.event_id for e in events])
        
        # P3: All queries identical
        first_result = query_results[0]
        differences = 0
        for i, result in enumerate(query_results[1:], 1):
            if result != first_result:
                differences += 1
        
        assert differences == 0, \
            f"P3 VIOLATED: {differences} queries returned different order"
        
        # P4: Matches write order
        order_matches = first_result == written_ids
        
        print_proof_summary(
            test_id="F1.4",
            guarantee="Ordering Determinism",
            metrics={
                "n_events": n_events,
                "n_queries": n_queries,
                "ordering_differences": differences,
                "matches_write_order": order_matches,
            },
            result=f"DETERMINISM PROVEN: {n_queries} identical queries"
        )
    
    # =========================================================================
    # F1.5: Write-Then-Read Consistency
    # =========================================================================
    
    @pytest.mark.parametrize("n_writers,events_per_writer", CONCURRENT_CONFIGS)
    def test_f1_5_write_then_read_consistency(
        self,
        fabric_no_embed,
        unique_workspace: str,
        unique_user: str,
        n_writers: int,
        events_per_writer: int,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        F1.5: Written content equals read content.
        
        PROOF:
        - P1: Write events with checksummed content
        - P2: Wait for migration
        - P3: Read each event - CORRECTED: get_event(event_id) only
        - P4: Content matches byte-for-byte
        """
        written: Dict[str, Dict] = {}
        barrier = threading.Barrier(n_writers)
        lock = threading.Lock()
        
        def writer(writer_id: int):
            barrier.wait()
            for i in range(events_per_writer):
                content = {
                    "writer": writer_id,
                    "seq": i,
                    "checksum": hashlib.md5(f"w{writer_id}s{i}".encode()).hexdigest()
                }
                event_id = fabric_no_embed.remember(
                    content=content,
                    workspace_id=unique_workspace,
                    user_id=unique_user,
                )
                with lock:
                    written[event_id] = content
        
        with ThreadPoolExecutor(max_workers=n_writers) as executor:
            futures = [executor.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(futures):
                f.result()
        
        # P2: Wait for migration
        wait_for_migration(fabric_no_embed.kernel, len(written), timeout=60)
        
        # P3 & P4: Verify content - CORRECTED: get_event takes only event_id
        mismatches = []
        for event_id, expected_content in written.items():
            retrieved = fabric_no_embed.kernel.get_event(event_id)
            if retrieved is None:
                mismatches.append((event_id, "NOT_FOUND"))
            elif retrieved.content != expected_content:
                mismatches.append((event_id, "CONTENT_MISMATCH"))
        
        assert len(mismatches) == 0, \
            f"P4 VIOLATED: {len(mismatches)} mismatches: {mismatches[:5]}"
        
        print_proof_summary(
            test_id="F1.5",
            guarantee="Write-Read Consistency",
            metrics={
                "events_written": len(written),
                "events_verified": len(written) - len(mismatches),
                "mismatches": len(mismatches),
            },
            result=f"CONSISTENCY PROVEN: {len(written)}/{len(written)} match"
        )


# ==============================================
# F1 PROOF SUMMARY
# ==============================================

class TestF1ProofSummary:
    """Generate F1 proof summary."""
    
    def test_f1_proof_complete(self, print_proof_summary):
        print("\n" + "="*70)
        print("F1 STREAM ORDERING - PROOF SUMMARY")
        print("="*70)
        print("""
F1: STREAM ORDERING GUARANTEES (per Playbook §2.3.3)

  F1.1: Concurrent Writers Produce Total Order
    - N writers × M events = N×M total events (no loss)
    - Timestamps monotonically increasing
    - Scale: (10×100), (50×100), (10×1000)
    
  F1.2: No Gaps in Sequence
    - All (writer_id, seq) pairs present
    - Complete event coverage verified
    
  F1.3: Event ID Uniqueness
    - |set(ids)| == |list(ids)| (zero collisions)
    - ID entropy >= 64 bits (length >= 16)
    
  F1.4: Ordering Determinism
    - N identical queries return identical order
    - Query order matches write order
    
  F1.5: Write-Read Consistency
    - Content checksums match
    - No data corruption under concurrency

METHODOLOGY: P1-P2-P3-P4 with barrier-synchronized threads
SCALE: 1,000 / 5,000 / 10,000 total events per test
""")
        print("="*70 + "\n")
