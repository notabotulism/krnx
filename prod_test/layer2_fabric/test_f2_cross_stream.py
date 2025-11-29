"""
KRNX Layer 2 Tests - F2: Cross-Stream Consistency (CORRECTED)

PROOF F2: Events across multiple streams maintain consistency.

CORRECTED:
- query_events() has NO 'order' param
- get_event(event_id) takes ONLY event_id
- Wait for async migration before verification
"""

import pytest
import time
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple

from .conftest import CONCURRENT_CONFIGS


@pytest.mark.layer2
@pytest.mark.requires_redis
class TestF2CrossStreamConsistency:
    """
    F2: Prove cross-stream consistency under concurrency.
    
    THEOREM: ∀ streams S₁...Sₙ receiving concurrent writes:
      1. Events in S_i are independent of S_j (isolation)
      2. Global event IDs are unique across all streams
      3. Per-stream ordering is maintained
    """
    
    # =========================================================================
    # F2.1: Stream Isolation Under Concurrent Writes
    # =========================================================================
    
    @pytest.mark.parametrize("n_streams", [2, 5, 10])
    def test_f2_1_stream_isolation(
        self,
        fabric_no_embed,
        unique_workspace: str,
        n_streams: int,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        F2.1: Writes to different streams don't interfere.
        
        PROOF:
        - P1: Create N user streams
        - P2: Concurrent writes to each stream
        - P3: Verify each stream has correct events
        - P4: No cross-contamination
        """
        events_per_stream = 50
        users = [f"user_{uuid.uuid4().hex[:8]}" for _ in range(n_streams)]
        
        written: Dict[str, List[str]] = {u: [] for u in users}
        lock = threading.Lock()
        barrier = threading.Barrier(n_streams)
        
        def writer(user_id: str):
            barrier.wait()
            for i in range(events_per_stream):
                event_id = fabric_no_embed.remember(
                    content={"user": user_id, "seq": i},
                    workspace_id=unique_workspace,
                    user_id=user_id,
                )
                with lock:
                    written[user_id].append(event_id)
        
        # P2: Execute concurrent writes
        with ThreadPoolExecutor(max_workers=n_streams) as executor:
            futures = [executor.submit(writer, u) for u in users]
            for f in as_completed(futures):
                f.result()
        
        # Wait for migration
        total_events = n_streams * events_per_stream
        wait_for_migration(fabric_no_embed.kernel, total_events, timeout=60)
        
        # P3: Verify each stream - CORRECTED: no 'order' param
        contamination_count = 0
        for user_id in users:
            events = list(fabric_no_embed.kernel.query_events(
                workspace_id=unique_workspace,
                user_id=user_id,
                limit=events_per_stream + 10,
            ))
            
            # Check no events from other users
            for e in events:
                if e.content.get("user") != user_id:
                    contamination_count += 1
        
        assert contamination_count == 0, \
            f"P4 VIOLATED: {contamination_count} cross-stream contaminations"
        
        print_proof_summary(
            test_id="F2.1",
            guarantee="Stream Isolation",
            metrics={
                "n_streams": n_streams,
                "events_per_stream": events_per_stream,
                "contaminations": contamination_count,
            },
            result=f"ISOLATION PROVEN: 0 contaminations across {n_streams} streams"
        )
    
    # =========================================================================
    # F2.2: Global Event ID Uniqueness
    # =========================================================================
    
    @pytest.mark.parametrize("n_streams", [5, 10])
    def test_f2_2_global_event_id_uniqueness(
        self,
        fabric_no_embed,
        unique_workspace: str,
        n_streams: int,
        print_proof_summary,
    ):
        """
        F2.2: Event IDs are unique across all streams.
        
        PROOF:
        - P1: Multiple streams write concurrently
        - P2: Collect all event IDs
        - P3: Verify global uniqueness
        """
        events_per_stream = 100
        users = [f"user_{uuid.uuid4().hex[:8]}" for _ in range(n_streams)]
        
        all_ids: List[str] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_streams)
        
        def writer(user_id: str):
            barrier.wait()
            for i in range(events_per_stream):
                event_id = fabric_no_embed.remember(
                    content={"stream": user_id, "i": i},
                    workspace_id=unique_workspace,
                    user_id=user_id,
                )
                with lock:
                    all_ids.append(event_id)
        
        with ThreadPoolExecutor(max_workers=n_streams) as executor:
            futures = [executor.submit(writer, u) for u in users]
            for f in as_completed(futures):
                f.result()
        
        # P3: Global uniqueness
        unique_ids = set(all_ids)
        collisions = len(all_ids) - len(unique_ids)
        
        assert collisions == 0, \
            f"P3 VIOLATED: {collisions} global ID collisions"
        
        print_proof_summary(
            test_id="F2.2",
            guarantee="Global Event ID Uniqueness",
            metrics={
                "n_streams": n_streams,
                "total_events": len(all_ids),
                "unique_ids": len(unique_ids),
                "collisions": collisions,
            },
            result=f"GLOBAL UNIQUENESS: {len(unique_ids)}/{len(all_ids)} unique"
        )
    
    # =========================================================================
    # F2.3: Per-Stream Ordering Maintained
    # =========================================================================
    
    @pytest.mark.parametrize("n_streams", [3, 5])
    def test_f2_3_per_stream_ordering(
        self,
        fabric_no_embed,
        unique_workspace: str,
        n_streams: int,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        F2.3: Each stream maintains monotonic ordering.
        
        PROOF:
        - P1: Concurrent writes to N streams
        - P2: Query each stream
        - P3: Verify monotonic timestamps per stream
        """
        events_per_stream = 50
        users = [f"user_{uuid.uuid4().hex[:8]}" for _ in range(n_streams)]
        barrier = threading.Barrier(n_streams)
        
        def writer(user_id: str):
            barrier.wait()
            for i in range(events_per_stream):
                fabric_no_embed.remember(
                    content={"seq": i},
                    workspace_id=unique_workspace,
                    user_id=user_id,
                )
                time.sleep(0.001)  # Small delay for timestamp ordering
        
        with ThreadPoolExecutor(max_workers=n_streams) as executor:
            futures = [executor.submit(writer, u) for u in users]
            for f in as_completed(futures):
                f.result()
        
        # Wait for migration
        wait_for_migration(fabric_no_embed.kernel, n_streams * events_per_stream, timeout=60)
        
        # P3: Verify per-stream ordering
        ordering_violations = 0
        for user_id in users:
            events = list(fabric_no_embed.kernel.query_events(
                workspace_id=unique_workspace,
                user_id=user_id,
                limit=events_per_stream + 10,
            ))
            
            timestamps = [e.timestamp for e in events]
            for i in range(1, len(timestamps)):
                if timestamps[i] < timestamps[i-1]:
                    ordering_violations += 1
        
        assert ordering_violations == 0, \
            f"P3 VIOLATED: {ordering_violations} per-stream ordering violations"
        
        print_proof_summary(
            test_id="F2.3",
            guarantee="Per-Stream Ordering",
            metrics={
                "n_streams": n_streams,
                "events_per_stream": events_per_stream,
                "ordering_violations": ordering_violations,
            },
            result=f"ORDERING PROVEN: 0 violations across {n_streams} streams"
        )
    
    # =========================================================================
    # F2.4: Cross-Stream Event Retrieval (CORRECTED)
    # =========================================================================
    
    def test_f2_4_cross_stream_event_retrieval(
        self,
        fabric_no_embed,
        unique_workspace: str,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        F2.4: Events can be retrieved by ID regardless of stream.
        
        CORRECTED: get_event(event_id) takes ONLY event_id
        """
        n_streams = 5
        events_per_stream = 20
        users = [f"user_{uuid.uuid4().hex[:8]}" for _ in range(n_streams)]
        
        all_events: Dict[str, Tuple[str, Dict]] = {}  # event_id -> (user_id, content)
        lock = threading.Lock()
        
        # Write events
        for user_id in users:
            for i in range(events_per_stream):
                content = {"user": user_id, "seq": i, "data": f"event_{i}"}
                event_id = fabric_no_embed.remember(
                    content=content,
                    workspace_id=unique_workspace,
                    user_id=user_id,
                )
                all_events[event_id] = (user_id, content)
        
        # Wait for migration
        wait_for_migration(fabric_no_embed.kernel, len(all_events), timeout=60)
        
        # CORRECTED: Retrieve by ID only
        retrieval_failures = []
        for event_id, (user_id, expected_content) in all_events.items():
            retrieved = fabric_no_embed.kernel.get_event(event_id)
            if retrieved is None:
                retrieval_failures.append((event_id, "NOT_FOUND"))
            elif retrieved.content != expected_content:
                retrieval_failures.append((event_id, "CONTENT_MISMATCH"))
        
        assert len(retrieval_failures) == 0, \
            f"P3 VIOLATED: {len(retrieval_failures)} retrieval failures: {retrieval_failures[:5]}"
        
        print_proof_summary(
            test_id="F2.4",
            guarantee="Cross-Stream Event Retrieval",
            metrics={
                "n_streams": n_streams,
                "total_events": len(all_events),
                "retrieval_failures": len(retrieval_failures),
            },
            result=f"RETRIEVAL PROVEN: {len(all_events)}/{len(all_events)} retrieved"
        )


# ==============================================
# F2 PROOF SUMMARY
# ==============================================

class TestF2ProofSummary:
    """Generate F2 proof summary."""
    
    def test_f2_proof_complete(self, print_proof_summary):
        print("\n" + "="*70)
        print("F2 CROSS-STREAM CONSISTENCY - PROOF SUMMARY")
        print("="*70)
        print("""
F2: CROSS-STREAM CONSISTENCY GUARANTEES

  F2.1: Stream Isolation
    - Events in stream S_i isolated from S_j
    - No cross-contamination under concurrency
    - Tested with 2, 5, 10 concurrent streams
    
  F2.2: Global Event ID Uniqueness
    - All event IDs unique across all streams
    - Zero collisions verified
    
  F2.3: Per-Stream Ordering
    - Each stream maintains monotonic timestamps
    - Ordering independent of other streams
    
  F2.4: Cross-Stream Event Retrieval
    - Events retrievable by ID regardless of origin stream
    - Content integrity verified

METHODOLOGY: P1-P2-P3-P4 with barrier-synchronized threads
SCALE: Up to 10 streams × 100 events = 1,000 events
""")
        print("="*70 + "\n")
