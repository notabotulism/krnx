"""
K2: Hash-Chain Integrity

THEOREM: KRNX maintains cryptographic integrity through hash chains.
Each event links to its predecessor via previous_hash, forming a
tamper-evident audit trail.

PROOF STRUCTURE:
Each test follows the format:
- P1 (Precondition): Establish known initial state
- P2 (Operation): Perform the operation under test
- P3 (Postcondition): Verify expected outcome
- P4 (Mechanism): Prove WHY the outcome is correct

Academic rigor requirements:
- Show actual hash values in assertions
- Verify chain linkage explicitly
- Demonstrate tamper detection
- Each test is independently reproducible
"""

import pytest
import time
import hashlib
import json
import uuid
from typing import List, Dict, Any, Optional

from chillbot.kernel.models import Event
from chillbot.kernel.controller import KRNXController

from ..config import KERNEL_TEST_CONFIG


@pytest.mark.layer1
@pytest.mark.requires_redis
class TestK2HashChain:
    """
    K2: Hash-Chain Integrity Proofs
    
    These tests prove that KRNX correctly implements hash-chain semantics:
    - Hashes are computed deterministically
    - Chains link correctly via previous_hash
    - Chain breaks are detectable
    - Chains are tamper-evident
    """
    
    # ==========================================
    # K2.1: Hash Computation Determinism
    # ==========================================
    
    def test_k2_1_hash_determinism(
        self,
        test_workspace: str,
        test_user: str,
    ):
        """
        K2.1: Hash computation is deterministic.
        
        THEOREM: For any event E, compute_hash(E) always returns the same value.
        
        PROOF:
        - P1: Create event with fixed content
        - P2: Compute hash N times
        - P3: All N hashes are identical
        - P4: Show actual hash value
        """
        N = 100
        
        # === P1: Create event with fixed, known content ===
        fixed_content = {
            "type": "determinism_proof",
            "value": 42,
            "nested": {"a": 1, "b": 2},
        }
        
        event = Event(
            event_id="k2-1-determinism-test",
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k2-1-session",
            content=fixed_content,
            timestamp=1000000.123456,  # Fixed timestamp
            created_at=1000000.123456,  # Fixed created_at
        )
        
        # === P2: Compute hash N times ===
        hashes = [event.compute_hash() for _ in range(N)]
        
        # === P3: All hashes identical ===
        unique_hashes = set(hashes)
        
        assert len(unique_hashes) == 1, (
            f"P3 VIOLATED: Hash computation non-deterministic!\n"
            f"  Unique hashes found: {len(unique_hashes)}\n"
            f"  Sample hashes: {list(unique_hashes)[:5]}"
        )
        
        # === P4: Show actual hash ===
        computed_hash = hashes[0]
        
        # Verify hash format (64 hex characters = SHA-256)
        assert len(computed_hash) == 64, f"Hash length wrong: {len(computed_hash)}"
        assert all(c in '0123456789abcdef' for c in computed_hash), "Hash contains non-hex characters"
        
        print(f"\n{'='*60}")
        print(f"K2.1 PROOF SUMMARY: Hash Computation Determinism")
        print(f"{'='*60}")
        print(f"  Iterations:       {N}")
        print(f"  Unique hashes:    {len(unique_hashes)}")
        print(f"  Hash value:       {computed_hash}")
        print(f"  Hash length:      {len(computed_hash)} chars (SHA-256)")
        print(f"  RESULT: DETERMINISTIC HASH COMPUTATION VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.2: Hash Uniqueness (Collision Resistance)
    # ==========================================
    
    def test_k2_2_hash_uniqueness(
        self,
        test_workspace: str,
        test_user: str,
    ):
        """
        K2.2: Different events produce different hashes.
        
        THEOREM: For events E1 ≠ E2, hash(E1) ≠ hash(E2) with overwhelming probability.
        
        PROOF:
        - P1: Create N events with unique content
        - P2: Compute hash for each
        - P3: All N hashes are unique
        - P4: Show collision rate = 0
        """
        N = 100
        
        # === P1 & P2: Create events and compute hashes ===
        events_and_hashes: List[Dict[str, Any]] = []
        
        for i in range(N):
            event = Event(
                event_id=f"k2-2-unique-{i:04d}",
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k2-2-session",
                content={"index": i, "unique_marker": uuid.uuid4().hex},
                timestamp=time.time() + i,
                created_at=time.time(),
            )
            
            events_and_hashes.append({
                "event_id": event.event_id,
                "hash": event.compute_hash(),
            })
        
        # === P3: All hashes unique ===
        all_hashes = [eh["hash"] for eh in events_and_hashes]
        unique_hashes = set(all_hashes)
        
        collisions = N - len(unique_hashes)
        
        assert collisions == 0, (
            f"P3 VIOLATED: Hash collisions detected!\n"
            f"  Total events: {N}\n"
            f"  Unique hashes: {len(unique_hashes)}\n"
            f"  Collisions: {collisions}"
        )
        
        # === P4: Show collision rate ===
        collision_rate = collisions / N
        
        print(f"\n{'='*60}")
        print(f"K2.2 PROOF SUMMARY: Hash Uniqueness")
        print(f"{'='*60}")
        print(f"  Events created:   {N}")
        print(f"  Unique hashes:    {len(unique_hashes)}")
        print(f"  Collisions:       {collisions}")
        print(f"  Collision rate:   {collision_rate:.6f}")
        print(f"  Sample hashes:")
        for eh in events_and_hashes[:3]:
            print(f"    {eh['event_id']}: {eh['hash'][:32]}...")
        print(f"  RESULT: ZERO COLLISIONS VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.3: Hash-Chain Creation (Auto-Linking)
    # ==========================================
    
    def test_k2_3_chain_auto_linking(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K2.3: Kernel automatically links events in hash chain when enabled.
        
        THEOREM: When enable_hash_chain=True, each event's previous_hash
        is set to the hash of the preceding event.
        
        PROOF:
        - P1: Write N events sequentially
        - P2: Retrieve all events
        - P3: Verify event[i].previous_hash == hash(event[i-1])
        - P4: Show actual chain linkage
        """
        N = 10
        base_time = time.time()
        
        # === P1: Write N events ===
        written_ids = []
        for i in range(N):
            event = Event(
                event_id=f"k2-3-chain-{i:04d}",
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k2-3-session",
                content={"index": i},
                timestamp=base_time + i,
                created_at=time.time(),
                # Don't set previous_hash - kernel should auto-link
            )
            kernel.write_event(test_workspace, test_user, event)
            written_ids.append(event.event_id)
        
        # Wait for migration
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # === P2: Retrieve all events in order ===
        retrieved = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 10
        )
        
        # Sort by timestamp
        retrieved_sorted = sorted(retrieved, key=lambda e: e.timestamp)
        
        assert len(retrieved_sorted) == N, (
            f"P2 VIOLATED: Expected {N} events, got {len(retrieved_sorted)}"
        )
        
        # === P3: Verify chain linkage ===
        chain_links: List[Dict[str, Any]] = []
        linkage_errors = []
        
        for i, event in enumerate(retrieved_sorted):
            if i == 0:
                # First event should have no previous_hash
                expected_prev = None
                link_info = {
                    "position": i,
                    "event_id": event.event_id,
                    "previous_hash": event.previous_hash,
                    "expected_previous": expected_prev,
                    "valid": event.previous_hash == expected_prev,
                }
            else:
                # Subsequent events should link to predecessor
                predecessor = retrieved_sorted[i - 1]
                expected_prev = predecessor.compute_hash()
                link_info = {
                    "position": i,
                    "event_id": event.event_id,
                    "previous_hash": event.previous_hash[:16] + "..." if event.previous_hash else None,
                    "expected_previous": expected_prev[:16] + "...",
                    "valid": event.previous_hash == expected_prev,
                }
                
                if event.previous_hash != expected_prev:
                    linkage_errors.append({
                        "position": i,
                        "event_id": event.event_id,
                        "expected": expected_prev,
                        "actual": event.previous_hash,
                    })
            
            chain_links.append(link_info)
        
        assert len(linkage_errors) == 0, (
            f"P3 VIOLATED: Chain linkage errors:\n" +
            "\n".join(
                f"  Position {e['position']}: expected {e['expected'][:16]}..., got {e['actual'][:16] if e['actual'] else None}..."
                for e in linkage_errors
            )
        )
        
        # === P4: Show chain structure ===
        print(f"\n{'='*60}")
        print(f"K2.3 PROOF SUMMARY: Hash-Chain Auto-Linking")
        print(f"{'='*60}")
        print(f"  Events in chain:  {N}")
        print(f"  Linkage errors:   {len(linkage_errors)}")
        print(f"  Chain structure:")
        for link in chain_links[:5]:
            status = "✓" if link["valid"] else "✗"
            print(f"    [{link['position']}] {link['event_id']}: prev={link['previous_hash']} {status}")
        if N > 5:
            print(f"    ... ({N - 5} more events)")
        print(f"  RESULT: CHAIN AUTO-LINKING VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.4: Chain Verification
    # ==========================================
    
    def test_k2_4_chain_verification(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K2.4: verify_hash_chain() correctly validates chain integrity.
        
        THEOREM: verify_hash_chain() returns valid=True iff all chain links
        are correct.
        
        PROOF:
        - P1: Create valid chain
        - P2: Call verify_hash_chain()
        - P3: Result shows valid=True, gaps=0, corrupted=0
        - P4: Verify each link was actually checked
        """
        N = 20
        base_time = time.time()
        
        # === P1: Create valid chain ===
        for i in range(N):
            event = Event(
                event_id=f"k2-4-verify-{i:04d}",
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k2-4-session",
                content={"index": i, "data": f"event-{i}"},
                timestamp=base_time + i,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
        
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # === P2: Call verify_hash_chain ===
        result = kernel.verify_hash_chain(test_workspace, test_user)
        
        # === P3: Validate result structure and values ===
        assert 'valid' in result, "P3 VIOLATED: Result missing 'valid' field"
        assert 'events_verified' in result, "P3 VIOLATED: Result missing 'events_verified'"
        assert 'gaps' in result, "P3 VIOLATED: Result missing 'gaps'"
        assert 'corrupted' in result, "P3 VIOLATED: Result missing 'corrupted'"
        assert 'issues' in result, "P3 VIOLATED: Result missing 'issues'"
        
        assert result['valid'] == True, (
            f"P3 VIOLATED: Chain should be valid but got valid={result['valid']}\n"
            f"  Issues: {result['issues']}"
        )
        
        assert result['gaps'] == 0, (
            f"P3 VIOLATED: Chain should have no gaps but found {result['gaps']}"
        )
        
        assert result['corrupted'] == 0, (
            f"P3 VIOLATED: Chain should have no corruption but found {result['corrupted']}"
        )
        
        # === P4: Verify all events were checked ===
        assert result['events_verified'] == N, (
            f"P4 VIOLATED: Expected {N} events verified, got {result['events_verified']}"
        )
        
        print(f"\n{'='*60}")
        print(f"K2.4 PROOF SUMMARY: Chain Verification")
        print(f"{'='*60}")
        print(f"  Events in chain:     {N}")
        print(f"  Events verified:     {result['events_verified']}")
        print(f"  Chain valid:         {result['valid']}")
        print(f"  Gaps detected:       {result['gaps']}")
        print(f"  Corrupted events:    {result['corrupted']}")
        print(f"  First event ID:      {result.get('first_event_id', 'N/A')}")
        print(f"  Last event ID:       {result.get('last_event_id', 'N/A')}")
        print(f"  RESULT: CHAIN VERIFICATION PASSED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.5: Chain Break Detection
    # ==========================================
    
    def test_k2_5_chain_break_detection(
        self,
        test_data_path,
        test_workspace: str,
        test_user: str,
    ):
        """
        K2.5: verify_hash_chain() detects broken chains.
        
        THEOREM: If event[i].previous_hash ≠ hash(event[i-1]), 
        verify_hash_chain() returns valid=False with issue details.
        
        PROOF:
        - P1: Create chain with enable_hash_chain=False (manual control)
        - P2: Write valid chain, then inject broken link
        - P3: Call verify_hash_chain()
        - P4: Verify break is detected with correct details
        """
        # === P1: Create kernel without auto-chaining ===
        kernel = KRNXController(
            data_path=str(test_data_path),
            enable_backpressure=False,
            enable_async_worker=True,
            enable_hash_chain=False,  # Manual control
        )
        
        try:
            N = 5
            base_time = time.time()
            
            # === P2: Write valid chain, then inject break ===
            previous_hash = None
            written_hashes = []
            
            # Write N-1 valid events
            for i in range(N - 1):
                event = Event(
                    event_id=f"k2-5-break-{i:04d}",
                    workspace_id=test_workspace,
                    user_id=test_user,
                    session_id="k2-5-session",
                    content={"index": i},
                    timestamp=base_time + i,
                    created_at=time.time(),
                    previous_hash=previous_hash,
                )
                kernel.write_event(test_workspace, test_user, event)
                current_hash = event.compute_hash()
                written_hashes.append({
                    "event_id": event.event_id,
                    "hash": current_hash,
                    "previous_hash": previous_hash,
                })
                previous_hash = current_hash
            
            # Inject broken link (wrong previous_hash)
            correct_previous = previous_hash
            wrong_previous = "0" * 64  # Intentionally wrong
            
            broken_event = Event(
                event_id=f"k2-5-break-{N-1:04d}",
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k2-5-session",
                content={"index": N - 1, "broken": True},
                timestamp=base_time + (N - 1),
                created_at=time.time(),
                previous_hash=wrong_previous,  # BROKEN!
            )
            kernel.write_event(test_workspace, test_user, broken_event)
            
            # Wait for migration
            start = time.time()
            while time.time() - start < 30:
                metrics = kernel.get_worker_metrics()
                if metrics.messages_processed >= N:
                    break
                time.sleep(0.5)
            
            # === P3: Call verify_hash_chain ===
            result = kernel.verify_hash_chain(test_workspace, test_user)
            
            # === P4: Verify break detected ===
            assert result['valid'] == False, (
                f"P4 VIOLATED: Chain with broken link reported as valid!\n"
                f"  Correct previous_hash: {correct_previous[:16]}...\n"
                f"  Injected previous_hash: {wrong_previous[:16]}...\n"
                f"  Result: {result}"
            )
            
            assert result['gaps'] >= 1, (
                f"P4 VIOLATED: Expected gaps >= 1, got {result['gaps']}"
            )
            
            assert len(result['issues']) >= 1, (
                f"P4 VIOLATED: Expected issues list to contain break details"
            )
            
            # Find the chain_break issue
            chain_break_issues = [i for i in result['issues'] if i.get('type') == 'chain_break']
            assert len(chain_break_issues) >= 1, (
                f"P4 VIOLATED: No chain_break issue found. Issues: {result['issues']}"
            )
            
            break_issue = chain_break_issues[0]
            
            print(f"\n{'='*60}")
            print(f"K2.5 PROOF SUMMARY: Chain Break Detection")
            print(f"{'='*60}")
            print(f"  Events in chain:      {N}")
            print(f"  Broken link position: {N - 1}")
            print(f"  Correct previous:     {correct_previous[:32]}...")
            print(f"  Injected previous:    {wrong_previous[:32]}...")
            print(f"  Chain valid:          {result['valid']}")
            print(f"  Gaps detected:        {result['gaps']}")
            print(f"  Issue details:")
            print(f"    Type:     {break_issue.get('type')}")
            print(f"    Event:    {break_issue.get('event_id')}")
            print(f"    Position: {break_issue.get('position')}")
            print(f"  RESULT: CHAIN BREAK CORRECTLY DETECTED")
            print(f"{'='*60}\n")
            
        finally:
            kernel.shutdown(timeout=5)
    
    # ==========================================
    # K2.6: Hash Lookup by Value
    # ==========================================
    
    def test_k2_6_hash_lookup(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K2.6: Events can be retrieved by their hash value.
        
        THEOREM: get_event_by_hash(hash) returns the event whose 
        computed hash equals the given hash.
        
        PROOF:
        - P1: Write event and compute its hash
        - P2: Retrieve event by hash
        - P3: Retrieved event matches original
        - P4: Retrieved event's computed hash matches lookup hash
        """
        # === P1: Write event and compute hash ===
        content = {"type": "hash_lookup_proof", "data": uuid.uuid4().hex}
        event_id = f"k2-6-lookup-{uuid.uuid4().hex[:8]}"
        
        original_event = Event(
            event_id=event_id,
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k2-6-session",
            content=content,
            timestamp=time.time(),
            created_at=time.time(),
        )
        
        expected_hash = original_event.compute_hash()
        kernel.write_event(test_workspace, test_user, original_event)
        
        assert wait_for_migration(kernel, 1, timeout=15), "Migration timeout"
        
        # === P2: Retrieve by hash ===
        retrieved = kernel.get_event_by_hash(expected_hash)
        
        assert retrieved is not None, (
            f"P2 VIOLATED: Event not found by hash {expected_hash[:16]}..."
        )
        
        # === P3: Verify match ===
        assert retrieved.event_id == original_event.event_id, (
            f"P3 VIOLATED: Wrong event returned!\n"
            f"  Expected event_id: {original_event.event_id}\n"
            f"  Got event_id: {retrieved.event_id}"
        )
        
        assert retrieved.content == original_event.content, (
            f"P3 VIOLATED: Content mismatch"
        )
        
        # === P4: Verify hash consistency ===
        retrieved_hash = retrieved.compute_hash()
        
        assert retrieved_hash == expected_hash, (
            f"P4 VIOLATED: Retrieved event's hash doesn't match lookup hash!\n"
            f"  Lookup hash:    {expected_hash}\n"
            f"  Retrieved hash: {retrieved_hash}"
        )
        
        print(f"\n{'='*60}")
        print(f"K2.6 PROOF SUMMARY: Hash Lookup")
        print(f"{'='*60}")
        print(f"  Event ID:        {event_id}")
        print(f"  Expected hash:   {expected_hash}")
        print(f"  Retrieved hash:  {retrieved_hash}")
        print(f"  Hashes match:    {retrieved_hash == expected_hash}")
        print(f"  Content match:   {retrieved.content == original_event.content}")
        print(f"  RESULT: HASH LOOKUP VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.7: Chain Walking (Backward Traversal)
    # ==========================================
    
    def test_k2_7_chain_walking(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K2.7: Hash chain can be walked backward via previous_hash links.
        
        THEOREM: Starting from event[N], following previous_hash links
        visits all events in reverse order until reaching genesis.
        
        PROOF:
        - P1: Create chain of N events
        - P2: Start at last event, walk backward via previous_hash
        - P3: Visit exactly N events in reverse order
        - P4: Final event has previous_hash=None (genesis)
        """
        N = 10
        base_time = time.time()
        
        # === P1: Create chain ===
        for i in range(N):
            event = Event(
                event_id=f"k2-7-walk-{i:04d}",
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k2-7-session",
                content={"index": i},
                timestamp=base_time + i,
                created_at=time.time(),
            )
            kernel.write_event(test_workspace, test_user, event)
        
        assert wait_for_migration(kernel, N, timeout=30), "Migration timeout"
        
        # Get last event
        all_events = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=N + 10
        )
        sorted_events = sorted(all_events, key=lambda e: e.timestamp)
        last_event = sorted_events[-1]
        
        # === P2: Walk backward ===
        walk_path: List[Dict[str, Any]] = []
        current = last_event
        visited_ids = set()
        max_steps = N + 5  # Safety limit
        
        while current is not None and len(walk_path) < max_steps:
            walk_path.append({
                "event_id": current.event_id,
                "index": current.content.get("index"),
                "previous_hash": current.previous_hash[:16] + "..." if current.previous_hash else None,
            })
            visited_ids.add(current.event_id)
            
            if current.previous_hash is None:
                break
            
            current = kernel.get_event_by_hash(current.previous_hash)
        
        # === P3: Verify we visited all N events ===
        assert len(walk_path) == N, (
            f"P3 VIOLATED: Expected to visit {N} events, visited {len(walk_path)}\n"
            f"  Walk path: {[w['event_id'] for w in walk_path]}"
        )
        
        # Verify reverse order
        indices = [w["index"] for w in walk_path]
        expected_indices = list(range(N - 1, -1, -1))
        
        assert indices == expected_indices, (
            f"P3 VIOLATED: Events not in reverse order!\n"
            f"  Expected indices: {expected_indices}\n"
            f"  Got indices: {indices}"
        )
        
        # === P4: Verify genesis ===
        genesis = walk_path[-1]
        
        assert genesis["previous_hash"] is None, (
            f"P4 VIOLATED: Genesis event should have previous_hash=None, "
            f"got {genesis['previous_hash']}"
        )
        
        assert genesis["index"] == 0, (
            f"P4 VIOLATED: Genesis should be index 0, got {genesis['index']}"
        )
        
        print(f"\n{'='*60}")
        print(f"K2.7 PROOF SUMMARY: Chain Walking")
        print(f"{'='*60}")
        print(f"  Chain length:     {N}")
        print(f"  Events visited:   {len(walk_path)}")
        print(f"  Walk order:       {indices}")
        print(f"  Genesis event:    {genesis['event_id']}")
        print(f"  Genesis prev:     {genesis['previous_hash']}")
        print(f"  RESULT: CHAIN WALKING VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.8: Hash Field Sensitivity
    # ==========================================
    
    def test_k2_8_hash_field_sensitivity(
        self,
        test_workspace: str,
        test_user: str,
    ):
        """
        K2.8: Hash changes when any critical field changes.
        
        THEOREM: Changing any of {content, workspace_id, user_id, session_id, 
        timestamp} produces a different hash.
        
        PROOF:
        - P1: Create base event and compute hash
        - P2: For each critical field, modify it and compute new hash
        - P3: Each modified hash differs from base hash
        - P4: previous_hash does NOT affect hash (correct design)
        """
        # === P1: Create base event ===
        base_event = Event(
            event_id="k2-8-sensitivity-base",
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k2-8-session",
            content={"type": "base", "value": 1},
            timestamp=1000000.0,
            created_at=1000000.0,
            previous_hash="abc123",
        )
        
        base_hash = base_event.compute_hash()
        
        # === P2 & P3: Test each field ===
        field_tests: List[Dict[str, Any]] = []
        
        # Test: content
        modified = Event(
            event_id=base_event.event_id,
            workspace_id=base_event.workspace_id,
            user_id=base_event.user_id,
            session_id=base_event.session_id,
            content={"type": "base", "value": 2},  # Changed
            timestamp=base_event.timestamp,
            created_at=base_event.created_at,
            previous_hash=base_event.previous_hash,
        )
        new_hash = modified.compute_hash()
        field_tests.append({
            "field": "content",
            "base_hash": base_hash[:16] + "...",
            "new_hash": new_hash[:16] + "...",
            "changed": new_hash != base_hash,
        })
        assert new_hash != base_hash, "content change should produce different hash"
        
        # Test: workspace_id
        modified = Event(
            event_id=base_event.event_id,
            workspace_id="different-workspace",  # Changed
            user_id=base_event.user_id,
            session_id=base_event.session_id,
            content=base_event.content,
            timestamp=base_event.timestamp,
            created_at=base_event.created_at,
            previous_hash=base_event.previous_hash,
        )
        new_hash = modified.compute_hash()
        field_tests.append({
            "field": "workspace_id",
            "base_hash": base_hash[:16] + "...",
            "new_hash": new_hash[:16] + "...",
            "changed": new_hash != base_hash,
        })
        assert new_hash != base_hash, "workspace_id change should produce different hash"
        
        # Test: user_id
        modified = Event(
            event_id=base_event.event_id,
            workspace_id=base_event.workspace_id,
            user_id="different-user",  # Changed
            session_id=base_event.session_id,
            content=base_event.content,
            timestamp=base_event.timestamp,
            created_at=base_event.created_at,
            previous_hash=base_event.previous_hash,
        )
        new_hash = modified.compute_hash()
        field_tests.append({
            "field": "user_id",
            "base_hash": base_hash[:16] + "...",
            "new_hash": new_hash[:16] + "...",
            "changed": new_hash != base_hash,
        })
        assert new_hash != base_hash, "user_id change should produce different hash"
        
        # Test: session_id
        modified = Event(
            event_id=base_event.event_id,
            workspace_id=base_event.workspace_id,
            user_id=base_event.user_id,
            session_id="different-session",  # Changed
            content=base_event.content,
            timestamp=base_event.timestamp,
            created_at=base_event.created_at,
            previous_hash=base_event.previous_hash,
        )
        new_hash = modified.compute_hash()
        field_tests.append({
            "field": "session_id",
            "base_hash": base_hash[:16] + "...",
            "new_hash": new_hash[:16] + "...",
            "changed": new_hash != base_hash,
        })
        assert new_hash != base_hash, "session_id change should produce different hash"
        
        # Test: timestamp
        modified = Event(
            event_id=base_event.event_id,
            workspace_id=base_event.workspace_id,
            user_id=base_event.user_id,
            session_id=base_event.session_id,
            content=base_event.content,
            timestamp=2000000.0,  # Changed
            created_at=base_event.created_at,
            previous_hash=base_event.previous_hash,
        )
        new_hash = modified.compute_hash()
        field_tests.append({
            "field": "timestamp",
            "base_hash": base_hash[:16] + "...",
            "new_hash": new_hash[:16] + "...",
            "changed": new_hash != base_hash,
        })
        assert new_hash != base_hash, "timestamp change should produce different hash"
        
        # === P4: Verify previous_hash does NOT affect hash ===
        modified = Event(
            event_id=base_event.event_id,
            workspace_id=base_event.workspace_id,
            user_id=base_event.user_id,
            session_id=base_event.session_id,
            content=base_event.content,
            timestamp=base_event.timestamp,
            created_at=base_event.created_at,
            previous_hash="different_previous_hash",  # Changed
        )
        new_hash = modified.compute_hash()
        field_tests.append({
            "field": "previous_hash",
            "base_hash": base_hash[:16] + "...",
            "new_hash": new_hash[:16] + "...",
            "changed": new_hash != base_hash,
            "note": "Should NOT change (correct design)",
        })
        assert new_hash == base_hash, (
            "previous_hash should NOT affect hash computation (hash-chain design)"
        )
        
        print(f"\n{'='*60}")
        print(f"K2.8 PROOF SUMMARY: Hash Field Sensitivity")
        print(f"{'='*60}")
        print(f"  Base hash: {base_hash}")
        print(f"  Field sensitivity tests:")
        for test in field_tests:
            status = "✓ changed" if test["changed"] else "✗ unchanged"
            note = f" ({test['note']})" if 'note' in test else ""
            print(f"    {test['field']:15s}: {status}{note}")
        print(f"  RESULT: HASH FIELD SENSITIVITY VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K2.9: Auto-Repair Guarantee (Security Feature)
    # ==========================================
    
    def test_k2_9_auto_repair_guarantee(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K2.9: Kernel auto-repairs broken chains when enable_hash_chain=True.
        
        THEOREM: When enable_hash_chain=True, even if application provides 
        incorrect previous_hash, kernel overwrites with correct value.
        
        PROOF:
        - P1: Write first event, compute its hash
        - P2: Write second event with WRONG previous_hash
        - P3: Verify kernel stored CORRECT previous_hash (not our wrong one)
        - P4: Verify chain is valid despite our attempt to break it
        
        This is a SECURITY FEATURE - applications cannot break the chain.
        """
        base_time = time.time()
        
        # === P1: Write first event ===
        event1 = Event(
            event_id=f"k2-9-repair-0",
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k2-9-session",
            content={"index": 0},
            timestamp=base_time,
            created_at=time.time(),
        )
        kernel.write_event(test_workspace, test_user, event1)
        
        correct_hash = event1.compute_hash()
        
        # === P2: Write second event with WRONG previous_hash ===
        wrong_hash = "0" * 64  # Intentionally wrong
        
        event2 = Event(
            event_id=f"k2-9-repair-1",
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k2-9-session",
            content={"index": 1},
            timestamp=base_time + 1,
            created_at=time.time(),
            previous_hash=wrong_hash,  # We're submitting wrong hash
        )
        
        # Verify we're actually submitting a wrong hash
        assert event2.previous_hash == wrong_hash, "Sanity check: we're submitting wrong hash"
        assert event2.previous_hash != correct_hash, "Sanity check: wrong != correct"
        
        kernel.write_event(test_workspace, test_user, event2)
        
        # Wait for migration
        assert wait_for_migration(kernel, 2, timeout=15), "Migration timeout"
        
        # === P3: Verify kernel stored CORRECT hash ===
        stored_event2 = kernel.get_event(f"k2-9-repair-1")
        
        assert stored_event2 is not None, "Event 2 should exist"
        
        assert stored_event2.previous_hash != wrong_hash, (
            f"P3 VIOLATED: Kernel stored our wrong hash!\n"
            f"  Wrong hash we submitted: {wrong_hash[:16]}...\n"
            f"  Stored previous_hash:    {stored_event2.previous_hash[:16] if stored_event2.previous_hash else None}..."
        )
        
        assert stored_event2.previous_hash == correct_hash, (
            f"P3 VIOLATED: Kernel didn't store correct hash!\n"
            f"  Correct hash:         {correct_hash[:16]}...\n"
            f"  Stored previous_hash: {stored_event2.previous_hash[:16] if stored_event2.previous_hash else None}..."
        )
        
        # === P4: Verify chain is valid ===
        result = kernel.verify_hash_chain(test_workspace, test_user)
        
        assert result['valid'] == True, (
            f"P4 VIOLATED: Chain should be valid after auto-repair!\n"
            f"  Result: {result}"
        )
        
        assert result['gaps'] == 0, f"P4 VIOLATED: No gaps expected, got {result['gaps']}"
        
        print(f"\n{'='*60}")
        print(f"K2.9 PROOF SUMMARY: Auto-Repair Guarantee")
        print(f"{'='*60}")
        print(f"  Event 1 hash:           {correct_hash}")
        print(f"  Wrong hash submitted:   {wrong_hash}")
        print(f"  Stored previous_hash:   {stored_event2.previous_hash}")
        print(f"  Kernel overwrote:       {stored_event2.previous_hash != wrong_hash}")
        print(f"  Kernel used correct:    {stored_event2.previous_hash == correct_hash}")
        print(f"  Chain valid:            {result['valid']}")
        print(f"  RESULT: AUTO-REPAIR GUARANTEE VERIFIED")
        print(f"  SECURITY: Applications cannot break the hash chain")
        print(f"{'='*60}\n")


# ==========================================
# PROOF SUMMARY
# ==========================================
"""
K2 Test Suite: Hash-Chain Integrity

Theorems Proven:
- K2.1: Hash computation is deterministic
- K2.2: Different events produce different hashes (collision resistance)
- K2.3: Kernel auto-links events in hash chain
- K2.4: verify_hash_chain() validates chain integrity
- K2.5: verify_hash_chain() detects broken chains
- K2.6: Events can be retrieved by hash value
- K2.7: Chain can be walked backward via previous_hash
- K2.8: Hash changes when critical fields change (but not previous_hash)
- K2.9: Kernel auto-repairs broken chains (security guarantee)

Each test provides:
- Numbered proof steps (P1, P2, P3, P4)
- Actual hash values shown
- Mechanism demonstration
- Proof summary output

Run with: pytest -v -s prod_test/layer1_kernel/test_k2_hash_chain.py
"""
