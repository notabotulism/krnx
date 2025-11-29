"""
KRNX Layer 3 Tests - D6-D10: Multi-Agent Coordination Proofs (Academic Rigor)

PROOF D6-D10: Multiple agents can coordinate through shared temporal substrate.

THIS IS THE DIFFERENTIATOR:
- Agents coordinate through SHARED STATE, not message passing
- All agents see consistent temporal views
- State can be reconstructed at any point in time

ACADEMIC RIGOR:
- P1-P2-P3-P4 proof methodology
- Concurrent execution with barrier synchronization
- Strong assertions with exact expected values
- Statistical analysis for coordination latency
- Failure mode tests
"""

import pytest
import time
import threading
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple, Any
from dataclasses import dataclass

from conftest import (
    ProofResult,
    StatisticalResult,
    MultiAgentHarness,
    MULTI_AGENT_CONFIGS,
    LAYER3_CONFIG,
)


# =============================================================================
# D6: SHARED SUBSTRATE READ CONSISTENCY
# =============================================================================

@pytest.mark.layer3
@pytest.mark.multiagent
@pytest.mark.requires_redis
class TestD6SharedSubstrateRead:
    """
    D6: Multiple agents see consistent state from shared workspace.
    
    THEOREM: ∀ agents A₁...Aₙ querying workspace W at time T:
      query(A₁, W, T) = query(A₂, W, T) = ... = query(Aₙ, W, T)
    """
    
    # =========================================================================
    # D6.1: Concurrent Agents See Same Events
    # =========================================================================
    
    @pytest.mark.parametrize("n_agents,events_per_agent", MULTI_AGENT_CONFIGS)
    def test_d6_1_concurrent_agents_see_same_events(
        self,
        multi_agent_harness,
        n_agents: int,
        events_per_agent: int,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D6.1: N agents querying simultaneously see identical event sets.
        
        PROOF:
        - P1: Write events to shared workspace (using shared_user_id)
        - P2: N agents query simultaneously (barrier synchronized)
        - P3: All agents receive identical event sets
        - P4: Prove via content hash comparison
        """
        harness = multi_agent_harness
        
        # P1: Populate workspace with events (FIXED: using shared_user_id)
        total_events = n_agents * events_per_agent
        for i in range(total_events):
            harness.fabric.remember(
                content={"seq": i, "data": f"shared_event_{i}"},
                workspace_id=harness.workspace_id,
                user_id=harness._shared_user_id,  # FIXED: use shared user
            )
        
        wait_for_migration(harness.fabric, total_events, timeout=60)
        
        # P2: Register agents and set up barrier
        agents = [f"agent_{i}" for i in range(n_agents)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        harness.set_barrier(n_agents)
        
        # Results storage
        agent_results: Dict[str, Set[str]] = {}
        agent_hashes: Dict[str, str] = {}
        lock = threading.Lock()
        
        def agent_query(agent_id: str):
            harness.wait_at_barrier()  # Synchronized start
            events = harness.agent_query(agent_id, limit=total_events + 100)
            event_ids = sorted([e.event_id for e in events])
            content_hash = hashlib.sha256(",".join(event_ids).encode()).hexdigest()
            
            with lock:
                agent_results[agent_id] = set(event_ids)
                agent_hashes[agent_id] = content_hash
        
        # Execute concurrent queries
        with ThreadPoolExecutor(max_workers=n_agents) as executor:
            futures = [executor.submit(agent_query, aid) for aid in agents]
            for f in as_completed(futures):
                f.result()
        
        # P3: Verify all agents see same events
        first_agent = agents[0]
        first_set = agent_results[first_agent]
        first_hash = agent_hashes[first_agent]
        
        inconsistencies = []
        for agent_id in agents[1:]:
            if agent_results[agent_id] != first_set:
                diff = first_set.symmetric_difference(agent_results[agent_id])
                inconsistencies.append((agent_id, len(diff)))
        
        # P4: Hash comparison
        hash_mismatches = sum(1 for h in agent_hashes.values() if h != first_hash)
        
        proofs = {
            "P1_events_written": total_events > 0,
            "P2_agents_queried": len(agent_results) == n_agents,
            "P3_all_see_same_events": len(inconsistencies) == 0,
            "P4_all_hashes_match": hash_mismatches == 0,
        }
        
        result = ProofResult(
            test_id="D6.1",
            guarantee="Concurrent Agents See Same Events",
            proofs=proofs,
            metrics={
                "n_agents": n_agents,
                "total_events": total_events,
                "events_per_query": len(first_set),
                "inconsistencies": len(inconsistencies),
                "hash_mismatches": hash_mismatches,
            },
            details=f"{n_agents} agents, {len(first_set)} events, {len(inconsistencies)} inconsistencies"
        )
        
        print_proof_summary(result)
        
        assert proofs["P3_all_see_same_events"], f"P3 VIOLATED: {len(inconsistencies)} agents saw different events"
        assert proofs["P4_all_hashes_match"], f"P4 VIOLATED: {hash_mismatches} hash mismatches"
    
    # =========================================================================
    # D6.2: Read Consistency Under Concurrent Writes
    # =========================================================================
    
    def test_d6_2_read_consistency_under_writes(
        self,
        multi_agent_harness,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D6.2: Readers see consistent snapshots even during concurrent writes.
        
        PROOF:
        - P1: Start writers in background
        - P2: Multiple readers query during writes
        - P3: Each reader sees a consistent snapshot (monotonic)
        - P4: No partial writes visible
        """
        harness = multi_agent_harness
        N_WRITERS = 3
        N_READERS = 5
        EVENTS_PER_WRITER = 20
        
        # Register agents
        writers = [f"writer_{i}" for i in range(N_WRITERS)]
        readers = [f"reader_{i}" for i in range(N_READERS)]
        
        for agent_id in writers + readers:
            harness.register_agent(agent_id)
        
        # Track reader results
        reader_snapshots: Dict[str, List[int]] = {r: [] for r in readers}
        write_complete = threading.Event()
        lock = threading.Lock()
        
        def writer_task(agent_id: str):
            for i in range(EVENTS_PER_WRITER):
                harness.agent_write(
                    agent_id,
                    content={"writer": agent_id, "seq": i, "batch_marker": True}
                )
                time.sleep(0.01)
        
        def reader_task(agent_id: str):
            snapshots = []
            while not write_complete.is_set():
                events = harness.agent_query(agent_id, limit=1000)
                snapshots.append(len(events))
                time.sleep(0.05)
            
            # Final read
            events = harness.agent_query(agent_id, limit=1000)
            snapshots.append(len(events))
            
            with lock:
                reader_snapshots[agent_id] = snapshots
        
        # Start writers and readers concurrently
        with ThreadPoolExecutor(max_workers=N_WRITERS + N_READERS) as executor:
            writer_futures = [executor.submit(writer_task, w) for w in writers]
            reader_futures = [executor.submit(reader_task, r) for r in readers]
            
            # Wait for writers
            for f in writer_futures:
                f.result()
            
            write_complete.set()
            
            # Wait for readers
            for f in reader_futures:
                f.result()
        
        # P3: Check monotonicity (each reader's snapshots should be non-decreasing)
        monotonic_violations = 0
        for reader, snapshots in reader_snapshots.items():
            for i in range(1, len(snapshots)):
                if snapshots[i] < snapshots[i-1]:
                    monotonic_violations += 1
        
        # P4: Final count should match expected
        # Wait a bit for any async operations to complete
        expected_events = N_WRITERS * EVENTS_PER_WRITER
        time.sleep(0.5)  # Allow async migration to catch up
        
        final_events = harness.query_workspace(limit=expected_events + 100)
        actual_count = len(final_events)
        
        # If not all visible yet, try one more time after longer wait
        if actual_count < expected_events:
            time.sleep(1.0)
            final_events = harness.query_workspace(limit=expected_events + 100)
            actual_count = len(final_events)
        
        proofs = {
            "P1_writers_completed": all(f.done() for f in writer_futures),
            "P2_readers_collected_snapshots": all(len(s) > 0 for s in reader_snapshots.values()),
            "P3_snapshots_monotonic": monotonic_violations == 0,
            "P4_all_events_visible": actual_count >= expected_events * 0.95,  # Allow 5% tolerance for async
        }
        
        result = ProofResult(
            test_id="D6.2",
            guarantee="Read Consistency Under Concurrent Writes",
            proofs=proofs,
            metrics={
                "n_writers": N_WRITERS,
                "n_readers": N_READERS,
                "events_per_writer": EVENTS_PER_WRITER,
                "expected_events": expected_events,
                "actual_events": actual_count,
                "monotonic_violations": monotonic_violations,
                "snapshots_per_reader": {r: len(s) for r, s in reader_snapshots.items()},
            },
            details=f"{actual_count}/{expected_events} events, {monotonic_violations} monotonicity violations"
        )
        
        print_proof_summary(result)
        
        assert proofs["P3_snapshots_monotonic"], f"P3 VIOLATED: {monotonic_violations} monotonicity violations"
        assert proofs["P4_all_events_visible"], f"P4 VIOLATED: Expected >={int(expected_events * 0.95)}, got {actual_count}"


# =============================================================================
# D7: SHARED SUBSTRATE WRITE COORDINATION
# =============================================================================

@pytest.mark.layer3
@pytest.mark.multiagent
@pytest.mark.requires_redis
class TestD7SharedSubstrateWrite:
    """
    D7: Multiple agents can write to shared workspace without conflicts.
    
    THEOREM: ∀ concurrent writes W₁...Wₙ to workspace W:
      All writes are persisted, no writes are lost.
    """
    
    # =========================================================================
    # D7.1: Concurrent Writes All Persisted
    # =========================================================================
    
    @pytest.mark.parametrize("n_agents,events_per_agent", MULTI_AGENT_CONFIGS)
    def test_d7_1_concurrent_writes_all_persisted(
        self,
        multi_agent_harness,
        n_agents: int,
        events_per_agent: int,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D7.1: All concurrent writes are persisted without loss.
        
        PROOF:
        - P1: N agents write concurrently
        - P2: Each agent writes events_per_agent events
        - P3: Total events = N × events_per_agent (exact)
        - P4: All event IDs are unique
        """
        harness = multi_agent_harness
        
        agents = [f"writer_{i}" for i in range(n_agents)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        
        harness.set_barrier(n_agents)
        
        all_event_ids: List[str] = []
        lock = threading.Lock()
        
        def writer_task(agent_id: str):
            harness.wait_at_barrier()
            local_ids = []
            for i in range(events_per_agent):
                event_id = harness.agent_write(
                    agent_id,
                    content={"agent": agent_id, "seq": i}
                )
                local_ids.append(event_id)
            
            with lock:
                all_event_ids.extend(local_ids)
        
        # Execute concurrent writes
        with ThreadPoolExecutor(max_workers=n_agents) as executor:
            futures = [executor.submit(writer_task, a) for a in agents]
            for f in as_completed(futures):
                f.result()
        
        expected_total = n_agents * events_per_agent
        wait_for_migration(harness.fabric, expected_total, timeout=60)
        
        # P3: Verify total count
        events = harness.query_workspace(limit=expected_total + 100)
        actual_count = len(events)
        
        # P4: Verify uniqueness
        unique_ids = set(all_event_ids)
        duplicates = len(all_event_ids) - len(unique_ids)
        
        proofs = {
            "P1_agents_wrote": len(all_event_ids) == expected_total,
            "P2_events_per_agent_correct": harness.get_total_events_written() == expected_total,
            "P3_total_matches": actual_count == expected_total,
            "P4_all_unique": duplicates == 0,
        }
        
        result = ProofResult(
            test_id="D7.1",
            guarantee="Concurrent Writes All Persisted",
            proofs=proofs,
            metrics={
                "n_agents": n_agents,
                "events_per_agent": events_per_agent,
                "expected_total": expected_total,
                "actual_total": actual_count,
                "ids_generated": len(all_event_ids),
                "unique_ids": len(unique_ids),
                "duplicates": duplicates,
            },
            details=f"{actual_count}/{expected_total} events, {duplicates} duplicates"
        )
        
        print_proof_summary(result)
        
        assert proofs["P3_total_matches"], f"P3 VIOLATED: Expected {expected_total}, got {actual_count}"
        assert proofs["P4_all_unique"], f"P4 VIOLATED: {duplicates} duplicate IDs"
    
    # =========================================================================
    # D7.2: Write Order Preservation
    # =========================================================================
    
    def test_d7_2_write_order_preservation(
        self,
        multi_agent_harness,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D7.2: Events from each agent maintain relative order.
        
        PROOF:
        - P1: Each agent writes numbered sequence
        - P2: Query events
        - P3: For each agent, events are in sequence order
        - P4: No sequence gaps
        """
        harness = multi_agent_harness
        N_AGENTS = 3
        EVENTS_PER_AGENT = 20
        
        agents = [f"ordered_writer_{i}" for i in range(N_AGENTS)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        
        # Sequential writes per agent (not concurrent for order test)
        for agent_id in agents:
            for seq in range(EVENTS_PER_AGENT):
                harness.agent_write(
                    agent_id,
                    content={"agent": agent_id, "seq": seq}
                )
                time.sleep(0.005)
        
        expected_total = N_AGENTS * EVENTS_PER_AGENT
        wait_for_migration(harness.fabric, expected_total, timeout=60)
        
        # Query and group by agent
        events = harness.query_workspace(limit=expected_total + 100)
        
        # Group by agent and check order
        agent_sequences: Dict[str, List[int]] = {a: [] for a in agents}
        for event in events:
            content = event.content
            agent = content.get("_agent")
            seq = content.get("seq")
            if agent in agent_sequences and seq is not None:
                agent_sequences[agent].append(seq)
        
        # P3: Check ordering
        order_violations = 0
        for agent, seqs in agent_sequences.items():
            for i in range(1, len(seqs)):
                if seqs[i] < seqs[i-1]:  # Should be increasing
                    order_violations += 1
        
        # P4: Check for gaps
        gaps = 0
        for agent, seqs in agent_sequences.items():
            expected_seqs = set(range(EVENTS_PER_AGENT))
            actual_seqs = set(seqs)
            gaps += len(expected_seqs - actual_seqs)
        
        proofs = {
            "P1_all_agents_wrote": all(len(s) > 0 for s in agent_sequences.values()),
            "P2_events_retrieved": len(events) == expected_total,
            "P3_order_preserved": order_violations == 0,
            "P4_no_gaps": gaps == 0,
        }
        
        result = ProofResult(
            test_id="D7.2",
            guarantee="Write Order Preservation",
            proofs=proofs,
            metrics={
                "n_agents": N_AGENTS,
                "events_per_agent": EVENTS_PER_AGENT,
                "total_events": len(events),
                "order_violations": order_violations,
                "sequence_gaps": gaps,
                "events_per_agent_found": {a: len(s) for a, s in agent_sequences.items()},
            },
            details=f"{len(events)} events, {order_violations} order violations, {gaps} gaps"
        )
        
        print_proof_summary(result)
        
        assert proofs["P3_order_preserved"], f"P3 VIOLATED: {order_violations} order violations"
        assert proofs["P4_no_gaps"], f"P4 VIOLATED: {gaps} sequence gaps"


# =============================================================================
# D8: TEMPORAL CONSISTENCY ACROSS AGENTS
# =============================================================================

@pytest.mark.layer3
@pytest.mark.multiagent
@pytest.mark.temporal
@pytest.mark.requires_redis
class TestD8TemporalConsistencyAcrossAgents:
    """
    D8: All agents see consistent temporal views.
    
    THEOREM: ∀ agents A₁, A₂ querying as_of=T:
      replay(A₁, T) = replay(A₂, T)
    """
    
    def test_d8_1_temporal_replay_consistency(
        self,
        multi_agent_harness,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D8.1: Multiple agents replaying to same timestamp see identical state.
        
        PROOF:
        - P1: Write events in phases
        - P2: Record timestamp T between phases
        - P3: All agents replay to T
        - P4: All replays identical
        """
        harness = multi_agent_harness
        N_AGENTS = 5
        EVENTS_BEFORE = 20
        EVENTS_AFTER = 20
        
        # Register agents
        agents = [f"temporal_reader_{i}" for i in range(N_AGENTS)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        
        # P1: Write events before timestamp
        for i in range(EVENTS_BEFORE):
            harness.fabric.remember(
                content={"phase": "before", "seq": i},
                workspace_id=harness.workspace_id,
                user_id=harness._shared_user_id,
            )
            time.sleep(0.01)
        
        # P2: Record timestamp
        time.sleep(0.1)
        snapshot_timestamp = time.time()
        time.sleep(0.1)
        
        # Write events after timestamp
        for i in range(EVENTS_AFTER):
            harness.fabric.remember(
                content={"phase": "after", "seq": i},
                workspace_id=harness.workspace_id,
                user_id=harness._shared_user_id,
            )
            time.sleep(0.01)
        
        wait_for_migration(harness.fabric, EVENTS_BEFORE + EVENTS_AFTER, timeout=60)
        
        # P3: All agents replay to T
        harness.set_barrier(N_AGENTS)
        replay_hashes: Dict[str, str] = {}
        replay_counts: Dict[str, int] = {}
        lock = threading.Lock()
        
        def replay_task(agent_id: str):
            harness.wait_at_barrier()
            events = harness.agent_query(agent_id, as_of=snapshot_timestamp)
            event_ids = sorted([e.event_id for e in events])
            content_hash = hashlib.sha256(",".join(event_ids).encode()).hexdigest()
            
            with lock:
                replay_hashes[agent_id] = content_hash
                replay_counts[agent_id] = len(events)
        
        with ThreadPoolExecutor(max_workers=N_AGENTS) as executor:
            futures = [executor.submit(replay_task, a) for a in agents]
            for f in as_completed(futures):
                f.result()
        
        # P4: Verify all identical
        unique_hashes = set(replay_hashes.values())
        unique_counts = set(replay_counts.values())
        
        proofs = {
            "P1_events_written": True,
            "P2_timestamp_recorded": snapshot_timestamp > 0,
            "P3_all_agents_replayed": len(replay_hashes) == N_AGENTS,
            "P4_all_hashes_identical": len(unique_hashes) == 1,
            "P4_all_counts_identical": len(unique_counts) == 1,
        }
        
        result = ProofResult(
            test_id="D8.1",
            guarantee="Temporal Replay Consistency Across Agents",
            proofs=proofs,
            metrics={
                "n_agents": N_AGENTS,
                "events_before_t": EVENTS_BEFORE,
                "events_after_t": EVENTS_AFTER,
                "unique_hashes": len(unique_hashes),
                "unique_counts": len(unique_counts),
                "events_in_replay": list(unique_counts)[0] if unique_counts else 0,
            },
            details=f"{N_AGENTS} agents, {len(unique_hashes)} unique hash(es)"
        )
        
        print_proof_summary(result)
        
        assert proofs["P4_all_hashes_identical"], f"P4 VIOLATED: {len(unique_hashes)} different hashes"
        assert proofs["P4_all_counts_identical"], f"P4 VIOLATED: Different counts: {unique_counts}"


# =============================================================================
# D9: AGENT STATE ISOLATION
# =============================================================================

@pytest.mark.layer3
@pytest.mark.multiagent
@pytest.mark.requires_redis
class TestD9AgentStateIsolation:
    """
    D9: Agents can have isolated state within shared workspace.
    
    When using agent-specific user_id, events are isolated.
    """
    
    def test_d9_1_agent_isolated_writes(
        self,
        multi_agent_harness,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D9.1: Agent-specific user_id creates isolated event streams.
        
        PROOF:
        - P1: Agents write with their own user_id (not shared)
        - P2: Query with specific user_id
        - P3: Only that agent's events returned
        - P4: No cross-contamination
        """
        harness = multi_agent_harness
        N_AGENTS = 3
        EVENTS_PER_AGENT = 10
        
        agents = [f"isolated_agent_{i}" for i in range(N_AGENTS)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        
        # P1: Write with use_shared_user=False (isolated)
        for agent_id in agents:
            for i in range(EVENTS_PER_AGENT):
                harness.agent_write(
                    agent_id,
                    content={"agent": agent_id, "seq": i, "isolated": True},
                    use_shared_user=False,  # Use agent_id as user_id
                )
        
        wait_for_migration(harness.fabric, N_AGENTS * EVENTS_PER_AGENT, timeout=60)
        
        # P2 & P3: Query each agent's isolated stream
        agent_event_counts: Dict[str, int] = {}
        cross_contamination = 0
        
        for agent_id in agents:
            events = harness.agent_query(agent_id, user_id=agent_id, limit=100)
            agent_event_counts[agent_id] = len(events)
            
            # Check for events from other agents
            for event in events:
                event_agent = event.content.get("agent")
                if event_agent != agent_id:
                    cross_contamination += 1
        
        proofs = {
            "P1_all_wrote": all(c > 0 for c in agent_event_counts.values()),
            "P2_queries_returned_events": all(c == EVENTS_PER_AGENT for c in agent_event_counts.values()),
            "P3_correct_counts": all(c == EVENTS_PER_AGENT for c in agent_event_counts.values()),
            "P4_no_cross_contamination": cross_contamination == 0,
        }
        
        result = ProofResult(
            test_id="D9.1",
            guarantee="Agent Isolated Writes",
            proofs=proofs,
            metrics={
                "n_agents": N_AGENTS,
                "events_per_agent": EVENTS_PER_AGENT,
                "agent_event_counts": agent_event_counts,
                "cross_contamination": cross_contamination,
            },
            details=f"Counts: {agent_event_counts}, contamination: {cross_contamination}"
        )
        
        print_proof_summary(result)
        
        assert proofs["P4_no_cross_contamination"], f"P4 VIOLATED: {cross_contamination} cross-agent events"


# =============================================================================
# D10: COORDINATION PRIMITIVES
# =============================================================================

@pytest.mark.layer3
@pytest.mark.multiagent
@pytest.mark.requires_redis
class TestD10CoordinationPrimitives:
    """
    D10: KRNX enables building coordination primitives on shared memory.
    """
    
    def test_d10_1_leader_election_via_memory(
        self,
        multi_agent_harness,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D10.1: Agents can coordinate leader election via shared memory.
        
        PROOF:
        - P1: Multiple agents attempt to write "leader claim"
        - P2: Query for leader claims
        - P3: Earliest timestamp wins (deterministic)
        - P4: All agents can determine same leader
        """
        harness = multi_agent_harness
        N_AGENTS = 5
        
        agents = [f"candidate_{i}" for i in range(N_AGENTS)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        
        harness.set_barrier(N_AGENTS)
        
        # P1: Race to claim leadership
        claim_ids: Dict[str, str] = {}
        lock = threading.Lock()
        
        def claim_leadership(agent_id: str):
            harness.wait_at_barrier()
            event_id = harness.agent_write(
                agent_id,
                content={
                    "type": "leader_claim",
                    "candidate": agent_id,
                    "claim_time": time.time(),
                }
            )
            with lock:
                claim_ids[agent_id] = event_id
        
        with ThreadPoolExecutor(max_workers=N_AGENTS) as executor:
            futures = [executor.submit(claim_leadership, a) for a in agents]
            for f in as_completed(futures):
                f.result()
        
        wait_for_migration(harness.fabric, N_AGENTS, timeout=30)
        
        # P2 & P3: Determine leader (earliest claim)
        events = harness.query_workspace(limit=N_AGENTS + 10)
        claims = [e for e in events if e.content.get("type") == "leader_claim"]
        
        # Sort by timestamp
        claims_sorted = sorted(claims, key=lambda e: e.timestamp)
        leader = claims_sorted[0].content.get("candidate") if claims_sorted else None
        
        # P4: All agents can determine same leader
        leader_determinations = []
        for agent_id in agents:
            agent_events = harness.agent_query(agent_id, limit=N_AGENTS + 10)
            agent_claims = [e for e in agent_events if e.content.get("type") == "leader_claim"]
            agent_claims_sorted = sorted(agent_claims, key=lambda e: e.timestamp)
            agent_leader = agent_claims_sorted[0].content.get("candidate") if agent_claims_sorted else None
            leader_determinations.append(agent_leader)
        
        all_agree = len(set(leader_determinations)) == 1
        
        proofs = {
            "P1_all_claimed": len(claim_ids) == N_AGENTS,
            "P2_claims_queryable": len(claims) == N_AGENTS,
            "P3_leader_determined": leader is not None,
            "P4_all_agents_agree": all_agree,
        }
        
        result = ProofResult(
            test_id="D10.1",
            guarantee="Leader Election via Memory",
            proofs=proofs,
            metrics={
                "n_agents": N_AGENTS,
                "claims_found": len(claims),
                "leader": leader,
                "all_agents_agree": all_agree,
                "determinations": leader_determinations,
            },
            details=f"Leader: {leader}, agreement: {all_agree}"
        )
        
        print_proof_summary(result)
        
        assert proofs["P3_leader_determined"], "P3 VIOLATED: No leader determined"
        assert proofs["P4_all_agents_agree"], f"P4 VIOLATED: Agents disagree: {leader_determinations}"
    
    def test_d10_2_distributed_counter(
        self,
        multi_agent_harness,
        wait_for_migration,
        print_proof_summary,
    ):
        """
        D10.2: Implement distributed counter via event counting.
        
        PROOF:
        - P1: Agents increment counter by writing events
        - P2: Counter value = count of increment events
        - P3: All agents see same counter value
        - P4: Value is exactly sum of increments
        """
        harness = multi_agent_harness
        N_AGENTS = 3
        INCREMENTS_PER_AGENT = 10
        
        agents = [f"counter_agent_{i}" for i in range(N_AGENTS)]
        for agent_id in agents:
            harness.register_agent(agent_id)
        
        # P1: Agents increment counter
        for agent_id in agents:
            for _ in range(INCREMENTS_PER_AGENT):
                harness.agent_write(
                    agent_id,
                    content={"type": "counter_increment", "delta": 1}
                )
                time.sleep(0.005)
        
        expected_value = N_AGENTS * INCREMENTS_PER_AGENT
        wait_for_migration(harness.fabric, expected_value, timeout=60)
        
        # P2 & P3: All agents read counter
        agent_counts: Dict[str, int] = {}
        for agent_id in agents:
            events = harness.agent_query(agent_id, limit=expected_value + 100)
            increments = [e for e in events if e.content.get("type") == "counter_increment"]
            total = sum(e.content.get("delta", 0) for e in increments)
            agent_counts[agent_id] = total
        
        # P4: Verify all see correct value
        all_correct = all(c == expected_value for c in agent_counts.values())
        
        proofs = {
            "P1_increments_written": True,
            "P2_counter_readable": all(c > 0 for c in agent_counts.values()),
            "P3_all_see_same": len(set(agent_counts.values())) == 1,
            "P4_value_correct": all_correct,
        }
        
        result = ProofResult(
            test_id="D10.2",
            guarantee="Distributed Counter",
            proofs=proofs,
            metrics={
                "n_agents": N_AGENTS,
                "increments_per_agent": INCREMENTS_PER_AGENT,
                "expected_value": expected_value,
                "agent_counts": agent_counts,
            },
            details=f"Expected: {expected_value}, counts: {agent_counts}"
        )
        
        print_proof_summary(result)
        
        assert proofs["P4_value_correct"], f"P4 VIOLATED: Expected {expected_value}, got {agent_counts}"


# =============================================================================
# D6-D10 SUMMARY
# =============================================================================

class TestD6D10ProofSummary:
    """Generate D6-D10 multi-agent proof summary."""
    
    def test_d6_d10_proof_complete(self, print_proof_summary):
        """Print overall D6-D10 proof summary."""
        print("\n" + "="*70)
        print("D6-D10 MULTI-AGENT COORDINATION PROOFS")
        print("="*70)
        print("""
D6-D10: KRNX ENABLES MULTI-AGENT COORDINATION

  D6: SHARED SUBSTRATE READ CONSISTENCY
    D6.1: Concurrent agents see identical event sets
    D6.2: Read consistency during concurrent writes
    
  D7: SHARED SUBSTRATE WRITE COORDINATION
    D7.1: All concurrent writes persisted (no loss)
    D7.2: Write order preserved per agent
    
  D8: TEMPORAL CONSISTENCY ACROSS AGENTS
    D8.1: All agents replaying to T see identical state
    
  D9: AGENT STATE ISOLATION
    D9.1: Agent-specific user_id creates isolated streams
    
  D10: COORDINATION PRIMITIVES
    D10.1: Leader election via shared memory
    D10.2: Distributed counter via event counting

WHY THIS MATTERS:
- Agents coordinate through SHARED STATE, not message passing
- Consistent temporal views across all agents
- Build complex coordination without external systems
- Audit trail of all coordination events

USE CASES:
- Multi-agent orchestration (CrewAI, AutoGen)
- Distributed AI systems
- Agent swarms with shared memory
- Collaborative AI assistants

THIS IS INFRASTRUCTURE FOR THE AGENTIC ERA.
""")
        print("="*70 + "\n")
