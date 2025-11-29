"""
K7: TTL and Expiry Semantics

THEOREM: KRNX correctly tracks TTL (time-to-live) and provides accurate
expiry detection. TTL is informational (stored but not auto-deleted).

PROOF STRUCTURE:
Each test follows the format:
- P1 (Precondition): Establish known initial state
- P2 (Operation): Perform the operation under test
- P3 (Postcondition): Verify expected outcome
- P4 (Mechanism): Prove WHY the outcome is correct

Academic rigor requirements:
- Verify TTL values stored exactly
- Test is_expired() with precise timestamps
- Prove boundary conditions
- Show actual TTL values in assertions
"""

import pytest
import time
import uuid
from typing import List, Dict, Any

from chillbot.kernel.models import Event
from chillbot.kernel.controller import KRNXController

from ..config import KERNEL_TEST_CONFIG


@pytest.mark.layer1
@pytest.mark.requires_redis
class TestK7TTLExpiry:
    """
    K7: TTL and Expiry Proofs
    
    These tests prove that KRNX correctly handles TTL:
    - TTL values are stored and retrieved accurately
    - is_expired() correctly identifies expired events
    - Events without TTL never expire
    - Expiry is informational (events not auto-deleted)
    """
    
    # ==========================================
    # K7.1: TTL Value Storage
    # ==========================================
    
    def test_k7_1_ttl_value_storage(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K7.1: TTL values are stored and retrieved exactly.
        
        THEOREM: For event e with ttl_seconds=T, stored(e).ttl_seconds = T.
        
        PROOF:
        - P1: Create event with specific TTL value
        - P2: Store and retrieve
        - P3: Retrieved TTL equals original
        - P4: Verify with multiple TTL values
        """
        test_ttls = [60, 3600, 86400, 604800, None]  # 1min, 1hr, 1day, 1week, none
        
        results: List[Dict[str, Any]] = []
        base_time = time.time()
        
        # === P1 & P2: Create and store events with various TTLs ===
        for i, ttl in enumerate(test_ttls):
            event_id = f"k7-1-ttl-{i:04d}"
            
            event = Event(
                event_id=event_id,
                workspace_id=test_workspace,
                user_id=test_user,
                session_id="k7-1-session",
                content={"index": i, "expected_ttl": ttl},
                timestamp=base_time + i,
                created_at=time.time(),
                ttl_seconds=ttl,
            )
            kernel.write_event(test_workspace, test_user, event)
            
            results.append({
                "event_id": event_id,
                "original_ttl": ttl,
                "retrieved_ttl": None,
                "match": None,
            })
        
        # Wait for migration
        assert wait_for_migration(kernel, len(test_ttls), timeout=30), "Migration timeout"
        
        # === P3 & P4: Retrieve and compare ===
        mismatches = []
        
        for result in results:
            retrieved = kernel.get_event(result["event_id"])
            
            assert retrieved is not None, f"Event {result['event_id']} not found"
            
            result["retrieved_ttl"] = retrieved.ttl_seconds
            result["match"] = result["original_ttl"] == result["retrieved_ttl"]
            
            if not result["match"]:
                mismatches.append(result)
        
        assert len(mismatches) == 0, (
            f"P3 VIOLATED: TTL mismatches:\n" +
            "\n".join(
                f"  {m['event_id']}: original={m['original_ttl']}, retrieved={m['retrieved_ttl']}"
                for m in mismatches
            )
        )
        
        print(f"\n{'='*60}")
        print(f"K7.1 PROOF SUMMARY: TTL Value Storage")
        print(f"{'='*60}")
        print(f"  TTL values tested:    {test_ttls}")
        print(f"  Mismatches:           {len(mismatches)}")
        print(f"  Results:")
        for r in results:
            status = "✓" if r['match'] else "✗"
            print(f"    TTL={r['original_ttl']}: {status}")
        print(f"  RESULT: TTL STORAGE VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.2: is_expired() Correctness
    # ==========================================
    
    def test_k7_2_is_expired_correctness(self):
        """
        K7.2: is_expired() correctly identifies expired events.
        
        THEOREM: is_expired() returns True iff age > ttl_seconds.
        
        PROOF:
        - P1: Create event with known age and TTL
        - P2: age < TTL → is_expired() = False
        - P3: age > TTL → is_expired() = True
        - P4: Test boundary (age ≈ TTL)
        """
        now = time.time()
        
        # === P2: Not expired (age < TTL) ===
        fresh_event = Event(
            event_id="k7-2-fresh",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-2-session",
            content={"type": "fresh"},
            timestamp=now - 30,  # 30 seconds ago
            created_at=now - 30,
            ttl_seconds=3600,  # 1 hour TTL
        )
        
        age_fresh = fresh_event.get_age_seconds()
        ttl_fresh = fresh_event.ttl_seconds
        is_expired_fresh = fresh_event.is_expired()
        
        assert not is_expired_fresh, (
            f"P2 VIOLATED: Fresh event marked as expired.\n"
            f"  Age: {age_fresh:.2f}s\n"
            f"  TTL: {ttl_fresh}s\n"
            f"  Expected: not expired"
        )
        
        # === P3: Expired (age > TTL) ===
        old_event = Event(
            event_id="k7-2-old",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-2-session",
            content={"type": "old"},
            timestamp=now - 7200,  # 2 hours ago
            created_at=now - 7200,
            ttl_seconds=3600,  # 1 hour TTL
        )
        
        age_old = old_event.get_age_seconds()
        ttl_old = old_event.ttl_seconds
        is_expired_old = old_event.is_expired()
        
        assert is_expired_old, (
            f"P3 VIOLATED: Old event not marked as expired.\n"
            f"  Age: {age_old:.2f}s\n"
            f"  TTL: {ttl_old}s\n"
            f"  Expected: expired"
        )
        
        # === P4: Boundary (age ≈ TTL) ===
        # Just under TTL
        boundary_under = Event(
            event_id="k7-2-boundary-under",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-2-session",
            content={"type": "boundary"},
            timestamp=now - 3599,  # 3599 seconds ago
            created_at=now - 3599,
            ttl_seconds=3600,  # 3600 second TTL
        )
        
        assert not boundary_under.is_expired(), (
            "P4 VIOLATED: Event just under TTL marked as expired"
        )
        
        # Just over TTL
        boundary_over = Event(
            event_id="k7-2-boundary-over",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-2-session",
            content={"type": "boundary"},
            timestamp=now - 3601,  # 3601 seconds ago
            created_at=now - 3601,
            ttl_seconds=3600,  # 3600 second TTL
        )
        
        assert boundary_over.is_expired(), (
            "P4 VIOLATED: Event just over TTL not marked as expired"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.2 PROOF SUMMARY: is_expired() Correctness")
        print(f"{'='*60}")
        print(f"  Fresh event (age={age_fresh:.0f}s, TTL={ttl_fresh}s):")
        print(f"    is_expired() = {is_expired_fresh} (expected: False) ✓")
        print(f"  Old event (age={age_old:.0f}s, TTL={ttl_old}s):")
        print(f"    is_expired() = {is_expired_old} (expected: True) ✓")
        print(f"  Boundary tests:")
        print(f"    age=3599s, TTL=3600s: is_expired()={boundary_under.is_expired()} ✓")
        print(f"    age=3601s, TTL=3600s: is_expired()={boundary_over.is_expired()} ✓")
        print(f"  RESULT: is_expired() CORRECTNESS VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.3: No TTL Means Never Expires
    # ==========================================
    
    def test_k7_3_no_ttl_never_expires(self):
        """
        K7.3: Events without TTL never expire.
        
        THEOREM: If ttl_seconds is None, is_expired() = False regardless of age.
        
        PROOF:
        - P1: Create very old event without TTL
        - P2: Call is_expired()
        - P3: Returns False
        - P4: Even extremely old events don't expire without TTL
        """
        now = time.time()
        
        # === P1: Very old event without TTL ===
        year_ago = now - (365 * 24 * 3600)  # 1 year ago
        
        ancient_event = Event(
            event_id="k7-3-ancient",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-3-session",
            content={"type": "ancient"},
            timestamp=year_ago,
            created_at=year_ago,
            ttl_seconds=None,  # No TTL
        )
        
        age_days = ancient_event.get_age_days()
        
        # === P2 & P3: Check expiry ===
        is_expired = ancient_event.is_expired()
        
        assert not is_expired, (
            f"P3 VIOLATED: Event without TTL marked as expired.\n"
            f"  Age: {age_days:.1f} days\n"
            f"  TTL: None\n"
            f"  Expected: never expires"
        )
        
        # === P4: Extremely old event ===
        decade_ago = now - (10 * 365 * 24 * 3600)  # 10 years ago
        
        very_ancient_event = Event(
            event_id="k7-3-very-ancient",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-3-session",
            content={"type": "very_ancient"},
            timestamp=decade_ago,
            created_at=decade_ago,
            ttl_seconds=None,
        )
        
        assert not very_ancient_event.is_expired(), (
            "P4 VIOLATED: 10-year-old event without TTL marked as expired"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.3 PROOF SUMMARY: No TTL Never Expires")
        print(f"{'='*60}")
        print(f"  Event age:            {age_days:.1f} days")
        print(f"  TTL:                  None")
        print(f"  is_expired():         {is_expired}")
        print(f"  10-year-old event:    is_expired()={very_ancient_event.is_expired()}")
        print(f"  RESULT: NO TTL NEVER EXPIRES VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.4: get_age_seconds() Accuracy
    # ==========================================
    
    def test_k7_4_get_age_seconds_accuracy(self):
        """
        K7.4: get_age_seconds() returns accurate age.
        
        THEOREM: get_age_seconds() = current_time - event.timestamp (±1s tolerance).
        
        PROOF:
        - P1: Create event with known timestamp
        - P2: Call get_age_seconds()
        - P3: Compare with expected age
        - P4: Within tolerance
        """
        now = time.time()
        known_offset = 60  # 60 seconds ago
        
        # === P1: Create event with known timestamp ===
        event = Event(
            event_id="k7-4-age-test",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-4-session",
            content={"type": "age_test"},
            timestamp=now - known_offset,
            created_at=now - known_offset,
        )
        
        # === P2: Get age ===
        reported_age = event.get_age_seconds()
        
        # === P3 & P4: Compare with tolerance ===
        expected_age = known_offset
        tolerance = 2.0  # 2 second tolerance for test execution time
        
        age_error = abs(reported_age - expected_age)
        
        assert age_error < tolerance, (
            f"P4 VIOLATED: Age calculation error exceeds tolerance.\n"
            f"  Expected age: ~{expected_age}s\n"
            f"  Reported age: {reported_age:.2f}s\n"
            f"  Error: {age_error:.2f}s\n"
            f"  Tolerance: {tolerance}s"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.4 PROOF SUMMARY: get_age_seconds() Accuracy")
        print(f"{'='*60}")
        print(f"  Known offset:         {known_offset}s")
        print(f"  Reported age:         {reported_age:.2f}s")
        print(f"  Error:                {age_error:.4f}s")
        print(f"  Tolerance:            {tolerance}s")
        print(f"  Within tolerance:     {age_error < tolerance}")
        print(f"  RESULT: AGE CALCULATION VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.5: get_age_days() Accuracy
    # ==========================================
    
    def test_k7_5_get_age_days_accuracy(self):
        """
        K7.5: get_age_days() returns accurate age in days.
        
        THEOREM: get_age_days() = get_age_seconds() / 86400.
        
        PROOF:
        - P1: Create event with known age
        - P2: Compare get_age_days() with get_age_seconds() / 86400
        - P3: Values match exactly
        """
        now = time.time()
        known_days = 3.5  # 3.5 days ago
        known_seconds = known_days * 86400
        
        # === P1: Create event ===
        event = Event(
            event_id="k7-5-days-test",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-5-session",
            content={"type": "days_test"},
            timestamp=now - known_seconds,
            created_at=now - known_seconds,
        )
        
        # === P2: Get both values ===
        age_seconds = event.get_age_seconds()
        age_days = event.get_age_days()
        
        # === P3: Compare ===
        expected_days = age_seconds / 86400
        
        assert abs(age_days - expected_days) < 0.0001, (
            f"P3 VIOLATED: get_age_days() inconsistent with get_age_seconds().\n"
            f"  get_age_seconds(): {age_seconds:.2f}\n"
            f"  get_age_days(): {age_days:.6f}\n"
            f"  Expected days: {expected_days:.6f}"
        )
        
        # Verify approximate correctness
        tolerance_days = 0.01  # ~15 minutes tolerance
        assert abs(age_days - known_days) < tolerance_days, (
            f"P3 VIOLATED: Age in days doesn't match expected.\n"
            f"  Expected: ~{known_days} days\n"
            f"  Got: {age_days:.4f} days"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.5 PROOF SUMMARY: get_age_days() Accuracy")
        print(f"{'='*60}")
        print(f"  Expected age:         {known_days} days")
        print(f"  get_age_seconds():    {age_seconds:.2f}s")
        print(f"  get_age_days():       {age_days:.4f} days")
        print(f"  Computed days:        {expected_days:.4f} days")
        print(f"  RESULT: AGE IN DAYS VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.6: is_recent() Threshold
    # ==========================================
    
    def test_k7_6_is_recent_threshold(self):
        """
        K7.6: is_recent(hours) correctly applies threshold.
        
        THEOREM: is_recent(H) = True iff age < H * 3600.
        
        PROOF:
        - P1: Create events with various ages
        - P2: Test with specific hour threshold
        - P3: Events under threshold return True
        - P4: Events over threshold return False
        """
        now = time.time()
        threshold_hours = 2  # 2 hour threshold
        threshold_seconds = threshold_hours * 3600
        
        # === P1: Create events ===
        # Recent (1 hour ago)
        recent_event = Event(
            event_id="k7-6-recent",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-6-session",
            content={"type": "recent"},
            timestamp=now - 3600,  # 1 hour ago
            created_at=now - 3600,
        )
        
        # Old (3 hours ago)
        old_event = Event(
            event_id="k7-6-old",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-6-session",
            content={"type": "old"},
            timestamp=now - 10800,  # 3 hours ago
            created_at=now - 10800,
        )
        
        # === P2, P3, P4: Test threshold ===
        recent_age = recent_event.get_age_seconds()
        old_age = old_event.get_age_seconds()
        
        is_recent_recent = recent_event.is_recent(threshold_hours)
        is_recent_old = old_event.is_recent(threshold_hours)
        
        assert is_recent_recent, (
            f"P3 VIOLATED: Event under threshold not marked as recent.\n"
            f"  Age: {recent_age:.0f}s\n"
            f"  Threshold: {threshold_seconds}s ({threshold_hours}h)\n"
            f"  Expected: is_recent=True"
        )
        
        assert not is_recent_old, (
            f"P4 VIOLATED: Event over threshold marked as recent.\n"
            f"  Age: {old_age:.0f}s\n"
            f"  Threshold: {threshold_seconds}s ({threshold_hours}h)\n"
            f"  Expected: is_recent=False"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.6 PROOF SUMMARY: is_recent() Threshold")
        print(f"{'='*60}")
        print(f"  Threshold:            {threshold_hours} hours ({threshold_seconds}s)")
        print(f"  Recent event:")
        print(f"    Age:                {recent_age:.0f}s")
        print(f"    is_recent({threshold_hours}):       {is_recent_recent} ✓")
        print(f"  Old event:")
        print(f"    Age:                {old_age:.0f}s")
        print(f"    is_recent({threshold_hours}):       {is_recent_old} ✓")
        print(f"  RESULT: is_recent() THRESHOLD VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.7: should_archive() Threshold
    # ==========================================
    
    def test_k7_7_should_archive_threshold(self):
        """
        K7.7: should_archive(days) correctly applies threshold.
        
        THEOREM: should_archive(D) = True iff age_days > D.
        
        PROOF:
        - P1: Create events with various ages
        - P2: Test with specific day threshold
        - P3: Events under threshold return False
        - P4: Events over threshold return True
        """
        now = time.time()
        threshold_days = 30
        
        # === P1: Create events ===
        # Recent (10 days old)
        recent_event = Event(
            event_id="k7-7-recent",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-7-session",
            content={"type": "recent"},
            timestamp=now - (10 * 86400),  # 10 days ago
            created_at=now - (10 * 86400),
        )
        
        # Old (45 days old)
        old_event = Event(
            event_id="k7-7-old",
            workspace_id="test-ws",
            user_id="test-user",
            session_id="k7-7-session",
            content={"type": "old"},
            timestamp=now - (45 * 86400),  # 45 days ago
            created_at=now - (45 * 86400),
        )
        
        # === P2, P3, P4: Test threshold ===
        recent_age_days = recent_event.get_age_days()
        old_age_days = old_event.get_age_days()
        
        should_archive_recent = recent_event.should_archive(threshold_days)
        should_archive_old = old_event.should_archive(threshold_days)
        
        assert not should_archive_recent, (
            f"P3 VIOLATED: Event under threshold marked for archive.\n"
            f"  Age: {recent_age_days:.1f} days\n"
            f"  Threshold: {threshold_days} days\n"
            f"  Expected: should_archive=False"
        )
        
        assert should_archive_old, (
            f"P4 VIOLATED: Event over threshold not marked for archive.\n"
            f"  Age: {old_age_days:.1f} days\n"
            f"  Threshold: {threshold_days} days\n"
            f"  Expected: should_archive=True"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.7 PROOF SUMMARY: should_archive() Threshold")
        print(f"{'='*60}")
        print(f"  Threshold:            {threshold_days} days")
        print(f"  Recent event:")
        print(f"    Age:                {recent_age_days:.1f} days")
        print(f"    should_archive():   {should_archive_recent} ✓")
        print(f"  Old event:")
        print(f"    Age:                {old_age_days:.1f} days")
        print(f"    should_archive():   {should_archive_old} ✓")
        print(f"  RESULT: should_archive() THRESHOLD VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.8: TTL Persists Through Storage
    # ==========================================
    
    def test_k7_8_ttl_persists_through_storage(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K7.8: TTL and retention_class persist through LTM storage.
        
        THEOREM: After STM→LTM migration, TTL and retention_class are preserved.
        
        PROOF:
        - P1: Write event with TTL and retention_class
        - P2: Wait for migration to LTM
        - P3: Retrieve event
        - P4: TTL and retention_class match original
        """
        ttl_value = 7200  # 2 hours
        retention_class = "extended"
        
        # === P1: Write event ===
        event_id = f"k7-8-persist-{uuid.uuid4().hex[:8]}"
        
        original_event = Event(
            event_id=event_id,
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k7-8-session",
            content={"type": "ttl_persist_test"},
            timestamp=time.time(),
            created_at=time.time(),
            ttl_seconds=ttl_value,
            retention_class=retention_class,
        )
        
        kernel.write_event(test_workspace, test_user, original_event)
        
        # === P2: Wait for migration ===
        assert wait_for_migration(kernel, 1, timeout=15), "Migration timeout"
        
        # === P3: Retrieve ===
        retrieved = kernel.get_event(event_id)
        
        assert retrieved is not None, f"P3 VIOLATED: Event {event_id} not found"
        
        # === P4: Verify TTL and retention_class ===
        assert retrieved.ttl_seconds == ttl_value, (
            f"P4 VIOLATED: TTL mismatch.\n"
            f"  Original: {ttl_value}\n"
            f"  Retrieved: {retrieved.ttl_seconds}"
        )
        
        assert retrieved.retention_class == retention_class, (
            f"P4 VIOLATED: retention_class mismatch.\n"
            f"  Original: {retention_class}\n"
            f"  Retrieved: {retrieved.retention_class}"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.8 PROOF SUMMARY: TTL Persists Through Storage")
        print(f"{'='*60}")
        print(f"  Original TTL:         {ttl_value}")
        print(f"  Retrieved TTL:        {retrieved.ttl_seconds}")
        print(f"  TTL match:            {retrieved.ttl_seconds == ttl_value}")
        print(f"  Original retention:   {retention_class}")
        print(f"  Retrieved retention:  {retrieved.retention_class}")
        print(f"  Retention match:      {retrieved.retention_class == retention_class}")
        print(f"  RESULT: TTL PERSISTENCE VERIFIED")
        print(f"{'='*60}\n")
    
    # ==========================================
    # K7.9: Expired Events Still Queryable
    # ==========================================
    
    def test_k7_9_expired_events_queryable(
        self,
        kernel: KRNXController,
        test_workspace: str,
        test_user: str,
        wait_for_migration,
    ):
        """
        K7.9: Expired events are still returned by queries (expiry is informational).
        
        THEOREM: query_events() includes expired events. Expiry doesn't auto-delete.
        
        PROOF:
        - P1: Write event that is already expired (old timestamp + short TTL)
        - P2: Wait for migration
        - P3: Query returns the expired event
        - P4: Event's is_expired() returns True
        """
        now = time.time()
        
        # === P1: Create already-expired event ===
        event_id = f"k7-9-expired-{uuid.uuid4().hex[:8]}"
        
        # Timestamp 1 hour ago, TTL 30 minutes = already expired
        expired_event = Event(
            event_id=event_id,
            workspace_id=test_workspace,
            user_id=test_user,
            session_id="k7-9-session",
            content={"type": "expired_queryable_test"},
            timestamp=now - 3600,  # 1 hour ago
            created_at=now - 3600,
            ttl_seconds=1800,  # 30 minute TTL (already expired)
        )
        
        # Verify it's expired before storage
        assert expired_event.is_expired(), (
            "P1: Event should be expired before storage"
        )
        
        kernel.write_event(test_workspace, test_user, expired_event)
        
        # === P2: Wait for migration ===
        assert wait_for_migration(kernel, 1, timeout=15), "Migration timeout"
        
        # === P3: Query returns expired event ===
        result = kernel.query_events(
            workspace_id=test_workspace,
            user_id=test_user,
            limit=100,
        )
        
        result_ids = [e.event_id for e in result]
        
        assert event_id in result_ids, (
            f"P3 VIOLATED: Expired event not returned by query.\n"
            f"  Event ID: {event_id}\n"
            f"  Result IDs: {result_ids}"
        )
        
        # === P4: Retrieved event shows as expired ===
        retrieved = kernel.get_event(event_id)
        
        assert retrieved is not None, f"Event {event_id} not found"
        assert retrieved.is_expired(), (
            f"P4 VIOLATED: Retrieved event should show as expired.\n"
            f"  Age: {retrieved.get_age_seconds():.0f}s\n"
            f"  TTL: {retrieved.ttl_seconds}s"
        )
        
        print(f"\n{'='*60}")
        print(f"K7.9 PROOF SUMMARY: Expired Events Queryable")
        print(f"{'='*60}")
        print(f"  Event ID:             {event_id}")
        print(f"  Event age:            {retrieved.get_age_seconds():.0f}s")
        print(f"  Event TTL:            {retrieved.ttl_seconds}s")
        print(f"  is_expired():         {retrieved.is_expired()}")
        print(f"  Returned by query:    {event_id in result_ids}")
        print(f"  RESULT: EXPIRED EVENTS QUERYABLE VERIFIED")
        print(f"  NOTE: Expiry is informational only - no auto-delete")
        print(f"{'='*60}\n")


# ==========================================
# PROOF SUMMARY
# ==========================================
"""
K7 Test Suite: TTL and Expiry Semantics

Theorems Proven:
- K7.1: TTL values are stored and retrieved exactly
- K7.2: is_expired() correctly identifies expired events
- K7.3: Events without TTL never expire
- K7.4: get_age_seconds() returns accurate age
- K7.5: get_age_days() returns accurate age in days
- K7.6: is_recent(hours) correctly applies threshold
- K7.7: should_archive(days) correctly applies threshold
- K7.8: TTL and retention_class persist through LTM storage
- K7.9: Expired events are still queryable (expiry is informational)

Each test provides:
- Numbered proof steps (P1, P2, P3, P4)
- Actual TTL and age values shown
- Boundary condition testing
- Proof summary output

Run with: pytest -v -s prod_test/layer1_kernel/test_k7_ttl_expiry.py
"""
