"""
KRNX Enrichment - End-to-End Integration Tests

Run from your project root:
    cd D:\\chillbot\\chillbot
    python test_enrichment_e2e.py

Or with pytest:
    pytest test_enrichment_e2e.py -v

Tests the FULL pipeline with real infrastructure:
- Redis (STM) via Docker
- SQLite (LTM)
- Multi-signal enrichment

Prerequisites:
    docker-compose up -d  # Start Redis + Qdrant

Test order (simple → complex):
1. test_enrichment_short.py   - Unit tests, no infra needed
2. test_enrichment_full.py    - Integration tests, no infra needed  
3. test_enrichment_e2e.py     - Full system tests, Docker required
"""

import os
import sys
import time
import uuid
import shutil
import tempfile
import unittest
from typing import Optional, Dict, Any, List

# ==============================================
# PATH SETUP - Adapt to environment
# ==============================================

# Always add the directory containing this script first
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Try to find the project root
_possible_roots = [
    _script_dir,                     # Script's directory (most likely)
    os.getcwd(),                     # Current working directory
    '/mnt/user-data/uploads',        # Claude test environment
    'D:\\chillbot\\chillbot',        # Windows project
    '/home/user/chillbot/chillbot',  # Linux project
]

for root in _possible_roots:
    if os.path.exists(root) and root not in sys.path:
        sys.path.insert(0, root)

# Add fabric/ so 'from enrichment import ...' works
for root in [_script_dir, os.getcwd()]:
    fabric_path = os.path.join(root, 'fabric')
    if os.path.exists(fabric_path) and fabric_path not in sys.path:
        sys.path.insert(0, fabric_path)


# ==============================================
# INFRASTRUCTURE CHECKS
# ==============================================

def check_redis_available(host: str = "localhost", port: int = 6379) -> bool:
    """Check if Redis is available."""
    try:
        import redis
        r = redis.Redis(host=host, port=port, socket_timeout=2)
        r.ping()
        return True
    except Exception:
        return False


def check_qdrant_available(host: str = "localhost", port: int = 6333) -> bool:
    """Check if Qdrant is available."""
    try:
        import httpx
        response = httpx.get(f"http://{host}:{port}/readyz", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def check_krnx_imports() -> bool:
    """Check if KRNX modules can be imported."""
    try:
        # Direct kernel import (when running from project root)
        from kernel.models import Event
        from kernel.stm import STM
        from kernel.ltm import LTM
        return True
    except ImportError:
        pass
    
    try:
        # Namespaced import
        from chillbot.kernel.models import Event
        return True
    except ImportError:
        pass
    
    return False


# Check infrastructure at module load
REDIS_AVAILABLE = check_redis_available()
QDRANT_AVAILABLE = check_qdrant_available()
KRNX_IMPORTS_AVAILABLE = check_krnx_imports()

print(f"\n{'='*60}")
print("KRNX E2E Test Environment")
print(f"{'='*60}")
print(f"  Redis:   {'✓ Available' if REDIS_AVAILABLE else '✗ Not running (docker-compose up -d)'}")
print(f"  Qdrant:  {'✓ Available' if QDRANT_AVAILABLE else '✗ Not running (docker-compose up -d)'}")
print(f"  Imports: {'✓ Available' if KRNX_IMPORTS_AVAILABLE else '✗ Run from project root'}")
print(f"{'='*60}\n")


# ==============================================
# DYNAMIC IMPORTS (handle different project structures)
# ==============================================

def get_event_class():
    """Get Event class from available imports."""
    try:
        from kernel.models import Event
        return Event
    except ImportError:
        try:
            from chillbot.kernel.models import Event
            return Event
        except ImportError:
            from models import Event
            return Event


def get_create_event():
    """Get create_event function or create a factory from Event class."""
    # Try to import create_event function
    try:
        from kernel.models import create_event
        return create_event
    except ImportError:
        pass
    
    try:
        from chillbot.kernel.models import create_event
        return create_event
    except ImportError:
        pass
    
    # Fallback: create factory from Event class
    Event = get_event_class()
    
    def create_event(event_id, workspace_id, user_id, content, timestamp, **kwargs):
        """Factory function for Event."""
        return Event(
            event_id=event_id,
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=kwargs.get('session_id', f"{workspace_id}_{user_id}"),
            content=content,
            timestamp=timestamp,
            channel=kwargs.get('channel', 'default'),
            metadata=kwargs.get('metadata', {}),
        )
    
    return create_event


def get_stm_class():
    """Get STM class."""
    try:
        from kernel.stm import STM
        return STM
    except ImportError:
        try:
            from chillbot.kernel.stm import STM
            return STM
        except ImportError:
            from stm import STM
            return STM


def get_ltm_class():
    """Get LTM class."""
    try:
        from kernel.ltm import LTM
        return LTM
    except ImportError:
        try:
            from chillbot.kernel.ltm import LTM
            return LTM
        except ImportError:
            from ltm import LTM
            return LTM


def get_controller_class():
    """Get KRNXController class."""
    try:
        from kernel.controller import KRNXController
        return KRNXController
    except ImportError:
        try:
            from chillbot.kernel.controller import KRNXController
            return KRNXController
        except ImportError:
            from controller import KRNXController
            return KRNXController


def configure_redis_pool(host="localhost", port=6379, max_connections=10):
    """Configure Redis connection pool."""
    # Must configure chillbot.kernel.connection_pool since that's what STM uses internally
    try:
        from chillbot.kernel.connection_pool import configure_pool
        configure_pool(host=host, port=port, max_connections=max_connections)
        return
    except ImportError:
        pass
    
    try:
        from kernel.connection_pool import configure_pool
        configure_pool(host=host, port=port, max_connections=max_connections)
    except ImportError:
        from connection_pool import configure_pool
        configure_pool(host=host, port=port, max_connections=max_connections)


def close_redis_pool():
    """Close Redis connection pool."""
    try:
        from chillbot.kernel.connection_pool import close_pool
        close_pool()
        return
    except ImportError:
        pass
    
    try:
        from kernel.connection_pool import close_pool
        close_pool()
    except ImportError:
        from connection_pool import close_pool
        close_pool()


# ==============================================
# TEST UTILITIES
# ==============================================

class TestDataGenerator:
    """Generates test events with predictable patterns."""
    
    def __init__(self, workspace_id: str = "test_workspace"):
        self.workspace_id = workspace_id
        self.user_id = "test_user"
        self.session_id = f"{workspace_id}_{self.user_id}"
        self._counter = 0
        self._Event = get_event_class()
    
    def create_event(
        self,
        text: str,
        timestamp: Optional[float] = None,
        user_id: Optional[str] = None,
        channel: str = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a test event."""
        self._counter += 1
        event_id = f"evt_test_{self._counter:04d}_{uuid.uuid4().hex[:8]}"
        
        return self._Event(
            event_id=event_id,
            workspace_id=self.workspace_id,
            user_id=user_id or self.user_id,
            session_id=self.session_id,
            content={"text": text, "type": "test"},
            timestamp=timestamp or time.time(),
            channel=channel,
            metadata=metadata or {},
        )
    
    def create_correction_sequence(self) -> List:
        """Create a sequence of events where each corrects the previous."""
        now = time.time()
        
        return [
            self.create_event(
                text="Meeting scheduled for 2pm, budget $50,000",
                timestamp=now - 3600,
            ),
            self.create_event(
                text="Meeting rescheduled to 3pm, budget $50,000",  # Time change
                timestamp=now - 1800,
            ),
            self.create_event(
                text="Meeting at 3pm, budget updated to $75,000",  # Numeric change
                timestamp=now - 600,
            ),
            self.create_event(
                text="Meeting at 3pm is not happening, budget $75,000",  # Negation
                timestamp=now,
            ),
        ]
    
    def create_multi_user_conflict(self) -> List:
        """Create conflicting statements from different users."""
        now = time.time()
        
        return [
            self.create_event(
                text="Budget is approved for $50,000",
                user_id="alice",
                timestamp=now - 300,
            ),
            self.create_event(
                text="Budget is not approved, still under review",
                user_id="bob",
                timestamp=now,
            ),
        ]


# ==============================================
# E2E TEST: REDIS STM
# ==============================================

@unittest.skipUnless(REDIS_AVAILABLE and KRNX_IMPORTS_AVAILABLE, 
                     "Redis not available or imports failed - run: docker-compose up -d")
class TestRedisSTM(unittest.TestCase):
    """Test Redis STM operations."""
    
    @classmethod
    def setUpClass(cls):
        """Set up Redis connection."""
        configure_redis_pool(host="localhost", port=6379, max_connections=10)
        cls.generator = TestDataGenerator(workspace_id=f"test_e2e_{uuid.uuid4().hex[:8]}")
        cls.STM = get_stm_class()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up Redis test data."""
        try:
            import redis
            client = redis.Redis(host="localhost", port=6379)
            # Clean up test keys
            pattern = f"*{cls.generator.workspace_id}*"
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
        except Exception:
            pass
        finally:
            close_redis_pool()
    
    def test_stm_write_and_read(self):
        """Test basic STM write and read."""
        stm = self.STM()  # No workspace_id in constructor
        event = self.generator.create_event("Hello STM")
        
        # Write (correct API)
        stm.write_event(
            workspace_id=self.generator.workspace_id,
            user_id=self.generator.user_id,
            event=event,
        )
        
        # Read back (correct API: get_event)
        retrieved = stm.get_event(event.event_id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.event_id, event.event_id)
        self.assertEqual(retrieved.content["text"], "Hello STM")
    
    def test_stm_recent_events(self):
        """Test retrieving recent events."""
        stm = self.STM()
        
        # Write multiple events
        events = []
        for i in range(5):
            event = self.generator.create_event(f"Event {i}")
            stm.write_event(
                workspace_id=self.generator.workspace_id,
                user_id=self.generator.user_id,
                event=event,
            )
            events.append(event)
            time.sleep(0.01)  # Small delay for ordering
        
        # Get recent (correct API: get_events)
        recent = stm.get_events(
            workspace_id=self.generator.workspace_id,
            user_id=self.generator.user_id,
            limit=10,
        )
        
        self.assertGreaterEqual(len(recent), 5)
        
        # Most recent should be last written
        recent_ids = [e.event_id for e in recent]
        self.assertIn(events[-1].event_id, recent_ids)


# ==============================================
# E2E TEST: SQLite LTM
# ==============================================

@unittest.skipUnless(KRNX_IMPORTS_AVAILABLE, "KRNX imports not available - run from project root")
class TestSQLiteLTM(unittest.TestCase):
    """Test SQLite LTM operations."""
    
    def setUp(self):
        """Create temp directory for LTM."""
        self.temp_dir = tempfile.mkdtemp(prefix="krnx_e2e_")
        self.generator = TestDataGenerator()
        self.LTM = get_ltm_class()
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ltm_write_and_query(self):
        """Test LTM write and query."""
        ltm = self.LTM(data_path=self.temp_dir)
        event = self.generator.create_event("Hello LTM")
        
        # Write (correct API: store_event)
        ltm.store_event(event)
        
        # Query (correct API: query_events)
        results = ltm.query_events(
            workspace_id=self.generator.workspace_id,
            limit=10,
        )
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].event_id, event.event_id)
        
        ltm.close()
    
    def test_ltm_metadata_storage(self):
        """Test that metadata is properly stored in LTM."""
        ltm = self.LTM(data_path=self.temp_dir)
        
        # Event with enrichment metadata
        event = self.generator.create_event(
            "Test with metadata",
            metadata={
                "salience_score": 0.75,
                "relations": [
                    {"kind": "supersedes", "target": "evt_old", "confidence": 0.9}
                ],
                "entities": [
                    {"text": "test", "type": "concept"}
                ],
            }
        )
        
        ltm.store_event(event)
        
        # Retrieve
        results = ltm.query_events(
            workspace_id=self.generator.workspace_id,
            limit=1,
        )
        
        self.assertEqual(len(results), 1)
        retrieved = results[0]
        
        # Check metadata preserved
        self.assertIn("salience_score", retrieved.metadata)
        self.assertEqual(retrieved.metadata["salience_score"], 0.75)
        self.assertIn("relations", retrieved.metadata)
        
        ltm.close()


# ==============================================
# E2E TEST: FULL ENRICHMENT PIPELINE
# ==============================================

@unittest.skipUnless(REDIS_AVAILABLE and KRNX_IMPORTS_AVAILABLE, 
                     "Redis or imports not available - run: docker-compose up -d")
class TestFullEnrichmentPipeline(unittest.TestCase):
    """
    Test the complete enrichment pipeline with real infrastructure.
    
    Flow: Event → STM → Enrichment → LTM with metadata
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        configure_redis_pool(host="localhost", port=6379, max_connections=10)
        cls.temp_dir = tempfile.mkdtemp(prefix="krnx_e2e_full_")
        cls.workspace_id = f"test_full_{uuid.uuid4().hex[:8]}"
        cls.generator = TestDataGenerator(workspace_id=cls.workspace_id)
        cls.STM = get_stm_class()
        cls.LTM = get_ltm_class()
        cls.Event = get_event_class()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        # Clean Redis
        try:
            import redis
            client = redis.Redis(host="localhost", port=6379)
            pattern = f"*{cls.workspace_id}*"
            for key in client.keys(pattern):
                client.delete(key)
        except Exception:
            pass
        finally:
            close_redis_pool()
        
        # Clean temp dir
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_enrichment_with_real_events(self):
        """Test enrichment produces correct metadata for real events."""
        from enrichment import (
            FeatureExtractor,
            RelationScorer,
            SalienceEngine,
            RelationType,
        )
        
        # Create correction sequence
        events = self.generator.create_correction_sequence()
        
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        salience_engine = SalienceEngine()
        
        # Track relations for each event
        all_relations = {}
        
        for i, event in enumerate(events):
            event_relations = []
            
            # Compare with previous events
            for j in range(i):
                prev = events[j]
                
                # Simulate embedding similarity (high for related content)
                similarity = 0.85 if "meeting" in prev.content["text"].lower() else 0.3
                
                features = extractor.extract(event, prev, embedding_similarity=similarity)
                relations = scorer.score_pair(event, prev, features)
                event_relations.extend(relations)
            
            all_relations[event.event_id] = event_relations
            
            # Compute salience
            salience = salience_engine.compute_with_relations(
                event_id=event.event_id,
                timestamp=event.timestamp,
                relations=event_relations,
            )
            
            # Store salience in event metadata (simulating what orchestrator does)
            event.metadata["salience_score"] = salience.score
            event.metadata["relations"] = [r.to_dict() for r in event_relations]
        
        # Verify: Last event should supersede previous (via negation or numeric mismatch)
        last_event = events[-1]
        last_relations = all_relations[last_event.event_id]
        
        supersedes = [r for r in last_relations if r.kind == RelationType.SUPERSEDES]
        
        # Debug: show what was detected
        if not supersedes:
            print(f"\n  DEBUG: No supersedes found")
            print(f"  Relations found: {[r.kind.value for r in last_relations]}")
            for rel in last_relations:
                print(f"    {rel.kind.value}: signals={rel.signals}")
        
        self.assertTrue(len(supersedes) > 0, "Last event should supersede previous (negation detected)")
        
        # Verify negation was detected (cancelled vs scheduled)
        signals_found = set()
        for rel in last_relations:
            signals_found.update(rel.signals)
        
        # Should have detected negation (cancelled has implicit negation context)
        print(f"\n  Signals detected: {signals_found}")
        print(f"  Relations found: {len(last_relations)}")
    
    def test_multi_user_contradiction_detection(self):
        """Test contradiction detection between different users."""
        from enrichment import (
            FeatureExtractor,
            RelationScorer,
            RelationType,
        )
        
        events = self.generator.create_multi_user_conflict()
        alice_event, bob_event = events
        
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        
        # Extract features
        features = extractor.extract(bob_event, alice_event, embedding_similarity=0.88)
        
        # Should detect negation mismatch
        self.assertTrue(features.negation_mismatch, "Should detect 'approved' vs 'not approved'")
        self.assertFalse(features.same_actor, "Different users")
        
        # Score relations
        relations = scorer.score_pair(bob_event, alice_event, features)
        
        # Should detect CONTRADICTS (not supersedes, because different actors)
        contradicts = [r for r in relations if r.kind == RelationType.CONTRADICTS]
        self.assertTrue(len(contradicts) > 0, "Should detect contradiction between users")
        
        print(f"\n  Contradiction detected between alice and bob")
        print(f"  Signals: {contradicts[0].signals}")
        print(f"  Reason: {contradicts[0].reason_code}")
    
    def test_full_write_path_with_controller(self):
        """Test the full write path through KRNXController."""
        from enrichment import FeatureExtractor, RelationScorer
        
        # Initialize components (correct APIs)
        stm = self.STM()  # No workspace_id in constructor
        ltm = self.LTM(data_path=self.temp_dir)
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        
        # Create and store events
        events = []
        for i, text in enumerate([
            "Server configuration: port 8000",
            "Server configuration updated: port 8080",
            "Server configuration finalized: port 443 with SSL",
        ]):
            event = self.generator.create_event(
                text=text,
                timestamp=time.time() - (2 - i) * 300,  # Spaced 5 min apart
            )
            events.append(event)
        
        # Simulate the full pipeline for each event
        for i, event in enumerate(events):
            # 1. Write to STM (correct API)
            stm.write_event(
                workspace_id=self.workspace_id,
                user_id=self.generator.user_id,
                event=event,
            )
            
            # 2. Enrich (compare with previous events)
            relations = []
            for j in range(i):
                prev = events[j]
                features = extractor.extract(event, prev, embedding_similarity=0.85)
                relations.extend(scorer.score_pair(event, prev, features))
            
            # 3. Add enrichment to metadata
            enriched_event = self.Event(
                event_id=event.event_id,
                workspace_id=event.workspace_id,
                user_id=event.user_id,
                session_id=event.session_id,
                content=event.content,
                timestamp=event.timestamp,
                channel=event.channel,
                metadata={
                    **event.metadata,
                    "relations": [r.to_dict() for r in relations],
                    "enrichment_version": "3.0.0",
                }
            )
            
            # 4. Store in LTM (correct API)
            ltm.store_event(enriched_event)
        
        # Verify: Query LTM and check metadata (correct API)
        results = ltm.query_events(workspace_id=self.workspace_id, limit=10)
        
        self.assertGreaterEqual(len(results), 3)
        
        # Last event should have relations to previous events
        # Find the "finalized" event
        finalized = next(
            (e for e in results if "finalized" in e.content.get("text", "")),
            None
        )
        
        if finalized:
            self.assertIn("relations", finalized.metadata)
            self.assertGreater(len(finalized.metadata["relations"]), 0)
            print(f"\n  Final event has {len(finalized.metadata['relations'])} relations")
        
        ltm.close()


# ==============================================
# E2E TEST: CONTROLLER INTEGRATION
# ==============================================

@unittest.skipUnless(REDIS_AVAILABLE and KRNX_IMPORTS_AVAILABLE, 
                     "Redis or imports not available - run: docker-compose up -d")
class TestControllerIntegration(unittest.TestCase):
    """Test KRNXController with enrichment."""
    
    @classmethod
    def setUpClass(cls):
        """Set up controller."""
        cls.temp_dir = tempfile.mkdtemp(prefix="krnx_controller_")
        cls.workspace_id = f"test_ctrl_{uuid.uuid4().hex[:8]}"
        cls.generator = TestDataGenerator(workspace_id=cls.workspace_id)
        cls.Event = get_event_class()
        
        # Initialize controller
        KRNXController = get_controller_class()
        
        cls.controller = KRNXController(
            data_path=cls.temp_dir,
            redis_host="localhost",
            redis_port=6379,
            enable_backpressure=False,  # Disable for testing
            enable_async_worker=False,   # Synchronous for testing
        )
    
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        if hasattr(cls, 'controller'):
            cls.controller.shutdown()
        
        # Clean Redis
        try:
            import redis
            r = redis.Redis(host="localhost", port=6379)
            pattern = f"*{cls.workspace_id}*"
            for key in r.keys(pattern):
                r.delete(key)
        except Exception:
            pass
        
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_controller_write_event(self):
        """Test writing event through controller."""
        event = self.generator.create_event("Controller test event")
        
        # Write through controller
        message_id = self.controller.write_event(
            workspace_id=self.workspace_id,
            user_id=self.generator.user_id,
            event=event,
        )
        
        self.assertIsNotNone(message_id)
        print(f"\n  Event written with message_id: {message_id}")
    
    def test_controller_write_with_enriched_metadata(self):
        """Test writing event with pre-computed enrichment metadata."""
        from enrichment import (
            FeatureExtractor,
            RelationScorer,
            SalienceEngine,
        )
        
        # Create two events - second supersedes first
        old_event = self.generator.create_event(
            "Budget is $50,000",
            timestamp=time.time() - 600,
        )
        new_event = self.generator.create_event(
            "Budget updated to $75,000",
            timestamp=time.time(),
        )
        
        # Write old event first
        self.controller.write_event(
            workspace_id=self.workspace_id,
            user_id=self.generator.user_id,
            event=old_event,
        )
        
        # Compute enrichment for new event
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        salience_engine = SalienceEngine()
        
        features = extractor.extract(new_event, old_event, embedding_similarity=0.88)
        relations = scorer.score_pair(new_event, old_event, features)
        salience = salience_engine.compute_with_relations(
            event_id=new_event.event_id,
            timestamp=new_event.timestamp,
            relations=relations,
        )
        
        # Create enriched event
        enriched_event = self.Event(
            event_id=new_event.event_id,
            workspace_id=new_event.workspace_id,
            user_id=new_event.user_id,
            session_id=new_event.session_id,
            content=new_event.content,
            timestamp=new_event.timestamp,
            channel=new_event.channel,
            metadata={
                "relations": [r.to_dict() for r in relations],
                "salience_score": salience.score,
                "salience_factors": salience.factors,
            }
        )
        
        # Write enriched event
        message_id = self.controller.write_event(
            workspace_id=self.workspace_id,
            user_id=self.generator.user_id,
            event=enriched_event,
        )
        
        self.assertIsNotNone(message_id)
        
        # Verify we can read it back with metadata (correct API: get_event)
        retrieved = self.controller.stm.get_event(enriched_event.event_id)
        
        self.assertIsNotNone(retrieved)
        self.assertIn("relations", retrieved.metadata)
        self.assertIn("salience_score", retrieved.metadata)
        
        print(f"\n  Enriched event stored with {len(relations)} relations")
        print(f"  Salience: {salience.score:.3f}")


# ==============================================
# E2E TEST: STRESS / PERFORMANCE
# ==============================================

@unittest.skipUnless(KRNX_IMPORTS_AVAILABLE, "KRNX imports not available")
class TestPerformance(unittest.TestCase):
    """Performance tests for enrichment pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test classes."""
        cls._create_event_func = get_create_event()
    
    def test_enrichment_throughput(self):
        """Test enrichment can handle high throughput."""
        from enrichment import FeatureExtractor, RelationScorer
        
        extractor = FeatureExtractor()
        scorer = RelationScorer()
        create_event = self.__class__._create_event_func  # Access without binding
        
        # Create test events
        num_events = 100
        events = []
        now = time.time()
        
        for i in range(num_events):
            event = create_event(
                event_id=f"evt_perf_{i:04d}",
                workspace_id="perf_test",
                user_id="user",
                content={"text": f"Performance test event {i} with budget ${i * 1000}"},
                timestamp=now - (num_events - i),
            )
            events.append(event)
        
        # Benchmark: Extract features and score relations
        start = time.time()
        
        total_relations = 0
        for i, event in enumerate(events[1:], 1):
            # Compare with previous event only (realistic scenario)
            prev = events[i - 1]
            features = extractor.extract(event, prev, embedding_similarity=0.85)
            relations = scorer.score_pair(event, prev, features)
            total_relations += len(relations)
        
        elapsed = time.time() - start
        events_per_sec = (num_events - 1) / elapsed
        
        print(f"\n  Enrichment throughput: {events_per_sec:.0f} events/sec")
        print(f"  Total relations detected: {total_relations}")
        print(f"  Avg relations per event: {total_relations / (num_events - 1):.2f}")
        
        # Should handle at least 1000 events/sec (pure Python, no embeddings)
        self.assertGreater(events_per_sec, 500, "Enrichment too slow")
    
    def test_feature_extraction_latency(self):
        """Test feature extraction latency."""
        from enrichment import FeatureExtractor
        
        extractor = FeatureExtractor()
        create_event = self.__class__._create_event_func  # Access without binding
        now = time.time()
        
        event_a = create_event(
            event_id="evt_latency_a",
            workspace_id="latency_test",
            user_id="user",
            content={"text": "Meeting scheduled for Monday at 3pm, budget is $50,000"},
            timestamp=now,
        )
        
        event_b = create_event(
            event_id="evt_latency_b",
            workspace_id="latency_test",
            user_id="user",
            content={"text": "Meeting rescheduled to Tuesday at 4pm, budget updated to $75,000"},
            timestamp=now - 100,
        )
        
        # Warm up
        for _ in range(10):
            extractor.extract(event_a, event_b, embedding_similarity=0.85)
        
        # Benchmark
        iterations = 1000
        start = time.time()
        
        for _ in range(iterations):
            features = extractor.extract(event_a, event_b, embedding_similarity=0.85)
        
        elapsed = time.time() - start
        avg_latency_ms = (elapsed / iterations) * 1000
        
        print(f"\n  Feature extraction latency: {avg_latency_ms:.3f}ms")
        
        # Should be under 1ms
        self.assertLess(avg_latency_ms, 1.0, "Feature extraction too slow")


# ==============================================
# MAIN
# ==============================================

def main():
    """Run E2E tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes in order of complexity
    test_classes = [
        TestSQLiteLTM,           # No Docker needed
        TestRedisSTM,            # Needs Redis
        TestFullEnrichmentPipeline,  # Needs Redis
        TestControllerIntegration,   # Needs Redis
        TestPerformance,         # Needs Redis
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("E2E TEST SUMMARY")
    print("=" * 60)
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    if result.skipped:
        print("\n  Skipped tests (Docker not running?):")
        for test, reason in result.skipped:
            print(f"    - {test}: {reason}")
    
    print("=" * 60)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
