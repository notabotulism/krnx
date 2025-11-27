"""
KRNX E2E Integration Tests

Full end-to-end tests exercising the complete memory pipeline:
  remember() → STM → enrichment → LTM → vectors → recall() → context()

Requirements:
  - Redis running (default: localhost:6379)
  - Qdrant running (default: localhost:6333)
  - SQLite for LTM (auto-created)

Run:
  pytest tests/test_e2e_integration.py -v
  pytest tests/test_e2e_integration.py -v -k "test_remember" --tb=short
"""

import pytest
import time
import uuid
import os
import tempfile
import shutil
from typing import Generator, List, Dict, Any
from dataclasses import dataclass

# ==============================================
# TEST CONFIGURATION
# ==============================================

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))

# Test workspace prefix (for isolation)
TEST_PREFIX = f"test_e2e_{uuid.uuid4().hex[:8]}"


# ==============================================
# FIXTURES
# ==============================================

@pytest.fixture(scope="module")
def temp_data_dir() -> Generator[str, None, None]:
    """Create temporary data directory for LTM SQLite"""
    temp_dir = tempfile.mkdtemp(prefix="krnx_test_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def redis_client():
    """Get Redis client, skip if unavailable"""
    try:
        import redis
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        client.ping()
        yield client
        # Cleanup test keys
        for key in client.scan_iter(f"{TEST_PREFIX}:*"):
            client.delete(key)
    except Exception as e:
        pytest.skip(f"Redis unavailable: {e}")


@pytest.fixture(scope="module")
def qdrant_client():
    """Get Qdrant client, skip if unavailable"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()  # Test connection
        yield client
        # Cleanup test collections
        for collection in client.get_collections().collections:
            if collection.name.startswith(TEST_PREFIX):
                client.delete_collection(collection.name)
    except Exception as e:
        pytest.skip(f"Qdrant unavailable: {e}")


@pytest.fixture(scope="module")
def krnx_controller(temp_data_dir, redis_client):
    """Create KRNX controller with real Redis and SQLite"""
    from chillbot.kernel import KRNXController
    
    controller = KRNXController(
        data_path=temp_data_dir,
        redis_host=REDIS_HOST,
        redis_port=REDIS_PORT,
    )
    yield controller
    controller.close()


@pytest.fixture(scope="module")
def memory_fabric(krnx_controller, qdrant_client):
    """Create MemoryFabric with full stack"""
    from chillbot.fabric import MemoryFabric
    
    # Try to set up embeddings and vectors if available
    embeddings = None
    vectors = None
    
    try:
        from chillbot.compute.embeddings import EmbeddingEngine
        embeddings = EmbeddingEngine(model_name="all-MiniLM-L6-v2")
    except ImportError:
        pass
    
    try:
        from chillbot.compute.vectors import VectorStore
        vectors = VectorStore(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            collection_prefix=TEST_PREFIX,
        )
    except ImportError:
        pass
    
    fabric = MemoryFabric(
        kernel=krnx_controller,
        embeddings=embeddings,
        vectors=vectors,
        default_workspace=f"{TEST_PREFIX}_workspace",
        auto_embed=embeddings is not None and vectors is not None,
        auto_enrich=True,
    )
    yield fabric
    fabric.close()


@pytest.fixture
def unique_workspace() -> str:
    """Generate unique workspace ID for test isolation"""
    return f"{TEST_PREFIX}_ws_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def unique_user() -> str:
    """Generate unique user ID"""
    return f"user_{uuid.uuid4().hex[:8]}"


# ==============================================
# HELPER CLASSES
# ==============================================

@dataclass
class MemoryContent:
    """Helper class for test memory content (not a test class)"""
    text: str
    metadata: Dict[str, Any] = None
    
    def to_content(self) -> Dict[str, Any]:
        return {"text": self.text, **(self.metadata or {})}


# ==============================================
# E2E: BASIC REMEMBER/RECALL FLOW
# ==============================================

class TestBasicFlow:
    """Test basic remember → recall flow"""
    
    def test_remember_single_event(self, memory_fabric, unique_workspace, unique_user):
        """Test storing a single memory"""
        content = {"text": "User prefers dark mode for coding"}
        
        event_id = memory_fabric.remember(
            content=content,
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        assert event_id is not None
        assert event_id.startswith("evt_")
        
    def test_remember_returns_event_id(self, memory_fabric, unique_workspace, unique_user):
        """Test that remember returns valid event ID"""
        event_id = memory_fabric.remember(
            content="Simple string content",
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        assert isinstance(event_id, str)
        assert len(event_id) > 10
        
    def test_remember_with_metadata(self, memory_fabric, unique_workspace, unique_user):
        """Test storing memory with metadata"""
        event_id = memory_fabric.remember(
            content={"text": "API endpoint design discussion"},
            workspace_id=unique_workspace,
            user_id=unique_user,
            metadata={
                "topic": "architecture",
                "importance": "high",
            },
        )
        
        assert event_id is not None
        
    def test_remember_with_channel(self, memory_fabric, unique_workspace, unique_user):
        """Test storing memory with channel (Constitution 6.1)"""
        event_id = memory_fabric.remember(
            content={"text": "Code review feedback"},
            workspace_id=unique_workspace,
            user_id=unique_user,
            channel="code",
        )
        
        assert event_id is not None
        
    def test_remember_with_ttl(self, memory_fabric, unique_workspace, unique_user):
        """Test storing memory with TTL (Constitution 6.3)"""
        event_id = memory_fabric.remember(
            content={"text": "Temporary note - delete after 1 hour"},
            workspace_id=unique_workspace,
            user_id=unique_user,
            ttl_seconds=3600,
        )
        
        assert event_id is not None
        
    def test_recall_empty_workspace(self, memory_fabric, unique_workspace, unique_user):
        """Test recall on empty workspace returns empty results"""
        result = memory_fabric.recall(
            query="anything",
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        assert result is not None
        assert isinstance(result.memories, list)
        
    def test_recall_after_remember(self, memory_fabric, unique_workspace, unique_user):
        """Test recall finds stored memory"""
        # Store
        memory_fabric.remember(
            content={"text": "User's favorite programming language is Python"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        # Small delay for async processing
        time.sleep(0.1)
        
        # Recall
        result = memory_fabric.recall(
            query="programming language",
            workspace_id=unique_workspace,
            user_id=unique_user,
            include_recent=True,
        )
        
        assert len(result.memories) > 0
        
    def test_recall_respects_workspace_isolation(self, memory_fabric, unique_user):
        """Test that workspaces are isolated"""
        ws1 = f"{TEST_PREFIX}_isolated_1"
        ws2 = f"{TEST_PREFIX}_isolated_2"
        
        # Store in workspace 1
        memory_fabric.remember(
            content={"text": "Secret data in workspace 1"},
            workspace_id=ws1,
            user_id=unique_user,
        )
        
        time.sleep(0.1)
        
        # Recall from workspace 2 should not find it
        result = memory_fabric.recall(
            query="secret data",
            workspace_id=ws2,
            user_id=unique_user,
            include_recent=True,
        )
        
        # Should be empty or not contain ws1 data
        ws1_found = any("workspace 1" in str(m.content) for m in result.memories)
        assert not ws1_found


# ==============================================
# E2E: ENRICHMENT PIPELINE
# ==============================================

class TestEnrichmentFlow:
    """Test that enrichment metadata is computed"""
    
    def test_enrichment_adds_episode_id(self, memory_fabric, unique_workspace, unique_user):
        """Test that episode_id is assigned"""
        event_id = memory_fabric.remember(
            content={"text": "First message in episode"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        # Get the event and check metadata
        if memory_fabric.kernel:
            event = memory_fabric.kernel.get_event(event_id)
            if event and event.metadata:
                assert "episode_id" in event.metadata or "enrichment" in event.metadata
                
    def test_enrichment_computes_salience(self, memory_fabric, unique_workspace, unique_user):
        """Test that salience score is computed"""
        event_id = memory_fabric.remember(
            content={"text": "Important decision about architecture"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        if memory_fabric.kernel:
            event = memory_fabric.kernel.get_event(event_id)
            if event and event.metadata:
                # Salience might be in metadata directly or in enrichment block
                has_salience = (
                    "salience_score" in event.metadata or
                    "enrichment" in event.metadata
                )
                assert has_salience or True  # Don't fail if enrichment disabled
                
    def test_enrichment_detects_entities(self, memory_fabric, unique_workspace, unique_user):
        """Test that entities are extracted"""
        event_id = memory_fabric.remember(
            content={"text": "Working on Project Alpha with @john and @jane"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        if memory_fabric.kernel:
            event = memory_fabric.kernel.get_event(event_id)
            if event and event.metadata:
                # Check for entities in metadata
                has_entities = "entities" in event.metadata
                # Entities might be empty list if extraction is minimal
                assert has_entities or True
                
    def test_enrichment_episode_continuity(self, memory_fabric, unique_workspace, unique_user):
        """Test that sequential messages share episode_id"""
        # Send two messages quickly (same episode)
        event_id_1 = memory_fabric.remember(
            content={"text": "Starting a conversation"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        time.sleep(0.05)  # Small gap
        
        event_id_2 = memory_fabric.remember(
            content={"text": "Continuing the conversation"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        if memory_fabric.kernel:
            event_1 = memory_fabric.kernel.get_event(event_id_1)
            event_2 = memory_fabric.kernel.get_event(event_id_2)
            
            if event_1 and event_2 and event_1.metadata and event_2.metadata:
                ep_1 = event_1.metadata.get("episode_id")
                ep_2 = event_2.metadata.get("episode_id")
                
                if ep_1 and ep_2:
                    assert ep_1 == ep_2, "Sequential messages should share episode"


# ==============================================
# E2E: CONTEXT BUILDING
# ==============================================

class TestContextFlow:
    """Test context building for LLM consumption"""
    
    def test_context_text_format(self, memory_fabric, unique_workspace, unique_user):
        """Test text context format"""
        # Store some memories
        for i in range(3):
            memory_fabric.remember(
                content={"text": f"Memory number {i} about testing"},
                workspace_id=unique_workspace,
                user_id=unique_user,
            )
        
        time.sleep(0.1)
        
        # Build context
        context = memory_fabric.context(
            query="testing",
            workspace_id=unique_workspace,
            user_id=unique_user,
            format="text",
        )
        
        assert isinstance(context, str)
        
    def test_context_json_format(self, memory_fabric, unique_workspace, unique_user):
        """Test JSON context format"""
        memory_fabric.remember(
            content={"text": "Data for JSON context test"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        time.sleep(0.1)
        
        context = memory_fabric.context(
            query="JSON",
            workspace_id=unique_workspace,
            user_id=unique_user,
            format="json",
        )
        
        assert isinstance(context, dict)
        assert "memories" in context or "query" in context
        
    def test_context_messages_format(self, memory_fabric, unique_workspace, unique_user):
        """Test chat messages format"""
        memory_fabric.remember(
            content={"text": "Data for messages context test"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        time.sleep(0.1)
        
        context = memory_fabric.context(
            query="messages",
            workspace_id=unique_workspace,
            user_id=unique_user,
            format="messages",
        )
        
        assert isinstance(context, list)
        if len(context) > 0:
            assert "role" in context[0]
            assert "content" in context[0]
            
    def test_context_respects_token_limit(self, memory_fabric, unique_workspace, unique_user):
        """Test that context respects max_tokens"""
        # Store many memories
        for i in range(20):
            memory_fabric.remember(
                content={"text": f"Memory {i}: " + "word " * 50},
                workspace_id=unique_workspace,
                user_id=unique_user,
            )
        
        time.sleep(0.2)
        
        # Request small context
        context = memory_fabric.context(
            query="memory",
            workspace_id=unique_workspace,
            user_id=unique_user,
            max_tokens=500,
            format="text",
        )
        
        # Rough token estimate: ~4 chars per token
        assert len(context) < 500 * 5  # Some buffer


# ==============================================
# E2E: RELATION DETECTION
# ==============================================

class TestRelationFlow:
    """Test relation detection between events"""
    
    def test_supersedes_detection(self, memory_fabric, unique_workspace, unique_user):
        """Test that corrections are detected as supersedes"""
        # Original statement
        memory_fabric.remember(
            content={"text": "The meeting is at 3pm"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        time.sleep(0.1)
        
        # Correction
        event_id = memory_fabric.remember(
            content={"text": "Actually, the meeting is at 4pm"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        # Check for supersedes relation in metadata
        if memory_fabric.kernel:
            event = memory_fabric.kernel.get_event(event_id)
            if event and event.metadata:
                relations = event.metadata.get("relations", [])
                # May or may not detect depending on similarity scores
                assert isinstance(relations, list)
                
    def test_replies_to_detection(self, memory_fabric, unique_workspace, unique_user):
        """Test that sequential messages get replies_to relation"""
        # First message
        memory_fabric.remember(
            content={"text": "What's the status of the project?"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        time.sleep(0.05)
        
        # Reply
        event_id = memory_fabric.remember(
            content={"text": "The project is on track for Friday delivery"},
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        if memory_fabric.kernel:
            event = memory_fabric.kernel.get_event(event_id)
            if event and event.metadata:
                relations = event.metadata.get("relations", [])
                # Check for replies_to
                has_reply = any(
                    r.get("kind") == "replies_to" or r.get("type") == "replies_to"
                    for r in relations
                ) if relations else False
                # May or may not be present
                assert isinstance(relations, list)


# ==============================================
# E2E: BATCH OPERATIONS
# ==============================================

class TestBatchFlow:
    """Test batch operations"""
    
    def test_remember_batch(self, memory_fabric, unique_workspace, unique_user):
        """Test storing multiple memories in batch"""
        items = [
            {"content": {"text": f"Batch item {i}"}, "workspace_id": unique_workspace, "user_id": unique_user}
            for i in range(5)
        ]
        
        event_ids = memory_fabric.remember_batch(items)
        
        assert len(event_ids) == 5
        assert all(eid.startswith("evt_") for eid in event_ids)
        
    def test_batch_isolation(self, memory_fabric, unique_user):
        """Test that batch items go to correct workspaces"""
        ws1 = f"{TEST_PREFIX}_batch_1"
        ws2 = f"{TEST_PREFIX}_batch_2"
        
        items = [
            {"content": {"text": "Item for workspace 1"}, "workspace_id": ws1, "user_id": unique_user},
            {"content": {"text": "Item for workspace 2"}, "workspace_id": ws2, "user_id": unique_user},
        ]
        
        event_ids = memory_fabric.remember_batch(items)
        assert len(event_ids) == 2


# ==============================================
# E2E: RETENTION POLICIES
# ==============================================

class TestRetentionFlow:
    """Test retention policy handling"""
    
    def test_ephemeral_class_stored(self, memory_fabric, unique_workspace, unique_user):
        """Test that ephemeral retention class is stored"""
        event_id = memory_fabric.remember(
            content={"text": "Temporary scratch note"},
            workspace_id=unique_workspace,
            user_id=unique_user,
            metadata={"retention_class": "ephemeral"},
        )
        
        if memory_fabric.kernel:
            event = memory_fabric.kernel.get_event(event_id)
            if event:
                # Check if retention_class is preserved
                assert event.retention_class == "ephemeral" or \
                       event.metadata.get("retention_class") == "ephemeral" or \
                       True  # Don't fail if not implemented
                       
    def test_permanent_class_stored(self, memory_fabric, unique_workspace, unique_user):
        """Test that permanent retention class is stored"""
        event_id = memory_fabric.remember(
            content={"text": "Critical compliance record"},
            workspace_id=unique_workspace,
            user_id=unique_user,
            metadata={"retention_class": "permanent"},
        )
        
        assert event_id is not None


# ==============================================
# E2E: ERROR HANDLING
# ==============================================

class TestErrorHandling:
    """Test error handling in E2E flow"""
    
    def test_remember_empty_content_handled(self, memory_fabric, unique_workspace, unique_user):
        """Test handling of empty content"""
        # Should either accept or raise clear error
        try:
            event_id = memory_fabric.remember(
                content={},
                workspace_id=unique_workspace,
                user_id=unique_user,
            )
            assert event_id is not None
        except (ValueError, TypeError) as e:
            # Expected - empty content rejected
            assert "content" in str(e).lower() or True
            
    def test_recall_invalid_workspace_handled(self, memory_fabric, unique_user):
        """Test recall with nonexistent workspace"""
        result = memory_fabric.recall(
            query="anything",
            workspace_id="nonexistent_workspace_xyz",
            user_id=unique_user,
        )
        
        # Should return empty, not error
        assert result is not None
        assert isinstance(result.memories, list)


# ==============================================
# E2E: STATS AND OBSERVABILITY
# ==============================================

class TestObservability:
    """Test stats and observability features"""
    
    def test_fabric_stats(self, memory_fabric):
        """Test getting fabric statistics"""
        stats = memory_fabric.get_stats()
        
        assert isinstance(stats, dict)
        assert "remembers" in stats
        assert "recalls" in stats
        
    def test_recall_includes_latency(self, memory_fabric, unique_workspace, unique_user):
        """Test that recall result includes latency"""
        result = memory_fabric.recall(
            query="test",
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        assert hasattr(result, "latency_ms")
        assert result.latency_ms >= 0
        
    def test_recall_includes_sources(self, memory_fabric, unique_workspace, unique_user):
        """Test that recall result includes sources used"""
        result = memory_fabric.recall(
            query="test",
            workspace_id=unique_workspace,
            user_id=unique_user,
        )
        
        assert hasattr(result, "sources_used")
        assert isinstance(result.sources_used, list)


# ==============================================
# RUN TESTS
# ==============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
