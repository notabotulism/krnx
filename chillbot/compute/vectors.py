"""
KRNX Compute - Vector Store

Qdrant-backed vector storage for semantic search.
Supports multiple workspaces, filtering, and batch operations.

FIXED: TTL-based collection caching to prevent high-concurrency bottlenecks.
FIXED v2: Better thread-safety - use fresh timestamp after lock acquisition.

Usage:
    vectors = VectorStore(url="http://localhost:6333")
    
    # Ensure collection exists (cached with TTL)
    vectors.ensure_collection("my-workspace", dimension=384)
    
    # Index a vector
    vectors.index(
        workspace_id="my-workspace",
        id="evt_123",
        vector=[0.1, 0.2, ...],
        payload={"text": "Hello world", "user_id": "user_1"}
    )
    
    # Search
    results = vectors.search(
        workspace_id="my-workspace",
        vector=query_vector,
        top_k=10
    )
"""

import logging
import threading
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class VectorStoreBackend(Enum):
    """Supported vector store backends."""
    QDRANT = "qdrant"
    MEMORY = "memory"  # In-memory for testing


@dataclass
class VectorMatch:
    """A vector search result."""
    id: str
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "id": self.id,
            "score": self.score,
            "payload": self.payload,
        }
        if self.vector:
            result["vector"] = self.vector
        return result


@dataclass
class CollectionInfo:
    """Information about a vector collection."""
    name: str
    vectors_count: int
    dimension: int
    distance: str


class VectorStore:
    """
    Qdrant vector store wrapper.
    
    Provides workspace-scoped vector storage with semantic search.
    Each workspace gets its own collection.
    
    FIXED: TTL-based caching prevents 50-thread bottleneck on collection checks.
    FIXED v2: Thread-safe initialization with fresh timestamps after lock.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_prefix: str = "krnx",
        backend: VectorStoreBackend = VectorStoreBackend.QDRANT,
        cache_ttl_seconds: int = 300,  # 5-minute cache TTL
    ):
        """
        Initialize vector store.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for Qdrant Cloud
            collection_prefix: Prefix for collection names
            backend: Which backend to use (qdrant or memory)
            cache_ttl_seconds: TTL for collection cache (default 5 minutes)
        """
        self.url = url
        self.api_key = api_key
        self.collection_prefix = collection_prefix
        self.backend = backend
        self.cache_ttl = cache_ttl_seconds
        
        self._client = None
        self._client_lock = threading.Lock()  # Thread-safe client initialization
        
        # FIXED: Collection cache with timestamps (workspace -> (dimension, verified_at))
        self._known_collections: Dict[str, tuple] = {}  # workspace -> (dimension, timestamp)
        self._collection_lock = threading.Lock()  # Separate lock for collection ops
        
        # In-memory storage for testing
        self._memory_store: Dict[str, Dict[str, tuple]] = {}  # collection -> {id: (vector, payload)}
    
    @property
    def client(self):
        """Lazy load Qdrant client."""
        if self._client is None and self.backend == VectorStoreBackend.QDRANT:
            self._connect()
        return self._client
    
    def _connect(self):
        """Connect to Qdrant (thread-safe)."""
        # Double-check locking pattern
        with self._client_lock:
            # Check again after acquiring lock
            if self._client is not None:
                return
            
            try:
                from qdrant_client import QdrantClient
                
                logger.info(f"[VECTOR] Connecting to Qdrant at {self.url}")
                
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                )
                
                # Test connection
                self._client.get_collections()
                logger.info("[VECTOR] Connected to Qdrant")
                
            except ImportError:
                raise ImportError(
                    "qdrant-client is required for vector storage. "
                    "Install with: pip install qdrant-client"
                )
            except Exception as e:
                logger.error(f"[VECTOR] Failed to connect to Qdrant: {e}")
                raise
    
    def _collection_name(self, workspace_id: str) -> str:
        """Get collection name for workspace."""
        return f"{self.collection_prefix}_{workspace_id}"
    
    def collection_exists(self, workspace_id: str) -> bool:
        """Check if collection exists for workspace."""
        if self.backend == VectorStoreBackend.MEMORY:
            return self._collection_name(workspace_id) in self._memory_store
        
        try:
            collections = self.client.get_collections().collections
            name = self._collection_name(workspace_id)
            return any(c.name == name for c in collections)
        except Exception:
            return False
    
    def ensure_collection(self, workspace_id: str, dimension: int):
        """
        Ensure collection exists (idempotent, cached, thread-safe).
        
        FIXED v2: Better thread-safety for high-concurrency scenarios.
        - Fast path: Cache hit (no network call, no lock)
        - Slow path: Acquire lock, re-check with FRESH timestamp, then verify/create
        - TTL ensures eventual consistency across processes
        
        Args:
            workspace_id: Workspace identifier
            dimension: Vector dimension (must match embedding model)
        """
        name = self._collection_name(workspace_id)
        now = time.time()
        
        # Memory backend - simple in-memory storage
        if self.backend == VectorStoreBackend.MEMORY:
            if name not in self._memory_store:
                self._memory_store[name] = {}
                self._known_collections[workspace_id] = (dimension, now)
            return
        
        # ==============================================
        # FAST PATH: Check cache WITHOUT lock
        # This is the 99% case after warm-up
        # ==============================================
        cached = self._known_collections.get(workspace_id)
        if cached:
            cached_dim, cached_at = cached
            if now - cached_at < self.cache_ttl:
                # Cache is still valid - fast return
                if cached_dim != dimension:
                    logger.warning(
                        f"[VECTOR] Dimension mismatch for {workspace_id}: "
                        f"cached={cached_dim}, requested={dimension}"
                    )
                return
        
        # ==============================================
        # SLOW PATH: Acquire lock, verify with Qdrant
        # Only happens on cache miss or TTL expiry
        # ==============================================
        with self._collection_lock:
            # CRITICAL FIX: Use FRESH timestamp after acquiring lock
            # (time may have passed while waiting for lock)
            fresh_now = time.time()
            
            # Double-check cache after acquiring lock
            # Another thread may have updated it while we waited
            cached = self._known_collections.get(workspace_id)
            if cached:
                cached_dim, cached_at = cached
                if fresh_now - cached_at < self.cache_ttl:
                    # Another thread already refreshed the cache
                    return
            
            try:
                from qdrant_client.models import Distance, VectorParams
                
                # Ensure client is connected
                if self._client is None:
                    self._connect()
                
                # Check if collection exists
                collections = self._client.get_collections().collections
                exists = any(c.name == name for c in collections)
                
                if not exists:
                    logger.info(f"[VECTOR] Creating collection: {name} (dim={dimension})")
                    
                    self._client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=dimension,
                            distance=Distance.COSINE,
                        ),
                    )
                else:
                    logger.debug(f"[VECTOR] Collection exists: {name}")
                
                # Update cache with FRESH timestamp (after all work is done)
                self._known_collections[workspace_id] = (dimension, time.time())
                
            except Exception as e:
                logger.error(f"[VECTOR] Failed to ensure collection {name}: {e}")
                raise
    
    def is_collection_cached(self, workspace_id: str) -> bool:
        """
        Check if collection is in valid cache (for testing/debugging).
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            True if collection is cached and not expired
        """
        cached = self._known_collections.get(workspace_id)
        if not cached:
            return False
        _, cached_at = cached
        return (time.time() - cached_at) < self.cache_ttl
    
    def index(
        self,
        workspace_id: str,
        id: str,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None,
    ):
        """
        Index a vector.
        
        NOTE: Assumes collection already exists (call ensure_collection first
        during initialization, not on every index call).
        
        Args:
            workspace_id: Workspace identifier
            id: Unique vector ID (usually event_id)
            vector: Vector to index
            payload: Optional metadata to store with vector
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name not in self._memory_store:
                self._memory_store[name] = {}
            self._memory_store[name][id] = (vector, payload or {})
            return
        
        try:
            from qdrant_client.models import PointStruct
            
            self.client.upsert(
                collection_name=name,
                points=[
                    PointStruct(
                        id=id,
                        vector=vector,
                        payload=payload or {},
                    )
                ],
            )
            
        except Exception as e:
            logger.error(f"[VECTOR] Index failed: {e}")
            raise
    
    def index_batch(
        self,
        workspace_id: str,
        vectors: List[Dict[str, Any]],
        batch_size: int = 100,
    ):
        """
        Index multiple vectors efficiently.
        
        NOTE: Assumes collection already exists.
        
        Args:
            workspace_id: Workspace identifier
            vectors: List of {"id": str, "vector": List[float], "payload": Dict}
            batch_size: Batch size for upsert operations
        """
        if not vectors:
            return
        
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name not in self._memory_store:
                self._memory_store[name] = {}
            for v in vectors:
                self._memory_store[name][v["id"]] = (v["vector"], v.get("payload", {}))
            return
        
        try:
            from qdrant_client.models import PointStruct
            
            # Process in batches
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                points = [
                    PointStruct(
                        id=v["id"],
                        vector=v["vector"],
                        payload=v.get("payload", {}),
                    )
                    for v in batch
                ]
                
                self.client.upsert(
                    collection_name=name,
                    points=points,
                )
            
            logger.debug(f"[VECTOR] Indexed {len(vectors)} vectors to {name}")
            
        except Exception as e:
            logger.error(f"[VECTOR] Batch index failed: {e}")
            raise
    
    def search(
        self,
        workspace_id: str,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[VectorMatch]:
        """
        Search for similar vectors.
        
        Args:
            workspace_id: Workspace identifier
            vector: Query vector
            top_k: Number of results
            filter: Optional Qdrant filter conditions
            score_threshold: Minimum score threshold
        
        Returns:
            List of VectorMatch objects sorted by score descending
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            return self._memory_search(name, vector, top_k, score_threshold)
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter if provided
            qdrant_filter = None
            if filter:
                must_conditions = []
                for key, value in filter.items():
                    if value is not None:
                        must_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                if must_conditions:
                    qdrant_filter = Filter(must=must_conditions)
            
            results = self.client.search(
                collection_name=name,
                query_vector=vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )
            
            return [
                VectorMatch(
                    id=r.id,
                    score=r.score,
                    payload=r.payload or {},
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"[VECTOR] Search failed: {e}")
            return []
    
    def _memory_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        score_threshold: Optional[float],
    ) -> List[VectorMatch]:
        """In-memory search for testing."""
        import numpy as np
        
        if collection_name not in self._memory_store:
            return []
        
        query = np.array(query_vector)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            return []
        
        scores = []
        for id, (vector, payload) in self._memory_store[collection_name].items():
            v = np.array(vector)
            v_norm = np.linalg.norm(v)
            
            if v_norm == 0:
                continue
            
            score = float(np.dot(query, v) / (query_norm * v_norm))
            
            if score_threshold and score < score_threshold:
                continue
            
            scores.append(VectorMatch(id=id, score=score, payload=payload))
        
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]
    
    def get(
        self,
        workspace_id: str,
        id: str,
        with_vector: bool = False,
    ) -> Optional[VectorMatch]:
        """
        Get a single vector by ID.
        
        Args:
            workspace_id: Workspace identifier
            id: Vector ID
            with_vector: Include vector in result
        
        Returns:
            VectorMatch or None if not found
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name in self._memory_store and id in self._memory_store[name]:
                vector, payload = self._memory_store[name][id]
                return VectorMatch(
                    id=id,
                    score=1.0,
                    payload=payload,
                    vector=vector if with_vector else None,
                )
            return None
        
        try:
            results = self.client.retrieve(
                collection_name=name,
                ids=[id],
                with_vectors=with_vector,
            )
            
            for r in results:
                return VectorMatch(
                    id=r.id,
                    score=1.0,
                    payload=r.payload or {},
                    vector=r.vector if with_vector else None,
                )
            return None
            
        except Exception as e:
            logger.error(f"[VECTOR] Get failed: {e}")
            return None
    
    def delete(self, workspace_id: str, id: str):
        """
        Delete a vector by ID.
        
        Args:
            workspace_id: Workspace identifier
            id: Vector ID to delete
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name in self._memory_store:
                self._memory_store[name].pop(id, None)
            return
        
        try:
            from qdrant_client.models import PointIdsList
            
            self.client.delete(
                collection_name=name,
                points_selector=PointIdsList(points=[id]),
            )
            
        except Exception as e:
            logger.error(f"[VECTOR] Delete failed: {e}")
    
    def delete_batch(self, workspace_id: str, ids: List[str]):
        """
        Delete multiple vectors by ID.
        
        Args:
            workspace_id: Workspace identifier
            ids: List of vector IDs to delete
        """
        if not ids:
            return
        
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name in self._memory_store:
                for id in ids:
                    self._memory_store[name].pop(id, None)
            return
        
        try:
            from qdrant_client.models import PointIdsList
            
            self.client.delete(
                collection_name=name,
                points_selector=PointIdsList(points=ids),
            )
            
        except Exception as e:
            logger.error(f"[VECTOR] Batch delete failed: {e}")
    
    def delete_by_filter(
        self,
        workspace_id: str,
        filter: Dict[str, Any],
    ) -> int:
        """
        Delete vectors matching filter.
        
        Args:
            workspace_id: Workspace identifier
            filter: Qdrant filter conditions
        
        Returns:
            Approximate number of deleted vectors
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            # Memory backend doesn't support filters well
            return 0
        
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Build filter
            must_conditions = []
            for key, value in filter.items():
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            
            self.client.delete(
                collection_name=name,
                points_selector=Filter(must=must_conditions),
            )
            
            return -1  # Qdrant doesn't return count
            
        except Exception as e:
            logger.error(f"[VECTOR] Filter delete failed: {e}")
            return 0
    
    def delete_collection(self, workspace_id: str):
        """
        Delete entire collection (GDPR).
        
        Args:
            workspace_id: Workspace identifier
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            self._memory_store.pop(name, None)
            self._known_collections.pop(workspace_id, None)
            return
        
        try:
            self.client.delete_collection(name)
            
            # Remove from cache
            with self._collection_lock:
                self._known_collections.pop(workspace_id, None)
            
            logger.info(f"[VECTOR] Deleted collection: {name}")
            
        except Exception as e:
            logger.error(f"[VECTOR] Failed to delete collection: {e}")
    
    def count(self, workspace_id: str) -> int:
        """
        Count vectors in workspace.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            Number of vectors
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            return len(self._memory_store.get(name, {}))
        
        try:
            info = self.client.get_collection(name)
            return info.vectors_count or 0
        except Exception:
            return 0
    
    def get_collection_info(self, workspace_id: str) -> Optional[CollectionInfo]:
        """
        Get collection information.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            CollectionInfo or None
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name in self._memory_store:
                dim = self._known_collections.get(workspace_id, (384, 0))[0]
                return CollectionInfo(
                    name=name,
                    vectors_count=len(self._memory_store[name]),
                    dimension=dim,
                    distance="cosine",
                )
            return None
        
        try:
            info = self.client.get_collection(name)
            return CollectionInfo(
                name=name,
                vectors_count=info.vectors_count or 0,
                dimension=info.config.params.vectors.size,
                distance=str(info.config.params.vectors.distance),
            )
        except Exception:
            return None
    
    def list_collections(self) -> List[str]:
        """List all workspace collections."""
        if self.backend == VectorStoreBackend.MEMORY:
            return list(self._memory_store.keys())
        
        try:
            collections = self.client.get_collections().collections
            return [
                c.name for c in collections
                if c.name.startswith(self.collection_prefix)
            ]
        except Exception:
            return []
    
    def health_check(self) -> bool:
        """Check if vector store is healthy."""
        if self.backend == VectorStoreBackend.MEMORY:
            return True
        
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get collection cache statistics.
        
        Returns:
            Cache stats including hit rate, size, TTL
        """
        now = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for workspace_id, (dim, cached_at) in self._known_collections.items():
            if now - cached_at < self.cache_ttl:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_cached": len(self._known_collections),
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "ttl_seconds": self.cache_ttl,
        }
    
    def close(self):
        """Close connection."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __repr__(self) -> str:
        return f"VectorStore(url='{self.url}', backend={self.backend.value}, cache_ttl={self.cache_ttl}s)"
