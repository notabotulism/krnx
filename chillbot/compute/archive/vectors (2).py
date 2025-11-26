"""
KRNX Compute - Vector Store

Qdrant-backed vector storage for semantic search.
Supports multiple workspaces, filtering, and batch operations.

Usage:
    vectors = VectorStore(url="http://localhost:6333")
    
    # Ensure collection exists
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
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_prefix: str = "krnx",
        backend: VectorStoreBackend = VectorStoreBackend.QDRANT,
    ):
        """
        Initialize vector store.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for Qdrant Cloud
            collection_prefix: Prefix for collection names
            backend: Which backend to use (qdrant or memory)
        """
        self.url = url
        self.api_key = api_key
        self.collection_prefix = collection_prefix
        self.backend = backend
        self._client = None
        self._client_lock = threading.Lock()  # Thread-safe client initialization
        self._collections_cache: Dict[str, int] = {}  # workspace -> dimension
        
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
        Create collection if not exists.
        
        Args:
            workspace_id: Workspace identifier
            dimension: Vector dimension (must match embedding model)
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name not in self._memory_store:
                self._memory_store[name] = {}
                self._collections_cache[workspace_id] = dimension
            return
        
        # Check cache first
        if workspace_id in self._collections_cache:
            return
        
        try:
            from qdrant_client.models import Distance, VectorParams
            
            # Check if exists
            collections = self.client.get_collections().collections
            exists = any(c.name == name for c in collections)
            
            if not exists:
                logger.info(f"[VECTOR] Creating collection: {name} (dim={dimension})")
                
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                )
            
            self._collections_cache[workspace_id] = dimension
            
        except Exception as e:
            logger.error(f"[VECTOR] Failed to ensure collection: {e}")
            raise
    
    def index(
        self,
        workspace_id: str,
        id: str,
        vector: List[float],
        payload: Optional[Dict[str, Any]] = None,
    ):
        """
        Index a single vector.
        
        Args:
            workspace_id: Workspace identifier
            id: Unique vector ID (typically event_id)
            vector: Embedding vector
            payload: Metadata to store with vector
        """
        name = self._collection_name(workspace_id)
        payload = payload or {}
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name not in self._memory_store:
                self._memory_store[name] = {}
            self._memory_store[name][id] = (vector, payload)
            return
        
        try:
            from qdrant_client.models import PointStruct
            
            self.client.upsert(
                collection_name=name,
                points=[
                    PointStruct(
                        id=id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            
        except Exception as e:
            logger.error(f"[VECTOR] Failed to index: {e}")
            raise
    
    def index_batch(
        self,
        workspace_id: str,
        points: List[tuple],  # [(id, vector, payload), ...]
    ):
        """
        Batch index multiple vectors.
        
        Args:
            workspace_id: Workspace identifier
            points: List of (id, vector, payload) tuples
        """
        if not points:
            return
        
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name not in self._memory_store:
                self._memory_store[name] = {}
            for id, vector, payload in points:
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
                    for id, vector, payload in points
                ],
            )
            
            logger.debug(f"[VECTOR] Indexed {len(points)} vectors to {name}")
            
        except Exception as e:
            logger.error(f"[VECTOR] Failed to batch index: {e}")
            raise
    
    def search(
        self,
        workspace_id: str,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False,
    ) -> List[VectorMatch]:
        """
        Search for similar vectors.
        
        Args:
            workspace_id: Workspace identifier
            vector: Query vector
            top_k: Maximum results to return
            filter: Qdrant filter conditions
            score_threshold: Minimum score threshold
            with_vectors: Include vectors in results
        
        Returns:
            List of VectorMatch results
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            return self._memory_search(name, vector, top_k, score_threshold)
        
        try:
            # Try new API first (qdrant-client >= 1.7)
            try:
                results = self.client.query_points(
                    collection_name=name,
                    query=vector,
                    limit=top_k,
                    query_filter=filter,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=with_vectors,
                ).points
            except AttributeError:
                # Fall back to old API (qdrant-client < 1.7)
                results = self.client.search(
                    collection_name=name,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=filter,
                    score_threshold=score_threshold,
                    with_vectors=with_vectors,
                )
            
            return [
                VectorMatch(
                    id=str(r.id),
                    score=r.score,
                    payload=r.payload or {},
                    vector=r.vector if with_vectors else None,
                )
                for r in results
            ]
            
        except Exception as e:
            logger.error(f"[VECTOR] Search failed: {e}")
            return []
    
    def _memory_search(
        self,
        collection_name: str,
        vector: List[float],
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> List[VectorMatch]:
        """In-memory search for testing."""
        import numpy as np
        
        if collection_name not in self._memory_store:
            return []
        
        query = np.array(vector)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            return []
        
        results = []
        for id, (vec, payload) in self._memory_store[collection_name].items():
            v = np.array(vec)
            v_norm = np.linalg.norm(v)
            
            if v_norm == 0:
                continue
            
            score = float(np.dot(query, v) / (query_norm * v_norm))
            
            if score_threshold is not None and score < score_threshold:
                continue
            
            results.append(VectorMatch(id=id, score=score, payload=payload))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
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
            VectorMatch or None
        """
        name = self._collection_name(workspace_id)
        
        if self.backend == VectorStoreBackend.MEMORY:
            if name in self._memory_store and id in self._memory_store[name]:
                vec, payload = self._memory_store[name][id]
                return VectorMatch(
                    id=id,
                    score=1.0,
                    payload=payload,
                    vector=vec if with_vector else None,
                )
            return None
        
        try:
            results = self.client.retrieve(
                collection_name=name,
                ids=[id],
                with_vectors=with_vector,
            )
            
            if results:
                r = results[0]
                return VectorMatch(
                    id=str(r.id),
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
            self._collections_cache.pop(workspace_id, None)
            return
        
        try:
            self.client.delete_collection(name)
            self._collections_cache.pop(workspace_id, None)
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
                return CollectionInfo(
                    name=name,
                    vectors_count=len(self._memory_store[name]),
                    dimension=self._collections_cache.get(workspace_id, 384),
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
    
    def close(self):
        """Close connection."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __repr__(self) -> str:
        return f"VectorStore(url='{self.url}', backend={self.backend.value})"
