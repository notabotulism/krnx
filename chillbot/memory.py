"""
Chillbot Memory - Simple Interface

The "it just works" interface for AI memory.
Three lines to durable, semantic memory.

Usage:
    from chillbot import Memory
    
    memory = Memory("my-agent")
    memory.remember("user loves hiking in the Alps")
    memories = memory.recall("outdoor activities")
    context = memory.context("plan a weekend trip", max_tokens=4000)

For power users who need infrastructure access:
    from chillbot.kernel import KRNXClient
    from chillbot.compute import EmbeddingEngine, VectorStore
    from chillbot.fabric import MemoryFabric
"""

import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class Memory:
    """
    Simple memory interface.
    
    Wraps the full Chillbot stack (Kernel + Compute + Fabric)
    into a dead-simple API.
    
    Features:
    - Automatic embedding and indexing
    - Semantic search
    - LLM-ready context building
    - Time travel (replay)
    """
    
    def __init__(
        self,
        agent_id: str,
        
        # Connection options
        api_key: Optional[str] = None,  # For hosted mode
        api_url: Optional[str] = None,  # For self-hosted
        
        # Local mode options
        data_path: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        qdrant_url: str = "http://localhost:6333",
        
        # Model options
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize Memory.
        
        For hosted (coming soon):
            memory = Memory("my-agent", api_key="chillbot_xxx")
        
        For local:
            memory = Memory("my-agent", data_path="./data")
        
        Args:
            agent_id: Unique identifier for this agent/application
            api_key: API key for hosted Chillbot (coming soon)
            api_url: Custom API URL for self-hosted
            data_path: Local data directory (enables local mode)
            redis_host: Redis host for local mode
            redis_port: Redis port for local mode
            qdrant_url: Qdrant URL for local mode
            embedding_model: Sentence transformer model
        """
        self.agent_id = agent_id
        self._mode = None
        self._fabric = None
        self._kernel = None
        self._embeddings = None
        self._vectors = None
        self._job_queue = None
        
        if api_key or api_url:
            # Hosted/remote mode
            self._init_remote(api_key, api_url)
        elif data_path:
            # Local mode
            self._init_local(
                data_path=data_path,
                redis_host=redis_host,
                redis_port=redis_port,
                qdrant_url=qdrant_url,
                embedding_model=embedding_model,
            )
        else:
            # Default to local with temp storage
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix=f"chillbot_{agent_id}_")
            logger.warning(f"[MEMORY] No data_path provided, using temp: {temp_dir}")
            self._init_local(
                data_path=temp_dir,
                redis_host=redis_host,
                redis_port=redis_port,
                qdrant_url=qdrant_url,
                embedding_model=embedding_model,
            )
    
    def _init_remote(self, api_key: Optional[str], api_url: Optional[str]):
        """Initialize in hosted/remote mode."""
        from chillbot.fabric import MemoryFabric
        
        self._mode = "remote"
        
        base_url = api_url or "https://api.chillbot.io"
        
        self._fabric = MemoryFabric(
            kernel_url=base_url,
            default_workspace=self.agent_id,
        )
        
        # TODO: Add API key header handling
        
        logger.info(f"[MEMORY] Initialized in remote mode (agent={self.agent_id})")
    
    def _init_local(
        self,
        data_path: str,
        redis_host: str,
        redis_port: int,
        qdrant_url: str,
        embedding_model: str,
    ):
        """Initialize in local mode with full stack."""
        self._mode = "local"
        
        # Initialize kernel
        try:
            from chillbot.kernel import KRNXController
            
            self._kernel = KRNXController(
                data_path=data_path,
                redis_host=redis_host,
                redis_port=redis_port,
                enable_async_worker=True,
            )
            logger.info("[MEMORY] Kernel initialized")
        except Exception as e:
            logger.warning(f"[MEMORY] Kernel init failed: {e}")
            self._kernel = None
        
        # Initialize compute components
        try:
            from chillbot.compute import EmbeddingEngine, VectorStore, JobQueue
            
            self._embeddings = EmbeddingEngine(model_name=embedding_model)
            self._vectors = VectorStore(url=qdrant_url)
            
            # ==============================================
            # FIX: Pre-warm collection BEFORE any threads start
            # This prevents 50-thread race condition on ensure_collection()
            # ==============================================
            if self._vectors and self._embeddings:
                # Force model load now (single-threaded, safe)
                _ = self._embeddings.dimension  # This triggers lazy load check
                
                # Pre-create collection for this agent
                self._vectors.ensure_collection(
                    self.agent_id, 
                    self._embeddings.dimension
                )
                logger.info(f"[MEMORY] Pre-warmed collection for {self.agent_id} (dim={self._embeddings.dimension})")
            
            self._job_queue = JobQueue()  # Redis-based, uses global connection pool
            logger.info("[MEMORY] Compute components initialized")
        except Exception as e:
            logger.warning(f"[MEMORY] Compute init failed: {e}")
            self._embeddings = None
            self._vectors = None
            self._job_queue = None
        
        # Initialize fabric
        from chillbot.fabric import MemoryFabric
        
        self._fabric = MemoryFabric(
            kernel=self._kernel,
            job_queue=self._job_queue,
            embeddings=self._embeddings,
            vectors=self._vectors,
            default_workspace=self.agent_id,
            auto_embed=True,
        )
        
        logger.info(f"[MEMORY] Initialized in local mode (agent={self.agent_id})")
    
    # ==============================================
    # SIMPLE API
    # ==============================================
    
    def remember(
        self,
        content: Union[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        channel: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Store a memory.
        
        Args:
            content: Memory content (string or dict)
            metadata: Additional metadata
            channel: Event channel for filtering
            user_id: User identifier (optional)
        
        Returns:
            event_id
        
        Example:
            memory.remember("User prefers dark mode")
            memory.remember({"preference": "dark_mode", "value": True})
        """
        return self._fabric.remember(
            content=content,
            workspace_id=self.agent_id,
            user_id=user_id,
            channel=channel,
            metadata=metadata,
        )
    
    def recall(
        self,
        query: str,
        top_k: int = 10,
        user_id: Optional[str] = None,
        include_recent: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Recall relevant memories.
        
        Args:
            query: Search query
            top_k: Maximum results
            user_id: Filter by user
            include_recent: Include recent memories from kernel
        
        Returns:
            List of memories with scores
        
        Example:
            memories = memory.recall("what are their preferences?")
            for m in memories:
                print(f"{m['score']:.2f}: {m['content']}")
        """
        result = self._fabric.recall(
            query=query,
            workspace_id=self.agent_id,
            user_id=user_id,
            top_k=top_k,
            include_recent=include_recent,
        )
        
        return [m.to_dict() for m in result.memories]
    
    def context(
        self,
        query: str,
        max_tokens: int = 4000,
        format: str = "text",
        user_id: Optional[str] = None,
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Build LLM-ready context from memories.
        
        Args:
            query: Query to find relevant memories
            max_tokens: Token budget
            format: Output format ('text', 'json', 'messages')
            user_id: Filter by user
        
        Returns:
            Formatted context
        
        Example:
            context = memory.context("help them plan a trip", max_tokens=4000)
            # Use context in your LLM prompt
        """
        return self._fabric.context(
            query=query,
            workspace_id=self.agent_id,
            user_id=user_id,
            max_tokens=max_tokens,
            format=format,
        )
    
    def replay(
        self,
        timestamp: float,
        user_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """
        Time travel: Get all memories up to a timestamp.
        
        Args:
            timestamp: Target timestamp
            user_id: User to replay
        
        Returns:
            List of events in chronological order
        
        Example:
            import time
            yesterday = time.time() - 86400
            history = memory.replay(yesterday)
        """
        if self._kernel:
            events = self._kernel.replay_to_timestamp(
                workspace_id=self.agent_id,
                user_id=user_id,
                timestamp=timestamp,
            )
            return [e.to_dict() for e in events]
        else:
            # Remote mode - use fabric
            return []  # TODO: Implement remote replay
    
    # ==============================================
    # MANAGEMENT
    # ==============================================
    
    def forget(self, event_id: str):
        """
        Delete a specific memory.
        
        Args:
            event_id: Event to delete
        """
        # TODO: Implement single event deletion
        raise NotImplementedError("Single event deletion not yet implemented")
    
    def forget_all(self, user_id: Optional[str] = None):
        """
        Delete all memories (GDPR erase).
        
        Args:
            user_id: If provided, only delete this user's memories
        """
        if self._kernel:
            if user_id:
                self._kernel.erase_user(self.agent_id, user_id)
            else:
                self._kernel.erase_workspace(self.agent_id)
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            "agent_id": self.agent_id,
            "mode": self._mode,
        }
        
        if self._kernel:
            stats["kernel"] = self._kernel.get_stats()
        
        if self._vectors:
            stats["vectors"] = {
                "count": self._vectors.count(self.agent_id),
            }
        
        if self._fabric:
            stats["fabric"] = self._fabric.get_stats()
        
        return stats
    
    def close(self):
        """Close all connections."""
        if self._fabric:
            self._fabric.close()
        
        if self._kernel:
            self._kernel.close()
        
        if self._vectors:
            self._vectors.close()
        
        if self._job_queue:
            self._job_queue.close()
        
        logger.info(f"[MEMORY] Closed (agent={self.agent_id})")
    
    def __repr__(self) -> str:
        return f"Memory(agent_id='{self.agent_id}', mode='{self._mode}')"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
