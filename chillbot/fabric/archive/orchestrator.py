"""
Memory Fabric - Orchestrator

The orchestration layer that ties Kernel and Compute together.
Routes requests, manages the remember→embed→index pipeline.

Philosophy:
- No opinions, just orchestration
- Kernel owns events, Compute owns vectors
- Fabric routes and coordinates

FIXED: Removed ensure_collection from remember() hot path.
Collection is now pre-warmed during Memory initialization.

Usage:
    from chillbot.fabric import MemoryFabric
    
    fabric = MemoryFabric(kernel_url="http://localhost:6380")
    
    # Remember (writes to kernel + enqueues embedding job)
    event_id = fabric.remember(
        workspace_id="my-app",
        user_id="user_1",
        content="User loves hiking in the Alps"
    )
    
    # Recall (searches vectors, enriches from kernel)
    memories = fabric.recall(
        workspace_id="my-app",
        query="outdoor activities"
    )
    
    # Context (builds LLM-ready context)
    context = fabric.context(
        workspace_id="my-app",
        user_id="user_1",
        query="plan a weekend trip",
        max_tokens=4000
    )
"""

import time
import uuid
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A retrieved memory with metadata."""
    event_id: str
    content: Union[str, Dict[str, Any]]
    timestamp: float
    score: float = 1.0  # Relevance score (0-1)
    source: str = "unknown"  # 'stm', 'ltm', 'vector'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "score": self.score,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @property
    def age_hours(self) -> float:
        return (time.time() - self.timestamp) / 3600
    
    @property
    def age_days(self) -> float:
        return self.age_hours / 24


@dataclass
class RecallResult:
    """Result of a recall operation."""
    memories: List[MemoryItem]
    query: str
    workspace_id: str
    latency_ms: float
    sources_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memories": [m.to_dict() for m in self.memories],
            "query": self.query,
            "workspace_id": self.workspace_id,
            "latency_ms": self.latency_ms,
            "sources_used": self.sources_used,
            "count": len(self.memories),
        }


class MemoryFabric:
    """
    Memory Fabric - Orchestration Layer.
    
    Routes between Kernel (events) and Compute (vectors/embeddings).
    Provides unified interface for memory operations.
    
    Architecture:
        remember() → Kernel.write_event() + Compute.enqueue(EMBED)
        recall()   → Compute.search() + Kernel.get_events()
        context()  → recall() + ContextBuilder.build()
    
    IMPORTANT: Collection must be pre-warmed before using remember().
    The Memory class handles this during initialization.
    """
    
    def __init__(
        self,
        # Kernel connection (direct or via HTTP)
        kernel=None,  # KRNXController instance
        kernel_url: Optional[str] = None,  # Or HTTP URL
        
        # Compute components
        job_queue=None,  # JobQueue instance
        embeddings=None,  # EmbeddingEngine instance
        vectors=None,  # VectorStore instance
        salience=None,  # SalienceEngine instance
        
        # Configuration
        default_workspace: str = "default",
        auto_embed: bool = True,  # Auto-enqueue embedding jobs
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize Memory Fabric.
        
        Args:
            kernel: Direct KRNXController instance (for local mode)
            kernel_url: HTTP URL for kernel API (for remote mode)
            job_queue: Compute job queue
            embeddings: Embedding engine (for local embedding)
            vectors: Vector store
            salience: Salience scoring engine
            default_workspace: Default workspace ID
            auto_embed: Automatically enqueue embedding jobs on remember()
            embedding_model: Model name for embeddings
        """
        self.kernel = kernel
        self.kernel_url = kernel_url
        self.job_queue = job_queue
        self.embeddings = embeddings
        self.vectors = vectors
        self.salience = salience
        
        self.default_workspace = default_workspace
        self.auto_embed = auto_embed
        self.embedding_model = embedding_model
        
        # HTTP client for remote kernel (lazy init)
        self._http_client = None
        
        # Stats
        self._stats = {
            "remembers": 0,
            "recalls": 0,
            "contexts_built": 0,
        }
        
        logger.info(f"[FABRIC] Initialized (workspace={default_workspace}, auto_embed={auto_embed})")
    
    # ==============================================
    # REMEMBER (Write Path)
    # ==============================================
    
    def remember(
        self,
        content: Union[str, Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        priority: int = 0,
    ) -> str:
        """
        Store a memory.
        
        Pipeline:
        1. Write event to Kernel (STM + LTM queue)
        2. Enqueue embedding job to Compute (if auto_embed)
        
        IMPORTANT: Collection must be pre-warmed before calling this method.
        The Memory class handles this during initialization.
        
        Args:
            content: Memory content (string or dict)
            workspace_id: Workspace (default: self.default_workspace)
            user_id: User identifier (default: "default")
            session_id: Session identifier (optional)
            channel: Event channel for filtering
            metadata: Additional metadata
            ttl_seconds: Time-to-live (None = forever)
            priority: Embedding job priority (higher = first)
        
        Returns:
            event_id
        """
        workspace_id = workspace_id or self.default_workspace
        user_id = user_id or "default"
        session_id = session_id or f"{workspace_id}_{user_id}"
        
        # Generate event ID
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        timestamp = time.time()
        
        # Normalize content to dict
        if isinstance(content, str):
            content_dict = {"text": content}
        else:
            content_dict = content
        
        # 1. Write to Kernel
        if self.kernel:
            # Local mode - direct kernel access
            from chillbot.kernel import Event
            
            event = Event(
                event_id=event_id,
                workspace_id=workspace_id,
                user_id=user_id,
                session_id=session_id,
                content=content_dict,
                timestamp=timestamp,
                channel=channel,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {},
            )
            
            # ==============================================
            # FIX: REMOVED ensure_collection from hot path!
            # Collection is pre-warmed in Memory._init_local()
            # This was causing 50-thread race condition/freeze
            # ==============================================
            # OLD CODE (caused freeze):
            # if self.vectors and self.embeddings:
            #     self.vectors.ensure_collection(workspace_id, self.embeddings.dimension)
            
            self.kernel.write_event(workspace_id, user_id, event)
            logger.debug(f"[FABRIC] Wrote event {event_id} to kernel")
        
        elif self.kernel_url:
            # Remote mode - HTTP API
            self._write_event_http(
                event_id=event_id,
                workspace_id=workspace_id,
                user_id=user_id,
                session_id=session_id,
                content=content_dict,
                channel=channel,
                metadata=metadata,
                ttl_seconds=ttl_seconds,
            )
            logger.debug(f"[FABRIC] Wrote event {event_id} via HTTP")
        
        else:
            raise RuntimeError("No kernel configured (need kernel or kernel_url)")
        
        # 2. Enqueue embedding job (if enabled and queue available)
        if self.auto_embed and self.job_queue:
            from chillbot.compute import JobType
            
            # Extract text for embedding
            text = self._extract_text(content_dict)
            
            if text:
                self.job_queue.enqueue(
                    job_type=JobType.EMBED,
                    workspace_id=workspace_id,
                    payload={
                        "event_id": event_id,
                        "text": text,
                        "metadata": {
                            "user_id": user_id,
                            "channel": channel,
                            "timestamp": timestamp,
                        },
                    },
                    priority=priority,
                )
                logger.debug(f"[FABRIC] Enqueued embedding job for {event_id}")
        
        self._stats["remembers"] += 1
        return event_id
    
    def remember_batch(
        self,
        items: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Store multiple memories efficiently.
        
        Args:
            items: List of {"content": ..., "metadata": ..., ...}
            workspace_id: Default workspace for all items
            user_id: Default user for all items
        
        Returns:
            List of event_ids
        """
        event_ids = []
        
        for item in items:
            event_id = self.remember(
                content=item.get("content"),
                workspace_id=item.get("workspace_id", workspace_id),
                user_id=item.get("user_id", user_id),
                channel=item.get("channel"),
                metadata=item.get("metadata"),
            )
            event_ids.append(event_id)
        
        return event_ids
    
    # ==============================================
    # RECALL (Read Path)
    # ==============================================
    
    def recall(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: int = 10,
        min_score: float = 0.0,
        include_recent: bool = True,
        recent_hours: int = 24,
        channel: Optional[str] = None,
    ) -> RecallResult:
        """
        Recall relevant memories.
        
        Search strategy:
        1. Vector search (semantic similarity)
        2. Recent events from kernel (time-based)
        3. Merge, dedupe, rank by score
        
        Args:
            query: Search query
            workspace_id: Workspace to search
            user_id: Optional user filter
            top_k: Maximum results
            min_score: Minimum relevance score
            include_recent: Include recent events from kernel
            recent_hours: How far back to look for recent events
            channel: Filter by channel
        
        Returns:
            RecallResult with ranked memories
        """
        start_time = time.time()
        workspace_id = workspace_id or self.default_workspace
        
        memories: List[MemoryItem] = []
        sources_used: List[str] = []
        
        # 1. Vector search (if available)
        if self.vectors and self.embeddings:
            try:
                # Embed query
                query_vector = self.embeddings.embed(query)
                
                # Search
                matches = self.vectors.search(
                    workspace_id=workspace_id,
                    vector=query_vector,
                    top_k=top_k * 2,  # Get more, filter later
                    filter={"user_id": user_id} if user_id else None,
                )
                
                for match in matches:
                    if match.score >= min_score:
                        memories.append(MemoryItem(
                            event_id=match.id,
                            content=match.payload.get("text_preview", ""),
                            timestamp=match.payload.get("timestamp", 0),
                            score=match.score,
                            source="vector",
                            metadata=match.payload,
                        ))
                
                sources_used.append("vector")
                logger.debug(f"[FABRIC] Vector search returned {len(matches)} matches")
            
            except Exception as e:
                logger.warning(f"[FABRIC] Vector search failed: {e}")
        
        # 2. Recent events from kernel (if enabled)
        if include_recent and self.kernel:
            try:
                # Calculate time range
                end_time = time.time()
                start_time_filter = end_time - (recent_hours * 3600)
                
                recent_events = self.kernel.query_events(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    start_time=start_time_filter,
                    end_time=end_time,
                    limit=top_k,
                )
                
                for event in recent_events:
                    # Check channel filter
                    if channel and event.channel != channel:
                        continue
                    
                    # Calculate recency score (newer = higher)
                    age_hours = (end_time - event.timestamp) / 3600
                    recency_score = max(0.1, 1.0 - (age_hours / recent_hours))
                    
                    memories.append(MemoryItem(
                        event_id=event.event_id,
                        content=event.content,
                        timestamp=event.timestamp,
                        score=recency_score,
                        source="stm" if age_hours < 24 else "ltm",
                        metadata=event.metadata,
                    ))
                
                sources_used.append("kernel")
                logger.debug(f"[FABRIC] Kernel returned {len(recent_events)} recent events")
            
            except Exception as e:
                logger.warning(f"[FABRIC] Kernel query failed: {e}")
        
        # 3. Deduplicate by event_id
        seen_ids = set()
        unique_memories = []
        for mem in memories:
            if mem.event_id not in seen_ids:
                seen_ids.add(mem.event_id)
                unique_memories.append(mem)
        
        # 4. Sort by score descending
        unique_memories.sort(key=lambda m: m.score, reverse=True)
        
        # 5. Apply limit
        final_memories = unique_memories[:top_k]
        
        latency_ms = (time.time() - start_time) * 1000
        
        self._stats["recalls"] += 1
        
        return RecallResult(
            memories=final_memories,
            query=query,
            workspace_id=workspace_id,
            latency_ms=latency_ms,
            sources_used=sources_used,
        )
    
    # ==============================================
    # CONTEXT (LLM-Ready)
    # ==============================================
    
    def context(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_tokens: int = 4000,
        format: str = "text",  # 'text', 'json', 'messages'
        include_metadata: bool = False,
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Build LLM-ready context from memories.
        
        Args:
            query: Query to find relevant memories
            workspace_id: Workspace to search
            user_id: Optional user filter
            max_tokens: Approximate token budget
            format: Output format ('text', 'json', 'messages')
            include_metadata: Include memory metadata
        
        Returns:
            Formatted context based on format parameter
        """
        from chillbot.fabric.context import ContextBuilder
        
        # Recall relevant memories
        result = self.recall(
            query=query,
            workspace_id=workspace_id,
            user_id=user_id,
            top_k=50,  # Get more, builder will trim
        )
        
        # Build context
        builder = ContextBuilder(max_tokens=max_tokens)
        context = builder.build(
            memories=result.memories,
            query=query,
            format=format,
            include_metadata=include_metadata,
        )
        
        self._stats["contexts_built"] += 1
        
        return context
    
    # ==============================================
    # ENRICHMENT
    # ==============================================
    
    def enrich(
        self,
        event_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """
        Enrich a memory item with full event data from kernel.
        
        Used after vector search to get complete content.
        
        Args:
            event_id: Event to enrich
            workspace_id: Workspace (for vector lookup)
        
        Returns:
            Enriched MemoryItem or None
        """
        workspace_id = workspace_id or self.default_workspace
        
        # Try kernel first
        if self.kernel:
            event = self.kernel.get_event(event_id)
            if event:
                return MemoryItem(
                    event_id=event.event_id,
                    content=event.content,
                    timestamp=event.timestamp,
                    score=1.0,
                    source="kernel",
                    metadata=event.metadata,
                )
        
        # Fall back to vector payload
        if self.vectors:
            match = self.vectors.get(workspace_id, event_id)
            if match:
                return MemoryItem(
                    event_id=match.id,
                    content=match.payload.get("text_preview", ""),
                    timestamp=match.payload.get("timestamp", 0),
                    score=1.0,
                    source="vector",
                    metadata=match.payload,
                )
        
        return None
    
    # ==============================================
    # UTILITY
    # ==============================================
    
    def _extract_text(self, content: Dict[str, Any]) -> Optional[str]:
        """Extract embeddable text from content dict."""
        # Try common text fields
        for field in ["text", "message", "content", "body", "query", "response"]:
            if field in content and isinstance(content[field], str):
                return content[field]
        
        # Try chat format
        if "role" in content and "content" in content:
            return content["content"]
        
        # Combine query + response
        if "query" in content and "response" in content:
            return f"{content['query']} {content['response']}"
        
        return None
    
    def _write_event_http(
        self,
        event_id: str,
        workspace_id: str,
        user_id: str,
        session_id: str,
        content: Dict[str, Any],
        channel: Optional[str],
        metadata: Optional[Dict[str, Any]],
        ttl_seconds: Optional[int],
    ):
        """Write event via HTTP API."""
        import httpx
        
        if self._http_client is None:
            self._http_client = httpx.Client(base_url=self.kernel_url, timeout=10.0)
        
        response = self._http_client.post(
            "/api/v1/events/write",
            json={
                "workspace_id": workspace_id,
                "user_id": user_id,
                "session_id": session_id,
                "content": content,
                "channel": channel,
                "metadata": metadata or {},
                "ttl_seconds": ttl_seconds,
            },
        )
        response.raise_for_status()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fabric statistics."""
        return {
            **self._stats,
            "kernel_mode": "local" if self.kernel else ("remote" if self.kernel_url else "none"),
            "auto_embed": self.auto_embed,
        }
    
    def close(self):
        """Close connections."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
    
    def __repr__(self) -> str:
        mode = "local" if self.kernel else ("remote" if self.kernel_url else "none")
        return f"MemoryFabric(mode={mode}, workspace={self.default_workspace})"
