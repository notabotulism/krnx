"""
Memory Fabric - Orchestrator v2 TURBO + Enrichment

PERFORMANCE FIX: Uses single-pipeline writes.
Before: 3 round-trips per remember() = ~3ms + contention
After:  1 round-trip per remember() = ~1ms, minimal contention

NEW: Metadata enrichment pipeline
- Temporal (episode threading, time gaps)
- Salience scoring
- Relation detection (replies_to, supersedes, etc.)
- Entity extraction
- Retention signals

FIX v3: Synchronous embedding fallback
- When embeddings + vectors are available but job_queue is None,
  embedding and indexing happens inline in remember()
- This ensures vectors work without requiring async job infrastructure
- The async job_queue path remains available as an optimization

Expected improvement: 443 events/sec → 1500+ events/sec
"""

import time
import uuid
import json
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
    score: float = 1.0
    source: str = "unknown"
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
    Memory Fabric - Orchestration Layer v2 TURBO + Enrichment
    
    Uses single-pipeline writes for 3x throughput.
    Includes metadata enrichment for intelligent memory.
    
    FIX v3: Supports synchronous embedding when job_queue is None.
    """
    
    def __init__(
        self,
        kernel=None,
        kernel_url: Optional[str] = None,
        job_queue=None,
        embeddings=None,
        vectors=None,
        salience=None,
        default_workspace: str = "default",
        auto_embed: bool = True,
        embedding_model: str = "all-MiniLM-L6-v2",
        # NEW: Enrichment options
        auto_enrich: bool = True,
        enrichment_config: Optional[Any] = None,
    ):
        self.kernel = kernel
        self.kernel_url = kernel_url
        self.job_queue = job_queue
        self.embeddings = embeddings
        self.vectors = vectors
        self.salience = salience
        
        self.default_workspace = default_workspace
        self.auto_embed = auto_embed
        self.embedding_model = embedding_model
        
        self._http_client = None
        
        # NEW: Enrichment pipeline
        self.auto_enrich = auto_enrich
        self._enricher = None
        self._enrichment_config = enrichment_config
        
        if auto_enrich:
            self._init_enricher()
        
        # Track previous events per workspace:user for temporal enrichment
        self._previous_events: Dict[str, Any] = {}
        
        # Track which collections have been initialized (for sync embedding)
        self._initialized_collections: Dict[str, int] = {}  # workspace_id -> dimension
        
        self._stats = {
            "remembers": 0,
            "recalls": 0,
            "contexts_built": 0,
            "enrichments": 0,
            "sync_embeds": 0,  # NEW: Track sync embeddings
        }
        
        logger.info(f"[FABRIC] Initialized TURBO (workspace={default_workspace}, auto_embed={auto_embed}, auto_enrich={auto_enrich})")
    
    def _init_enricher(self):
        """Initialize the metadata enrichment pipeline."""
        try:
            from chillbot.fabric.enrichment import MetadataEnricher, EnrichmentConfig
            
            config = self._enrichment_config or EnrichmentConfig()
            self._enricher = MetadataEnricher(config=config)
            logger.info("[FABRIC] Enrichment pipeline initialized")
        except ImportError as e:
            logger.warning(f"[FABRIC] Enrichment not available: {e}")
            self._enricher = None
            self.auto_enrich = False
    
    # ==============================================
    # REMEMBER (Write Path) - TURBO + ENRICHMENT
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
        # NEW: Enrichment options
        enrich: Optional[bool] = None,
        similarity_scores: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Store a memory using TURBO single-pipeline write.
        
        Before: 3 Redis round-trips (STM + LTM queue + job queue)
        After:  1 Redis round-trip (all combined)
        
        NEW: Automatic metadata enrichment (temporal, salience, relations, entities)
        
        FIX v3: When auto_embed=True and embeddings+vectors are available but
        job_queue is None, embedding happens synchronously inline.
        """
        workspace_id = workspace_id or self.default_workspace
        user_id = user_id or "default"
        session_id = session_id or f"{workspace_id}_{user_id}"
        
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        timestamp = time.time()
        
        # Normalize content
        if isinstance(content, str):
            content_dict = {"text": content}
        else:
            content_dict = content
        
        # Initialize metadata
        event_metadata = metadata.copy() if metadata else {}
        
        # ==============================================
        # NEW: METADATA ENRICHMENT
        # ==============================================
        should_enrich = enrich if enrich is not None else self.auto_enrich
        
        if should_enrich and self._enricher:
            try:
                from chillbot.fabric.enrichment import EnrichmentContext
                
                # Get previous event for this stream
                stream_key = f"{workspace_id}:{user_id}"
                previous_event = self._previous_events.get(stream_key)
                
                # Build enrichment context
                context = EnrichmentContext(
                    previous_event=previous_event,
                    similarity_scores=similarity_scores,
                    access_count=0,  # New event, no accesses yet
                    now=timestamp,
                )
                
                # Create a temporary event-like object for enrichment
                class TempEvent:
                    pass
                temp_event = TempEvent()
                temp_event.event_id = event_id
                temp_event.workspace_id = workspace_id
                temp_event.user_id = user_id
                temp_event.timestamp = timestamp
                temp_event.content = content_dict
                temp_event.metadata = event_metadata
                temp_event.retention_class = event_metadata.get("retention_class")
                
                # Run enrichment
                enriched = self._enricher.enrich(temp_event, context)
                
                # Merge enriched metadata
                event_metadata.update(enriched.to_dict())
                
                self._stats["enrichments"] += 1
                logger.debug(f"[FABRIC] Enriched {event_id}: salience={enriched.salience_score}")
                
            except Exception as e:
                logger.warning(f"[FABRIC] Enrichment failed for {event_id}: {e}")
        
        # Build job data if auto_embed is enabled AND we have a job queue
        job_data = None
        if self.auto_embed and self.job_queue:
            text = self._extract_text(content_dict)
            if text:
                from chillbot.compute import JobType
                job_id = f"job_{uuid.uuid4().hex[:16]}"
                job_data = {
                    "job_id": job_id,
                    "job_type": JobType.EMBED.value,
                    "workspace_id": workspace_id,
                    "payload": {
                        "event_id": event_id,
                        "text": text,
                        "metadata": {
                            "user_id": user_id,
                            "channel": channel,
                            "timestamp": timestamp,
                        },
                    },
                    "priority": priority,
                    "created_at": timestamp,
                }
        
        # Write using TURBO path (single pipeline)
        if self.kernel:
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
                metadata=event_metadata,  # Now includes enriched metadata
            )
            
            # Check if kernel supports turbo method
            if hasattr(self.kernel, 'write_event_turbo'):
                # TURBO: Single pipeline for STM + LTM + job queue
                self.kernel.write_event_turbo(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    event=event,
                    job_data=job_data,
                )
                logger.debug(f"[FABRIC] TURBO wrote event {event_id}")
            else:
                # Fallback to legacy path (3 round-trips)
                self.kernel.write_event(workspace_id, user_id, event)
                
                if job_data and self.job_queue:
                    from chillbot.compute import JobType
                    self.job_queue.enqueue(
                        job_type=JobType.EMBED,
                        workspace_id=workspace_id,
                        payload=job_data["payload"],
                        priority=priority,
                    )
                
                logger.debug(f"[FABRIC] Legacy wrote event {event_id}")
            
            # ==============================================
            # NEW: Track this event as previous for next enrichment
            # ==============================================
            stream_key = f"{workspace_id}:{user_id}"
            self._previous_events[stream_key] = event
        
        elif self.kernel_url:
            self._write_event_http(
                event_id=event_id,
                workspace_id=workspace_id,
                user_id=user_id,
                session_id=session_id,
                content=content_dict,
                channel=channel,
                metadata=event_metadata,
                ttl_seconds=ttl_seconds,
            )
            logger.debug(f"[FABRIC] Wrote event {event_id} via HTTP")
        
        else:
            raise RuntimeError("No kernel configured")
        
        # ==============================================
        # FIX v3: SYNCHRONOUS EMBEDDING FALLBACK
        # When auto_embed is enabled but no job_queue exists,
        # embed and index synchronously using embeddings + vectors
        # ==============================================
        logger.info(f"[FABRIC] Sync embed check: auto_embed={self.auto_embed}, job_queue={self.job_queue is not None}, embeddings={self.embeddings is not None}, vectors={self.vectors is not None}")
        
        if self.auto_embed and not self.job_queue and self.embeddings and self.vectors:
            text = self._extract_text(content_dict)
            logger.info(f"[FABRIC] Sync embed: extracted text={text[:50] if text else None}...")
            if text:
                try:
                    # Generate embedding
                    logger.info(f"[FABRIC] Generating embedding for {event_id}...")
                    embedding = self.embeddings.embed(text)
                    logger.info(f"[FABRIC] Generated embedding dim={len(embedding)}")
                    
                    # Ensure collection exists (critical for first event!)
                    dimension = len(embedding)
                    if workspace_id not in self._initialized_collections:
                        logger.info(f"[FABRIC] Creating collection for workspace={workspace_id}, dim={dimension}")
                        self.vectors.ensure_collection(workspace_id, dimension)
                        self._initialized_collections[workspace_id] = dimension
                        logger.info(f"[FABRIC] Collection created for {workspace_id}")
                    
                    # Index the vector
                    logger.info(f"[FABRIC] Indexing vector {event_id} in workspace={workspace_id}")
                    self.vectors.index(
                        workspace_id=workspace_id,
                        id=event_id,
                        vector=embedding,
                        payload={
                            "event_id": event_id,
                            "text_preview": text[:500],  # Store preview for recall
                            "user_id": user_id,
                            "channel": channel,
                            "timestamp": timestamp,
                        }
                    )
                    
                    self._stats["sync_embeds"] += 1
                    logger.info(f"[FABRIC] ✓ Sync embedded and indexed {event_id}")
                    
                except Exception as e:
                    logger.error(f"[FABRIC] ✗ Sync embedding failed for {event_id}: {e}", exc_info=True)
        
        self._stats["remembers"] += 1
        return event_id
    
    def remember_batch(
        self,
        items: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Store multiple memories."""
        event_ids = []
        for item in items:
            event_id = self.remember(
                content=item.get("content"),
                workspace_id=workspace_id or item.get("workspace_id"),
                user_id=user_id or item.get("user_id"),
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
        recent_hours: float = 24.0,
        channel: Optional[str] = None,
    ) -> RecallResult:
        """Recall relevant memories."""
        workspace_id = workspace_id or self.default_workspace
        start_time = time.time()
        memories = []
        sources_used = []
        
        # Vector search
        if self.vectors and self.embeddings:
            try:
                query_embedding = self.embeddings.embed(query)
                matches = self.vectors.search(
                    workspace_id=workspace_id,
                    vector=query_embedding,
                    top_k=top_k,
                    score_threshold=min_score,
                )
                
                for match in matches:
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
        
        # Recent events from kernel
        if include_recent and self.kernel:
            try:
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
                    if channel and event.channel != channel:
                        continue
                    
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
        
        # Deduplicate
        seen_ids = set()
        unique_memories = []
        for mem in memories:
            if mem.event_id not in seen_ids:
                seen_ids.add(mem.event_id)
                unique_memories.append(mem)
        
        # Sort by score
        unique_memories.sort(key=lambda m: m.score, reverse=True)
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
    # CONTEXT
    # ==============================================
    
    def context(
        self,
        query: str,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        max_tokens: int = 4000,
        format: str = "text",
        include_metadata: bool = False,
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """Build LLM-ready context from memories."""
        from chillbot.fabric.context import ContextBuilder
        
        result = self.recall(
            query=query,
            workspace_id=workspace_id,
            user_id=user_id,
            top_k=50,
        )
        
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
    # NEW: ENRICHMENT UTILITIES
    # ==============================================
    
    def enrich_event(
        self,
        event: Any,
        similarity_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Manually enrich an event with metadata.
        
        Useful for re-enriching historical events or testing.
        
        Args:
            event: Event object with event_id, timestamp, content
            similarity_scores: Pre-computed similarity scores
        
        Returns:
            Enriched metadata dict
        """
        if not self._enricher:
            raise RuntimeError("Enrichment not configured")
        
        from chillbot.fabric.enrichment import EnrichmentContext
        
        stream_key = f"{event.workspace_id}:{event.user_id}"
        previous_event = self._previous_events.get(stream_key)
        
        context = EnrichmentContext(
            previous_event=previous_event,
            similarity_scores=similarity_scores,
            access_count=getattr(event, 'access_count', 0),
        )
        
        enriched = self._enricher.enrich(event, context)
        return enriched.to_dict()
    
    def get_enricher_config(self) -> Optional[Dict[str, Any]]:
        """Get current enrichment configuration."""
        if self._enricher:
            return self._enricher.get_stats()
        return None
    
    # ==============================================
    # UTILITY
    # ==============================================
    
    def _extract_text(self, content: Dict[str, Any]) -> Optional[str]:
        """Extract embeddable text from content dict."""
        for field in ["text", "message", "content", "body", "query", "response"]:
            if field in content and isinstance(content[field], str):
                return content[field]
        
        if "role" in content and "content" in content:
            return content["content"]
        
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
    
    def enrich(
        self,
        event_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[MemoryItem]:
        """Enrich a memory item with full event data (legacy method name)."""
        workspace_id = workspace_id or self.default_workspace
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fabric statistics."""
        stats = {
            **self._stats,
            "kernel_mode": "local" if self.kernel else ("remote" if self.kernel_url else "none"),
            "auto_embed": self.auto_embed,
            "auto_enrich": self.auto_enrich,
            "turbo_enabled": hasattr(self.kernel, 'write_event_turbo') if self.kernel else False,
            "sync_embed_mode": self.auto_embed and not self.job_queue and self.embeddings is not None,
            "initialized_collections": list(self._initialized_collections.keys()),
        }
        
        if self._enricher:
            stats["enricher"] = self._enricher.get_stats()
        
        return stats
    
    def close(self):
        """Close connections."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        
        # Clear previous events cache
        self._previous_events.clear()
        
        # Clear initialized collections tracking
        self._initialized_collections.clear()
    
    def __repr__(self) -> str:
        mode = "local" if self.kernel else ("remote" if self.kernel_url else "none")
        sync_mode = "sync" if (self.auto_embed and not self.job_queue and self.embeddings) else "async"
        return f"MemoryFabric(mode={mode}, workspace={self.default_workspace}, turbo=True, enrich={self.auto_enrich}, embed={sync_mode})"
