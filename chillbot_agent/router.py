"""
ChillBot Agent - Retrieval Router

Routes queries to appropriate retrieval methods based on QueryPlan.

Retrieval strategies:
- TEMPORAL: kernel.replay_to_timestamp() / get_events_in_range()
- ENTITY: Filter by extracted entities
- SEMANTIC: fabric.recall() with vector search
- MULTI_HOP: Iterative retrieval with expansion
- FACTUAL: Targeted retrieval with reranking

Leverages KRNX's unique capabilities:
- Temporal replay (no other system has this)
- Event relations (supersession chains)
- Multi-tier storage (STM + LTM)

Constitution-safe: Uses kernel/fabric APIs, no direct DB access.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set

from .planner import QueryPlan, QueryIntent

logger = logging.getLogger(__name__)


# ==============================================
# RETRIEVAL RESULT
# ==============================================

@dataclass
class RetrievalResult:
    """
    Result of retrieval operation.
    
    Contains events from one or more retrieval strategies.
    """
    events: List[Any] = field(default_factory=list)
    
    # Metadata
    strategy: str = "semantic"
    sources_used: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    
    # For multi-hop
    hop_count: int = 1
    intermediate_results: List[List[Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_count": len(self.events),
            "strategy": self.strategy,
            "sources_used": self.sources_used,
            "latency_ms": round(self.latency_ms, 2),
            "hop_count": self.hop_count,
        }


# ==============================================
# RETRIEVAL ROUTER
# ==============================================

class RetrievalRouter:
    """
    Routes queries to appropriate retrieval methods.
    
    Uses QueryPlan to determine strategy, then executes
    against kernel and fabric.
    """
    
    def __init__(
        self,
        fabric=None,
        kernel=None,
        embeddings=None,
        vectors=None,
        max_multi_hop_depth: int = 3,
    ):
        """
        Initialize retrieval router.
        
        Args:
            fabric: MemoryFabric instance
            kernel: KRNXController instance
            embeddings: Embedding model (for entity expansion)
            vectors: VectorStore instance
            max_multi_hop_depth: Maximum iterations for multi-hop
        """
        self.fabric = fabric
        self.kernel = kernel
        self.embeddings = embeddings
        self.vectors = vectors
        self.max_multi_hop_depth = max_multi_hop_depth
        
        logger.info("[ROUTER] Initialized retrieval router")
    
    def retrieve(
        self,
        plan: QueryPlan,
        query: str,
        workspace_id: str,
        user_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Execute retrieval based on query plan.
        
        Args:
            plan: QueryPlan from planner
            query: Original query string
            workspace_id: Workspace to search
            user_id: Optional user filter
        
        Returns:
            RetrievalResult with events
        """
        start_time = time.time()
        
        # Route based on intent
        if plan.intent == QueryIntent.TEMPORAL:
            result = self._retrieve_temporal(plan, query, workspace_id, user_id)
        
        elif plan.intent == QueryIntent.ENTITY:
            result = self._retrieve_entity(plan, query, workspace_id, user_id)
        
        elif plan.intent == QueryIntent.MULTI_HOP:
            result = self._retrieve_multi_hop(plan, query, workspace_id, user_id)
        
        else:  # SEMANTIC, FACTUAL, default
            result = self._retrieve_semantic(plan, query, workspace_id, user_id)
        
        # Add timing
        result.latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"[ROUTER] Retrieved {len(result.events)} events "
            f"via {result.strategy} in {result.latency_ms:.1f}ms"
        )
        
        return result
    
    # ==============================================
    # TEMPORAL RETRIEVAL
    # ==============================================
    
    def _retrieve_temporal(
        self,
        plan: QueryPlan,
        query: str,
        workspace_id: str,
        user_id: Optional[str],
    ) -> RetrievalResult:
        """
        Retrieve using temporal replay - KRNX's differentiator.
        
        Uses kernel's replay_to_timestamp() and get_events_in_range().
        """
        events = []
        sources = []
        
        # If we have a temporal range, use it
        if plan.temporal_range and self.kernel:
            tr = plan.temporal_range
            
            if tr.start and tr.end:
                # Range query
                temporal_events = self.kernel.get_events_in_range(
                    workspace_id=workspace_id,
                    user_id=user_id or "default",
                    start_time=tr.start,
                    end_time=tr.end,
                )
                events.extend(temporal_events)
                sources.append("kernel:time_range")
                logger.debug(f"[ROUTER] Time range query returned {len(temporal_events)} events")
            
            elif tr.end:
                # Replay to timestamp
                temporal_events = self.kernel.replay_to_timestamp(
                    workspace_id=workspace_id,
                    user_id=user_id or "default",
                    timestamp=tr.end,
                )
                events.extend(temporal_events)
                sources.append("kernel:replay")
                logger.debug(f"[ROUTER] Replay query returned {len(temporal_events)} events")
        
        # Also do semantic search to catch relevant events
        if self.fabric:
            recall_result = self.fabric.recall(
                query=query,
                workspace_id=workspace_id,
                user_id=user_id,
                top_k=plan.top_k,
            )
            
            # Add semantic results (will be deduplicated later)
            for memory in recall_result.memories:
                events.append(memory)
            sources.append("fabric:semantic")
        
        return RetrievalResult(
            events=events,
            strategy="temporal",
            sources_used=sources,
        )
    
    # ==============================================
    # ENTITY RETRIEVAL
    # ==============================================
    
    def _retrieve_entity(
        self,
        plan: QueryPlan,
        query: str,
        workspace_id: str,
        user_id: Optional[str],
    ) -> RetrievalResult:
        """
        Retrieve events mentioning specific entities.
        
        Combines entity filtering with semantic search.
        """
        events = []
        sources = []
        
        # Semantic search first
        if self.fabric:
            recall_result = self.fabric.recall(
                query=query,
                workspace_id=workspace_id,
                user_id=user_id,
                top_k=plan.top_k * 2,  # Get more, filter down
            )
            
            # Filter by entities if we have them
            if plan.entities:
                entity_ids = {e["id"].lower() for e in plan.entities}
                
                for memory in recall_result.memories:
                    # Check if memory mentions any of our entities
                    content = self._get_text(memory)
                    content_lower = content.lower()
                    
                    if any(eid in content_lower for eid in entity_ids):
                        events.append(memory)
                
                sources.append("fabric:entity_filtered")
                logger.debug(f"[ROUTER] Entity filter: {len(recall_result.memories)} -> {len(events)}")
            else:
                events.extend(recall_result.memories)
                sources.append("fabric:semantic")
        
        # If no results from filtering, fall back to semantic
        if not events and self.fabric:
            recall_result = self.fabric.recall(
                query=query,
                workspace_id=workspace_id,
                user_id=user_id,
                top_k=plan.top_k,
            )
            events.extend(recall_result.memories)
            sources.append("fabric:semantic_fallback")
        
        return RetrievalResult(
            events=events[:plan.top_k],
            strategy="entity",
            sources_used=sources,
        )
    
    # ==============================================
    # SEMANTIC RETRIEVAL
    # ==============================================
    
    def _retrieve_semantic(
        self,
        plan: QueryPlan,
        query: str,
        workspace_id: str,
        user_id: Optional[str],
    ) -> RetrievalResult:
        """
        Standard semantic similarity retrieval.
        
        Uses fabric.recall() which combines vector search + recent events.
        """
        events = []
        sources = []
        
        if self.fabric:
            recall_result = self.fabric.recall(
                query=query,
                workspace_id=workspace_id,
                user_id=user_id,
                top_k=plan.top_k,
            )
            
            events.extend(recall_result.memories)
            sources.extend(recall_result.sources_used)
        
        return RetrievalResult(
            events=events,
            strategy="semantic",
            sources_used=sources,
        )
    
    # ==============================================
    # MULTI-HOP RETRIEVAL
    # ==============================================
    
    def _retrieve_multi_hop(
        self,
        plan: QueryPlan,
        query: str,
        workspace_id: str,
        user_id: Optional[str],
    ) -> RetrievalResult:
        """
        Iterative retrieval for complex queries.
        
        Steps:
        1. Initial retrieval
        2. Analyze results for expansion
        3. Additional retrieval based on findings
        4. Repeat until sufficient or max depth
        """
        all_events = []
        intermediate = []
        sources = []
        seen_ids: Set[str] = set()
        
        # Current query starts as original
        current_query = query
        
        for hop in range(self.max_multi_hop_depth):
            # Retrieve for current query
            if self.fabric:
                recall_result = self.fabric.recall(
                    query=current_query,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    top_k=plan.top_k,
                )
                
                hop_events = []
                for memory in recall_result.memories:
                    event_id = getattr(memory, 'event_id', str(id(memory)))
                    if event_id not in seen_ids:
                        seen_ids.add(event_id)
                        hop_events.append(memory)
                        all_events.append(memory)
                
                intermediate.append(hop_events)
                sources.append(f"hop_{hop + 1}:fabric")
                
                logger.debug(f"[ROUTER] Hop {hop + 1}: {len(hop_events)} new events")
                
                # Check if we have enough
                if len(all_events) >= plan.top_k * 2:
                    break
                
                # Try to expand query from results
                expansion = self._expand_query_from_results(query, hop_events)
                if expansion and expansion != current_query:
                    current_query = expansion
                    logger.debug(f"[ROUTER] Expanded query: {expansion[:50]}...")
                else:
                    break  # No expansion possible
        
        return RetrievalResult(
            events=all_events[:plan.top_k * 2],
            strategy="multi_hop",
            sources_used=sources,
            hop_count=len(intermediate),
            intermediate_results=intermediate,
        )
    
    def _expand_query_from_results(
        self,
        original_query: str,
        results: List[Any],
    ) -> Optional[str]:
        """
        Try to expand query based on retrieved results.
        
        Looks for entities, concepts, or related terms in results
        that might help find more relevant events.
        """
        if not results:
            return None
        
        # Extract potential expansion terms from results
        expansion_terms = set()
        
        for result in results[:5]:  # Look at top 5
            content = self._get_text(result)
            
            # Extract @mentions
            mentions = set(m.group(1) for m in 
                          __import__('re').finditer(r'@(\w+)', content))
            expansion_terms.update(mentions)
            
            # Extract #tags
            tags = set(m.group(1) for m in 
                      __import__('re').finditer(r'#(\w+)', content))
            expansion_terms.update(tags)
        
        # If we found new terms, add them to query
        if expansion_terms:
            expansion = " ".join(expansion_terms)
            return f"{original_query} {expansion}"
        
        return None
    
    # ==============================================
    # UTILITY
    # ==============================================
    
    def _get_text(self, memory_or_event: Any) -> str:
        """Extract text from memory item or event."""
        # Try content attribute
        content = getattr(memory_or_event, 'content', None)
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            for field in ["text", "message", "content", "body"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            
            # Try query/response combo
            if "query" in content and "response" in content:
                return f"{content['query']} {content['response']}"
            
            return str(content)
        
        return str(memory_or_event)


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'RetrievalResult',
    'RetrievalRouter',
]
