"""
ChillBot Agent - Result Processor

Processes retrieved results to maximize relevance and accuracy.

Processing steps:
1. Cross-encoder reranking (higher precision than bi-encoder)
2. Relation chain resolution (get current version of superseded events)
3. Deduplication (by content hash)
4. Retention filtering (skip ephemeral, prefer durable)
5. Salience scoring (for final ordering)

Uses KRNX enrichment components:
- cross_encoder.py for reranking
- relations.py for supersession detection
- summary.py for chain resolution
- salience.py for importance scoring

Constitution-safe: Read-only processing, no mutations.
"""

import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple

from .planner import QueryPlan
from .router import RetrievalResult

logger = logging.getLogger(__name__)


# ==============================================
# PROCESSED RESULT
# ==============================================

@dataclass
class ProcessedResult:
    """
    Result after processing (reranked, filtered, deduplicated).
    """
    events: List[Any] = field(default_factory=list)
    
    # Processing stats
    original_count: int = 0
    after_rerank: int = 0
    after_dedup: int = 0
    after_filter: int = 0
    
    # Scores
    rerank_scores: Dict[str, float] = field(default_factory=dict)
    salience_scores: Dict[str, float] = field(default_factory=dict)
    
    # Processing info
    latency_ms: float = 0.0
    steps_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_count": len(self.events),
            "original_count": self.original_count,
            "after_rerank": self.after_rerank,
            "after_dedup": self.after_dedup,
            "after_filter": self.after_filter,
            "latency_ms": round(self.latency_ms, 2),
            "steps_applied": self.steps_applied,
        }


# ==============================================
# RESULT PROCESSOR
# ==============================================

class ResultProcessor:
    """
    Processes retrieval results for maximum relevance.
    
    Steps:
    1. Cross-encoder reranking
    2. Relation chain resolution
    3. Deduplication
    4. Retention filtering
    5. Salience scoring
    """
    
    def __init__(
        self,
        cross_encoder=None,
        relation_scorer=None,
        salience_engine=None,
        enable_reranking: bool = True,
        enable_relation_filter: bool = True,
        enable_retention_filter: bool = True,
        dedup_threshold: float = 0.95,
    ):
        """
        Initialize result processor.
        
        Args:
            cross_encoder: CrossEncoderReranker instance
            relation_scorer: RelationScorer instance
            salience_engine: SalienceEngine instance
            enable_reranking: Whether to use cross-encoder
            enable_relation_filter: Whether to filter superseded events
            enable_retention_filter: Whether to filter by retention class
            dedup_threshold: Similarity threshold for dedup
        """
        self.cross_encoder = cross_encoder
        self.relation_scorer = relation_scorer
        self.salience_engine = salience_engine
        
        self.enable_reranking = enable_reranking
        self.enable_relation_filter = enable_relation_filter
        self.enable_retention_filter = enable_retention_filter
        self.dedup_threshold = dedup_threshold
        
        logger.info(
            f"[PROCESSOR] Initialized (rerank={enable_reranking}, "
            f"relation_filter={enable_relation_filter}, "
            f"retention_filter={enable_retention_filter})"
        )
    
    def process(
        self,
        retrieval_result: RetrievalResult,
        query: str,
        plan: QueryPlan,
    ) -> ProcessedResult:
        """
        Process retrieval results.
        
        Args:
            retrieval_result: Raw retrieval results
            query: Original query string
            plan: Query plan (for hints)
        
        Returns:
            ProcessedResult with filtered and ranked events
        """
        start_time = time.time()
        events = list(retrieval_result.events)
        steps = []
        
        original_count = len(events)
        
        # Step 1: Cross-encoder reranking
        rerank_scores = {}
        if self.enable_reranking and plan.use_reranking and self.cross_encoder:
            events, rerank_scores = self._rerank(events, query)
            steps.append("cross_encoder_rerank")
        after_rerank = len(events)
        
        # Step 2: Deduplication
        events = self._deduplicate(events)
        steps.append("deduplicate")
        after_dedup = len(events)
        
        # Step 3: Relation chain resolution (get current versions)
        if self.enable_relation_filter and plan.filter_stale:
            events = self._filter_superseded(events)
            steps.append("relation_filter")
        
        # Step 4: Retention filtering
        if self.enable_retention_filter:
            events = self._filter_by_retention(events)
            steps.append("retention_filter")
        after_filter = len(events)
        
        # Step 5: Salience scoring
        salience_scores = {}
        if self.salience_engine:
            salience_scores = self._compute_salience(events, query)
            # Re-sort by salience
            events = sorted(
                events,
                key=lambda e: salience_scores.get(self._get_id(e), 0),
                reverse=True
            )
            steps.append("salience_sort")
        
        # Limit to top_k
        events = events[:plan.top_k]
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"[PROCESSOR] {original_count} -> {len(events)} events "
            f"({', '.join(steps)}) in {latency_ms:.1f}ms"
        )
        
        return ProcessedResult(
            events=events,
            original_count=original_count,
            after_rerank=after_rerank,
            after_dedup=after_dedup,
            after_filter=after_filter,
            rerank_scores=rerank_scores,
            salience_scores=salience_scores,
            latency_ms=latency_ms,
            steps_applied=steps,
        )
    
    # ==============================================
    # CROSS-ENCODER RERANKING
    # ==============================================
    
    def _rerank(
        self,
        events: List[Any],
        query: str,
    ) -> Tuple[List[Any], Dict[str, float]]:
        """
        Rerank events using cross-encoder.
        
        Cross-encoders process query+document together,
        achieving higher accuracy than bi-encoder embeddings.
        """
        if not events:
            return events, {}
        
        try:
            # Get texts for reranking
            texts = [self._get_text(e) for e in events]
            
            # Rerank
            reranked = self.cross_encoder.rerank(
                query=query,
                candidates=texts,
                top_k=len(events),
            )
            
            # Build result
            scores = {}
            reranked_events = []
            
            for idx, score in reranked:
                event = events[idx]
                event_id = self._get_id(event)
                scores[event_id] = score
                reranked_events.append(event)
            
            logger.debug(f"[PROCESSOR] Reranked {len(events)} events")
            return reranked_events, scores
            
        except Exception as e:
            logger.warning(f"[PROCESSOR] Reranking failed: {e}")
            return events, {}
    
    # ==============================================
    # DEDUPLICATION
    # ==============================================
    
    def _deduplicate(self, events: List[Any]) -> List[Any]:
        """
        Remove duplicate events based on content hash.
        
        Uses content hash to detect exact duplicates and
        similarity threshold for near-duplicates.
        """
        if not events:
            return events
        
        seen_hashes: Set[str] = set()
        unique_events = []
        
        for event in events:
            content = self._get_text(event)
            content_hash = self._hash_content(content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_events.append(event)
        
        if len(unique_events) < len(events):
            logger.debug(
                f"[PROCESSOR] Dedup: {len(events)} -> {len(unique_events)} events"
            )
        
        return unique_events
    
    def _hash_content(self, content: str) -> str:
        """Generate content hash for deduplication."""
        # Normalize content
        normalized = content.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    # ==============================================
    # RELATION FILTERING
    # ==============================================
    
    def _filter_superseded(self, events: List[Any]) -> List[Any]:
        """
        Filter out superseded events, keeping only current versions.
        
        Uses relation metadata to find supersession chains
        and returns only the "current" (non-superseded) events.
        """
        if not events:
            return events
        
        # Build set of superseded event IDs
        superseded_ids: Set[str] = set()
        
        for event in events:
            metadata = getattr(event, 'metadata', {}) or {}
            
            # Check if this event supersedes others
            relations = metadata.get('relations', [])
            if isinstance(relations, list):
                for rel in relations:
                    if isinstance(rel, dict):
                        if rel.get('type') == 'supersedes':
                            target = rel.get('target_event_id')
                            if target:
                                superseded_ids.add(target)
            
            # Also check summary info
            summary = metadata.get('relation_summary', {})
            if isinstance(summary, dict):
                if summary.get('is_superseded'):
                    event_id = self._get_id(event)
                    superseded_ids.add(event_id)
        
        # Filter out superseded events
        filtered = []
        for event in events:
            event_id = self._get_id(event)
            if event_id not in superseded_ids:
                filtered.append(event)
        
        if len(filtered) < len(events):
            logger.debug(
                f"[PROCESSOR] Filtered {len(events) - len(filtered)} superseded events"
            )
        
        return filtered
    
    # ==============================================
    # RETENTION FILTERING
    # ==============================================
    
    def _filter_by_retention(self, events: List[Any]) -> List[Any]:
        """
        Filter by retention class, preferring durable events.
        
        Skips ephemeral events unless we have very few results.
        """
        if not events:
            return events
        
        # Separate by retention class
        durable = []
        other = []
        ephemeral = []
        
        for event in events:
            metadata = getattr(event, 'metadata', {}) or {}
            retention = metadata.get('retention_class', 'durable')
            
            if retention == 'ephemeral':
                ephemeral.append(event)
            elif retention == 'durable' or retention == 'permanent':
                durable.append(event)
            else:
                other.append(event)
        
        # Prefer durable, then other, then ephemeral (only if needed)
        result = durable + other
        
        # Only add ephemeral if we have very few results
        if len(result) < 3:
            result.extend(ephemeral)
        
        if len(ephemeral) > 0 and len(result) < len(events):
            logger.debug(
                f"[PROCESSOR] Filtered {len(ephemeral)} ephemeral events"
            )
        
        return result
    
    # ==============================================
    # SALIENCE SCORING
    # ==============================================
    
    def _compute_salience(
        self,
        events: List[Any],
        query: str,
    ) -> Dict[str, float]:
        """
        Compute salience scores for events.
        
        Uses existing salience from metadata if available,
        otherwise computes fresh scores.
        """
        scores = {}
        now = time.time()
        
        for event in events:
            event_id = self._get_id(event)
            metadata = getattr(event, 'metadata', {}) or {}
            
            # Try to get pre-computed salience
            salience_data = metadata.get('salience', {})
            if isinstance(salience_data, dict) and 'final' in salience_data:
                scores[event_id] = salience_data['final']
            elif isinstance(salience_data, (int, float)):
                scores[event_id] = float(salience_data)
            else:
                # Compute basic salience from timestamp + score
                timestamp = getattr(event, 'timestamp', now)
                age_hours = (now - timestamp) / 3600
                recency_score = max(0.1, 1.0 - (age_hours / 168))  # 1 week decay
                
                # Combine with retrieval score if available
                retrieval_score = getattr(event, 'score', 0.5)
                
                scores[event_id] = (recency_score + retrieval_score) / 2
        
        return scores
    
    # ==============================================
    # UTILITY
    # ==============================================
    
    def _get_id(self, event: Any) -> str:
        """Get event ID."""
        return getattr(event, 'event_id', str(id(event)))
    
    def _get_text(self, event: Any) -> str:
        """Extract text from event."""
        content = getattr(event, 'content', None)
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            for field in ["text", "message", "content", "body"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            
            if "query" in content and "response" in content:
                return f"{content['query']} {content['response']}"
            
            return str(content)
        
        # Fallback for MemoryItem
        if hasattr(event, 'content'):
            return str(event.content)
        
        return str(event)


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'ProcessedResult',
    'ResultProcessor',
]
