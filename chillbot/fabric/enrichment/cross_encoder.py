"""
KRNX Enrichment - Cross-Encoder Reranker

High-accuracy semantic similarity scoring using cross-encoder models.
Cross-encoders process both texts together, achieving higher accuracy
than bi-encoder embeddings at the cost of speed.

Use case: Rerank top-N candidates from bi-encoder search for precision.

Performance:
- Bi-encoder (embedding): ~15ms for 1000 comparisons
- Cross-encoder: ~5ms per pair = ~100ms for top-20

Constitution-safe: Optional, async-friendly, no side effects.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class CrossEncoderConfig:
    """Configuration for cross-encoder reranking."""
    
    # Model selection
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Reranking settings
    top_k: int = 20                 # How many candidates to rerank
    batch_size: int = 16            # Batch size for inference
    
    # Score normalization
    normalize_scores: bool = True   # Normalize to [0, 1]
    
    # Device
    device: Optional[str] = None    # 'cpu', 'cuda', None = auto


# ==============================================
# CROSS-ENCODER INTERFACE
# ==============================================

class CrossEncoderBase(ABC):
    """Abstract base for cross-encoder implementations."""
    
    @abstractmethod
    def score_pair(self, text_a: str, text_b: str) -> float:
        """Score a single text pair."""
        pass
    
    @abstractmethod
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score multiple text pairs (batched)."""
        pass
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank candidates by relevance to query.
        
        Returns list of (original_index, score) sorted by score descending.
        """
        pass


# ==============================================
# SENTENCE TRANSFORMERS IMPLEMENTATION
# ==============================================

class SentenceTransformerCrossEncoder(CrossEncoderBase):
    """
    Cross-encoder using sentence-transformers library.
    
    Recommended model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Fast (~5ms per pair on CPU)
    - Good accuracy for semantic similarity
    - 22M parameters
    
    Alternative for higher accuracy: cross-encoder/stsb-roberta-large
    - Slower (~20ms per pair on CPU)
    - Better accuracy
    - 355M parameters
    """
    
    def __init__(self, config: Optional[CrossEncoderConfig] = None):
        """
        Initialize cross-encoder.
        
        Args:
            config: Configuration options
        """
        self.config = config or CrossEncoderConfig()
        self._model = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy-load the model."""
        if self._loaded:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"[CROSS_ENCODER] Loading model: {self.config.model_name}")
            
            device = self.config.device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._model = CrossEncoder(
                self.config.model_name,
                device=device,
            )
            self._loaded = True
            
            logger.info(f"[CROSS_ENCODER] Loaded on device: {device}")
            
        except ImportError:
            logger.warning(
                "[CROSS_ENCODER] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    
    def score_pair(self, text_a: str, text_b: str) -> float:
        """
        Score a single text pair.
        
        Args:
            text_a: First text
            text_b: Second text
        
        Returns:
            Similarity score (higher = more similar)
        """
        self._ensure_loaded()
        
        score = self._model.predict([(text_a, text_b)])[0]
        
        if self.config.normalize_scores:
            score = self._normalize(score)
        
        return float(score)
    
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score multiple text pairs (batched).
        
        Args:
            pairs: List of (text_a, text_b) tuples
        
        Returns:
            List of scores
        """
        if not pairs:
            return []
        
        self._ensure_loaded()
        
        scores = self._model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
        )
        
        if self.config.normalize_scores:
            scores = [self._normalize(s) for s in scores]
        
        return [float(s) for s in scores]
    
    def rerank(
        self,
        query: str,
        candidates: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Rerank candidates by relevance to query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Return top K results (None = all)
        
        Returns:
            List of (original_index, score) sorted by score descending
        """
        if not candidates:
            return []
        
        top_k = top_k or self.config.top_k
        
        # Create pairs
        pairs = [(query, candidate) for candidate in candidates]
        
        # Score all pairs
        scores = self.score_pairs(pairs)
        
        # Create (index, score) tuples
        indexed_scores = list(enumerate(scores))
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k
        return indexed_scores[:top_k]
    
    def _normalize(self, score: float) -> float:
        """
        Normalize score to [0, 1] range.
        
        MS-MARCO model outputs logits, roughly in [-10, 10] range.
        We use sigmoid to normalize.
        """
        import math
        return 1 / (1 + math.exp(-score))
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            logger.info("[CROSS_ENCODER] Model unloaded")


# ==============================================
# CROSS-ENCODER RERANKER
# ==============================================

class CrossEncoderReranker:
    """
    High-level reranker for KRNX relation scoring.
    
    Takes candidates from bi-encoder search and reranks
    using cross-encoder for higher precision.
    """
    
    def __init__(
        self,
        config: Optional[CrossEncoderConfig] = None,
        encoder: Optional[CrossEncoderBase] = None,
    ):
        """
        Initialize reranker.
        
        Args:
            config: Configuration
            encoder: Optional custom encoder (default: SentenceTransformerCrossEncoder)
        """
        self.config = config or CrossEncoderConfig()
        self._encoder = encoder
        self._enabled = True
    
    @property
    def encoder(self) -> CrossEncoderBase:
        """Lazy-initialize encoder."""
        if self._encoder is None:
            self._encoder = SentenceTransformerCrossEncoder(self.config)
        return self._encoder
    
    def rerank_events(
        self,
        query_event: Any,
        candidates: List[Any],
        bi_encoder_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Rerank candidate events by similarity to query event.
        
        Args:
            query_event: The query event
            candidates: List of candidate events
            bi_encoder_scores: Optional bi-encoder scores for fallback
        
        Returns:
            List of (event, score) sorted by score descending
        """
        if not candidates:
            return []
        
        if not self._enabled:
            # Return candidates with bi-encoder scores
            return [
                (c, bi_encoder_scores.get(
                    getattr(c, 'event_id', str(id(c))), 0.0
                ))
                for c in candidates
            ]
        
        # Extract query text
        query_text = self._get_text(query_event)
        if not query_text:
            logger.warning("[RERANKER] Empty query text, using bi-encoder scores")
            return [
                (c, bi_encoder_scores.get(
                    getattr(c, 'event_id', str(id(c))), 0.0
                ))
                for c in candidates
            ]
        
        # Extract candidate texts
        candidate_texts = [self._get_text(c) for c in candidates]
        
        try:
            # Rerank
            reranked = self.encoder.rerank(
                query=query_text,
                candidates=candidate_texts,
                top_k=len(candidates),
            )
            
            # Map back to events
            return [(candidates[idx], score) for idx, score in reranked]
            
        except Exception as e:
            logger.error(f"[RERANKER] Cross-encoder failed: {e}")
            # Fallback to bi-encoder scores
            if bi_encoder_scores:
                results = [
                    (c, bi_encoder_scores.get(
                        getattr(c, 'event_id', str(id(c))), 0.0
                    ))
                    for c in candidates
                ]
                results.sort(key=lambda x: x[1], reverse=True)
                return results
            return [(c, 0.0) for c in candidates]
    
    def score_event_pair(
        self,
        event_a: Any,
        event_b: Any,
    ) -> float:
        """
        Score similarity between two events.
        
        Args:
            event_a: First event
            event_b: Second event
        
        Returns:
            Similarity score [0, 1]
        """
        text_a = self._get_text(event_a)
        text_b = self._get_text(event_b)
        
        if not text_a or not text_b:
            return 0.0
        
        try:
            return self.encoder.score_pair(text_a, text_b)
        except Exception as e:
            logger.error(f"[RERANKER] Cross-encoder pair scoring failed: {e}")
            return 0.0
    
    def score_event_pairs(
        self,
        query_event: Any,
        candidates: List[Any],
    ) -> Dict[str, float]:
        """
        Score multiple event pairs.
        
        Args:
            query_event: Query event
            candidates: List of candidate events
        
        Returns:
            Dict of event_id -> cross-encoder score
        """
        if not candidates:
            return {}
        
        query_text = self._get_text(query_event)
        if not query_text:
            return {}
        
        pairs = []
        event_ids = []
        
        for candidate in candidates:
            text = self._get_text(candidate)
            if text:
                pairs.append((query_text, text))
                event_ids.append(getattr(candidate, 'event_id', str(id(candidate))))
        
        if not pairs:
            return {}
        
        try:
            scores = self.encoder.score_pairs(pairs)
            return dict(zip(event_ids, scores))
        except Exception as e:
            logger.error(f"[RERANKER] Batch scoring failed: {e}")
            return {}
    
    def _get_text(self, event: Any) -> str:
        """Extract text from event."""
        content = getattr(event, 'content', {})
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            for field in ["text", "message", "content", "body", "query", "response"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            
            if "query" in content and "response" in content:
                return f"{content['query']} {content['response']}"
        
        return str(content) if content else ""
    
    def enable(self):
        """Enable cross-encoder reranking."""
        self._enabled = True
    
    def disable(self):
        """Disable cross-encoder reranking (use bi-encoder scores only)."""
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self._enabled
    
    def unload(self):
        """Unload model to free memory."""
        if self._encoder is not None:
            self._encoder.unload()


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

_default_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(config: Optional[CrossEncoderConfig] = None) -> CrossEncoderReranker:
    """
    Get or create default reranker.
    
    Args:
        config: Optional configuration
    
    Returns:
        CrossEncoderReranker instance
    """
    global _default_reranker
    
    if _default_reranker is None:
        _default_reranker = CrossEncoderReranker(config)
    
    return _default_reranker


def rerank_candidates(
    query_text: str,
    candidate_texts: List[str],
    top_k: int = 20,
) -> List[Tuple[int, float]]:
    """
    Convenience function to rerank text candidates.
    
    Args:
        query_text: Query string
        candidate_texts: List of candidate strings
        top_k: Return top K
    
    Returns:
        List of (original_index, score)
    """
    reranker = get_reranker()
    return reranker.encoder.rerank(query_text, candidate_texts, top_k)


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Config
    'CrossEncoderConfig',
    
    # Base class
    'CrossEncoderBase',
    
    # Implementations
    'SentenceTransformerCrossEncoder',
    'CrossEncoderReranker',
    
    # Convenience
    'get_reranker',
    'rerank_candidates',
]
