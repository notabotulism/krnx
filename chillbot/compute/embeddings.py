"""
KRNX Compute - Embedding Engine

Local-first embedding generation using sentence-transformers.
Falls back gracefully, supports batching, caches model.

Supported models:
- all-MiniLM-L6-v2 (384 dims, fast, good quality) - DEFAULT
- all-mpnet-base-v2 (768 dims, slower, better quality)
- paraphrase-multilingual-MiniLM-L12-v2 (384 dims, multilingual)

Usage:
    engine = EmbeddingEngine()
    
    # Single embedding
    vector = engine.embed("Hello world")
    
    # Batch (more efficient)
    vectors = engine.embed_batch(["Hello", "World", "Test"])
    
    # Similarity
    score = engine.similarity(vec1, vec2)
"""

import logging
from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_seq_length": 256,
        "description": "Fast, good quality, English-focused",
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_seq_length": 384,
        "description": "Slower, higher quality, English-focused",
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "max_seq_length": 128,
        "description": "Multilingual support, 50+ languages",
    },
    "text-embedding-3-small": {
        "dimension": 1536,
        "max_seq_length": 8191,
        "description": "OpenAI API model (not local)",
    },
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    vector: List[float]
    text_preview: str
    model: str
    dimension: int


class EmbeddingEngine:
    """
    Generate embeddings for text content.
    
    Local-first using sentence-transformers. Lazy loads model on first use.
    Thread-safe for inference (model is read-only after load).
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize embedding engine.
        
        Args:
            model_name: Name of sentence-transformer model
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
            normalize: Whether to L2-normalize vectors (recommended for cosine similarity)
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None
        self._loaded = False
        
        # Validate model name
        if model_name not in MODEL_CONFIGS:
            logger.warning(
                f"Unknown model '{model_name}', using dimension=384. "
                f"Known models: {list(MODEL_CONFIGS.keys())}"
            )
    
    @property
    def dimension(self) -> int:
        """Vector dimension for this model."""
        config = MODEL_CONFIGS.get(self.model_name, {})
        return config.get("dimension", 384)
    
    @property
    def max_seq_length(self) -> int:
        """Maximum sequence length for this model."""
        config = MODEL_CONFIGS.get(self.model_name, {})
        return config.get("max_seq_length", 256)
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"[EMBED] Loading model: {self.model_name}")
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._loaded = True
            
            logger.info(
                f"[EMBED] Model loaded: {self.model_name} "
                f"(dim={self.dimension}, device={self._model.device})"
            )
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"[EMBED] Failed to load model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def embed(self, text: str) -> List[float]:
        """
        Embed single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Vector as list of floats
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.dimension
        
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        
        return vector.tolist()
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Embed batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            List of vectors
        """
        if not texts:
            return []
        
        # Handle empty texts
        non_empty_indices = []
        non_empty_texts = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        
        # Embed non-empty texts
        if non_empty_texts:
            vectors = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
            )
        else:
            vectors = np.array([])
        
        # Reconstruct full results with zero vectors for empty texts
        results = []
        vector_idx = 0
        zero_vector = [0.0] * self.dimension
        
        for i in range(len(texts)):
            if i in non_empty_indices:
                results.append(vectors[vector_idx].tolist())
                vector_idx += 1
            else:
                results.append(zero_vector)
        
        return results
    
    def embed_with_metadata(self, text: str) -> EmbeddingResult:
        """
        Embed text and return with metadata.
        
        Args:
            text: Text to embed
        
        Returns:
            EmbeddingResult with vector and metadata
        """
        vector = self.embed(text)
        
        return EmbeddingResult(
            vector=vector,
            text_preview=text[:100] if text else "",
            model=self.model_name,
            dimension=len(vector),
        )
    
    def similarity(
        self,
        vec1: Union[List[float], np.ndarray],
        vec2: Union[List[float], np.ndarray],
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Similarity score (-1 to 1, higher is more similar)
        """
        a = np.array(vec1)
        b = np.array(vec2)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def batch_similarity(
        self,
        query_vector: Union[List[float], np.ndarray],
        vectors: List[Union[List[float], np.ndarray]],
    ) -> List[float]:
        """
        Compute similarity of query against multiple vectors.
        
        Args:
            query_vector: Query vector
            vectors: List of vectors to compare against
        
        Returns:
            List of similarity scores
        """
        if not vectors:
            return []
        
        query = np.array(query_vector)
        query_norm = np.linalg.norm(query)
        
        if query_norm == 0:
            return [0.0] * len(vectors)
        
        scores = []
        for vec in vectors:
            v = np.array(vec)
            v_norm = np.linalg.norm(v)
            
            if v_norm == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.dot(query, v) / (query_norm * v_norm)))
        
        return scores
    
    def most_similar(
        self,
        query_vector: Union[List[float], np.ndarray],
        vectors: List[Union[List[float], np.ndarray]],
        top_k: int = 10,
    ) -> List[tuple]:
        """
        Find most similar vectors to query.
        
        Args:
            query_vector: Query vector
            vectors: List of vectors to search
            top_k: Number of results
        
        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        scores = self.batch_similarity(query_vector, vectors)
        
        # Create (index, score) pairs and sort
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:top_k]
    
    def extract_text(self, content: any) -> Optional[str]:
        """
        Extract embeddable text from various content formats.
        
        Args:
            content: String, dict, or other content
        
        Returns:
            Extracted text or None
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            # Try common text fields
            for field in ["text", "message", "content", "body", "description", "summary"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            
            # Try role-based messages (chat format)
            if "role" in content and "content" in content:
                return content["content"]
        
        # Last resort: stringify
        try:
            return str(content)
        except Exception:
            return None
    
    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"EmbeddingEngine(model='{self.model_name}', dim={self.dimension}, {status})"
