"""
KRNX Enrichment - Multi-Signal Feature Extraction v2

Extracts feature vectors from event pairs for relation scoring.
The core insight: embedding similarity alone fails critical memory cases.

Features captured:
- Semantic: embedding similarity, entity overlap
- Contradiction: negation mismatch, numeric mismatch, temporal mismatch, antonyms
- Structural: timestamp delta, same episode, same actor, reply pattern

v2 Changes:
- Lexicons loaded from external JSON (signals/*.json)
- User overrides supported (~/.krnx/signals/)
- Cancellation/correction phrase detection
- Reply pattern detection

Constitution-safe: Pure, deterministic, no LLM calls.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple

from .signals import (
    get_negation_lexicon,
    get_cancellation_lexicon,
    get_antonym_lexicon,
    get_temporal_lexicon,
    get_reply_pattern_lexicon,
)

logger = logging.getLogger(__name__)


# ==============================================
# NEGATION DETECTION
# ==============================================

def has_negation(text: str) -> bool:
    """
    Check if text contains negation.
    
    Args:
        text: Input text
    
    Returns:
        True if negation detected
    """
    return get_negation_lexicon().has_negation(text)


def negation_mismatch(text_a: str, text_b: str) -> bool:
    """
    Check if one text is negated and the other is not.
    
    This catches contradictions like:
    - "Project is approved" vs "Project is not approved"
    - "User can access" vs "User cannot access"
    
    Args:
        text_a: First text
        text_b: Second text
    
    Returns:
        True if negation mismatch detected
    """
    return has_negation(text_a) != has_negation(text_b)


# ==============================================
# CANCELLATION/CORRECTION DETECTION
# ==============================================

def has_cancellation(text: str) -> bool:
    """
    Check if text contains cancellation verbs.
    
    Args:
        text: Input text
    
    Returns:
        True if cancellation verb detected
    """
    return get_cancellation_lexicon().has_cancellation(text)


def has_correction(text: str) -> bool:
    """
    Check if text contains correction phrases.
    
    Args:
        text: Input text
    
    Returns:
        True if correction phrase detected
    """
    return get_cancellation_lexicon().has_correction(text)


def cancellation_mismatch(text_a: str, text_b: str) -> bool:
    """
    Check if one text cancels/corrects the other.
    
    Returns True if text_a contains cancellation/correction language
    about similar content in text_b.
    
    Args:
        text_a: First text (typically newer)
        text_b: Second text (typically older)
    
    Returns:
        True if cancellation detected
    """
    lexicon = get_cancellation_lexicon()
    
    # Check if newer text has cancellation/correction language
    has_cancel = lexicon.has_cancellation(text_a) or lexicon.has_correction(text_a)
    
    return has_cancel


# ==============================================
# NUMERIC EXTRACTION
# ==============================================

NUMERIC_PATTERN = re.compile(
    r'\$[\d,]+\.?\d*|'           # Currency: $50,000
    r'\d{1,2}:\d{2}\s*[ap]m?|'   # Time: 3:00pm, 3:00 pm
    r'\d+\.?\d*%|'               # Percentage: 15%, 3.5%
    r'\d+(?:st|nd|rd|th)|'       # Ordinals: 1st, 2nd, 3rd
    r'\d+\.?\d*'                 # Plain numbers: 42, 3.14
, re.IGNORECASE)


def extract_numerics(text: str) -> Set[str]:
    """
    Extract numeric values from text.
    
    Captures:
    - Currency ($50,000)
    - Times (3:00pm)
    - Percentages (15%)
    - Ordinals (1st, 2nd)
    - Plain numbers (42, 3.14)
    
    Args:
        text: Input text
    
    Returns:
        Set of normalized numeric strings
    """
    matches = NUMERIC_PATTERN.findall(text)
    # Normalize: lowercase, strip whitespace
    return {m.lower().strip() for m in matches}


def numeric_mismatch(text_a: str, text_b: str) -> bool:
    """
    Check if texts have conflicting numeric values.
    
    Returns True if:
    - Both texts contain numbers
    - The numbers are different
    - The contexts appear similar (both have numbers)
    
    This catches:
    - "Budget is $50,000" vs "Budget is $75,000"
    - "Meeting at 3pm" vs "Meeting at 4pm"
    
    Args:
        text_a: First text
        text_b: Second text
    
    Returns:
        True if numeric mismatch detected
    """
    nums_a = extract_numerics(text_a)
    nums_b = extract_numerics(text_b)
    
    # Both must have numbers
    if not nums_a or not nums_b:
        return False
    
    # If they have some overlap but aren't identical, that's a mismatch
    if nums_a != nums_b:
        return True
    
    return False


# ==============================================
# TEMPORAL EXTRACTION
# ==============================================

TIME_PATTERN = re.compile(
    r'\d{1,2}:\d{2}\s*[ap]m?|'   # 3:00pm, 3:00 pm
    r'\d{1,2}\s*[ap]m'           # 3pm, 4 pm
, re.IGNORECASE)

DATE_PATTERN = re.compile(
    r'\d{1,2}/\d{1,2}/\d{2,4}|'    # MM/DD/YYYY
    r'\d{4}-\d{2}-\d{2}|'          # YYYY-MM-DD
    r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}'  # Month DD
, re.IGNORECASE)


@dataclass
class TemporalInfo:
    """Extracted temporal information from text."""
    days: Set[str] = field(default_factory=set)
    relative: Set[str] = field(default_factory=set)
    times: Set[str] = field(default_factory=set)
    dates: Set[str] = field(default_factory=set)


def extract_temporal(text: str) -> TemporalInfo:
    """
    Extract temporal references from text.
    
    Uses external lexicon for days and relative times.
    
    Args:
        text: Input text
    
    Returns:
        TemporalInfo with extracted components
    """
    lower = text.lower()
    lexicon = get_temporal_lexicon()
    
    return TemporalInfo(
        days={d for d in lexicon.days if d in lower},
        relative={r for r in lexicon.relative_times if r in lower},
        times={t.lower() for t in TIME_PATTERN.findall(text)},
        dates={d.lower() for d in DATE_PATTERN.findall(text)},
    )


def temporal_mismatch(text_a: str, text_b: str) -> bool:
    """
    Check if texts have conflicting temporal references.
    
    Returns True if:
    - Same day mentioned but different times
    - Conflicting relative terms (today vs tomorrow)
    - Same context but different dates
    
    Args:
        text_a: First text
        text_b: Second text
    
    Returns:
        True if temporal mismatch detected
    """
    temp_a = extract_temporal(text_a)
    temp_b = extract_temporal(text_b)
    lexicon = get_temporal_lexicon()
    
    # Same day mentioned but different times
    if temp_a.days & temp_b.days:  # Overlap in days
        if temp_a.times and temp_b.times and temp_a.times != temp_b.times:
            return True
    
    # Conflicting relative terms (from lexicon)
    if temp_a.relative and temp_b.relative:
        for term_a in temp_a.relative:
            for term_b in temp_b.relative:
                if lexicon.is_conflict(term_a, term_b):
                    return True
    
    # Different specific dates
    if temp_a.dates and temp_b.dates and temp_a.dates != temp_b.dates:
        return True
    
    return False


# ==============================================
# ANTONYM DETECTION
# ==============================================

def find_antonyms(text: str) -> Set[str]:
    """
    Find words in text that have known antonyms.
    
    Args:
        text: Input text
    
    Returns:
        Set of words that have antonyms in our dictionary
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    lexicon = get_antonym_lexicon()
    return {w for w in words if lexicon.get_antonyms(w)}


def antonym_detected(text_a: str, text_b: str) -> bool:
    """
    Check if texts contain antonym pairs.
    
    Args:
        text_a: First text
        text_b: Second text
    
    Returns:
        True if antonym pair detected
    """
    words_a = set(re.findall(r'\b\w+\b', text_a.lower()))
    words_b = set(re.findall(r'\b\w+\b', text_b.lower()))
    
    return get_antonym_lexicon().has_antonym_pair(words_a, words_b)


# ==============================================
# ENTITY OVERLAP
# ==============================================

def entity_overlap(entities_a: List[Dict[str, Any]], entities_b: List[Dict[str, Any]]) -> float:
    """
    Compute Jaccard similarity of entity sets.
    
    Args:
        entities_a: List of entity dicts from event A
        entities_b: List of entity dicts from event B
    
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    # Extract normalized entity text
    set_a = {e.get("text", e.get("id", "")).lower() for e in entities_a}
    set_b = {e.get("text", e.get("id", "")).lower() for e in entities_b}
    
    # Remove empty strings
    set_a.discard("")
    set_b.discard("")
    
    if not set_a or not set_b:
        return 0.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


# ==============================================
# REPLY PATTERN DETECTION
# ==============================================

def detect_reply_pattern(text: str) -> Optional[str]:
    """
    Detect if text is a reply/acknowledgment/correction.
    
    Args:
        text: Input text
    
    Returns:
        Pattern type: 'acknowledgment', 'follow_up', 'correction', 'question', or None
    """
    return get_reply_pattern_lexicon().get_pattern_type(text)


def is_reply_like(text: str) -> bool:
    """
    Check if text looks like a reply to something.
    
    Args:
        text: Input text
    
    Returns:
        True if text appears to be a reply
    """
    return detect_reply_pattern(text) is not None


# ==============================================
# PAIR FEATURES DATACLASS
# ==============================================

@dataclass
class PairFeatures:
    """
    Features extracted from an event pair for relation scoring.
    
    This is the core data structure for multi-signal relation detection.
    Embedding similarity alone fails critical cases - these features
    capture what embeddings miss.
    """
    
    # Semantic signals
    embedding_similarity: float = 0.0       # Cosine similarity from bi-encoder
    entity_overlap: float = 0.0             # Jaccard similarity of entity sets
    
    # Contradiction signals
    negation_mismatch: bool = False         # One negated, other not
    numeric_mismatch: bool = False          # Same context, different numbers
    temporal_mismatch: bool = False         # Same pattern, different day/time
    antonym_detected: bool = False          # Antonym pair detected
    cancellation_detected: bool = False     # Cancellation/correction phrase
    
    # Structural signals
    timestamp_delta: float = 0.0            # Seconds between events
    same_episode: bool = False              # Same episode_id
    same_actor: bool = False                # Same actor/user
    reply_pattern: Optional[str] = None     # 'acknowledgment', 'correction', etc.
    
    # High-accuracy scoring (optional, async)
    cross_encoder_score: Optional[float] = None
    
    @property
    def contradiction_count(self) -> int:
        """Count of contradiction signals that fired."""
        count = 0
        if self.negation_mismatch:
            count += 1
        if self.numeric_mismatch:
            count += 1
        if self.temporal_mismatch:
            count += 1
        if self.antonym_detected:
            count += 1
        if self.cancellation_detected:
            count += 1
        return count
    
    @property
    def has_contradiction(self) -> bool:
        """True if any contradiction signal fired."""
        return self.contradiction_count > 0
    
    @property
    def strict_contradiction(self) -> bool:
        """True if 2+ contradiction signals fired (high confidence)."""
        return self.contradiction_count >= 2
    
    def get_fired_signals(self) -> List[str]:
        """Get list of contradiction signals that fired."""
        signals = []
        if self.negation_mismatch:
            signals.append("negation_mismatch")
        if self.numeric_mismatch:
            signals.append("numeric_mismatch")
        if self.temporal_mismatch:
            signals.append("temporal_mismatch")
        if self.antonym_detected:
            signals.append("antonym_detected")
        if self.cancellation_detected:
            signals.append("cancellation_detected")
        return signals
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding_similarity": self.embedding_similarity,
            "entity_overlap": self.entity_overlap,
            "negation_mismatch": self.negation_mismatch,
            "numeric_mismatch": self.numeric_mismatch,
            "temporal_mismatch": self.temporal_mismatch,
            "antonym_detected": self.antonym_detected,
            "cancellation_detected": self.cancellation_detected,
            "timestamp_delta": self.timestamp_delta,
            "same_episode": self.same_episode,
            "same_actor": self.same_actor,
            "reply_pattern": self.reply_pattern,
            "cross_encoder_score": self.cross_encoder_score,
            "contradiction_count": self.contradiction_count,
            "strict_contradiction": self.strict_contradiction,
        }


# ==============================================
# FEATURE EXTRACTOR
# ==============================================

class FeatureExtractor:
    """
    Extracts PairFeatures from two events.
    
    Pure, deterministic, no side effects.
    """
    
    def __init__(self):
        """Initialize feature extractor."""
        pass
    
    def extract(
        self,
        event_a: Any,
        event_b: Any,
        embedding_similarity: float = 0.0,
        cross_encoder_score: Optional[float] = None,
    ) -> PairFeatures:
        """
        Extract features from an event pair.
        
        Args:
            event_a: First event (typically newer)
            event_b: Second event (typically older/candidate)
            embedding_similarity: Pre-computed cosine similarity
            cross_encoder_score: Optional cross-encoder score
        
        Returns:
            PairFeatures for this pair
        """
        # Extract text content
        text_a = self._get_text(event_a)
        text_b = self._get_text(event_b)
        
        # Extract entities
        entities_a = self._get_entities(event_a)
        entities_b = self._get_entities(event_b)
        
        # Extract structural info
        ts_a = getattr(event_a, 'timestamp', 0)
        ts_b = getattr(event_b, 'timestamp', 0)
        
        episode_a = self._get_episode(event_a)
        episode_b = self._get_episode(event_b)
        
        actor_a = getattr(event_a, 'actor_id', None) or getattr(event_a, 'user_id', None)
        actor_b = getattr(event_b, 'actor_id', None) or getattr(event_b, 'user_id', None)
        
        return PairFeatures(
            # Semantic
            embedding_similarity=embedding_similarity,
            entity_overlap=entity_overlap(entities_a, entities_b),
            
            # Contradiction signals
            negation_mismatch=negation_mismatch(text_a, text_b),
            numeric_mismatch=numeric_mismatch(text_a, text_b),
            temporal_mismatch=temporal_mismatch(text_a, text_b),
            antonym_detected=antonym_detected(text_a, text_b),
            cancellation_detected=cancellation_mismatch(text_a, text_b),
            
            # Structural
            timestamp_delta=abs(ts_a - ts_b),
            same_episode=(episode_a == episode_b and episode_a is not None),
            same_actor=(actor_a == actor_b and actor_a is not None),
            reply_pattern=detect_reply_pattern(text_a),
            
            # Optional high-accuracy
            cross_encoder_score=cross_encoder_score,
        )
    
    def extract_batch(
        self,
        new_event: Any,
        candidates: List[Any],
        similarities: Dict[str, float],
    ) -> Dict[str, PairFeatures]:
        """
        Extract features for multiple candidate pairs.
        
        Args:
            new_event: The new event
            candidates: List of candidate events
            similarities: Dict of event_id -> embedding similarity
        
        Returns:
            Dict of event_id -> PairFeatures
        """
        results = {}
        
        for candidate in candidates:
            event_id = getattr(candidate, 'event_id', str(id(candidate)))
            similarity = similarities.get(event_id, 0.0)
            
            features = self.extract(
                event_a=new_event,
                event_b=candidate,
                embedding_similarity=similarity,
            )
            
            results[event_id] = features
        
        return results
    
    def _get_text(self, event: Any) -> str:
        """Extract text from event."""
        content = getattr(event, 'content', {})
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            # Try common text fields
            for field in ["text", "message", "content", "body", "query", "response"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            
            # Concatenate query + response if present
            if "query" in content and "response" in content:
                return f"{content['query']} {content['response']}"
        
        return str(content) if content else ""
    
    def _get_entities(self, event: Any) -> List[Dict[str, Any]]:
        """Extract entities from event metadata."""
        metadata = getattr(event, 'metadata', {})
        if isinstance(metadata, dict):
            entities = metadata.get('entities', [])
            if isinstance(entities, list):
                return entities
        return []
    
    def _get_episode(self, event: Any) -> Optional[str]:
        """Extract episode_id from event."""
        # Try direct attribute
        episode = getattr(event, 'episode_id', None)
        if episode:
            return episode
        
        # Try metadata
        metadata = getattr(event, 'metadata', {})
        if isinstance(metadata, dict):
            return metadata.get('episode_id')
        
        return None


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def extract_pair_features(
    event_a: Any,
    event_b: Any,
    embedding_similarity: float = 0.0,
) -> PairFeatures:
    """
    Convenience function to extract features from an event pair.
    
    Args:
        event_a: First event
        event_b: Second event
        embedding_similarity: Pre-computed similarity
    
    Returns:
        PairFeatures
    """
    extractor = FeatureExtractor()
    return extractor.extract(event_a, event_b, embedding_similarity)


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Core dataclass
    'PairFeatures',
    
    # Extractor
    'FeatureExtractor',
    'extract_pair_features',
    
    # Individual detectors
    'has_negation',
    'negation_mismatch',
    'has_cancellation',
    'has_correction',
    'cancellation_mismatch',
    'extract_numerics',
    'numeric_mismatch',
    'extract_temporal',
    'temporal_mismatch',
    'find_antonyms',
    'antonym_detected',
    'entity_overlap',
    'detect_reply_pattern',
    'is_reply_like',
    
    # Types
    'TemporalInfo',
]
