"""
KRNX Enrichment - Lexicon Loader

Loads signal lexicons from JSON files with support for:
- Package defaults (signals/*.json)
- User overrides (~/.krnx/signals/*.json)

Lexicons are merged at startup, with user entries taking precedence.

Constitution-safe: Pure data loading, no side effects beyond caching.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==============================================
# PATHS
# ==============================================

# Package signals directory (relative to this file)
_PACKAGE_SIGNALS_DIR = Path(__file__).parent

# User override directory
_USER_SIGNALS_DIR = Path.home() / ".krnx" / "signals"


# ==============================================
# LEXICON DATA CLASSES
# ==============================================

@dataclass
class NegationLexicon:
    """Loaded negation lexicon."""
    negators: Set[str] = field(default_factory=set)
    
    def has_negation(self, text: str) -> bool:
        """Check if text contains negation."""
        tokens = set(text.lower().split())
        return bool(tokens & self.negators)


@dataclass
class CancellationLexicon:
    """Loaded cancellation/correction lexicon."""
    cancellation_verbs: Set[str] = field(default_factory=set)
    correction_phrases: List[str] = field(default_factory=list)
    
    def has_cancellation(self, text: str) -> bool:
        """Check if text contains cancellation verb."""
        tokens = set(text.lower().split())
        return bool(tokens & self.cancellation_verbs)
    
    def has_correction(self, text: str) -> bool:
        """Check if text contains correction phrase."""
        lower = text.lower()
        return any(phrase in lower for phrase in self.correction_phrases)


@dataclass
class AntonymLexicon:
    """Loaded antonym pairs lexicon."""
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    _map: Dict[str, Set[str]] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Build bidirectional lookup map."""
        self._rebuild_map()
    
    def _rebuild_map(self):
        """Rebuild the lookup map from pairs."""
        self._map = {}
        for a, b in self.pairs:
            self._map.setdefault(a.lower(), set()).add(b.lower())
            self._map.setdefault(b.lower(), set()).add(a.lower())
    
    def add_pair(self, word_a: str, word_b: str):
        """Add an antonym pair."""
        a, b = word_a.lower(), word_b.lower()
        self.pairs.append((a, b))
        self._map.setdefault(a, set()).add(b)
        self._map.setdefault(b, set()).add(a)
    
    def get_antonyms(self, word: str) -> Set[str]:
        """Get all antonyms for a word."""
        return self._map.get(word.lower(), set())
    
    def has_antonym_pair(self, words_a: Set[str], words_b: Set[str]) -> bool:
        """Check if two word sets contain an antonym pair."""
        words_a_lower = {w.lower() for w in words_a}
        words_b_lower = {w.lower() for w in words_b}
        
        for word in words_a_lower:
            if word in self._map:
                if words_b_lower & self._map[word]:
                    return True
        return False


@dataclass
class TemporalLexicon:
    """Loaded temporal conflicts lexicon."""
    days: Set[str] = field(default_factory=set)
    relative_times: Set[str] = field(default_factory=set)
    conflict_pairs: List[Tuple[str, str]] = field(default_factory=list)
    _conflict_set: Set[Tuple[str, str]] = field(default_factory=set, repr=False)
    
    def __post_init__(self):
        """Build conflict lookup set."""
        self._rebuild_conflicts()
    
    def _rebuild_conflicts(self):
        """Rebuild conflict set from pairs."""
        self._conflict_set = set()
        for a, b in self.conflict_pairs:
            # Store both directions for O(1) lookup
            self._conflict_set.add((a.lower(), b.lower()))
            self._conflict_set.add((b.lower(), a.lower()))
    
    def is_conflict(self, term_a: str, term_b: str) -> bool:
        """Check if two temporal terms conflict."""
        return (term_a.lower(), term_b.lower()) in self._conflict_set


@dataclass
class ReplyPatternLexicon:
    """Loaded reply pattern lexicon."""
    acknowledgments: Set[str] = field(default_factory=set)
    follow_ups: Set[str] = field(default_factory=set)
    corrections: Set[str] = field(default_factory=set)
    questions: Set[str] = field(default_factory=set)
    
    def get_pattern_type(self, text: str) -> Optional[str]:
        """
        Detect reply pattern type in text.
        
        Returns: 'acknowledgment', 'follow_up', 'correction', 'question', or None
        """
        lower = text.lower().strip()
        
        # Check corrections first (most specific)
        for phrase in self.corrections:
            if phrase in lower:
                return "correction"
        
        # Check follow-ups
        for phrase in self.follow_ups:
            if phrase in lower:
                return "follow_up"
        
        # Check acknowledgments (often at start)
        first_words = lower.split()[:3]
        for ack in self.acknowledgments:
            if ack in first_words or lower.startswith(ack):
                return "acknowledgment"
        
        # Check questions
        if "?" in text:
            return "question"
        for q in self.questions:
            if lower.startswith(q):
                return "question"
        
        return None


# ==============================================
# LOADER
# ==============================================

class LexiconLoader:
    """
    Loads and caches signal lexicons.
    
    Supports:
    - Package defaults from signals/*.json
    - User overrides from ~/.krnx/signals/*.json
    - Automatic merging (user entries extend/override package defaults)
    """
    
    def __init__(
        self,
        package_dir: Optional[Path] = None,
        user_dir: Optional[Path] = None,
        enable_user_overrides: bool = True,
    ):
        """
        Initialize lexicon loader.
        
        Args:
            package_dir: Package signals directory (default: signals/)
            user_dir: User override directory (default: ~/.krnx/signals/)
            enable_user_overrides: Whether to load user overrides
        """
        self.package_dir = package_dir or _PACKAGE_SIGNALS_DIR
        self.user_dir = user_dir or _USER_SIGNALS_DIR
        self.enable_user_overrides = enable_user_overrides
        
        # Cached lexicons
        self._negation: Optional[NegationLexicon] = None
        self._cancellation: Optional[CancellationLexicon] = None
        self._antonyms: Optional[AntonymLexicon] = None
        self._temporal: Optional[TemporalLexicon] = None
        self._reply_patterns: Optional[ReplyPatternLexicon] = None
    
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load and merge JSON from package and user directories.
        
        Args:
            filename: JSON filename (e.g., 'negation.json')
        
        Returns:
            Merged dictionary
        """
        result = {}
        
        # Load package default
        package_path = self.package_dir / filename
        if package_path.exists():
            try:
                with open(package_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                logger.debug(f"[LEXICON] Loaded package: {package_path}")
            except Exception as e:
                logger.warning(f"[LEXICON] Failed to load {package_path}: {e}")
        
        # Load and merge user override
        if self.enable_user_overrides:
            user_path = self.user_dir / filename
            if user_path.exists():
                try:
                    with open(user_path, 'r', encoding='utf-8') as f:
                        user_data = json.load(f)
                    result = self._merge_dicts(result, user_data)
                    logger.info(f"[LEXICON] Merged user override: {user_path}")
                except Exception as e:
                    logger.warning(f"[LEXICON] Failed to load user override {user_path}: {e}")
        
        return result
    
    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """
        Merge two dictionaries, extending lists and sets.
        
        Override values take precedence for scalars.
        Lists/sets are extended (not replaced).
        """
        result = dict(base)
        
        for key, value in override.items():
            if key.startswith("_"):
                continue  # Skip metadata
            
            if key not in result:
                result[key] = value
            elif isinstance(value, list) and isinstance(result[key], list):
                # Extend lists, dedupe
                combined = result[key] + value
                result[key] = list(dict.fromkeys(combined))
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @property
    def negation(self) -> NegationLexicon:
        """Get negation lexicon (cached)."""
        if self._negation is None:
            data = self._load_json("negation.json")
            self._negation = NegationLexicon(
                negators=set(data.get("negators", []))
            )
        return self._negation
    
    @property
    def cancellation(self) -> CancellationLexicon:
        """Get cancellation lexicon (cached)."""
        if self._cancellation is None:
            data = self._load_json("cancellation.json")
            self._cancellation = CancellationLexicon(
                cancellation_verbs=set(data.get("cancellation_verbs", [])),
                correction_phrases=data.get("correction_phrases", []),
            )
        return self._cancellation
    
    @property
    def antonyms(self) -> AntonymLexicon:
        """Get antonyms lexicon (cached)."""
        if self._antonyms is None:
            data = self._load_json("antonyms.json")
            pairs = [tuple(p) for p in data.get("pairs", []) if len(p) == 2]
            self._antonyms = AntonymLexicon(pairs=pairs)
        return self._antonyms
    
    @property
    def temporal(self) -> TemporalLexicon:
        """Get temporal conflicts lexicon (cached)."""
        if self._temporal is None:
            data = self._load_json("temporal_conflicts.json")
            conflict_pairs = [tuple(p) for p in data.get("conflict_pairs", []) if len(p) == 2]
            self._temporal = TemporalLexicon(
                days=set(data.get("days", [])),
                relative_times=set(data.get("relative_times", [])),
                conflict_pairs=conflict_pairs,
            )
        return self._temporal
    
    @property
    def reply_patterns(self) -> ReplyPatternLexicon:
        """Get reply patterns lexicon (cached)."""
        if self._reply_patterns is None:
            data = self._load_json("reply_patterns.json")
            self._reply_patterns = ReplyPatternLexicon(
                acknowledgments=set(data.get("acknowledgments", [])),
                follow_ups=set(data.get("follow_ups", [])),
                corrections=set(data.get("corrections", [])),
                questions=set(data.get("questions", [])),
            )
        return self._reply_patterns
    
    def reload(self):
        """Clear cache and reload all lexicons."""
        self._negation = None
        self._cancellation = None
        self._antonyms = None
        self._temporal = None
        self._reply_patterns = None
        logger.info("[LEXICON] Cache cleared, will reload on next access")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            "package_dir": str(self.package_dir),
            "user_dir": str(self.user_dir),
            "user_overrides_enabled": self.enable_user_overrides,
            "user_dir_exists": self.user_dir.exists(),
            "loaded": {
                "negation": self._negation is not None,
                "cancellation": self._cancellation is not None,
                "antonyms": self._antonyms is not None,
                "temporal": self._temporal is not None,
                "reply_patterns": self._reply_patterns is not None,
            },
        }


# ==============================================
# GLOBAL INSTANCE
# ==============================================

_default_loader: Optional[LexiconLoader] = None


def get_loader() -> LexiconLoader:
    """Get or create the default lexicon loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = LexiconLoader()
    return _default_loader


def get_negation_lexicon() -> NegationLexicon:
    """Get the negation lexicon."""
    return get_loader().negation


def get_cancellation_lexicon() -> CancellationLexicon:
    """Get the cancellation lexicon."""
    return get_loader().cancellation


def get_antonym_lexicon() -> AntonymLexicon:
    """Get the antonyms lexicon."""
    return get_loader().antonyms


def get_temporal_lexicon() -> TemporalLexicon:
    """Get the temporal conflicts lexicon."""
    return get_loader().temporal


def get_reply_pattern_lexicon() -> ReplyPatternLexicon:
    """Get the reply patterns lexicon."""
    return get_loader().reply_patterns


def reload_lexicons():
    """Reload all lexicons from disk."""
    get_loader().reload()


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Data classes
    'NegationLexicon',
    'CancellationLexicon',
    'AntonymLexicon',
    'TemporalLexicon',
    'ReplyPatternLexicon',
    
    # Loader
    'LexiconLoader',
    
    # Global access
    'get_loader',
    'get_negation_lexicon',
    'get_cancellation_lexicon',
    'get_antonym_lexicon',
    'get_temporal_lexicon',
    'get_reply_pattern_lexicon',
    'reload_lexicons',
]
