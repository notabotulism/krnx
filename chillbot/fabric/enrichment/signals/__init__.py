"""
KRNX Enrichment - Signal Lexicons

External JSON lexicons for deterministic signal detection.
Supports user overrides at ~/.krnx/signals/

Lexicons:
- negation.json: Negation words for contradiction detection
- cancellation.json: Cancellation/correction verbs
- antonyms.json: Antonym pairs for contradiction detection
- temporal_conflicts.json: Temporal reference conflicts
- reply_patterns.json: Reply/acknowledgment patterns

Usage:
    from chillbot.fabric.enrichment.signals import get_negation_lexicon, get_antonym_lexicon
    
    negation = get_negation_lexicon()
    if negation.has_negation("This is not approved"):
        print("Negation detected!")
    
    antonyms = get_antonym_lexicon()
    if antonyms.has_antonym_pair({"approved"}, {"rejected"}):
        print("Antonym pair detected!")
"""

from .loader import (
    # Data classes
    NegationLexicon,
    CancellationLexicon,
    AntonymLexicon,
    TemporalLexicon,
    ReplyPatternLexicon,
    
    # Loader
    LexiconLoader,
    
    # Global access
    get_loader,
    get_negation_lexicon,
    get_cancellation_lexicon,
    get_antonym_lexicon,
    get_temporal_lexicon,
    get_reply_pattern_lexicon,
    reload_lexicons,
)

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
