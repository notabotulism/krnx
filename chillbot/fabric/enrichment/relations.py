"""
KRNX Enrichment - Multi-Signal Relation Scoring v3

Detects and scores relationships between events using multi-signal feature vectors.
The core insight: embedding similarity alone fails critical memory cases.

Relation Taxonomy:
- duplicates: Semantically identical (sim ≥ 0.95, no contradiction)
- supersedes: Newer replaces older (sim ≥ 0.70, contradiction, newer timestamp)
- expands_on: Adds detail (sim 0.50-0.90, entity overlap, no contradiction)
- contradicts: Conflicting info (sim ≥ 0.70, contradiction, different actor)
- replies_to: Direct response (temporal proximity, same episode)

Design principle: Detect contradictions BEFORE scoring similarity.
Similarity tells you events are related; signals tell you HOW.

Constitution-safe: Pure, deterministic, no LLM calls.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

from .features import PairFeatures, FeatureExtractor

logger = logging.getLogger(__name__)


# ==============================================
# RELATION TYPES
# ==============================================

class RelationType(Enum):
    """Types of event relationships."""
    DUPLICATES = "duplicates"       # Near-identical content
    SUPERSEDES = "supersedes"       # Replaces/updates previous
    EXPANDS_ON = "expands_on"       # Related/elaborates on
    CONTRADICTS = "contradicts"     # Conflicting information
    REPLIES_TO = "replies_to"       # Response to previous event
    REFERENCES = "references"       # Mentions another event
    CAUSED_BY = "caused_by"         # Causal dependency


# ==============================================
# CONFIGURATION
# ==============================================

@dataclass
class RelationScoringConfig:
    """Configuration for relation scoring thresholds."""
    
    # Similarity thresholds
    duplicate_threshold: float = 0.95       # Very high = duplicate
    supersede_threshold: float = 0.70       # High + contradiction = supersede
    expand_threshold: float = 0.50          # Moderate = expands_on
    contradict_threshold: float = 0.70      # High + contradiction + diff actor
    
    # Reply detection
    reply_gap_threshold: float = 300.0      # 5 min max gap for auto-reply
    
    # Entity overlap for expands_on
    expand_entity_overlap_min: float = 0.3
    
    # Cross-encoder settings
    enable_cross_encoder: bool = True
    cross_encoder_top_k: int = 20           # Rerank top 20 candidates
    
    # Confidence adjustments
    strict_contradiction_boost: float = 0.1  # Boost for 2+ signals


# ==============================================
# RELATION RESULT
# ==============================================

@dataclass
class RelationResult:
    """
    Result of relation scoring for a single pair.
    
    Includes reason_code for human-readable explanations
    and strict_contradiction flag for high-confidence decisions.
    """
    kind: RelationType              # supersedes, contradicts, etc.
    target: str                     # Target event ID
    confidence: float               # 0.0-1.0
    signals: List[str]              # Which signals fired
    reason_code: str                # Human-readable explanation
    strict_contradiction: bool      # 2+ contradiction signals
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "kind": self.kind.value,
            "target": self.target,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
            "reason_code": self.reason_code,
            "strict_contradiction": self.strict_contradiction,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationResult":
        """Create from dictionary."""
        return cls(
            kind=RelationType(data["kind"]),
            target=data["target"],
            confidence=data.get("confidence", 1.0),
            signals=data.get("signals", []),
            reason_code=data.get("reason_code", ""),
            strict_contradiction=data.get("strict_contradiction", False),
            metadata=data.get("metadata", {}),
        )


# ==============================================
# MULTI-SIGNAL RELATION SCORER
# ==============================================

class RelationScorer:
    """
    Multi-signal relation scoring engine.
    
    Scores all possible relations between event pairs using:
    - Embedding similarity (semantic relatedness)
    - Contradiction signals (negation, numeric, temporal, antonym)
    - Structural signals (timestamp, episode, actor)
    
    Returns scored relations with reason codes for transparency.
    """
    
    def __init__(self, config: Optional[RelationScoringConfig] = None):
        """
        Initialize relation scorer.
        
        Args:
            config: Scoring configuration
        """
        self.config = config or RelationScoringConfig()
        self._feature_extractor = FeatureExtractor()
    
    def score_pair(
        self,
        new_event: Any,
        old_event: Any,
        features: PairFeatures,
    ) -> List[RelationResult]:
        """
        Score all possible relations for an event pair.
        
        Args:
            new_event: The newer event
            old_event: The older/candidate event
            features: Pre-extracted PairFeatures
        
        Returns:
            List of RelationResult (may be empty, or multiple)
        """
        results = []
        
        new_id = getattr(new_event, 'event_id', str(id(new_event)))
        old_id = getattr(old_event, 'event_id', str(id(old_event)))
        new_ts = getattr(new_event, 'timestamp', 0)
        old_ts = getattr(old_event, 'timestamp', 0)
        
        sim = features.embedding_similarity
        signals = features.get_fired_signals()
        contradiction_count = features.contradiction_count
        has_contradiction = features.has_contradiction
        strict = features.strict_contradiction
        
        # ==============================================
        # 1. DUPLICATES: High similarity, no contradictions
        # ==============================================
        if sim >= self.config.duplicate_threshold and not has_contradiction:
            results.append(RelationResult(
                kind=RelationType.DUPLICATES,
                target=old_id,
                confidence=sim,
                signals=["semantic_similarity"],
                reason_code="duplicate: semantic match",
                strict_contradiction=False,
                metadata={"similarity": sim},
            ))
        
        # ==============================================
        # 2. SUPERSEDES: Similar content + contradiction + newer
        # ==============================================
        if sim >= self.config.supersede_threshold and has_contradiction:
            if new_ts > old_ts:
                # Base confidence from similarity
                base_confidence = sim * (0.7 + 0.1 * contradiction_count)
                
                # Boost for strict contradiction
                if strict:
                    base_confidence = min(base_confidence + self.config.strict_contradiction_boost, 1.0)
                
                reason_parts = signals + ["newer_event"]
                results.append(RelationResult(
                    kind=RelationType.SUPERSEDES,
                    target=old_id,
                    confidence=min(base_confidence, 1.0),
                    signals=signals,
                    reason_code=f"supersedes: {' + '.join(reason_parts)}",
                    strict_contradiction=strict,
                    metadata={
                        "similarity": sim,
                        "time_delta": new_ts - old_ts,
                    },
                ))
        
        # ==============================================
        # 3. CONTRADICTS: Similar + contradiction + DIFFERENT actor
        # ==============================================
        if sim >= self.config.contradict_threshold and has_contradiction:
            if not features.same_actor:
                base_confidence = sim * (0.6 + 0.1 * contradiction_count)
                
                if strict:
                    base_confidence = min(base_confidence + self.config.strict_contradiction_boost, 1.0)
                
                reason_parts = signals + ["different_actor"]
                results.append(RelationResult(
                    kind=RelationType.CONTRADICTS,
                    target=old_id,
                    confidence=min(base_confidence, 1.0),
                    signals=signals,
                    reason_code=f"contradicts: {' + '.join(reason_parts)}",
                    strict_contradiction=strict,
                    metadata={
                        "similarity": sim,
                    },
                ))
        
        # ==============================================
        # 4. EXPANDS_ON: Moderate similarity, entity overlap, no contradiction
        # ==============================================
        if (self.config.expand_threshold <= sim < self.config.duplicate_threshold 
            and not has_contradiction):
            
            if features.entity_overlap >= self.config.expand_entity_overlap_min:
                confidence = sim * (0.5 + 0.5 * features.entity_overlap)
                
                results.append(RelationResult(
                    kind=RelationType.EXPANDS_ON,
                    target=old_id,
                    confidence=confidence,
                    signals=["entity_overlap", "semantic_similarity"],
                    reason_code=f"expands_on: entity overlap {features.entity_overlap:.2f}",
                    strict_contradiction=False,
                    metadata={
                        "similarity": sim,
                        "entity_overlap": features.entity_overlap,
                    },
                ))
        
        return results
    
    def score_replies_to(
        self,
        new_event: Any,
        previous_event: Optional[Any],
    ) -> Optional[RelationResult]:
        """
        Detect replies_to relationship.
        
        Based on temporal proximity and episode continuity.
        
        Args:
            new_event: The new event
            previous_event: Immediately previous event in stream
        
        Returns:
            RelationResult if reply detected, None otherwise
        """
        if previous_event is None:
            return None
        
        prev_ts = getattr(previous_event, 'timestamp', None)
        prev_id = getattr(previous_event, 'event_id', None)
        new_ts = getattr(new_event, 'timestamp', 0)
        
        if prev_ts is None or prev_id is None:
            return None
        
        gap = new_ts - prev_ts
        
        if gap <= self.config.reply_gap_threshold:
            # Check episode continuity
            new_meta = getattr(new_event, 'metadata', {}) or {}
            prev_meta = getattr(previous_event, 'metadata', {}) or {}
            
            new_episode = new_meta.get('episode_id')
            prev_episode = prev_meta.get('episode_id')
            same_episode = (new_episode == prev_episode and new_episode is not None)
            
            confidence = 1.0 - (gap / self.config.reply_gap_threshold) * 0.3
            if same_episode:
                confidence = min(confidence + 0.1, 1.0)
            
            signals = ["temporal_proximity"]
            reason = f"replies_to: {gap:.1f}s gap"
            if same_episode:
                signals.append("same_episode")
                reason += " + same episode"
            
            return RelationResult(
                kind=RelationType.REPLIES_TO,
                target=prev_id,
                confidence=confidence,
                signals=signals,
                reason_code=reason,
                strict_contradiction=False,
                metadata={"gap_seconds": gap},
            )
        
        return None
    
    def score_candidates(
        self,
        new_event: Any,
        candidates: List[Any],
        similarities: Dict[str, float],
        previous_event: Optional[Any] = None,
    ) -> List[RelationResult]:
        """
        Score all relations for a new event against candidates.
        
        Args:
            new_event: The new event
            candidates: List of candidate events from vector search
            similarities: Dict of event_id -> embedding similarity
            previous_event: Optional previous event for replies_to
        
        Returns:
            List of all detected relations
        """
        all_results = []
        
        # Check replies_to first
        reply = self.score_replies_to(new_event, previous_event)
        if reply:
            all_results.append(reply)
        
        # Score each candidate
        for candidate in candidates:
            candidate_id = getattr(candidate, 'event_id', str(id(candidate)))
            similarity = similarities.get(candidate_id, 0.0)
            
            # Skip if similarity too low
            if similarity < self.config.expand_threshold:
                continue
            
            # Extract features
            features = self._feature_extractor.extract(
                event_a=new_event,
                event_b=candidate,
                embedding_similarity=similarity,
            )
            
            # Score pair
            pair_results = self.score_pair(new_event, candidate, features)
            all_results.extend(pair_results)
        
        # Sort by confidence
        all_results.sort(key=lambda r: r.confidence, reverse=True)
        
        return all_results


# ==============================================
# LEGACY COMPATIBILITY
# ==============================================

@dataclass
class Relation:
    """
    Legacy Relation class for backward compatibility.
    
    Use RelationResult for new code.
    """
    kind: RelationType
    target: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "kind": self.kind.value,
            "target": self.target,
            "confidence": self.confidence,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        return cls(
            kind=RelationType(data["kind"]),
            target=data["target"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata"),
        )


class RelationEnricher:
    """
    Legacy RelationEnricher for backward compatibility.
    
    Wraps RelationScorer with the old API.
    """
    
    def __init__(
        self,
        duplicate_threshold: float = 0.95,
        expand_threshold: float = 0.70,
        supersede_threshold: float = 0.90,
        reply_gap_threshold: float = 300.0,
    ):
        config = RelationScoringConfig(
            duplicate_threshold=duplicate_threshold,
            expand_threshold=expand_threshold,
            supersede_threshold=supersede_threshold,
            reply_gap_threshold=reply_gap_threshold,
        )
        self._scorer = RelationScorer(config)
    
    def detect(
        self,
        event_id: str,
        timestamp: float,
        content: Any,
        previous_event: Optional[Any] = None,
        similarity_scores: Optional[Dict[str, float]] = None,
        workspace_events: Optional[List[Any]] = None,
    ) -> List[Relation]:
        """
        Detect relations (legacy API).
        
        Returns List[Relation] for backward compatibility.
        """
        # Create a minimal event object
        class TempEvent:
            pass
        
        new_event = TempEvent()
        new_event.event_id = event_id
        new_event.timestamp = timestamp
        new_event.content = content
        new_event.metadata = {}
        
        # Score candidates
        candidates = workspace_events or []
        similarities = similarity_scores or {}
        
        results = self._scorer.score_candidates(
            new_event=new_event,
            candidates=candidates,
            similarities=similarities,
            previous_event=previous_event,
        )
        
        # Convert to legacy Relation format
        return [
            Relation(
                kind=r.kind,
                target=r.target,
                confidence=r.confidence,
                metadata={
                    "signals": r.signals,
                    "reason_code": r.reason_code,
                    "strict_contradiction": r.strict_contradiction,
                    **(r.metadata or {}),
                },
            )
            for r in results
        ]
    
    def find_supersession_chain(
        self,
        event_id: str,
        events_with_relations: Dict[str, List[Relation]],
    ) -> List[str]:
        """Follow supersession chain to find all superseded events."""
        chain = [event_id]
        visited = {event_id}
        current = event_id
        
        while True:
            relations = events_with_relations.get(current, [])
            supersedes = [
                r for r in relations
                if r.kind == RelationType.SUPERSEDES
            ]
            
            if not supersedes:
                break
            
            target = supersedes[0].target
            if target in visited:
                break
            
            chain.append(target)
            visited.add(target)
            current = target
        
        return chain
    
    def get_current_version(
        self,
        event_id: str,
        all_events_relations: Dict[str, List[Relation]],
    ) -> str:
        """Find the most current version of an event."""
        # Build reverse index
        superseded_by = {}
        for eid, relations in all_events_relations.items():
            for r in relations:
                if r.kind == RelationType.SUPERSEDES:
                    superseded_by[r.target] = eid
        
        # Follow forward
        current = event_id
        visited = {current}
        
        while current in superseded_by:
            next_id = superseded_by[current]
            if next_id in visited:
                break
            current = next_id
            visited.add(current)
        
        return current


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Core types
    'RelationType',
    'RelationResult',
    'RelationScoringConfig',
    
    # Scorer
    'RelationScorer',
    
    # Legacy compatibility
    'Relation',
    'RelationEnricher',
]
