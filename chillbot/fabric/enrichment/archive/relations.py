"""
KRNX Fabric - Relation Enricher

Detects relationships between events:
- replies_to: This event responds to previous event
- duplicates: Near-identical content (similarity > 0.95)
- expands_on: Related content (similarity 0.7-0.95)
- supersedes: This event replaces/updates a previous event

The "supersedes" relation is key for the "metadata as graph" thesis:
- Enables correction chains without deletion
- Audit trail preserved
- Query can follow supersession chain to get "current truth"

Constitution-safe: Pure, deterministic, no side effects.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class RelationType(Enum):
    """Types of event relationships."""
    REPLIES_TO = "replies_to"       # Response to previous event
    DUPLICATES = "duplicates"       # Near-identical content
    EXPANDS_ON = "expands_on"       # Related/elaborates on
    SUPERSEDES = "supersedes"       # Replaces/updates previous
    CAUSED_BY = "caused_by"         # Causal dependency
    REFERENCES = "references"       # Mentions another event


@dataclass
class Relation:
    """A detected relationship between events."""
    kind: RelationType
    target: str                     # Target event_id
    confidence: float = 1.0         # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Create from dictionary."""
        return cls(
            kind=RelationType(data["kind"]),
            target=data["target"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata"),
        )


class RelationEnricher:
    """
    Detects relationships between events.
    
    Uses:
    - Temporal proximity (replies_to)
    - Semantic similarity (duplicates, expands_on)
    - Content signals (supersedes)
    """
    
    def __init__(
        self,
        duplicate_threshold: float = 0.95,
        expand_threshold: float = 0.70,
        supersede_threshold: float = 0.90,
        reply_gap_threshold: float = 300.0,  # 5 min
    ):
        """
        Initialize relation enricher.
        
        Args:
            duplicate_threshold: Similarity for duplicate detection
            expand_threshold: Minimum similarity for expands_on
            supersede_threshold: Similarity + recency for supersedes
            reply_gap_threshold: Max gap for automatic replies_to
        """
        self.duplicate_threshold = duplicate_threshold
        self.expand_threshold = expand_threshold
        self.supersede_threshold = supersede_threshold
        self.reply_gap_threshold = reply_gap_threshold
    
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
        Detect all relationships for an event.
        
        Args:
            event_id: Current event ID
            timestamp: Current event timestamp
            content: Event content
            previous_event: Immediately previous event
            similarity_scores: Pre-computed similarities {event_id: score}
            workspace_events: Recent events for comparison
        
        Returns:
            List of detected Relations
        """
        relations = []
        
        # 1. REPLIES_TO: Previous event in stream
        if previous_event is not None:
            reply_relation = self._detect_reply(
                timestamp=timestamp,
                previous_event=previous_event,
            )
            if reply_relation:
                relations.append(reply_relation)
        
        # 2. DUPLICATES / EXPANDS_ON / SUPERSEDES from similarity scores
        if similarity_scores:
            semantic_relations = self._detect_from_similarity(
                event_id=event_id,
                timestamp=timestamp,
                content=content,
                similarity_scores=similarity_scores,
                workspace_events=workspace_events,
            )
            relations.extend(semantic_relations)
        
        # 3. SUPERSEDES from content signals (corrections, updates)
        content_supersedes = self._detect_supersedes_from_content(
            content=content,
            workspace_events=workspace_events,
        )
        relations.extend(content_supersedes)
        
        return relations
    
    def _detect_reply(
        self,
        timestamp: float,
        previous_event: Any,
    ) -> Optional[Relation]:
        """
        Detect replies_to relationship.
        
        Simple rule: If previous event exists and gap is reasonable,
        this event replies to it.
        """
        prev_timestamp = getattr(previous_event, 'timestamp', None)
        prev_id = getattr(previous_event, 'event_id', None)
        
        if prev_timestamp is None or prev_id is None:
            return None
        
        gap = timestamp - prev_timestamp
        
        # Only create reply if gap is reasonable
        if gap <= self.reply_gap_threshold:
            return Relation(
                kind=RelationType.REPLIES_TO,
                target=prev_id,
                confidence=1.0,
                metadata={"gap_seconds": gap},
            )
        
        return None
    
    def _detect_from_similarity(
        self,
        event_id: str,
        timestamp: float,
        content: Any,
        similarity_scores: Dict[str, float],
        workspace_events: Optional[List[Any]] = None,
    ) -> List[Relation]:
        """
        Detect relations based on semantic similarity.
        
        Thresholds:
        - >= duplicate_threshold: DUPLICATES
        - >= supersede_threshold + recency: SUPERSEDES
        - >= expand_threshold: EXPANDS_ON
        """
        relations = []
        
        # Build event lookup for timestamps
        event_lookup = {}
        if workspace_events:
            for e in workspace_events:
                eid = getattr(e, 'event_id', None)
                if eid:
                    event_lookup[eid] = e
        
        for target_id, similarity in similarity_scores.items():
            # Skip self
            if target_id == event_id:
                continue
            
            # DUPLICATES: Very high similarity
            if similarity >= self.duplicate_threshold:
                relations.append(Relation(
                    kind=RelationType.DUPLICATES,
                    target=target_id,
                    confidence=similarity,
                    metadata={"similarity": similarity},
                ))
                continue  # Don't also mark as expands_on
            
            # SUPERSEDES: High similarity + this is newer
            if similarity >= self.supersede_threshold:
                target_event = event_lookup.get(target_id)
                if target_event:
                    target_ts = getattr(target_event, 'timestamp', 0)
                    if timestamp > target_ts:
                        # This event is newer and very similar = supersedes
                        relations.append(Relation(
                            kind=RelationType.SUPERSEDES,
                            target=target_id,
                            confidence=similarity,
                            metadata={
                                "similarity": similarity,
                                "time_delta": timestamp - target_ts,
                            },
                        ))
                        continue
            
            # EXPANDS_ON: Moderate similarity
            if similarity >= self.expand_threshold:
                relations.append(Relation(
                    kind=RelationType.EXPANDS_ON,
                    target=target_id,
                    confidence=similarity,
                    metadata={"similarity": similarity},
                ))
        
        return relations
    
    def _detect_supersedes_from_content(
        self,
        content: Any,
        workspace_events: Optional[List[Any]] = None,
    ) -> List[Relation]:
        """
        Detect supersedes from content signals.
        
        Looks for explicit markers:
        - "correction:" prefix
        - "update:" prefix
        - "supersedes: evt_xxx"
        - metadata.supersedes field
        """
        relations = []
        
        # Check if content is dict
        if not isinstance(content, dict):
            return relations
        
        # Explicit supersedes in content
        if "supersedes" in content:
            target = content["supersedes"]
            if isinstance(target, str):
                relations.append(Relation(
                    kind=RelationType.SUPERSEDES,
                    target=target,
                    confidence=1.0,
                    metadata={"source": "explicit"},
                ))
            elif isinstance(target, list):
                for t in target:
                    relations.append(Relation(
                        kind=RelationType.SUPERSEDES,
                        target=t,
                        confidence=1.0,
                        metadata={"source": "explicit"},
                    ))
        
        # Check for correction markers in text
        text = content.get("text", "") or content.get("message", "")
        if isinstance(text, str):
            text_lower = text.lower()
            
            # "correction: ..." or "actually, ..."
            if text_lower.startswith(("correction:", "update:", "actually,")):
                # This is a correction, but we don't know what it supersedes
                # without more context. Mark it for potential linking.
                pass
        
        return relations
    
    def find_supersession_chain(
        self,
        event_id: str,
        events_with_relations: Dict[str, List[Relation]],
    ) -> List[str]:
        """
        Follow supersession chain to find all superseded events.
        
        Useful for finding the "original" event that led to corrections.
        
        Args:
            event_id: Starting event
            events_with_relations: Map of event_id -> relations
        
        Returns:
            Chain of event IDs (newest to oldest)
        """
        chain = [event_id]
        visited = {event_id}
        current = event_id
        
        while True:
            relations = events_with_relations.get(current, [])
            supersedes = [
                r for r in relations
                if isinstance(r, Relation) and r.kind == RelationType.SUPERSEDES
            ]
            
            if not supersedes:
                break
            
            # Follow first supersedes link
            target = supersedes[0].target
            if target in visited:
                break  # Cycle detection
            
            chain.append(target)
            visited.add(target)
            current = target
        
        return chain
    
    def get_current_version(
        self,
        event_id: str,
        all_events_relations: Dict[str, List[Relation]],
    ) -> str:
        """
        Find the most current version of an event (follow supersession forward).
        
        Args:
            event_id: Event to check
            all_events_relations: Map of all events -> their relations
        
        Returns:
            ID of the most current version
        """
        # Build reverse index: who supersedes whom
        superseded_by = {}
        for eid, relations in all_events_relations.items():
            for r in relations:
                if isinstance(r, Relation) and r.kind == RelationType.SUPERSEDES:
                    superseded_by[r.target] = eid
        
        # Follow forward
        current = event_id
        visited = {current}
        
        while current in superseded_by:
            next_id = superseded_by[current]
            if next_id in visited:
                break  # Cycle
            current = next_id
            visited.add(current)
        
        return current
    
    def __repr__(self) -> str:
        return (
            f"RelationEnricher("
            f"dup={self.duplicate_threshold}, "
            f"exp={self.expand_threshold}, "
            f"sup={self.supersede_threshold})"
        )
