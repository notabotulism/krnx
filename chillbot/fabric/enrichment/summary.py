"""
KRNX Enrichment - Relation Summary

Computes summary statistics for an event's position in the relation graph.
Provides instant visibility for TUI/debugging without graph traversal.

Summary fields:
- edge_count: Total number of relations
- by_kind: Count per relation type
- chain_depth: Depth in supersession chains
- is_leaf: True if no events supersede this one
- is_root: True if this event supersedes nothing
- has_strict_contradiction: True if any relation has 2+ signals

Constitution-safe: Pure, deterministic, no side effects.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from .relations import RelationType, RelationResult

logger = logging.getLogger(__name__)


# ==============================================
# RELATION SUMMARY
# ==============================================

@dataclass
class RelationSummary:
    """
    Summary of an event's position in the relation graph.
    
    Computed for fast access without graph traversal.
    """
    
    # Edge counts
    edge_count: int = 0
    by_kind: Dict[str, int] = field(default_factory=dict)
    
    # Graph position
    chain_depth: int = 0            # Depth in supersession chain
    is_leaf: bool = True            # No events supersede this one
    is_root: bool = True            # This event supersedes nothing
    
    # Flags
    has_strict_contradiction: bool = False  # Any relation has 2+ signals
    has_supersedes: bool = False            # This event supersedes something
    has_contradicts: bool = False           # This event contradicts something
    is_superseded: bool = False             # Something supersedes this event
    is_duplicate: bool = False              # This is a duplicate of something
    
    # Chain info
    supersedes_chain: List[str] = field(default_factory=list)  # IDs we supersede
    superseded_by: Optional[str] = None     # ID that supersedes us (if any)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/display."""
        return {
            "edge_count": self.edge_count,
            "by_kind": self.by_kind,
            "chain_depth": self.chain_depth,
            "is_leaf": self.is_leaf,
            "is_root": self.is_root,
            "has_strict_contradiction": self.has_strict_contradiction,
            "has_supersedes": self.has_supersedes,
            "has_contradicts": self.has_contradicts,
            "is_superseded": self.is_superseded,
            "is_duplicate": self.is_duplicate,
        }
    
    def to_tui_string(self) -> str:
        """Format for TUI display."""
        parts = []
        
        # Edge summary
        if self.edge_count > 0:
            kind_str = ", ".join(f"{count} {kind}" for kind, count in self.by_kind.items())
            parts.append(f"Relations: {kind_str}")
        else:
            parts.append("Relations: none")
        
        # Chain info
        if self.chain_depth > 0:
            chain_str = " → ".join(["supersedes"] * self.chain_depth)
            parts.append(f"Chain depth: {self.chain_depth} ({chain_str})")
        
        # Status
        status = []
        if self.is_leaf:
            status.append("leaf node")
        if self.is_root:
            status.append("root node")
        if self.is_superseded:
            status.append("SUPERSEDED")
        if status:
            parts.append(f"Status: {', '.join(status)}")
        
        # Flags
        if self.has_strict_contradiction:
            parts.append("Strict contradiction: yes")
        
        return "\n".join(parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationSummary":
        """Create from dictionary."""
        return cls(
            edge_count=data.get("edge_count", 0),
            by_kind=data.get("by_kind", {}),
            chain_depth=data.get("chain_depth", 0),
            is_leaf=data.get("is_leaf", True),
            is_root=data.get("is_root", True),
            has_strict_contradiction=data.get("has_strict_contradiction", False),
            has_supersedes=data.get("has_supersedes", False),
            has_contradicts=data.get("has_contradicts", False),
            is_superseded=data.get("is_superseded", False),
            is_duplicate=data.get("is_duplicate", False),
        )


# ==============================================
# SUMMARY COMPUTER
# ==============================================

class RelationSummaryComputer:
    """
    Computes RelationSummary for events.
    
    Can work incrementally (single event) or batch (full graph).
    """
    
    def __init__(self):
        """Initialize summary computer."""
        pass
    
    def compute(
        self,
        event_id: str,
        relations: List[RelationResult],
        all_relations: Optional[Dict[str, List[RelationResult]]] = None,
    ) -> RelationSummary:
        """
        Compute relation summary for a single event.
        
        Args:
            event_id: The event ID
            relations: Relations FROM this event (outgoing)
            all_relations: Optional full graph for chain/leaf detection
        
        Returns:
            RelationSummary
        """
        summary = RelationSummary()
        
        # Count edges by kind
        by_kind: Dict[str, int] = {}
        strict_contradiction = False
        supersedes_chain = []
        
        for rel in relations:
            kind = rel.kind.value if isinstance(rel.kind, RelationType) else str(rel.kind)
            by_kind[kind] = by_kind.get(kind, 0) + 1
            
            if rel.strict_contradiction:
                strict_contradiction = True
            
            if rel.kind == RelationType.SUPERSEDES:
                supersedes_chain.append(rel.target)
        
        summary.edge_count = len(relations)
        summary.by_kind = by_kind
        summary.has_strict_contradiction = strict_contradiction
        summary.supersedes_chain = supersedes_chain
        
        # Set flags from relations
        summary.has_supersedes = RelationType.SUPERSEDES.value in by_kind
        summary.has_contradicts = RelationType.CONTRADICTS.value in by_kind
        summary.is_duplicate = RelationType.DUPLICATES.value in by_kind
        
        # Determine if root (doesn't supersede anything)
        summary.is_root = not summary.has_supersedes
        
        # If we have the full graph, compute chain depth and leaf status
        if all_relations is not None:
            summary.chain_depth = self._compute_chain_depth(event_id, all_relations)
            summary.is_leaf, summary.superseded_by = self._check_leaf_status(
                event_id, all_relations
            )
            summary.is_superseded = summary.superseded_by is not None
        
        return summary
    
    def compute_batch(
        self,
        all_relations: Dict[str, List[RelationResult]],
    ) -> Dict[str, RelationSummary]:
        """
        Compute summaries for all events in a graph.
        
        Args:
            all_relations: Dict of event_id -> list of relations
        
        Returns:
            Dict of event_id -> RelationSummary
        """
        summaries = {}
        
        for event_id, relations in all_relations.items():
            summaries[event_id] = self.compute(
                event_id=event_id,
                relations=relations,
                all_relations=all_relations,
            )
        
        return summaries
    
    def _compute_chain_depth(
        self,
        event_id: str,
        all_relations: Dict[str, List[RelationResult]],
    ) -> int:
        """
        Compute depth in supersession chain.
        
        Follows supersedes relations backward to find chain length.
        """
        depth = 0
        current = event_id
        visited: Set[str] = {current}
        
        while True:
            relations = all_relations.get(current, [])
            supersedes = [
                r for r in relations
                if r.kind == RelationType.SUPERSEDES
            ]
            
            if not supersedes:
                break
            
            # Follow first supersedes link
            target = supersedes[0].target
            if target in visited:
                break  # Cycle detection
            
            depth += 1
            visited.add(target)
            current = target
        
        return depth
    
    def _check_leaf_status(
        self,
        event_id: str,
        all_relations: Dict[str, List[RelationResult]],
    ) -> tuple:
        """
        Check if event is a leaf (nothing supersedes it).
        
        Returns:
            (is_leaf: bool, superseded_by: Optional[str])
        """
        # Check if any other event supersedes this one
        for other_id, relations in all_relations.items():
            if other_id == event_id:
                continue
            
            for rel in relations:
                if rel.kind == RelationType.SUPERSEDES and rel.target == event_id:
                    return False, other_id
        
        return True, None
    
    def find_current_version(
        self,
        event_id: str,
        all_relations: Dict[str, List[RelationResult]],
    ) -> str:
        """
        Find the most current version of an event.
        
        Follows supersession chain forward to find the leaf.
        
        Args:
            event_id: Starting event ID
            all_relations: Full relation graph
        
        Returns:
            ID of the most current version
        """
        # Build reverse index: who supersedes whom
        superseded_by: Dict[str, str] = {}
        
        for eid, relations in all_relations.items():
            for rel in relations:
                if rel.kind == RelationType.SUPERSEDES:
                    superseded_by[rel.target] = eid
        
        # Follow forward
        current = event_id
        visited: Set[str] = {current}
        
        while current in superseded_by:
            next_id = superseded_by[current]
            if next_id in visited:
                break  # Cycle
            current = next_id
            visited.add(current)
        
        return current
    
    def find_supersession_chain(
        self,
        event_id: str,
        all_relations: Dict[str, List[RelationResult]],
    ) -> List[str]:
        """
        Get the full supersession chain ending at this event.
        
        Returns list of event IDs from newest to oldest.
        
        Args:
            event_id: Event ID (should be the current/newest version)
            all_relations: Full relation graph
        
        Returns:
            List of event IDs in chain order
        """
        chain = [event_id]
        visited: Set[str] = {event_id}
        current = event_id
        
        while True:
            relations = all_relations.get(current, [])
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
    
    def get_effective_events(
        self,
        all_relations: Dict[str, List[RelationResult]],
    ) -> List[str]:
        """
        Get all "effective" events (leaf nodes in supersession graph).
        
        These are events that haven't been superseded by anything else.
        
        Args:
            all_relations: Full relation graph
        
        Returns:
            List of event IDs that are current/effective
        """
        # Find all events that are superseded
        superseded: Set[str] = set()
        
        for relations in all_relations.values():
            for rel in relations:
                if rel.kind == RelationType.SUPERSEDES:
                    superseded.add(rel.target)
        
        # Return events not in superseded set
        return [
            event_id for event_id in all_relations.keys()
            if event_id not in superseded
        ]


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def compute_relation_summary(
    event_id: str,
    relations: List[RelationResult],
    all_relations: Optional[Dict[str, List[RelationResult]]] = None,
) -> RelationSummary:
    """
    Convenience function to compute relation summary.
    
    Args:
        event_id: Event ID
        relations: Relations from this event
        all_relations: Optional full graph
    
    Returns:
        RelationSummary
    """
    computer = RelationSummaryComputer()
    return computer.compute(event_id, relations, all_relations)


def get_current_version(
    event_id: str,
    all_relations: Dict[str, List[RelationResult]],
) -> str:
    """
    Convenience function to find current version of an event.
    
    Args:
        event_id: Event ID
        all_relations: Full relation graph
    
    Returns:
        ID of the most current version
    """
    computer = RelationSummaryComputer()
    return computer.find_current_version(event_id, all_relations)


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'RelationSummary',
    'RelationSummaryComputer',
    'compute_relation_summary',
    'get_current_version',
]
