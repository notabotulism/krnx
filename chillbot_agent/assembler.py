"""
ChillBot Agent - Context Assembler

Builds LLM-ready context from processed results.

Features:
- Salience-based ordering (most important first)
- Episode grouping (chronological within episodes)
- Token budget management
- Multiple output formats (text, messages, JSON)
- Source citation support

Extends KRNX's ContextBuilder with:
- Episode-aware grouping
- Salience-based prioritization
- Relation-aware context (shows supersession info)

Constitution-safe: Read-only assembly, no mutations.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from collections import defaultdict

from .processor import ProcessedResult

logger = logging.getLogger(__name__)


# ==============================================
# CONTEXT CONFIGURATION
# ==============================================

@dataclass
class ContextConfig:
    """Configuration for context assembly."""
    
    # Token limits
    max_tokens: int = 4000
    reserve_tokens: int = 500       # Reserve for query + response
    
    # Formatting
    separator: str = "\n---\n"
    include_timestamps: bool = True
    include_sources: bool = True
    time_format: str = "relative"   # relative, absolute, both
    
    # Ordering
    order_by: str = "salience"      # salience, chronological, episode
    group_by_episode: bool = True
    
    # Truncation
    truncate_long: bool = True
    max_memory_chars: int = 1000
    
    # Output
    output_format: str = "text"     # text, messages, json


# ==============================================
# ASSEMBLED CONTEXT
# ==============================================

@dataclass
class AssembledContext:
    """
    Result of context assembly.
    
    Contains formatted context ready for LLM.
    """
    # The assembled context
    context: Union[str, List[Dict], Dict] = ""
    
    # Metadata
    events_included: int = 0
    events_truncated: int = 0
    tokens_used: int = 0
    episodes_included: int = 0
    
    # Source tracking (for citations)
    source_map: Dict[int, str] = field(default_factory=dict)  # index -> event_id
    
    # Assembly info
    latency_ms: float = 0.0
    config_used: Optional[ContextConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "events_included": self.events_included,
            "events_truncated": self.events_truncated,
            "tokens_used": self.tokens_used,
            "episodes_included": self.episodes_included,
            "latency_ms": round(self.latency_ms, 2),
        }


# ==============================================
# CONTEXT ASSEMBLER
# ==============================================

class ContextAssembler:
    """
    Assembles processed results into LLM-ready context.
    
    Features:
    - Episode-aware grouping
    - Salience-based prioritization
    - Token budget management
    - Source citations
    """
    
    # Token estimation (conservative)
    CHARS_PER_TOKEN = 4
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """
        Initialize context assembler.
        
        Args:
            config: Assembly configuration
        """
        self.config = config or ContextConfig()
        logger.info(f"[ASSEMBLER] Initialized (max_tokens={self.config.max_tokens})")
    
    def assemble(
        self,
        processed: ProcessedResult,
        query: str,
        config: Optional[ContextConfig] = None,
    ) -> AssembledContext:
        """
        Assemble context from processed results.
        
        Args:
            processed: ProcessedResult from processor
            query: Original query (for header)
            config: Optional override config
        
        Returns:
            AssembledContext ready for LLM
        """
        start_time = time.time()
        config = config or self.config
        
        events = processed.events
        salience_scores = processed.salience_scores
        
        # Group by episode if configured
        if config.group_by_episode:
            grouped = self._group_by_episode(events)
        else:
            grouped = {"default": events}
        
        # Order within groups
        ordered_groups = self._order_groups(grouped, salience_scores, config)
        
        # Build context based on format
        if config.output_format == "messages":
            result = self._build_messages(ordered_groups, query, config)
        elif config.output_format == "json":
            result = self._build_json(ordered_groups, query, config)
        else:
            result = self._build_text(ordered_groups, query, config)
        
        result.latency_ms = (time.time() - start_time) * 1000
        result.config_used = config
        
        logger.info(
            f"[ASSEMBLER] Built {config.output_format} context: "
            f"{result.events_included} events, ~{result.tokens_used} tokens"
        )
        
        return result
    
    # ==============================================
    # EPISODE GROUPING
    # ==============================================
    
    def _group_by_episode(
        self,
        events: List[Any],
    ) -> Dict[str, List[Any]]:
        """
        Group events by episode_id.
        
        Events without episode_id go to "unknown" group.
        """
        groups = defaultdict(list)
        
        for event in events:
            metadata = getattr(event, 'metadata', {}) or {}
            
            # Try to get episode_id from various places
            episode_id = None
            
            # Check temporal metadata
            temporal = metadata.get('temporal', {})
            if isinstance(temporal, dict):
                episode_id = temporal.get('episode_id')
            
            # Direct episode_id
            if not episode_id:
                episode_id = metadata.get('episode_id')
            
            # Fallback
            if not episode_id:
                episode_id = "unknown"
            
            groups[episode_id].append(event)
        
        return dict(groups)
    
    def _order_groups(
        self,
        groups: Dict[str, List[Any]],
        salience_scores: Dict[str, float],
        config: ContextConfig,
    ) -> List[Tuple[str, List[Any]]]:
        """
        Order episode groups and events within them.
        
        Returns list of (episode_id, events) tuples.
        """
        result = []
        
        for episode_id, events in groups.items():
            # Order events within episode
            if config.order_by == "chronological":
                # Oldest first
                events = sorted(events, key=lambda e: getattr(e, 'timestamp', 0))
            elif config.order_by == "salience":
                # Highest salience first
                events = sorted(
                    events,
                    key=lambda e: salience_scores.get(self._get_id(e), 0),
                    reverse=True
                )
            # else: keep original order
            
            result.append((episode_id, events))
        
        # Order episodes by average salience of their events
        def episode_salience(item):
            ep_id, ep_events = item
            if not ep_events:
                return 0
            scores = [salience_scores.get(self._get_id(e), 0) for e in ep_events]
            return sum(scores) / len(scores)
        
        result.sort(key=episode_salience, reverse=True)
        
        return result
    
    # ==============================================
    # TEXT FORMAT
    # ==============================================
    
    def _build_text(
        self,
        ordered_groups: List[Tuple[str, List[Any]]],
        query: str,
        config: ContextConfig,
    ) -> AssembledContext:
        """Build plain text context."""
        available_tokens = config.max_tokens - config.reserve_tokens
        
        parts = []
        source_map = {}
        tokens_used = 0
        events_included = 0
        events_truncated = 0
        episodes_included = set()
        memory_index = 1
        
        # Header
        header = f"Relevant context for: {query}\n"
        header_tokens = self._estimate_tokens(header)
        parts.append(header)
        tokens_used += header_tokens
        
        # Process each episode group
        for episode_id, events in ordered_groups:
            # Episode header (if grouping)
            if config.group_by_episode and episode_id != "unknown":
                ep_header = f"\n[Episode: {episode_id}]\n"
                ep_tokens = self._estimate_tokens(ep_header)
                if tokens_used + ep_tokens > available_tokens:
                    break
                parts.append(ep_header)
                tokens_used += ep_tokens
                episodes_included.add(episode_id)
            
            # Add events
            for event in events:
                memory_text = self._format_event_text(
                    event, memory_index, config
                )
                memory_tokens = self._estimate_tokens(memory_text)
                
                # Check budget
                if tokens_used + memory_tokens > available_tokens:
                    # Try truncating
                    if config.truncate_long:
                        remaining = available_tokens - tokens_used - 20
                        if remaining > 50:
                            truncated = self._truncate(memory_text, remaining)
                            parts.append(truncated)
                            tokens_used += self._estimate_tokens(truncated)
                            events_truncated += 1
                    break
                
                parts.append(memory_text)
                tokens_used += memory_tokens
                source_map[memory_index] = self._get_id(event)
                events_included += 1
                memory_index += 1
        
        context = config.separator.join(parts)
        
        return AssembledContext(
            context=context,
            events_included=events_included,
            events_truncated=events_truncated,
            tokens_used=tokens_used,
            episodes_included=len(episodes_included),
            source_map=source_map,
        )
    
    def _format_event_text(
        self,
        event: Any,
        index: int,
        config: ContextConfig,
    ) -> str:
        """Format a single event as text."""
        parts = []
        
        # Index
        parts.append(f"[{index}]")
        
        # Timestamp
        if config.include_timestamps:
            timestamp = getattr(event, 'timestamp', None)
            if timestamp:
                time_str = self._format_time(timestamp, config.time_format)
                parts.append(f"({time_str})")
        
        # Content
        content = self._get_text(event)
        
        # Truncate if needed
        if len(content) > config.max_memory_chars:
            content = content[:config.max_memory_chars - 3] + "..."
        
        parts.append(content)
        
        # Source
        if config.include_sources:
            event_id = self._get_id(event)
            parts.append(f"[ref:{event_id[:12]}]")
        
        return " ".join(parts)
    
    # ==============================================
    # MESSAGES FORMAT (for chat APIs)
    # ==============================================
    
    def _build_messages(
        self,
        ordered_groups: List[Tuple[str, List[Any]]],
        query: str,
        config: ContextConfig,
    ) -> AssembledContext:
        """Build chat messages format."""
        available_tokens = config.max_tokens - config.reserve_tokens
        
        memory_texts = []
        source_map = {}
        tokens_used = 100  # Base overhead
        events_included = 0
        memory_index = 1
        
        for episode_id, events in ordered_groups:
            for event in events:
                memory_text = self._format_event_text(event, memory_index, config)
                memory_tokens = self._estimate_tokens(memory_text)
                
                if tokens_used + memory_tokens > available_tokens:
                    break
                
                memory_texts.append(memory_text)
                source_map[memory_index] = self._get_id(event)
                tokens_used += memory_tokens
                events_included += 1
                memory_index += 1
        
        system_content = (
            "You have access to the following memories from past conversations:\n\n"
            + "\n\n".join(memory_texts)
            + "\n\nUse these memories to answer the user's question accurately. "
            "If citing a memory, reference it by its number [N]."
        )
        
        messages = [
            {"role": "system", "content": system_content},
        ]
        
        return AssembledContext(
            context=messages,
            events_included=events_included,
            events_truncated=0,
            tokens_used=tokens_used,
            source_map=source_map,
        )
    
    # ==============================================
    # JSON FORMAT
    # ==============================================
    
    def _build_json(
        self,
        ordered_groups: List[Tuple[str, List[Any]]],
        query: str,
        config: ContextConfig,
    ) -> AssembledContext:
        """Build JSON format."""
        available_tokens = config.max_tokens - config.reserve_tokens
        
        memories = []
        source_map = {}
        tokens_used = 100
        events_included = 0
        
        for episode_id, events in ordered_groups:
            for event in events:
                memory_dict = {
                    "id": self._get_id(event),
                    "content": self._get_text(event),
                    "timestamp": getattr(event, 'timestamp', None),
                    "episode": episode_id if episode_id != "unknown" else None,
                }
                
                # Estimate tokens for this entry
                import json
                entry_tokens = self._estimate_tokens(json.dumps(memory_dict))
                
                if tokens_used + entry_tokens > available_tokens:
                    break
                
                memories.append(memory_dict)
                source_map[events_included + 1] = memory_dict["id"]
                tokens_used += entry_tokens
                events_included += 1
        
        context = {
            "query": query,
            "memories": memories,
            "count": len(memories),
        }
        
        return AssembledContext(
            context=context,
            events_included=events_included,
            events_truncated=0,
            tokens_used=tokens_used,
            source_map=source_map,
        )
    
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
                return f"Q: {content['query']}\nA: {content['response']}"
            
            return str(content)
        
        return str(content) if content else ""
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // self.CHARS_PER_TOKEN
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token limit."""
        max_chars = max_tokens * self.CHARS_PER_TOKEN
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."
    
    def _format_time(self, timestamp: float, format: str) -> str:
        """Format timestamp."""
        from datetime import datetime
        
        now = time.time()
        age_seconds = now - timestamp
        
        if format == "absolute":
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        
        elif format == "both":
            absolute = datetime.fromtimestamp(timestamp).strftime("%m/%d %H:%M")
            relative = self._relative_time(age_seconds)
            return f"{absolute}, {relative}"
        
        else:  # relative
            return self._relative_time(age_seconds)
    
    def _relative_time(self, seconds: float) -> str:
        """Convert seconds to relative time string."""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds / 3600)}h ago"
        elif seconds < 604800:
            return f"{int(seconds / 86400)}d ago"
        else:
            return f"{int(seconds / 604800)}w ago"


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'ContextConfig',
    'AssembledContext',
    'ContextAssembler',
]
