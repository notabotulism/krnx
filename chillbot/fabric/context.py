"""
Memory Fabric - Context Builder

Assembles memories into LLM-ready context.
Handles token budgeting, formatting, and prioritization.

Philosophy:
- Token-aware: Respects LLM context limits
- Flexible formats: text, JSON, chat messages
- Priority-based: Most relevant memories first

Usage:
    builder = ContextBuilder(max_tokens=4000)
    
    # Build text context
    context = builder.build(memories, query, format="text")
    
    # Build chat messages
    messages = builder.build(memories, query, format="messages")
"""

import json
import logging
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Approximate tokens per character (conservative estimate)
CHARS_PER_TOKEN = 4


@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_tokens: int = 4000
    reserve_tokens: int = 500  # Reserve for query/response
    separator: str = "\n---\n"
    include_scores: bool = False
    include_timestamps: bool = True
    time_format: str = "relative"  # 'relative', 'absolute', 'both'
    truncate_long_memories: bool = True
    max_memory_tokens: int = 500  # Max tokens per memory


class ContextBuilder:
    """
    Build LLM-ready context from memories.
    
    Handles:
    - Token budgeting (fits within limit)
    - Multiple formats (text, JSON, messages)
    - Prioritization (highest score first)
    - Metadata inclusion (optional)
    """
    
    def __init__(
        self,
        max_tokens: int = 4000,
        config: Optional[ContextConfig] = None,
    ):
        """
        Initialize context builder.
        
        Args:
            max_tokens: Maximum tokens for context
            config: Optional detailed configuration
        """
        self.config = config or ContextConfig(max_tokens=max_tokens)
        if max_tokens != 4000:
            self.config.max_tokens = max_tokens
    
    def build(
        self,
        memories: List[Any],  # List of MemoryItem
        query: str,
        format: str = "text",
        include_metadata: bool = False,
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Build context from memories.
        
        Args:
            memories: List of MemoryItem objects
            query: The query for context (for relevance hints)
            format: Output format ('text', 'json', 'messages')
            include_metadata: Include memory metadata
        
        Returns:
            Formatted context based on format parameter
        """
        if format == "text":
            return self._build_text(memories, query, include_metadata)
        elif format == "json":
            return self._build_json(memories, query, include_metadata)
        elif format == "messages":
            return self._build_messages(memories, query, include_metadata)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _build_text(
        self,
        memories: List[Any],
        query: str,
        include_metadata: bool,
    ) -> str:
        """Build plain text context."""
        available_tokens = self.config.max_tokens - self.config.reserve_tokens
        
        parts = []
        tokens_used = 0
        
        # Header
        header = f"Relevant memories for: {query}\n"
        header_tokens = self._estimate_tokens(header)
        tokens_used += header_tokens
        parts.append(header)
        
        # Add memories in priority order (already sorted by score)
        for i, memory in enumerate(memories):
            memory_text = self._format_memory_text(memory, i + 1, include_metadata)
            memory_tokens = self._estimate_tokens(memory_text)
            
            # Check if we have room
            if tokens_used + memory_tokens > available_tokens:
                # Try truncating
                if self.config.truncate_long_memories:
                    truncated = self._truncate_to_tokens(
                        memory_text,
                        available_tokens - tokens_used - 50  # Leave buffer
                    )
                    if truncated:
                        parts.append(truncated)
                        tokens_used += self._estimate_tokens(truncated)
                break
            
            parts.append(memory_text)
            tokens_used += memory_tokens
        
        result = self.config.separator.join(parts)
        
        logger.debug(f"[CONTEXT] Built text context: {tokens_used} tokens, {len(memories)} memories")
        
        return result
    
    def _build_json(
        self,
        memories: List[Any],
        query: str,
        include_metadata: bool,
    ) -> Dict[str, Any]:
        """Build JSON context."""
        available_tokens = self.config.max_tokens - self.config.reserve_tokens
        
        items = []
        tokens_used = 100  # Base JSON overhead
        
        for memory in memories:
            item = self._format_memory_json(memory, include_metadata)
            item_json = json.dumps(item)
            item_tokens = self._estimate_tokens(item_json)
            
            if tokens_used + item_tokens > available_tokens:
                break
            
            items.append(item)
            tokens_used += item_tokens
        
        result = {
            "query": query,
            "memories": items,
            "count": len(items),
            "truncated": len(items) < len(memories),
        }
        
        logger.debug(f"[CONTEXT] Built JSON context: {tokens_used} tokens, {len(items)} memories")
        
        return result
    
    def _build_messages(
        self,
        memories: List[Any],
        query: str,
        include_metadata: bool,
    ) -> List[Dict[str, Any]]:
        """Build chat messages format."""
        available_tokens = self.config.max_tokens - self.config.reserve_tokens
        
        # System message with memories
        memory_texts = []
        tokens_used = 100  # Base message overhead
        
        for memory in memories:
            memory_text = self._format_memory_text(memory, None, include_metadata)
            memory_tokens = self._estimate_tokens(memory_text)
            
            if tokens_used + memory_tokens > available_tokens:
                break
            
            memory_texts.append(memory_text)
            tokens_used += memory_tokens
        
        system_content = (
            "You have access to the following memories from past conversations:\n\n"
            + "\n\n".join(memory_texts)
        )
        
        messages = [
            {"role": "system", "content": system_content},
        ]
        
        logger.debug(f"[CONTEXT] Built messages context: {tokens_used} tokens, {len(memory_texts)} memories")
        
        return messages
    
    def _format_memory_text(
        self,
        memory: Any,
        index: Optional[int],
        include_metadata: bool,
    ) -> str:
        """Format a single memory as text."""
        parts = []
        
        # Index
        if index is not None:
            parts.append(f"[{index}]")
        
        # Timestamp
        if self.config.include_timestamps:
            time_str = self._format_time(memory.timestamp)
            parts.append(f"({time_str})")
        
        # Score
        if self.config.include_scores:
            parts.append(f"[score: {memory.score:.2f}]")
        
        # Content
        content = memory.content
        if isinstance(content, dict):
            # Extract text from dict
            if "text" in content:
                content = content["text"]
            elif "message" in content:
                content = content["message"]
            elif "query" in content and "response" in content:
                content = f"Q: {content['query']}\nA: {content['response']}"
            else:
                content = json.dumps(content)
        
        parts.append(str(content))
        
        # Metadata
        if include_metadata and memory.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in memory.metadata.items() if k not in ["text", "timestamp"])
            if meta_str:
                parts.append(f"[{meta_str}]")
        
        return " ".join(parts)
    
    def _format_memory_json(
        self,
        memory: Any,
        include_metadata: bool,
    ) -> Dict[str, Any]:
        """Format a single memory as JSON."""
        item = {
            "id": memory.event_id,
            "content": memory.content,
            "timestamp": memory.timestamp,
            "score": round(memory.score, 3),
        }
        
        if include_metadata and memory.metadata:
            item["metadata"] = memory.metadata
        
        return item
    
    def _format_time(self, timestamp: float) -> str:
        """Format timestamp based on config."""
        import time as time_module
        from datetime import datetime
        
        now = time_module.time()
        age_seconds = now - timestamp
        
        if self.config.time_format == "absolute":
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        
        elif self.config.time_format == "both":
            absolute = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
            relative = self._relative_time(age_seconds)
            return f"{absolute} ({relative})"
        
        else:  # relative
            return self._relative_time(age_seconds)
    
    def _relative_time(self, seconds: float) -> str:
        """Convert seconds to relative time string."""
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days}d ago"
        elif seconds < 2592000:
            weeks = int(seconds / 604800)
            return f"{weeks}w ago"
        else:
            months = int(seconds / 2592000)
            return f"{months}mo ago"
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // CHARS_PER_TOKEN
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> Optional[str]:
        """Truncate text to fit within token limit."""
        if max_tokens <= 0:
            return None
        
        max_chars = max_tokens * CHARS_PER_TOKEN
        
        if len(text) <= max_chars:
            return text
        
        # Truncate with ellipsis
        return text[:max_chars - 3] + "..."
    
    def estimate_context_size(self, memories: List[Any]) -> Dict[str, int]:
        """
        Estimate context size for memories.
        
        Returns:
            {'total_tokens': int, 'fits': bool, 'would_use': int}
        """
        total_tokens = 0
        
        for memory in memories:
            text = self._format_memory_text(memory, None, False)
            total_tokens += self._estimate_tokens(text)
        
        available = self.config.max_tokens - self.config.reserve_tokens
        
        return {
            "total_tokens": total_tokens,
            "available_tokens": available,
            "fits": total_tokens <= available,
            "memories_count": len(memories),
        }


def build_context(
    memories: List[Any],
    query: str,
    max_tokens: int = 4000,
    format: str = "text",
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Convenience function to build context.
    
    Args:
        memories: List of MemoryItem objects
        query: Query for context
        max_tokens: Token limit
        format: Output format
    
    Returns:
        Formatted context
    """
    builder = ContextBuilder(max_tokens=max_tokens)
    return builder.build(memories, query, format=format)
