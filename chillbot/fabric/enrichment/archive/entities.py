"""
KRNX Fabric - Entity Extractor

Extracts entities referenced in event content.
This is how "metadata becomes graph edges" without a graph DB.

Entity types:
- user: @mentions, user IDs
- project: Project references
- file: File paths, URLs
- code: Function names, classes
- concept: Tagged concepts (#topic)
- custom: User-defined patterns

Query pattern:
    "Find all events about Project Alpha"
    → Filter by entities.type="project" AND entities.id="project_alpha"

Constitution-safe: Pure, deterministic, no side effects.
No LLM calls - pure regex/pattern matching for v1.
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Pattern

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of extractable entities."""
    USER = "user"           # @mentions, user:xxx
    PROJECT = "project"     # project:xxx, Project Xxx
    FILE = "file"           # /path/to/file, file.ext
    URL = "url"             # https://...
    CODE = "code"           # function(), ClassName
    CONCEPT = "concept"     # #hashtag
    EVENT = "event"         # evt_xxx references
    WORKSPACE = "workspace" # workspace:xxx
    CUSTOM = "custom"       # User-defined


@dataclass
class Entity:
    """An extracted entity."""
    type: EntityType
    id: str                         # Normalized identifier
    raw: str                        # Original text match
    confidence: float = 1.0         # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type.value,
            "id": self.id,
        }
        if self.raw != self.id:
            result["raw"] = self.raw
        if self.confidence < 1.0:
            result["confidence"] = self.confidence
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        return cls(
            type=EntityType(data["type"]),
            id=data["id"],
            raw=data.get("raw", data["id"]),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata"),
        )


# ==============================================
# DEFAULT PATTERNS
# ==============================================

DEFAULT_PATTERNS: Dict[EntityType, List[Pattern]] = {
    EntityType.USER: [
        re.compile(r'@([a-zA-Z0-9_-]+)'),           # @username
        re.compile(r'user:([a-zA-Z0-9_-]+)'),       # user:xxx
        re.compile(r'user_id[=:\s]+([a-zA-Z0-9_-]+)', re.I),  # user_id = xxx
    ],
    EntityType.PROJECT: [
        re.compile(r'project:([a-zA-Z0-9_-]+)'),    # project:xxx
        re.compile(r'Project\s+([A-Z][a-zA-Z0-9_-]*)'),  # Project Alpha
        re.compile(r'proj[_-]([a-zA-Z0-9_-]+)'),    # proj_xxx, proj-xxx
    ],
    EntityType.FILE: [
        re.compile(r'(/[a-zA-Z0-9_.-]+)+'),         # /path/to/file
        re.compile(r'[a-zA-Z0-9_-]+\.(py|js|ts|go|rs|md|txt|json|yaml|yml|toml)'),  # file.ext
    ],
    EntityType.URL: [
        re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),  # URLs
    ],
    EntityType.CODE: [
        re.compile(r'\b([A-Z][a-zA-Z0-9]*(?:Class|Service|Manager|Handler|Controller|Factory|Builder))\b'),  # ClassNames
        re.compile(r'\b([a-z_][a-z0-9_]*)\s*\('),   # function_calls(
        re.compile(r'def\s+([a-z_][a-z0-9_]*)'),    # def function_name
        re.compile(r'class\s+([A-Z][a-zA-Z0-9]*)'), # class ClassName
        re.compile(r'fn\s+([a-z_][a-z0-9_]*)'),     # fn rust_function
    ],
    EntityType.CONCEPT: [
        re.compile(r'#([a-zA-Z][a-zA-Z0-9_-]*)'),   # #hashtag
        re.compile(r'\[([a-zA-Z][a-zA-Z0-9_-]*)\]'), # [tag]
    ],
    EntityType.EVENT: [
        re.compile(r'evt_([a-zA-Z0-9]+)'),          # evt_xxx
        re.compile(r'event:([a-zA-Z0-9_-]+)'),      # event:xxx
    ],
    EntityType.WORKSPACE: [
        re.compile(r'workspace:([a-zA-Z0-9_-]+)'),  # workspace:xxx
        re.compile(r'ws:([a-zA-Z0-9_-]+)'),         # ws:xxx
    ],
}


class EntityExtractor:
    """
    Extracts entities from event content.
    
    Pure regex-based extraction (no LLM).
    Fast, deterministic, predictable.
    """
    
    def __init__(
        self,
        custom_patterns: Optional[Dict[str, str]] = None,
        enabled_types: Optional[List[EntityType]] = None,
    ):
        """
        Initialize entity extractor.
        
        Args:
            custom_patterns: Custom regex patterns {name: pattern_string}
            enabled_types: Which entity types to extract (None = all)
        """
        self.patterns = dict(DEFAULT_PATTERNS)
        self.enabled_types = enabled_types
        
        # Add custom patterns
        if custom_patterns:
            for name, pattern in custom_patterns.items():
                try:
                    compiled = re.compile(pattern)
                    if EntityType.CUSTOM not in self.patterns:
                        self.patterns[EntityType.CUSTOM] = []
                    self.patterns[EntityType.CUSTOM].append(compiled)
                except re.error as e:
                    logger.warning(f"Invalid custom pattern '{name}': {e}")
    
    def extract(
        self,
        content: Any,
    ) -> List[Entity]:
        """
        Extract entities from content.
        
        Args:
            content: Event content (str or dict)
        
        Returns:
            List of extracted Entity objects
        """
        # Get text from content
        text = self._get_text(content)
        if not text:
            return []
        
        entities = []
        seen = set()  # Dedupe by (type, id)
        
        for entity_type, patterns in self.patterns.items():
            # Skip disabled types
            if self.enabled_types and entity_type not in self.enabled_types:
                continue
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get the captured group (or full match)
                    raw = match.group(1) if match.lastindex else match.group(0)
                    
                    # Normalize ID
                    entity_id = self._normalize_id(raw, entity_type)
                    
                    # Dedupe
                    key = (entity_type, entity_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    entities.append(Entity(
                        type=entity_type,
                        id=entity_id,
                        raw=raw,
                        confidence=1.0,
                    ))
        
        return entities
    
    def extract_typed(
        self,
        content: Any,
        entity_type: EntityType,
    ) -> List[Entity]:
        """
        Extract entities of a specific type.
        
        Args:
            content: Event content
            entity_type: Type to extract
        
        Returns:
            List of entities of that type
        """
        all_entities = self.extract(content)
        return [e for e in all_entities if e.type == entity_type]
    
    def extract_users(self, content: Any) -> List[str]:
        """Extract user IDs/mentions."""
        entities = self.extract_typed(content, EntityType.USER)
        return [e.id for e in entities]
    
    def extract_projects(self, content: Any) -> List[str]:
        """Extract project references."""
        entities = self.extract_typed(content, EntityType.PROJECT)
        return [e.id for e in entities]
    
    def extract_concepts(self, content: Any) -> List[str]:
        """Extract concept tags (#hashtags)."""
        entities = self.extract_typed(content, EntityType.CONCEPT)
        return [e.id for e in entities]
    
    def _get_text(self, content: Any) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        
        if isinstance(content, dict):
            # Try common text fields
            for field in ["text", "message", "content", "body", "query", "response"]:
                if field in content and isinstance(content[field], str):
                    return content[field]
            
            # Concatenate multiple text fields
            parts = []
            for field in ["query", "response", "text", "message"]:
                if field in content and isinstance(content[field], str):
                    parts.append(content[field])
            if parts:
                return " ".join(parts)
            
            # Last resort: stringify
            try:
                return str(content)
            except Exception:
                return ""
        
        # Unknown type
        try:
            return str(content)
        except Exception:
            return ""
    
    def _normalize_id(self, raw: str, entity_type: EntityType) -> str:
        """
        Normalize entity ID for consistent storage/lookup.
        
        Examples:
        - "Project Alpha" → "project_alpha"
        - "@UserName" → "username"
        - "/path/to/file.py" → "/path/to/file.py" (paths preserved)
        """
        if entity_type == EntityType.FILE:
            return raw  # Preserve file paths as-is
        
        if entity_type == EntityType.URL:
            return raw  # Preserve URLs as-is
        
        if entity_type == EntityType.CODE:
            return raw  # Preserve code identifiers as-is
        
        # For other types: lowercase and normalize
        normalized = raw.lower()
        normalized = re.sub(r'[^a-z0-9_-]', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')
        
        return normalized
    
    def add_pattern(
        self,
        entity_type: EntityType,
        pattern: str,
    ):
        """
        Add a custom extraction pattern.
        
        Args:
            entity_type: Type to associate with matches
            pattern: Regex pattern (should have one capture group)
        """
        try:
            compiled = re.compile(pattern)
            if entity_type not in self.patterns:
                self.patterns[entity_type] = []
            self.patterns[entity_type].append(compiled)
        except re.error as e:
            raise ValueError(f"Invalid pattern: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            "enabled_types": (
                [t.value for t in self.enabled_types]
                if self.enabled_types
                else "all"
            ),
            "pattern_counts": {
                t.value: len(patterns)
                for t, patterns in self.patterns.items()
            },
        }
    
    def __repr__(self) -> str:
        types = self.enabled_types or list(EntityType)
        return f"EntityExtractor(types={len(types)})"
