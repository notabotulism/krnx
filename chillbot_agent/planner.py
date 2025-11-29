"""
ChillBot Agent - Query Planner

Analyzes incoming queries to determine retrieval strategy.
Supports three modes:
- rules: Fast, deterministic, zero API cost
- llm: Accurate, handles ambiguity, costs per query
- hybrid: Rules first, LLM fallback when uncertain (default)

Intent Types:
- TEMPORAL: Time-based queries ("when did", "last Tuesday")
- ENTITY: Entity-focused ("@user", "#project", "Project Alpha")
- SEMANTIC: General similarity search
- MULTI_HOP: Requires multiple retrieval steps
- FACTUAL: Direct fact lookup

Constitution-safe: Deterministic rules, transparent LLM prompts.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ==============================================
# INTENT TYPES
# ==============================================

class QueryIntent(Enum):
    """Types of query intents."""
    TEMPORAL = "temporal"       # Time-based queries
    ENTITY = "entity"           # Entity-focused queries
    SEMANTIC = "semantic"       # General similarity search
    MULTI_HOP = "multi_hop"     # Requires multiple retrievals
    FACTUAL = "factual"         # Direct fact lookup


# ==============================================
# QUERY PLAN
# ==============================================

@dataclass
class TemporalRange:
    """Time range for temporal queries."""
    start: Optional[float] = None   # Unix timestamp
    end: Optional[float] = None     # Unix timestamp
    description: str = ""           # Human-readable ("last week")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "description": self.description,
        }


@dataclass
class QueryPlan:
    """
    Result of query analysis.
    
    Tells the retrieval router what strategy to use.
    """
    # Primary intent
    intent: QueryIntent = QueryIntent.SEMANTIC
    confidence: float = 0.5
    
    # Query transformations
    expanded_query: Optional[str] = None    # Expanded/rewritten query
    search_terms: List[str] = field(default_factory=list)
    
    # Temporal info
    is_temporal: bool = False
    temporal_range: Optional[TemporalRange] = None
    
    # Entity info
    entities: List[Dict[str, str]] = field(default_factory=list)  # [{type, id}]
    
    # Multi-hop info
    requires_multi_hop: bool = False
    hop_queries: List[str] = field(default_factory=list)
    
    # Retrieval hints
    top_k: int = 10
    use_reranking: bool = True
    filter_stale: bool = True       # Filter superseded events
    
    # Provenance
    planner_mode: str = "hybrid"    # rules, llm, hybrid
    rule_signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "is_temporal": self.is_temporal,
            "temporal_range": self.temporal_range.to_dict() if self.temporal_range else None,
            "entities": self.entities,
            "requires_multi_hop": self.requires_multi_hop,
            "top_k": self.top_k,
            "use_reranking": self.use_reranking,
            "planner_mode": self.planner_mode,
            "rule_signals": self.rule_signals,
        }


# ==============================================
# RULE-BASED PLANNER
# ==============================================

class RuleBasedPlanner:
    """
    Fast, deterministic query classification using patterns.
    
    Signals:
    - Temporal: keywords, date patterns, relative time
    - Entity: @mentions, #tags, quoted terms, patterns
    - Multi-hop: causal language, comparisons, multiple entities
    """
    
    # Temporal patterns
    TEMPORAL_KEYWORDS = {
        "when", "date", "time", "day", "week", "month", "year",
        "yesterday", "today", "tomorrow", "ago", "before", "after",
        "recently", "earlier", "later", "last", "next", "previous",
        "morning", "afternoon", "evening", "night", "session",
    }
    
    TEMPORAL_PATTERNS = [
        # Relative time: "3 days ago", "last week"
        re.compile(r'\b(\d+)\s*(day|week|month|year|hour|minute)s?\s*ago\b', re.I),
        re.compile(r'\blast\s+(day|week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.I),
        re.compile(r'\b(this|next)\s+(week|month|year)\b', re.I),
        
        # Dates: "May 2023", "05/15", "2023-05-15"
        re.compile(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(,?\s*\d{4})?\b', re.I),
        re.compile(r'\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b'),
        re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'),
        
        # Session references
        re.compile(r'\bsession[_\s]*(\d+|one|two|three|four|five)\b', re.I),
        
        # Time of day
        re.compile(r'\b(morning|afternoon|evening|night)\s+(of|on)\b', re.I),
    ]
    
    # Entity patterns
    ENTITY_PATTERNS = [
        (re.compile(r'@([a-zA-Z0-9_-]+)'), "user"),
        (re.compile(r'#([a-zA-Z][a-zA-Z0-9_-]*)'), "concept"),
        (re.compile(r'"([^"]+)"'), "quoted"),
        (re.compile(r"'([^']+)'"), "quoted"),
        (re.compile(r'\bproject[:\s]+([a-zA-Z][a-zA-Z0-9_-]*)', re.I), "project"),
        (re.compile(r'\buser[:\s]+([a-zA-Z][a-zA-Z0-9_-]*)', re.I), "user"),
    ]
    
    # Multi-hop signals
    MULTI_HOP_PATTERNS = [
        re.compile(r'\b(how|why)\s+did\s+\w+\s+(change|affect|impact|lead|cause)\b', re.I),
        re.compile(r'\bcompare\s+.+\s+(to|with|and)\s+', re.I),
        re.compile(r'\bwhat\s+(changed|happened)\s+(since|after|before)\b', re.I),
        re.compile(r'\b(because|therefore|thus|hence|consequently)\b', re.I),
        re.compile(r'\brelationship\s+between\b', re.I),
        re.compile(r'\bdifference\s+between\b', re.I),
    ]
    
    # Factual question patterns
    FACTUAL_PATTERNS = [
        re.compile(r'^(who|what|where|which)\s+(is|was|are|were)\b', re.I),
        re.compile(r'^(how\s+many|how\s+much)\b', re.I),
        re.compile(r'\bname\s+of\b', re.I),
    ]
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize rule-based planner.
        
        Args:
            confidence_threshold: Minimum confidence to return without LLM fallback
        """
        self.confidence_threshold = confidence_threshold
    
    def plan(self, query: str) -> QueryPlan:
        """
        Analyze query using rules.
        
        Args:
            query: The question/query string
        
        Returns:
            QueryPlan with classification and extraction
        """
        query_lower = query.lower()
        signals = []
        
        # Score each intent type
        temporal_score = self._score_temporal(query, query_lower, signals)
        entity_score, entities = self._score_entity(query, signals)
        multi_hop_score = self._score_multi_hop(query, query_lower, signals)
        factual_score = self._score_factual(query, query_lower, signals)
        
        # Determine primary intent
        scores = {
            QueryIntent.TEMPORAL: temporal_score,
            QueryIntent.ENTITY: entity_score,
            QueryIntent.MULTI_HOP: multi_hop_score,
            QueryIntent.FACTUAL: factual_score,
            QueryIntent.SEMANTIC: 0.3,  # Default fallback
        }
        
        # Pick highest scoring intent
        intent = max(scores, key=scores.get)
        confidence = scores[intent]
        
        # If no strong signal, default to semantic
        if confidence < 0.4:
            intent = QueryIntent.SEMANTIC
            confidence = 0.5
        
        # Build plan
        plan = QueryPlan(
            intent=intent,
            confidence=confidence,
            is_temporal=(temporal_score >= 0.5),
            entities=entities,
            requires_multi_hop=(multi_hop_score >= 0.5),
            planner_mode="rules",
            rule_signals=signals,
        )
        
        # Extract temporal range if temporal
        if plan.is_temporal:
            plan.temporal_range = self._extract_temporal_range(query, query_lower)
        
        # Set top_k based on intent
        if intent == QueryIntent.MULTI_HOP:
            plan.top_k = 20  # Need more candidates for multi-hop
        elif intent == QueryIntent.TEMPORAL:
            plan.top_k = 15  # Temporal queries might span time
        
        return plan
    
    def _score_temporal(self, query: str, query_lower: str, signals: List[str]) -> float:
        """Score temporal intent."""
        score = 0.0
        
        # Keyword matching
        words = set(re.findall(r'\b\w+\b', query_lower))
        temporal_matches = words & self.TEMPORAL_KEYWORDS
        if temporal_matches:
            score += 0.3 * min(len(temporal_matches), 3) / 3
            signals.append(f"temporal_keywords: {temporal_matches}")
        
        # Pattern matching
        for pattern in self.TEMPORAL_PATTERNS:
            if pattern.search(query):
                score += 0.4
                signals.append(f"temporal_pattern: {pattern.pattern[:30]}...")
                break
        
        return min(score, 1.0)
    
    def _score_entity(self, query: str, signals: List[str]) -> Tuple[float, List[Dict]]:
        """Score entity intent and extract entities."""
        score = 0.0
        entities = []
        
        for pattern, entity_type in self.ENTITY_PATTERNS:
            for match in pattern.finditer(query):
                entity_id = match.group(1)
                entities.append({"type": entity_type, "id": entity_id})
                score += 0.3
                signals.append(f"entity_{entity_type}: {entity_id}")
        
        return min(score, 1.0), entities
    
    def _score_multi_hop(self, query: str, query_lower: str, signals: List[str]) -> float:
        """Score multi-hop intent."""
        score = 0.0
        
        for pattern in self.MULTI_HOP_PATTERNS:
            if pattern.search(query):
                score += 0.4
                signals.append(f"multi_hop_pattern: {pattern.pattern[:30]}...")
        
        # Multiple question marks suggest multi-part question
        if query.count('?') > 1:
            score += 0.2
            signals.append("multiple_questions")
        
        return min(score, 1.0)
    
    def _score_factual(self, query: str, query_lower: str, signals: List[str]) -> float:
        """Score factual query intent."""
        score = 0.0
        
        for pattern in self.FACTUAL_PATTERNS:
            if pattern.search(query):
                score += 0.5
                signals.append(f"factual_pattern: {pattern.pattern[:30]}...")
                break
        
        return min(score, 1.0)
    
    def _extract_temporal_range(self, query: str, query_lower: str) -> Optional[TemporalRange]:
        """Extract temporal range from query."""
        now = datetime.now()
        
        # "X days/weeks ago"
        ago_match = re.search(r'(\d+)\s*(day|week|month|year|hour)s?\s*ago', query_lower)
        if ago_match:
            amount = int(ago_match.group(1))
            unit = ago_match.group(2)
            
            if unit == "hour":
                delta = timedelta(hours=amount)
            elif unit == "day":
                delta = timedelta(days=amount)
            elif unit == "week":
                delta = timedelta(weeks=amount)
            elif unit == "month":
                delta = timedelta(days=amount * 30)
            elif unit == "year":
                delta = timedelta(days=amount * 365)
            else:
                delta = timedelta(days=1)
            
            target = now - delta
            # Give a window around the target
            return TemporalRange(
                start=(target - timedelta(hours=12)).timestamp(),
                end=(target + timedelta(hours=12)).timestamp(),
                description=f"{amount} {unit}(s) ago",
            )
        
        # "last week/month"
        last_match = re.search(r'last\s+(day|week|month|year)', query_lower)
        if last_match:
            unit = last_match.group(1)
            if unit == "day":
                start = now - timedelta(days=1)
                end = now
            elif unit == "week":
                start = now - timedelta(weeks=1)
                end = now
            elif unit == "month":
                start = now - timedelta(days=30)
                end = now
            elif unit == "year":
                start = now - timedelta(days=365)
                end = now
            else:
                start = now - timedelta(days=7)
                end = now
            
            return TemporalRange(
                start=start.timestamp(),
                end=end.timestamp(),
                description=f"last {unit}",
            )
        
        # Session references (common in LOCOMO)
        session_match = re.search(r'session[_\s]*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)', query_lower)
        if session_match:
            return TemporalRange(
                description=f"session {session_match.group(1)}",
            )
        
        return None


# ==============================================
# LLM-BASED PLANNER
# ==============================================

class LLMPlanner:
    """
    Accurate query classification using LLM.
    
    Handles ambiguous queries and complex intent detection.
    """
    
    SYSTEM_PROMPT = """You are a query analyzer for a memory retrieval system.
Analyze the query and return JSON with:
{
  "intent": "temporal" | "entity" | "semantic" | "multi_hop" | "factual",
  "confidence": 0.0-1.0,
  "is_temporal": true/false,
  "temporal_description": "string if temporal, else null",
  "entities": [{"type": "user|project|concept", "id": "name"}],
  "requires_multi_hop": true/false,
  "reasoning": "brief explanation"
}

Intent definitions:
- temporal: Query about WHEN something happened, time-based filtering needed
- entity: Query focuses on specific person, project, or concept
- semantic: General similarity-based search will work
- multi_hop: Needs multiple retrieval steps (comparisons, causality, changes over time)
- factual: Direct fact lookup (who/what/where/which)

Return ONLY valid JSON, no markdown."""

    def __init__(self, llm_client=None):
        """
        Initialize LLM planner.
        
        Args:
            llm_client: LLM client with complete(prompt, system) method
        """
        self.llm = llm_client
    
    def plan(self, query: str) -> QueryPlan:
        """
        Analyze query using LLM.
        
        Args:
            query: The question/query string
        
        Returns:
            QueryPlan with classification
        """
        if self.llm is None:
            logger.warning("[PLANNER] No LLM client, falling back to semantic")
            return QueryPlan(
                intent=QueryIntent.SEMANTIC,
                confidence=0.5,
                planner_mode="llm",
            )
        
        try:
            response = self.llm.complete(
                prompt=f"Analyze this query:\n\n{query}",
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=500,
            )
            
            # Parse JSON response
            result = self._parse_response(response)
            
            # Build plan
            intent = QueryIntent(result.get("intent", "semantic"))
            
            plan = QueryPlan(
                intent=intent,
                confidence=result.get("confidence", 0.7),
                is_temporal=result.get("is_temporal", False),
                entities=result.get("entities", []),
                requires_multi_hop=result.get("requires_multi_hop", False),
                planner_mode="llm",
                rule_signals=[f"llm_reasoning: {result.get('reasoning', '')}"],
            )
            
            # Extract temporal description
            if plan.is_temporal and result.get("temporal_description"):
                plan.temporal_range = TemporalRange(
                    description=result["temporal_description"]
                )
            
            return plan
            
        except Exception as e:
            logger.error(f"[PLANNER] LLM planning failed: {e}")
            return QueryPlan(
                intent=QueryIntent.SEMANTIC,
                confidence=0.5,
                planner_mode="llm",
                rule_signals=[f"llm_error: {str(e)}"],
            )
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in response
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


# ==============================================
# HYBRID PLANNER
# ==============================================

class QueryPlanner:
    """
    Unified query planner with three modes.
    
    - rules: Fast, deterministic
    - llm: Accurate, handles ambiguity
    - hybrid: Rules first, LLM fallback (default)
    """
    
    def __init__(
        self,
        llm_client=None,
        mode: str = "hybrid",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize query planner.
        
        Args:
            llm_client: LLM client for llm/hybrid modes
            mode: "rules", "llm", or "hybrid"
            confidence_threshold: Confidence needed to skip LLM in hybrid mode
        """
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        
        self._rule_planner = RuleBasedPlanner(confidence_threshold)
        self._llm_planner = LLMPlanner(llm_client)
        
        logger.info(f"[PLANNER] Initialized in {mode} mode (threshold={confidence_threshold})")
    
    def plan(self, query: str) -> QueryPlan:
        """
        Analyze query and produce retrieval plan.
        
        Args:
            query: The question/query string
        
        Returns:
            QueryPlan with classification and strategy
        """
        if self.mode == "rules":
            plan = self._rule_planner.plan(query)
            plan.planner_mode = "rules"
            return plan
        
        elif self.mode == "llm":
            plan = self._llm_planner.plan(query)
            plan.planner_mode = "llm"
            return plan
        
        else:  # hybrid
            # Try rules first
            rule_plan = self._rule_planner.plan(query)
            
            # If confident enough, use rule result
            if rule_plan.confidence >= self.confidence_threshold:
                rule_plan.planner_mode = "hybrid:rules"
                logger.debug(f"[PLANNER] Rules confident ({rule_plan.confidence:.2f}), skipping LLM")
                return rule_plan
            
            # Otherwise, use LLM
            logger.debug(f"[PLANNER] Rules uncertain ({rule_plan.confidence:.2f}), using LLM")
            llm_plan = self._llm_planner.plan(query)
            
            # Merge: keep rule entities if LLM missed them
            if rule_plan.entities and not llm_plan.entities:
                llm_plan.entities = rule_plan.entities
            
            # Merge: keep temporal range from rules if more specific
            if rule_plan.temporal_range and not llm_plan.temporal_range:
                llm_plan.temporal_range = rule_plan.temporal_range
            
            llm_plan.planner_mode = "hybrid:llm"
            llm_plan.rule_signals = rule_plan.rule_signals + llm_plan.rule_signals
            
            return llm_plan
    
    def set_mode(self, mode: str):
        """Change planner mode."""
        if mode not in ("rules", "llm", "hybrid"):
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        logger.info(f"[PLANNER] Mode changed to {mode}")


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    # Types
    'QueryIntent',
    'TemporalRange',
    'QueryPlan',
    
    # Planners
    'RuleBasedPlanner',
    'LLMPlanner',
    'QueryPlanner',
]
