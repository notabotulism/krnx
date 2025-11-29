"""
KRNX API Schemas

Pydantic models for request/response validation.
Follows the playbook specification for consistent API structure.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import time
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class EventType(str, Enum):
    """Event type classification."""
    MESSAGE = "message"
    DECISION = "decision"
    OBSERVATION = "observation"
    ACTION = "action"
    SYSTEM = "system"
    FINDING = "finding"
    TASK = "task"


class RetentionClass(str, Enum):
    """Retention class for events (Constitution 6.3)."""
    EPHEMERAL = "ephemeral"
    STANDARD = "standard"
    PERMANENT = "permanent"


class RelationKind(str, Enum):
    """Types of event relations."""
    REPLIES_TO = "replies_to"
    SUPERSEDES = "supersedes"
    SUPERSEDED_BY = "superseded_by"
    REFERENCES = "references"
    CAUSED_BY = "caused_by"


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# =============================================================================
# RESPONSE ENVELOPE (consistent structure per playbook 1.1.3)
# =============================================================================

class ResponseMeta(BaseModel):
    """Metadata for all API responses."""
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    timestamp: float = Field(default_factory=time.time)
    duration_ms: Optional[float] = None
    workspace_id: Optional[str] = None
    as_of: Optional[float] = None


class Pagination(BaseModel):
    """Pagination info for list responses."""
    cursor: Optional[str] = None
    has_more: bool = False
    total: Optional[int] = None


class APIResponse(BaseModel):
    """Standard API response envelope."""
    data: Any
    meta: ResponseMeta = Field(default_factory=ResponseMeta)
    pagination: Optional[Pagination] = None


# =============================================================================
# ERROR RESPONSE (RFC 7807 per playbook 1.1.4)
# =============================================================================

class ErrorDetail(BaseModel):
    """RFC 7807 Problem Details."""
    type: str = "https://krnx.dev/errors/internal"
    code: str = "INTERNAL_ERROR"
    title: str = "Internal Error"
    detail: str = "An unexpected error occurred"
    instance: Optional[str] = None
    status: int = 500


class ErrorResponse(BaseModel):
    """Error response envelope."""
    error: ErrorDetail
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class EventRelation(BaseModel):
    """A relation between events."""
    kind: RelationKind
    target: str  # Event ID
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class EventProvenance(BaseModel):
    """Provenance information for an event."""
    hash: Optional[str] = None
    previous_hash: Optional[str] = None


class EventEnrichment(BaseModel):
    """Enrichment metadata for an event."""
    salience_score: Optional[float] = None
    episode_id: Optional[str] = None
    time_gap_seconds: Optional[float] = None
    entities: Optional[List[str]] = None
    topics: Optional[List[str]] = None
    status: Optional[str] = None


class EventBase(BaseModel):
    """Base event fields."""
    content: Union[str, Dict[str, Any]]
    user_id: Optional[str] = "default"
    session_id: Optional[str] = None
    channel: Optional[str] = None
    event_type: EventType = EventType.MESSAGE
    ttl_seconds: Optional[int] = None
    retention_class: Optional[RetentionClass] = None
    metadata: Optional[Dict[str, Any]] = None


class EventCreate(EventBase):
    """Request model for creating an event."""
    parent_event_id: Optional[str] = None
    target_agents: Optional[List[str]] = None
    branch: Optional[str] = None


class EventBatchCreate(BaseModel):
    """Request model for batch event creation."""
    events: List[EventCreate]
    branch: Optional[str] = None


class EventResponse(BaseModel):
    """Full event response."""
    event_id: str
    workspace_id: str
    user_id: str
    session_id: Optional[str] = None
    content: Union[str, Dict[str, Any]]
    channel: Optional[str] = None
    event_type: str = "message"
    timestamp: float
    created_at: float
    provenance: Optional[EventProvenance] = None
    relations: Optional[List[EventRelation]] = None
    enrichment: Optional[EventEnrichment] = None
    metadata: Optional[Dict[str, Any]] = None
    # Query-time additions
    score: Optional[float] = None  # Similarity score from recall


class EventListResponse(BaseModel):
    """Response for event list queries."""
    events: List[EventResponse]
    count: int


# =============================================================================
# TEMPORAL SCHEMAS
# =============================================================================

class StateResponse(BaseModel):
    """Workspace state at a point in time."""
    workspace_id: str
    as_of: Optional[float] = None
    event_count: int
    latest_event: Optional[EventResponse] = None
    earliest_event: Optional[EventResponse] = None
    users: List[str] = []
    channels: List[str] = []


class ReplayRequest(BaseModel):
    """Request for replaying events."""
    start: float
    end: float
    user_id: Optional[str] = None
    channel: Optional[str] = None
    branch: Optional[str] = None


class ReplayResponse(BaseModel):
    """Response for replay queries."""
    events: List[EventResponse]
    start: float
    end: float
    count: int
    duration_ms: float


class TimelineBucket(BaseModel):
    """A single bucket in timeline distribution."""
    timestamp: float
    count: int
    channels: Optional[Dict[str, int]] = None


class TimelineResponse(BaseModel):
    """Event distribution over time."""
    buckets: List[TimelineBucket]
    bucket_size: str  # minute, hour, day, week
    start: float
    end: float
    total_events: int


# =============================================================================
# PROVENANCE SCHEMAS
# =============================================================================

class ProvenanceChainResponse(BaseModel):
    """Provenance chain from event to root."""
    event_id: str
    chain: List[EventResponse]
    depth: int
    verified: bool
    complete: bool
    gaps: List[str] = []


class VerificationResponse(BaseModel):
    """Result of verifying event integrity."""
    event_id: str
    verified: bool
    event_hash_valid: bool
    chain_valid: bool
    chain_complete: bool
    computed_hash: Optional[str] = None
    expected_hash: Optional[str] = None
    issues: List[Dict[str, Any]] = []
    verification_time_ms: float


class AncestryResponse(BaseModel):
    """Ancestors/children of an event."""
    event_id: str
    events: List[EventResponse]
    depth: int


# =============================================================================
# SUPERSESSION SCHEMAS
# =============================================================================

class VersionChainResponse(BaseModel):
    """All versions of a fact."""
    versions: List[EventResponse]
    current: Optional[EventResponse] = None
    queried_event_id: str
    queried_position: int
    total_versions: int


class SupersessionResponse(BaseModel):
    """What an event supersedes or is superseded by."""
    event_id: str
    supersedes: Optional[EventResponse] = None
    superseded_by: Optional[EventResponse] = None
    confidence: Optional[float] = None
    signals: List[str] = []


# =============================================================================
# CONTEXT SCHEMAS
# =============================================================================

class ContextRequest(BaseModel):
    """Request for building LLM context."""
    query: str
    max_tokens: int = 4000
    format: str = "text"  # text, json, messages
    user_id: Optional[str] = None
    channel: Optional[str] = None
    as_of: Optional[float] = None
    include_metadata: bool = False
    branch: Optional[str] = None


class ContextResponse(BaseModel):
    """LLM-ready context."""
    context: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    token_count: int
    event_count: int
    format: str
    query: str


class RecallRequest(BaseModel):
    """Request for semantic search."""
    query: str
    top_k: int = 10
    user_id: Optional[str] = None
    channel: Optional[str] = None
    as_of: Optional[float] = None
    min_score: float = 0.0
    include_superseded: bool = False
    branch: Optional[str] = None


class RecallResponse(BaseModel):
    """Semantic search results."""
    memories: List[EventResponse]
    query: str
    count: int
    latency_ms: float


# =============================================================================
# AGENT SCHEMAS
# =============================================================================

class AgentCreate(BaseModel):
    """Request for registering an agent."""
    agent_id: str
    agent_type: str = "generic"
    metadata: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    """Agent registration info."""
    agent_id: str
    workspace_id: str
    agent_type: str
    registered_at: float
    last_seen: Optional[float] = None
    events_processed: int = 0
    metadata: Optional[Dict[str, Any]] = None


class AgentListResponse(BaseModel):
    """List of agents."""
    agents: List[AgentResponse]
    count: int


class PublishRequest(BaseModel):
    """Request to publish an event to workspace stream."""
    event_type: str
    content: Union[str, Dict[str, Any]]
    agent_id: str
    target_agents: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConsumeRequest(BaseModel):
    """Request to consume events from stream."""
    agent_id: str
    agent_type: str
    count: int = 10
    block_ms: int = 1000


class ConsumeResponse(BaseModel):
    """Events consumed from stream."""
    events: List[Dict[str, Any]]  # Includes message_id for acking
    count: int


class AckRequest(BaseModel):
    """Request to acknowledge event processing."""
    agent_id: str
    agent_type: str
    message_ids: List[str]


# =============================================================================
# BRANCH SCHEMAS
# =============================================================================

class BranchCreate(BaseModel):
    """Request to create a branch."""
    branch_id: str
    parent_branch: str = "main"
    fork_event_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BranchResponse(BaseModel):
    """Branch information."""
    branch_id: str
    workspace_id: str
    parent_branch: str
    fork_event_id: Optional[str] = None
    fork_timestamp: float
    head_event_id: Optional[str] = None
    event_count: int = 0
    created_at: float
    updated_at: float
    archived: bool = False
    metadata: Optional[Dict[str, Any]] = None


class BranchListResponse(BaseModel):
    """List of branches."""
    branches: List[BranchResponse]
    count: int


class MergeRequest(BaseModel):
    """Request to merge a branch."""
    target_branch: str = "main"
    strategy: str = "append"  # append, rebase
    archive_source: bool = True


class MergeResponse(BaseModel):
    """Result of merging branches."""
    source_branch: str
    target_branch: str
    events_merged: int
    conflicts: List[Dict[str, Any]] = []
    source_archived: bool


class BranchCompareRequest(BaseModel):
    """Request to compare branches."""
    branches: List[str]


class BranchCompareResponse(BaseModel):
    """Comparison of multiple branches."""
    branches: List[str]
    fork_point: Optional[EventResponse] = None
    common_events: int
    divergent_events: Dict[str, int]  # branch_id -> unique event count


# =============================================================================
# OPERATIONS SCHEMAS
# =============================================================================

class ComponentHealth(BaseModel):
    """Health of a single component."""
    status: HealthStatus
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """System health status."""
    status: HealthStatus
    version: str = "0.3.10"
    components: Dict[str, ComponentHealth]
    uptime_seconds: float


class WorkspaceStats(BaseModel):
    """Workspace statistics."""
    workspace_id: str
    total_events: int
    stm_events: int
    ltm_warm_events: int
    ltm_cold_events: int
    users: int
    channels: int
    storage_bytes: int
    oldest_event: Optional[float] = None
    newest_event: Optional[float] = None


class WorkerMetricsResponse(BaseModel):
    """Worker health metrics."""
    queue_depth: int
    lag_seconds: float
    messages_processed: int
    errors_last_hour: int
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None
    worker_running: bool
    healthy: bool


class CompactRequest(BaseModel):
    """Request to trigger compaction."""
    force: bool = False
    archive_days: int = 30


class CompactResponse(BaseModel):
    """Result of compaction."""
    events_archived: int
    events_deleted: int
    space_reclaimed_bytes: int
    duration_ms: float


class EnrichmentStatusResponse(BaseModel):
    """Enrichment status for an event."""
    event_id: str
    status: str  # pending, complete, failed
    enrichment: Optional[EventEnrichment] = None
    error: Optional[str] = None
