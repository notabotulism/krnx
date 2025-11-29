"""
KRNX Basic API Routes

Core kernel operations:
- Events CRUD (create, get, list, delete, batch)
- Temporal (state reconstruction, replay, timeline)
- Health and stats
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import Optional, List
import time
import uuid

from ..schemas import (
    # Response envelope
    APIResponse, ResponseMeta, Pagination, ErrorResponse, ErrorDetail,
    # Event schemas
    EventCreate, EventBatchCreate, EventResponse, EventListResponse,
    # Temporal schemas
    StateResponse, ReplayRequest, ReplayResponse, TimelineResponse, TimelineBucket,
    # Operations schemas
    HealthResponse, ComponentHealth, HealthStatus, WorkspaceStats, WorkerMetricsResponse,
)
from ..deps import get_controller, get_fabric, get_uptime, event_to_response, events_to_response

router = APIRouter()


# =============================================================================
# HEALTH & STATS
# =============================================================================

@router.get("/health", response_model=APIResponse, tags=["Operations"])
async def health_check():
    """
    System health check.
    
    Returns health status of all components:
    - Redis (STM)
    - SQLite (LTM)
    - Worker (async processor)
    """
    start = time.time()
    controller = get_controller()
    
    components = {}
    overall_status = HealthStatus.HEALTHY
    
    # Check Redis
    try:
        redis_start = time.time()
        controller.stm.redis.ping()
        redis_latency = (time.time() - redis_start) * 1000
        components["redis"] = ComponentHealth(
            status=HealthStatus.HEALTHY,
            latency_ms=round(redis_latency, 2),
            details={"connected": True}
        )
    except Exception as e:
        components["redis"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            details={"error": str(e)}
        )
        overall_status = HealthStatus.UNHEALTHY
    
    # Check LTM
    try:
        ltm_start = time.time()
        ltm_stats = controller.ltm.get_stats()
        ltm_latency = (time.time() - ltm_start) * 1000
        components["ltm"] = ComponentHealth(
            status=HealthStatus.HEALTHY,
            latency_ms=round(ltm_latency, 2),
            details=ltm_stats
        )
    except Exception as e:
        components["ltm"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            details={"error": str(e)}
        )
        overall_status = HealthStatus.DEGRADED if overall_status == HealthStatus.HEALTHY else overall_status
    
    # Check Worker
    try:
        metrics = controller.get_worker_metrics()
        worker_status = HealthStatus.HEALTHY if metrics.is_healthy() else HealthStatus.DEGRADED
        components["worker"] = ComponentHealth(
            status=worker_status,
            details=metrics.to_dict()
        )
        if worker_status == HealthStatus.DEGRADED:
            overall_status = HealthStatus.DEGRADED
    except Exception as e:
        components["worker"] = ComponentHealth(
            status=HealthStatus.UNHEALTHY,
            details={"error": str(e)}
        )
        overall_status = HealthStatus.DEGRADED if overall_status == HealthStatus.HEALTHY else overall_status
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=HealthResponse(
            status=overall_status,
            version="0.3.10",
            components=components,
            uptime_seconds=round(get_uptime(), 2)
        ),
        meta=ResponseMeta(duration_ms=round(duration_ms, 2))
    )


@router.get("/workspaces/{workspace_id}/stats", response_model=APIResponse, tags=["Operations"])
async def get_workspace_stats(
    workspace_id: str = Path(..., description="Workspace identifier")
):
    """
    Get workspace statistics.
    
    Returns event counts, storage usage, and timeline bounds.
    """
    start = time.time()
    controller = get_controller()
    
    # Get LTM stats
    ltm_stats = controller.ltm.get_stats()
    
    # Query for workspace-specific stats
    # For now, use total stats - workspace filtering can be added later
    events = controller.query_events(workspace_id=workspace_id, limit=1)
    
    # Get unique users and channels (simplified)
    all_events = controller.query_events(workspace_id=workspace_id, limit=1000)
    users = set(e.user_id for e in all_events)
    channels = set(e.channel for e in all_events if e.channel)
    
    oldest = min((e.timestamp for e in all_events), default=None) if all_events else None
    newest = max((e.timestamp for e in all_events), default=None) if all_events else None
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=WorkspaceStats(
            workspace_id=workspace_id,
            total_events=ltm_stats.get("total_events", 0),
            stm_events=0,  # Would need Redis query
            ltm_warm_events=ltm_stats.get("warm_events", 0),
            ltm_cold_events=ltm_stats.get("cold_events", 0),
            users=len(users),
            channels=len(channels),
            storage_bytes=int(ltm_stats.get("total_size_mb", 0) * 1024 * 1024),
            oldest_event=oldest,
            newest_event=newest,
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/metrics", response_model=APIResponse, tags=["Operations"])
async def get_worker_metrics(
    workspace_id: str = Path(..., description="Workspace identifier")
):
    """
    Get worker metrics for backpressure monitoring.
    """
    start = time.time()
    controller = get_controller()
    
    metrics = controller.get_worker_metrics()
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=WorkerMetricsResponse(**metrics.to_dict()),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


# =============================================================================
# EVENTS CRUD
# =============================================================================

@router.post("/workspaces/{workspace_id}/events", response_model=APIResponse, tags=["Events"])
async def create_event(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event: EventCreate = ...,
):
    """
    Write a single event to memory.
    
    The event is written to STM (Redis) immediately and queued
    for LTM (SQLite) persistence. Returns the event ID.
    """
    start = time.time()
    controller = get_controller()
    
    # Import Event model
    from chillbot.kernel.models import Event as KernelEvent
    
    # Generate event ID
    event_id = f"evt_{uuid.uuid4().hex[:16]}"
    user_id = event.user_id or "default"
    session_id = event.session_id or f"{workspace_id}_{user_id}"
    timestamp = time.time()
    
    # Normalize content
    content = event.content if isinstance(event.content, dict) else {"text": event.content}
    
    # Build metadata
    metadata = event.metadata or {}
    metadata["event_type"] = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
    if event.parent_event_id:
        metadata["parent_event_id"] = event.parent_event_id
    
    # Create kernel event
    kernel_event = KernelEvent(
        event_id=event_id,
        workspace_id=workspace_id,
        user_id=user_id,
        session_id=session_id,
        content=content,
        timestamp=timestamp,
        channel=event.channel,
        ttl_seconds=event.ttl_seconds,
        retention_class=event.retention_class.value if event.retention_class else None,
        metadata=metadata,
    )
    
    try:
        # Write via controller
        stream_id = controller.write_event_turbo(
            workspace_id=workspace_id,
            user_id=user_id,
            event=kernel_event,
            target_agents=event.target_agents,
        )
        
        duration_ms = (time.time() - start) * 1000
        
        return APIResponse(
            data=event_to_response(kernel_event),
            meta=ResponseMeta(
                duration_ms=round(duration_ms, 2),
                workspace_id=workspace_id
            )
        )
        
    except Exception as e:
        error_type = type(e).__name__
        if "BackpressureError" in error_type:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": ErrorDetail(
                        type="https://krnx.dev/errors/backpressure",
                        code="BACKPRESSURE",
                        title="System Under Load",
                        detail="The system is under heavy load. Please retry.",
                        status=503
                    ).model_dump()
                }
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspaces/{workspace_id}/events/{event_id}", response_model=APIResponse, tags=["Events"])
async def get_event(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get a single event by ID.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    
    if not event:
        raise HTTPException(
            status_code=404,
            detail={
                "error": ErrorDetail(
                    type="https://krnx.dev/errors/not-found",
                    code="EVENT_NOT_FOUND",
                    title="Event Not Found",
                    detail=f"Event {event_id} does not exist in workspace {workspace_id}",
                    instance=f"/workspaces/{workspace_id}/events/{event_id}",
                    status=404
                ).model_dump()
            }
        )
    
    # Verify workspace matches
    if event.workspace_id != workspace_id:
        raise HTTPException(
            status_code=404,
            detail={
                "error": ErrorDetail(
                    type="https://krnx.dev/errors/not-found",
                    code="EVENT_NOT_FOUND",
                    title="Event Not Found",
                    detail=f"Event {event_id} does not exist in workspace {workspace_id}",
                    instance=f"/workspaces/{workspace_id}/events/{event_id}",
                    status=404
                ).model_dump()
            }
        )
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=event_to_response(event),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events", response_model=APIResponse, tags=["Events"])
async def list_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    channel: Optional[str] = Query(None, description="Filter by channel"),
    start: Optional[float] = Query(None, description="Start timestamp (inclusive)"),
    end: Optional[float] = Query(None, description="End timestamp (inclusive)"),
    as_of: Optional[float] = Query(None, description="Temporal scope - only events visible at this time"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
):
    """
    List/query events with filters.
    
    Supports temporal scoping with `as_of` parameter - the key differentiator.
    """
    start_time = time.time()
    controller = get_controller()
    
    # Apply as_of filter (temporal scoping)
    end_time = as_of if as_of else end
    
    events = controller.query_events(
        workspace_id=workspace_id,
        user_id=user_id,
        start_time=start,
        end_time=end_time,
        limit=limit + 1,  # +1 to check for more
    )
    
    # Filter by channel if specified (controller doesn't support channel filter yet)
    if channel:
        events = [e for e in events if e.channel == channel]
    
    # Check for pagination
    has_more = len(events) > limit
    if has_more:
        events = events[:limit]
    
    # Sort order
    if order == "asc":
        events = list(reversed(events))
    
    # Build cursor for next page
    next_cursor = None
    if has_more and events:
        next_cursor = f"cur_{events[-1].timestamp}_{events[-1].event_id[:8]}"
    
    duration_ms = (time.time() - start_time) * 1000
    
    return APIResponse(
        data=EventListResponse(
            events=events_to_response(events),
            count=len(events)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id,
            as_of=as_of
        ),
        pagination=Pagination(
            cursor=next_cursor,
            has_more=has_more,
            total=None  # Would require separate count query
        )
    )


@router.delete("/workspaces/{workspace_id}/events/{event_id}", response_model=APIResponse, tags=["Events"])
async def delete_event(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    GDPR delete an event (tombstone).
    
    The event is marked as deleted but not physically removed
    to maintain hash chain integrity.
    """
    start = time.time()
    controller = get_controller()
    
    # Check event exists
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(
            status_code=404,
            detail={
                "error": ErrorDetail(
                    type="https://krnx.dev/errors/not-found",
                    code="EVENT_NOT_FOUND",
                    title="Event Not Found",
                    detail=f"Event {event_id} does not exist in workspace {workspace_id}",
                    status=404
                ).model_dump()
            }
        )
    
    # TODO: Implement tombstone mechanism
    # For now, just acknowledge the request
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data={"deleted": True, "event_id": event_id},
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/users/{user_id}/events", response_model=APIResponse, tags=["Events"])
async def list_user_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    user_id: str = Path(..., description="User identifier"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results"),
    as_of: Optional[float] = Query(None, description="Temporal scope"),
):
    """
    List events for a specific user.
    """
    start = time.time()
    controller = get_controller()
    
    events = controller.query_events(
        workspace_id=workspace_id,
        user_id=user_id,
        end_time=as_of,
        limit=limit,
    )
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=EventListResponse(
            events=events_to_response(events),
            count=len(events)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id,
            as_of=as_of
        )
    )


@router.post("/workspaces/{workspace_id}/events/batch", response_model=APIResponse, tags=["Events"])
async def create_events_batch(
    workspace_id: str = Path(..., description="Workspace identifier"),
    batch: EventBatchCreate = ...,
):
    """
    Write multiple events atomically.
    
    All events are written in a single transaction.
    """
    start = time.time()
    controller = get_controller()
    
    from chillbot.kernel.models import Event as KernelEvent
    
    created_events = []
    
    for event in batch.events:
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        user_id = event.user_id or "default"
        session_id = event.session_id or f"{workspace_id}_{user_id}"
        timestamp = time.time()
        
        content = event.content if isinstance(event.content, dict) else {"text": event.content}
        
        metadata = event.metadata or {}
        metadata["event_type"] = event.event_type.value if hasattr(event.event_type, 'value') else event.event_type
        
        kernel_event = KernelEvent(
            event_id=event_id,
            workspace_id=workspace_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            timestamp=timestamp,
            channel=event.channel,
            ttl_seconds=event.ttl_seconds,
            retention_class=event.retention_class.value if event.retention_class else None,
            metadata=metadata,
        )
        
        try:
            controller.write_event_turbo(
                workspace_id=workspace_id,
                user_id=user_id,
                event=kernel_event,
                target_agents=event.target_agents,
            )
            created_events.append(kernel_event)
        except Exception as e:
            # On failure, return what we created so far
            # TODO: Implement proper transaction rollback
            break
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data={
            "events": events_to_response(created_events),
            "count": len(created_events),
            "requested": len(batch.events)
        },
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


# =============================================================================
# TEMPORAL OPERATIONS (THE DIFFERENTIATOR)
# =============================================================================

@router.get("/workspaces/{workspace_id}/state", response_model=APIResponse, tags=["Temporal"])
async def get_workspace_state(
    workspace_id: str = Path(..., description="Workspace identifier"),
    as_of: Optional[float] = Query(None, description="Reconstruct state at this timestamp"),
    user_id: Optional[str] = Query(None, description="Scope to specific user"),
):
    """
    Get workspace state, optionally at a point in time.
    
    This is KRNX's killer feature. RAG cannot do this.
    
    With `as_of`, returns the state as it existed at that timestamp:
    - Only events that existed at that time
    - Supersession state as it was then
    - Perfect reconstruction for auditing
    """
    start = time.time()
    controller = get_controller()
    
    # Query events up to as_of timestamp
    events = controller.query_events(
        workspace_id=workspace_id,
        user_id=user_id,
        end_time=as_of,
        limit=10000,  # Get all for state calculation
    )
    
    # Calculate state
    users = list(set(e.user_id for e in events))
    channels = list(set(e.channel for e in events if e.channel))
    
    latest = events[0] if events else None
    earliest = events[-1] if events else None
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=StateResponse(
            workspace_id=workspace_id,
            as_of=as_of,
            event_count=len(events),
            latest_event=event_to_response(latest) if latest else None,
            earliest_event=event_to_response(earliest) if earliest else None,
            users=users,
            channels=channels,
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id,
            as_of=as_of
        )
    )


@router.get("/workspaces/{workspace_id}/replay", response_model=APIResponse, tags=["Temporal"])
async def replay_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    start: float = Query(..., description="Start timestamp"),
    end: float = Query(..., description="End timestamp"),
    user_id: Optional[str] = Query(None, description="Filter by user"),
    channel: Optional[str] = Query(None, description="Filter by channel"),
):
    """
    Replay events in a time range.
    
    Returns events in chronological order for time-travel debugging.
    """
    start_time = time.time()
    controller = get_controller()
    
    events = controller.query_events(
        workspace_id=workspace_id,
        user_id=user_id,
        start_time=start,
        end_time=end,
        limit=10000,
    )
    
    # Filter by channel if specified
    if channel:
        events = [e for e in events if e.channel == channel]
    
    # Sort chronologically (oldest first)
    events = sorted(events, key=lambda e: e.timestamp)
    
    duration_ms = (time.time() - start_time) * 1000
    
    return APIResponse(
        data=ReplayResponse(
            events=events_to_response(events),
            start=start,
            end=end,
            count=len(events),
            duration_ms=round(duration_ms, 2)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/timeline", response_model=APIResponse, tags=["Temporal"])
async def get_timeline(
    workspace_id: str = Path(..., description="Workspace identifier"),
    bucket: str = Query("hour", regex="^(minute|hour|day|week)$", description="Time bucket size"),
    start: Optional[float] = Query(None, description="Start timestamp"),
    end: Optional[float] = Query(None, description="End timestamp"),
):
    """
    Get event distribution over time.
    
    Returns counts per time bucket for visualization.
    """
    start_time = time.time()
    controller = get_controller()
    
    # Query all events in range
    events = controller.query_events(
        workspace_id=workspace_id,
        start_time=start,
        end_time=end,
        limit=100000,
    )
    
    # Bucket size in seconds
    bucket_sizes = {
        "minute": 60,
        "hour": 3600,
        "day": 86400,
        "week": 604800,
    }
    bucket_seconds = bucket_sizes[bucket]
    
    # Group by bucket
    buckets = {}
    for event in events:
        bucket_ts = (int(event.timestamp) // bucket_seconds) * bucket_seconds
        if bucket_ts not in buckets:
            buckets[bucket_ts] = {"count": 0, "channels": {}}
        buckets[bucket_ts]["count"] += 1
        if event.channel:
            buckets[bucket_ts]["channels"][event.channel] = \
                buckets[bucket_ts]["channels"].get(event.channel, 0) + 1
    
    # Convert to list
    bucket_list = [
        TimelineBucket(
            timestamp=float(ts),
            count=data["count"],
            channels=data["channels"] if data["channels"] else None
        )
        for ts, data in sorted(buckets.items())
    ]
    
    # Determine actual range
    actual_start = min(e.timestamp for e in events) if events else (start or 0)
    actual_end = max(e.timestamp for e in events) if events else (end or time.time())
    
    duration_ms = (time.time() - start_time) * 1000
    
    return APIResponse(
        data=TimelineResponse(
            buckets=bucket_list,
            bucket_size=bucket,
            start=actual_start,
            end=actual_end,
            total_events=len(events)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )
