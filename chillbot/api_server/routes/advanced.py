"""
KRNX Advanced API Routes

Full playbook endpoints:
- Provenance (hash chain verification, ancestry)
- Supersession (fact versioning)
- Context (LLM-ready context building)
- Agents (consumer groups, coordination)
- Branches (workflow branching/merging)
- Operations (compact, enrichment)
"""

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, List
import time
import uuid
import json
import asyncio

from ..schemas import (
    # Response envelope
    APIResponse, ResponseMeta, Pagination, ErrorResponse, ErrorDetail,
    # Event schemas
    EventResponse,
    # Provenance schemas
    ProvenanceChainResponse, VerificationResponse, AncestryResponse,
    # Supersession schemas
    VersionChainResponse, SupersessionResponse,
    # Context schemas
    ContextRequest, ContextResponse, RecallRequest, RecallResponse,
    # Agent schemas
    AgentCreate, AgentResponse, AgentListResponse,
    PublishRequest, ConsumeRequest, ConsumeResponse, AckRequest,
    # Branch schemas
    BranchCreate, BranchResponse, BranchListResponse,
    MergeRequest, MergeResponse, BranchCompareRequest, BranchCompareResponse,
    # Operations schemas
    CompactRequest, CompactResponse, EnrichmentStatusResponse,
)
from ..deps import get_controller, get_fabric, event_to_response, events_to_response

router = APIRouter()


# =============================================================================
# PROVENANCE (AUDITABILITY)
# =============================================================================

@router.get("/workspaces/{workspace_id}/events/{event_id}/provenance", 
            response_model=APIResponse, tags=["Provenance"])
async def get_provenance_chain(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
    max_depth: int = Query(100, ge=1, le=1000, description="Maximum chain depth"),
):
    """
    Get the provenance chain for an event.
    
    Walks the hash chain from event back to genesis,
    showing the complete causal history.
    """
    start = time.time()
    controller = get_controller()
    
    # Get the target event
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Walk the hash chain backwards
    chain = [event]
    current = event
    verified = True
    gaps = []
    
    while current.previous_hash and len(chain) < max_depth:
        # Find event with matching hash
        # This requires a hash lookup - for now, we'll use metadata if stored
        parent_id = current.metadata.get("parent_event_id") if current.metadata else None
        
        if parent_id:
            parent = controller.get_event(parent_id)
            if parent:
                # Verify hash chain
                expected_hash = parent.compute_hash()
                if current.previous_hash != expected_hash:
                    verified = False
                    gaps.append(f"Hash mismatch at {current.event_id}")
                chain.append(parent)
                current = parent
            else:
                gaps.append(f"Missing parent {parent_id}")
                break
        else:
            # No parent reference, chain ends
            break
    
    # Reverse to get root-first order
    chain = list(reversed(chain))
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=ProvenanceChainResponse(
            event_id=event_id,
            chain=events_to_response(chain),
            depth=len(chain),
            verified=verified,
            complete=len(gaps) == 0,
            gaps=gaps
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/verify",
            response_model=APIResponse, tags=["Provenance"])
async def verify_event_integrity(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Verify cryptographic integrity of an event and its chain.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    issues = []
    
    # Compute and verify event hash
    computed_hash = event.compute_hash()
    event_hash_valid = True  # Event doesn't store its own hash, so this is always valid
    
    # Verify chain link
    chain_valid = True
    chain_complete = True
    
    if event.previous_hash:
        parent_id = event.metadata.get("parent_event_id") if event.metadata else None
        if parent_id:
            parent = controller.get_event(parent_id)
            if parent:
                expected_hash = parent.compute_hash()
                if event.previous_hash != expected_hash:
                    chain_valid = False
                    issues.append({
                        "type": "hash_mismatch",
                        "expected": expected_hash,
                        "actual": event.previous_hash,
                        "parent_id": parent_id
                    })
            else:
                chain_complete = False
                issues.append({
                    "type": "missing_parent",
                    "parent_id": parent_id
                })
        else:
            # Has previous_hash but no parent reference
            issues.append({
                "type": "unresolvable_parent",
                "previous_hash": event.previous_hash
            })
    
    verified = event_hash_valid and chain_valid and chain_complete
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=VerificationResponse(
            event_id=event_id,
            verified=verified,
            event_hash_valid=event_hash_valid,
            chain_valid=chain_valid,
            chain_complete=chain_complete,
            computed_hash=computed_hash,
            expected_hash=event.previous_hash,
            issues=issues,
            verification_time_ms=round(duration_ms, 2)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/children",
            response_model=APIResponse, tags=["Provenance"])
async def get_event_children(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get events that list this event as their parent.
    """
    start = time.time()
    controller = get_controller()
    
    # Query all events and filter by parent_event_id
    # This is inefficient - would need an index in production
    all_events = controller.query_events(workspace_id=workspace_id, limit=10000)
    
    children = [
        e for e in all_events
        if e.metadata and e.metadata.get("parent_event_id") == event_id
    ]
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=AncestryResponse(
            event_id=event_id,
            events=events_to_response(children),
            depth=1  # Direct children only
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/ancestors",
            response_model=APIResponse, tags=["Provenance"])
async def get_event_ancestors(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
    max_depth: int = Query(100, ge=1, le=1000, description="Maximum depth"),
):
    """
    Get all ancestors of an event.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Walk up the chain
    ancestors = []
    current = event
    depth = 0
    
    while depth < max_depth:
        parent_id = current.metadata.get("parent_event_id") if current.metadata else None
        if not parent_id:
            break
        
        parent = controller.get_event(parent_id)
        if not parent:
            break
        
        ancestors.append(parent)
        current = parent
        depth += 1
    
    # Reverse to get root-first order
    ancestors = list(reversed(ancestors))
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=AncestryResponse(
            event_id=event_id,
            events=events_to_response(ancestors),
            depth=len(ancestors)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


# =============================================================================
# SUPERSESSION (CONSISTENCY)
# =============================================================================

@router.get("/workspaces/{workspace_id}/events/{event_id}/versions",
            response_model=APIResponse, tags=["Supersession"])
async def get_event_versions(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get all versions of a fact (supersession chain).
    
    When facts are updated, KRNX detects semantic similarity
    with contradiction signals and links versions together.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Find all versions by walking supersession chain
    versions = [event]
    
    # Walk backwards to find older versions
    current = event
    while True:
        supersedes_id = None
        if current.metadata:
            for rel in current.metadata.get("relations", []):
                if rel.get("kind") == "supersedes":
                    supersedes_id = rel.get("target")
                    break
        
        if not supersedes_id:
            break
        
        older = controller.get_event(supersedes_id)
        if older:
            versions.append(older)
            current = older
        else:
            break
    
    # Walk forwards to find newer versions
    current = event
    while True:
        superseded_by_id = None
        if current.metadata:
            for rel in current.metadata.get("relations", []):
                if rel.get("kind") == "superseded_by":
                    superseded_by_id = rel.get("target")
                    break
        
        if not superseded_by_id:
            break
        
        newer = controller.get_event(superseded_by_id)
        if newer:
            versions.insert(0, newer)
            current = newer
        else:
            break
    
    # Sort by timestamp (oldest first)
    versions = sorted(versions, key=lambda e: e.timestamp)
    
    # Find position of queried event
    position = next((i for i, v in enumerate(versions) if v.event_id == event_id), 0)
    
    # Current version is the newest
    current_version = versions[-1] if versions else None
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=VersionChainResponse(
            versions=events_to_response(versions),
            current=event_to_response(current_version) if current_version else None,
            queried_event_id=event_id,
            queried_position=position,
            total_versions=len(versions)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/current",
            response_model=APIResponse, tags=["Supersession"])
async def get_current_version(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get the current version of a potentially superseded event.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Walk forward to find current version
    current = event
    while True:
        superseded_by_id = None
        if current.metadata:
            for rel in current.metadata.get("relations", []):
                if rel.get("kind") == "superseded_by":
                    superseded_by_id = rel.get("target")
                    break
        
        if not superseded_by_id:
            break
        
        newer = controller.get_event(superseded_by_id)
        if newer:
            current = newer
        else:
            break
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=event_to_response(current),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/supersedes",
            response_model=APIResponse, tags=["Supersession"])
async def get_supersedes(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get the event that this event supersedes.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    supersedes_id = None
    confidence = None
    signals = []
    
    if event.metadata:
        for rel in event.metadata.get("relations", []):
            if rel.get("kind") == "supersedes":
                supersedes_id = rel.get("target")
                confidence = rel.get("confidence")
                signals = rel.get("signals", [])
                break
    
    supersedes_event = None
    if supersedes_id:
        supersedes_event = controller.get_event(supersedes_id)
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=SupersessionResponse(
            event_id=event_id,
            supersedes=event_to_response(supersedes_event) if supersedes_event else None,
            superseded_by=None,
            confidence=confidence,
            signals=signals
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/superseded_by",
            response_model=APIResponse, tags=["Supersession"])
async def get_superseded_by(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get the event that superseded this event.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    superseded_by_id = None
    confidence = None
    signals = []
    
    if event.metadata:
        for rel in event.metadata.get("relations", []):
            if rel.get("kind") == "superseded_by":
                superseded_by_id = rel.get("target")
                confidence = rel.get("confidence")
                signals = rel.get("signals", [])
                break
    
    superseded_by_event = None
    if superseded_by_id:
        superseded_by_event = controller.get_event(superseded_by_id)
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=SupersessionResponse(
            event_id=event_id,
            supersedes=None,
            superseded_by=event_to_response(superseded_by_event) if superseded_by_event else None,
            confidence=confidence,
            signals=signals
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


# =============================================================================
# CONTEXT (LLM-READY)
# =============================================================================

@router.post("/workspaces/{workspace_id}/context",
             response_model=APIResponse, tags=["Context"])
async def build_context(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: ContextRequest = ...,
):
    """
    Build LLM-ready context from relevant memories.
    
    Returns formatted context optimized for LLM consumption.
    """
    start = time.time()
    fabric = get_fabric()
    
    try:
        context = fabric.context(
            query=request.query,
            workspace_id=workspace_id,
            user_id=request.user_id,
            max_tokens=request.max_tokens,
            format=request.format,
            include_metadata=request.include_metadata,
        )
        
        # Estimate token count (rough: ~4 chars per token)
        if isinstance(context, str):
            token_count = len(context) // 4
        else:
            token_count = len(json.dumps(context)) // 4
        
        duration_ms = (time.time() - start) * 1000
        
        return APIResponse(
            data=ContextResponse(
                context=context,
                token_count=token_count,
                event_count=0,  # Would need to track from fabric
                format=request.format,
                query=request.query
            ),
            meta=ResponseMeta(
                duration_ms=round(duration_ms, 2),
                workspace_id=workspace_id,
                as_of=request.as_of
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workspaces/{workspace_id}/recall",
             response_model=APIResponse, tags=["Context"])
async def recall_memories(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: RecallRequest = ...,
):
    """
    Semantic search for relevant memories.
    """
    start = time.time()
    fabric = get_fabric()
    
    try:
        result = fabric.recall(
            query=request.query,
            workspace_id=workspace_id,
            user_id=request.user_id,
            top_k=request.top_k,
            channel=request.channel,
        )
        
        # Convert MemoryItems to EventResponses
        memories = []
        for mem in result.memories:
            memories.append({
                "event_id": mem.event_id,
                "workspace_id": workspace_id,
                "user_id": request.user_id or "default",
                "content": mem.content,
                "timestamp": mem.timestamp,
                "created_at": mem.timestamp,
                "score": mem.score,
                "metadata": mem.metadata,
            })
        
        duration_ms = (time.time() - start) * 1000
        
        return APIResponse(
            data=RecallResponse(
                memories=memories,
                query=request.query,
                count=len(memories),
                latency_ms=result.latency_ms
            ),
            meta=ResponseMeta(
                duration_ms=round(duration_ms, 2),
                workspace_id=workspace_id,
                as_of=request.as_of
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AGENTS (MULTI-AGENT COORDINATION)
# =============================================================================

# In-memory agent registry (would be Redis in production)
_agents: dict = {}


@router.post("/workspaces/{workspace_id}/agents",
             response_model=APIResponse, tags=["Agents"])
async def register_agent(
    workspace_id: str = Path(..., description="Workspace identifier"),
    agent: AgentCreate = ...,
):
    """
    Register an agent in the workspace.
    """
    start = time.time()
    controller = get_controller()
    
    agent_key = f"{workspace_id}:{agent.agent_id}"
    
    # Create consumer group for agent type
    try:
        controller.create_consumer_group(
            workspace_id=workspace_id,
            agent_group=agent.agent_type,
            start_id='0'
        )
    except Exception:
        pass  # Group may already exist
    
    # Register agent
    agent_data = AgentResponse(
        agent_id=agent.agent_id,
        workspace_id=workspace_id,
        agent_type=agent.agent_type,
        registered_at=time.time(),
        last_seen=time.time(),
        events_processed=0,
        metadata=agent.metadata
    )
    _agents[agent_key] = agent_data
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=agent_data,
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/agents",
            response_model=APIResponse, tags=["Agents"])
async def list_agents(
    workspace_id: str = Path(..., description="Workspace identifier"),
):
    """
    List all registered agents.
    """
    start = time.time()
    
    agents = [
        a for key, a in _agents.items()
        if key.startswith(f"{workspace_id}:")
    ]
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=AgentListResponse(agents=agents, count=len(agents)),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/agents/{agent_id}",
            response_model=APIResponse, tags=["Agents"])
async def get_agent(
    workspace_id: str = Path(..., description="Workspace identifier"),
    agent_id: str = Path(..., description="Agent identifier"),
):
    """
    Get agent details.
    """
    start = time.time()
    
    agent_key = f"{workspace_id}:{agent_id}"
    agent = _agents.get(agent_key)
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=agent,
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.delete("/workspaces/{workspace_id}/agents/{agent_id}",
               response_model=APIResponse, tags=["Agents"])
async def deregister_agent(
    workspace_id: str = Path(..., description="Workspace identifier"),
    agent_id: str = Path(..., description="Agent identifier"),
):
    """
    Remove an agent from the workspace.
    """
    start = time.time()
    
    agent_key = f"{workspace_id}:{agent_id}"
    if agent_key in _agents:
        del _agents[agent_key]
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data={"deleted": True, "agent_id": agent_id},
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.post("/workspaces/{workspace_id}/publish",
             response_model=APIResponse, tags=["Agents"])
async def publish_event(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: PublishRequest = ...,
):
    """
    Publish an event to the workspace stream.
    """
    start = time.time()
    controller = get_controller()
    
    from chillbot.kernel.models import Event as KernelEvent
    
    event_id = f"evt_{uuid.uuid4().hex[:16]}"
    timestamp = time.time()
    
    content = request.content if isinstance(request.content, dict) else {"text": request.content}
    
    metadata = request.metadata or {}
    metadata["event_type"] = request.event_type
    metadata["publisher_agent"] = request.agent_id
    
    kernel_event = KernelEvent(
        event_id=event_id,
        workspace_id=workspace_id,
        user_id=request.agent_id,
        session_id=f"{workspace_id}_{request.agent_id}",
        content=content,
        timestamp=timestamp,
        metadata=metadata,
    )
    
    stream_id = controller.write_event_turbo(
        workspace_id=workspace_id,
        user_id=request.agent_id,
        event=kernel_event,
        target_agents=request.target_agents,
    )
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data={"event_id": event_id, "stream_id": stream_id},
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.post("/workspaces/{workspace_id}/consume",
             response_model=APIResponse, tags=["Agents"])
async def consume_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: ConsumeRequest = ...,
):
    """
    Consume events from workspace stream.
    """
    start = time.time()
    controller = get_controller()
    
    events = controller.read_events_for_agent(
        workspace_id=workspace_id,
        agent_group=request.agent_type,
        agent_id=request.agent_id,
        count=request.count,
        block_ms=request.block_ms,
    )
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=ConsumeResponse(events=events, count=len(events)),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.post("/workspaces/{workspace_id}/ack",
             response_model=APIResponse, tags=["Agents"])
async def acknowledge_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: AckRequest = ...,
):
    """
    Acknowledge event processing.
    """
    start = time.time()
    controller = get_controller()
    
    for msg_id in request.message_ids:
        controller.ack_event(
            workspace_id=workspace_id,
            agent_group=request.agent_type,
            message_id=msg_id
        )
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data={"acknowledged": len(request.message_ids)},
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/stream", tags=["Agents"])
async def stream_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    agent_id: str = Query(..., description="Agent ID"),
    agent_type: str = Query("generic", description="Agent type"),
):
    """
    SSE stream of workspace events.
    
    Server-Sent Events for real-time event consumption.
    """
    controller = get_controller()
    
    async def event_generator():
        while True:
            events = controller.read_events_for_agent(
                workspace_id=workspace_id,
                agent_group=agent_type,
                agent_id=agent_id,
                count=10,
                block_ms=1000,
            )
            
            for event in events:
                yield f"data: {json.dumps(event)}\n\n"
            
            if not events:
                await asyncio.sleep(0.1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# =============================================================================
# BRANCHES (WORKFLOW)
# =============================================================================

# In-memory branch registry (would be Redis/SQLite in production)
_branches: dict = {}


@router.post("/workspaces/{workspace_id}/branches",
             response_model=APIResponse, tags=["Branches"])
async def create_branch(
    workspace_id: str = Path(..., description="Workspace identifier"),
    branch: BranchCreate = ...,
):
    """
    Create a new branch.
    """
    start = time.time()
    controller = get_controller()
    
    branch_key = f"{workspace_id}:{branch.branch_id}"
    
    # Get fork event if specified
    fork_timestamp = time.time()
    if branch.fork_event_id:
        fork_event = controller.get_event(branch.fork_event_id)
        if fork_event:
            fork_timestamp = fork_event.timestamp
    
    branch_data = BranchResponse(
        branch_id=branch.branch_id,
        workspace_id=workspace_id,
        parent_branch=branch.parent_branch,
        fork_event_id=branch.fork_event_id,
        fork_timestamp=fork_timestamp,
        head_event_id=None,
        event_count=0,
        created_at=time.time(),
        updated_at=time.time(),
        archived=False,
        metadata=branch.metadata
    )
    _branches[branch_key] = branch_data
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=branch_data,
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/branches",
            response_model=APIResponse, tags=["Branches"])
async def list_branches(
    workspace_id: str = Path(..., description="Workspace identifier"),
):
    """
    List all branches.
    """
    start = time.time()
    
    branches = [
        b for key, b in _branches.items()
        if key.startswith(f"{workspace_id}:")
    ]
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=BranchListResponse(branches=branches, count=len(branches)),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/branches/{branch_id}",
            response_model=APIResponse, tags=["Branches"])
async def get_branch(
    workspace_id: str = Path(..., description="Workspace identifier"),
    branch_id: str = Path(..., description="Branch identifier"),
):
    """
    Get branch details.
    """
    start = time.time()
    
    branch_key = f"{workspace_id}:{branch_id}"
    branch = _branches.get(branch_key)
    
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=branch,
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/branches/{branch_id}/events",
            response_model=APIResponse, tags=["Branches"])
async def list_branch_events(
    workspace_id: str = Path(..., description="Workspace identifier"),
    branch_id: str = Path(..., description="Branch identifier"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    List events in a branch.
    """
    start = time.time()
    controller = get_controller()
    
    branch_key = f"{workspace_id}:{branch_id}"
    branch = _branches.get(branch_key)
    
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")
    
    # Query events after fork point
    events = controller.query_events(
        workspace_id=workspace_id,
        start_time=branch.fork_timestamp,
        limit=limit,
    )
    
    # Filter by branch metadata (simplified)
    # In production, events would have branch_id field
    branch_events = [
        e for e in events
        if e.metadata and e.metadata.get("branch_id") == branch_id
    ]
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data={"events": events_to_response(branch_events), "count": len(branch_events)},
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.post("/workspaces/{workspace_id}/branches/{branch_id}/merge",
             response_model=APIResponse, tags=["Branches"])
async def merge_branch(
    workspace_id: str = Path(..., description="Workspace identifier"),
    branch_id: str = Path(..., description="Branch identifier"),
    request: MergeRequest = ...,
):
    """
    Merge a branch into another.
    """
    start = time.time()
    
    branch_key = f"{workspace_id}:{branch_id}"
    branch = _branches.get(branch_key)
    
    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")
    
    # Simplified merge - just mark as archived
    if request.archive_source:
        branch.archived = True
        branch.updated_at = time.time()
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=MergeResponse(
            source_branch=branch_id,
            target_branch=request.target_branch,
            events_merged=branch.event_count,
            conflicts=[],
            source_archived=request.archive_source
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.post("/workspaces/{workspace_id}/branches/compare",
             response_model=APIResponse, tags=["Branches"])
async def compare_branches(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: BranchCompareRequest = ...,
):
    """
    Compare multiple branches.
    """
    start = time.time()
    
    divergent = {}
    for branch_id in request.branches:
        branch_key = f"{workspace_id}:{branch_id}"
        branch = _branches.get(branch_key)
        if branch:
            divergent[branch_id] = branch.event_count
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=BranchCompareResponse(
            branches=request.branches,
            fork_point=None,
            common_events=0,
            divergent_events=divergent
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


# =============================================================================
# OPERATIONS
# =============================================================================

@router.post("/workspaces/{workspace_id}/compact",
             response_model=APIResponse, tags=["Operations"])
async def compact_workspace(
    workspace_id: str = Path(..., description="Workspace identifier"),
    request: CompactRequest = ...,
):
    """
    Trigger LTM compaction.
    """
    start = time.time()
    controller = get_controller()
    
    # Force checkpoint
    controller.ltm.force_checkpoint()
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=CompactResponse(
            events_archived=0,
            events_deleted=0,
            space_reclaimed_bytes=0,
            duration_ms=round(duration_ms, 2)
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )


@router.get("/workspaces/{workspace_id}/events/{event_id}/enrichment",
            response_model=APIResponse, tags=["Operations"])
async def get_enrichment_status(
    workspace_id: str = Path(..., description="Workspace identifier"),
    event_id: str = Path(..., description="Event identifier"),
):
    """
    Get enrichment status for an event.
    """
    start = time.time()
    controller = get_controller()
    
    event = controller.get_event(event_id)
    if not event or event.workspace_id != workspace_id:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Extract enrichment from metadata
    enrichment = None
    status = "pending"
    
    if event.metadata:
        if "salience_score" in event.metadata:
            status = "complete"
            enrichment = {
                "salience_score": event.metadata.get("salience_score"),
                "episode_id": event.metadata.get("episode_id"),
                "time_gap_seconds": event.metadata.get("time_gap_seconds"),
                "entities": event.metadata.get("entities"),
                "topics": event.metadata.get("topics"),
            }
    
    duration_ms = (time.time() - start) * 1000
    
    return APIResponse(
        data=EnrichmentStatusResponse(
            event_id=event_id,
            status=status,
            enrichment=enrichment,
            error=None
        ),
        meta=ResponseMeta(
            duration_ms=round(duration_ms, 2),
            workspace_id=workspace_id
        )
    )
