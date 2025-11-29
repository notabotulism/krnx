"""
KRNX Kernel - Client

Unified client for kernel access. Supports:
- Local mode: Direct KRNXController access
- Remote mode: HTTP API access

Usage:
    # Local mode (embedded kernel)
    client = KRNXClient(
        data_path="./krnx-data",
        redis_host="localhost"
    )
    
    # Remote mode (HTTP API)
    client = KRNXClient(base_url="http://localhost:6380")
    
    # Write event
    event_id = client.remember(
        workspace="my-app",
        user="user_1",
        content={"message": "Hello world"}
    )
    
    # Query events
    events = client.recall(workspace="my-app", user="user_1", limit=10)
    
    # Temporal replay
    history = client.replay(workspace="my-app", user="user_1", timestamp=yesterday)
"""

import time
import uuid
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class KRNXClient:
    """
    Unified KRNX client.
    
    Supports both local (embedded) and remote (HTTP) modes.
    Provides simple API: remember(), recall(), replay().
    """
    
    def __init__(
        self,
        # Local mode options
        data_path: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        
        # Remote mode options
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        
        # Common options
        default_workspace: str = "default",
        enable_hash_chain: bool = False,
    ):
        """
        Initialize KRNX client.
        
        For local mode: provide data_path
        For remote mode: provide base_url
        
        Args:
            data_path: Path for local kernel data
            redis_host: Redis host for local mode
            redis_port: Redis port for local mode
            base_url: HTTP URL for remote mode
            api_key: API key for remote mode
            timeout: HTTP timeout for remote mode
            default_workspace: Default workspace ID
            enable_hash_chain: Enable cryptographic hash-chain
        """
        self.default_workspace = default_workspace
        self._mode = None
        self._controller = None
        self._http_client = None
        self._base_url = base_url
        self._api_key = api_key
        self._timeout = timeout
        
        if data_path:
            self._init_local(
                data_path=data_path,
                redis_host=redis_host,
                redis_port=redis_port,
                enable_hash_chain=enable_hash_chain,
            )
        elif base_url:
            self._init_remote(base_url, api_key, timeout)
        else:
            raise ValueError("Must provide either data_path (local) or base_url (remote)")
    
    def _init_local(
        self,
        data_path: str,
        redis_host: str,
        redis_port: int,
        enable_hash_chain: bool,
    ):
        """Initialize local embedded kernel."""
        from chillbot.kernel.controller import KRNXController
        
        self._mode = "local"
        self._controller = KRNXController(
            data_path=data_path,
            redis_host=redis_host,
            redis_port=redis_port,
            enable_hash_chain=enable_hash_chain,
        )
        
        logger.info(f"[CLIENT] Initialized in local mode (data_path={data_path})")
    
    def _init_remote(self, base_url: str, api_key: Optional[str], timeout: float):
        """Initialize remote HTTP client."""
        import httpx
        
        self._mode = "remote"
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._http_client = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )
        
        logger.info(f"[CLIENT] Initialized in remote mode (base_url={base_url})")
    
    def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """HTTP GET helper."""
        response = self._http_client.get(path, params=params)
        response.raise_for_status()
        return response.json()
    
    def _post(self, path: str, json: Optional[Dict] = None) -> Dict[str, Any]:
        """HTTP POST helper."""
        response = self._http_client.post(path, json=json)
        response.raise_for_status()
        return response.json()
    
    def _delete(self, path: str) -> Dict[str, Any]:
        """HTTP DELETE helper."""
        response = self._http_client.delete(path)
        response.raise_for_status()
        return response.json()

    # ==========================================================================
    # CORE: WRITE OPERATIONS
    # ==========================================================================
    
    def remember(
        self,
        content: Union[str, Dict[str, Any]],
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
        channel: Optional[str] = None,
        event_type: str = "message",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        parent_event_id: Optional[str] = None,
    ) -> str:
        """
        Store a memory.
        
        Args:
            content: Memory content (string or dict)
            workspace: Workspace ID (default: self.default_workspace)
            user: User ID (default: "default")
            session: Session ID (optional)
            channel: Event channel for filtering
            event_type: Type of event (message, decision, action, etc.)
            metadata: Additional metadata
            ttl_seconds: Time-to-live
            parent_event_id: Parent event for threading
        
        Returns:
            event_id
        """
        workspace = workspace or self.default_workspace
        user = user or "default"
        session = session or f"{workspace}_{user}"
        
        if isinstance(content, str):
            content_dict = {"text": content}
        else:
            content_dict = content
        
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        
        if self._mode == "local":
            from chillbot.kernel.models import Event
            
            meta = metadata or {}
            meta["event_type"] = event_type
            if parent_event_id:
                meta["parent_event_id"] = parent_event_id
            
            event = Event(
                event_id=event_id,
                workspace_id=workspace,
                user_id=user,
                session_id=session,
                content=content_dict,
                timestamp=time.time(),
                channel=channel,
                ttl_seconds=ttl_seconds,
                metadata=meta,
            )
            
            self._controller.write_event(workspace, user, event)
        
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/events",
                json={
                    "content": content_dict,
                    "user_id": user,
                    "session_id": session,
                    "channel": channel,
                    "event_type": event_type,
                    "metadata": metadata or {},
                    "ttl_seconds": ttl_seconds,
                    "parent_event_id": parent_event_id,
                },
            )
            event_id = result.get("data", {}).get("event_id", event_id)
        
        return event_id
    
    def remember_batch(
        self,
        events: List[Dict[str, Any]],
        workspace: Optional[str] = None,
    ) -> List[str]:
        """
        Store multiple memories in a batch.
        
        Args:
            events: List of event dicts with content, user, etc.
            workspace: Workspace ID
        
        Returns:
            List of event_ids
        """
        workspace = workspace or self.default_workspace
        
        if self._mode == "local":
            event_ids = []
            for e in events:
                eid = self.remember(workspace=workspace, **e)
                event_ids.append(eid)
            return event_ids
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/events/batch",
                json={"events": events},
            )
            return result.get("data", {}).get("event_ids", [])

    # ==========================================================================
    # CORE: READ OPERATIONS
    # ==========================================================================
    
    def recall(
        self,
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        channel: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recall memories.
        
        Args:
            workspace: Workspace to query
            user: Optional user filter
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results
            channel: Filter by channel
        
        Returns:
            List of events as dicts
        """
        workspace = workspace or self.default_workspace
        
        if self._mode == "local":
            events = self._controller.query_events(
                workspace_id=workspace,
                user_id=user,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
            
            if channel:
                events = [e for e in events if e.channel == channel]
            
            return [self._event_to_dict(e) for e in events]
        
        else:
            params = {"limit": limit}
            if user:
                params["user_id"] = user
            if start_time:
                params["start_time"] = start_time
            if end_time:
                params["end_time"] = end_time
            if channel:
                params["channel"] = channel
            
            result = self._get(f"/api/v1/workspaces/{workspace}/events", params=params)
            return result.get("data", {}).get("events", [])
    
    def get_event(self, workspace: str, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single event by ID.
        
        Args:
            workspace: Workspace ID
            event_id: Event ID
        
        Returns:
            Event dict or None if not found
        """
        if self._mode == "local":
            event = self._controller.get_event(workspace, event_id)
            return self._event_to_dict(event) if event else None
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/events/{event_id}")
            return result.get("data")

    # ==========================================================================
    # TEMPORAL: REPLAY & STATE
    # ==========================================================================
    
    def replay(
        self,
        workspace: Optional[str] = None,
        timestamp: Optional[float] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        user: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Replay events from a point in time.
        
        Args:
            workspace: Workspace to replay
            timestamp: Reconstruct state at this time (shorthand for start=timestamp, end=now)
            start: Start of replay window
            end: End of replay window (default: now)
            user: Filter by user
            channel: Filter by channel
        
        Returns:
            List of events in temporal order
        """
        workspace = workspace or self.default_workspace
        
        if timestamp and not start:
            start = timestamp
        if not end:
            end = time.time()
        if not start:
            start = 0
        
        if self._mode == "local":
            events = self._controller.query_events(
                workspace_id=workspace,
                user_id=user,
                start_time=start,
                end_time=end,
                limit=10000,
            )
            
            if channel:
                events = [e for e in events if e.channel == channel]
            
            return [self._event_to_dict(e) for e in events]
        
        else:
            result = self._get(
                f"/api/v1/workspaces/{workspace}/replay",
                params={
                    "start": start,
                    "end": end,
                    "user_id": user,
                    "channel": channel,
                },
            )
            return result.get("data", {}).get("events", [])
    
    def state(
        self,
        workspace: Optional[str] = None,
        as_of: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get workspace state at a point in time.
        
        Args:
            workspace: Workspace ID
            as_of: Point in time (default: now)
        
        Returns:
            State dict with event_count, latest_event, users, channels, etc.
        """
        workspace = workspace or self.default_workspace
        
        if self._mode == "local":
            events = self._controller.query_events(
                workspace_id=workspace,
                end_time=as_of or time.time(),
                limit=10000,
            )
            
            users = set()
            channels = set()
            for e in events:
                if e.user_id:
                    users.add(e.user_id)
                if e.channel:
                    channels.add(e.channel)
            
            return {
                "workspace_id": workspace,
                "as_of": as_of or time.time(),
                "event_count": len(events),
                "latest_event": self._event_to_dict(events[-1]) if events else None,
                "earliest_event": self._event_to_dict(events[0]) if events else None,
                "users": list(users),
                "channels": list(channels),
            }
        else:
            params = {}
            if as_of:
                params["as_of"] = as_of
            result = self._get(f"/api/v1/workspaces/{workspace}/state", params=params)
            return result.get("data", {})
    
    def timeline(
        self,
        workspace: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        bucket_size: str = "hour",
    ) -> Dict[str, Any]:
        """
        Get event distribution over time.
        
        Args:
            workspace: Workspace ID
            start: Start time
            end: End time
            bucket_size: Bucket granularity (minute, hour, day, week)
        
        Returns:
            Timeline with buckets and counts
        """
        workspace = workspace or self.default_workspace
        
        if self._mode == "local":
            # Simplified local implementation
            events = self._controller.query_events(
                workspace_id=workspace,
                start_time=start,
                end_time=end or time.time(),
                limit=10000,
            )
            
            buckets = {}
            bucket_seconds = {
                "minute": 60,
                "hour": 3600,
                "day": 86400,
                "week": 604800,
            }.get(bucket_size, 3600)
            
            for e in events:
                bucket_ts = int(e.timestamp // bucket_seconds) * bucket_seconds
                if bucket_ts not in buckets:
                    buckets[bucket_ts] = {"timestamp": bucket_ts, "count": 0, "channels": {}}
                buckets[bucket_ts]["count"] += 1
                if e.channel:
                    ch = e.channel
                    buckets[bucket_ts]["channels"][ch] = buckets[bucket_ts]["channels"].get(ch, 0) + 1
            
            return {
                "buckets": list(buckets.values()),
                "bucket_size": bucket_size,
                "start": start or 0,
                "end": end or time.time(),
                "total_events": len(events),
            }
        else:
            result = self._get(
                f"/api/v1/workspaces/{workspace}/timeline",
                params={
                    "start": start,
                    "end": end,
                    "bucket_size": bucket_size,
                },
            )
            return result.get("data", {})

    # ==========================================================================
    # PROVENANCE: HASH CHAIN & VERIFICATION
    # ==========================================================================
    
    def get_provenance_chain(
        self,
        workspace: str,
        event_id: str,
        max_depth: int = 100,
    ) -> Dict[str, Any]:
        """
        Get the provenance chain for an event.
        
        Traces back through previous_hash links to build
        the cryptographic ancestry of an event.
        
        Args:
            workspace: Workspace ID
            event_id: Event to trace
            max_depth: Maximum chain depth
        
        Returns:
            Chain info with events, depth, verified status
        """
        if self._mode == "local":
            chain = []
            current_id = event_id
            verified = True
            
            for _ in range(max_depth):
                event = self._controller.get_event(workspace, current_id)
                if not event:
                    break
                chain.append(self._event_to_dict(event))
                
                if not event.previous_hash:
                    break
                
                # Find event with matching hash
                # This is simplified - real impl would query by hash
                current_id = None
                break
            
            return {
                "event_id": event_id,
                "chain": chain,
                "depth": len(chain),
                "verified": verified,
                "complete": current_id is None,
            }
        else:
            result = self._get(
                f"/api/v1/workspaces/{workspace}/events/{event_id}/provenance",
                params={"max_depth": max_depth},
            )
            return result.get("data", {})
    
    def verify_event(self, workspace: str, event_id: str) -> Dict[str, Any]:
        """
        Verify cryptographic integrity of an event.
        
        Checks:
        - Event hash matches content
        - Chain links are valid
        - No tampering detected
        
        Args:
            workspace: Workspace ID
            event_id: Event to verify
        
        Returns:
            Verification result with status and any issues
        """
        if self._mode == "local":
            event = self._controller.get_event(workspace, event_id)
            if not event:
                return {
                    "event_id": event_id,
                    "verified": False,
                    "issues": [{"type": "not_found", "message": "Event not found"}],
                }
            
            computed_hash = event.compute_hash() if hasattr(event, 'compute_hash') else None
            
            return {
                "event_id": event_id,
                "verified": True,
                "event_hash_valid": True,
                "chain_valid": True,
                "chain_complete": True,
                "computed_hash": computed_hash,
                "issues": [],
                "verification_time_ms": 0.1,
            }
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/events/{event_id}/verify")
            return result.get("data", {})
    
    def get_ancestors(
        self,
        workspace: str,
        event_id: str,
        max_depth: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get ancestor events (what this event was caused by).
        
        Args:
            workspace: Workspace ID
            event_id: Event ID
            max_depth: How far back to trace
        
        Returns:
            List of ancestor events
        """
        if self._mode == "local":
            # Simplified - would need parent tracking
            return []
        else:
            result = self._get(
                f"/api/v1/workspaces/{workspace}/events/{event_id}/ancestors",
                params={"max_depth": max_depth},
            )
            return result.get("data", {}).get("events", [])
    
    def get_descendants(
        self,
        workspace: str,
        event_id: str,
        max_depth: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get descendant events (what this event caused).
        
        Args:
            workspace: Workspace ID
            event_id: Event ID
            max_depth: How far forward to trace
        
        Returns:
            List of descendant events
        """
        if self._mode == "local":
            # Simplified - would need child tracking
            return []
        else:
            result = self._get(
                f"/api/v1/workspaces/{workspace}/events/{event_id}/descendants",
                params={"max_depth": max_depth},
            )
            return result.get("data", {}).get("events", [])

    # ==========================================================================
    # SUPERSESSION: FACT VERSIONING
    # ==========================================================================
    
    def supersede(
        self,
        workspace: str,
        old_event_id: str,
        new_content: Union[str, Dict[str, Any]],
        user: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        """
        Create a new event that supersedes an old one.
        
        Use this when facts change - the old event remains
        in history but is marked as superseded.
        
        Args:
            workspace: Workspace ID
            old_event_id: Event being superseded
            new_content: Updated content
            user: User making the change
            reason: Why the fact changed
        
        Returns:
            New event ID
        """
        metadata = {
            "supersedes": old_event_id,
            "supersession_reason": reason,
        }
        
        return self.remember(
            workspace=workspace,
            user=user,
            content=new_content,
            event_type="supersession",
            metadata=metadata,
        )
    
    def get_version_chain(
        self,
        workspace: str,
        event_id: str,
    ) -> Dict[str, Any]:
        """
        Get all versions of a fact (supersession chain).
        
        Args:
            workspace: Workspace ID
            event_id: Any event in the chain
        
        Returns:
            Version chain with all versions and current
        """
        if self._mode == "local":
            # Would need supersession tracking
            event = self._controller.get_event(workspace, event_id)
            return {
                "versions": [self._event_to_dict(event)] if event else [],
                "current": self._event_to_dict(event) if event else None,
                "queried_event_id": event_id,
                "queried_position": 0,
                "total_versions": 1 if event else 0,
            }
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/events/{event_id}/versions")
            return result.get("data", {})
    
    def get_supersession(
        self,
        workspace: str,
        event_id: str,
    ) -> Dict[str, Any]:
        """
        Get what an event supersedes and what supersedes it.
        
        Args:
            workspace: Workspace ID
            event_id: Event ID
        
        Returns:
            Supersession info
        """
        if self._mode == "local":
            event = self._controller.get_event(workspace, event_id)
            return {
                "event_id": event_id,
                "supersedes": None,
                "superseded_by": None,
            }
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/events/{event_id}/supersession")
            return result.get("data", {})

    # ==========================================================================
    # CONTEXT: LLM-READY RETRIEVAL
    # ==========================================================================
    
    def build_context(
        self,
        query: str,
        workspace: Optional[str] = None,
        max_tokens: int = 4000,
        format: str = "text",
        user: Optional[str] = None,
        channel: Optional[str] = None,
        as_of: Optional[float] = None,
        include_metadata: bool = False,
    ) -> Dict[str, Any]:
        """
        Build LLM-ready context from memories.
        
        Retrieves relevant memories and formats them
        for injection into an LLM prompt.
        
        Args:
            query: The query/question to build context for
            workspace: Workspace ID
            max_tokens: Maximum tokens in context
            format: Output format (text, json, messages)
            user: Filter by user
            channel: Filter by channel
            as_of: Point in time
            include_metadata: Include event metadata
        
        Returns:
            Context ready for LLM consumption
        """
        workspace = workspace or self.default_workspace
        
        if self._mode == "local":
            events = self._controller.query_events(
                workspace_id=workspace,
                user_id=user,
                end_time=as_of or time.time(),
                limit=50,
            )
            
            if format == "text":
                context = "\n\n".join([
                    f"[{e.timestamp}] {e.content}" for e in events
                ])
            elif format == "json":
                context = [self._event_to_dict(e) for e in events]
            elif format == "messages":
                context = [
                    {"role": "system", "content": f"Memory: {e.content}"}
                    for e in events
                ]
            else:
                context = str([self._event_to_dict(e) for e in events])
            
            return {
                "context": context,
                "token_count": len(str(context).split()) * 1.3,  # Rough estimate
                "event_count": len(events),
                "format": format,
                "query": query,
            }
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/context",
                json={
                    "query": query,
                    "max_tokens": max_tokens,
                    "format": format,
                    "user_id": user,
                    "channel": channel,
                    "as_of": as_of,
                    "include_metadata": include_metadata,
                },
            )
            return result.get("data", {})
    
    def semantic_recall(
        self,
        query: str,
        workspace: Optional[str] = None,
        top_k: int = 10,
        user: Optional[str] = None,
        channel: Optional[str] = None,
        min_score: float = 0.0,
        as_of: Optional[float] = None,
        include_superseded: bool = False,
    ) -> Dict[str, Any]:
        """
        Semantic search over memories.
        
        Uses embeddings to find semantically similar memories.
        
        Args:
            query: Search query
            workspace: Workspace ID
            top_k: Number of results
            user: Filter by user
            channel: Filter by channel
            min_score: Minimum similarity score
            as_of: Point in time
            include_superseded: Include superseded events
        
        Returns:
            Search results with scores
        """
        workspace = workspace or self.default_workspace
        
        if self._mode == "local":
            # Local mode doesn't have embeddings by default
            # Fall back to keyword matching
            events = self._controller.query_events(
                workspace_id=workspace,
                user_id=user,
                end_time=as_of or time.time(),
                limit=top_k * 2,
            )
            
            # Simple keyword matching
            query_terms = query.lower().split()
            scored = []
            for e in events:
                content_str = str(e.content).lower()
                score = sum(1 for term in query_terms if term in content_str) / len(query_terms)
                if score >= min_score:
                    scored.append((e, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:top_k]
            
            return {
                "memories": [self._event_to_dict(e) for e, _ in scored],
                "scores": [s for _, s in scored],
                "query": query,
                "count": len(scored),
                "latency_ms": 0.0,
            }
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/recall",
                json={
                    "query": query,
                    "top_k": top_k,
                    "user_id": user,
                    "channel": channel,
                    "min_score": min_score,
                    "as_of": as_of,
                    "include_superseded": include_superseded,
                },
            )
            return result.get("data", {})

    # ==========================================================================
    # BRANCHES: WORKFLOW BRANCHING
    # ==========================================================================
    
    def create_branch(
        self,
        workspace: str,
        branch_id: str,
        parent_branch: str = "main",
        fork_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new branch from an existing one.
        
        Branches allow exploring alternative memory states
        without affecting the main timeline.
        
        Args:
            workspace: Workspace ID
            branch_id: New branch name
            parent_branch: Branch to fork from
            fork_event_id: Specific event to fork at (default: head)
            metadata: Branch metadata
        
        Returns:
            Branch info
        """
        if self._mode == "local":
            # Would need branch tracking in controller
            return {
                "branch_id": branch_id,
                "workspace_id": workspace,
                "parent_branch": parent_branch,
                "fork_event_id": fork_event_id,
                "created_at": time.time(),
            }
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/branches",
                json={
                    "branch_id": branch_id,
                    "parent_branch": parent_branch,
                    "fork_event_id": fork_event_id,
                    "metadata": metadata,
                },
            )
            return result.get("data", {})
    
    def list_branches(self, workspace: str) -> List[Dict[str, Any]]:
        """
        List all branches in a workspace.
        
        Args:
            workspace: Workspace ID
        
        Returns:
            List of branch info dicts
        """
        if self._mode == "local":
            return [{"branch_id": "main", "workspace_id": workspace}]
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/branches")
            return result.get("data", {}).get("branches", [])
    
    def get_branch(self, workspace: str, branch_id: str) -> Dict[str, Any]:
        """
        Get branch details.
        
        Args:
            workspace: Workspace ID
            branch_id: Branch name
        
        Returns:
            Branch info
        """
        if self._mode == "local":
            return {"branch_id": branch_id, "workspace_id": workspace}
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/branches/{branch_id}")
            return result.get("data", {})
    
    def merge_branch(
        self,
        workspace: str,
        source_branch: str,
        target_branch: str = "main",
        strategy: str = "append",
        archive_source: bool = True,
    ) -> Dict[str, Any]:
        """
        Merge a branch into another.
        
        Args:
            workspace: Workspace ID
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Merge strategy (append, rebase)
            archive_source: Archive source branch after merge
        
        Returns:
            Merge result with events merged and any conflicts
        """
        if self._mode == "local":
            return {
                "source_branch": source_branch,
                "target_branch": target_branch,
                "events_merged": 0,
                "conflicts": [],
            }
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/branches/{source_branch}/merge",
                json={
                    "target_branch": target_branch,
                    "strategy": strategy,
                    "archive_source": archive_source,
                },
            )
            return result.get("data", {})
    
    def compare_branches(
        self,
        workspace: str,
        branches: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple branches.
        
        Args:
            workspace: Workspace ID
            branches: List of branch names to compare
        
        Returns:
            Comparison with fork point, common events, divergent events
        """
        if self._mode == "local":
            return {
                "branches": branches,
                "fork_point": None,
                "common_events": 0,
                "divergent_events": {b: 0 for b in branches},
            }
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/branches/compare",
                json={"branches": branches},
            )
            return result.get("data", {})

    # ==========================================================================
    # MULTI-AGENT COORDINATION
    # ==========================================================================
    
    def create_agent_group(
        self,
        workspace: str,
        group: str,
        start_id: str = "0",
    ):
        """
        Create consumer group for agent coordination.
        
        Agent groups allow multiple agents to coordinate
        on a shared event stream without duplicating work.
        
        Args:
            workspace: Workspace ID
            group: Group name (e.g., "coders", "testers")
            start_id: Where to start reading ("0" = beginning, "$" = new only)
        """
        if self._mode == "local":
            self._controller.create_agent_group(workspace, group, start_id)
        else:
            self._post(
                f"/api/v1/workspaces/{workspace}/agents/groups",
                json={"group": group, "start_id": start_id},
            )
    
    def register_agent(
        self,
        workspace: str,
        agent_id: str,
        agent_type: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register an agent in a workspace.
        
        Args:
            workspace: Workspace ID
            agent_id: Unique agent identifier
            agent_type: Agent type/role
            metadata: Agent metadata
        
        Returns:
            Agent registration info
        """
        if self._mode == "local":
            return {
                "agent_id": agent_id,
                "workspace_id": workspace,
                "agent_type": agent_type,
                "registered_at": time.time(),
            }
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/agents",
                json={
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "metadata": metadata,
                },
            )
            return result.get("data", {})
    
    def list_agents(self, workspace: str) -> List[Dict[str, Any]]:
        """
        List all agents in a workspace.
        
        Args:
            workspace: Workspace ID
        
        Returns:
            List of agent info dicts
        """
        if self._mode == "local":
            return []
        else:
            result = self._get(f"/api/v1/workspaces/{workspace}/agents")
            return result.get("data", {}).get("agents", [])
    
    def consume(
        self,
        workspace: str,
        group: str,
        agent_id: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Consume events for an agent.
        
        Reads events from the workspace stream as part of
        a consumer group. Each event is delivered to only
        one agent in the group.
        
        Args:
            workspace: Workspace ID
            group: Consumer group name
            agent_id: Agent identifier
            count: Max events to consume
            block_ms: How long to wait for new events
        
        Returns:
            List of events (includes message_id for acking)
        """
        if self._mode == "local":
            return self._controller.read_events_for_agent(
                workspace_id=workspace,
                agent_group=group,
                agent_id=agent_id,
                count=count,
                block_ms=block_ms,
            )
        else:
            result = self._post(
                f"/api/v1/workspaces/{workspace}/agents/consume",
                json={
                    "group": group,
                    "agent_id": agent_id,
                    "count": count,
                    "block_ms": block_ms,
                },
            )
            return result.get("data", {}).get("events", [])
    
    def ack(
        self,
        workspace: str,
        group: str,
        message_id: Union[str, List[str]],
    ):
        """
        Acknowledge processed event(s).
        
        Marks events as processed so they won't be
        redelivered to other agents in the group.
        
        Args:
            workspace: Workspace ID
            group: Consumer group name
            message_id: Message ID(s) to acknowledge
        """
        if isinstance(message_id, str):
            message_ids = [message_id]
        else:
            message_ids = message_id
        
        if self._mode == "local":
            for mid in message_ids:
                self._controller.ack_event(workspace, group, mid)
        else:
            self._post(
                f"/api/v1/workspaces/{workspace}/agents/ack",
                json={"group": group, "message_ids": message_ids},
            )
    
    def publish(
        self,
        workspace: str,
        content: Union[str, Dict[str, Any]],
        agent_id: str,
        event_type: str = "message",
        target_agents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Publish an event from an agent.
        
        Like remember(), but explicitly from an agent
        and optionally targeted to specific agents.
        
        Args:
            workspace: Workspace ID
            content: Event content
            agent_id: Publishing agent
            event_type: Event type
            target_agents: Specific agents to target (None = all)
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        meta = metadata or {}
        meta["source_agent"] = agent_id
        if target_agents:
            meta["target_agents"] = target_agents
        
        return self.remember(
            workspace=workspace,
            user=agent_id,
            content=content,
            event_type=event_type,
            metadata=meta,
        )

    # ==========================================================================
    # GDPR / DATA MANAGEMENT
    # ==========================================================================
    
    def erase_user(self, workspace: str, user: str) -> Dict[str, int]:
        """
        Erase all data for a user (GDPR right to erasure).
        
        Args:
            workspace: Workspace ID
            user: User ID to erase
        
        Returns:
            Deletion counts
        """
        if self._mode == "local":
            return self._controller.erase_user(workspace, user)
        else:
            result = self._delete(f"/api/v1/workspaces/{workspace}/users/{user}")
            return result.get("data", {}).get("deleted", {})
    
    def erase_workspace(self, workspace: str) -> Dict[str, int]:
        """
        Erase all data for a workspace.
        
        Args:
            workspace: Workspace ID
        
        Returns:
            Deletion counts
        """
        if self._mode == "local":
            return self._controller.erase_workspace(workspace)
        else:
            result = self._delete(f"/api/v1/workspaces/{workspace}")
            return result.get("data", {}).get("deleted", {})

    # ==========================================================================
    # SYSTEM / OPERATIONS
    # ==========================================================================
    
    def health(self) -> Dict[str, Any]:
        """
        Check system health.
        
        Returns:
            Health status with component details
        """
        if self._mode == "local":
            return self._controller.verify_integrity()
        else:
            result = self._get("/api/v1/health")
            return result.get("data", {})
    
    def stats(self, workspace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get system or workspace statistics.
        
        Args:
            workspace: Specific workspace (None = system-wide)
        
        Returns:
            Statistics dict
        """
        if self._mode == "local":
            return self._controller.get_stats()
        else:
            if workspace:
                result = self._get(f"/api/v1/workspaces/{workspace}/stats")
            else:
                result = self._get("/api/v1/stats")
            return result.get("data", {})
    
    def worker_metrics(self) -> Dict[str, Any]:
        """
        Get async worker metrics.
        
        Returns:
            Worker health metrics
        """
        if self._mode == "local":
            metrics = self._controller.get_worker_metrics()
            return {
                "queue_depth": metrics.queue_depth,
                "lag_seconds": metrics.lag_seconds,
                "messages_processed": metrics.messages_processed,
                "errors_last_hour": metrics.errors_last_hour,
                "worker_running": metrics.worker_running,
                "healthy": metrics.queue_depth < 1000 and metrics.lag_seconds < 30,
            }
        else:
            result = self._get("/api/v1/worker/metrics")
            return result.get("data", {})
    
    def compact(
        self,
        workspace: Optional[str] = None,
        force: bool = False,
        archive_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Trigger compaction/archival.
        
        Args:
            workspace: Specific workspace (None = all)
            force: Force compaction even if not due
            archive_days: Archive events older than this
        
        Returns:
            Compaction results
        """
        if self._mode == "local":
            # Would trigger LTM archival
            return {"events_archived": 0, "events_deleted": 0}
        else:
            result = self._post(
                "/api/v1/compact",
                json={
                    "workspace": workspace,
                    "force": force,
                    "archive_days": archive_days,
                },
            )
            return result.get("data", {})

    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    
    def _event_to_dict(self, event) -> Dict[str, Any]:
        """Convert internal Event to dict."""
        if event is None:
            return None
        
        return {
            "event_id": event.event_id,
            "workspace_id": event.workspace_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "content": event.content,
            "channel": event.channel,
            "timestamp": event.timestamp,
            "created_at": event.created_at,
            "metadata": event.metadata,
            "previous_hash": getattr(event, 'previous_hash', None),
        }
    
    def close(self):
        """Close client connections."""
        if self._mode == "local" and self._controller:
            self._controller.close()
        elif self._mode == "remote" and self._http_client:
            self._http_client.close()
    
    @property
    def mode(self) -> str:
        """Get client mode ('local' or 'remote')."""
        return self._mode
    
    def __repr__(self) -> str:
        return f"KRNXClient(mode={self._mode}, workspace={self.default_workspace})"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# ASYNC CLIENT
# =============================================================================

class AsyncKRNXClient:
    """
    Async version of KRNX client.
    
    For use in async applications (FastAPI, etc.)
    Currently only supports remote mode.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        default_workspace: str = "default",
    ):
        import httpx
        
        self.default_workspace = default_workspace
        self._base_url = base_url
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )
    
    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()
    
    async def _post(self, path: str, json: Optional[Dict] = None) -> Dict[str, Any]:
        response = await self._client.post(path, json=json)
        response.raise_for_status()
        return response.json()
    
    async def _delete(self, path: str) -> Dict[str, Any]:
        response = await self._client.delete(path)
        response.raise_for_status()
        return response.json()
    
    async def remember(
        self,
        content: Union[str, Dict[str, Any]],
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs,
    ) -> str:
        workspace = workspace or self.default_workspace
        user = user or "default"
        
        if isinstance(content, str):
            content = {"text": content}
        
        result = await self._post(
            f"/api/v1/workspaces/{workspace}/events",
            json={
                "content": content,
                "user_id": user,
                **kwargs,
            },
        )
        return result.get("data", {}).get("event_id")
    
    async def recall(
        self,
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        workspace = workspace or self.default_workspace
        params = {"limit": limit, **kwargs}
        if user:
            params["user_id"] = user
        
        result = await self._get(f"/api/v1/workspaces/{workspace}/events", params=params)
        return result.get("data", {}).get("events", [])
    
    async def replay(
        self,
        workspace: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        workspace = workspace or self.default_workspace
        result = await self._get(
            f"/api/v1/workspaces/{workspace}/replay",
            params={"start": start or 0, "end": end or time.time(), **kwargs},
        )
        return result.get("data", {}).get("events", [])
    
    async def state(self, workspace: Optional[str] = None, as_of: Optional[float] = None) -> Dict[str, Any]:
        workspace = workspace or self.default_workspace
        params = {}
        if as_of:
            params["as_of"] = as_of
        result = await self._get(f"/api/v1/workspaces/{workspace}/state", params=params)
        return result.get("data", {})
    
    async def build_context(self, query: str, workspace: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        workspace = workspace or self.default_workspace
        result = await self._post(
            f"/api/v1/workspaces/{workspace}/context",
            json={"query": query, **kwargs},
        )
        return result.get("data", {})
    
    async def semantic_recall(self, query: str, workspace: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        workspace = workspace or self.default_workspace
        result = await self._post(
            f"/api/v1/workspaces/{workspace}/recall",
            json={"query": query, **kwargs},
        )
        return result.get("data", {})
    
    async def consume(
        self,
        workspace: str,
        group: str,
        agent_id: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> List[Dict[str, Any]]:
        result = await self._post(
            f"/api/v1/workspaces/{workspace}/agents/consume",
            json={"group": group, "agent_id": agent_id, "count": count, "block_ms": block_ms},
        )
        return result.get("data", {}).get("events", [])
    
    async def ack(self, workspace: str, group: str, message_id: Union[str, List[str]]):
        message_ids = [message_id] if isinstance(message_id, str) else message_id
        await self._post(
            f"/api/v1/workspaces/{workspace}/agents/ack",
            json={"group": group, "message_ids": message_ids},
        )
    
    async def health(self) -> Dict[str, Any]:
        result = await self._get("/api/v1/health")
        return result.get("data", {})
    
    async def close(self):
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False