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
            # Local mode
            self._init_local(
                data_path=data_path,
                redis_host=redis_host,
                redis_port=redis_port,
                enable_hash_chain=enable_hash_chain,
            )
        elif base_url:
            # Remote mode
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
    
    # ==============================================
    # WRITE OPERATIONS
    # ==============================================
    
    def remember(
        self,
        content: Union[str, Dict[str, Any]],
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        session: Optional[str] = None,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """
        Store a memory.
        
        Args:
            content: Memory content (string or dict)
            workspace: Workspace ID (default: self.default_workspace)
            user: User ID (default: "default")
            session: Session ID (optional)
            channel: Event channel for filtering
            metadata: Additional metadata
            ttl_seconds: Time-to-live
        
        Returns:
            event_id
        """
        workspace = workspace or self.default_workspace
        user = user or "default"
        session = session or f"{workspace}_{user}"
        
        # Normalize content
        if isinstance(content, str):
            content_dict = {"text": content}
        else:
            content_dict = content
        
        event_id = f"evt_{uuid.uuid4().hex[:16]}"
        
        if self._mode == "local":
            from chillbot.kernel.models import Event
            
            event = Event(
                event_id=event_id,
                workspace_id=workspace,
                user_id=user,
                session_id=session,
                content=content_dict,
                timestamp=time.time(),
                channel=channel,
                ttl_seconds=ttl_seconds,
                metadata=metadata or {},
            )
            
            self._controller.write_event(workspace, user, event)
        
        else:
            response = self._http_client.post(
                "/api/v1/events/write",
                json={
                    "workspace_id": workspace,
                    "user_id": user,
                    "session_id": session,
                    "content": content_dict,
                    "channel": channel,
                    "metadata": metadata or {},
                    "ttl_seconds": ttl_seconds,
                },
            )
            response.raise_for_status()
            result = response.json()
            event_id = result.get("event_id", event_id)
        
        return event_id
    
    # ==============================================
    # READ OPERATIONS
    # ==============================================
    
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
            
            # Filter by channel if specified
            if channel:
                events = [e for e in events if e.channel == channel]
            
            return [e.to_dict() for e in events]
        
        else:
            response = self._http_client.post(
                "/api/v1/events/query",
                json={
                    "workspace_id": workspace,
                    "user_id": user,
                    "start_time": start_time,
                    "end_time": end_time,
                    "limit": limit,
                    "channel": channel,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("events", [])
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single event by ID.
        
        Args:
            event_id: Event identifier
        
        Returns:
            Event dict or None
        """
        if self._mode == "local":
            event = self._controller.get_event(event_id)
            return event.to_dict() if event else None
        
        else:
            response = self._http_client.get(f"/api/v1/events/{event_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
    
    # ==============================================
    # TEMPORAL REPLAY
    # ==============================================
    
    def replay(
        self,
        workspace: str,
        user: str,
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """
        Temporal replay to a specific point in time.
        
        Args:
            workspace: Workspace ID
            user: User ID
            timestamp: Target timestamp
        
        Returns:
            List of events up to timestamp, in chronological order
        """
        if self._mode == "local":
            events = self._controller.replay_to_timestamp(
                workspace_id=workspace,
                user_id=user,
                timestamp=timestamp,
            )
            return [e.to_dict() for e in events]
        
        else:
            response = self._http_client.post(
                "/api/v1/replay",
                json={
                    "workspace_id": workspace,
                    "user_id": user,
                    "timestamp": timestamp,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("events", [])
    
    # ==============================================
    # MULTI-AGENT
    # ==============================================
    
    def create_agent_group(
        self,
        workspace: str,
        group: str,
        start_id: str = "0",
    ):
        """Create consumer group for agent coordination."""
        if self._mode == "local":
            self._controller.create_agent_group(workspace, group, start_id)
        else:
            response = self._http_client.post(
                f"/api/v1/workspaces/{workspace}/agents/{group}/create",
                params={"start_id": start_id},
            )
            response.raise_for_status()
    
    def consume(
        self,
        workspace: str,
        group: str,
        agent_id: str,
        count: int = 10,
        block_ms: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Consume events for an agent."""
        if self._mode == "local":
            return self._controller.read_events_for_agent(
                workspace_id=workspace,
                agent_group=group,
                agent_id=agent_id,
                count=count,
                block_ms=block_ms,
            )
        else:
            response = self._http_client.post(
                "/api/v1/agents/consume",
                json={
                    "workspace_id": workspace,
                    "agent_group": group,
                    "agent_id": agent_id,
                    "count": count,
                    "block_ms": block_ms,
                },
            )
            response.raise_for_status()
            return response.json().get("events", [])
    
    def ack(self, workspace: str, group: str, message_id: str):
        """Acknowledge processed event."""
        if self._mode == "local":
            self._controller.ack_event(workspace, group, message_id)
        else:
            response = self._http_client.post(
                "/api/v1/agents/ack",
                json={
                    "workspace_id": workspace,
                    "agent_group": group,
                    "stream_ids": [message_id],
                },
            )
            response.raise_for_status()
    
    # ==============================================
    # GDPR
    # ==============================================
    
    def erase_user(self, workspace: str, user: str) -> Dict[str, int]:
        """Erase all data for a user (GDPR)."""
        if self._mode == "local":
            return self._controller.erase_user(workspace, user)
        else:
            response = self._http_client.delete(
                f"/api/v1/workspaces/{workspace}/users/{user}"
            )
            response.raise_for_status()
            return response.json().get("deleted", {})
    
    def erase_workspace(self, workspace: str) -> Dict[str, int]:
        """Erase all data for a workspace (GDPR)."""
        if self._mode == "local":
            return self._controller.erase_workspace(workspace)
        else:
            response = self._http_client.delete(f"/api/v1/workspaces/{workspace}")
            response.raise_for_status()
            return response.json().get("deleted", {})
    
    # ==============================================
    # SYSTEM
    # ==============================================
    
    def health(self) -> Dict[str, Any]:
        """Check system health."""
        if self._mode == "local":
            return self._controller.verify_integrity()
        else:
            response = self._http_client.get("/api/v1/health")
            response.raise_for_status()
            return response.json()
    
    def stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        if self._mode == "local":
            return self._controller.get_stats()
        else:
            response = self._http_client.get("/api/v1/stats")
            response.raise_for_status()
            return response.json()
    
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
        """
        Initialize async client.
        
        Args:
            base_url: HTTP URL for KRNX API
            api_key: Optional API key
            timeout: Request timeout
            default_workspace: Default workspace
        """
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
        
        logger.info(f"[ASYNC_CLIENT] Initialized (base_url={base_url})")
    
    async def remember(
        self,
        content: Union[str, Dict[str, Any]],
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Store a memory (async)."""
        workspace = workspace or self.default_workspace
        user = user or "default"
        
        if isinstance(content, str):
            content_dict = {"text": content}
        else:
            content_dict = content
        
        response = await self._client.post(
            "/api/v1/events/write",
            json={
                "workspace_id": workspace,
                "user_id": user,
                "session_id": kwargs.get("session", f"{workspace}_{user}"),
                "content": content_dict,
                "channel": kwargs.get("channel"),
                "metadata": kwargs.get("metadata", {}),
                "ttl_seconds": kwargs.get("ttl_seconds"),
            },
        )
        response.raise_for_status()
        return response.json().get("event_id")
    
    async def recall(
        self,
        workspace: Optional[str] = None,
        user: Optional[str] = None,
        limit: int = 100,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Recall memories (async)."""
        workspace = workspace or self.default_workspace
        
        response = await self._client.post(
            "/api/v1/events/query",
            json={
                "workspace_id": workspace,
                "user_id": user,
                "limit": limit,
                **kwargs,
            },
        )
        response.raise_for_status()
        return response.json().get("events", [])
    
    async def health(self) -> Dict[str, Any]:
        """Check health (async)."""
        response = await self._client.get("/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close async client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False
