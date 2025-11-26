"""
KRNX STM - Short-Term Memory (Redis/KeyDB)

Optimized for:
- Multi-agent coordination (workspace streams)
- Fast writes (<1ms with pipelines)
- Event replay for new agents
- Consumer groups for agent types

Architecture:
  workspace:{workspace_id}:events  → Single stream per workspace
    ↓ Consumer groups
  agent-type:coder    → Coders subscribe
  agent-type:tester   → Testers subscribe
  agent-type:architect → Architects subscribe
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any

from chillbot.kernel.connection_pool import get_redis_client
from chillbot.kernel.models import Event

logger = logging.getLogger(__name__)


class STM:
    """
    Short-Term Memory using Redis/KeyDB.
    
    Hot tier: 24-hour retention, <1ms writes.
    Optimized with pipelines (3x faster than separate calls).
    
    Uses global connection pool for efficient resource management.
    """
    
    def __init__(
        self,
        ttl_hours: int = 24,
        use_connection_pool: bool = True
    ):
        """
        Initialize Redis STM.
        
        Args:
            ttl_hours: Time-to-live for events in hours (default 24)
            use_connection_pool: Use global connection pool (default True)
        
        Raises:
            RuntimeError: If connection pool not configured
        """
        if not use_connection_pool:
            raise RuntimeError(
                "STM requires connection pool. "
                "Call configure_pool() before creating STM."
            )
        
        self.redis = get_redis_client()
        self.ttl_seconds = ttl_hours * 3600
        
        # Test connection
        try:
            self.redis.ping()
            logger.info(f"[OK] STM connected to Redis (TTL: {ttl_hours}h)")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
    
    # ==============================================
    # EVENT OPERATIONS (Optimized with Pipelines)
    # ==============================================
    
    def write_event(
        self,
        workspace_id: str,
        user_id: str,
        event: Event,
        target_agents: Optional[List[str]] = None
    ) -> str:
        """
        Write event to STM using pipeline (3x faster).
        
        Before: 3 roundtrips (~2.4ms)
        After:  1 roundtrip (~0.8ms)
        
        Args:
            workspace_id: Workspace ID
            user_id: User ID
            event: Event to store
            target_agents: Optional list of agent types to notify
                          (e.g., ['coder', 'tester']). If None, all agents see it.
        
        Returns:
            Stream message ID
        """
        # Use pipeline for atomic multi-command execution
        pipe = self.redis.pipeline()
        
        # 1. Store event data (for quick retrieval)
        event_key = f"event:{event.event_id}"
        pipe.setex(
            event_key,
            self.ttl_seconds,
            event.to_json()
        )
        
        # 2. Add to workspace stream (for agent coordination)
        stream_name = f"workspace:{workspace_id}:events"
        
        # Optimize content_preview: only for small content
        content_str = str(event.content)
        if len(content_str) < 1000:
            content_preview = content_str[:200]
        else:
            # For large content, just use type or truncate aggressively
            content_preview = f"<{event.content.get('type', 'large_content')}>"
        
        stream_data = {
            'event_id': event.event_id,
            'user_id': user_id,
            'timestamp': str(event.timestamp),
            'target_agents': ','.join(target_agents) if target_agents else '*',  # '*' = all agents
            'content_preview': content_preview
        }
        pipe.xadd(stream_name, stream_data, maxlen=10000)  # Keep last 10k events
        
        # 3. Add to user's recent events list
        user_recent_key = f"user:{workspace_id}:{user_id}:recent"
        pipe.lpush(user_recent_key, event.event_id)
        pipe.ltrim(user_recent_key, 0, 99)  # Keep last 100 events
        pipe.expire(user_recent_key, self.ttl_seconds)
        
        # Execute all commands in one roundtrip
        results = pipe.execute()
        
        # Return stream message ID (from xadd command)
        stream_message_id = results[1]
        return stream_message_id
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get single event by ID"""
        event_key = f"event:{event_id}"
        event_json = self.redis.get(event_key)
        
        if not event_json:
            return None
        
        return Event.from_json(event_json)
    
    def get_events(
        self,
        workspace_id: str,
        user_id: str,
        limit: int = 10
    ) -> List[Event]:
        """
        Get recent events for user (fast user-specific retrieval).
        
        Uses user's recent events list, then fetches event data with pipeline.
        """
        user_recent_key = f"user:{workspace_id}:{user_id}:recent"
        event_ids = self.redis.lrange(user_recent_key, 0, limit - 1)
        
        if not event_ids:
            return []
        
        # Use pipeline to fetch all events at once
        pipe = self.redis.pipeline()
        for event_id in event_ids:
            pipe.get(f"event:{event_id}")
        
        event_jsons = pipe.execute()
        
        events = []
        for event_json in event_jsons:
            if event_json:
                events.append(Event.from_json(event_json))
        
        return events
    
    def query_events(
        self,
        workspace_id: str,
        user_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Event]:
        """
        Query events from STM with time range filtering.
        
        Gets events from user's recent list, then filters by timestamp.
        
        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            start_time: Optional start timestamp filter
            end_time: Optional end timestamp filter
        
        Returns:
            List of events matching filters
        """
        # Get all recent events for user
        user_recent_key = f"user:{workspace_id}:{user_id}:recent"
        event_ids = self.redis.lrange(user_recent_key, 0, -1)  # Get all
        
        if not event_ids:
            return []
        
        # Fetch events with pipeline
        pipe = self.redis.pipeline()
        for event_id in event_ids:
            pipe.get(f"event:{event_id}")
        
        event_jsons = pipe.execute()
        
        # Parse and filter by timestamp
        events = []
        for event_json in event_jsons:
            if event_json:
                event = Event.from_json(event_json)
                
                # Apply time filters
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                events.append(event)
        
        return events
    
    # ==============================================
    # WORKSPACE STREAMS (Multi-Agent Coordination)
    # ==============================================
    
    def create_consumer_group(
        self,
        workspace_id: str,
        group_name: str,
        start_id: str = '0'
    ):
        """
        Create consumer group for agent type.
        
        Example groups:
        - 'agent-type:coder'
        - 'agent-type:tester'
        - 'agent-type:architect'
        
        Args:
            workspace_id: Workspace ID
            group_name: Consumer group name (usually 'agent-type:X')
            start_id: Where to start reading ('0' = from beginning, '$' = new only)
        """
        stream_name = f"workspace:{workspace_id}:events"
        try:
            self.redis.xgroup_create(
                stream_name,
                group_name,
                id=start_id,
                mkstream=True
            )
            logger.info(f"[OK] Created consumer group '{group_name}' for workspace '{workspace_id}'")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise
    
    def read_events_for_agent(
        self,
        workspace_id: str,
        agent_group: str,
        agent_id: str,
        count: int = 10,
        block_ms: int = 1000,
        filter_targets: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Read events from workspace stream for specific agent type.
        
        Uses consumer groups for:
        - Guaranteed delivery (XACK required)
        - Load balancing (multiple agents of same type share work)
        - No duplicate processing within group
        
        Args:
            workspace_id: Workspace ID
            agent_group: Consumer group name (e.g., 'agent-type:coder')
            agent_id: Unique agent identifier
            count: Max events to read
            block_ms: How long to wait for new events (0 = non-blocking)
            filter_targets: Optional agent types to filter for (e.g., ['coder', 'tester'])
        
        Returns:
            List of events with metadata
        """
        stream_name = f"workspace:{workspace_id}:events"
        
        # Read from consumer group (Redis handles coordination)
        messages = self.redis.xreadgroup(
            groupname=agent_group,
            consumername=agent_id,
            streams={stream_name: '>'},  # '>' = unread messages
            count=count,
            block=block_ms
        )
        
        if not messages:
            return []
        
        events = []
        for stream, msg_list in messages:
            for msg_id, data in msg_list:
                # Apply target filter if specified
                target_agents = data.get('target_agents', '*')
                
                if filter_targets and target_agents != '*':
                    # Check if this event is targeted to this agent type
                    targets = target_agents.split(',')
                    if not any(ft in targets for ft in filter_targets):
                        # Not for us, but ack anyway to move consumer forward
                        self.redis.xack(stream_name, agent_group, msg_id)
                        continue
                
                events.append({
                    'message_id': msg_id,
                    'event_id': data.get('event_id'),
                    'user_id': data.get('user_id'),
                    'timestamp': float(data.get('timestamp', 0)),
                    'target_agents': target_agents,
                    'content_preview': data.get('content_preview', '')
                })
        
        return events
    
    def ack_event(
        self,
        workspace_id: str,
        agent_group: str,
        message_id: str
    ):
        """
        Acknowledge event processing.
        
        Required for consumer group pattern - tells Redis
        "I processed this, remove from pending list"
        """
        stream_name = f"workspace:{workspace_id}:events"
        self.redis.xack(stream_name, agent_group, message_id)
    
    def get_pending_events(
        self,
        workspace_id: str,
        agent_group: str,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get events that were delivered but not acknowledged.
        
        Used for crash recovery - if agent dies mid-processing,
        pending events can be reclaimed.
        """
        stream_name = f"workspace:{workspace_id}:events"
        
        # Get pending events for this consumer
        pending = self.redis.xpending_range(
            stream_name,
            agent_group,
            min='-',
            max='+',
            count=100,
            consumername=agent_id
        )
        
        return [
            {
                'message_id': p['message_id'],
                'consumer': p['consumer'],
                'time_since_delivered': p['time_since_delivered']
            }
            for p in pending
        ]
    
    def replay_events(
        self,
        workspace_id: str,
        since_timestamp: Optional[float] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Replay events from workspace stream.
        
        Used when:
        - New agent joins and needs history
        - Time-travel debugging
        - Audit/compliance
        
        Args:
            workspace_id: Workspace ID
            since_timestamp: Unix timestamp to replay from (None = from beginning)
            limit: Max events to return
        
        Returns:
            List of historical events
        """
        stream_name = f"workspace:{workspace_id}:events"
        
        # Convert timestamp to stream ID if provided
        start_id = '0'
        if since_timestamp:
            # Redis stream IDs are {timestamp_ms}-{sequence}
            start_id = f"{int(since_timestamp * 1000)}-0"
        
        # Read from stream without consumer group (no ack needed)
        messages = self.redis.xrange(
            stream_name,
            min=start_id,
            max='+',
            count=limit
        )
        
        events = []
        for msg_id, data in messages:
            events.append({
                'message_id': msg_id,
                'event_id': data.get('event_id'),
                'user_id': data.get('user_id'),
                'timestamp': float(data.get('timestamp', 0)),
                'target_agents': data.get('target_agents', '*'),
                'content_preview': data.get('content_preview', '')
            })
        
        return events
    
    # ==============================================
    # DELETE OPERATIONS (Constitution 6.4)
    # ==============================================
    
    def delete_workspace(self, workspace_id: str) -> int:
        """
        Delete all data for workspace.
        
        Deletes:
        - Workspace stream
        - All user:workspace:user:recent lists
        - All event:* keys for this workspace (approximate)
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            Approximate number of keys deleted
        """
        deleted = 0
        
        # Delete workspace stream
        stream_name = f"workspace:{workspace_id}:events"
        deleted += self.redis.delete(stream_name)
        
        # Delete user recent lists (scan for all users in workspace)
        pattern = f"user:{workspace_id}:*:recent"
        for key in self.redis.scan_iter(match=pattern, count=100):
            deleted += self.redis.delete(key)
        
        # Note: event:* keys expire naturally after TTL
        # We don't explicitly delete them to avoid full scan
        
        logger.info(f"[OK] Deleted {deleted} keys for workspace '{workspace_id}'")
        return deleted
    
    def delete_user(self, workspace_id: str, user_id: str) -> int:
        """
        Delete all data for user within workspace.
        
        Deletes:
        - User recent events list
        - Event keys for this user (best effort via recent list)
        
        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
        
        Returns:
            Number of keys deleted
        """
        deleted = 0
        
        # Get user's recent events to delete them
        user_recent_key = f"user:{workspace_id}:{user_id}:recent"
        event_ids = self.redis.lrange(user_recent_key, 0, -1)
        
        # Delete event keys
        for event_id in event_ids:
            event_key = f"event:{event_id}"
            deleted += self.redis.delete(event_key)
        
        # Delete user recent list
        deleted += self.redis.delete(user_recent_key)
        
        logger.info(f"[OK] Deleted {deleted} keys for user '{workspace_id}/{user_id}'")
        return deleted
    
    # ==============================================
    # STATS & LIFECYCLE
    # ==============================================
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for workspace stream"""
        stream_name = f"workspace:{workspace_id}:events"
        
        try:
            info = self.redis.xinfo_stream(stream_name)
            groups = self.redis.xinfo_groups(stream_name)
            
            # Handle empty stream
            first_entry = info.get('first-entry')
            last_entry = info.get('last-entry')
            
            return {
                'stream_length': info.get('length', 0),
                'first_entry_id': first_entry[0] if first_entry else None,
                'last_entry_id': last_entry[0] if last_entry else None,
                'consumer_groups': [
                    {
                        'name': g['name'],
                        'consumers': g['consumers'],
                        'pending': g['pending'],
                        'last_delivered_id': g['last-delivered-id']
                    }
                    for g in groups
                ]
            }
        except Exception as e:
            logger.warning(f"[STATS] Failed to get workspace stats for '{workspace_id}': {e}")
            return {
                'stream_length': 0,
                'consumer_groups': []
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall STM statistics"""
        info = self.redis.info()
        
        return {
            'connected': True,
            'used_memory_mb': round(info['used_memory'] / (1024 * 1024), 2),
            'total_keys': info['db0']['keys'] if 'db0' in info else 0,
            'uptime_seconds': info['uptime_in_seconds']
        }
    
    def close(self):
        """Close Redis connection"""
        if self.redis:
            self.redis.close()
