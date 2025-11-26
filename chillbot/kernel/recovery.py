"""
KRNX Crash Recovery - Reclaim Pending Messages

Handles recovery from worker crashes by reclaiming pending messages
that were delivered but never acknowledged.

Features:
- Detects pending messages on startup
- Reclaims messages from dead consumers
- Configurable idle timeout
- Safe reclamation (only touches truly dead consumers)

Usage:
    from chillbot.kernel.recovery import CrashRecovery
    
    recovery = CrashRecovery(redis_client)
    
    # On startup
    reclaimed = recovery.reclaim_pending_messages(
        stream="krnx:ltm:queue",
        group="krnx-ltm-workers",
        idle_timeout_ms=300000  # 5 minutes
    )
"""

import logging
import time
from typing import List, Dict, Any, Optional
import redis

logger = logging.getLogger(__name__)


class CrashRecovery:
    """
    Crash recovery for Redis Streams consumer groups.
    
    Reclaims pending messages from dead/stuck consumers.
    """
    
    def __init__(self, redis_client: redis.Redis):
        """
        Initialize crash recovery.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
    
    def reclaim_pending_messages(
        self,
        stream: str,
        group: str,
        idle_timeout_ms: int = 300000,  # 5 minutes
        claim_to_consumer: Optional[str] = None,
        max_messages: int = 1000
    ) -> Dict[str, Any]:
        """
        Reclaim pending messages from dead consumers.
        
        Process:
        1. Get pending messages summary
        2. For each consumer with pending messages:
           - Check if idle > timeout
           - Claim messages to new consumer
        3. Return statistics
        
        Args:
            stream: Redis stream name
            group: Consumer group name
            idle_timeout_ms: Consider consumer dead after this many ms
            claim_to_consumer: Consumer to claim messages to (default: auto-generated)
            max_messages: Maximum messages to reclaim per consumer
        
        Returns:
            {
                'total_pending': int,
                'consumers_checked': int,
                'dead_consumers': List[str],
                'messages_reclaimed': int,
                'claimed_to': str
            }
        """
        try:
            logger.info(f"[RECOVERY] Checking pending messages in {stream}:{group}")
            
            # Get pending summary
            pending_info = self.redis.xpending(stream, group)
            
            if not pending_info or pending_info.get('pending', 0) == 0:
                logger.info("[RECOVERY] No pending messages found")
                return {
                    'total_pending': 0,
                    'consumers_checked': 0,
                    'dead_consumers': [],
                    'messages_reclaimed': 0,
                    'claimed_to': None
                }
            
            total_pending = pending_info['pending']
            consumers = pending_info.get('consumers', [])
            
            logger.info(f"[RECOVERY] Found {total_pending} pending messages across {len(consumers)} consumers")
            
            # Generate claim target consumer if not provided
            if not claim_to_consumer:
                claim_to_consumer = f"recovery-{int(time.time())}"
            
            dead_consumers = []
            total_reclaimed = 0
            
            # Check each consumer
            for consumer_info in consumers:
                consumer_name = consumer_info['name']
                consumer_pending = consumer_info['pending']
                
                logger.debug(f"[RECOVERY] Checking consumer '{consumer_name}' ({consumer_pending} pending)")
                
                # Get pending messages for this consumer
                pending_msgs = self.redis.xpending_range(
                    stream,
                    group,
                    min='-',
                    max='+',
                    count=max_messages,
                    consumername=consumer_name
                )
                
                if not pending_msgs:
                    continue
                
                # Check if any message is idle longer than timeout
                now_ms = int(time.time() * 1000)
                
                for msg_info in pending_msgs:
                    msg_id = msg_info['message_id']
                    if isinstance(msg_id, bytes):
                        msg_id = msg_id.decode('utf-8')
                    
                    # Get message timestamp from ID (format: {timestamp_ms}-{seq})
                    msg_timestamp_ms = int(msg_id.split('-')[0])
                    idle_ms = now_ms - msg_timestamp_ms
                    
                    if idle_ms > idle_timeout_ms:
                        # Consumer is dead - claim this message
                        try:
                            # XCLAIM: Claim message to new consumer
                            claimed = self.redis.xclaim(
                                stream,
                                group,
                                claim_to_consumer,
                                min_idle_time=idle_timeout_ms,
                                message_ids=[msg_id]
                            )
                            
                            if claimed:
                                total_reclaimed += len(claimed)
                                if consumer_name not in dead_consumers:
                                    dead_consumers.append(consumer_name)
                                    logger.warning(
                                        f"[RECOVERY] Consumer '{consumer_name}' appears dead "
                                        f"(idle {idle_ms}ms > {idle_timeout_ms}ms)"
                                    )
                        
                        except Exception as e:
                            logger.error(f"[RECOVERY] Failed to claim message {msg_id}: {e}")
            
            result = {
                'total_pending': total_pending,
                'consumers_checked': len(consumers),
                'dead_consumers': dead_consumers,
                'messages_reclaimed': total_reclaimed,
                'claimed_to': claim_to_consumer
            }
            
            if total_reclaimed > 0:
                logger.warning(
                    f"[RECOVERY] Reclaimed {total_reclaimed} messages from "
                    f"{len(dead_consumers)} dead consumers to '{claim_to_consumer}'"
                )
            else:
                logger.info("[RECOVERY] No dead consumers found")
            
            return result
        
        except redis.ResponseError as e:
            # Stream or group doesn't exist yet
            logger.debug(f"[RECOVERY] Stream/group doesn't exist yet: {e}")
            return {
                'total_pending': 0,
                'consumers_checked': 0,
                'dead_consumers': [],
                'messages_reclaimed': 0,
                'claimed_to': None
            }
        
        except Exception as e:
            logger.error(f"[RECOVERY] Crash recovery failed: {e}")
            raise
    
    def delete_dead_consumers(
        self,
        stream: str,
        group: str,
        dead_consumers: List[str]
    ) -> int:
        """
        Delete dead consumers from consumer group.
        
        This cleans up the consumer list after reclaiming their messages.
        
        Args:
            stream: Redis stream name
            group: Consumer group name
            dead_consumers: List of consumer names to delete
        
        Returns:
            Number of consumers deleted
        """
        deleted = 0
        
        for consumer_name in dead_consumers:
            try:
                # XGROUP DELCONSUMER: Remove consumer from group
                self.redis.xgroup_delconsumer(stream, group, consumer_name)
                deleted += 1
                logger.info(f"[RECOVERY] Deleted dead consumer '{consumer_name}'")
            
            except Exception as e:
                logger.error(f"[RECOVERY] Failed to delete consumer '{consumer_name}': {e}")
        
        return deleted
    
    def get_recovery_stats(self, stream: str, group: str) -> Dict[str, Any]:
        """
        Get recovery statistics for monitoring.
        
        Args:
            stream: Redis stream name
            group: Consumer group name
        
        Returns:
            {
                'total_pending': int,
                'consumers': List[Dict],
                'oldest_pending_ms': int,
                'recovery_needed': bool
            }
        """
        try:
            pending_info = self.redis.xpending(stream, group)
            
            if not pending_info or pending_info.get('pending', 0) == 0:
                return {
                    'total_pending': 0,
                    'consumers': [],
                    'oldest_pending_ms': 0,
                    'recovery_needed': False
                }
            
            # Get oldest pending message age
            oldest_pending_ms = 0
            if 'min' in pending_info and pending_info['min']:
                msg_id = pending_info['min']
                if isinstance(msg_id, bytes):
                    msg_id = msg_id.decode('utf-8')
                
                msg_timestamp_ms = int(msg_id.split('-')[0])
                now_ms = int(time.time() * 1000)
                oldest_pending_ms = now_ms - msg_timestamp_ms
            
            consumers = [
                {
                    'name': c['name'],
                    'pending': c['pending']
                }
                for c in pending_info.get('consumers', [])
            ]
            
            # Recovery needed if oldest message > 5 minutes
            recovery_needed = oldest_pending_ms > 300000
            
            return {
                'total_pending': pending_info['pending'],
                'consumers': consumers,
                'oldest_pending_ms': oldest_pending_ms,
                'recovery_needed': recovery_needed
            }
        
        except redis.ResponseError:
            return {
                'total_pending': 0,
                'consumers': [],
                'oldest_pending_ms': 0,
                'recovery_needed': False
            }


# ==============================================
# CONVENIENCE FUNCTION
# ==============================================

def run_recovery(
    redis_client: redis.Redis,
    stream: str = "krnx:ltm:queue",
    group: str = "krnx-ltm-workers",
    idle_timeout_ms: int = 300000,
    delete_dead: bool = True
) -> Dict[str, Any]:
    """
    Convenience function: Run full recovery process.
    
    Usage:
        from chillbot.kernel.recovery import run_recovery
        from chillbot.kernel.connection_pool import get_redis_client
        
        redis = get_redis_client()
        result = run_recovery(redis)
    
    Args:
        redis_client: Redis client
        stream: Stream name
        group: Consumer group name
        idle_timeout_ms: Idle timeout in milliseconds
        delete_dead: Delete dead consumers after reclaiming
    
    Returns:
        Recovery statistics
    """
    recovery = CrashRecovery(redis_client)
    
    # Reclaim messages
    result = recovery.reclaim_pending_messages(
        stream=stream,
        group=group,
        idle_timeout_ms=idle_timeout_ms
    )
    
    # Delete dead consumers
    if delete_dead and result['dead_consumers']:
        deleted = recovery.delete_dead_consumers(
            stream=stream,
            group=group,
            dead_consumers=result['dead_consumers']
        )
        result['dead_consumers_deleted'] = deleted
    
    return result


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'CrashRecovery',
    'run_recovery'
]
