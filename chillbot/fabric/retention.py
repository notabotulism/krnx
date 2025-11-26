"""
Memory Fabric - Retention Policies

Defines and evaluates retention policies for memories.
Triggers actions like archival, deletion, consolidation.

Philosophy:
- Policy definitions only (kernel executes)
- Time-based and rule-based triggers
- No automatic execution (fabric triggers, kernel acts)

Usage:
    retention = RetentionManager()
    
    # Define policies
    retention.add_policy(RetentionPolicy(
        name="ephemeral",
        ttl_seconds=3600,  # 1 hour
        action="delete"
    ))
    
    # Evaluate memory against policies
    action = retention.evaluate(memory)
    if action == RetentionAction.DELETE:
        kernel.delete_event(memory.event_id)
"""

import time
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RetentionAction(Enum):
    """Actions that can be triggered by retention policies."""
    KEEP = "keep"           # Keep as-is
    ARCHIVE = "archive"     # Move to cold storage
    DELETE = "delete"       # Delete permanently
    CONSOLIDATE = "consolidate"  # Merge/summarize
    COMPRESS = "compress"   # Compress in place


class RetentionClass(Enum):
    """Pre-defined retention classes (Constitution 6.3)."""
    EPHEMERAL = "ephemeral"   # Short-lived, auto-delete
    STANDARD = "standard"     # Normal retention
    PERMANENT = "permanent"   # Never delete
    AUDIT = "audit"           # Compliance retention


@dataclass
class RetentionPolicy:
    """
    A retention policy definition.
    
    Defines when and how memories should be retained/archived/deleted.
    """
    name: str
    
    # Time-based triggers
    ttl_seconds: Optional[int] = None  # Delete after this many seconds
    archive_after_days: Optional[int] = None  # Archive after this many days
    
    # Action to take
    action: RetentionAction = RetentionAction.KEEP
    
    # Conditions
    retention_class: Optional[RetentionClass] = None  # Match specific class
    channels: Optional[List[str]] = None  # Match specific channels
    workspace_pattern: Optional[str] = None  # Workspace regex pattern
    
    # Priority (higher = evaluated first)
    priority: int = 0
    
    # Metadata
    description: Optional[str] = None
    enabled: bool = True
    
    def matches(self, memory: Any) -> bool:
        """Check if this policy applies to a memory."""
        # Check retention class
        if self.retention_class:
            memory_class = getattr(memory, 'retention_class', None)
            if memory_class != self.retention_class.value:
                return False
        
        # Check channel
        if self.channels:
            memory_channel = getattr(memory, 'channel', None)
            if memory_channel not in self.channels:
                return False
        
        # Check workspace pattern
        if self.workspace_pattern:
            import re
            workspace_id = getattr(memory, 'workspace_id', '')
            if not re.match(self.workspace_pattern, workspace_id):
                return False
        
        return True
    
    def should_trigger(self, memory: Any, now: Optional[float] = None) -> bool:
        """Check if action should be triggered for this memory."""
        now = now or time.time()
        timestamp = getattr(memory, 'timestamp', now)
        age_seconds = now - timestamp
        
        # Check TTL
        if self.ttl_seconds and age_seconds > self.ttl_seconds:
            return True
        
        # Check archive age
        if self.archive_after_days:
            age_days = age_seconds / 86400
            if age_days > self.archive_after_days:
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "ttl_seconds": self.ttl_seconds,
            "archive_after_days": self.archive_after_days,
            "action": self.action.value,
            "retention_class": self.retention_class.value if self.retention_class else None,
            "channels": self.channels,
            "workspace_pattern": self.workspace_pattern,
            "priority": self.priority,
            "description": self.description,
            "enabled": self.enabled,
        }


@dataclass
class RetentionEvaluation:
    """Result of evaluating a memory against retention policies."""
    memory_id: str
    action: RetentionAction
    policy_name: Optional[str] = None
    reason: Optional[str] = None
    scheduled_at: Optional[float] = None  # When action should occur
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "action": self.action.value,
            "policy_name": self.policy_name,
            "reason": self.reason,
            "scheduled_at": self.scheduled_at,
        }


class RetentionManager:
    """
    Manage retention policies and evaluate memories.
    
    This is the "brain" of retention - it defines policies
    and evaluates memories, but doesn't execute actions.
    The fabric/kernel is responsible for execution.
    """
    
    def __init__(self):
        """Initialize retention manager."""
        self._policies: Dict[str, RetentionPolicy] = {}
        
        # Add default policies
        self._add_default_policies()
        
        logger.info("[RETENTION] Initialized with default policies")
    
    def _add_default_policies(self):
        """Add default retention policies (Constitution 6.3)."""
        # Ephemeral: Delete after 1 hour
        self.add_policy(RetentionPolicy(
            name="ephemeral_cleanup",
            retention_class=RetentionClass.EPHEMERAL,
            ttl_seconds=3600,
            action=RetentionAction.DELETE,
            priority=100,
            description="Delete ephemeral memories after 1 hour",
        ))
        
        # Standard: Archive after 30 days
        self.add_policy(RetentionPolicy(
            name="standard_archive",
            retention_class=RetentionClass.STANDARD,
            archive_after_days=30,
            action=RetentionAction.ARCHIVE,
            priority=50,
            description="Archive standard memories after 30 days",
        ))
        
        # Permanent: Never delete
        self.add_policy(RetentionPolicy(
            name="permanent_keep",
            retention_class=RetentionClass.PERMANENT,
            action=RetentionAction.KEEP,
            priority=200,  # High priority to override others
            description="Never delete permanent memories",
        ))
        
        # Audit: Keep for 7 years (compliance)
        self.add_policy(RetentionPolicy(
            name="audit_compliance",
            retention_class=RetentionClass.AUDIT,
            archive_after_days=30,
            action=RetentionAction.ARCHIVE,
            priority=150,
            description="Archive audit logs, keep for compliance",
        ))
    
    # ==============================================
    # POLICY MANAGEMENT
    # ==============================================
    
    def add_policy(self, policy: RetentionPolicy):
        """Add or update a retention policy."""
        self._policies[policy.name] = policy
        logger.debug(f"[RETENTION] Added policy: {policy.name}")
    
    def remove_policy(self, name: str):
        """Remove a retention policy."""
        if name in self._policies:
            del self._policies[name]
            logger.debug(f"[RETENTION] Removed policy: {name}")
    
    def get_policy(self, name: str) -> Optional[RetentionPolicy]:
        """Get a policy by name."""
        return self._policies.get(name)
    
    def list_policies(self, enabled_only: bool = True) -> List[RetentionPolicy]:
        """List all policies, sorted by priority."""
        policies = list(self._policies.values())
        
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        
        # Sort by priority (highest first)
        policies.sort(key=lambda p: p.priority, reverse=True)
        
        return policies
    
    # ==============================================
    # EVALUATION
    # ==============================================
    
    def evaluate(
        self,
        memory: Any,
        now: Optional[float] = None,
    ) -> RetentionEvaluation:
        """
        Evaluate a memory against all policies.
        
        Returns the action to take (from highest priority matching policy).
        
        Args:
            memory: Memory object with timestamp, retention_class, etc.
            now: Current time (defaults to time.time())
        
        Returns:
            RetentionEvaluation with recommended action
        """
        now = now or time.time()
        memory_id = getattr(memory, 'event_id', str(id(memory)))
        
        # Check memory's own TTL first (Constitution 6.3)
        memory_ttl = getattr(memory, 'ttl_seconds', None)
        if memory_ttl:
            timestamp = getattr(memory, 'timestamp', now)
            age_seconds = now - timestamp
            if age_seconds > memory_ttl:
                return RetentionEvaluation(
                    memory_id=memory_id,
                    action=RetentionAction.DELETE,
                    policy_name="memory_ttl",
                    reason=f"TTL exceeded ({age_seconds:.0f}s > {memory_ttl}s)",
                )
        
        # Evaluate policies in priority order
        for policy in self.list_policies(enabled_only=True):
            if policy.matches(memory) and policy.should_trigger(memory, now):
                return RetentionEvaluation(
                    memory_id=memory_id,
                    action=policy.action,
                    policy_name=policy.name,
                    reason=policy.description,
                )
        
        # Default: keep
        return RetentionEvaluation(
            memory_id=memory_id,
            action=RetentionAction.KEEP,
            policy_name=None,
            reason="No matching policy",
        )
    
    def evaluate_batch(
        self,
        memories: List[Any],
        now: Optional[float] = None,
    ) -> List[RetentionEvaluation]:
        """Evaluate multiple memories."""
        now = now or time.time()
        return [self.evaluate(m, now) for m in memories]
    
    def find_expired(
        self,
        memories: List[Any],
        now: Optional[float] = None,
    ) -> List[RetentionEvaluation]:
        """Find memories that need action (expired, need archival, etc.)."""
        evaluations = self.evaluate_batch(memories, now)
        return [e for e in evaluations if e.action != RetentionAction.KEEP]
    
    # ==============================================
    # SCHEDULING
    # ==============================================
    
    def schedule_evaluation(
        self,
        memory: Any,
        now: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate when this memory should next be evaluated.
        
        Returns timestamp of next evaluation, or None if no future evaluation needed.
        """
        now = now or time.time()
        timestamp = getattr(memory, 'timestamp', now)
        
        next_check = None
        
        # Check memory TTL
        memory_ttl = getattr(memory, 'ttl_seconds', None)
        if memory_ttl:
            ttl_expiry = timestamp + memory_ttl
            if ttl_expiry > now:
                next_check = ttl_expiry
        
        # Check policies
        for policy in self.list_policies(enabled_only=True):
            if not policy.matches(memory):
                continue
            
            # Calculate when this policy would trigger
            if policy.ttl_seconds:
                trigger_time = timestamp + policy.ttl_seconds
                if trigger_time > now:
                    if next_check is None or trigger_time < next_check:
                        next_check = trigger_time
            
            if policy.archive_after_days:
                trigger_time = timestamp + (policy.archive_after_days * 86400)
                if trigger_time > now:
                    if next_check is None or trigger_time < next_check:
                        next_check = trigger_time
        
        return next_check
    
    # ==============================================
    # UTILITIES
    # ==============================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retention manager statistics."""
        policies = self.list_policies(enabled_only=False)
        enabled = [p for p in policies if p.enabled]
        
        return {
            "total_policies": len(policies),
            "enabled_policies": len(enabled),
            "policy_names": [p.name for p in enabled],
        }
    
    def explain_policy(self, name: str) -> Optional[str]:
        """Get human-readable explanation of a policy."""
        policy = self.get_policy(name)
        if not policy:
            return None
        
        parts = [f"Policy: {policy.name}"]
        
        if policy.description:
            parts.append(f"  Description: {policy.description}")
        
        if policy.retention_class:
            parts.append(f"  Applies to: {policy.retention_class.value} class")
        
        if policy.channels:
            parts.append(f"  Channels: {', '.join(policy.channels)}")
        
        if policy.ttl_seconds:
            hours = policy.ttl_seconds / 3600
            parts.append(f"  TTL: {hours:.1f} hours")
        
        if policy.archive_after_days:
            parts.append(f"  Archive after: {policy.archive_after_days} days")
        
        parts.append(f"  Action: {policy.action.value}")
        parts.append(f"  Priority: {policy.priority}")
        parts.append(f"  Enabled: {policy.enabled}")
        
        return "\n".join(parts)


# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

def get_retention_class(name: str) -> Optional[RetentionClass]:
    """Get RetentionClass by name."""
    try:
        return RetentionClass(name.lower())
    except ValueError:
        return None


def create_ttl_policy(
    name: str,
    ttl_seconds: int,
    channels: Optional[List[str]] = None,
) -> RetentionPolicy:
    """Create a simple TTL-based deletion policy."""
    return RetentionPolicy(
        name=name,
        ttl_seconds=ttl_seconds,
        action=RetentionAction.DELETE,
        channels=channels,
        description=f"Delete after {ttl_seconds}s",
    )


def create_archive_policy(
    name: str,
    days: int,
    retention_class: Optional[RetentionClass] = None,
) -> RetentionPolicy:
    """Create a simple archive policy."""
    return RetentionPolicy(
        name=name,
        archive_after_days=days,
        action=RetentionAction.ARCHIVE,
        retention_class=retention_class,
        description=f"Archive after {days} days",
    )
