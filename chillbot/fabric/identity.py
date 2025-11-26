"""
Memory Fabric - Identity Resolver

Resolves agent and scope identities for multi-tenant/multi-agent systems.
Maps abstract identities to concrete workspace/user/session IDs.

Philosophy:
- Flexible identity schemes
- Hierarchical scopes
- No authentication (that's app layer)

Usage:
    resolver = IdentityResolver()
    
    # Register identity mappings
    resolver.register_agent("coder-1", workspace="project-x", role="coder")
    
    # Resolve identity
    identity = resolver.resolve("coder-1")
    print(identity.workspace_id)  # "project-x"
    
    # Scope-based resolution
    identity = resolver.resolve_scope("org:acme/project:alpha/user:bob")
"""

import re
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IdentityType(Enum):
    """Types of identities."""
    USER = "user"
    AGENT = "agent"
    SERVICE = "service"
    SYSTEM = "system"


@dataclass
class Identity:
    """Resolved identity with scope information."""
    id: str
    type: IdentityType
    workspace_id: str
    user_id: str
    session_id: Optional[str] = None
    
    # Additional attributes
    role: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Hierarchical scope
    org_id: Optional[str] = None
    project_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "permissions": self.permissions,
            "metadata": self.metadata,
            "org_id": self.org_id,
            "project_id": self.project_id,
        }
    
    @property
    def scope_path(self) -> str:
        """Get hierarchical scope path."""
        parts = []
        if self.org_id:
            parts.append(f"org:{self.org_id}")
        if self.project_id:
            parts.append(f"project:{self.project_id}")
        parts.append(f"workspace:{self.workspace_id}")
        parts.append(f"user:{self.user_id}")
        return "/".join(parts)


@dataclass
class AgentRegistration:
    """Agent registration information."""
    agent_id: str
    workspace_id: str
    role: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdentityResolver:
    """
    Resolve identities for memory operations.
    
    Supports:
    - Direct ID resolution (user_123 → Identity)
    - Scope-based resolution (org:acme/project:alpha → Identity)
    - Agent registration and lookup
    - Default identity fallback
    """
    
    def __init__(
        self,
        default_workspace: str = "default",
        default_user: str = "default",
    ):
        """
        Initialize identity resolver.
        
        Args:
            default_workspace: Default workspace for unresolved identities
            default_user: Default user for unresolved identities
        """
        self.default_workspace = default_workspace
        self.default_user = default_user
        
        # Registered agents
        self._agents: Dict[str, AgentRegistration] = {}
        
        # Scope mappings (scope_path → workspace_id)
        self._scope_mappings: Dict[str, str] = {}
        
        logger.info(f"[IDENTITY] Initialized (default_workspace={default_workspace})")
    
    # ==============================================
    # AGENT REGISTRATION
    # ==============================================
    
    def register_agent(
        self,
        agent_id: str,
        workspace_id: str,
        role: Optional[str] = None,
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Register an agent identity.
        
        Args:
            agent_id: Unique agent identifier
            workspace_id: Workspace this agent belongs to
            role: Agent role (e.g., 'coder', 'reviewer')
            permissions: List of permissions
            metadata: Additional metadata
        """
        self._agents[agent_id] = AgentRegistration(
            agent_id=agent_id,
            workspace_id=workspace_id,
            role=role,
            permissions=permissions or [],
            metadata=metadata or {},
        )
        
        logger.debug(f"[IDENTITY] Registered agent: {agent_id} → {workspace_id}")
    
    def unregister_agent(self, agent_id: str):
        """Remove agent registration."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.debug(f"[IDENTITY] Unregistered agent: {agent_id}")
    
    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration."""
        return self._agents.get(agent_id)
    
    def list_agents(self, workspace_id: Optional[str] = None) -> List[AgentRegistration]:
        """List registered agents, optionally filtered by workspace."""
        agents = list(self._agents.values())
        
        if workspace_id:
            agents = [a for a in agents if a.workspace_id == workspace_id]
        
        return agents
    
    # ==============================================
    # SCOPE MAPPING
    # ==============================================
    
    def register_scope(self, scope_path: str, workspace_id: str):
        """
        Register a scope to workspace mapping.
        
        Args:
            scope_path: Hierarchical scope (e.g., "org:acme/project:alpha")
            workspace_id: Workspace ID to map to
        """
        self._scope_mappings[scope_path] = workspace_id
        logger.debug(f"[IDENTITY] Registered scope: {scope_path} → {workspace_id}")
    
    def unregister_scope(self, scope_path: str):
        """Remove scope mapping."""
        if scope_path in self._scope_mappings:
            del self._scope_mappings[scope_path]
    
    # ==============================================
    # RESOLUTION
    # ==============================================
    
    def resolve(
        self,
        id: str,
        type_hint: Optional[IdentityType] = None,
    ) -> Identity:
        """
        Resolve an identity by ID.
        
        Resolution order:
        1. Check if it's a registered agent
        2. Check if it's a scope path
        3. Parse as direct identity
        4. Fall back to defaults
        
        Args:
            id: Identity string to resolve
            type_hint: Optional type hint
        
        Returns:
            Resolved Identity
        """
        # 1. Check agents
        if id in self._agents:
            agent = self._agents[id]
            return Identity(
                id=id,
                type=IdentityType.AGENT,
                workspace_id=agent.workspace_id,
                user_id=f"agent:{id}",
                role=agent.role,
                permissions=agent.permissions,
                metadata=agent.metadata,
            )
        
        # 2. Check scope paths
        if "/" in id or ":" in id:
            return self.resolve_scope(id)
        
        # 3. Direct resolution (assume user)
        return Identity(
            id=id,
            type=type_hint or IdentityType.USER,
            workspace_id=self.default_workspace,
            user_id=id,
        )
    
    def resolve_scope(self, scope_path: str) -> Identity:
        """
        Resolve a hierarchical scope path.
        
        Scope format: "org:acme/project:alpha/user:bob"
        
        Args:
            scope_path: Hierarchical scope path
        
        Returns:
            Resolved Identity
        """
        # Parse scope components
        org_id = None
        project_id = None
        workspace_id = self.default_workspace
        user_id = self.default_user
        
        parts = scope_path.split("/")
        
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                
                if key == "org":
                    org_id = value
                elif key == "project":
                    project_id = value
                elif key == "workspace":
                    workspace_id = value
                elif key == "user":
                    user_id = value
        
        # Check scope mappings for workspace override
        for prefix, mapped_workspace in self._scope_mappings.items():
            if scope_path.startswith(prefix):
                workspace_id = mapped_workspace
                break
        
        # Build workspace from hierarchy if not explicitly set
        if workspace_id == self.default_workspace and project_id:
            if org_id:
                workspace_id = f"{org_id}_{project_id}"
            else:
                workspace_id = project_id
        
        return Identity(
            id=scope_path,
            type=IdentityType.USER,
            workspace_id=workspace_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
        )
    
    def resolve_workspace(
        self,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> str:
        """
        Resolve workspace ID from various inputs.
        
        Args:
            workspace_id: Explicit workspace ID
            org_id: Organization ID
            project_id: Project ID
        
        Returns:
            Resolved workspace ID
        """
        if workspace_id:
            return workspace_id
        
        if project_id:
            if org_id:
                return f"{org_id}_{project_id}"
            return project_id
        
        if org_id:
            return org_id
        
        return self.default_workspace
    
    # ==============================================
    # VALIDATION
    # ==============================================
    
    def validate_identity(self, identity: Identity) -> List[str]:
        """
        Validate an identity.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        if not identity.workspace_id:
            errors.append("workspace_id is required")
        
        if not identity.user_id:
            errors.append("user_id is required")
        
        if len(identity.workspace_id) > 255:
            errors.append("workspace_id too long (max 255)")
        
        if len(identity.user_id) > 255:
            errors.append("user_id too long (max 255)")
        
        # Validate characters (alphanumeric, underscore, dash)
        valid_pattern = re.compile(r'^[a-zA-Z0-9_\-:]+$')
        
        if not valid_pattern.match(identity.workspace_id):
            errors.append("workspace_id contains invalid characters")
        
        if not valid_pattern.match(identity.user_id):
            errors.append("user_id contains invalid characters")
        
        return errors
    
    # ==============================================
    # UTILITIES
    # ==============================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resolver statistics."""
        return {
            "registered_agents": len(self._agents),
            "scope_mappings": len(self._scope_mappings),
            "default_workspace": self.default_workspace,
            "default_user": self.default_user,
        }
    
    def clear(self):
        """Clear all registrations."""
        self._agents.clear()
        self._scope_mappings.clear()
        logger.info("[IDENTITY] Cleared all registrations")


def parse_scope(scope_string: str) -> Dict[str, str]:
    """
    Parse a scope string into components.
    
    Example: "org:acme/project:alpha/user:bob"
    Returns: {"org": "acme", "project": "alpha", "user": "bob"}
    """
    result = {}
    
    parts = scope_string.split("/")
    for part in parts:
        if ":" in part:
            key, value = part.split(":", 1)
            result[key] = value
    
    return result
