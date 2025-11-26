"""
Memory Fabric - Orchestration Layer

Routes requests across Kernel and Compute. Provides:
- Identity resolution (agents, scopes)
- Context building (assemble memories for LLM)
- Retention policy triggers
- Read orchestration

This is the "smart routing" layer - but no opinions, just orchestration.

Usage:
    from chillbot.fabric import MemoryFabric, ContextBuilder
    
    fabric = MemoryFabric(kernel=krnx, vectors=vectors, embeddings=embeddings)
    
    # Store (routes to kernel + enqueues embedding job)
    event_id = fabric.remember(content="User loves hiking")
    
    # Recall (searches vectors, enriches from kernel)
    memories = fabric.recall(query="outdoor activities")
    
    # Build LLM context
    context = fabric.context(query="plan a trip", max_tokens=4000)
"""

__version__ = "0.1.0"

# Orchestrator
from chillbot.fabric.orchestrator import (
    MemoryFabric,
    MemoryItem,
    RecallResult,
)

# Context Builder
from chillbot.fabric.context import (
    ContextBuilder,
    ContextConfig,
    build_context,
)

# Identity Resolution
from chillbot.fabric.identity import (
    IdentityResolver,
    Identity,
    IdentityType,
    AgentRegistration,
    parse_scope,
)

# Retention Policies
from chillbot.fabric.retention import (
    RetentionManager,
    RetentionPolicy,
    RetentionAction,
    RetentionClass,
    RetentionEvaluation,
    create_ttl_policy,
    create_archive_policy,
)

__all__ = [
    # Orchestrator
    "MemoryFabric",
    "MemoryItem",
    "RecallResult",
    
    # Context
    "ContextBuilder",
    "ContextConfig",
    "build_context",
    
    # Identity
    "IdentityResolver",
    "Identity",
    "IdentityType",
    "AgentRegistration",
    "parse_scope",
    
    # Retention
    "RetentionManager",
    "RetentionPolicy",
    "RetentionAction",
    "RetentionClass",
    "RetentionEvaluation",
    "create_ttl_policy",
    "create_archive_policy",
]
