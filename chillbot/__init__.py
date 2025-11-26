"""
Chillbot - AI Memory Infrastructure

The substrate for AI agents that remember.

Quick Start:
    from chillbot import Memory
    
    memory = Memory("my-agent")
    memory.remember("user likes hiking")
    memories = memory.recall("outdoor activities")
    context = memory.context("plan a weekend trip")

Components:
    - Memory: Simple, "it just works" interface
    - Kernel: Event store, STM/LTM, replay engine (KRNX)
    - Compute: Embeddings, vectors, salience scoring
    - Fabric: Orchestration, context building, identity

For Infrastructure Access:
    from chillbot.compute import JobQueue, EmbeddingEngine, VectorStore
    from chillbot.kernel import KRNXClient, KRNXController, Event
    from chillbot.fabric import MemoryFabric, ContextBuilder
"""

__version__ = "0.1.0"
__author__ = "Ben (ChillBot)"

# Top-level exports
from chillbot.memory import Memory

# Kernel exports (for power users)
from chillbot.kernel import (
    KRNXClient,
    AsyncKRNXClient,
    KRNXController,
    Event,
    create_event,
)

# Fabric exports
from chillbot.fabric import (
    MemoryFabric,
    ContextBuilder,
    IdentityResolver,
)

__all__ = [
    # Main interface
    "Memory",
    
    # Kernel
    "KRNXClient",
    "AsyncKRNXClient",
    "KRNXController",
    "Event",
    "create_event",
    
    # Fabric
    "MemoryFabric",
    "ContextBuilder",
    "IdentityResolver",
]
