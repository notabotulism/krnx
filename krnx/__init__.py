"""
KRNX - Temporal Memory Fabric for AI Agents

Reconstruct any moment. Debug any decision.
Durable memory infrastructure in one import.

Usage:
    from krnx import KRNXClient
    
    client = KRNXClient(data_path="./krnx-data", redis_host="localhost")
    client.remember(content="User prefers dark mode", workspace="app")
    events = client.recall(workspace="app", limit=10)

Links:
    Docs: https://krnx.dev/docs
    GitHub: https://github.com/chillbot/krnx
"""

__version__ = "0.3.10"

# Re-export from internal package
from chillbot.kernel import (
    # Client
    KRNXClient,
    AsyncKRNXClient,
    
    # Models
    Event,
    
    # Controller (power users)
    KRNXController,
    create_krnx,
    
    # Exceptions
    KRNXError,
    BackpressureError,
    NotFoundError,
    ValidationError,
    
    # Recovery
    CrashRecovery,
    run_recovery,
)

__all__ = [
    "KRNXClient",
    "AsyncKRNXClient",
    "Event",
    "KRNXController",
    "create_krnx",
    "KRNXError",
    "BackpressureError",
    "NotFoundError",
    "ValidationError",
    "CrashRecovery",
    "run_recovery",
]