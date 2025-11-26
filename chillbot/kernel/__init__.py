"""
KRNX Kernel - Temporal Memory Infrastructure

The foundational layer. Append-only event store with:
- STM (Redis/KeyDB) - hot storage, 0-24h
- LTM (SQLite WAL) - warm storage, 0-30d  
- Archival (compressed SQLite) - cold storage, 30d+
- Replay engine - temporal queries
- Hash-chain provenance - integrity verification

Usage:
    # Local mode (embedded kernel)
    from chillbot.kernel import KRNXClient
    
    client = KRNXClient(data_path="./data", redis_host="localhost")
    client.remember(content="User loves hiking")
    events = client.recall(workspace="default")
    
    # Direct controller access (power users)
    from chillbot.kernel import KRNXController, Event
    
    krnx = KRNXController(data_path="./data")
    event = Event(
        event_id="evt_123",
        workspace_id="default",
        user_id="user_1",
        session_id="session_1",
        content={"message": "hello"},
        timestamp=time.time()
    )
    krnx.write_event("default", "user_1", event)
    
    # Remote mode (HTTP API)
    client = KRNXClient(base_url="http://localhost:6380")
"""

__version__ = "1.0.0"

# Client (unified access)
from chillbot.kernel.client import KRNXClient, AsyncKRNXClient

# Controller (direct access)
from chillbot.kernel.controller import (
    KRNXController,
    create_krnx,
    WorkerMetrics,
    RetrievalTelemetry,
)

# Models
from chillbot.kernel.models import (
    Event,
    create_event,
    validate_event_id,
    validate_workspace_id,
    validate_user_id,
)

# Connection pool
from chillbot.kernel.connection_pool import (
    configure_pool,
    get_redis_client,
    close_pool,
)

# Exceptions
from chillbot.kernel.exceptions import (
    KRNXError,
    RedisUnavailableError,
    BackpressureError,
    NotFoundError,
    ValidationError,
    StorageError,
    IntegrityError,
    LTMStorageError,
    LTMArchivalError,
)

# Recovery
from chillbot.kernel.recovery import CrashRecovery, run_recovery

__all__ = [
    # Client
    "KRNXClient",
    "AsyncKRNXClient",
    
    # Controller
    "KRNXController",
    "create_krnx",
    "WorkerMetrics",
    "RetrievalTelemetry",
    
    # Models
    "Event",
    "create_event",
    "validate_event_id",
    "validate_workspace_id",
    "validate_user_id",
    
    # Connection pool
    "configure_pool",
    "get_redis_client",
    "close_pool",
    
    # Exceptions
    "KRNXError",
    "RedisUnavailableError",
    "BackpressureError",
    "NotFoundError",
    "ValidationError",
    "StorageError",
    "IntegrityError",
    "LTMStorageError",
    "LTMArchivalError",
    
    # Recovery
    "CrashRecovery",
    "run_recovery",
]
