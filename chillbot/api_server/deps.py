"""
KRNX API Dependencies

Dependency injection for controller and fabric instances.
"""

from typing import Optional, Generator
from functools import lru_cache
import time

from .config import get_settings, Settings

# These will be initialized at startup
_controller = None
_fabric = None
_start_time: float = 0.0


def get_controller():
    """
    Get the KRNX controller instance.
    
    The controller is initialized once at startup and reused.
    """
    global _controller
    if _controller is None:
        raise RuntimeError("Controller not initialized. Call init_dependencies() first.")
    return _controller


def get_fabric():
    """
    Get the Memory Fabric instance.
    
    The fabric is initialized once at startup and reused.
    """
    global _fabric
    if _fabric is None:
        raise RuntimeError("Fabric not initialized. Call init_dependencies() first.")
    return _fabric


def get_uptime() -> float:
    """Get server uptime in seconds."""
    global _start_time
    return time.time() - _start_time


def init_dependencies(settings: Optional[Settings] = None):
    """
    Initialize controller and fabric.
    
    Called once at server startup.
    """
    global _controller, _fabric, _start_time
    
    if settings is None:
        settings = get_settings()
    
    _start_time = time.time()
    
    # Import here to avoid circular imports
    from chillbot.kernel.controller import KRNXController
    from chillbot.fabric.orchestrator import MemoryFabric
    
    print(f"[INIT] Initializing KRNX API server...")
    print(f"[INIT] Redis: {settings.redis_host}:{settings.redis_port}")
    print(f"[INIT] Data path: {settings.data_path}")
    
    # Initialize controller
    _controller = KRNXController(
        data_path=settings.data_path,
        redis_host=settings.redis_host,
        redis_port=settings.redis_port,
        redis_password=settings.redis_password,
        redis_max_connections=settings.redis_max_connections,
        enable_backpressure=settings.enable_backpressure,
        enable_hash_chain=settings.enable_hash_chain,
        max_queue_depth=settings.max_queue_depth,
        max_lag_seconds=settings.max_lag_seconds,
        warm_retention_days=settings.warm_retention_days,
        enable_async_worker=True,
    )
    
    # Initialize fabric with kernel
    _fabric = MemoryFabric(
        kernel=_controller,
        default_workspace=settings.default_workspace,
        auto_embed=settings.auto_embed,
        auto_enrich=settings.auto_enrich,
    )
    
    print(f"[OK] KRNX API server initialized")
    return _controller, _fabric


def shutdown_dependencies():
    """
    Shutdown controller and fabric.
    
    Called at server shutdown.
    """
    global _controller, _fabric
    
    print(f"[SHUTDOWN] Shutting down KRNX API server...")
    
    if _fabric:
        _fabric.close()
        _fabric = None
    
    if _controller:
        _controller.shutdown(close_connection_pool=True)
        _controller = None
    
    print(f"[OK] KRNX API server shutdown complete")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def event_to_response(event, score: Optional[float] = None) -> dict:
    """
    Convert kernel Event to API response dict.
    
    Handles the mapping between internal Event dataclass
    and the API EventResponse schema.
    """
    return {
        "event_id": event.event_id,
        "workspace_id": event.workspace_id,
        "user_id": event.user_id,
        "session_id": event.session_id,
        "content": event.content,
        "channel": event.channel,
        "event_type": event.metadata.get("event_type", "message") if event.metadata else "message",
        "timestamp": event.timestamp,
        "created_at": event.created_at,
        "provenance": {
            "hash": event.compute_hash() if hasattr(event, 'compute_hash') else None,
            "previous_hash": event.previous_hash,
        },
        "relations": event.metadata.get("relations", []) if event.metadata else [],
        "enrichment": {
            k: v for k, v in (event.metadata or {}).items()
            if k in ["salience_score", "episode_id", "time_gap_seconds", "entities", "topics"]
        } or None,
        "metadata": {
            k: v for k, v in (event.metadata or {}).items()
            if k not in ["relations", "salience_score", "episode_id", "time_gap_seconds", "entities", "topics", "event_type"]
        } or None,
        "score": score,
    }


def events_to_response(events, scores: Optional[list] = None) -> list:
    """Convert list of kernel Events to API response dicts."""
    if scores:
        return [event_to_response(e, s) for e, s in zip(events, scores)]
    return [event_to_response(e) for e in events]
