"""
KRNX API Routes

Basic routes: CRUD + Temporal (core kernel operations)
Advanced routes: Full 30 endpoints (fabric, agents, branches, etc.)
"""

from .basic import router as basic_router
from .advanced import router as advanced_router

__all__ = ["basic_router", "advanced_router"]
