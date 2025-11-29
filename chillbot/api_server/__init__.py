"""
KRNX API Server

FastAPI-based REST API for the KRNX temporal memory kernel.

Basic API:
- Events CRUD
- Temporal operations (state, replay, timeline)
- Health and stats

Advanced API:
- Provenance (hash chain verification)
- Supersession (fact versioning)
- Context (LLM-ready retrieval)
- Agents (multi-agent coordination)
- Branches (workflow branching)
"""

from .main import app
from .config import Settings, get_settings

__all__ = ["app", "Settings", "get_settings"]
__version__ = "0.3.10"
