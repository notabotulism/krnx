"""
krnx — Git for ML Agent State

Record. Branch. Replay. Verify.
Put your agents in production with confidence.
"""

from .substrate import Substrate, Event, IntegrityError, init

__version__ = "0.1.0"
__all__ = ["Substrate", "Event", "IntegrityError", "init"]

def studio(workspace=None):
    """Launch Krnx Studio TUI."""
    from .studio import run_studio
    run_studio(workspace)
