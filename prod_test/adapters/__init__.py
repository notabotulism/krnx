"""
Test Adapters

Protocol-based adapters for comparing KRNX against baselines:
- KRNXAdapter: Full KRNX kernel
- NaiveRAGAdapter: Simple vector-only retrieval
- NoMemoryAdapter: Stateless baseline

Used in Layer 3 demos for scientific comparison.
"""

from .base import BaseAdapter
from .krnx_adapter import KRNXAdapter

__all__ = ['BaseAdapter', 'KRNXAdapter']
