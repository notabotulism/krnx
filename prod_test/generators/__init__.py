"""
Test Data Generators

Utilities for generating test data:
- events: Event factory with various patterns
- haystack: Needle-in-haystack test scenarios
- conversations: Realistic conversation generation
"""

from .events import EventGenerator

__all__ = ['EventGenerator']
