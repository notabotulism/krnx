"""
Metrics Collection & Statistical Validation

Tools for collecting test metrics and validating results:
- collector: Metrics collection and aggregation
- statistical: Effect size, confidence intervals, significance tests
"""

from .collector import MetricsCollector

__all__ = ['MetricsCollector']
