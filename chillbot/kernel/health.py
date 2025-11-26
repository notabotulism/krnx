"""
KRNX Health Check System - Enhanced Production Monitoring

Comprehensive health checks with:
- Configurable thresholds
- Actionable error messages
- Degradation detection
- Automated recovery suggestions

Usage:
    from chillbot.kernel.health import HealthChecker
    
    checker = HealthChecker(krnx)
    report = checker.full_health_check()
    
    if not report.is_healthy():
        print(report.get_actions())
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


# ==============================================
# HEALTH STATUS
# ==============================================

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# ==============================================
# CHECK RESULTS
# ==============================================

@dataclass
class HealthCheckResult:
    """Result from a single health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    action: Optional[str] = None  # What to do if unhealthy
    
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Complete health report"""
    timestamp: float
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    
    def is_healthy(self) -> bool:
        return self.overall_status == HealthStatus.HEALTHY
    
    def is_degraded(self) -> bool:
        return self.overall_status == HealthStatus.DEGRADED
    
    def is_critical(self) -> bool:
        return self.overall_status == HealthStatus.CRITICAL
    
    def get_actions(self) -> List[str]:
        """Get list of recommended actions"""
        actions = []
        for check in self.checks:
            if check.action and not check.is_healthy():
                actions.append(f"[{check.name}] {check.action}")
        return actions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'overall_status': self.overall_status.value,
            'healthy': self.is_healthy(),
            'checks': [
                {
                    'name': c.name,
                    'status': c.status.value,
                    'message': c.message,
                    'details': c.details,
                    'action': c.action
                }
                for c in self.checks
            ],
            'actions': self.get_actions()
        }


# ==============================================
# THRESHOLDS (Configurable)
# ==============================================

@dataclass
class HealthThresholds:
    """Configurable health check thresholds"""
    
    # Queue thresholds
    queue_depth_warning: int = 5000
    queue_depth_critical: int = 10000
    
    # Lag thresholds
    lag_warning_seconds: float = 30.0
    lag_critical_seconds: float = 60.0
    
    # Error rate thresholds
    error_rate_warning: int = 50
    error_rate_critical: int = 100
    
    # Memory thresholds (MB)
    redis_memory_warning_mb: float = 1000.0
    redis_memory_critical_mb: float = 2000.0
    
    # Storage thresholds (MB)
    ltm_size_warning_mb: float = 10000.0  # 10 GB
    ltm_size_critical_mb: float = 50000.0  # 50 GB


# ==============================================
# HEALTH CHECKER
# ==============================================

class HealthChecker:
    """
    Comprehensive health checker with actionable diagnostics.
    """
    
    def __init__(
        self,
        krnx,
        thresholds: Optional[HealthThresholds] = None
    ):
        """
        Initialize health checker.
        
        Args:
            krnx: KRNXController instance
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.krnx = krnx
        self.thresholds = thresholds or HealthThresholds()
    
    def full_health_check(self) -> HealthReport:
        """
        Run complete health check suite.
        
        Returns:
            HealthReport with all check results
        """
        checks = [
            self._check_redis_connection(),
            self._check_ltm_integrity(),
            self._check_worker_running(),
            self._check_queue_depth(),
            self._check_worker_lag(),
            self._check_error_rate(),
            self._check_redis_memory(),
            self._check_ltm_storage()
        ]
        
        # Determine overall status
        if any(c.status == HealthStatus.CRITICAL for c in checks):
            overall_status = HealthStatus.CRITICAL
        elif any(c.status == HealthStatus.DEGRADED for c in checks):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return HealthReport(
            timestamp=time.time(),
            overall_status=overall_status,
            checks=checks
        )
    
    # ==============================================
    # INDIVIDUAL CHECKS
    # ==============================================
    
    def _check_redis_connection(self) -> HealthCheckResult:
        """Check Redis (STM) connectivity"""
        try:
            stats = self.krnx.stm.get_stats()
            
            if stats.get('connected'):
                return HealthCheckResult(
                    name="redis_connection",
                    status=HealthStatus.HEALTHY,
                    message="Redis connected",
                    details={
                        'uptime_seconds': stats.get('uptime_seconds', 0),
                        'total_keys': stats.get('total_keys', 0)
                    }
                )
            else:
                return HealthCheckResult(
                    name="redis_connection",
                    status=HealthStatus.CRITICAL,
                    message="Redis not connected",
                    details={},
                    action="Check Redis is running: docker-compose up -d redis"
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="redis_connection",
                status=HealthStatus.CRITICAL,
                message=f"Redis connection error: {e}",
                details={'error': str(e)},
                action="Verify Redis host/port in configuration"
            )
    
    def _check_ltm_integrity(self) -> HealthCheckResult:
        """Check LTM (SQLite) database integrity"""
        try:
            integrity = self.krnx.ltm.verify_integrity()
            
            if integrity.get('healthy'):
                return HealthCheckResult(
                    name="ltm_integrity",
                    status=HealthStatus.HEALTHY,
                    message="LTM databases healthy",
                    details={
                        'warm_events': integrity['warm_tier']['event_count'],
                        'cold_events': integrity['cold_tier']['event_count'],
                        'total_size_mb': integrity['total_size_mb']
                    }
                )
            else:
                return HealthCheckResult(
                    name="ltm_integrity",
                    status=HealthStatus.CRITICAL,
                    message="LTM database corruption detected",
                    details=integrity,
                    action="Run: krnx doctor --fix-integrity (if available) or restore from backup"
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="ltm_integrity",
                status=HealthStatus.CRITICAL,
                message=f"LTM integrity check failed: {e}",
                details={'error': str(e)},
                action="Check data directory permissions and disk space"
            )
    
    def _check_worker_running(self) -> HealthCheckResult:
        """Check if async worker is running"""
        running = self.krnx._worker_running
        
        if running:
            return HealthCheckResult(
                name="worker_running",
                status=HealthStatus.HEALTHY,
                message="Async worker running",
                details={'thread_alive': self.krnx._ltm_worker_thread.is_alive()}
            )
        else:
            return HealthCheckResult(
                name="worker_running",
                status=HealthStatus.CRITICAL,
                message="Async worker not running",
                details={},
                action="Restart KRNX with enable_async_worker=True"
            )
    
    def _check_queue_depth(self) -> HealthCheckResult:
        """Check LTM queue depth"""
        try:
            metrics = self.krnx.get_worker_metrics()
            queue_depth = metrics.queue_depth
            
            if queue_depth >= self.thresholds.queue_depth_critical:
                return HealthCheckResult(
                    name="queue_depth",
                    status=HealthStatus.CRITICAL,
                    message=f"Queue critically deep: {queue_depth} events",
                    details={'queue_depth': queue_depth},
                    action="Worker is overwhelmed. Consider: (1) Reduce write rate, (2) Increase worker batch size, (3) Scale horizontally"
                )
            
            elif queue_depth >= self.thresholds.queue_depth_warning:
                return HealthCheckResult(
                    name="queue_depth",
                    status=HealthStatus.DEGRADED,
                    message=f"Queue depth elevated: {queue_depth} events",
                    details={'queue_depth': queue_depth},
                    action="Monitor closely. Worker may be falling behind."
                )
            
            else:
                return HealthCheckResult(
                    name="queue_depth",
                    status=HealthStatus.HEALTHY,
                    message=f"Queue depth normal: {queue_depth} events",
                    details={'queue_depth': queue_depth}
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="queue_depth",
                status=HealthStatus.DEGRADED,
                message=f"Could not check queue depth: {e}",
                details={'error': str(e)},
                action="Check Redis connectivity"
            )
    
    def _check_worker_lag(self) -> HealthCheckResult:
        """Check worker processing lag"""
        try:
            metrics = self.krnx.get_worker_metrics()
            lag_seconds = metrics.lag_seconds
            
            if lag_seconds >= self.thresholds.lag_critical_seconds:
                return HealthCheckResult(
                    name="worker_lag",
                    status=HealthStatus.CRITICAL,
                    message=f"Worker critically lagging: {lag_seconds:.1f}s behind",
                    details={'lag_seconds': lag_seconds},
                    action="Worker cannot keep up. Increase batch size or reduce write rate."
                )
            
            elif lag_seconds >= self.thresholds.lag_warning_seconds:
                return HealthCheckResult(
                    name="worker_lag",
                    status=HealthStatus.DEGRADED,
                    message=f"Worker lagging: {lag_seconds:.1f}s behind",
                    details={'lag_seconds': lag_seconds},
                    action="Monitor worker performance. May need tuning."
                )
            
            else:
                return HealthCheckResult(
                    name="worker_lag",
                    status=HealthStatus.HEALTHY,
                    message=f"Worker lag acceptable: {lag_seconds:.1f}s",
                    details={'lag_seconds': lag_seconds}
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="worker_lag",
                status=HealthStatus.DEGRADED,
                message=f"Could not check worker lag: {e}",
                details={'error': str(e)},
                action="Check Redis streams configuration"
            )
    
    def _check_error_rate(self) -> HealthCheckResult:
        """Check error rate (last hour)"""
        try:
            metrics = self.krnx.get_worker_metrics()
            errors = metrics.errors_last_hour
            
            if errors >= self.thresholds.error_rate_critical:
                return HealthCheckResult(
                    name="error_rate",
                    status=HealthStatus.CRITICAL,
                    message=f"High error rate: {errors} errors/hour",
                    details={
                        'errors_last_hour': errors,
                        'last_error': metrics.last_error
                    },
                    action=f"Investigate error: {metrics.last_error}"
                )
            
            elif errors >= self.thresholds.error_rate_warning:
                return HealthCheckResult(
                    name="error_rate",
                    status=HealthStatus.DEGRADED,
                    message=f"Elevated error rate: {errors} errors/hour",
                    details={
                        'errors_last_hour': errors,
                        'last_error': metrics.last_error
                    },
                    action="Check logs for error patterns"
                )
            
            else:
                return HealthCheckResult(
                    name="error_rate",
                    status=HealthStatus.HEALTHY,
                    message=f"Error rate normal: {errors} errors/hour",
                    details={'errors_last_hour': errors}
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="error_rate",
                status=HealthStatus.DEGRADED,
                message=f"Could not check error rate: {e}",
                details={'error': str(e)},
                action="Verify error tracking is working"
            )
    
    def _check_redis_memory(self) -> HealthCheckResult:
        """Check Redis memory usage"""
        try:
            stats = self.krnx.stm.get_stats()
            memory_mb = stats.get('used_memory_mb', 0)
            
            if memory_mb >= self.thresholds.redis_memory_critical_mb:
                return HealthCheckResult(
                    name="redis_memory",
                    status=HealthStatus.CRITICAL,
                    message=f"Redis memory critical: {memory_mb:.1f} MB",
                    details={'used_memory_mb': memory_mb},
                    action="Redis may run out of memory. Check TTL settings and maxmemory policy."
                )
            
            elif memory_mb >= self.thresholds.redis_memory_warning_mb:
                return HealthCheckResult(
                    name="redis_memory",
                    status=HealthStatus.DEGRADED,
                    message=f"Redis memory elevated: {memory_mb:.1f} MB",
                    details={'used_memory_mb': memory_mb},
                    action="Monitor Redis memory growth"
                )
            
            else:
                return HealthCheckResult(
                    name="redis_memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Redis memory normal: {memory_mb:.1f} MB",
                    details={'used_memory_mb': memory_mb}
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="redis_memory",
                status=HealthStatus.DEGRADED,
                message=f"Could not check Redis memory: {e}",
                details={'error': str(e)},
                action="Verify Redis INFO command works"
            )
    
    def _check_ltm_storage(self) -> HealthCheckResult:
        """Check LTM storage size"""
        try:
            integrity = self.krnx.ltm.verify_integrity()
            total_size_mb = integrity.get('total_size_mb', 0)
            
            if total_size_mb >= self.thresholds.ltm_size_critical_mb:
                return HealthCheckResult(
                    name="ltm_storage",
                    status=HealthStatus.CRITICAL,
                    message=f"LTM storage critical: {total_size_mb:.1f} MB",
                    details={
                        'total_size_mb': total_size_mb,
                        'warm_mb': integrity['warm_tier']['size_mb'],
                        'cold_mb': integrity['cold_tier']['size_mb']
                    },
                    action="Run archival: krnx archive run --dry-run (check first)"
                )
            
            elif total_size_mb >= self.thresholds.ltm_size_warning_mb:
                return HealthCheckResult(
                    name="ltm_storage",
                    status=HealthStatus.DEGRADED,
                    message=f"LTM storage elevated: {total_size_mb:.1f} MB",
                    details={
                        'total_size_mb': total_size_mb,
                        'warm_mb': integrity['warm_tier']['size_mb'],
                        'cold_mb': integrity['cold_tier']['size_mb']
                    },
                    action="Consider enabling auto-archival or running manual archival"
                )
            
            else:
                return HealthCheckResult(
                    name="ltm_storage",
                    status=HealthStatus.HEALTHY,
                    message=f"LTM storage normal: {total_size_mb:.1f} MB",
                    details={
                        'total_size_mb': total_size_mb,
                        'warm_mb': integrity['warm_tier']['size_mb'],
                        'cold_mb': integrity['cold_tier']['size_mb']
                    }
                )
        
        except Exception as e:
            return HealthCheckResult(
                name="ltm_storage",
                status=HealthStatus.DEGRADED,
                message=f"Could not check LTM storage: {e}",
                details={'error': str(e)},
                action="Check disk space and database accessibility"
            )


# ==============================================
# EXPORTS
# ==============================================

__all__ = [
    'HealthChecker',
    'HealthReport',
    'HealthCheckResult',
    'HealthStatus',
    'HealthThresholds'
]
