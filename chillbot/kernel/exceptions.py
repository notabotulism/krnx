"""
KRNX Kernel - Exceptions

Explicit exception types for kernel operations.
Clear error handling, no silent failures.
"""


class KRNXError(Exception):
    """Base exception for KRNX kernel errors."""
    pass


class ConnectionError(KRNXError):
    """Failed to connect to Redis or database."""
    pass


class RedisUnavailableError(KRNXError):
    """Redis is unavailable."""
    pass


class BackpressureError(KRNXError):
    """System is under load and rejecting writes."""
    pass


class NotFoundError(KRNXError):
    """Requested resource not found."""
    pass


class ValidationError(KRNXError):
    """Input validation failed."""
    pass


class StorageError(KRNXError):
    """Storage operation failed."""
    pass


class IntegrityError(KRNXError):
    """Data integrity check failed."""
    pass


class TimeoutError(KRNXError):
    """Operation timed out."""
    pass


# LTM-specific exceptions
class LTMStorageError(StorageError):
    """LTM storage operation failed."""
    pass


class LTMArchivalError(LTMStorageError):
    """Archival operation failed."""
    pass


class LTMSnapshotError(LTMStorageError):
    """Snapshot operation failed."""
    pass


class LTMIntegrityError(IntegrityError):
    """LTM database integrity check failed."""
    pass


__all__ = [
    "KRNXError",
    "ConnectionError",
    "RedisUnavailableError",
    "BackpressureError",
    "NotFoundError",
    "ValidationError",
    "StorageError",
    "IntegrityError",
    "TimeoutError",
    "LTMStorageError",
    "LTMArchivalError",
    "LTMSnapshotError",
    "LTMIntegrityError",
]
