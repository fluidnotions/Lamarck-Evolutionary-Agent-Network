"""Resilience components for error handling and recovery."""

from .circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
from .degradation import (
    CompositeDegradationStrategy,
    DegradationStrategy,
    ReturnPartialResults,
    SkipFailedValidators,
    UseCachedResults,
    UseSimplifiedValidation,
)
from .error_context import ErrorContext, ErrorContextCapture, capture_error_context
from .recovery import CheckpointManager, RecoveryStrategy, WorkflowRecovery

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpen",
    # Degradation
    "DegradationStrategy",
    "SkipFailedValidators",
    "UseCachedResults",
    "UseSimplifiedValidation",
    "CompositeDegradationStrategy",
    "ReturnPartialResults",
    # Error Context
    "ErrorContext",
    "ErrorContextCapture",
    "capture_error_context",
    # Recovery
    "RecoveryStrategy",
    "WorkflowRecovery",
    "CheckpointManager",
]
