"""Comprehensive error context capture for debugging and recovery."""

import json
import logging
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class ErrorContext:
    """Comprehensive error context for debugging and recovery."""

    def __init__(
        self,
        exception: Exception,
        state_snapshot: Optional[dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
        system_metrics: Optional[dict[str, Any]] = None,
        retry_attempts: int = 0,
        additional_context: Optional[dict[str, Any]] = None,
    ):
        """Initialize error context.

        Args:
            exception: The exception that occurred
            state_snapshot: Snapshot of ValidationState at time of error
            stack_trace: Full stack trace
            system_metrics: System metrics (CPU, memory, etc.)
            retry_attempts: Number of retry attempts made
            additional_context: Additional custom context
        """
        self.exception = exception
        self.exception_type = type(exception).__name__
        self.exception_message = str(exception)
        self.state_snapshot = state_snapshot or {}
        self.stack_trace = stack_trace or traceback.format_exc()
        self.system_metrics = system_metrics or {}
        self.retry_attempts = retry_attempts
        self.additional_context = additional_context or {}
        self.timestamp = datetime.now()

    def to_dict(self, redact_pii: bool = True) -> dict[str, Any]:
        """Convert error context to dictionary.

        Args:
            redact_pii: Whether to redact potentially sensitive information

        Returns:
            Dictionary representation of error context
        """
        state_snapshot = self.state_snapshot
        if redact_pii:
            state_snapshot = self._redact_sensitive_data(state_snapshot)

        return {
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
            "stack_trace": self.stack_trace,
            "state_snapshot": state_snapshot,
            "system_metrics": self.system_metrics,
            "retry_attempts": self.retry_attempts,
            "additional_context": self.additional_context,
        }

    def to_json(self, redact_pii: bool = True, indent: int = 2) -> str:
        """Convert error context to JSON string.

        Args:
            redact_pii: Whether to redact potentially sensitive information
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(redact_pii=redact_pii), indent=indent, default=str)

    def _redact_sensitive_data(self, data: Any) -> Any:
        """Recursively redact potentially sensitive data.

        Args:
            data: Data to redact

        Returns:
            Data with sensitive fields redacted
        """
        if isinstance(data, dict):
            redacted = {}
            for key, value in data.items():
                # Check if key looks sensitive
                if self._is_sensitive_key(key):
                    redacted[key] = "[REDACTED]"
                else:
                    redacted[key] = self._redact_sensitive_data(value)
            return redacted
        elif isinstance(data, list):
            return [self._redact_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            # Check if value looks like sensitive data
            if self._is_sensitive_value(data):
                return "[REDACTED]"
            return data
        else:
            return data

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key name suggests sensitive data.

        Args:
            key: Dictionary key name

        Returns:
            True if key suggests sensitive data
        """
        sensitive_patterns = [
            r"password",
            r"passwd",
            r"pwd",
            r"secret",
            r"token",
            r"api[_-]?key",
            r"auth",
            r"credential",
            r"ssn",
            r"social[_-]?security",
            r"credit[_-]?card",
            r"cc[_-]?number",
            r"cvv",
            r"pin",
        ]

        key_lower = key.lower()
        return any(re.search(pattern, key_lower) for pattern in sensitive_patterns)

    def _is_sensitive_value(self, value: str) -> bool:
        """Check if a value looks like sensitive data.

        Args:
            value: String value to check

        Returns:
            True if value looks sensitive
        """
        # Check for patterns that look like secrets
        patterns = [
            r"^sk-[A-Za-z0-9]{20,}$",  # API keys
            r"^[A-Za-z0-9+/]{40,}={0,2}$",  # Base64 encoded secrets
            r"^\d{3}-\d{2}-\d{4}$",  # SSN
            r"^\d{16}$",  # Credit card number
        ]

        return any(re.match(pattern, value) for pattern in patterns)


class ErrorContextCapture:
    """Utility for capturing comprehensive error context."""

    def __init__(self, capture_system_metrics: bool = True):
        """Initialize error context capture.

        Args:
            capture_system_metrics: Whether to capture system metrics
        """
        self.capture_system_metrics = capture_system_metrics

    def capture(
        self,
        exception: Exception,
        state: Optional[dict[str, Any]] = None,
        additional_context: Optional[dict[str, Any]] = None,
    ) -> ErrorContext:
        """Capture comprehensive error context.

        Args:
            exception: The exception that occurred
            state: Current ValidationState
            additional_context: Additional custom context

        Returns:
            ErrorContext object
        """
        # Capture stack trace
        stack_trace = self._capture_stack_trace()

        # Create state snapshot (deep copy to avoid mutations)
        state_snapshot = self._create_state_snapshot(state) if state else {}

        # Capture system metrics if enabled
        system_metrics = self._capture_system_metrics() if self.capture_system_metrics else {}

        # Get retry attempts from state
        retry_attempts = 0
        if state:
            retry_attempts = state.get("metadata", {}).get("retry_count", 0)

        return ErrorContext(
            exception=exception,
            state_snapshot=state_snapshot,
            stack_trace=stack_trace,
            system_metrics=system_metrics,
            retry_attempts=retry_attempts,
            additional_context=additional_context,
        )

    def _capture_stack_trace(self) -> str:
        """Capture full stack trace.

        Returns:
            Formatted stack trace string
        """
        return traceback.format_exc()

    def _create_state_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        """Create a snapshot of the state.

        Args:
            state: ValidationState to snapshot

        Returns:
            State snapshot (safe for serialization)
        """
        try:
            # Try to serialize to JSON to ensure it's serializable
            # This also creates a deep copy
            serialized = json.dumps(state, default=str)
            return json.loads(serialized)
        except Exception as e:
            logger.warning(f"Failed to create state snapshot: {e}")
            # Fall back to a simple representation
            return {
                "error": "Failed to create state snapshot",
                "keys": list(state.keys()) if isinstance(state, dict) else str(type(state)),
            }

    def _capture_system_metrics(self) -> dict[str, Any]:
        """Capture current system metrics.

        Returns:
            Dictionary of system metrics
        """
        metrics = {}

        if HAS_PSUTIL:
            try:
                # CPU metrics
                metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                metrics["cpu_count"] = psutil.cpu_count()

                # Memory metrics
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_available_mb"] = memory.available / (1024 * 1024)
                metrics["memory_total_mb"] = memory.total / (1024 * 1024)

                # Disk metrics (for root partition)
                disk = psutil.disk_usage("/")
                metrics["disk_percent"] = disk.percent
                metrics["disk_free_gb"] = disk.free / (1024 * 1024 * 1024)

                # Process metrics
                process = psutil.Process()
                metrics["process_cpu_percent"] = process.cpu_percent(interval=0.1)
                metrics["process_memory_mb"] = process.memory_info().rss / (1024 * 1024)
                metrics["process_threads"] = process.num_threads()

            except Exception as e:
                logger.warning(f"Failed to capture system metrics: {e}")
                metrics["error"] = str(e)
        else:
            metrics["note"] = "psutil not available, install for system metrics"

        return metrics


def capture_error_context(
    exception: Exception,
    state: Optional[dict[str, Any]] = None,
    capture_system_metrics: bool = True,
    additional_context: Optional[dict[str, Any]] = None,
) -> ErrorContext:
    """Convenience function to capture error context.

    Args:
        exception: The exception that occurred
        state: Current ValidationState
        capture_system_metrics: Whether to capture system metrics
        additional_context: Additional custom context

    Returns:
        ErrorContext object
    """
    capturer = ErrorContextCapture(capture_system_metrics=capture_system_metrics)
    return capturer.capture(
        exception=exception,
        state=state,
        additional_context=additional_context,
    )


class ErrorLogger:
    """Logger for error contexts with structured output."""

    def __init__(self, logger_name: str = __name__, log_to_file: bool = False):
        """Initialize error logger.

        Args:
            logger_name: Name for the logger
            log_to_file: Whether to log to file
        """
        self.logger = logging.getLogger(logger_name)
        self.log_to_file = log_to_file

    def log_error_context(
        self,
        error_context: ErrorContext,
        level: int = logging.ERROR,
        redact_pii: bool = True,
    ) -> None:
        """Log error context.

        Args:
            error_context: ErrorContext to log
            level: Logging level
            redact_pii: Whether to redact PII
        """
        # Log summary
        self.logger.log(
            level,
            f"Error: {error_context.exception_type} - {error_context.exception_message}"
        )

        # Log detailed context
        context_dict = error_context.to_dict(redact_pii=redact_pii)
        self.logger.log(level, f"Error Context: {json.dumps(context_dict, indent=2)}")

        # Optionally write to file
        if self.log_to_file:
            self._write_to_file(error_context, redact_pii=redact_pii)

    def _write_to_file(self, error_context: ErrorContext, redact_pii: bool = True) -> None:
        """Write error context to file.

        Args:
            error_context: ErrorContext to write
            redact_pii: Whether to redact PII
        """
        filename = (
            f"error_{error_context.timestamp.strftime('%Y%m%d_%H%M%S')}_"
            f"{error_context.exception_type}.json"
        )

        try:
            with open(filename, "w") as f:
                f.write(error_context.to_json(redact_pii=redact_pii))
            self.logger.info(f"Error context written to {filename}")
        except Exception as e:
            self.logger.warning(f"Failed to write error context to file: {e}")
