"""Structured logging for HVAS-Mini.

This module provides structured logging with correlation IDs, performance
metrics, and configurable output formats (text and JSON).
"""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to the log record.

        Args:
            record: Log record to filter

        Returns:
            Always True (doesn't actually filter)
        """
        record.correlation_id = correlation_id_var.get() or "N/A"
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs logs in JSON format."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "N/A"),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add any custom attributes
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
                "extra_fields",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured text logs with key-value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured key-value pairs.

        Args:
            record: Log record to format

        Returns:
            Formatted log string
        """
        # Base format
        base_format = (
            f"{datetime.fromtimestamp(record.created).isoformat()} | "
            f"{record.levelname:8s} | "
            f"{record.name:20s} | "
            f"correlation_id={getattr(record, 'correlation_id', 'N/A'):36s} | "
            f"{record.getMessage()}"
        )

        # Add extra fields if present
        extra_parts = []
        if hasattr(record, "extra_fields"):
            for key, value in record.extra_fields.items():
                extra_parts.append(f"{key}={value}")

        if extra_parts:
            base_format += " | " + " | ".join(extra_parts)

        # Add exception if present
        if record.exc_info:
            base_format += "\n" + self.formatException(record.exc_info)

        return base_format


def setup_logging(
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = False,
    log_file: str = "hvas-mini.log",
    enable_json: bool = False,
) -> None:
    """Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        log_file: Path to log file
        enable_json: Whether to use JSON format
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Create correlation ID filter
    correlation_filter = CorrelationIdFilter()

    # Setup console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if enable_json:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(StructuredFormatter())

        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)

    # Setup file handler
    if enable_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        if enable_json:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(StructuredFormatter())

        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for the current context.

    Args:
        correlation_id: Correlation ID to set (generates new UUID if None)

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID.

    Returns:
        Current correlation ID or None if not set
    """
    return correlation_id_var.get()


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    correlation_id_var.set(None)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra fields to all log records.

    This adapter makes it easy to add consistent contextual information
    to all logs from a specific component.
    """

    def process(
        self, msg: str, kwargs: Any
    ) -> tuple[str, dict[str, Any]]:
        """Process log message and add extra fields.

        Args:
            msg: Log message
            kwargs: Keyword arguments

        Returns:
            Tuple of (message, kwargs) with extra fields added
        """
        # Merge extra fields
        extra = kwargs.get("extra", {})
        extra.update(self.extra)

        # Store in a way that the formatter can access
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"]["extra_fields"] = extra

        return msg, kwargs


def get_structured_logger(
    name: str, extra: Optional[dict[str, Any]] = None
) -> LoggerAdapter:
    """Get a structured logger with extra fields.

    Args:
        name: Logger name
        extra: Extra fields to include in all log records

    Returns:
        LoggerAdapter with extra fields
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, extra or {})


class PerformanceLogger:
    """Context manager for logging performance metrics.

    Example:
        with PerformanceLogger("operation_name"):
            # code to measure
            pass
    """

    def __init__(
        self,
        operation_name: str,
        logger: Optional[logging.Logger] = None,
        level: int = logging.INFO,
        extra: Optional[dict[str, Any]] = None,
    ):
        """Initialize performance logger.

        Args:
            operation_name: Name of the operation being measured
            logger: Logger to use (creates default if None)
            level: Log level for performance metrics
            extra: Extra fields to include in logs
        """
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.level = level
        self.extra = extra or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self) -> "PerformanceLogger":
        """Start timing the operation."""
        import time

        self.start_time = time.time()
        self.logger.log(
            self.level,
            f"Starting operation: {self.operation_name}",
            extra={**self.extra, "operation": self.operation_name},
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and log the results."""
        import time

        self.end_time = time.time()

        if self.start_time is not None:
            duration = self.end_time - self.start_time

            log_extra = {
                **self.extra,
                "operation": self.operation_name,
                "duration": duration,
                "duration_ms": duration * 1000,
            }

            if exc_type is not None:
                self.logger.log(
                    logging.ERROR,
                    f"Operation failed: {self.operation_name} (duration: {duration:.3f}s)",
                    extra=log_extra,
                    exc_info=(exc_type, exc_val, exc_tb),
                )
            else:
                self.logger.log(
                    self.level,
                    f"Operation completed: {self.operation_name} (duration: {duration:.3f}s)",
                    extra=log_extra,
                )


# Initialize default logging on module import
setup_logging()
