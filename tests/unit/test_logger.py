"""Unit tests for logger module."""

import json
import logging
import pytest
from unittest.mock import Mock, patch
import time

from src.utils.logger import (
    CorrelationIdFilter,
    JSONFormatter,
    LoggerAdapter,
    PerformanceLogger,
    StructuredFormatter,
    clear_correlation_id,
    get_correlation_id,
    get_logger,
    get_structured_logger,
    set_correlation_id,
    setup_logging,
)


class TestCorrelationId:
    """Tests for correlation ID functionality."""

    def setup_method(self):
        """Clear correlation ID before each test."""
        clear_correlation_id()

    def teardown_method(self):
        """Clear correlation ID after each test."""
        clear_correlation_id()

    def test_set_correlation_id(self):
        """Test setting correlation ID."""
        corr_id = set_correlation_id("test-123")

        assert corr_id == "test-123"
        assert get_correlation_id() == "test-123"

    def test_set_correlation_id_auto_generate(self):
        """Test auto-generating correlation ID."""
        corr_id = set_correlation_id()

        assert corr_id is not None
        assert len(corr_id) > 0
        assert get_correlation_id() == corr_id

    def test_get_correlation_id_not_set(self):
        """Test getting correlation ID when not set."""
        assert get_correlation_id() is None

    def test_clear_correlation_id(self):
        """Test clearing correlation ID."""
        set_correlation_id("test-123")
        clear_correlation_id()

        assert get_correlation_id() is None


class TestCorrelationIdFilter:
    """Tests for CorrelationIdFilter."""

    def setup_method(self):
        """Clear correlation ID before each test."""
        clear_correlation_id()

    def test_filter_adds_correlation_id(self):
        """Test that filter adds correlation ID to record."""
        set_correlation_id("test-123")

        filter_obj = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)

        assert result is True
        assert record.correlation_id == "test-123"

    def test_filter_adds_na_when_not_set(self):
        """Test that filter adds N/A when correlation ID not set."""
        filter_obj = CorrelationIdFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = filter_obj.filter(record)

        assert result is True
        assert record.correlation_id == "N/A"


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_json_formatter_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-123"

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["correlation_id"] == "test-123"
        assert "timestamp" in data

    def test_json_formatter_with_exception(self):
        """Test JSON formatting with exception."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )
            record.correlation_id = "test-123"

            formatted = formatter.format(record)
            data = json.loads(formatted)

            assert "exception" in data
            assert "ValueError" in data["exception"]
            assert "Test error" in data["exception"]


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_structured_formatter_basic(self):
        """Test basic structured formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-123"

        formatted = formatter.format(record)

        assert "INFO" in formatted
        assert "test.logger" in formatted
        assert "Test message" in formatted
        assert "test-123" in formatted

    def test_structured_formatter_with_extra_fields(self):
        """Test structured formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.correlation_id = "test-123"
        record.extra_fields = {"user": "testuser", "action": "test"}

        formatted = formatter.format(record)

        assert "user=testuser" in formatted
        assert "action=test" in formatted


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_console(self):
        """Test setting up console logging."""
        setup_logging(level="DEBUG", enable_console=True, enable_file=False)

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_json(self):
        """Test setting up JSON logging."""
        setup_logging(
            level="INFO",
            enable_console=True,
            enable_file=False,
            enable_json=True
        )

        logger = logging.getLogger()
        console_handler = logger.handlers[0]
        assert isinstance(console_handler.formatter, JSONFormatter)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = get_logger("test.module")

        assert logger.name == "test.module"
        assert isinstance(logger, logging.Logger)

    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1.name == "module1"
        assert logger2.name == "module2"
        assert logger1 is not logger2


class TestLoggerAdapter:
    """Tests for LoggerAdapter."""

    def test_logger_adapter_adds_extra_fields(self):
        """Test that adapter adds extra fields."""
        base_logger = get_logger("test")
        adapter = LoggerAdapter(base_logger, {"component": "test_component"})

        # Process a log message
        msg, kwargs = adapter.process("Test message", {"extra": {"key": "value"}})

        assert msg == "Test message"
        assert "extra_fields" in kwargs["extra"]
        assert kwargs["extra"]["extra_fields"]["component"] == "test_component"
        assert kwargs["extra"]["extra_fields"]["key"] == "value"


class TestGetStructuredLogger:
    """Tests for get_structured_logger."""

    def test_get_structured_logger(self):
        """Test getting a structured logger."""
        logger = get_structured_logger("test", {"component": "test_comp"})

        assert isinstance(logger, LoggerAdapter)
        assert logger.extra["component"] == "test_comp"


class TestPerformanceLogger:
    """Tests for PerformanceLogger."""

    def test_performance_logger_measures_duration(self):
        """Test that performance logger measures duration."""
        logger = Mock()

        with PerformanceLogger("test_operation", logger=logger):
            time.sleep(0.05)

        # Should have two log calls: start and complete
        assert logger.log.call_count == 2

        # Check completion log
        last_call = logger.log.call_args_list[-1]
        assert "completed" in last_call[0][1].lower()
        extra = last_call[1]["extra"]
        assert extra["operation"] == "test_operation"
        assert extra["duration"] >= 0.05

    def test_performance_logger_logs_failure(self):
        """Test that performance logger logs failures."""
        logger = Mock()

        with pytest.raises(ValueError):
            with PerformanceLogger("test_operation", logger=logger):
                raise ValueError("Test error")

        # Should have two log calls: start and error
        assert logger.log.call_count == 2

        # Check error log
        last_call = logger.log.call_args_list[-1]
        assert last_call[0][0] == logging.ERROR
        assert "failed" in last_call[0][1].lower()

    def test_performance_logger_with_extra_fields(self):
        """Test performance logger with extra fields."""
        logger = Mock()
        extra_fields = {"user": "testuser", "request_id": "123"}

        with PerformanceLogger(
            "test_operation",
            logger=logger,
            extra=extra_fields
        ):
            time.sleep(0.01)

        # Check that extra fields are included
        last_call = logger.log.call_args_list[-1]
        extra = last_call[1]["extra"]
        assert extra["user"] == "testuser"
        assert extra["request_id"] == "123"

    def test_performance_logger_custom_level(self):
        """Test performance logger with custom log level."""
        logger = Mock()

        with PerformanceLogger(
            "test_operation",
            logger=logger,
            level=logging.DEBUG
        ):
            pass

        # Check that custom level is used
        first_call = logger.log.call_args_list[0]
        assert first_call[0][0] == logging.DEBUG
