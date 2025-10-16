"""Tests for error context capture."""

import json

import pytest

from src.resilience.error_context import (
    ErrorContext,
    ErrorContextCapture,
    capture_error_context,
)


class TestErrorContext:
    """Tests for ErrorContext."""

    def test_create_error_context(self):
        """Test creating error context."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={"data": "test"},
        )

        assert context.exception_type == "ValueError"
        assert context.exception_message == "Test error"
        assert context.state_snapshot == {"data": "test"}

    def test_to_dict(self):
        """Test converting error context to dict."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={"data": "test"},
            retry_attempts=3,
        )

        data = context.to_dict(redact_pii=False)

        assert data["exception_type"] == "ValueError"
        assert data["exception_message"] == "Test error"
        assert data["state_snapshot"] == {"data": "test"}
        assert data["retry_attempts"] == 3
        assert "timestamp" in data

    def test_to_json(self):
        """Test converting error context to JSON."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={"data": "test"},
        )

        json_str = context.to_json(redact_pii=False)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["exception_type"] == "ValueError"

    def test_redact_sensitive_keys(self):
        """Test PII redaction for sensitive keys."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "password": "secret123",
                "api_key": "abc123",
                "normal_data": "visible",
            },
        )

        data = context.to_dict(redact_pii=True)

        assert data["state_snapshot"]["password"] == "[REDACTED]"
        assert data["state_snapshot"]["api_key"] == "[REDACTED]"
        assert data["state_snapshot"]["normal_data"] == "visible"

    def test_redact_nested_sensitive_data(self):
        """Test PII redaction in nested structures."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "user": {
                    "name": "John",
                    "password": "secret",
                },
                "config": {
                    "api_token": "xyz789",
                },
            },
        )

        data = context.to_dict(redact_pii=True)

        assert data["state_snapshot"]["user"]["name"] == "John"
        assert data["state_snapshot"]["user"]["password"] == "[REDACTED]"
        assert data["state_snapshot"]["config"]["api_token"] == "[REDACTED]"

    def test_redact_sensitive_values(self):
        """Test PII redaction for sensitive-looking values."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "token": "sk-abcdefghij1234567890",  # Looks like API key
                "normal": "regular_value",
            },
        )

        data = context.to_dict(redact_pii=True)

        assert data["state_snapshot"]["token"] == "[REDACTED]"
        assert data["state_snapshot"]["normal"] == "regular_value"

    def test_redact_list_items(self):
        """Test PII redaction in lists."""
        exception = ValueError("Test error")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "items": [
                    {"password": "secret1"},
                    {"password": "secret2"},
                ],
            },
        )

        data = context.to_dict(redact_pii=True)

        assert data["state_snapshot"]["items"][0]["password"] == "[REDACTED]"
        assert data["state_snapshot"]["items"][1]["password"] == "[REDACTED]"


class TestErrorContextCapture:
    """Tests for ErrorContextCapture."""

    def test_capture_basic_context(self):
        """Test capturing basic error context."""
        capturer = ErrorContextCapture(capture_system_metrics=False)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = capturer.capture(e)

        assert context.exception_type == "ValueError"
        assert context.exception_message == "Test error"
        assert "ValueError" in context.stack_trace

    def test_capture_with_state(self):
        """Test capturing context with state."""
        capturer = ErrorContextCapture(capture_system_metrics=False)

        state = {
            "completed_validators": ["val1", "val2"],
            "metadata": {"retry_count": 3},
        }

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = capturer.capture(e, state=state)

        assert context.state_snapshot["completed_validators"] == ["val1", "val2"]
        assert context.retry_attempts == 3

    def test_capture_with_additional_context(self):
        """Test capturing with additional context."""
        capturer = ErrorContextCapture(capture_system_metrics=False)

        additional = {
            "validator_name": "test_validator",
            "input_size": 1024,
        }

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = capturer.capture(e, additional_context=additional)

        assert context.additional_context["validator_name"] == "test_validator"
        assert context.additional_context["input_size"] == 1024

    def test_capture_with_system_metrics(self):
        """Test capturing system metrics."""
        capturer = ErrorContextCapture(capture_system_metrics=True)

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = capturer.capture(e)

        # Should have some metrics (or note about psutil)
        assert context.system_metrics is not None
        assert len(context.system_metrics) > 0

    def test_capture_unserializable_state(self):
        """Test capturing state with unserializable objects."""
        capturer = ErrorContextCapture(capture_system_metrics=False)

        # Create state with unserializable object
        class UnserializableClass:
            pass

        state = {
            "data": "test",
            "object": UnserializableClass(),
        }

        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = capturer.capture(e, state=state)

        # Should handle gracefully
        assert context.state_snapshot is not None


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_capture_error_context_function(self):
        """Test capture_error_context convenience function."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = capture_error_context(
                e,
                capture_system_metrics=False,
            )

        assert context.exception_type == "ValueError"
        assert isinstance(context, ErrorContext)


class TestSensitivePatterns:
    """Tests for sensitive data pattern detection."""

    def test_password_patterns(self):
        """Test detection of password-like keys."""
        exception = ValueError("Test")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "password": "secret",
                "user_password": "secret",
                "pwd": "secret",
                "passwd": "secret",
            },
        )

        data = context.to_dict(redact_pii=True)

        for key in ["password", "user_password", "pwd", "passwd"]:
            assert data["state_snapshot"][key] == "[REDACTED]"

    def test_token_patterns(self):
        """Test detection of token-like keys."""
        exception = ValueError("Test")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "token": "abc123",
                "auth_token": "abc123",
                "api_key": "abc123",
                "apikey": "abc123",
            },
        )

        data = context.to_dict(redact_pii=True)

        for key in ["token", "auth_token", "api_key", "apikey"]:
            assert data["state_snapshot"][key] == "[REDACTED]"

    def test_credit_card_patterns(self):
        """Test detection of credit card-like keys."""
        exception = ValueError("Test")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "credit_card": "1234567890123456",
                "cc_number": "1234567890123456",
                "cvv": "123",
            },
        )

        data = context.to_dict(redact_pii=True)

        for key in ["credit_card", "cc_number", "cvv"]:
            assert data["state_snapshot"][key] == "[REDACTED]"

    def test_ssn_pattern(self):
        """Test detection of SSN pattern in values."""
        exception = ValueError("Test")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "id": "123-45-6789",  # SSN pattern
                "normal_id": "ABC-123",
            },
        )

        data = context.to_dict(redact_pii=True)

        assert data["state_snapshot"]["id"] == "[REDACTED]"
        assert data["state_snapshot"]["normal_id"] == "ABC-123"

    def test_api_key_pattern(self):
        """Test detection of API key pattern in values."""
        exception = ValueError("Test")
        context = ErrorContext(
            exception=exception,
            state_snapshot={
                "key": "sk-1234567890abcdefghij",  # OpenAI-style
                "normal_key": "feature_flag_key",
            },
        )

        data = context.to_dict(redact_pii=True)

        assert data["state_snapshot"]["key"] == "[REDACTED]"
        assert data["state_snapshot"]["normal_key"] == "feature_flag_key"
