"""Unit tests for retry utilities."""

import pytest
import time
from unittest.mock import Mock, patch

from src.utils.retry import (
    RetryContext,
    RetryError,
    calculate_delay,
    retry,
    should_retry,
)


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_calculate_delay_first_attempt(self):
        """Test delay calculation for first retry."""
        delay = calculate_delay(
            attempt=0,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert delay == 1.0

    def test_calculate_delay_exponential(self):
        """Test exponential delay increase."""
        delay = calculate_delay(
            attempt=3,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
        )

        # 1.0 * 2^3 = 8.0
        assert delay == 8.0

    def test_calculate_delay_max_cap(self):
        """Test that delay is capped at max_delay."""
        delay = calculate_delay(
            attempt=10,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert delay == 60.0

    def test_calculate_delay_with_jitter(self):
        """Test delay with jitter."""
        delay = calculate_delay(
            attempt=2,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Jitter should reduce delay by up to 50%
        # Base delay would be 4.0, with jitter it's 2.0-4.0
        assert 2.0 <= delay <= 4.0


class TestShouldRetry:
    """Tests for retry decision logic."""

    def test_should_retry_timeout(self):
        """Test that timeout errors are retryable."""
        assert should_retry(TimeoutError()) is True

    def test_should_retry_connection(self):
        """Test that connection errors are retryable."""
        assert should_retry(ConnectionError()) is True

    def test_should_retry_value_error(self):
        """Test that value errors are not retryable by default."""
        assert should_retry(ValueError()) is False

    def test_should_retry_custom_exceptions(self):
        """Test retry with custom exception list."""
        assert should_retry(
            ValueError(),
            retryable_exceptions=(ValueError, TypeError)
        ) is True

    def test_should_retry_pattern_matching(self):
        """Test retry decision based on exception name patterns."""
        class RateLimitError(Exception):
            pass

        assert should_retry(RateLimitError()) is True


class TestRetryDecorator:
    """Tests for retry decorator."""

    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = [0]

        def test_func():
            call_count[0] += 1
            return "success"

        decorated = retry(max_attempts=3)(test_func)
        result = decorated()

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_success_after_failures(self):
        """Test successful execution after retries."""
        call_count = [0]

        def test_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise TimeoutError()
            return "success"

        decorated = retry(max_attempts=3, initial_delay=0.01)(test_func)
        result = decorated()

        assert result == "success"
        assert call_count[0] == 3

    def test_retry_exhausted(self):
        """Test all retry attempts exhausted."""
        call_count = [0]

        def test_func():
            call_count[0] += 1
            raise TimeoutError("Test timeout")

        decorated = retry(max_attempts=3, initial_delay=0.01)(test_func)

        with pytest.raises(RetryError) as exc_info:
            decorated()

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, TimeoutError)
        assert call_count[0] == 3

    def test_retry_non_retryable_error(self):
        """Test that non-retryable errors are not retried."""
        call_count = [0]

        def test_func():
            call_count[0] += 1
            raise ValueError("Test error")

        decorated = retry(max_attempts=3, initial_delay=0.01)(test_func)

        with pytest.raises(ValueError, match="Test error"):
            decorated()

        assert call_count[0] == 1

    def test_retry_with_callback(self):
        """Test retry with on_retry callback."""
        callback_calls = []
        call_count = [0]

        def callback(exc, attempt):
            callback_calls.append((exc, attempt))

        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError()
            return "success"

        decorated = retry(
            max_attempts=3,
            initial_delay=0.01,
            on_retry=callback
        )(test_func)

        result = decorated()

        assert result == "success"
        assert len(callback_calls) == 1

    def test_retry_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""
        @retry(max_attempts=3)
        def sample_function():
            """Sample docstring."""
            return "result"

        assert sample_function.__name__ == "sample_function"
        assert sample_function.__doc__ == "Sample docstring."

    def test_retry_with_arguments(self):
        """Test retry with function arguments."""
        received_args = []

        def test_func(*args, **kwargs):
            received_args.append((args, kwargs))
            return "success"

        decorated = retry(max_attempts=3)(test_func)
        result = decorated("arg1", kwarg1="value1")

        assert result == "success"
        assert len(received_args) == 1
        assert received_args[0] == (("arg1",), {"kwarg1": "value1"})

    def test_retry_delays_between_attempts(self):
        """Test that delays occur between retry attempts."""
        call_count = [0]

        def test_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError()
            return "success"

        decorated = retry(
            max_attempts=3,
            initial_delay=0.1,
            jitter=False
        )(test_func)

        start_time = time.time()
        result = decorated()
        elapsed_time = time.time() - start_time

        assert result == "success"
        # Should have at least one delay of 0.1 seconds
        assert elapsed_time >= 0.1


class TestRetryContext:
    """Tests for RetryContext."""

    def test_retry_context_success(self):
        """Test retry context with successful execution."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(return_value="success")

        result = ctx.execute(mock_func)

        assert result == "success"
        assert len(ctx.attempt_history) == 1
        assert ctx.attempt_history[0]["success"] is True

    def test_retry_context_with_retries(self):
        """Test retry context with multiple attempts."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(side_effect=[TimeoutError(), TimeoutError(), "success"])

        result = ctx.execute(mock_func)

        assert result == "success"
        assert len(ctx.attempt_history) == 3
        assert ctx.attempt_history[0]["success"] is False
        assert ctx.attempt_history[1]["success"] is False
        assert ctx.attempt_history[2]["success"] is True

    def test_retry_context_exhausted(self):
        """Test retry context when all attempts fail."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(side_effect=TimeoutError("Test timeout"))

        with pytest.raises(RetryError):
            ctx.execute(mock_func)

        assert len(ctx.attempt_history) == 3
        assert all(not h["success"] for h in ctx.attempt_history)

    def test_retry_context_with_args(self):
        """Test retry context with function arguments."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(return_value="success")

        result = ctx.execute(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_context_get_summary(self):
        """Test getting retry context summary."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(side_effect=[TimeoutError(), "success"])

        result = ctx.execute(mock_func)
        summary = ctx.get_summary()

        assert summary["total_attempts"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["attempt_history"]) == 2

    def test_retry_context_adaptive_delay(self):
        """Test adaptive delay calculation in retry context."""
        ctx = RetryContext(
            max_attempts=5,
            initial_delay=0.01,
            exponential_base=2.0,
            jitter=False
        )

        # First delay
        delay1 = ctx._calculate_adaptive_delay(0)
        assert delay1 == 0.01

        # Simulate failed attempts
        ctx.attempt_history = [
            {"attempt": 1, "success": False, "error": "test", "duration": 0.1},
            {"attempt": 2, "success": False, "error": "test", "duration": 0.1},
        ]

        # Delay should be increased due to consecutive failures
        delay2 = ctx._calculate_adaptive_delay(2)
        assert delay2 > 0.04  # Base would be 0.04, increased by 50%

    def test_retry_context_records_duration(self):
        """Test that retry context records execution duration."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)

        def slow_func():
            time.sleep(0.05)
            return "success"

        result = ctx.execute(slow_func)

        assert result == "success"
        assert ctx.attempt_history[0]["duration"] >= 0.05

    def test_retry_context_non_retryable_error(self):
        """Test retry context with non-retryable error."""
        ctx = RetryContext(max_attempts=3, initial_delay=0.01)
        mock_func = Mock(side_effect=ValueError("Test error"))

        with pytest.raises(ValueError, match="Test error"):
            ctx.execute(mock_func)

        # Should only have 1 attempt since error is not retryable
        assert len(ctx.attempt_history) == 1
