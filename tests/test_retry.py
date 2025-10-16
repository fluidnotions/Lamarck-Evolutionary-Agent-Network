"""Tests for retry logic with exponential backoff."""

import asyncio
import time

import pytest

from src.utils.retry import (
    MaxRetriesExceeded,
    RetryBudgetExceeded,
    RetryConfig,
    retry_async,
    retry_with_context,
)


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_calculate_delay(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
        )

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_with_max(self):
        """Test delay calculation respects max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False,
        )

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 5.0  # Capped at max_delay
        assert config.calculate_delay(4) == 5.0

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            base_delay=10.0,
            exponential_base=2.0,
            jitter=True,
            jitter_range=(0.8, 1.2),
        )

        delays = [config.calculate_delay(1) for _ in range(100)]

        # All delays should be within jitter range
        assert all(16.0 <= d <= 24.0 for d in delays)

        # Delays should vary (not all the same)
        assert len(set(delays)) > 10

    def test_should_retry(self):
        """Test exception retry decision."""
        config = RetryConfig(
            retry_on=(ValueError, RuntimeError),
            exclude_on=(KeyError,),
        )

        assert config.should_retry(ValueError("test"))
        assert config.should_retry(RuntimeError("test"))
        assert not config.should_retry(KeyError("test"))
        assert not config.should_retry(TypeError("test"))


class TestRetryWithContext:
    """Tests for retry_with_context decorator."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = 0

        @retry_with_context(RetryConfig(max_attempts=3))
        async def test_func(state):
            nonlocal call_count
            call_count += 1
            return "success"

        state = {"metadata": {}}
        result = await test_func(state)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test successful execution after retries."""
        call_count = 0

        @retry_with_context(RetryConfig(max_attempts=3, base_delay=0.01))
        async def test_func(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        state = {"metadata": {}}
        result = await test_func(state)

        assert result == "success"
        assert call_count == 3
        assert state["metadata"]["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test max retries exceeded."""

        @retry_with_context(RetryConfig(max_attempts=3, base_delay=0.01))
        async def test_func(state):
            raise ValueError("Permanent failure")

        state = {"metadata": {}}

        with pytest.raises(MaxRetriesExceeded):
            await test_func(state)

        # 3 attempts = 2 retries (first attempt + 2 retries)
        assert state["metadata"]["retry_count"] == 2

    @pytest.mark.asyncio
    async def test_retry_budget_exceeded(self):
        """Test retry budget enforcement."""

        @retry_with_context(RetryConfig(max_attempts=3))
        async def test_func(state):
            raise ValueError("Failure")

        state = {"metadata": {"retry_count": 10, "retry_budget": 10}}

        with pytest.raises(RetryBudgetExceeded):
            await test_func(state)

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test non-retryable exception is raised immediately."""
        call_count = 0

        @retry_with_context(
            RetryConfig(max_attempts=3, retry_on=(ValueError,))
        )
        async def test_func(state):
            nonlocal call_count
            call_count += 1
            raise KeyError("Non-retryable")

        state = {"metadata": {}}

        with pytest.raises(KeyError):
            await test_func(state)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_total_retry_time(self):
        """Test max total retry time enforcement."""

        @retry_with_context(
            RetryConfig(
                max_attempts=10,
                base_delay=0.1,
                max_total_retry_time=0.3,
            )
        )
        async def test_func(state):
            raise ValueError("Failure")

        state = {"metadata": {}}
        start_time = time.time()

        with pytest.raises(MaxRetriesExceeded):
            await test_func(state)

        elapsed = time.time() - start_time
        # Should stop due to time limit, not exhaust all retries
        # Allow some margin for timing variations
        assert elapsed < 1.0  # Should stop reasonably quickly
        assert elapsed >= 0.3  # Should have waited at least the timeout

    def test_sync_retry(self):
        """Test sync retry wrapper."""
        call_count = 0

        @retry_with_context(RetryConfig(max_attempts=3, base_delay=0.01))
        def test_func(state):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        state = {"metadata": {}}
        result = test_func(state)

        assert result == "success"
        assert call_count == 2


class TestRetryAsync:
    """Tests for retry_async decorator."""

    @pytest.mark.asyncio
    async def test_simple_retry(self):
        """Test simple async retry without state."""
        call_count = 0

        @retry_async(max_attempts=3, base_delay=0.01)
        async def test_func(value):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return value * 2

        result = await test_func(5)

        assert result == 10
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_different_exceptions(self):
        """Test retry only on specified exceptions."""
        call_count = 0

        @retry_async(
            max_attempts=3,
            base_delay=0.01,
            retry_on=(ValueError,)
        )
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retry this")
            else:
                raise KeyError("Don't retry this")

        with pytest.raises(KeyError):
            await test_func()

        assert call_count == 2


class TestRetryHistory:
    """Tests for retry history tracking."""

    @pytest.mark.asyncio
    async def test_retry_history_recorded(self):
        """Test that retry history is recorded in state."""
        call_count = 0

        @retry_with_context(RetryConfig(max_attempts=3, base_delay=0.01))
        async def test_func(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        state = {"metadata": {}}
        await test_func(state)

        # Check retry history
        history = state["metadata"]["retry_history"]
        assert len(history) == 2  # 2 retries before success

        # Check history entries
        for entry in history:
            assert "function" in entry
            assert "attempt" in entry
            assert "delay" in entry
            assert "error" in entry
            assert "timestamp" in entry
