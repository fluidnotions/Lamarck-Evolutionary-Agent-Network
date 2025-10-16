"""Tests for circuit breaker pattern."""

import asyncio
import time

import pytest

from src.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerRegistry,
    CircuitState,
    get_circuit_breaker,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test", failure_threshold=3)

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_successful_call(self):
        """Test successful call passes through."""
        cb = CircuitBreaker("test", failure_threshold=3)

        def test_func():
            return "success"

        result = cb.call(test_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_failed_call_increments_count(self):
        """Test failed call increments failure count."""
        cb = CircuitBreaker("test", failure_threshold=3)

        def test_func():
            raise ValueError("Failure")

        with pytest.raises(ValueError):
            cb.call(test_func)

        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker("test", failure_threshold=3)

        def test_func():
            raise ValueError("Failure")

        # Fail 3 times
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(test_func)

        assert cb.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        """Test OPEN circuit rejects calls immediately."""
        cb = CircuitBreaker("test", failure_threshold=2)

        def test_func():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(test_func)

        assert cb.state == CircuitState.OPEN

        # Next call should be rejected
        def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpen):
            cb.call(success_func)

    def test_half_open_transition(self):
        """Test transition to HALF_OPEN after timeout."""
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)

        def test_func():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(test_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Next call should attempt HALF_OPEN
        def success_func():
            return "success"

        result = cb.call(success_func)
        assert result == "success"
        # After one success in HALF_OPEN, still in HALF_OPEN
        # (needs success_threshold successes)

    def test_half_open_to_closed(self):
        """Test transition from HALF_OPEN to CLOSED after successes."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            timeout=0.1,
            success_threshold=2,
        )

        def fail_func():
            raise ValueError("Failure")

        def success_func():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Succeed twice in HALF_OPEN
        cb.call(success_func)
        cb.call(success_func)

        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        """Test HALF_OPEN returns to OPEN on failure."""
        cb = CircuitBreaker("test", failure_threshold=2, timeout=0.1)

        def fail_func():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail_func)

        # Wait for timeout
        time.sleep(0.15)

        # Fail in HALF_OPEN
        with pytest.raises(ValueError):
            cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_async_call(self):
        """Test async circuit breaker calls."""
        cb = CircuitBreaker("test", failure_threshold=2)

        async def success_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await cb.call_async(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_failure(self):
        """Test async circuit breaker with failures."""
        cb = CircuitBreaker("test", failure_threshold=2)

        async def fail_func():
            await asyncio.sleep(0.01)
            raise ValueError("Failure")

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call_async(fail_func)

        assert cb.state == CircuitState.OPEN

    def test_half_open_max_calls(self):
        """Test HALF_OPEN enforces max calls limit."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            timeout=0.1,
            half_open_max_calls=2,
        )

        def fail_func():
            raise ValueError("Failure")

        def success_func():
            return "success"

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail_func)

        # Wait for timeout
        time.sleep(0.15)

        # Try 3 calls in HALF_OPEN (limit is 2)
        cb.call(success_func)
        cb.call(success_func)

        with pytest.raises(CircuitBreakerOpen):
            cb.call(success_func)

    def test_get_state(self):
        """Test get_state returns circuit breaker info."""
        cb = CircuitBreaker("test", failure_threshold=3)

        state = cb.get_state()

        assert state["name"] == "test"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0
        assert state["last_failure_time"] is None

    def test_reset(self):
        """Test manual reset of circuit breaker."""
        cb = CircuitBreaker("test", failure_threshold=2)

        def fail_func():
            raise ValueError("Failure")

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(fail_func)

        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_unexpected_exception_not_counted(self):
        """Test unexpected exceptions don't count as failures."""
        cb = CircuitBreaker(
            "test",
            failure_threshold=2,
            expected_exception=ValueError,
        )

        def type_error_func():
            raise TypeError("Unexpected")

        # This should raise but not count as failure
        with pytest.raises(TypeError):
            cb.call(type_error_func)

        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("test1")
        cb2 = registry.get_or_create("test1")  # Same name

        assert cb1 is cb2  # Should return same instance

        cb3 = registry.get_or_create("test2")
        assert cb3 is not cb1  # Different name

    def test_get(self):
        """Test getting existing circuit breaker."""
        registry = CircuitBreakerRegistry()

        registry.get_or_create("test")
        cb = registry.get("test")

        assert cb is not None
        assert cb.name == "test"

        cb_missing = registry.get("nonexistent")
        assert cb_missing is None

    def test_get_all_states(self):
        """Test getting all circuit breaker states."""
        registry = CircuitBreakerRegistry()

        registry.get_or_create("test1")
        registry.get_or_create("test2")

        states = registry.get_all_states()

        assert len(states) == 2
        assert "test1" in states
        assert "test2" in states

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("test1", failure_threshold=1)
        cb2 = registry.get_or_create("test2", failure_threshold=1)

        # Open both circuits
        def fail_func():
            raise ValueError("Failure")

        with pytest.raises(ValueError):
            cb1.call(fail_func)
        with pytest.raises(ValueError):
            cb2.call(fail_func)

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.OPEN

        # Reset all
        registry.reset_all()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED


class TestGlobalRegistry:
    """Tests for global circuit breaker registry."""

    def test_get_circuit_breaker(self):
        """Test global get_circuit_breaker function."""
        cb1 = get_circuit_breaker("global_test")
        cb2 = get_circuit_breaker("global_test")

        assert cb1 is cb2  # Same instance
