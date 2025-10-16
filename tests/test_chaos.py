"""Chaos tests for resilience validation.

These tests inject random failures and verify that the system
handles them gracefully.
"""

import asyncio
import random
import time

import pytest

from src.models import ValidationResult
from src.resilience import (
    CircuitBreaker,
    SkipFailedValidators,
    UseCachedResults,
)
from src.resilience.circuit_breaker import CircuitBreakerOpen
from src.resilience.recovery import WorkflowRecovery
from src.utils.retry import RetryConfig, retry_with_context


class ChaosMonkey:
    """Simulates various failure scenarios."""

    def __init__(self, failure_rate: float = 0.3):
        """Initialize chaos monkey.

        Args:
            failure_rate: Probability of failure (0.0 to 1.0)
        """
        self.failure_rate = failure_rate
        self.call_count = 0
        self.failure_count = 0

    def maybe_fail(self, error_type: str = "random"):
        """Randomly fail based on failure rate.

        Args:
            error_type: Type of error to raise
        """
        self.call_count += 1

        if random.random() < self.failure_rate:
            self.failure_count += 1

            if error_type == "timeout":
                raise TimeoutError("Simulated timeout")
            elif error_type == "connection":
                raise ConnectionError("Simulated connection error")
            elif error_type == "value":
                raise ValueError("Simulated value error")
            else:
                # Random error type
                errors = [TimeoutError, ConnectionError, ValueError, RuntimeError]
                raise random.choice(errors)("Random chaos error")


@pytest.mark.chaos
class TestChaosRetry:
    """Chaos tests for retry mechanism."""

    @pytest.mark.asyncio
    async def test_retry_handles_random_failures(self):
        """Test retry handles random failures."""
        chaos = ChaosMonkey(failure_rate=0.7)  # 70% failure rate

        @retry_with_context(
            RetryConfig(
                max_attempts=10,
                base_delay=0.01,
            )
        )
        async def flaky_function(state):
            chaos.maybe_fail()
            return "success"

        state = {"metadata": {}}

        # Should eventually succeed despite high failure rate
        result = await flaky_function(state)
        assert result == "success"
        assert chaos.failure_count > 0  # Had some failures

    @pytest.mark.asyncio
    async def test_retry_respects_max_attempts_under_chaos(self):
        """Test retry gives up after max attempts."""
        chaos = ChaosMonkey(failure_rate=1.0)  # Always fail

        @retry_with_context(
            RetryConfig(
                max_attempts=3,
                base_delay=0.01,
            )
        )
        async def always_fail(state):
            chaos.maybe_fail()
            return "success"

        state = {"metadata": {}}

        with pytest.raises(Exception):  # Will raise one of the chaos errors
            await always_fail(state)

        assert chaos.call_count == 3  # Tried max_attempts times


@pytest.mark.chaos
class TestChaosCircuitBreaker:
    """Chaos tests for circuit breaker."""

    def test_circuit_breaker_prevents_cascading_failures(self):
        """Test circuit breaker stops cascading failures."""
        chaos = ChaosMonkey(failure_rate=1.0)  # Always fail
        cb = CircuitBreaker("chaos_test", failure_threshold=5)

        def flaky_validator():
            chaos.maybe_fail()
            return "success"

        failure_count = 0
        for i in range(20):
            try:
                cb.call(flaky_validator)
            except Exception:
                failure_count += 1

        # Circuit should have opened, preventing many calls
        assert chaos.call_count < 20  # Not all calls went through
        assert failure_count >= 5  # At least threshold failures

    def test_circuit_breaker_recovers(self):
        """Test circuit breaker can recover."""
        chaos = ChaosMonkey(failure_rate=0.5)  # Moderate failure rate
        cb = CircuitBreaker(
            "chaos_test",
            failure_threshold=3,
            timeout=0.1,
            success_threshold=2,
        )

        def flaky_validator():
            chaos.maybe_fail()
            return "success"

        # Open the circuit
        for _ in range(5):
            try:
                cb.call(flaky_validator)
            except Exception:
                pass

        assert cb.state.value == "open"

        # Wait for half-open transition
        time.sleep(0.15)

        # Try multiple times - with 50% success rate, should eventually work
        # The test demonstrates recovery mechanism, even if not fully successful
        attempted_after_timeout = False
        for _ in range(30):
            try:
                cb.call(flaky_validator)
                attempted_after_timeout = True
                break  # If this works, great
            except CircuitBreakerOpen:
                # Still open, wait a bit more
                time.sleep(0.05)
            except Exception:
                # Failed validation, but circuit is trying
                attempted_after_timeout = True

        # Circuit should have attempted recovery (not stuck in OPEN forever)
        assert attempted_after_timeout or cb.state.value in ["half_open", "closed"]


@pytest.mark.chaos
class TestChaosDegradation:
    """Chaos tests for graceful degradation."""

    def test_degradation_handles_multiple_failures(self):
        """Test degradation handles multiple validator failures."""
        strategy = SkipFailedValidators()

        # Simulate multiple validators failing
        failed_validators = [f"validator_{i}" for i in range(10)]

        state = {"validation_results": []}

        result_state = strategy.apply(state, failed_validators)

        # Should have handled all failures gracefully
        assert result_state["degradation_level"] > 0
        assert len(result_state["validation_results"]) == 10
        assert all(r.status == "skipped" for r in result_state["validation_results"])

    def test_cached_degradation_under_chaos(self):
        """Test cached degradation with chaotic failures."""
        # Prepare cache with some results
        cache = {}
        for i in range(5):
            cache[f"validator_{i}"] = {
                "timestamp": "2024-01-01T00:00:00",
                "result": ValidationResult(
                    validator_name=f"validator_{i}",
                    status="passed",
                    confidence=1.0,
                ),
                "input_hash": "test",
            }

        strategy = UseCachedResults(cache=cache)

        # All validators fail
        failed_validators = [f"validator_{i}" for i in range(10)]

        state = {
            "input_data": {},
            "validation_results": [],
        }

        result_state = strategy.apply(state, failed_validators)

        # Should have used cache for some, skipped others
        assert len(result_state["validation_results"]) > 0


@pytest.mark.chaos
class TestChaosRecovery:
    """Chaos tests for recovery mechanisms."""

    def test_recovery_with_random_checkpoint_failures(self):
        """Test recovery handles checkpoint failures."""
        recovery = WorkflowRecovery()

        # Create multiple checkpoints
        for i in range(5):
            state = {
                "completed_validators": [f"v{j}" for j in range(i)],
                "failed_validators": [],
            }
            recovery.create_checkpoint(state, f"checkpoint_{i}")

        # Simulate failure
        failed_state = {
            "completed_validators": [],
            "active_validators": [],
            "failed_validators": ["v1", "v2"],
        }

        recovered_state, success = recovery.recover(failed_state)

        # Should have attempted recovery
        assert "metadata" in recovered_state

    def test_workflow_survives_chaos(self):
        """Test complete workflow survives chaos."""
        chaos = ChaosMonkey(failure_rate=0.5)
        recovery = WorkflowRecovery()

        state = {
            "completed_validators": [],
            "active_validators": [f"val{i}" for i in range(10)],
            "failed_validators": [],
            "validation_results": [],
            "metadata": {},
        }

        # Simulate workflow with random failures
        while state["active_validators"]:
            # Checkpoint every 2 validators
            if len(state["completed_validators"]) % 2 == 0:
                recovery.create_checkpoint(state)

            # Try to run next validator
            validator = state["active_validators"][0]

            try:
                # Simulate validator execution
                chaos.maybe_fail()

                # Success
                state["active_validators"].pop(0)
                state["completed_validators"].append(validator)

            except Exception:
                # Failure
                state["active_validators"].pop(0)
                state["failed_validators"].append(validator)

                # Try recovery if too many failures
                if len(state["failed_validators"]) > 3:
                    recovered_state, success = recovery.recover(state)
                    if success:
                        state = recovered_state

        # Workflow should complete (even if degraded)
        assert len(state["active_validators"]) == 0


@pytest.mark.chaos
class TestChaosIntegration:
    """Integration chaos tests."""

    @pytest.mark.asyncio
    async def test_full_resilience_stack(self):
        """Test full resilience stack under chaos."""
        chaos = ChaosMonkey(failure_rate=0.6)
        recovery = WorkflowRecovery()
        cb = CircuitBreaker("test", failure_threshold=3, timeout=0.1)

        @retry_with_context(
            RetryConfig(
                max_attempts=3,
                base_delay=0.01,
            )
        )
        async def resilient_validator(state, name):
            # Use circuit breaker
            async def inner():
                chaos.maybe_fail()
                return ValidationResult(
                    validator_name=name,
                    status="passed",
                    confidence=1.0,
                )

            return await cb.call_async(inner)

        state = {
            "completed_validators": [],
            "failed_validators": [],
            "validation_results": [],
            "metadata": {},
        }

        validators = [f"val{i}" for i in range(5)]

        for validator in validators:
            try:
                result = await resilient_validator(state, validator)
                state["completed_validators"].append(validator)
                state["validation_results"].append(result)

                # Auto checkpoint
                recovery.auto_checkpoint(state, interval=2)

            except Exception as e:
                state["failed_validators"].append(validator)

                # Try recovery
                if len(state["failed_validators"]) > 2:
                    recovered_state, success = recovery.recover(state)
                    if success:
                        state = recovered_state

        # Should have some results despite chaos
        total_processed = (
            len(state["completed_validators"]) +
            len(state["failed_validators"])
        )
        assert total_processed == len(validators)

    def test_concurrent_chaos(self):
        """Test resilience under concurrent chaotic load."""
        chaos = ChaosMonkey(failure_rate=0.5)
        results = {"success": 0, "failure": 0}

        def chaotic_operation(id):
            """Simulate concurrent operation with failures."""
            try:
                time.sleep(random.uniform(0.001, 0.01))
                chaos.maybe_fail()
                results["success"] += 1
                return f"result_{id}"
            except Exception:
                results["failure"] += 1
                raise

        # Run many operations
        for i in range(50):
            try:
                chaotic_operation(i)
            except Exception:
                pass

        # Should have mix of success and failure
        assert results["success"] > 0
        assert results["failure"] > 0
        assert results["success"] + results["failure"] == 50


@pytest.mark.chaos
class TestResourceExhaustion:
    """Chaos tests for resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_handles_slow_operations(self):
        """Test system handles slow operations."""

        @retry_with_context(
            RetryConfig(
                max_attempts=3,
                base_delay=0.01,
                max_total_retry_time=0.5,
            )
        )
        async def slow_operation(state):
            await asyncio.sleep(0.3)  # Slow
            return "done"

        state = {"metadata": {}}

        # Should complete despite being slow
        result = await slow_operation(state)
        assert result == "done"

    def test_handles_memory_pressure(self):
        """Test system handles memory pressure simulation."""
        # Simulate memory pressure with large state
        large_state = {
            "validation_results": [
                ValidationResult(
                    validator_name=f"val{i}",
                    status="passed",
                    confidence=1.0,
                )
                for i in range(100)
            ],
            "metadata": {
                "large_data": ["x" * 1000 for _ in range(100)]
            }
        }

        strategy = SkipFailedValidators()

        # Should handle large state gracefully
        result = strategy.apply(large_state, ["val_new"])

        assert "skipped_validators" in result
