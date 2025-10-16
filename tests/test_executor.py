"""Tests for parallel execution engine."""

import asyncio

import pytest

from src.models.error_detail import ErrorDetail
from src.models.validation_result import ValidationResult
from src.validators.executor import (
    ValidatorExecutor,
    aggregate_results,
    execute_validators_parallel,
    execute_validators_sequential,
)


def simple_validator(data: dict) -> ValidationResult:
    """Simple test validator."""
    return ValidationResult(
        validator_name="simple_validator",
        status="passed",
        timing=0.01,
    )


def failing_validator(data: dict) -> ValidationResult:
    """Validator that fails."""
    return ValidationResult(
        validator_name="failing_validator",
        status="failed",
        errors=[ErrorDetail(path="test", message="Test error")],
    )


def slow_validator(data: dict) -> ValidationResult:
    """Slow validator for timeout testing."""
    import time
    time.sleep(2)
    return ValidationResult(
        validator_name="slow_validator",
        status="passed",
    )


def exception_validator(data: dict) -> ValidationResult:
    """Validator that raises exception."""
    raise ValueError("Test exception")


async def async_validator(data: dict) -> ValidationResult:
    """Async validator."""
    await asyncio.sleep(0.01)
    return ValidationResult(
        validator_name="async_validator",
        status="passed",
    )


class TestValidatorExecutor:
    """Test validator executor."""

    @pytest.mark.asyncio
    async def test_execute_simple_validator(self) -> None:
        """Test executing a simple validator."""
        executor = ValidatorExecutor()
        result = await executor.execute_validator(simple_validator, {})

        assert result.status == "passed"
        assert result.validator_name == "simple_validator"

    @pytest.mark.asyncio
    async def test_execute_async_validator(self) -> None:
        """Test executing an async validator."""
        executor = ValidatorExecutor()
        result = await executor.execute_validator(async_validator, {})

        assert result.status == "passed"
        assert result.validator_name == "async_validator"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self) -> None:
        """Test validator timeout."""
        executor = ValidatorExecutor(timeout=1.0)
        result = await executor.execute_validator(slow_validator, {}, timeout=1.0)

        assert result.status == "failed"
        assert len(result.errors) > 0
        assert "timeout" in result.errors[0].code.lower()

    @pytest.mark.asyncio
    async def test_execute_with_exception(self) -> None:
        """Test validator that raises exception."""
        executor = ValidatorExecutor()
        result = await executor.execute_validator(exception_validator, {})

        assert result.status == "failed"
        assert len(result.errors) > 0
        assert "exception" in result.errors[0].code.lower()

    @pytest.mark.asyncio
    async def test_execute_parallel(self) -> None:
        """Test executing multiple validators in parallel."""
        executor = ValidatorExecutor()

        validators = [
            (simple_validator, ({},), {}),
            (simple_validator, ({},), {}),
            (simple_validator, ({},), {}),
        ]

        results = await executor.execute_parallel(validators)

        assert len(results) == 3
        assert all(r.status == "passed" for r in results)

    @pytest.mark.asyncio
    async def test_execute_parallel_with_failures(self) -> None:
        """Test parallel execution with some failures."""
        executor = ValidatorExecutor()

        validators = [
            (simple_validator, ({},), {}),
            (failing_validator, ({},), {}),
            (simple_validator, ({},), {}),
        ]

        results = await executor.execute_parallel(validators)

        assert len(results) == 3
        assert sum(1 for r in results if r.status == "passed") == 2
        assert sum(1 for r in results if r.status == "failed") == 1

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self) -> None:
        """Test executing validators with dependencies."""
        executor = ValidatorExecutor()

        validators = {
            "validator1": (simple_validator, ({},), {}, []),
            "validator2": (simple_validator, ({},), {}, ["validator1"]),
            "validator3": (simple_validator, ({},), {}, ["validator1", "validator2"]),
        }

        results = await executor.execute_with_dependencies(validators)

        assert len(results) == 3
        assert all(r.status == "passed" for r in results.values())

    @pytest.mark.asyncio
    async def test_execute_with_failed_dependency(self) -> None:
        """Test that dependent validators are skipped when dependency fails."""
        executor = ValidatorExecutor()

        validators = {
            "validator1": (failing_validator, ({},), {}, []),
            "validator2": (simple_validator, ({},), {}, ["validator1"]),
        }

        results = await executor.execute_with_dependencies(validators)

        assert results["validator1"].status == "failed"
        assert results["validator2"].status == "skipped"

    def test_get_execution_metrics(self) -> None:
        """Test getting execution metrics."""
        executor = ValidatorExecutor()
        executor._execution_times = {
            "validator1": 0.1,
            "validator2": 0.2,
            "validator3": 0.15,
        }

        metrics = executor.get_execution_metrics()

        assert metrics["total_validators"] == 3
        assert metrics["total_time"] == 0.45
        assert metrics["min_time"] == 0.1
        assert metrics["max_time"] == 0.2

    def test_get_execution_metrics_empty(self) -> None:
        """Test getting metrics with no executions."""
        executor = ValidatorExecutor()
        metrics = executor.get_execution_metrics()

        assert metrics["total_validators"] == 0
        assert metrics["total_time"] == 0.0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_validators_parallel(self) -> None:
        """Test parallel execution convenience function."""
        validators = [
            (simple_validator, ({},), {}),
            (simple_validator, ({},), {}),
        ]

        results = await execute_validators_parallel(validators)

        assert len(results) == 2
        assert all(r.status == "passed" for r in results)

    @pytest.mark.asyncio
    async def test_execute_validators_sequential(self) -> None:
        """Test sequential execution convenience function."""
        validators = [
            (simple_validator, ({},), {}),
            (simple_validator, ({},), {}),
        ]

        results = await execute_validators_sequential(validators)

        assert len(results) == 2
        assert all(r.status == "passed" for r in results)

    @pytest.mark.asyncio
    async def test_execute_sequential_stop_on_failure(self) -> None:
        """Test sequential execution with stop on failure."""
        validators = [
            (failing_validator, ({},), {}),
            (simple_validator, ({},), {}),
            (simple_validator, ({},), {}),
        ]

        results = await execute_validators_sequential(validators, stop_on_failure=True)

        # Should stop after first failure
        assert len(results) == 1
        assert results[0].status == "failed"

    def test_aggregate_results_all_passed(self) -> None:
        """Test aggregating results when all pass."""
        results = [
            ValidationResult("validator1", "passed", timing=0.1),
            ValidationResult("validator2", "passed", timing=0.2),
        ]

        aggregated = aggregate_results(results)

        assert aggregated.status == "passed"
        assert abs(aggregated.timing - 0.3) < 0.0001  # Use approximate comparison for floats
        assert aggregated.metadata["validators_count"] == 2
        assert aggregated.metadata["passed_count"] == 2

    def test_aggregate_results_with_failures(self) -> None:
        """Test aggregating results with failures."""
        results = [
            ValidationResult("validator1", "passed"),
            ValidationResult(
                "validator2",
                "failed",
                errors=[ErrorDetail(path="test", message="Error")],
            ),
        ]

        aggregated = aggregate_results(results)

        assert aggregated.status == "failed"
        assert len(aggregated.errors) == 1
        assert aggregated.metadata["failed_count"] == 1
