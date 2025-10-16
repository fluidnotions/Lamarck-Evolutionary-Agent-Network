"""Parallel execution engine for validators."""

import asyncio
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine

from ..models.error_detail import ErrorDetail
from ..models.validation_result import ValidationResult


class ValidatorExecutor:
    """
    Executes validators in parallel with dependency resolution.

    Handles timeouts, partial failures, and execution metrics.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        """
        Initialize validator executor.

        Args:
            timeout: Default timeout per validator in seconds
        """
        self.timeout = timeout
        self._execution_times: dict[str, float] = {}

    async def execute_validator(
        self,
        validator_func: Callable[..., ValidationResult],
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Execute a single validator with timeout.

        Args:
            validator_func: Validator function to execute
            *args: Positional arguments for validator
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Keyword arguments for validator

        Returns:
            ValidationResult from validator
        """
        validator_name = getattr(validator_func, "__name__", "unknown_validator")
        timeout = timeout or self.timeout

        try:
            # Check if validator is async
            if asyncio.iscoroutinefunction(validator_func):
                result = await asyncio.wait_for(
                    validator_func(*args, **kwargs),
                    timeout=timeout,
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: validator_func(*args, **kwargs)),
                    timeout=timeout,
                )

            self._execution_times[validator_name] = result.timing
            return result

        except asyncio.TimeoutError:
            return ValidationResult(
                validator_name=validator_name,
                status="failed",
                errors=[
                    ErrorDetail(
                        path="",
                        message=f"Validator timed out after {timeout}s",
                        severity="error",
                        code="timeout",
                        context={"timeout": timeout},
                    )
                ],
                timing=timeout,
            )

        except Exception as e:
            return ValidationResult(
                validator_name=validator_name,
                status="failed",
                errors=[
                    ErrorDetail(
                        path="",
                        message=f"Validator raised exception: {str(e)}",
                        severity="error",
                        code="exception",
                        context={"exception_type": type(e).__name__, "exception": str(e)},
                    )
                ],
            )

    async def execute_parallel(
        self,
        validators: list[tuple[Callable[..., ValidationResult], tuple[Any, ...], dict[str, Any]]],
        timeout: float | None = None,
    ) -> list[ValidationResult]:
        """
        Execute multiple validators in parallel.

        Args:
            validators: List of (validator_func, args, kwargs) tuples
            timeout: Timeout per validator in seconds

        Returns:
            List of ValidationResult objects
        """
        tasks = [
            self.execute_validator(func, *args, timeout=timeout, **kwargs)
            for func, args, kwargs in validators
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)
        return results

    async def execute_with_dependencies(
        self,
        validators: dict[str, tuple[Callable[..., ValidationResult], tuple[Any, ...], dict[str, Any], list[str]]],
        timeout: float | None = None,
    ) -> dict[str, ValidationResult]:
        """
        Execute validators respecting dependencies.

        Args:
            validators: Dict mapping validator_name to (func, args, kwargs, dependencies)
            timeout: Timeout per validator in seconds

        Returns:
            Dict mapping validator_name to ValidationResult
        """
        results: dict[str, ValidationResult] = {}
        pending = set(validators.keys())
        in_progress: set[str] = set()

        # Build reverse dependency graph
        dependents: dict[str, list[str]] = defaultdict(list)
        for name, (_, _, _, deps) in validators.items():
            for dep in deps:
                dependents[dep].append(name)

        while pending or in_progress:
            # Find validators that can be executed now
            ready = []
            to_skip = []
            for name in list(pending):  # Create a list copy to iterate safely
                _, _, _, deps = validators[name]
                if all(dep in results for dep in deps):
                    # Check if all dependencies passed
                    deps_passed = all(
                        results[dep].status != "failed"
                        for dep in deps
                        if dep in results
                    )

                    if deps_passed:
                        ready.append(name)
                    else:
                        # Dependencies failed, skip this validator
                        results[name] = ValidationResult(
                            validator_name=name,
                            status="skipped",
                            info=[
                                ErrorDetail(
                                    path="",
                                    message="Skipped due to dependency failure",
                                    severity="info",
                                    code="dependency_failed",
                                )
                            ],
                        )
                        to_skip.append(name)

            # Remove skipped validators from pending
            for name in to_skip:
                pending.remove(name)

            if not ready and not in_progress:
                # No progress can be made
                break

            # Execute ready validators in parallel
            if ready:
                tasks = []
                for name in ready:
                    func, args, kwargs, _ = validators[name]
                    tasks.append((name, self.execute_validator(func, *args, timeout=timeout, **kwargs)))

                in_progress.update([name for name, _ in tasks])
                pending.difference_update([name for name, _ in tasks])

                # Wait for all tasks to complete
                task_results = await asyncio.gather(*[task for _, task in tasks])

                for (name, _), result in zip(tasks, task_results):
                    results[name] = result
                    in_progress.remove(name)

        return results

    def get_execution_metrics(self) -> dict[str, Any]:
        """
        Get execution metrics for all validators.

        Returns:
            Dict with execution metrics
        """
        if not self._execution_times:
            return {
                "total_validators": 0,
                "total_time": 0.0,
                "average_time": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
            }

        times = list(self._execution_times.values())
        return {
            "total_validators": len(times),
            "total_time": sum(times),
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "validators": dict(self._execution_times),
        }


# Convenience functions
async def execute_validators_parallel(
    validators: list[tuple[Callable[..., ValidationResult], tuple[Any, ...], dict[str, Any]]],
    timeout: float = 30.0,
) -> list[ValidationResult]:
    """
    Execute multiple validators in parallel.

    Args:
        validators: List of (validator_func, args, kwargs) tuples
        timeout: Timeout per validator in seconds

    Returns:
        List of ValidationResult objects
    """
    executor = ValidatorExecutor(timeout=timeout)
    return await executor.execute_parallel(validators, timeout=timeout)


async def execute_validators_sequential(
    validators: list[tuple[Callable[..., ValidationResult], tuple[Any, ...], dict[str, Any]]],
    timeout: float = 30.0,
    stop_on_failure: bool = False,
) -> list[ValidationResult]:
    """
    Execute validators sequentially.

    Args:
        validators: List of (validator_func, args, kwargs) tuples
        timeout: Timeout per validator in seconds
        stop_on_failure: Whether to stop execution on first failure

    Returns:
        List of ValidationResult objects
    """
    executor = ValidatorExecutor(timeout=timeout)
    results = []

    for func, args, kwargs in validators:
        result = await executor.execute_validator(func, *args, timeout=timeout, **kwargs)
        results.append(result)

        if stop_on_failure and result.status == "failed":
            break

    return results


def aggregate_results(results: list[ValidationResult]) -> ValidationResult:
    """
    Aggregate multiple validation results into a single result.

    Args:
        results: List of ValidationResult objects

    Returns:
        Aggregated ValidationResult
    """
    all_errors = []
    all_warnings = []
    all_info = []
    total_time = 0.0

    for result in results:
        all_errors.extend(result.errors)
        all_warnings.extend(result.warnings)
        all_info.extend(result.info)
        total_time += result.timing

    # Overall status: failed if any validator failed
    status = "passed"
    for result in results:
        if result.status == "failed":
            status = "failed"
            break

    return ValidationResult(
        validator_name="aggregated",
        status=status,
        errors=all_errors,
        warnings=all_warnings,
        info=all_info,
        timing=total_time,
        metadata={
            "validators_count": len(results),
            "passed_count": sum(1 for r in results if r.status == "passed"),
            "failed_count": sum(1 for r in results if r.status == "failed"),
            "skipped_count": sum(1 for r in results if r.status == "skipped"),
        },
    )
