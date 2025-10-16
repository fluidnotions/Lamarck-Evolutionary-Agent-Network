"""Retry logic with exponential backoff and context awareness."""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MaxRetriesExceeded(Exception):
    """Raised when maximum retry attempts are exceeded."""

    pass


class RetryBudgetExceeded(Exception):
    """Raised when retry budget for the workflow is exceeded."""

    pass


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_range: tuple[float, float] = (0.5, 1.5),
        retry_on: tuple[Type[Exception], ...] = (Exception,),
        exclude_on: tuple[Type[Exception], ...] = (),
        max_total_retry_time: float | None = None,
    ):
        """Initialize retry configuration.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff calculation
            jitter: Whether to add jitter to prevent thundering herd
            jitter_range: Range for jitter multiplier (min, max)
            retry_on: Tuple of exception types to retry on
            exclude_on: Tuple of exception types to never retry
            max_total_retry_time: Maximum total time to spend retrying (seconds)
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.retry_on = retry_on
        self.exclude_on = exclude_on
        self.max_total_retry_time = max_total_retry_time

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # Add jitter to prevent thundering herd problem
        if self.jitter:
            jitter_multiplier = random.uniform(*self.jitter_range)
            delay = delay * jitter_multiplier

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """Determine if we should retry based on the exception.

        Args:
            exception: The exception that was raised

        Returns:
            True if we should retry, False otherwise
        """
        # Never retry excluded exceptions
        if isinstance(exception, self.exclude_on):
            return False

        # Retry if it matches retry_on exceptions
        return isinstance(exception, self.retry_on)


def retry_with_context(config: RetryConfig | None = None):
    """Retry decorator that uses context from ValidationState.

    This decorator checks the state for previous retry attempts and
    adjusts behavior accordingly. It integrates with the workflow's
    retry budget to prevent excessive retries.

    Args:
        config: Retry configuration. If None, uses default config.

    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(state: dict[str, Any], *args, **kwargs) -> T:
            """Async wrapper for retry logic."""
            attempt = 0
            last_exception = None
            start_time = time.time()

            # Check retry budget from state
            metadata = state.get("metadata", {})
            workflow_retry_count = metadata.get("retry_count", 0)
            retry_budget = metadata.get("retry_budget", float("inf"))

            if workflow_retry_count >= retry_budget:
                raise RetryBudgetExceeded(
                    f"Workflow retry budget exceeded: {workflow_retry_count}/{retry_budget}"
                )

            while attempt < config.max_attempts:
                try:
                    # Call the function
                    result = await func(state, *args, **kwargs)

                    # Success - log if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"{func.__name__} succeeded on attempt {attempt + 1}/{config.max_attempts}"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not config.should_retry(e):
                        logger.error(
                            f"{func.__name__} raised non-retryable exception: {e}"
                        )
                        raise

                    attempt += 1

                    # Check if we've exceeded max attempts
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        break

                    # Check if we've exceeded max total retry time
                    if config.max_total_retry_time is not None:
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= config.max_total_retry_time:
                            logger.error(
                                f"{func.__name__} exceeded max retry time "
                                f"({elapsed_time:.2f}s >= {config.max_total_retry_time}s)"
                            )
                            raise MaxRetriesExceeded(
                                f"Max retry time exceeded: {elapsed_time:.2f}s"
                            )

                    # Calculate delay with exponential backoff
                    delay = config.calculate_delay(attempt)

                    # Log retry attempt
                    logger.warning(
                        f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    # Update state with retry info
                    state.setdefault("metadata", {})
                    state["metadata"]["retry_count"] = workflow_retry_count + attempt
                    state["metadata"]["last_error"] = str(e)
                    state["metadata"].setdefault("retry_history", []).append({
                        "function": func.__name__,
                        "attempt": attempt,
                        "delay": delay,
                        "error": str(e),
                        "timestamp": time.time(),
                    })

                    # Wait before retrying
                    await asyncio.sleep(delay)

            # All retries exhausted
            raise MaxRetriesExceeded(
                f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
            ) from last_exception

        @wraps(func)
        def sync_wrapper(state: dict[str, Any], *args, **kwargs) -> T:
            """Sync wrapper for retry logic."""
            attempt = 0
            last_exception = None
            start_time = time.time()

            # Check retry budget from state
            metadata = state.get("metadata", {})
            workflow_retry_count = metadata.get("retry_count", 0)
            retry_budget = metadata.get("retry_budget", float("inf"))

            if workflow_retry_count >= retry_budget:
                raise RetryBudgetExceeded(
                    f"Workflow retry budget exceeded: {workflow_retry_count}/{retry_budget}"
                )

            while attempt < config.max_attempts:
                try:
                    # Call the function
                    result = func(state, *args, **kwargs)

                    # Success - log if this was a retry
                    if attempt > 0:
                        logger.info(
                            f"{func.__name__} succeeded on attempt {attempt + 1}/{config.max_attempts}"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not config.should_retry(e):
                        logger.error(
                            f"{func.__name__} raised non-retryable exception: {e}"
                        )
                        raise

                    attempt += 1

                    # Check if we've exceeded max attempts
                    if attempt >= config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts: {e}"
                        )
                        break

                    # Check if we've exceeded max total retry time
                    if config.max_total_retry_time is not None:
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= config.max_total_retry_time:
                            logger.error(
                                f"{func.__name__} exceeded max retry time "
                                f"({elapsed_time:.2f}s >= {config.max_total_retry_time}s)"
                            )
                            raise MaxRetriesExceeded(
                                f"Max retry time exceeded: {elapsed_time:.2f}s"
                            )

                    # Calculate delay with exponential backoff
                    delay = config.calculate_delay(attempt)

                    # Log retry attempt
                    logger.warning(
                        f"Retry attempt {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    # Update state with retry info
                    state.setdefault("metadata", {})
                    state["metadata"]["retry_count"] = workflow_retry_count + attempt
                    state["metadata"]["last_error"] = str(e)
                    state["metadata"].setdefault("retry_history", []).append({
                        "function": func.__name__,
                        "attempt": attempt,
                        "delay": delay,
                        "error": str(e),
                        "timestamp": time.time(),
                    })

                    # Wait before retrying
                    time.sleep(delay)

            # All retries exhausted
            raise MaxRetriesExceeded(
                f"Max retries ({config.max_attempts}) exceeded for {func.__name__}"
            ) from last_exception

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def retry_async(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple[Type[Exception], ...] = (Exception,),
):
    """Simple async retry decorator without state context.

    Use this for utility functions that don't have access to ValidationState.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on

    Returns:
        Decorated function
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        retry_on=retry_on,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            attempt = 0
            last_exception = None

            while attempt < config.max_attempts:
                try:
                    return await func(*args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    attempt += 1

                    if attempt >= config.max_attempts:
                        break

                    delay = config.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt}/{config.max_attempts} for {func.__name__} "
                        f"after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

            raise MaxRetriesExceeded(
                f"Max retries ({config.max_attempts}) exceeded"
            ) from last_exception

        return wrapper

    return decorator
