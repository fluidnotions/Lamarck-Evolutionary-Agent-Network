"""Retry utilities with exponential backoff for HVAS-Mini.

This module provides retry decorators and utilities for handling transient
failures in validation operations. It supports exponential backoff, jitter,
and context-aware retry strategies.
"""

import functools
import random
import time
from typing import Any, Callable, Optional, Type, TypeVar, cast

from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryError(Exception):
    """Exception raised when all retry attempts are exhausted.

    Attributes:
        attempts: Number of attempts made
        last_exception: The final exception that caused the failure
    """

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Failed after {attempts} attempts. Last error: {last_exception}"
        )


def calculate_delay(
    attempt: int,
    initial_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """Calculate delay for the next retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Delay in seconds before next retry
    """
    # Calculate exponential delay
    delay = initial_delay * (exponential_base ** attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter if enabled (random value between 0 and delay)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


def should_retry(
    exception: Exception,
    retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
) -> bool:
    """Determine if an exception should trigger a retry.

    Args:
        exception: Exception that occurred
        retryable_exceptions: Tuple of exception types that are retryable

    Returns:
        True if the exception should trigger a retry, False otherwise
    """
    if retryable_exceptions:
        return isinstance(exception, retryable_exceptions)

    # Default retryable exceptions (network, timeouts, rate limits)
    default_retryable = (
        TimeoutError,
        ConnectionError,
    )

    # Check exception type name for common patterns
    exception_name = type(exception).__name__
    retryable_patterns = [
        "Timeout",
        "Connection",
        "RateLimit",
        "ServiceUnavailable",
        "TooManyRequests",
    ]

    if isinstance(exception, default_retryable):
        return True

    for pattern in retryable_patterns:
        if pattern in exception_name:
            return True

    return False


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Optional[tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Callback function called on each retry (exception, attempt)

    Returns:
        Decorated function with retry logic

    Example:
        @retry(max_attempts=3, initial_delay=1.0)
        def fetch_data():
            # code that might fail
            pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    logger.debug(
                        f"Attempting {func.__name__} (attempt {attempt + 1}/{max_attempts})"
                    )
                    result = func(*args, **kwargs)

                    if attempt > 0:
                        logger.info(
                            f"Successfully executed {func.__name__} after {attempt + 1} attempts"
                        )

                    return result

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    if not should_retry(e, retryable_exceptions):
                        logger.warning(
                            f"Non-retryable exception in {func.__name__}: {e}"
                        )
                        raise

                    # Check if we have more attempts
                    if attempt < max_attempts - 1:
                        delay = calculate_delay(
                            attempt, initial_delay, max_delay, exponential_base, jitter
                        )

                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )

                        # Call retry callback if provided
                        if on_retry:
                            on_retry(e, attempt + 1)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}: {e}"
                        )

            # All attempts exhausted
            if last_exception:
                raise RetryError(max_attempts, last_exception)

            # Should never reach here
            raise RuntimeError("Retry logic error: no exception but no result")

        return wrapper

    return decorator


class RetryContext:
    """Context manager for retry operations with state tracking.

    This class provides a context-aware retry mechanism that can
    examine previous failures and adjust retry strategy accordingly.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter
        attempt_history: History of previous attempts and failures
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry context.

        Args:
            max_attempts: Maximum number of attempts
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.attempt_history: list[dict[str, Any]] = []
        self.current_attempt = 0

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function

        Raises:
            RetryError: If all attempts fail
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            self.current_attempt = attempt
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Record successful attempt
                self.attempt_history.append({
                    "attempt": attempt + 1,
                    "success": True,
                    "duration": time.time() - start_time,
                })

                return result

            except Exception as e:
                last_exception = e
                duration = time.time() - start_time

                # Record failed attempt
                self.attempt_history.append({
                    "attempt": attempt + 1,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration,
                })

                if not should_retry(e):
                    raise

                if attempt < self.max_attempts - 1:
                    delay = self._calculate_adaptive_delay(attempt)
                    logger.info(
                        f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                        extra={
                            "error": str(e),
                            "attempt": attempt + 1,
                            "max_attempts": self.max_attempts,
                        },
                    )
                    time.sleep(delay)

        if last_exception:
            raise RetryError(self.max_attempts, last_exception)

        raise RuntimeError("Retry logic error")

    def _calculate_adaptive_delay(self, attempt: int) -> float:
        """Calculate adaptive delay based on attempt history.

        Adjusts delay based on patterns in previous failures.

        Args:
            attempt: Current attempt number

        Returns:
            Delay in seconds
        """
        base_delay = calculate_delay(
            attempt,
            self.initial_delay,
            self.max_delay,
            self.exponential_base,
            self.jitter,
        )

        # If we've seen consistent failures, increase delay
        if len(self.attempt_history) >= 2:
            recent_failures = [
                h for h in self.attempt_history[-2:]
                if not h["success"]
            ]
            if len(recent_failures) == 2:
                # Consecutive failures - increase delay by 50%
                base_delay *= 1.5

        return min(base_delay, self.max_delay)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of retry attempts.

        Returns:
            Dictionary containing retry statistics
        """
        total_attempts = len(self.attempt_history)
        successful = sum(1 for h in self.attempt_history if h["success"])
        failed = total_attempts - successful

        return {
            "total_attempts": total_attempts,
            "successful": successful,
            "failed": failed,
            "attempt_history": self.attempt_history,
        }
