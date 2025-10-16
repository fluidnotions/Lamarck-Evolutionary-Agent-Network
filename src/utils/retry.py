"""Retry logic and error handling utilities."""
import time
import logging
from typing import Any, Callable, Optional, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter to delays
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds
    """
    import random

    # Calculate exponential delay
    delay = min(
        config.initial_delay * (config.exponential_base**attempt), config.max_delay
    )

    # Add jitter if enabled
    if config.jitter:
        delay = delay * (0.5 + random.random())

    return delay


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        config: Retry configuration (uses defaults if None)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback called on each retry attempt

    Returns:
        Decorator function
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"Retry attempt {attempt + 1}/{config.max_retries} "
                            f"for {func.__name__} after error: {e}. "
                            f"Waiting {delay:.2f}s"
                        )

                        if on_retry:
                            on_retry(attempt, e)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Max retries ({config.max_retries}) exceeded for {func.__name__}"
                        )

            # If we get here, all retries failed
            raise last_exception  # type: ignore

        return wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes needed to close circuit
            timeout: Time in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.failures = 0
        self.successes = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half_open

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            # Check if timeout has elapsed
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.timeout
            ):
                self.state = "half_open"
                self.successes = 0
                logger.info("Circuit breaker entering half-open state")
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failures = 0

        if self.state == "half_open":
            self.successes += 1
            if self.successes >= self.success_threshold:
                self.state = "closed"
                logger.info("Circuit breaker closed after successful recovery")

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.state == "half_open":
            self.state = "open"
            logger.warning("Circuit breaker opened from half-open state")
        elif self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failures} failures"
            )


def with_timeout(seconds: float) -> Callable:
    """Decorator to add timeout to function execution.

    Args:
        seconds: Timeout in seconds

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import signal

            def timeout_handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds}s")

            # Set the signal handler and alarm
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))

            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)

            return result

        return wrapper

    return decorator
