"""Circuit breaker pattern implementation for validator resilience."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing recovery, allow limited requests


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejects requests."""

    pass


class CircuitBreaker:
    """Circuit breaker for validator resilience.

    The circuit breaker tracks failure rates and prevents cascading failures
    by "opening" when too many failures occur, temporarily blocking requests.

    States:
        - CLOSED: Normal operation, all requests pass through
        - OPEN: Too many failures, reject requests immediately
        - HALF_OPEN: Testing recovery, allow limited requests to check if service recovered
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2,
        expected_exception: type[Exception] = Exception,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker (usually validator name)
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset (OPEN -> HALF_OPEN)
            success_threshold: Number of successes needed in HALF_OPEN to close circuit
            expected_exception: Exception type that counts as a failure
            half_open_max_calls: Maximum calls to allow in HALF_OPEN state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.expected_exception = expected_exception
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time: datetime | None = None
        self._lock = Lock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, timeout={timeout}s"
        )

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection (sync).

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception raised by func
        """
        # Check if we should allow the call
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' is open. "
                        f"Last failure: {self.last_failure_time}"
                    )

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' HALF_OPEN call limit reached"
                    )
                self.half_open_calls += 1

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
        except Exception as e:
            # Unexpected exception, don't count as failure
            logger.warning(
                f"Circuit breaker '{self.name}' encountered unexpected exception: {e}"
            )
            raise

    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception raised by func
        """
        # Check if we should allow the call
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self.half_open_calls = 0
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' is open. "
                        f"Last failure: {self.last_failure_time}"
                    )

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        f"Circuit breaker '{self.name}' HALF_OPEN call limit reached"
                    )
                self.half_open_calls += 1

        # Execute the function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
        except Exception as e:
            # Unexpected exception, don't count as failure
            logger.warning(
                f"Circuit breaker '{self.name}' encountered unexpected exception: {e}"
            )
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                logger.info(
                    f"Circuit breaker '{self.name}' HALF_OPEN success "
                    f"{self.success_count}/{self.success_threshold}"
                )
                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit breaker '{self.name}' closing (recovered)")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == CircuitState.HALF_OPEN:
                # Failure in HALF_OPEN -> back to OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' failed in HALF_OPEN, "
                    "reopening circuit"
                )
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.CLOSED:
                # Check if we should open
                if self.failure_count >= self.failure_threshold:
                    logger.error(
                        f"Circuit breaker '{self.name}' opening after "
                        f"{self.failure_count} failures"
                    )
                    self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset.

        Returns:
            True if we should attempt reset (OPEN -> HALF_OPEN)
        """
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state.

        Returns:
            Dictionary with state information
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": (
                    self.last_failure_time.isoformat()
                    if self.last_failure_time
                    else None
                ),
                "seconds_since_last_failure": (
                    (datetime.now() - self.last_failure_time).total_seconds()
                    if self.last_failure_time
                    else None
                ),
            }

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state."""
        with self._lock:
            logger.info(f"Circuit breaker '{self.name}' manually reset")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = Lock()

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one.

        Args:
            name: Circuit breaker name
            failure_threshold: Failure threshold for new breaker
            timeout: Timeout for new breaker
            success_threshold: Success threshold for new breaker

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    success_threshold=success_threshold,
                )
            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name.

        Args:
            name: Circuit breaker name

        Returns:
            CircuitBreaker instance or None if not found
        """
        with self._lock:
            return self._breakers.get(name)

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get state of all circuit breakers.

        Returns:
            Dictionary mapping names to state dicts
        """
        with self._lock:
            return {name: breaker.get_state() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global circuit breaker registry
_global_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    success_threshold: int = 2,
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry.

    Args:
        name: Circuit breaker name
        failure_threshold: Failure threshold for new breaker
        timeout: Timeout for new breaker
        success_threshold: Success threshold for new breaker

    Returns:
        CircuitBreaker instance
    """
    return _global_registry.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=success_threshold,
    )
