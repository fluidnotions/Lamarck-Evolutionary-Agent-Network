# Task 6: Error Handling & Resilience
## Implementation Plan

## Objective
Implement comprehensive error handling and resilience mechanisms including retry logic with exponential backoff, graceful degradation, error context capture, and recovery mechanisms.

## Dependencies
- Task 1: Core Infrastructure (enhances retry.py, requires ValidationState)
- All other tasks (provides error handling for entire system)

## Components to Implement

### 6.1 Retry Logic with Exponential Backoff
**File**: `src/utils/retry.py` (enhancement)
**Actions**:
- Implement configurable retry decorator
- Support exponential backoff with jitter
- Add retry budgets (max retries per validator, per workflow)
- Implement circuit breaker pattern
- Support retry on specific exception types
- Add retry metrics and logging
- Integrate with ValidationState for context-aware retries

**Innovation**: Context-aware retry that learns from previous failures in the state

### 6.2 Graceful Degradation
**File**: `src/resilience/degradation.py`
**Actions**:
- Implement graceful degradation strategies:
  - Skip failed validators (continue with others)
  - Use cached results from previous runs
  - Fall back to simpler validation methods
  - Return partial results with lower confidence
- Define degradation policies per validator
- Add degradation decision logic
- Track degradation occurrences

### 6.3 Error Context Capture
**File**: `src/resilience/error_context.py`
**Actions**:
- Capture comprehensive error context:
  - Stack traces
  - State snapshot at failure time
  - Input data that caused failure
  - System metrics (memory, CPU)
  - Previous retry attempts
- Implement error context serialization
- Add context to ErrorDetail objects
- Support PII redaction in error context

### 6.4 Recovery Mechanisms
**File**: `src/resilience/recovery.py`
**Actions**:
- Implement workflow recovery strategies:
  - Resume from last successful checkpoint
  - Replay failed validators only
  - Reset state to known good point
- Add checkpoint/snapshot functionality
- Implement state rollback logic
- Support manual intervention points

### 6.5 Circuit Breaker
**File**: `src/resilience/circuit_breaker.py`
**Actions**:
- Implement circuit breaker pattern for validators
- Track failure rates per validator
- Support states: Closed, Open, Half-Open
- Configurable failure threshold and timeout
- Add circuit breaker status monitoring

### 6.6 Error Handler Agent
**File**: `src/agents/error_handler.py`
**Actions**:
- Create dedicated error handler agent node in LangGraph
- Analyze errors and determine recovery strategy
- Use LLM to suggest fixes for common errors
- Update state with error handling decisions
- Integrate with routing for error recovery paths

## Testing Strategy

### Unit Tests
**File**: `tests/test_retry.py`
- Test retry decorator with various failure scenarios
- Test exponential backoff timing
- Test retry budget exhaustion
- Test circuit breaker triggering

**File**: `tests/test_degradation.py`
- Test degradation strategies
- Test policy application
- Test partial result handling

**File**: `tests/test_error_context.py`
- Test context capture completeness
- Test PII redaction
- Test serialization/deserialization

**File**: `tests/test_recovery.py`
- Test checkpoint creation and restoration
- Test state rollback
- Test replay logic

**File**: `tests/test_circuit_breaker.py`
- Test circuit breaker state transitions
- Test failure rate calculation
- Test timeout behavior

### Integration Tests
**File**: `tests/test_resilience_integration.py`
- Test retry in full workflow
- Test graceful degradation with real validators
- Test error recovery in LangGraph
- Test circuit breaker preventing cascading failures

### Chaos Tests
**File**: `tests/test_chaos.py`
- Inject random failures and test recovery
- Test concurrent failures
- Test resource exhaustion scenarios
- Test network timeout simulation

## Technical Specifications

### Retry Decorator with Context
```python
from functools import wraps
import asyncio
import random
from typing import Callable, Type

class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_on: tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_on = retry_on

def retry_with_context(config: RetryConfig):
    """
    Retry decorator that uses context from ValidationState.

    Checks state for previous retry attempts and adjusts behavior.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(state: ValidationState, *args, **kwargs):
            attempt = 0
            last_exception = None

            # Check retry budget from state
            retry_count = state.get("metadata", {}).get("retry_count", 0)
            if retry_count >= config.max_attempts:
                raise MaxRetriesExceeded(f"Exceeded retry budget: {retry_count}")

            while attempt < config.max_attempts:
                try:
                    return await func(state, *args, **kwargs)
                except config.retry_on as e:
                    last_exception = e
                    attempt += 1

                    if attempt >= config.max_attempts:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay = delay * (0.5 + random.random())

                    # Log retry attempt
                    logger.warning(
                        f"Retry attempt {attempt}/{config.max_attempts} "
                        f"after {delay:.2f}s delay. Error: {e}"
                    )

                    # Update state with retry info
                    state.setdefault("metadata", {})["retry_count"] = retry_count + attempt
                    state.setdefault("metadata", {})["last_error"] = str(e)

                    await asyncio.sleep(delay)

            raise last_exception

        return wrapper
    return decorator
```

### Circuit Breaker
```python
from enum import Enum
from datetime import datetime, timedelta
from threading import Lock

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for validator resilience."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self._lock = Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpen("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            else:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
```

### Graceful Degradation
```python
class DegradationStrategy:
    """Defines how to degrade gracefully on failures."""

    def __init__(self, name: str):
        self.name = name

    def apply(self, state: ValidationState, failed_validators: list[str]) -> ValidationState:
        """Apply degradation strategy to state."""
        raise NotImplementedError

class SkipFailedValidators(DegradationStrategy):
    """Continue with successful validators, skip failed ones."""

    def apply(self, state: ValidationState, failed_validators: list[str]) -> ValidationState:
        new_state = state.copy()
        # Mark failed validators as skipped
        for validator in failed_validators:
            new_state["validation_results"].append(ValidationResult(
                validator_name=validator,
                status="skipped",
                errors=[ErrorDetail(
                    path="",
                    message=f"Validator skipped due to failure",
                    severity="warning",
                    code="degraded"
                )]
            ))
        return new_state

class UseCachedResults(DegradationStrategy):
    """Use cached results from previous successful runs."""

    def __init__(self, cache):
        super().__init__("use_cached")
        self.cache = cache

    def apply(self, state: ValidationState, failed_validators: list[str]) -> ValidationState:
        new_state = state.copy()
        for validator in failed_validators:
            cached_result = self.cache.get(validator, state["input_data"])
            if cached_result:
                new_state["validation_results"].append(cached_result)
        return new_state
```

## Innovation Highlights

1. **Context-Aware Retry**: Retry logic examines ValidationState for previous attempts and adjusts behavior
2. **Circuit Breaker per Validator**: Prevents cascading failures by isolating problematic validators
3. **Multi-Strategy Degradation**: Flexible degradation policies (skip, cache, fallback)
4. **Comprehensive Error Context**: Captures full context including state snapshots for debugging
5. **LLM-Assisted Error Recovery**: Error handler agent uses LLM to suggest fixes

## Acceptance Criteria

- ✅ Retry decorator works with exponential backoff and jitter
- ✅ Circuit breaker prevents cascading failures
- ✅ Graceful degradation allows workflow to complete partially
- ✅ Error context captured comprehensively
- ✅ Recovery mechanisms can restore workflow from failure
- ✅ Error handler agent provides useful suggestions
- ✅ All unit tests passing (>90% coverage)
- ✅ Integration tests show resilience under failures
- ✅ Chaos tests demonstrate system robustness

## Implementation Order

1. Implement retry decorator with exponential backoff
2. Implement circuit breaker pattern
3. Implement error context capture
4. Implement graceful degradation strategies
5. Implement recovery mechanisms (checkpoints, rollback)
6. Implement error handler agent
7. Write comprehensive unit tests
8. Write integration and chaos tests
9. Performance testing under failure scenarios

## Estimated Complexity
**Medium-High** - Robust error handling is challenging, testing is critical

## Notes
- Error handling should not significantly impact performance in happy path
- Circuit breaker state should be observable/monitorable
- Degradation strategies should be configurable per deployment
- Error context must handle sensitive data carefully (PII redaction)
- Consider distributed tracing integration for error tracking
- Chaos testing essential to validate resilience claims
