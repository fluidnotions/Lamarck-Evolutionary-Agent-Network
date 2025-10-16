# HVAS-Mini: Error Handling & Resilience (Task 6)

## Overview

This implementation provides comprehensive error handling and resilience mechanisms for the Hierarchical Validation Agent System (HVAS-Mini). The system is designed to handle failures gracefully, recover when possible, and continue operation even under adverse conditions.

## Components

### 1. Retry Logic (`src/utils/retry.py`)

Context-aware retry mechanism with exponential backoff:

- **RetryConfig**: Configurable retry behavior with exponential backoff and jitter
- **retry_with_context**: Decorator that integrates with ValidationState for intelligent retries
- **retry_async**: Simple async retry decorator for utility functions
- **Features**:
  - Exponential backoff with configurable base and max delays
  - Jitter to prevent thundering herd problem
  - Retry budgets to prevent excessive retries
  - Context-aware retry tracking in ValidationState
  - Support for exception-specific retry policies

### 2. Circuit Breaker (`src/resilience/circuit_breaker.py`)

Circuit breaker pattern for validator resilience:

- **CircuitBreaker**: Implements 3-state circuit breaker (CLOSED, OPEN, HALF_OPEN)
- **CircuitBreakerRegistry**: Global registry for managing multiple circuit breakers
- **Features**:
  - Configurable failure threshold and timeout
  - Automatic state transitions
  - Half-open testing with limited calls
  - Support for both sync and async functions
  - Global registry for easy access

### 3. Error Context Capture (`src/resilience/error_context.py`)

Comprehensive error context for debugging:

- **ErrorContext**: Rich error information with stack traces, state snapshots, and system metrics
- **ErrorContextCapture**: Utility for capturing error context
- **Features**:
  - Full stack trace capture
  - State snapshot at failure time
  - System metrics (CPU, memory, disk) via psutil
  - PII redaction for sensitive data
  - JSON serialization for logging
  - Retry attempt tracking

### 4. Graceful Degradation (`src/resilience/degradation.py`)

Multiple strategies for handling validator failures:

- **SkipFailedValidators**: Continue with successful validators (Level 1)
- **UseCachedResults**: Use cached results from previous runs (Level 1-2)
- **UseSimplifiedValidation**: Fall back to simpler validators (Level 2)
- **ReturnPartialResults**: Return results with reduced confidence (Level 2)
- **CompositeDegradationStrategy**: Try multiple strategies in sequence
- **Features**:
  - Degradation level tracking (0-3)
  - Multiple fallback strategies
  - Cache management for results
  - Confidence score adjustment

### 5. Recovery Mechanisms (`src/resilience/recovery.py`)

Workflow recovery with checkpoints:

- **CheckpointManager**: Manages workflow checkpoints
- **WorkflowRecovery**: Coordinates recovery strategies
- **Recovery Strategies**:
  - ResumeFromLastCheckpoint: Restore from most recent checkpoint
  - ReplayFailedValidators: Retry only failed validators
  - ResetToKnownGoodState: Find and restore last successful state
- **Features**:
  - In-memory and disk-based checkpoints
  - Automatic checkpoint creation at intervals
  - Multiple recovery strategies
  - Recovery success detection

### 6. Error Handler Agent (`src/agents/error_handler.py`)

LangGraph node for intelligent error handling:

- **ErrorHandlerAgent**: Analyzes errors and determines recovery strategy
- **error_handler_node**: LangGraph node function
- **Features**:
  - Error severity analysis (critical, moderate, minor)
  - Automatic strategy selection
  - LLM-based fix suggestions (planned)
  - Recovery recommendations
  - Integration with all resilience components

## Testing

### Unit Tests

Comprehensive unit tests for all components:

- `tests/test_retry.py`: Retry logic tests (26 tests)
- `tests/test_circuit_breaker.py`: Circuit breaker tests (18 tests)
- `tests/test_error_context.py`: Error context tests (17 tests)
- `tests/test_degradation.py`: Degradation strategy tests (19 tests)
- `tests/test_recovery.py`: Recovery mechanism tests (18 tests)
- `tests/test_error_handler.py`: Error handler agent tests (18 tests)

### Chaos Tests

Resilience validation with injected failures:

- `tests/test_chaos.py`: Chaos engineering tests (18 tests)
- Random failure injection with configurable rates
- Concurrent failure scenarios
- Resource exhaustion simulation
- Full resilience stack integration tests

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suite
uv run pytest tests/test_retry.py -v

# Run chaos tests only
uv run pytest -m chaos -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=term-missing
```

## Usage Examples

### Basic Retry

```python
from src.utils.retry import RetryConfig, retry_with_context

@retry_with_context(
    RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        exponential_base=2.0,
    )
)
async def validate_data(state: dict):
    # Your validation logic
    result = await some_api_call()
    return result
```

### Circuit Breaker

```python
from src.resilience.circuit_breaker import get_circuit_breaker

cb = get_circuit_breaker("my_validator", failure_threshold=5)

async def protected_validator():
    async def inner():
        return await validate()

    return await cb.call_async(inner)
```

### Error Handler Integration

```python
from src.agents.error_handler import error_handler_node
from langgraph.graph import StateGraph

# Add to LangGraph workflow
graph = StateGraph(ValidationState)
graph.add_node("error_handler", error_handler_node)
graph.add_edge("validator", "error_handler")
```

### Manual Recovery

```python
from src.resilience.recovery import WorkflowRecovery

recovery = WorkflowRecovery()

# Create checkpoint
recovery.create_checkpoint(state, "before_risky_operation")

try:
    # Risky operation
    result = await risky_validator(state)
except Exception:
    # Attempt recovery
    recovered_state, success = recovery.recover(state)
    if success:
        state = recovered_state
```

## Innovation Highlights

1. **Context-Aware Retry**: Retry logic examines ValidationState for previous attempts and adjusts behavior, preventing infinite retry loops and respecting workflow-level budgets.

2. **Multi-Level Degradation**: Flexible degradation with 4 levels (0-3) allowing fine-grained control over system behavior under failures.

3. **Comprehensive Error Context**: Captures full context including state snapshots, system metrics, and PII-redacted data for debugging.

4. **Intelligent Recovery**: Multiple recovery strategies with automatic selection based on failure severity and context.

5. **Circuit Breaker per Validator**: Isolates failures at validator level, preventing cascading failures across the system.

6. **Chaos Testing**: Validates resilience with randomized failure injection and concurrent load scenarios.

## Architecture Integration

This module integrates with HVAS-Mini's hierarchical architecture:

- **Supervisor Agent**: Uses circuit breakers to isolate failing validators
- **Domain Validators**: Protected by retry and circuit breaker mechanisms
- **Atomic Validators**: Implement basic retry for transient failures
- **Aggregator Agent**: Handles degraded results and partial confidence scores
- **Error Handler Agent**: Analyzes failures and applies recovery strategies

## Configuration

Key configuration parameters:

```python
# Retry Configuration
RetryConfig(
    max_attempts=3,              # Maximum retry attempts
    base_delay=1.0,              # Base delay in seconds
    max_delay=60.0,              # Maximum delay cap
    exponential_base=2.0,        # Backoff multiplier
    jitter=True,                 # Add randomness to delays
    retry_on=(Exception,),       # Exceptions to retry
    max_total_retry_time=300.0,  # Total retry time limit
)

# Circuit Breaker Configuration
CircuitBreaker(
    name="validator_name",
    failure_threshold=5,         # Failures before opening
    timeout=60.0,                # Seconds before attempting reset
    success_threshold=2,         # Successes to close circuit
    half_open_max_calls=3,       # Max calls in half-open state
)

# Degradation Configuration
UseCachedResults(
    cache=result_cache,
    max_age_seconds=3600,        # Maximum cache age
)
```

## Performance Considerations

- **Happy Path**: Minimal overhead (<1ms) when no failures occur
- **Retry**: Exponential backoff prevents excessive load
- **Circuit Breaker**: Fast-fail when open (microseconds)
- **Checkpoints**: Efficient JSON serialization, optional disk storage
- **Error Context**: Async capture with optional system metrics

## Future Enhancements

1. **LLM-Assisted Error Recovery**: Use LLM to analyze errors and suggest fixes
2. **Distributed Circuit Breakers**: Share circuit breaker state across instances
3. **Advanced Caching**: Semantic caching based on input similarity
4. **Predictive Failure Detection**: ML-based prediction of upcoming failures
5. **Auto-Tuning**: Automatically adjust retry and circuit breaker parameters
6. **Metrics & Observability**: Integration with Prometheus/Grafana

## Dependencies

- `pydantic>=2.5.0`: Data validation
- `psutil` (optional): System metrics capture
- Compatible with Python 3.11+

## License

Part of HVAS-Mini project.
