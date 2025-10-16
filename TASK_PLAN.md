# Task 1: Core Infrastructure & State Management
## Implementation Plan

## Objective
Establish the foundational infrastructure for HVAS-Mini, including state schemas, base agent classes, and project structure.

## Dependencies
- None (foundation task)

## Components to Implement

### 1.1 Project Structure Setup
**File**: Project root
**Actions**:
- Create all required directories (src/, tests/, docs/, examples/)
- Setup pyproject.toml with UV configuration
- Configure dependencies: langgraph, langchain, openai/anthropic
- Add .gitignore for Python projects
- Setup __init__.py files for proper module structure

### 1.2 State Schema Definitions
**File**: `src/graph/state.py`
**Actions**:
- Define `ValidationState` TypedDict with all required fields
- Define `ValidationResult` dataclass for individual validation results
- Define `ErrorDetail` dataclass for error tracking
- Add type hints and docstrings
- Implement state helper functions (get_active_validators, add_result, etc.)

**Innovation**: Use LangGraph's reducer pattern for state updates to handle concurrent validator results

### 1.3 Base Agent Class
**File**: `src/agents/base.py`
**Actions**:
- Create abstract `BaseAgent` class
- Define standard agent interface: `execute(state: ValidationState) -> ValidationState`
- Implement common functionality:
  - State validation
  - Error handling wrapper
  - Logging integration
  - Metrics collection hooks
- Add agent metadata (name, description, capabilities)

### 1.4 Configuration Management
**File**: `src/utils/config.py`
**Actions**:
- Create `Config` dataclass for system configuration
- Support environment variables for API keys
- Define default values for:
  - LLM model selection
  - Retry policies
  - Timeout values
  - Confidence thresholds
- Add configuration validation

### 1.5 Common Utilities
**File**: `src/utils/retry.py`
**Actions**:
- Implement retry decorator with exponential backoff
- Add configurable retry policies
- Support context-aware retry (learn from previous failures)
- Integrate with state management

**File**: `src/utils/logger.py`
**Actions**:
- Setup structured logging
- Add correlation IDs for request tracking
- Configure log levels
- Add performance metrics logging

## Testing Strategy

### Unit Tests
**File**: `tests/test_state.py`
- Test state schema validation
- Test state update operations
- Test state serialization/deserialization

**File**: `tests/test_base_agent.py`
- Test base agent initialization
- Test error handling wrapper
- Test state validation

**File**: `tests/test_config.py`
- Test configuration loading
- Test environment variable handling
- Test default values

## Technical Specifications

### State Schema
```python
class ValidationState(TypedDict):
    """Main state object passed through the LangGraph workflow."""
    input_data: dict
    validation_request: dict
    active_validators: list[str]
    completed_validators: list[str]
    validation_results: list[ValidationResult]
    errors: list[ErrorDetail]
    overall_status: Literal["pending", "in_progress", "completed", "failed"]
    confidence_score: float
    final_report: Optional[dict]
    metadata: dict  # For extensibility
```

### Dependencies
```toml
[project]
name = "hvas-mini"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
]
```

## Innovation Highlights

1. **Reducer-based State Updates**: Use LangGraph reducers to safely merge validation results from parallel executions
2. **Type-safe State Management**: Leverage TypedDict and runtime validation for state integrity
3. **Extensible Metadata**: Include metadata field in state for custom extensions
4. **Context-aware Retry**: Retry logic that examines previous failure context from state

## Acceptance Criteria

- ✅ All project directories created
- ✅ pyproject.toml configured and dependencies installable via `uv sync`
- ✅ State schemas defined with proper types
- ✅ BaseAgent class implemented with standard interface
- ✅ Configuration system working with env vars
- ✅ All unit tests passing (>90% coverage)
- ✅ Code follows Python best practices (black, ruff)

## Implementation Order

1. Setup project structure and pyproject.toml
2. Implement state schemas (state.py)
3. Create base agent class (base.py)
4. Add configuration management (config.py)
5. Implement utilities (retry.py, logger.py)
6. Write comprehensive tests
7. Document all components

## Estimated Complexity
**Medium** - Foundational work requiring careful design decisions that affect entire system

## Notes
- State schema is critical - must support both sequential and parallel validator execution
- Base agent class should be minimal but extensible
- Configuration should support both local development and production deployment
