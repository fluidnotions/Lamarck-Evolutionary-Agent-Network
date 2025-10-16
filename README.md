# HVAS-Mini

**Hierarchical Validation Agent System - Minimalist Implementation**

A lightweight validation system built with LangGraph for orchestrating multiple validation agents in a hierarchical workflow.

## Features

- **State Management**: Type-safe validation state with LangGraph reducers
- **Agent Framework**: Abstract base classes for building validation agents
- **Configuration Management**: Environment-based configuration with validation
- **Retry Utilities**: Exponential backoff with context-aware retry logic
- **Structured Logging**: Correlation IDs and performance metrics

## Installation

Using UV (recommended):

```bash
uv sync
```

## Project Structure

```
hvas-mini/
├── src/
│   ├── graph/          # LangGraph state management
│   ├── agents/         # Validation agent base classes
│   └── utils/          # Utilities (config, retry, logger)
├── tests/
│   └── unit/           # Unit tests
├── docs/               # Documentation
└── examples/           # Usage examples
```

## Quick Start

```python
from src.graph.state import create_initial_state, ValidationResult
from src.agents.base import ValidationAgent
from src.utils.config import load_config

# Load configuration
config = load_config()

# Create initial state
state = create_initial_state(
    input_data={"code": "def hello(): pass"},
    validation_request={"type": "syntax_check"}
)

# Implement custom validator
class MyValidator(ValidationAgent):
    def _execute(self, state):
        # Your validation logic here
        return ValidationResult(
            validator_name=self.name,
            status="passed",
            confidence=0.95
        )
```

## Testing

Run tests with coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Development

This is Task 1 of the HVAS-Mini implementation, providing the core infrastructure for the validation system.

## License

MIT
