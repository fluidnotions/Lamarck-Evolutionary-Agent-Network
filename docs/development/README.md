# HVAS-Mini Development Guide

Guide for developers contributing to HVAS-Mini.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- UV package manager
- Git
- (Optional) GraphViz for visualization

### Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd hvas-mini

# Install dependencies with dev tools
uv sync --dev

# Verify installation
uv run pytest --version
uv run black --version
uv run ruff --version
```

### Environment Configuration

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with required API keys and configuration.

## Project Structure

```
hvas-mini/
├── src/                    # Source code
│   ├── agents/            # Validation agents
│   │   ├── base.py        # Base agent class
│   │   ├── supervisor.py  # Supervisor agent
│   │   ├── schema_validator.py
│   │   ├── business_rules.py
│   │   ├── data_quality.py
│   │   └── aggregator.py
│   ├── graph/             # LangGraph workflow
│   │   ├── state.py       # State management
│   │   ├── workflow.py    # Main workflow
│   │   └── routing.py     # Routing logic
│   ├── validators/        # Validation logic
│   │   ├── json_schema.py # Schema validation
│   │   ├── rule_engine.py # Business rules
│   │   └── quality_checks.py
│   ├── models/            # Data models
│   │   └── validation_result.py
│   ├── utils/             # Utilities
│   │   ├── config.py      # Configuration
│   │   └── retry.py       # Retry logic
│   └── visualization/     # Graph export
│       └── graph_export.py
├── tests/                 # Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_state.py      # State tests
│   ├── test_validators.py # Validator tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── examples/              # Usage examples
├── docs/                  # Documentation
│   ├── api/              # API documentation
│   └── development/       # Development guide
├── pyproject.toml         # Project configuration
├── README.md              # Main documentation
└── .gitignore            # Git ignore rules
```

## Code Style

### Python Style Guide

HVAS-Mini follows PEP 8 with some modifications:

- Line length: 100 characters
- Use double quotes for strings
- Use type hints for all functions
- Document all public APIs with docstrings

### Formatting

Use Black for code formatting:

```bash
# Format all code
uv run black src tests examples

# Check formatting
uv run black --check src tests examples
```

### Linting

Use Ruff for linting:

```bash
# Lint code
uv run ruff check src tests examples

# Fix auto-fixable issues
uv run ruff check --fix src tests examples
```

### Type Checking

Use mypy for type checking:

```bash
# Type check source code
uv run mypy src

# Type check with strict mode
uv run mypy --strict src
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_state.py

# Run specific test
uv run pytest tests/test_state.py::TestCreateInitialState::test_creates_valid_state

# Run with verbose output
uv run pytest -v

# Run only unit tests
uv run pytest tests/ -k "not integration and not e2e"

# Run only integration tests
uv run pytest tests/integration/

# Run only e2e tests
uv run pytest tests/e2e/
```

### Writing Tests

#### Test Structure

```python
"""Test module docstring."""
import pytest
from src.module import function


class TestFeature:
    """Test class for feature."""

    def test_specific_behavior(self):
        """Test that specific behavior works correctly."""
        # Arrange
        input_data = {"key": "value"}

        # Act
        result = function(input_data)

        # Assert
        assert result == expected
```

#### Using Fixtures

```python
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value"}


def test_with_fixture(sample_data):
    """Test using fixture."""
    assert "key" in sample_data
```

#### Mocking

```python
from unittest.mock import Mock, patch


def test_with_mock():
    """Test with mocked dependency."""
    mock_llm = Mock()
    mock_llm.invoke.return_value = Mock(content="response")

    # Use mock in test
    agent = Agent(llm=mock_llm)
    result = agent.process(state)

    assert mock_llm.invoke.called
```

### Test Coverage

Maintain >85% test coverage:

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

## Adding New Features

### Adding a New Validator

1. **Create validator module**:

```python
# src/validators/my_validator.py
from typing import Dict, Any, List
from src.models.validation_result import ErrorDetail


def validate_my_feature(data: Dict[str, Any]) -> tuple[bool, List[ErrorDetail]]:
    """Validate my feature.

    Args:
        data: Data to validate

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    # Validation logic here

    return len(errors) == 0, errors
```

2. **Create agent**:

```python
# src/agents/my_agent.py
from src.agents.base import BaseAgent
from src.validators.my_validator import validate_my_feature


class MyValidatorAgent(BaseAgent):
    """Agent for my validator."""

    def __init__(self, **kwargs):
        super().__init__(name="my_validator", **kwargs)

    def process(self, state):
        data = state["input_data"]
        is_valid, errors = validate_my_feature(data)

        result = self.create_result(
            status="passed" if is_valid else "failed",
            confidence=1.0 if is_valid else 0.0,
            errors=errors,
        )

        return {
            "validation_results": [result],
            "completed_validators": [self.name],
            "errors": errors,
        }
```

3. **Add to workflow**:

```python
# src/graph/workflow.py
from src.agents.my_agent import MyValidatorAgent


class ValidationWorkflow:
    def __init__(self):
        # ... existing code ...
        self.my_validator = MyValidatorAgent()

        # Add to graph
        graph.add_node("my_validator", self.my_validator)
```

4. **Write tests**:

```python
# tests/test_my_validator.py
import pytest
from src.validators.my_validator import validate_my_feature


class TestMyValidator:
    def test_validates_valid_data(self):
        """Test validation of valid data."""
        data = {"valid": True}
        is_valid, errors = validate_my_feature(data)

        assert is_valid is True
        assert len(errors) == 0

    def test_detects_invalid_data(self):
        """Test detection of invalid data."""
        data = {"valid": False}
        is_valid, errors = validate_my_feature(data)

        assert is_valid is False
        assert len(errors) > 0
```

### Adding Custom Rules

```python
from src.validators.rule_engine import Rule

# Create rule
my_rule = Rule(
    name="my_rule",
    condition=lambda data: check_condition(data),
    error_message="Condition not met",
    severity="error",
)

# Add to workflow
workflow = ValidationWorkflow()
workflow.business_rules.add_rule(my_rule)
```

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Use LangChain Tracing

Enable in `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=hvas-mini-debug
```

### Debug with LangGraph Studio

```bash
# Start studio
uv run langgraph studio

# Open http://localhost:8000
```

### Use Python Debugger

```python
import pdb

def my_function():
    pdb.set_trace()  # Breakpoint
    # Continue debugging
```

## Performance Optimization

### Profiling

```bash
# Install profiling tools
uv pip install py-spy

# Profile running application
py-spy top --pid <pid>

# Generate flame graph
py-spy record -o profile.svg -- python examples/basic_validation.py
```

### Optimization Tips

1. **Use parallel execution**: Configure `parallel_validators=True`
2. **Set appropriate timeouts**: Avoid waiting too long for slow validators
3. **Cache LLM responses**: Use caching for repeated validations
4. **Optimize schemas**: Keep JSON schemas simple and focused
5. **Batch operations**: Validate multiple items together when possible

## CI/CD

### GitHub Actions

Example workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install UV
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --dev
      - name: Run tests
        run: uv run pytest --cov=src
      - name: Check coverage
        run: uv run pytest --cov=src --cov-fail-under=85
```

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Run full test suite**: `uv run pytest`
4. **Build package**: `uv build`
5. **Tag release**: `git tag v0.1.0`
6. **Push tag**: `git push origin v0.1.0`
7. **Create GitHub release**
8. **Publish to PyPI**: `uv publish`

## Contributing Checklist

Before submitting a pull request:

- [ ] Code follows style guide (Black + Ruff)
- [ ] All tests pass (`uv run pytest`)
- [ ] Coverage remains >85%
- [ ] Type hints added for new code
- [ ] Docstrings added for public APIs
- [ ] Examples updated if needed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented)

## Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: Search or create GitHub issues
- **Discussions**: Use GitHub Discussions
- **Stack Overflow**: Tag with `hvas-mini`

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## License

MIT License - see LICENSE file.
