# HVAS-Mini: Hierarchical Validation Agent System

A powerful, LangGraph-based multi-agent system for hierarchical data validation. HVAS-Mini orchestrates specialized validation agents to provide comprehensive, confidence-scored validation with detailed error reporting.

## Features

- **Hierarchical Multi-Agent Architecture**: Supervisor-coordinated validation with specialized domain validators
- **LangGraph Integration**: Built on LangGraph for robust state management and workflow orchestration
- **Parallel Execution**: Independent validators run concurrently for optimal performance
- **Comprehensive Error Reporting**: Detailed error tracking with severity levels and context
- **Confidence Scoring**: AI-powered confidence scores for validation results
- **Flexible Configuration**: Easy customization for different validation scenarios
- **Multiple Validators**:
  - **Schema Validator**: JSON schema compliance checking
  - **Business Rules**: Domain-specific business logic validation
  - **Data Quality**: Completeness, accuracy, and consistency checks
- **Rich Reporting**: Generate reports in text, markdown, or JSON formats
- **Visualization Support**: Export workflow graphs for analysis and documentation
- **Retry Logic**: Built-in resilience with configurable retry and error handling

## Installation

### Prerequisites

- Python 3.11 or higher
- [UV package manager](https://github.com/astral-sh/uv)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd hvas-mini

# Install dependencies using UV
uv sync

# Create environment file
cp .env.example .env

# Edit .env with your API keys
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### Dependencies

HVAS-Mini requires:
- LangGraph (>=0.2.0)
- LangChain (>=0.3.0)
- Pydantic (>=2.0.0)
- JSONSchema (>=4.20.0)

Development dependencies include pytest, black, ruff, and coverage tools.

## Quick Start

### Basic Validation

```python
from src.graph.workflow import ValidationWorkflow

# Sample data to validate
data = {
    "user": {
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
    },
    "order": {
        "id": "ORD-123",
        "total": 99.99,
    },
}

# Create workflow
workflow = ValidationWorkflow()

# Run validation
result = workflow.run(
    data=data,
    validators=["schema", "business", "quality"],
    config={
        "schema": {
            "schema": {
                "type": "object",
                "required": ["user", "order"]
            }
        }
    },
)

# Check results
print(f"Status: {result.overall_status}")
print(f"Confidence: {result.confidence_score:.2%}")
print(f"Errors: {result.total_errors}")

# Generate report
report = result.get_report("markdown")
```

### Custom Validation Rules

```python
from src.validators.rule_engine import Rule

# Create workflow
workflow = ValidationWorkflow()

# Add custom rule
age_rule = Rule(
    name="minimum_age",
    condition=lambda data: data.get("age", 0) >= 18,
    error_message="Must be 18 or older",
)
workflow.business_rules.add_rule(age_rule)

# Run validation
result = workflow.run(
    data={"age": 25},
    validators=["business"],
)
```

### Streaming Validation

```python
# Stream validation progress
for state_update in workflow.stream(data=data, validators=["schema", "quality"]):
    print(f"Processing: {list(state_update.keys())}")
```

## Configuration

### Environment Variables

```bash
# LLM Provider (openai or anthropic)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# LangChain Tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
LANGCHAIN_PROJECT=hvas-mini

# Application Settings
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

### Validation Configuration

Configure validators through the `config` parameter:

```python
config = {
    "schema": {
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            }
        }
    },
    "data_quality": {
        "required_fields": ["name", "email"],
        "types": {"age": "int"},
        "patterns": {
            "email": r"^[\w\.-]+@[\w\.-]+\.\w+$"
        },
        "ranges": {
            "age": {"min": 0, "max": 120}
        }
    },
    "business_rules": {
        "rules": ["rule1", "rule2"]
    }
}
```

## LangGraph Visualization

### View Workflow Graph

```python
from src.visualization.graph_export import print_graph_ascii, generate_graph_html

# ASCII visualization
workflow = ValidationWorkflow()
print_graph_ascii(workflow)

# HTML visualization
generate_graph_html(workflow, "workflow.html")
```

### LangGraph Studio

For interactive debugging and visualization:

```bash
# Start LangGraph Studio
uv run langgraph studio

# Open browser to http://localhost:8000
```

LangGraph Studio provides:
- Interactive graph visualization
- Step-by-step execution debugging
- State inspection at each node
- Execution replay and analysis

## Examples

See the `examples/` directory for complete examples:

- **basic_validation.py**: Simple validation workflow
- **custom_validator.py**: Creating custom validation rules
- **advanced_workflow.py**: Complex multi-validator scenarios
- **streaming_validation.py**: Real-time validation streaming
- **visualization_demo.py**: Graph export and visualization

Run examples:

```bash
uv run python examples/basic_validation.py
```

## Architecture

HVAS-Mini uses a hierarchical multi-agent architecture:

```
┌─────────────┐
│ Supervisor  │  ← Orchestrates workflow
└──────┬──────┘
       │
   ┌───┴───┬───────┬──────────┐
   │       │       │          │
┌──▼──┐ ┌──▼──┐ ┌──▼───────┐ │
│Schema│ │Rules│ │Quality   │ │
└──┬───┘ └──┬──┘ └──┬───────┘ │
   │        │       │          │
   └────────┴───────┴──────────┘
              │
        ┌─────▼──────┐
        │ Aggregator │  ← Synthesizes results
        └────────────┘
```

### Components

- **Supervisor Agent**: Analyzes requests and routes to validators
- **Schema Validator**: JSON schema compliance checking
- **Business Rules Agent**: Domain-specific business logic
- **Data Quality Agent**: Completeness and consistency checks
- **Aggregator Agent**: Result synthesis and reporting

### State Management

LangGraph manages workflow state with:
- Validation progress tracking
- Result accumulation
- Error collection
- Confidence scoring

## API Documentation

### ValidationWorkflow

Main workflow orchestrator.

```python
workflow = ValidationWorkflow(config=optional_config)

# Synchronous execution
result = workflow.run(data, validators, config)

# Asynchronous execution
result = await workflow.arun(data, validators, config)

# Streaming execution
for state in workflow.stream(data, validators, config):
    process(state)
```

### ValidationResult

Result from a single validator.

```python
result.validator_name  # Validator that produced result
result.status          # "passed", "failed", "error"
result.confidence      # 0.0 to 1.0
result.errors          # List of ErrorDetail
result.warnings        # List of ErrorDetail
result.execution_time_ms  # Execution time
```

### AggregatedResult

Final aggregated results.

```python
result.overall_status    # "passed", "failed", "partial"
result.confidence_score  # Overall confidence
result.validation_results  # List of ValidationResult
result.total_errors      # Total error count
result.total_warnings    # Total warning count

# Generate reports
text_report = result.get_report("text")
markdown_report = result.get_report("markdown")
json_report = result.get_report("json")
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run specific test file
uv run pytest tests/test_state.py

# Run with verbose output
uv run pytest -v

# Run only integration tests
uv run pytest tests/integration/
```

### Code Quality

```bash
# Format code
uv run black src tests examples

# Lint code
uv run ruff check src tests examples

# Type checking
uv run mypy src
```

### Project Structure

```
hvas-mini/
├── src/
│   ├── agents/           # Validation agents
│   ├── graph/            # LangGraph workflow
│   ├── validators/       # Validation logic
│   ├── models/           # Data models
│   ├── utils/            # Utilities
│   └── visualization/    # Graph export tools
├── tests/
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── examples/             # Usage examples
├── docs/                 # Documentation
└── pyproject.toml        # Project configuration
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd hvas-mini

# Install with dev dependencies
uv sync --dev

# Run tests before committing
uv run pytest

# Format and lint
uv run black .
uv run ruff check .
```

## Testing

HVAS-Mini has comprehensive test coverage (>85%):

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and workflows
- **End-to-End Tests**: Test realistic validation scenarios

See `tests/` directory for test examples.

## Performance

- **Parallel Execution**: Validators run concurrently when possible
- **Efficient State Management**: LangGraph's optimized state handling
- **Configurable Timeouts**: Control execution time limits
- **Retry Logic**: Automatic retry with exponential backoff

Typical performance:
- Simple validation: <100ms
- Complex multi-validator: <500ms
- Large datasets (1000+ records): <2s

## Troubleshooting

### Common Issues

**ImportError: No module named 'langgraph'**
```bash
uv sync  # Reinstall dependencies
```

**LLM API Errors**
- Check API keys in `.env`
- Verify API key permissions
- Check rate limits

**Validation Failures**
- Review error messages in result
- Check schema configuration
- Verify data format

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### LangGraph Tracing

Enable LangChain tracing for detailed execution logs:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use HVAS-Mini in your research or project, please cite:

```bibtex
@software{hvas_mini,
  title = {HVAS-Mini: Hierarchical Validation Agent System},
  author = {HVAS Team},
  year = {2024},
  url = {https://github.com/yourusername/hvas-mini}
}
```

## Support

- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## Roadmap

Future enhancements:
- [ ] Human-in-the-loop validation
- [ ] Machine learning for confidence scoring
- [ ] Custom validator plugins
- [ ] Distributed execution
- [ ] Real-time streaming validation
- [ ] Advanced analytics dashboard

## Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- [Pydantic](https://github.com/pydantic/pydantic) - Data validation

---

**HVAS-Mini** - Intelligent, hierarchical data validation made simple.
