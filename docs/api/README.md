# HVAS-Mini API Documentation

Complete API reference for HVAS-Mini components.

## Core Workflow

### ValidationWorkflow

Main workflow orchestrator for running validation tasks.

```python
from src.graph.workflow import ValidationWorkflow

workflow = ValidationWorkflow(config: Optional[Dict[str, Any]] = None)
```

#### Methods

**run(data, validators, config, workflow_id)**
- **Parameters**:
  - `data` (Dict[str, Any]): Data to validate
  - `validators` (List[str], optional): List of validator names
  - `config` (Dict[str, Any], optional): Validation configuration
  - `workflow_id` (str, optional): Custom workflow identifier
- **Returns**: AggregatedResult
- **Description**: Synchronously runs the validation workflow

**arun(data, validators, config, workflow_id)**
- **Parameters**: Same as `run()`
- **Returns**: AggregatedResult
- **Description**: Asynchronously runs the validation workflow

**stream(data, validators, config, workflow_id)**
- **Parameters**: Same as `run()`
- **Yields**: State updates as they occur
- **Description**: Streams validation execution for real-time monitoring

## State Management

### ValidationState

TypedDict defining the workflow state structure.

```python
from src.graph.state import ValidationState

state = ValidationState(
    input_data=dict,
    validation_request=dict,
    config=dict,
    active_validators=list,
    completed_validators=list,
    pending_validators=list,
    validation_results=list,
    errors=list,
    overall_status=str,
    current_step=str,
    retry_count=int,
    confidence_score=float,
    final_report=Optional[AggregatedResult],
    workflow_id=str,
    timestamp=datetime,
    execution_time_ms=float,
)
```

### State Functions

**create_initial_state(input_data, validators, config, workflow_id)**
- Creates initial validation state
- Returns: ValidationState

**update_state_with_result(state, result)**
- Updates state with a validation result
- Returns: Dict of state updates

**calculate_overall_confidence(results)**
- Calculates overall confidence from results
- Returns: float (0.0 to 1.0)

**determine_overall_status(results)**
- Determines overall validation status
- Returns: str ("passed", "failed", or "partial")

## Models

### ValidationResult

Result from a single validator.

```python
from src.models.validation_result import ValidationResult

result = ValidationResult(
    validator_name=str,
    status=str,  # "passed", "failed", "error", "skipped"
    confidence=float,  # 0.0 to 1.0
    errors=List[ErrorDetail],
    warnings=List[ErrorDetail],
    metadata=Dict[str, Any],
    execution_time_ms=float,
    timestamp=datetime,
)
```

### ErrorDetail

Details about a validation error.

```python
from src.models.validation_result import ErrorDetail

error = ErrorDetail(
    path=str,  # JSON path to error location
    message=str,  # Human-readable message
    code=str,  # Error code
    severity=str,  # "error", "warning", "info"
    context=Dict[str, Any],  # Additional context
)
```

### AggregatedResult

Aggregated results from all validators.

```python
from src.models.validation_result import AggregatedResult

result = AggregatedResult(
    overall_status=str,
    confidence_score=float,
    validation_results=List[ValidationResult],
    summary=str,
    total_errors=int,
    total_warnings=int,
    execution_time_ms=float,
    timestamp=datetime,
)
```

#### Methods

**get_all_errors() -> List[ErrorDetail]**
- Returns all errors from all validators

**get_all_warnings() -> List[ErrorDetail]**
- Returns all warnings from all validators

**get_report(format: str) -> str**
- Generates formatted report
- `format`: "text", "markdown", or "json"
- Returns: Formatted report string

## Agents

### BaseAgent

Abstract base class for all validation agents.

```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="custom", **kwargs)

    def process(self, state: ValidationState) -> Dict[str, Any]:
        # Implement validation logic
        return state_updates
```

### SupervisorAgent

Orchestrates validation workflow.

```python
from src.agents.supervisor import SupervisorAgent

supervisor = SupervisorAgent()
```

### SchemaValidatorAgent

Validates data against JSON schemas.

```python
from src.agents.schema_validator import SchemaValidatorAgent

validator = SchemaValidatorAgent()
```

### BusinessRulesAgent

Applies business rule validation.

```python
from src.agents.business_rules import BusinessRulesAgent

agent = BusinessRulesAgent()
agent.add_rule(rule)  # Add custom rule
```

### DataQualityAgent

Validates data quality metrics.

```python
from src.agents.data_quality import DataQualityAgent

agent = DataQualityAgent()
```

### AggregatorAgent

Aggregates validation results.

```python
from src.agents.aggregator import AggregatorAgent

aggregator = AggregatorAgent()
```

## Validators

### JSON Schema Validation

```python
from src.validators.json_schema import validate_json_schema

is_valid, errors = validate_json_schema(data, schema)
```

### Rule Engine

```python
from src.validators.rule_engine import RuleEngine, Rule

# Create engine
engine = RuleEngine()

# Add rule
rule = Rule(
    name="rule_name",
    condition=lambda data: bool,
    error_message="Error message",
    severity="error",
)
engine.add_rule(rule)

# Validate
passed, errors = engine.validate(data)
```

#### Rule Builders

```python
from src.validators.rule_engine import (
    create_range_rule,
    create_required_field_rule,
    create_comparison_rule,
)

# Range rule
rule = create_range_rule("age", min_val=0, max_val=120)

# Required field
rule = create_required_field_rule("email")

# Comparison
rule = create_comparison_rule("start_date", "<", "end_date")
```

### Quality Checks

```python
from src.validators.quality_checks import (
    check_completeness,
    check_data_types,
    check_string_patterns,
    check_value_ranges,
    check_consistency,
    check_uniqueness,
)

# Completeness
errors = check_completeness(data, required_fields=["name", "email"])

# Types
errors = check_data_types(data, type_specs={"age": int})

# Patterns
errors = check_string_patterns(data, patterns={"email": r"regex"})

# Ranges
errors = check_value_ranges(data, ranges={"age": (0, 120)})

# Consistency
errors = check_consistency(data, rules=[("min", "<", "max")])
```

## Utilities

### Configuration

```python
from src.utils.config import Config, get_config, set_config

# Get global config
config = get_config()

# Create from environment
config = Config.from_env()

# Set custom config
set_config(custom_config)
```

### Retry Logic

```python
from src.utils.retry import retry_with_backoff, RetryConfig

# Configure retry
config = RetryConfig(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
)

# Apply decorator
@retry_with_backoff(config=config)
def my_function():
    # Function with retry logic
    pass
```

### Circuit Breaker

```python
from src.utils.retry import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    success_threshold=2,
    timeout=60.0,
)

result = breaker.call(function, *args, **kwargs)
```

## Visualization

### Graph Export

```python
from src.visualization.graph_export import (
    export_graph_to_mermaid,
    export_graph_to_png,
    generate_graph_html,
    print_graph_ascii,
)

# Mermaid diagram
mermaid_code = export_graph_to_mermaid(workflow)

# PNG export (requires graphviz)
success = export_graph_to_png(workflow, "graph.png")

# HTML visualization
html = generate_graph_html(workflow, "graph.html")

# ASCII art
print_graph_ascii(workflow)
```

## Error Handling

All agents implement automatic retry logic and graceful error handling. Errors are captured in ErrorDetail objects with:
- Path to error location
- Human-readable message
- Error code for programmatic handling
- Severity level
- Additional context

## Type Hints

HVAS-Mini is fully typed using Python type hints. Import types:

```python
from typing import Dict, List, Any, Optional
from src.graph.state import ValidationState
from src.models.validation_result import ValidationResult, ErrorDetail
```

## Best Practices

1. **Always configure schemas**: Provide explicit JSON schemas for schema validation
2. **Use custom rules**: Add domain-specific business rules for your use case
3. **Handle errors**: Check `result.total_errors` and process error details
4. **Monitor confidence**: Use confidence scores to assess validation reliability
5. **Enable tracing**: Use LangChain tracing for debugging
6. **Stream for long operations**: Use `stream()` for real-time progress

## Examples

See the `examples/` directory for complete working examples of all API features.
