# Task 4: Atomic Validators & Execution
## Implementation Plan

## Objective
Implement leaf-node atomic validators that perform single, focused validation checks. These are the workhorses called by domain validators. Also implement parallel execution infrastructure for independent atomic validators.

## Dependencies
- Task 1: Core Infrastructure (requires ValidationState, ValidationResult)
- Task 3: Domain Validators (called by domain validators)

## Components to Implement

### 4.1 JSON Schema Validator
**File**: `src/validators/json_schema.py`
**Actions**:
- Implement JSON Schema validation using jsonschema library
- Support Draft 7 and Draft 2020-12
- Generate detailed error paths (e.g., "data.users[0].email")
- Handle nested schema validation
- Support custom format validators
- Provide validation timing metrics

### 4.2 Rule Engine
**File**: `src/validators/rule_engine.py`
**Actions**:
- Implement rule definition DSL (domain-specific language)
- Support rule types:
  - Constraint rules (must/must-not conditions)
  - Derivation rules (calculated fields)
  - Inference rules (logical deductions)
- Implement rule evaluation engine
- Support rule dependencies and ordering
- Add rule conflict detection
- Cache rule evaluation results

**Innovation**: Simple Python-based DSL for rules that's both powerful and readable

### 4.3 Quality Check Functions
**File**: `src/validators/quality_checks.py`
**Actions**:
- Implement completeness checks:
  - Missing value detection
  - Required field validation
  - Null handling policies
- Implement consistency checks:
  - Cross-field validation
  - Logical consistency (e.g., end_date > start_date)
  - Format consistency
- Implement accuracy checks:
  - Format validation (email, phone, URL, etc.)
  - Range validation
  - Domain value validation (enums, allowed values)
- Implement statistical quality checks:
  - Outlier detection
  - Distribution analysis
  - Duplicate detection

### 4.4 Cross-Reference Validators
**File**: `src/validators/cross_reference.py`
**Actions**:
- Implement foreign key validation
- Implement referential integrity checks
- Implement cardinality validation (1-to-1, 1-to-many, many-to-many)
- Implement cyclic dependency detection
- Support external reference validation (API calls, database lookups)
- Add caching for reference lookups

### 4.5 Parallel Execution Engine
**File**: `src/validators/executor.py`
**Actions**:
- Implement parallel execution of independent validators
- Use asyncio for concurrent execution
- Implement dependency resolution for ordered execution
- Add timeout handling per validator
- Collect results from parallel executions
- Handle partial failures gracefully
- Add execution metrics (timing, success rate)

### 4.6 Validator Result Models
**File**: `src/models/validation_result.py`
**Actions**:
- Define `ValidationResult` dataclass
- Include fields: validator_name, status, errors, warnings, info, timing
- Support result serialization
- Add result comparison utilities

**File**: `src/models/error_detail.py`
**Actions**:
- Define `ErrorDetail` dataclass
- Include fields: path, message, severity, code, context
- Support error grouping and categorization

## Testing Strategy

### Unit Tests
**File**: `tests/test_json_schema.py`
- Test with various JSON schemas
- Test error path generation
- Test nested schema validation
- Test custom formats

**File**: `tests/test_rule_engine.py`
- Test rule definition parsing
- Test rule evaluation
- Test rule dependencies
- Test rule conflicts

**File**: `tests/test_quality_checks.py`
- Test each quality check function independently
- Test with edge cases (empty data, null values, etc.)
- Test statistical checks with known datasets

**File**: `tests/test_cross_reference.py`
- Test foreign key validation
- Test referential integrity
- Test cyclic dependency detection
- Mock external reference lookups

**File**: `tests/test_executor.py`
- Test parallel execution
- Test dependency resolution
- Test timeout handling
- Test partial failure handling

## Technical Specifications

### JSON Schema Validator
```python
def validate_json_schema(data: dict, schema: dict) -> ValidationResult:
    """
    Validate data against JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema definition

    Returns:
        ValidationResult with detailed error paths
    """
    import jsonschema
    from jsonschema import Draft7Validator

    validator = Draft7Validator(schema)
    errors = []

    for error in validator.iter_errors(data):
        errors.append(ErrorDetail(
            path=".".join(str(p) for p in error.path),
            message=error.message,
            severity="error",
            code="schema_validation",
            context={"schema_path": ".".join(str(p) for p in error.schema_path)}
        ))

    return ValidationResult(
        validator_name="json_schema",
        status="passed" if not errors else "failed",
        errors=errors,
        timing=execution_time
    )
```

### Rule Engine DSL
```python
# Rule definition example
class Rule:
    def __init__(self, name: str, condition: Callable, message: str):
        self.name = name
        self.condition = condition
        self.message = message

    def evaluate(self, data: dict) -> bool:
        return self.condition(data)

# Example rules
rules = [
    Rule(
        name="price_positive",
        condition=lambda data: data.get("price", 0) > 0,
        message="Price must be positive"
    ),
    Rule(
        name="end_after_start",
        condition=lambda data: data.get("end_date") > data.get("start_date"),
        message="End date must be after start date"
    ),
]

class RuleEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules

    def evaluate(self, data: dict) -> ValidationResult:
        """Evaluate all rules against data."""
        violations = []
        for rule in self.rules:
            if not rule.evaluate(data):
                violations.append(ErrorDetail(
                    path="",
                    message=rule.message,
                    severity="error",
                    code=f"rule_{rule.name}",
                    context={"rule": rule.name}
                ))

        return ValidationResult(
            validator_name="rule_engine",
            status="passed" if not violations else "failed",
            errors=violations
        )
```

### Parallel Execution Engine
```python
import asyncio
from typing import List, Callable

async def execute_validators_parallel(
    validators: List[Callable],
    data: dict,
    timeout: float = 30.0
) -> List[ValidationResult]:
    """
    Execute multiple validators in parallel.

    Args:
        validators: List of validator functions
        data: Data to validate
        timeout: Timeout per validator in seconds

    Returns:
        List of ValidationResult objects
    """
    async def run_validator(validator: Callable) -> ValidationResult:
        try:
            return await asyncio.wait_for(
                validator(data),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return ValidationResult(
                validator_name=validator.__name__,
                status="failed",
                errors=[ErrorDetail(
                    path="",
                    message=f"Validator timed out after {timeout}s",
                    severity="error",
                    code="timeout"
                )]
            )
        except Exception as e:
            return ValidationResult(
                validator_name=validator.__name__,
                status="failed",
                errors=[ErrorDetail(
                    path="",
                    message=str(e),
                    severity="error",
                    code="exception"
                )]
            )

    tasks = [run_validator(v) for v in validators]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, ValidationResult)]
```

## Innovation Highlights

1. **Python-based Rule DSL**: Simple, readable rule definitions using Python lambdas and functions
2. **Async Parallel Execution**: Efficiently execute independent validators concurrently
3. **Detailed Error Paths**: JSON Schema validator provides exact paths to errors
4. **Statistical Quality Checks**: Beyond basic validation, includes outlier detection and distribution analysis
5. **Graceful Degradation**: Parallel executor handles partial failures without stopping entire validation

## Acceptance Criteria

- ✅ JSON Schema validator supports Draft 7 and 2020-12
- ✅ Rule engine evaluates complex rule sets correctly
- ✅ Quality checks cover completeness, consistency, and accuracy
- ✅ Cross-reference validators handle foreign keys and cycles
- ✅ Parallel execution engine runs validators concurrently
- ✅ All validators return structured ValidationResult objects
- ✅ Error paths are precise and actionable
- ✅ All unit tests passing (>90% coverage)
- ✅ Performance benchmarks met (100+ rules in <1s)

## Implementation Order

1. Define ValidationResult and ErrorDetail models
2. Implement JSON Schema validator (most fundamental)
3. Implement quality check functions (building blocks)
4. Implement rule engine with DSL
5. Implement cross-reference validators
6. Implement parallel execution engine
7. Write comprehensive tests for all components
8. Performance testing and optimization

## Estimated Complexity
**Medium-High** - Multiple independent components, parallel execution adds complexity

## Notes
- Atomic validators should have minimal dependencies
- Each validator should complete in <1s for good UX
- Consider memory usage when processing large datasets
- Parallel execution engine is critical for system performance
- Rule DSL should be intuitive for non-programmers to define simple rules
