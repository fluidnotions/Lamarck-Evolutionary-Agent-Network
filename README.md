# HVAS-Mini: Task 4 - Atomic Validators & Execution

This branch implements the atomic validators and parallel execution engine for HVAS-Mini.

## Implemented Components

### Models
- `ErrorDetail`: Detailed error information with path, message, severity, and context
- `ValidationResult`: Structured validation results with errors, warnings, and timing

### Validators
- `JSONSchemaValidator`: JSON Schema validation (Draft 7 and Draft 2020-12)
- `RuleEngine`: Business rule validation with DSL and dependency resolution
- `QualityChecker`: Data quality checks (completeness, consistency, accuracy, statistical)
- `CrossReferenceValidator`: Referential integrity and foreign key validation
- `ValidatorExecutor`: Parallel async execution with dependency resolution

## Features

- Atomic, focused validators that can run independently
- Parallel execution of independent validators
- Dependency resolution for ordered execution
- Comprehensive error reporting with precise paths
- Statistical quality checks (outlier detection, duplicate detection)
- Rule engine with Python DSL
- Foreign key and cardinality validation
- Cycle detection in hierarchical data

## Performance

- 100+ rules evaluated in <1s
- Efficient parallel execution
- Optimized for large datasets

## Testing

Run tests with:
```bash
uv run pytest
```

Run performance benchmarks:
```bash
uv run python tests/test_performance.py
```
