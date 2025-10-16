# HVAS-Mini: Hierarchical Validation Agent System

A hierarchical multi-agent system built on LangGraph for validating complex data through specialized domain validators.

## Features

- **Schema Validation**: JSON Schema and Pydantic model validation
- **Data Quality**: Multi-dimensional quality scoring (completeness, consistency, accuracy)
- **Business Rules**: Custom rule evaluation with LLM-powered explanations
- **Cross-Reference**: Referential integrity and relationship validation

## Installation

```bash
uv sync
```

## Usage

See `examples/` directory for usage examples.

## Testing

```bash
uv run pytest
```

## Task 3: Domain Validators

This implementation focuses on the four main domain validator agents that coordinate atomic validation checks within their respective domains.
