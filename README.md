# HVAS-Mini: Hierarchical Validation Agent System

## Task 5: Aggregator & Reporting

This branch implements the aggregator agent and comprehensive reporting functionality for HVAS-Mini.

## Features

- **Result Merging & Deduplication**: Intelligent merging of validation results from parallel executions
- **Multi-Factor Confidence Scoring**: Confidence calculation based on pass rate, error severity, coverage, and reliability
- **Error Analysis**: Pattern detection, error grouping, and prioritization
- **Multi-Format Reports**: JSON, Markdown, HTML, and PDF report generation
- **Visualizations**: Charts and graphs for validation results
- **Comprehensive Testing**: >85% test coverage with unit and integration tests

## Installation

```bash
uv sync
```

## Running Tests

```bash
uv run pytest
```

## Usage

See `tests/test_integration.py` for complete usage examples.
