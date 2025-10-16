"""Pytest fixtures for testing."""

import pytest
from datetime import datetime
from src.models import ValidationResult, ErrorDetail


@pytest.fixture
def sample_error():
    """Create a sample error."""
    return ErrorDetail(
        severity="error",
        message="Test error message",
        path="data.field",
        validator="test_validator",
        context={"detail": "test"},
    )


@pytest.fixture
def sample_critical_error():
    """Create a sample critical error."""
    return ErrorDetail(
        severity="critical",
        message="Critical error message",
        path="data.critical_field",
        validator="test_validator",
        context={"detail": "critical"},
    )


@pytest.fixture
def sample_warning():
    """Create a sample warning."""
    return ErrorDetail(
        severity="warning",
        message="Test warning message",
        path="data.field",
        validator="test_validator",
    )


@pytest.fixture
def passing_result():
    """Create a passing validation result."""
    return ValidationResult(
        validator_name="passing_validator",
        status="passed",
        errors=[],
        warnings=[],
        execution_time=0.5,
        coverage=1.0,
    )


@pytest.fixture
def failing_result(sample_error, sample_critical_error):
    """Create a failing validation result."""
    return ValidationResult(
        validator_name="failing_validator",
        status="failed",
        errors=[sample_error, sample_critical_error],
        warnings=[],
        execution_time=1.2,
        coverage=0.95,
    )


@pytest.fixture
def result_with_warnings(sample_warning):
    """Create a result with warnings."""
    return ValidationResult(
        validator_name="warning_validator",
        status="passed",
        errors=[],
        warnings=[sample_warning],
        execution_time=0.8,
        coverage=1.0,
    )


@pytest.fixture
def mixed_results(passing_result, failing_result, result_with_warnings):
    """Create a list of mixed validation results."""
    return [passing_result, failing_result, result_with_warnings]


@pytest.fixture
def validation_state(mixed_results):
    """Create a sample validation state."""
    return {
        "input_data": {"test": "data"},
        "validation_request": {"type": "full"},
        "active_validators": [],
        "completed_validators": ["passing_validator", "failing_validator", "warning_validator"],
        "validation_results": mixed_results,
        "errors": [],
        "overall_status": "in_progress",
        "confidence_score": None,
        "final_report": None,
    }
