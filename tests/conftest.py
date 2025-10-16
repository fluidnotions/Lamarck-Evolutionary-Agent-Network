"""Pytest configuration and fixtures for HVAS-Mini tests."""
import pytest
from typing import Any, Dict
from unittest.mock import Mock

from src.models.validation_result import ValidationResult, ErrorDetail
from src.graph.state import ValidationState, create_initial_state


@pytest.fixture
def sample_data() -> Dict[str, Any]:
    """Sample validation data."""
    return {
        "user": {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        },
        "order": {
            "id": "ORD-123",
            "total": 99.99,
            "items": [
                {"product": "Widget", "quantity": 2, "price": 49.99}
            ],
        },
    }


@pytest.fixture
def invalid_data() -> Dict[str, Any]:
    """Invalid validation data."""
    return {
        "user": {
            "name": "",  # Empty name
            "email": "invalid-email",  # Invalid email
            "age": -5,  # Invalid age
        },
        "order": {
            "total": "not a number",  # Wrong type
        },
    }


@pytest.fixture
def sample_schema() -> Dict[str, Any]:
    """Sample JSON schema."""
    return {
        "type": "object",
        "required": ["user", "order"],
        "properties": {
            "user": {
                "type": "object",
                "required": ["name", "email", "age"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "email": {"type": "string", "format": "email"},
                    "age": {"type": "integer", "minimum": 0},
                },
            },
            "order": {
                "type": "object",
                "required": ["id", "total"],
                "properties": {
                    "id": {"type": "string"},
                    "total": {"type": "number", "minimum": 0},
                },
            },
        },
    }


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration."""
    return {
        "schema": {
            "schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        },
        "data_quality": {
            "required_fields": ["user", "order"],
            "types": {
                "user": "dict",
                "order": "dict",
            },
        },
    }


@pytest.fixture
def initial_state(sample_data: Dict[str, Any]) -> ValidationState:
    """Create initial validation state."""
    return create_initial_state(
        input_data=sample_data,
        validators=["schema", "business", "quality"],
        config={},
    )


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="Mocked LLM response"))
    return mock


@pytest.fixture
def sample_validation_result() -> ValidationResult:
    """Sample validation result."""
    return ValidationResult(
        validator_name="test_validator",
        status="passed",
        confidence=1.0,
        errors=[],
        warnings=[],
        metadata={"test": True},
        execution_time_ms=10.5,
    )


@pytest.fixture
def sample_error_detail() -> ErrorDetail:
    """Sample error detail."""
    return ErrorDetail(
        path="user.email",
        message="Invalid email format",
        code="INVALID_EMAIL",
        severity="error",
        context={"expected": "email", "actual": "invalid-email"},
    )


@pytest.fixture
def failed_validation_result(sample_error_detail: ErrorDetail) -> ValidationResult:
    """Failed validation result with errors."""
    return ValidationResult(
        validator_name="test_validator",
        status="failed",
        confidence=0.0,
        errors=[sample_error_detail],
        warnings=[],
        metadata={},
        execution_time_ms=15.3,
    )
