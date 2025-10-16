"""Tests for validation models."""

import pytest

from src.models.error_detail import ErrorDetail
from src.models.validation_result import ValidationResult


class TestErrorDetail:
    """Test ErrorDetail model."""

    def test_create_error_detail(self) -> None:
        """Test creating an error detail."""
        error = ErrorDetail(
            path="data.users[0].email",
            message="Invalid email format",
            severity="error",
            code="format_error",
        )

        assert error.path == "data.users[0].email"
        assert error.message == "Invalid email format"
        assert error.severity == "error"
        assert error.code == "format_error"

    def test_invalid_severity(self) -> None:
        """Test that invalid severity raises error."""
        with pytest.raises(ValueError):
            ErrorDetail(
                path="test",
                message="test",
                severity="critical",  # Invalid severity
            )

    def test_to_dict(self) -> None:
        """Test converting error to dictionary."""
        error = ErrorDetail(
            path="test.path",
            message="Test message",
            context={"key": "value"},
        )

        result = error.to_dict()
        assert result["path"] == "test.path"
        assert result["message"] == "Test message"
        assert result["context"]["key"] == "value"

    def test_from_dict(self) -> None:
        """Test creating error from dictionary."""
        data = {
            "path": "test.path",
            "message": "Test message",
            "severity": "warning",
            "code": "test_code",
            "context": {"key": "value"},
        }

        error = ErrorDetail.from_dict(data)
        assert error.path == "test.path"
        assert error.severity == "warning"
        assert error.code == "test_code"

    def test_str_representation(self) -> None:
        """Test string representation."""
        error = ErrorDetail(path="test", message="Test message", severity="error")
        assert "[ERROR]" in str(error)
        assert "test" in str(error)
        assert "Test message" in str(error)


class TestValidationResult:
    """Test ValidationResult model."""

    def test_create_validation_result(self) -> None:
        """Test creating a validation result."""
        result = ValidationResult(
            validator_name="test_validator",
            status="passed",
        )

        assert result.validator_name == "test_validator"
        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_invalid_status(self) -> None:
        """Test that invalid status raises error."""
        with pytest.raises(ValueError):
            ValidationResult(
                validator_name="test",
                status="invalid",  # Invalid status
            )

    def test_has_errors_property(self) -> None:
        """Test has_errors property."""
        result = ValidationResult(
            validator_name="test",
            status="failed",
            errors=[ErrorDetail(path="test", message="error")],
        )

        assert result.has_errors is True
        assert result.error_count == 1

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        error = ErrorDetail(path="test", message="error")
        result = ValidationResult(
            validator_name="test",
            status="failed",
            errors=[error],
            timing=0.5,
        )

        data = result.to_dict()
        assert data["validator_name"] == "test"
        assert data["status"] == "failed"
        assert len(data["errors"]) == 1
        assert data["timing"] == 0.5

    def test_from_dict(self) -> None:
        """Test creating result from dictionary."""
        data = {
            "validator_name": "test",
            "status": "passed",
            "errors": [],
            "warnings": [{"path": "test", "message": "warning", "severity": "warning"}],
            "info": [],
            "timing": 0.5,
            "metadata": {"key": "value"},
        }

        result = ValidationResult.from_dict(data)
        assert result.validator_name == "test"
        assert result.status == "passed"
        assert len(result.warnings) == 1
        assert result.timing == 0.5
