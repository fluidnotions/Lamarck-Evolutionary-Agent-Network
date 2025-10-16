"""Unit tests for state management module."""

import pytest
from datetime import datetime

from src.graph.state import (
    ErrorDetail,
    ValidationResult,
    ValidationState,
    add_error,
    add_validation_result,
    create_initial_state,
    get_active_validators,
    get_completed_validators,
    get_validation_summary,
    is_validation_complete,
    update_confidence_score,
)


class TestErrorDetail:
    """Tests for ErrorDetail dataclass."""

    def test_error_detail_creation(self):
        """Test creating an ErrorDetail instance."""
        error = ErrorDetail(
            error_type="ValueError",
            message="Invalid input",
            validator_name="test_validator",
        )

        assert error.error_type == "ValueError"
        assert error.message == "Invalid input"
        assert error.validator_name == "test_validator"
        assert error.recoverable is True
        assert isinstance(error.timestamp, datetime)

    def test_error_detail_to_dict(self):
        """Test converting ErrorDetail to dictionary."""
        error = ErrorDetail(
            error_type="ValueError",
            message="Invalid input",
            validator_name="test_validator",
            context={"key": "value"},
        )

        error_dict = error.to_dict()

        assert error_dict["error_type"] == "ValueError"
        assert error_dict["message"] == "Invalid input"
        assert error_dict["validator_name"] == "test_validator"
        assert error_dict["context"] == {"key": "value"}
        assert error_dict["recoverable"] is True
        assert "timestamp" in error_dict


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult instance."""
        result = ValidationResult(
            validator_name="test_validator",
            status="passed",
            confidence=0.95,
            findings=["All checks passed"],
            recommendations=["No action needed"],
        )

        assert result.validator_name == "test_validator"
        assert result.status == "passed"
        assert result.confidence == 0.95
        assert result.findings == ["All checks passed"]
        assert result.recommendations == ["No action needed"]
        assert isinstance(result.timestamp, datetime)

    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult(
            validator_name="test_validator",
            status="failed",
            confidence=0.5,
            findings=["Issue found"],
            metadata={"severity": "high"},
        )

        result_dict = result.to_dict()

        assert result_dict["validator_name"] == "test_validator"
        assert result_dict["status"] == "failed"
        assert result_dict["confidence"] == 0.5
        assert result_dict["findings"] == ["Issue found"]
        assert result_dict["metadata"] == {"severity": "high"}
        assert "timestamp" in result_dict


class TestValidationState:
    """Tests for ValidationState and state helper functions."""

    def test_create_initial_state(self):
        """Test creating initial validation state."""
        input_data = {"key": "value"}
        validation_request = {"type": "test"}

        state = create_initial_state(input_data, validation_request)

        assert state["input_data"] == input_data
        assert state["validation_request"] == validation_request
        assert state["active_validators"] == []
        assert state["completed_validators"] == []
        assert state["validation_results"] == []
        assert state["errors"] == []
        assert state["overall_status"] == "pending"
        assert state["confidence_score"] == 0.0
        assert state["final_report"] is None
        assert state["metadata"] == {}

    def test_create_initial_state_with_metadata(self):
        """Test creating initial state with custom metadata."""
        metadata = {"request_id": "123", "user": "test"}

        state = create_initial_state(
            {"key": "value"}, {"type": "test"}, metadata
        )

        assert state["metadata"] == metadata

    def test_add_validation_result(self):
        """Test adding a validation result to state."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["active_validators"] = ["validator1"]

        result = ValidationResult(
            validator_name="validator1",
            status="passed",
            confidence=0.9,
        )

        new_state = add_validation_result(state, result)

        assert len(new_state["validation_results"]) == 1
        assert new_state["validation_results"][0] == result
        assert "validator1" not in new_state["active_validators"]
        assert "validator1" in new_state["completed_validators"]

    def test_add_validation_result_updates_status(self):
        """Test that adding results updates overall status."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["active_validators"] = ["validator1", "validator2"]

        result1 = ValidationResult(
            validator_name="validator1",
            status="passed",
            confidence=0.9,
        )

        new_state = add_validation_result(state, result1)
        assert new_state["overall_status"] == "in_progress"

        result2 = ValidationResult(
            validator_name="validator2",
            status="passed",
            confidence=0.95,
        )

        final_state = add_validation_result(new_state, result2)
        assert final_state["overall_status"] == "completed"
        assert len(final_state["active_validators"]) == 0

    def test_add_error(self):
        """Test adding an error to state."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["active_validators"] = ["validator1"]

        error = ErrorDetail(
            error_type="ValueError",
            message="Test error",
            validator_name="validator1",
        )

        new_state = add_error(state, error)

        assert len(new_state["errors"]) == 1
        assert new_state["errors"][0] == error
        assert "validator1" not in new_state["active_validators"]

    def test_update_confidence_score_no_results(self):
        """Test updating confidence score with no results."""
        state = create_initial_state({"key": "value"}, {"type": "test"})

        new_state = update_confidence_score(state)

        assert new_state["confidence_score"] == 0.0

    def test_update_confidence_score_passed(self):
        """Test confidence score with all passed validations."""
        state = create_initial_state({"key": "value"}, {"type": "test"})

        result1 = ValidationResult(
            validator_name="v1", status="passed", confidence=0.9
        )
        result2 = ValidationResult(
            validator_name="v2", status="passed", confidence=0.8
        )

        state["validation_results"] = [result1, result2]
        new_state = update_confidence_score(state)

        # Average: (0.9*1.0 + 0.8*1.0) / (1.0 + 1.0) = 0.85
        assert abs(new_state["confidence_score"] - 0.85) < 0.001

    def test_update_confidence_score_mixed(self):
        """Test confidence score with mixed results."""
        state = create_initial_state({"key": "value"}, {"type": "test"})

        result1 = ValidationResult(
            validator_name="v1", status="passed", confidence=0.9
        )
        result2 = ValidationResult(
            validator_name="v2", status="warning", confidence=0.8
        )
        result3 = ValidationResult(
            validator_name="v3", status="failed", confidence=0.6
        )

        state["validation_results"] = [result1, result2, result3]
        new_state = update_confidence_score(state)

        # Weighted: (0.9*1.0 + 0.8*0.75 + 0.6*0.5) / (1.0 + 0.75 + 0.5)
        expected = (0.9 + 0.6 + 0.3) / 2.25
        assert abs(new_state["confidence_score"] - expected) < 0.01

    def test_get_active_validators(self):
        """Test getting active validators."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["active_validators"] = ["v1", "v2"]

        active = get_active_validators(state)

        assert active == ["v1", "v2"]

    def test_get_completed_validators(self):
        """Test getting completed validators."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["completed_validators"] = ["v1", "v2"]

        completed = get_completed_validators(state)

        assert completed == ["v1", "v2"]

    def test_is_validation_complete_true(self):
        """Test validation complete check when no active validators."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["active_validators"] = []

        assert is_validation_complete(state) is True

    def test_is_validation_complete_false(self):
        """Test validation complete check when validators are active."""
        state = create_initial_state({"key": "value"}, {"type": "test"})
        state["active_validators"] = ["v1"]

        assert is_validation_complete(state) is False

    def test_get_validation_summary(self):
        """Test getting validation summary."""
        state = create_initial_state({"key": "value"}, {"type": "test"})

        result1 = ValidationResult(
            validator_name="v1", status="passed", confidence=0.9
        )
        result2 = ValidationResult(
            validator_name="v2", status="failed", confidence=0.6
        )
        result3 = ValidationResult(
            validator_name="v3", status="warning", confidence=0.7
        )

        error = ErrorDetail(
            error_type="ValueError",
            message="Test error",
            validator_name="v4",
        )

        state["validation_results"] = [result1, result2, result3]
        state["errors"] = [error]
        state["overall_status"] = "completed"
        state["confidence_score"] = 0.75
        state["active_validators"] = ["v5"]
        state["completed_validators"] = ["v1", "v2", "v3"]

        summary = get_validation_summary(state)

        assert summary["overall_status"] == "completed"
        assert summary["confidence_score"] == 0.75
        assert summary["total_validators"] == 3
        assert summary["status_counts"]["passed"] == 1
        assert summary["status_counts"]["failed"] == 1
        assert summary["status_counts"]["warning"] == 1
        assert summary["error_count"] == 1
        assert summary["active_validators"] == 1
        assert summary["completed_validators"] == 3
