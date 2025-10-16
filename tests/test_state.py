"""Tests for state management."""
import pytest
from src.graph.state import (
    create_initial_state,
    update_state_with_result,
    should_continue_validation,
    calculate_overall_confidence,
    determine_overall_status,
)
from src.models.validation_result import ValidationResult, ErrorDetail


class TestCreateInitialState:
    """Tests for create_initial_state function."""

    def test_creates_valid_state(self, sample_data):
        """Test that initial state is created with correct structure."""
        state = create_initial_state(
            input_data=sample_data,
            validators=["schema", "rules"],
            config={"test": True},
        )

        assert state["input_data"] == sample_data
        assert state["pending_validators"] == ["schema", "rules"]
        assert state["completed_validators"] == []
        assert state["validation_results"] == []
        assert state["overall_status"] == "pending"
        assert state["confidence_score"] == 0.0
        assert "workflow_id" in state

    def test_creates_with_custom_workflow_id(self, sample_data):
        """Test that custom workflow ID is used."""
        custom_id = "test-workflow-123"
        state = create_initial_state(
            input_data=sample_data,
            validators=["schema"],
            workflow_id=custom_id,
        )

        assert state["workflow_id"] == custom_id

    def test_creates_with_empty_validators(self, sample_data):
        """Test state creation with no validators."""
        state = create_initial_state(
            input_data=sample_data,
            validators=[],
        )

        assert state["pending_validators"] == []


class TestUpdateStateWithResult:
    """Tests for update_state_with_result function."""

    def test_updates_with_passed_result(self, initial_state, sample_validation_result):
        """Test updating state with a passed result."""
        updates = update_state_with_result(initial_state, sample_validation_result)

        assert sample_validation_result in updates["validation_results"]
        assert sample_validation_result.validator_name in updates["completed_validators"]

    def test_updates_with_failed_result(self, initial_state, failed_validation_result):
        """Test updating state with a failed result."""
        updates = update_state_with_result(initial_state, failed_validation_result)

        assert failed_validation_result in updates["validation_results"]
        assert "errors" in updates
        assert len(updates["errors"]) > 0

    def test_removes_from_pending(self, initial_state):
        """Test that validator is removed from pending list."""
        initial_state["pending_validators"] = ["test_validator", "other"]

        result = ValidationResult(
            validator_name="test_validator",
            status="passed",
            confidence=1.0,
        )

        updates = update_state_with_result(initial_state, result)

        assert "test_validator" not in updates.get("pending_validators", [])


class TestShouldContinueValidation:
    """Tests for should_continue_validation function."""

    def test_continues_with_pending_validators(self, initial_state):
        """Test that validation continues when validators are pending."""
        initial_state["pending_validators"] = ["schema", "rules"]
        initial_state["overall_status"] = "in_progress"

        assert should_continue_validation(initial_state) is True

    def test_continues_with_active_validators(self, initial_state):
        """Test that validation continues when validators are active."""
        initial_state["pending_validators"] = []
        initial_state["active_validators"] = ["schema"]
        initial_state["overall_status"] = "in_progress"

        assert should_continue_validation(initial_state) is True

    def test_stops_when_no_validators(self, initial_state):
        """Test that validation stops when no validators remain."""
        initial_state["pending_validators"] = []
        initial_state["active_validators"] = []

        assert should_continue_validation(initial_state) is False

    def test_stops_when_failed(self, initial_state):
        """Test that validation stops when status is failed."""
        initial_state["pending_validators"] = ["schema"]
        initial_state["overall_status"] = "failed"

        assert should_continue_validation(initial_state) is False


class TestCalculateOverallConfidence:
    """Tests for calculate_overall_confidence function."""

    def test_with_empty_results(self):
        """Test confidence calculation with no results."""
        confidence = calculate_overall_confidence([])
        assert confidence == 0.0

    def test_with_all_passed(self):
        """Test confidence with all passed results."""
        results = [
            ValidationResult(validator_name="v1", status="passed", confidence=1.0),
            ValidationResult(validator_name="v2", status="passed", confidence=1.0),
            ValidationResult(validator_name="v3", status="passed", confidence=1.0),
        ]

        confidence = calculate_overall_confidence(results)
        assert confidence == 1.0

    def test_with_mixed_results(self):
        """Test confidence with mixed results."""
        results = [
            ValidationResult(validator_name="v1", status="passed", confidence=1.0),
            ValidationResult(
                validator_name="v2",
                status="failed",
                confidence=0.0,
                errors=[ErrorDetail(path="test", message="error", code="ERR")],
            ),
        ]

        confidence = calculate_overall_confidence(results)
        assert 0.0 < confidence < 1.0

    def test_with_all_failed(self):
        """Test confidence with all failed results."""
        results = [
            ValidationResult(
                validator_name="v1",
                status="failed",
                confidence=0.0,
                errors=[ErrorDetail(path="test", message="error", code="ERR")],
            ),
            ValidationResult(
                validator_name="v2",
                status="failed",
                confidence=0.0,
                errors=[ErrorDetail(path="test", message="error", code="ERR")],
            ),
        ]

        confidence = calculate_overall_confidence(results)
        assert confidence == 0.0


class TestDetermineOverallStatus:
    """Tests for determine_overall_status function."""

    def test_with_empty_results(self):
        """Test status determination with no results."""
        status = determine_overall_status([])
        assert status == "pending"

    def test_with_all_passed(self):
        """Test status with all passed results."""
        results = [
            ValidationResult(validator_name="v1", status="passed", confidence=1.0),
            ValidationResult(validator_name="v2", status="passed", confidence=1.0),
        ]

        status = determine_overall_status(results)
        assert status == "passed"

    def test_with_all_failed(self):
        """Test status with all failed results."""
        results = [
            ValidationResult(validator_name="v1", status="failed", confidence=0.0),
            ValidationResult(validator_name="v2", status="failed", confidence=0.0),
        ]

        status = determine_overall_status(results)
        assert status == "failed"

    def test_with_partial_failure(self):
        """Test status with some passed and some failed."""
        results = [
            ValidationResult(validator_name="v1", status="passed", confidence=1.0),
            ValidationResult(validator_name="v2", status="failed", confidence=0.0),
        ]

        status = determine_overall_status(results)
        assert status == "partial"

    def test_with_errors(self):
        """Test status with error results."""
        results = [
            ValidationResult(validator_name="v1", status="error", confidence=0.0),
            ValidationResult(validator_name="v2", status="passed", confidence=1.0),
        ]

        status = determine_overall_status(results)
        assert status == "partial"
