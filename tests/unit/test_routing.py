"""Unit tests for routing logic."""

import pytest
from langgraph.graph import END

from src.graph.routing import (
    route_to_validators,
    route_from_validator,
    update_state_after_validator,
    _all_validators_complete,
    should_handle_errors,
)
from src.graph.state import create_initial_state, create_error_detail


class TestRoutingLogic:
    """Test routing decision logic."""

    def test_route_to_first_validator_sequential(self) -> None:
        """Test routing to first validator in sequential mode."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = ["validator1"]
        state["pending_validators"] = ["validator1", "validator2"]
        state["workflow_metadata"] = {"execution_mode": "sequential"}

        result = route_to_validators(state)

        assert result == "validator1"

    def test_route_to_parallel_validators(self) -> None:
        """Test routing to multiple validators in parallel mode."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = ["validator1", "validator2"]
        state["pending_validators"] = ["validator1", "validator2"]
        state["workflow_metadata"] = {"execution_mode": "parallel"}

        result = route_to_validators(state)

        assert result == ["validator1", "validator2"]

    def test_route_to_aggregator_when_complete(self) -> None:
        """Test routing to aggregator when all validators complete."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = []
        state["completed_validators"] = ["validator1", "validator2"]
        state["workflow_metadata"] = {
            "execution_mode": "sequential",
            "supervisor_decision": {
                "validators": ["validator1", "validator2"]
            }
        }

        result = route_to_validators(state)

        assert result == "aggregator"

    def test_route_ends_on_failure(self) -> None:
        """Test routing ends workflow on failure status."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "failed"

        result = route_to_validators(state)

        assert result == END

    def test_route_activates_pending_sequential(self) -> None:
        """Test routing activates next pending validator in sequential mode."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = ["validator2", "validator3"]
        state["completed_validators"] = ["validator1"]
        state["workflow_metadata"] = {"execution_mode": "sequential"}

        result = route_to_validators(state)

        assert result == "validator2"

    def test_route_activates_all_pending_parallel(self) -> None:
        """Test routing activates all pending validators in parallel mode."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = ["validator2", "validator3"]
        state["completed_validators"] = ["validator1"]
        state["workflow_metadata"] = {"execution_mode": "parallel"}

        result = route_to_validators(state)

        assert result == ["validator2", "validator3"]

    def test_route_from_validator_sequential_continues(self) -> None:
        """Test routing from validator continues to next in sequential mode."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = ["validator2"]
        state["completed_validators"] = ["validator1"]
        state["workflow_metadata"] = {"execution_mode": "sequential"}

        result = route_from_validator(state)

        assert result == "validator2"

    def test_route_from_validator_sequential_to_aggregator(self) -> None:
        """Test routing from validator to aggregator when complete."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = []
        state["completed_validators"] = ["validator1", "validator2"]
        state["workflow_metadata"] = {
            "execution_mode": "sequential",
            "supervisor_decision": {
                "validators": ["validator1", "validator2"]
            }
        }

        result = route_from_validator(state)

        assert result == "aggregator"

    def test_route_from_validator_parallel_to_aggregator(self) -> None:
        """Test routing from validator in parallel mode when all complete."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = []
        state["completed_validators"] = ["validator1", "validator2"]
        state["workflow_metadata"] = {
            "execution_mode": "parallel",
            "supervisor_decision": {
                "validators": ["validator1", "validator2"]
            }
        }

        result = route_from_validator(state)

        assert result == "aggregator"

    def test_update_state_after_validator_completion(self) -> None:
        """Test state updates correctly after validator completes."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["active_validators"] = ["validator1"]
        state["pending_validators"] = ["validator1", "validator2"]
        state["completed_validators"] = []
        state["workflow_metadata"] = {"execution_mode": "sequential"}

        result = update_state_after_validator(state, "validator1")

        assert "validator1" not in result["active_validators"]
        assert "validator1" in result["completed_validators"]
        assert "validator1" not in result["pending_validators"]
        # In sequential mode, should activate next validator
        assert "validator2" in result["active_validators"]

    def test_update_state_parallel_doesnt_activate_next(self) -> None:
        """Test parallel mode doesn't auto-activate next validator."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["active_validators"] = ["validator1", "validator2"]
        state["pending_validators"] = ["validator1", "validator2"]
        state["completed_validators"] = []
        state["workflow_metadata"] = {"execution_mode": "parallel"}

        result = update_state_after_validator(state, "validator1")

        assert "validator1" in result["completed_validators"]
        # validator2 should still be active from initial setup
        assert "validator2" in result["active_validators"]

    def test_all_validators_complete_check(self) -> None:
        """Test checking if all validators are complete."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["completed_validators"] = ["validator1", "validator2"]
        state["workflow_metadata"] = {
            "supervisor_decision": {
                "validators": ["validator1", "validator2"]
            }
        }

        assert _all_validators_complete(state) is True

    def test_all_validators_not_complete(self) -> None:
        """Test checking when not all validators are complete."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["completed_validators"] = ["validator1"]
        state["workflow_metadata"] = {
            "supervisor_decision": {
                "validators": ["validator1", "validator2"]
            }
        }

        assert _all_validators_complete(state) is False

    def test_all_validators_complete_when_no_validators(self) -> None:
        """Test completion check when no validators were selected."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["completed_validators"] = []
        state["workflow_metadata"] = {
            "supervisor_decision": {
                "validators": []
            }
        }

        assert _all_validators_complete(state) is True

    def test_should_handle_errors_with_critical_error(self) -> None:
        """Test error handling detection with critical errors."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["errors"] = [
            create_error_detail(
                error_type="critical_error",
                message="Critical error occurred"
            )
        ]

        assert should_handle_errors(state) is True

    def test_should_handle_errors_with_supervisor_error(self) -> None:
        """Test error handling detection with supervisor errors."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["errors"] = [
            create_error_detail(
                error_type="supervisor_analysis_error",
                message="Supervisor failed"
            )
        ]

        assert should_handle_errors(state) is True

    def test_should_not_handle_minor_errors(self) -> None:
        """Test error handling skips minor errors."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["errors"] = [
            create_error_detail(
                error_type="validation_warning",
                message="Minor issue"
            )
        ]

        assert should_handle_errors(state) is False

    def test_should_not_handle_when_no_errors(self) -> None:
        """Test error handling when no errors exist."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )

        assert should_handle_errors(state) is False

    def test_route_ends_when_nothing_to_do(self) -> None:
        """Test routing ends when there's nothing left to do."""
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        state["overall_status"] = "in_progress"
        state["active_validators"] = []
        state["pending_validators"] = []
        state["completed_validators"] = []
        state["workflow_metadata"] = {}

        result = route_to_validators(state)

        assert result == END
