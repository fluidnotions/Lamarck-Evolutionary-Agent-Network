"""Tests for error handler agent."""

import pytest

from src.agents.error_handler import (
    ErrorHandlerAgent,
    create_error_handler_node,
    error_handler_node,
)
from src.models import ErrorDetail, ValidationResult
from src.resilience.recovery import WorkflowRecovery


class TestErrorHandlerAgent:
    """Tests for ErrorHandlerAgent."""

    def test_handle_error_no_failures(self):
        """Test handling state with no failures."""
        agent = ErrorHandlerAgent()
        state = {"failed_validators": []}

        result = agent.handle_error(state)

        assert result == state  # No changes

    def test_handle_error_minor_failures(self):
        """Test handling minor failures."""
        agent = ErrorHandlerAgent()
        state = {
            "failed_validators": ["val1"],
            "completed_validators": ["val2", "val3", "val4"],
            "validation_results": [],
            "errors": [
                ErrorDetail(
                    path="test",
                    message="Minor error",
                    severity="warning",
                )
            ],
        }

        result = agent.handle_error(state)

        # Should have applied degradation
        assert "metadata" in result
        assert len(result["validation_results"]) > 0

    def test_handle_error_moderate_failures(self):
        """Test handling moderate failures."""
        agent = ErrorHandlerAgent()
        state = {
            "failed_validators": ["val1", "val2"],
            "completed_validators": ["val3", "val4"],
            "validation_results": [],
            "errors": [],
        }

        result = agent.handle_error(state)

        # Should have applied degradation
        assert "degradation_applied" in result.get("metadata", {})

    def test_handle_error_critical_failures(self):
        """Test handling critical failures."""
        agent = ErrorHandlerAgent()
        state = {
            "failed_validators": ["val1", "val2", "val3"],
            "completed_validators": ["val4"],
            "validation_results": [],
            "errors": [
                ErrorDetail(
                    path="test",
                    message="Critical error",
                    severity="critical",
                )
            ],
        }

        result = agent.handle_error(state)

        # Should have attempted recovery
        assert "metadata" in result

    def test_analyze_error_severity_critical(self):
        """Test critical error severity detection."""
        agent = ErrorHandlerAgent()

        # Critical error present
        state = {
            "errors": [
                ErrorDetail(
                    path="test",
                    message="Critical error",
                    severity="critical",
                )
            ],
            "failed_validators": [],
            "completed_validators": [],
        }

        severity = agent._analyze_error_severity(state)
        assert severity == "critical"

    def test_analyze_error_severity_high_failure_rate(self):
        """Test critical severity for high failure rate."""
        agent = ErrorHandlerAgent()

        state = {
            "errors": [],
            "failed_validators": ["val1", "val2", "val3"],
            "completed_validators": ["val4"],  # 75% failure rate
        }

        severity = agent._analyze_error_severity(state)
        assert severity == "critical"

    def test_analyze_error_severity_moderate(self):
        """Test moderate error severity detection."""
        agent = ErrorHandlerAgent()

        state = {
            "errors": [],
            "failed_validators": ["val1"],
            "completed_validators": ["val2", "val3", "val4"],  # 25% failure rate
        }

        severity = agent._analyze_error_severity(state)
        assert severity == "moderate"

    def test_analyze_error_severity_minor(self):
        """Test minor error severity detection."""
        agent = ErrorHandlerAgent()

        state = {
            "errors": [],
            "failed_validators": ["val1"],
            "completed_validators": [f"val{i}" for i in range(2, 11)],  # 10% failure
        }

        severity = agent._analyze_error_severity(state)
        assert severity == "minor"

    def test_suggest_fixes_timeout(self):
        """Test fix suggestions for timeout errors."""
        agent = ErrorHandlerAgent()

        state = {
            "errors": [
                ErrorDetail(
                    path="test",
                    message="Timeout error",
                    code="timeout",
                )
            ]
        }

        suggestions = agent.suggest_fixes(state)

        assert len(suggestions) > 0
        assert any("timeout" in s.lower() for s in suggestions)

    def test_suggest_fixes_validation_error(self):
        """Test fix suggestions for validation errors."""
        agent = ErrorHandlerAgent()

        state = {
            "errors": [
                ErrorDetail(
                    path="test",
                    message="Validation failed",
                    code="validation_error",
                )
            ]
        }

        suggestions = agent.suggest_fixes(state)

        assert len(suggestions) > 0
        assert any("validation" in s.lower() or "schema" in s.lower() for s in suggestions)

    def test_suggest_fixes_high_retries(self):
        """Test fix suggestions for high retry count."""
        agent = ErrorHandlerAgent()

        state = {
            "errors": [],
            "metadata": {"retry_count": 10}
        }

        suggestions = agent.suggest_fixes(state)

        assert len(suggestions) > 0
        assert any("retries" in s.lower() or "retry" in s.lower() for s in suggestions)

    def test_get_recovery_recommendations(self):
        """Test getting recovery recommendations."""
        agent = ErrorHandlerAgent()

        state = {
            "failed_validators": ["val1", "val2"],
            "completed_validators": ["val3"],
            "degradation_level": 2,
            "errors": [],
        }

        recommendations = agent.get_recovery_recommendations(state)

        assert "severity" in recommendations
        assert "failed_validators" in recommendations
        assert "suggested_actions" in recommendations
        assert "fix_suggestions" in recommendations
        assert len(recommendations["suggested_actions"]) > 0

    def test_handle_error_with_exception(self):
        """Test handling error with exception object."""
        agent = ErrorHandlerAgent()

        exception = ValueError("Test exception")
        state = {
            "failed_validators": ["val1"],
            "completed_validators": [],
            "validation_results": [],
            "errors": [],
        }

        result = agent.handle_error(state, error=exception)

        # Should have captured error context
        assert "last_error_context" in result["metadata"]


class TestErrorHandlerNode:
    """Tests for error_handler_node function."""

    def test_node_with_errors(self):
        """Test node function with errors."""
        state = {
            "failed_validators": ["val1"],
            "completed_validators": [],
            "validation_results": [],
            "errors": [
                ErrorDetail(
                    path="test",
                    message="Error",
                    severity="error",
                )
            ],
        }

        result = error_handler_node(state)

        # Should have processed errors
        assert "metadata" in result
        assert "error_recommendations" in result["metadata"]

    def test_node_without_errors(self):
        """Test node function without errors."""
        state = {
            "failed_validators": [],
            "errors": [],
        }

        result = error_handler_node(state)

        # Should pass through unchanged
        assert result == state

    def test_create_custom_node(self):
        """Test creating custom error handler node."""
        recovery = WorkflowRecovery()
        cache = {}

        node_func = create_error_handler_node(
            use_llm=False,
            workflow_recovery=recovery,
            result_cache=cache,
        )

        state = {
            "failed_validators": ["val1"],
            "completed_validators": [],
            "validation_results": [],
            "errors": [],
        }

        result = node_func(state)

        # Should have processed errors
        assert "metadata" in result


class TestErrorHandlerIntegration:
    """Integration tests for error handler."""

    def test_full_error_handling_flow(self):
        """Test complete error handling flow."""
        agent = ErrorHandlerAgent()

        # Initial state with failures
        state = {
            "input_data": {"test": "data"},
            "failed_validators": ["val1", "val2"],
            "completed_validators": ["val3", "val4"],
            "validation_results": [
                ValidationResult(
                    validator_name="val3",
                    status="passed",
                    confidence=1.0,
                ),
                ValidationResult(
                    validator_name="val4",
                    status="passed",
                    confidence=1.0,
                ),
            ],
            "errors": [
                ErrorDetail(
                    path="val1",
                    message="Validation failed",
                    severity="error",
                )
            ],
            "metadata": {},
        }

        # Handle error
        result = agent.handle_error(state)

        # Get recommendations
        recommendations = agent.get_recovery_recommendations(result)

        # Verify complete handling
        assert result["degradation_level"] > 0
        assert "degradation_applied" in result["metadata"]
        assert len(recommendations["suggested_actions"]) > 0
        assert len(result["validation_results"]) > 2  # Added degraded results

    def test_recovery_after_degradation(self):
        """Test recovery after degradation is applied."""
        recovery = WorkflowRecovery()

        # Create checkpoint before failure
        good_state = {
            "completed_validators": ["val1", "val2"],
            "failed_validators": [],
            "active_validators": ["val3"],
        }
        recovery.create_checkpoint(good_state, "before_failure")

        # Simulate failure and degradation
        agent = ErrorHandlerAgent(workflow_recovery=recovery)

        failed_state = {
            "completed_validators": ["val1", "val2"],
            "failed_validators": ["val3"],
            "active_validators": [],
            "validation_results": [],
            "errors": [
                ErrorDetail(
                    path="val3",
                    message="Critical failure",
                    severity="critical",
                )
            ],
        }

        # Handle error (should trigger recovery for critical error)
        result = agent.handle_error(failed_state)

        # Should have attempted recovery
        assert "metadata" in result
