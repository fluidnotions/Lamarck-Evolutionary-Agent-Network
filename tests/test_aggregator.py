"""Unit tests for aggregator agent."""

import pytest
from src.agents.aggregator import AggregatorAgent
from src.models import ValidationResult, ErrorDetail


class TestAggregatorAgent:
    """Test cases for AggregatorAgent."""

    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        agent = AggregatorAgent()
        assert agent.name == "aggregator"
        assert agent.description is not None

    def test_execute_with_results(self, validation_state):
        """Test execution with validation results."""
        agent = AggregatorAgent()
        result_state = agent.execute(validation_state)

        assert "overall_status" in result_state
        assert "confidence_score" in result_state
        assert "final_report" in result_state
        assert result_state["confidence_score"] is not None

    def test_execute_with_empty_results(self):
        """Test execution with no results."""
        agent = AggregatorAgent()
        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": [],
            "validation_results": [],
            "errors": [],
            "overall_status": "pending",
            "confidence_score": None,
            "final_report": None,
        }

        result_state = agent.execute(state)
        assert result_state["overall_status"] == "error"
        assert result_state["confidence_score"] == 0.0

    def test_determine_overall_status_all_pass(self):
        """Test status determination when all pass."""
        agent = AggregatorAgent()

        results = [
            ValidationResult(validator_name=f"v{i}", status="passed")
            for i in range(3)
        ]

        status = agent._determine_overall_status(results)
        assert status == "completed"

    def test_determine_overall_status_with_failures(self, failing_result):
        """Test status determination with failures."""
        agent = AggregatorAgent()
        status = agent._determine_overall_status([failing_result])
        assert status == "failed"

    def test_final_report_structure(self, validation_state):
        """Test that final report has correct structure."""
        agent = AggregatorAgent()
        result_state = agent.execute(validation_state)

        report = result_state["final_report"]
        assert "content" in report
        assert "format" in report
        assert "metadata" in report
        assert "error_analysis" in report
        assert "confidence_breakdown" in report

    def test_custom_report_format(self, validation_state):
        """Test custom report format."""
        agent = AggregatorAgent(report_format="html")
        result_state = agent.execute(validation_state)

        report = result_state["final_report"]
        assert report["format"] == "html"

    def test_visualizations_generated(self, validation_state):
        """Test that visualizations are generated."""
        agent = AggregatorAgent(generate_visualizations=True)
        result_state = agent.execute(validation_state)

        report = result_state["final_report"]
        assert "visualizations" in report
        assert report["visualizations"] is not None

    def test_visualizations_disabled(self, validation_state):
        """Test that visualizations can be disabled."""
        agent = AggregatorAgent(generate_visualizations=False)
        result_state = agent.execute(validation_state)

        report = result_state["final_report"]
        assert report["visualizations"] is None

    def test_custom_confidence_weights(self, validation_state):
        """Test custom confidence weights."""
        custom_weights = {
            "pass_rate": 0.5,
            "severity": 0.2,
            "coverage": 0.2,
            "reliability": 0.1,
        }

        agent = AggregatorAgent(confidence_weights=custom_weights)
        result_state = agent.execute(validation_state)

        assert result_state["confidence_score"] is not None

    def test_generate_summary_report(self, validation_state):
        """Test summary report generation."""
        agent = AggregatorAgent()
        summary = agent.generate_summary_report(validation_state)

        assert "status" in summary
        assert "confidence" in summary
        assert "confidence_level" in summary
        assert "recommendation" in summary
        assert "summary" in summary

    def test_summary_with_no_results(self):
        """Test summary report with no results."""
        agent = AggregatorAgent()
        state = {
            "validation_results": [],
        }

        summary = agent.generate_summary_report(state)
        assert summary["status"] == "no_results"

    def test_export_results_json(self, validation_state):
        """Test exporting results as JSON."""
        agent = AggregatorAgent()
        exported = agent.export_results_for_analysis(validation_state, format="json")

        import json
        data = json.loads(exported)
        assert "results" in data
        assert "metadata" in data

    def test_export_unsupported_format(self, validation_state):
        """Test exporting with unsupported format."""
        agent = AggregatorAgent()
        exported = agent.export_results_for_analysis(validation_state, format="xml")
        assert "Unsupported" in exported

    def test_get_failed_validators(self):
        """Test getting list of failed validators."""
        agent = AggregatorAgent()

        state = {
            "validation_results": [
                ValidationResult(validator_name="pass1", status="passed"),
                ValidationResult(
                    validator_name="fail1",
                    status="failed",
                    errors=[ErrorDetail(
                        severity="error",
                        message="Error",
                        validator="fail1",
                    )],
                ),
                ValidationResult(validator_name="pass2", status="passed"),
            ],
        }

        failed = agent.get_failed_validators(state)
        assert len(failed) == 1
        assert "fail1" in failed

    def test_get_critical_errors(self):
        """Test getting critical errors."""
        agent = AggregatorAgent()

        state = {
            "validation_results": [
                ValidationResult(
                    validator_name="test",
                    status="failed",
                    errors=[
                        ErrorDetail(
                            severity="critical",
                            message="Critical error",
                            validator="test",
                        ),
                        ErrorDetail(
                            severity="error",
                            message="Regular error",
                            validator="test",
                        ),
                    ],
                ),
            ],
        }

        critical = agent.get_critical_errors(state)
        assert len(critical) == 1
        assert critical[0]["message"] == "Critical error"

    def test_completed_validators_updated(self, validation_state):
        """Test that completed_validators list is updated."""
        agent = AggregatorAgent()
        result_state = agent.execute(validation_state)

        assert "aggregator" in result_state["completed_validators"]

    def test_state_immutability(self, validation_state):
        """Test that original state is not mutated."""
        agent = AggregatorAgent()
        original_status = validation_state["overall_status"]

        result_state = agent.execute(validation_state)

        # Original state should not be modified
        assert validation_state["overall_status"] == original_status
        # Result state should have updated status
        assert result_state["overall_status"] != original_status

    def test_callable_interface(self, validation_state):
        """Test that agent is callable (for LangGraph)."""
        agent = AggregatorAgent()
        result_state = agent(validation_state)

        assert result_state is not None
        assert "overall_status" in result_state

    def test_error_deduplication_applied(self):
        """Test that error deduplication is applied."""
        agent = AggregatorAgent()

        # Create duplicate errors
        error = ErrorDetail(
            severity="error",
            message="Duplicate error",
            path="field",
            validator="v1",
        )

        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": [],
            "validation_results": [
                ValidationResult(
                    validator_name="v1",
                    status="failed",
                    errors=[error],
                ),
                ValidationResult(
                    validator_name="v1",
                    status="failed",
                    errors=[error],
                ),
            ],
            "errors": [],
            "overall_status": "pending",
            "confidence_score": None,
            "final_report": None,
        }

        result_state = agent.execute(state)
        # Should have merged results
        assert len(result_state["validation_results"]) == 1

    def test_confidence_breakdown_in_report(self, validation_state):
        """Test that confidence breakdown is included in report."""
        agent = AggregatorAgent()
        result_state = agent.execute(validation_state)

        breakdown = result_state["final_report"]["confidence_breakdown"]
        assert "confidence" in breakdown
        assert "breakdown" in breakdown
        assert "pass_rate" in breakdown["breakdown"]
