"""Integration tests for complete aggregator workflow."""

import pytest
from src.agents.aggregator import AggregatorAgent
from src.models import ValidationResult, ErrorDetail
from src.aggregator.confidence import ConfidenceCalculator
from src.aggregator.error_analysis import ErrorAnalyzer
from src.aggregator.report_generator import ReportGenerator
from src.aggregator.result_merger import ResultMerger
from src.aggregator.visualization import Visualizer


class TestAggregatorIntegration:
    """Integration tests for complete aggregator workflow."""

    def test_complete_workflow_with_mixed_results(self):
        """Test complete workflow from results to final report."""
        # Create realistic validation results
        results = [
            ValidationResult(
                validator_name="schema_validator",
                status="passed",
                errors=[],
                warnings=[],
                execution_time=0.5,
                coverage=1.0,
                metadata={"schema_version": "1.0"},
            ),
            ValidationResult(
                validator_name="business_rules",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="critical",
                        message="Required field 'customer_id' is missing",
                        path="order.customer_id",
                        validator="business_rules",
                    ),
                    ErrorDetail(
                        severity="error",
                        message="Invalid date format",
                        path="order.date",
                        validator="business_rules",
                    ),
                ],
                warnings=[],
                execution_time=1.2,
                coverage=0.98,
            ),
            ValidationResult(
                validator_name="data_quality",
                status="passed",
                errors=[],
                warnings=[
                    ErrorDetail(
                        severity="warning",
                        message="Field 'description' is empty",
                        path="order.description",
                        validator="data_quality",
                    ),
                ],
                execution_time=0.8,
                coverage=1.0,
            ),
        ]

        # Create state
        state = {
            "input_data": {"order": {"id": "123"}},
            "validation_request": {"type": "full_validation"},
            "active_validators": [],
            "completed_validators": ["schema_validator", "business_rules", "data_quality"],
            "validation_results": results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        # Execute aggregator
        agent = AggregatorAgent(report_format="markdown", generate_visualizations=True)
        result_state = agent.execute(state)

        # Verify complete workflow
        assert result_state["overall_status"] == "failed"
        assert result_state["confidence_score"] is not None
        assert 0.0 <= result_state["confidence_score"] <= 1.0

        # Verify report structure
        report = result_state["final_report"]
        assert report is not None
        assert "content" in report
        assert "metadata" in report
        assert "error_analysis" in report
        assert "visualizations" in report

        # Verify metadata
        metadata = report["metadata"]
        assert metadata["validator_count"] == 3
        assert metadata["total_errors"] == 2
        assert metadata["total_warnings"] == 1

        # Verify error analysis
        error_analysis = report["error_analysis"]
        assert error_analysis["total_errors"] == 2
        assert error_analysis["total_warnings"] == 1
        assert "by_severity" in error_analysis
        assert "recommendations" in error_analysis

        # Verify visualizations
        viz = report["visualizations"]
        assert "pass_fail_chart" in viz
        assert "severity_distribution" in viz
        assert "confidence_gauge" in viz

    def test_end_to_end_with_all_passing(self):
        """Test end-to-end workflow with all validators passing."""
        results = [
            ValidationResult(
                validator_name=f"validator_{i}",
                status="passed",
                execution_time=0.5,
                coverage=1.0,
            )
            for i in range(5)
        ]

        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": [f"validator_{i}" for i in range(5)],
            "validation_results": results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        agent = AggregatorAgent()
        result_state = agent.execute(state)

        assert result_state["overall_status"] == "completed"
        assert result_state["confidence_score"] == 1.0

        report = result_state["final_report"]
        assert report["error_analysis"]["total_errors"] == 0

    def test_components_work_together(self):
        """Test that all aggregator components work together."""
        # Create test data
        results = [
            ValidationResult(
                validator_name="test_validator",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Test error",
                        validator="test_validator",
                    ),
                ],
                execution_time=1.0,
            ),
        ]

        # Test result merger
        merger = ResultMerger()
        merged = merger.merge_results(results)
        assert len(merged) == 1

        # Test confidence calculator
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate(merged)
        assert 0.0 <= confidence <= 1.0

        # Test error analyzer
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(merged)
        assert analysis["total_errors"] == 1

        # Test report generator
        generator = ReportGenerator()
        report = generator.generate(
            results=merged,
            confidence=confidence,
            error_analysis=analysis,
            format="markdown",
        )
        assert "Validation Report" in report["content"]

        # Test visualizer
        visualizer = Visualizer()
        viz = visualizer.generate_all_visualizations(merged, confidence, analysis)
        assert "pass_fail_chart" in viz

    def test_duplicate_error_handling_across_workflow(self):
        """Test that duplicate errors are handled throughout workflow."""
        # Create duplicate errors
        results = [
            ValidationResult(
                validator_name="validator1",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Duplicate error",
                        path="field",
                        validator="validator1",
                    ),
                ],
            ),
            ValidationResult(
                validator_name="validator2",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Duplicate error",
                        path="field",
                        validator="validator2",
                    ),
                ],
            ),
        ]

        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": ["validator1", "validator2"],
            "validation_results": results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        agent = AggregatorAgent()
        result_state = agent.execute(state)

        # Check that deduplication was applied
        merged_results = result_state["validation_results"]
        for result in merged_results:
            if result.errors:
                # Should have deduplication markers
                assert result.errors[0].context.get("is_duplicate") == True

    def test_pattern_detection_in_workflow(self):
        """Test that error patterns are detected in full workflow."""
        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Invalid value at field1",
                        path="section.field1",
                        validator="test",
                    ),
                    ErrorDetail(
                        severity="error",
                        message="Invalid value at field2",
                        path="section.field2",
                        validator="test",
                    ),
                    ErrorDetail(
                        severity="error",
                        message="Invalid value at field3",
                        path="section.field3",
                        validator="test",
                    ),
                ],
            ),
        ]

        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": ["test"],
            "validation_results": results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        agent = AggregatorAgent()
        result_state = agent.execute(state)

        error_analysis = result_state["final_report"]["error_analysis"]
        patterns = error_analysis["patterns"]

        # Should detect patterns
        assert len(patterns) > 0

    def test_report_format_consistency(self):
        """Test that reports are consistent across formats."""
        results = [
            ValidationResult(
                validator_name="test",
                status="passed",
            ),
        ]

        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": ["test"],
            "validation_results": results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        # Test all formats
        for format_type in ["json", "markdown", "html"]:
            agent = AggregatorAgent(report_format=format_type)
            result_state = agent.execute(state)

            report = result_state["final_report"]
            assert report["format"] == format_type
            assert len(report["content"]) > 0

    def test_confidence_affects_recommendations(self):
        """Test that confidence score affects recommendations."""
        # High confidence scenario
        high_conf_results = [
            ValidationResult(validator_name=f"v{i}", status="passed")
            for i in range(5)
        ]

        # Low confidence scenario
        low_conf_results = [
            ValidationResult(
                validator_name=f"v{i}",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="critical",
                        message="Critical error",
                        validator=f"v{i}",
                    ),
                ],
            )
            for i in range(5)
        ]

        agent = AggregatorAgent()

        # Test high confidence
        state_high = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": [],
            "validation_results": high_conf_results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        result_high = agent.execute(state_high)
        recs_high = result_high["final_report"]["error_analysis"]["recommendations"]

        # Test low confidence
        state_low = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": [],
            "validation_results": low_conf_results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        result_low = agent.execute(state_low)
        recs_low = result_low["final_report"]["error_analysis"]["recommendations"]

        # Recommendations should be different
        assert recs_high != recs_low

    def test_large_scale_aggregation(self):
        """Test aggregation with many validators and errors."""
        # Create 20 validators with various results
        results = []
        for i in range(20):
            if i % 3 == 0:
                # Failed validator
                results.append(
                    ValidationResult(
                        validator_name=f"validator_{i}",
                        status="failed",
                        errors=[
                            ErrorDetail(
                                severity="error",
                                message=f"Error {j}",
                                validator=f"validator_{i}",
                            )
                            for j in range(5)
                        ],
                        execution_time=1.0,
                    )
                )
            else:
                # Passing validator
                results.append(
                    ValidationResult(
                        validator_name=f"validator_{i}",
                        status="passed",
                        execution_time=0.5,
                    )
                )

        state = {
            "input_data": {},
            "validation_request": {},
            "active_validators": [],
            "completed_validators": [f"validator_{i}" for i in range(20)],
            "validation_results": results,
            "errors": [],
            "overall_status": "in_progress",
            "confidence_score": None,
            "final_report": None,
        }

        agent = AggregatorAgent()
        result_state = agent.execute(state)

        # Should handle large scale without issues
        assert result_state["overall_status"] in ["completed", "failed"]
        assert result_state["confidence_score"] is not None

        report = result_state["final_report"]
        assert report["metadata"]["validator_count"] == 20
