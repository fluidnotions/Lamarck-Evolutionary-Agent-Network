"""Unit tests for error analyzer."""

import pytest
from src.aggregator.error_analysis import ErrorAnalyzer
from src.models import ValidationResult, ErrorDetail


class TestErrorAnalyzer:
    """Test cases for ErrorAnalyzer."""

    def test_analyze_empty_results(self):
        """Test analysis with no results."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze([])

        assert analysis["total_errors"] == 0
        assert analysis["total_warnings"] == 0
        assert len(analysis["by_severity"]) == 0

    def test_analyze_with_errors(self, failing_result):
        """Test analysis with errors."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze([failing_result])

        assert analysis["total_errors"] > 0
        assert "by_severity" in analysis
        assert "by_validator" in analysis
        assert "statistics" in analysis

    def test_group_by_severity(self, mixed_results):
        """Test grouping errors by severity."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(mixed_results)

        by_severity = analysis["by_severity"]
        # Should have different severity levels
        assert len(by_severity) > 0

        # Each severity should have count and errors list
        for severity, data in by_severity.items():
            assert "count" in data
            assert "errors" in data
            assert isinstance(data["count"], int)
            assert isinstance(data["errors"], list)

    def test_group_by_validator(self, mixed_results):
        """Test grouping by validator."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(mixed_results)

        by_validator = analysis["by_validator"]
        assert len(by_validator) == 3  # Three validators in mixed_results

        for validator, data in by_validator.items():
            assert "status" in data
            assert "error_count" in data
            assert "warning_count" in data
            assert "total_issues" in data

    def test_group_by_path(self):
        """Test grouping errors by path."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Error 1",
                        path="section.field1",
                        validator="test",
                    ),
                    ErrorDetail(
                        severity="error",
                        message="Error 2",
                        path="section.field2",
                        validator="test",
                    ),
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        by_path = analysis["by_path"]

        # Should have entries for the paths
        assert len(by_path) > 0
        for path, data in by_path.items():
            assert "count" in data
            assert "severities" in data
            assert "validators" in data

    def test_detect_similar_message_patterns(self):
        """Test detection of similar error messages."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Invalid value at field1",
                        path="field1",
                        validator="test",
                    ),
                    ErrorDetail(
                        severity="error",
                        message="Invalid value at field2",
                        path="field2",
                        validator="test",
                    ),
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        patterns = analysis["patterns"]

        # Should detect pattern of similar messages
        assert len(patterns) > 0
        pattern = patterns[0]
        assert pattern["type"] == "similar_messages"
        assert pattern["count"] >= 2

    def test_detect_common_path_prefix_pattern(self):
        """Test detection of errors in same section."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message=f"Error {i}",
                        path=f"section.field{i}",
                        validator="test",
                    )
                    for i in range(4)
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        patterns = analysis["patterns"]

        # Should detect pattern of common path prefix
        common_prefix_patterns = [p for p in patterns if p["type"] == "common_path_prefix"]
        assert len(common_prefix_patterns) > 0

    def test_top_errors_ranked_by_severity_and_count(self):
        """Test that top errors are properly ranked."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    # Multiple instances of same error
                    ErrorDetail(
                        severity="critical",
                        message="Critical error",
                        validator="test",
                    ),
                    ErrorDetail(
                        severity="critical",
                        message="Critical error",
                        validator="test",
                    ),
                    ErrorDetail(
                        severity="warning",
                        message="Warning message",
                        validator="test",
                    ),
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        top_errors = analysis["top_errors"]

        assert len(top_errors) > 0
        # Critical error should be first despite same count
        assert top_errors[0]["severity"] == "critical"

    def test_statistics_calculation(self, mixed_results):
        """Test error statistics calculation."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(mixed_results)

        stats = analysis["statistics"]
        assert "unique_errors" in stats
        assert "affected_validators" in stats
        assert "error_rate" in stats
        assert "avg_errors_per_validator" in stats
        assert "total_validators" in stats

        assert stats["total_validators"] == 3
        assert 0.0 <= stats["error_rate"] <= 1.0

    def test_recommendations_generation(self):
        """Test that recommendations are generated."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="critical",
                        message="Critical error",
                        validator="test",
                    ),
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        recommendations = analysis["recommendations"]

        assert len(recommendations) > 0
        # Should mention critical errors
        assert any("critical" in rec.lower() for rec in recommendations)

    def test_recommendations_for_warnings(self):
        """Test recommendations when there are many warnings."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="passed",
                warnings=[
                    ErrorDetail(
                        severity="warning",
                        message=f"Warning {i}",
                        validator="test",
                    )
                    for i in range(10)
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        recommendations = analysis["recommendations"]

        # Should recommend reviewing warnings
        assert any("warning" in rec.lower() for rec in recommendations)

    def test_pattern_recommendations(self):
        """Test that pattern-based recommendations are generated."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="failed",
                errors=[
                    ErrorDetail(
                        severity="error",
                        message="Invalid value",
                        path=f"section.field{i}",
                        validator="test",
                    )
                    for i in range(4)
                ],
            )
        ]

        analysis = analyzer.analyze(results)
        recommendations = analysis["recommendations"]

        # Should have pattern-based recommendations
        assert len(recommendations) > 0

    def test_no_errors_recommendation(self):
        """Test recommendation when there are no errors."""
        analyzer = ErrorAnalyzer()

        results = [
            ValidationResult(
                validator_name="test",
                status="passed",
            )
        ]

        analysis = analyzer.analyze(results)
        recommendations = analysis["recommendations"]

        assert len(recommendations) > 0
        assert any("good" in rec.lower() or "no" in rec.lower() for rec in recommendations)

    def test_analyze_with_llm_flag(self, mixed_results):
        """Test analysis with LLM flag (note: actual LLM not implemented)."""
        analyzer = ErrorAnalyzer()
        analysis = analyzer.analyze(mixed_results, use_llm=True)

        # Should still work even though LLM is not implemented
        assert "recommendations" in analysis
