"""Unit tests for report generator."""

import pytest
import json
from src.aggregator.report_generator import ReportGenerator
from src.aggregator.error_analysis import ErrorAnalyzer
from src.models import ValidationResult


class TestReportGenerator:
    """Test cases for ReportGenerator."""

    def test_generate_json_report(self, mixed_results):
        """Test JSON report generation."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="json",
        )

        assert report["format"] == "json"
        assert "content" in report
        assert "metadata" in report

        # Should be valid JSON
        content = json.loads(report["content"])
        assert "summary" in content
        assert "overview" in content
        assert "detailed_findings" in content

    def test_generate_markdown_report(self, mixed_results):
        """Test Markdown report generation."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="markdown",
        )

        assert report["format"] == "markdown"
        content = report["content"]

        # Check for markdown elements
        assert "# Validation Report" in content
        assert "## Executive Summary" in content
        assert "## Validation Overview" in content
        assert "##" in content  # Has headers

    def test_generate_html_report(self, mixed_results):
        """Test HTML report generation."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="html",
        )

        assert report["format"] == "html"
        content = report["content"]

        # Check for HTML elements
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<body>" in content
        assert "<h1>" in content
        assert "<table>" in content

    def test_generate_pdf_report(self, mixed_results):
        """Test PDF report generation (returns instructions)."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="pdf",
        )

        assert report["format"] == "pdf"
        # PDF generation returns HTML + instructions
        assert "weasyprint" in report["content"]

    def test_report_includes_confidence_score(self, mixed_results):
        """Test that report includes confidence score."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.85,
            error_analysis=error_analysis,
            format="json",
        )

        content = json.loads(report["content"])
        assert content["metadata"]["confidence_score"] == 0.85

    def test_report_includes_error_analysis(self, failing_result):
        """Test that report includes error analysis."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze([failing_result])

        report = generator.generate(
            results=[failing_result],
            confidence=0.3,
            error_analysis=error_analysis,
            format="json",
        )

        content = json.loads(report["content"])
        assert "error_analysis" in content
        assert content["error_analysis"]["total_errors"] > 0

    def test_report_includes_recommendations(self, mixed_results):
        """Test that report includes recommendations."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="json",
        )

        content = json.loads(report["content"])
        assert "recommendations" in content
        assert len(content["recommendations"]) > 0

    def test_summary_status_all_passed(self):
        """Test summary status when all validators pass."""
        generator = ReportGenerator()

        results = [
            ValidationResult(
                validator_name=f"v{i}",
                status="passed",
            )
            for i in range(3)
        ]

        report = generator.generate(
            results=results,
            confidence=1.0,
            error_analysis={"total_errors": 0, "total_warnings": 0},
            format="json",
        )

        content = json.loads(report["content"])
        assert content["summary"]["overall_status"] == "PASSED"

    def test_summary_status_with_failures(self, failing_result):
        """Test summary status when there are failures."""
        generator = ReportGenerator()

        report = generator.generate(
            results=[failing_result],
            confidence=0.3,
            error_analysis={"total_errors": 2, "total_warnings": 0},
            format="json",
        )

        content = json.loads(report["content"])
        assert content["summary"]["overall_status"] == "FAILED"

    def test_overview_includes_execution_stats(self, mixed_results):
        """Test that overview includes execution statistics."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="json",
        )

        content = json.loads(report["content"])
        overview = content["overview"]

        assert "total_execution_time" in overview
        assert "avg_execution_time" in overview
        assert "avg_coverage" in overview
        assert "validator_breakdown" in overview

    def test_detailed_findings_per_validator(self, mixed_results):
        """Test detailed findings for each validator."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="json",
        )

        content = json.loads(report["content"])
        findings = content["detailed_findings"]

        assert len(findings) == 3  # Three validators
        for finding in findings:
            assert "validator" in finding
            assert "status" in finding
            assert "errors" in finding
            assert "warnings" in finding

    def test_markdown_table_formatting(self, mixed_results):
        """Test that markdown report includes proper table formatting."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="markdown",
        )

        content = report["content"]
        # Check for markdown table
        assert "| Validator |" in content
        assert "|-----------|" in content

    def test_html_styling(self, mixed_results):
        """Test that HTML report includes styling."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="html",
        )

        content = report["content"]
        assert "<style>" in content
        assert "font-family" in content
        assert ".container" in content

    def test_confidence_breakdown_included(self, mixed_results):
        """Test that confidence breakdown is included when provided."""
        generator = ReportGenerator()
        analyzer = ErrorAnalyzer()
        error_analysis = analyzer.analyze(mixed_results)

        confidence_breakdown = {
            "confidence": 0.75,
            "breakdown": {
                "pass_rate": 0.8,
                "severity": 0.7,
                "coverage": 0.9,
                "reliability": 1.0,
            },
        }

        report = generator.generate(
            results=mixed_results,
            confidence=0.75,
            error_analysis=error_analysis,
            format="json",
            confidence_breakdown=confidence_breakdown,
        )

        content = json.loads(report["content"])
        assert "confidence_breakdown" in content
        assert content["confidence_breakdown"]["breakdown"]["pass_rate"] == 0.8

    def test_empty_results_report(self):
        """Test report generation with empty results."""
        generator = ReportGenerator()

        report = generator.generate(
            results=[],
            confidence=0.0,
            error_analysis={},
            format="json",
        )

        assert report["format"] == "json"
        content = json.loads(report["content"])
        assert content["metadata"]["validator_count"] == 0
