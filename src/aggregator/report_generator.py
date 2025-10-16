"""Comprehensive report generator supporting multiple output formats."""

import json
from datetime import datetime
from typing import Any, Optional
from src.models import ValidationResult


class ReportGenerator:
    """
    Generates validation reports in multiple formats.

    Supported formats:
    - JSON: Structured data for programmatic consumption
    - Markdown: Human-readable text format
    - HTML: Interactive web format
    - PDF: Formal document format (requires HTML conversion)
    """

    def __init__(self):
        """Initialize report generator."""
        pass

    def generate(
        self,
        results: list[ValidationResult],
        confidence: float,
        error_analysis: dict[str, Any],
        format: str = "markdown",
        confidence_breakdown: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            results: Validation results to report on
            confidence: Overall confidence score
            error_analysis: Error analysis data
            format: Output format (json, markdown, html, pdf)
            confidence_breakdown: Optional detailed confidence breakdown

        Returns:
            Report dictionary with content and metadata
        """
        # Build report data structure
        report_data = self._build_report_data(
            results, confidence, error_analysis, confidence_breakdown
        )

        # Format based on requested type
        if format == "markdown":
            content = self._format_as_markdown(report_data)
        elif format == "html":
            content = self._format_as_html(report_data)
        elif format == "pdf":
            # PDF generation requires HTML first, then conversion
            html_content = self._format_as_html(report_data)
            content = self._format_as_pdf(html_content, report_data)
        else:  # Default to JSON
            content = self._format_as_json(report_data)

        return {
            "content": content,
            "format": format,
            "metadata": report_data["metadata"],
        }

    def _build_report_data(
        self,
        results: list[ValidationResult],
        confidence: float,
        error_analysis: dict[str, Any],
        confidence_breakdown: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Build structured report data.

        Args:
            results: Validation results
            confidence: Confidence score
            error_analysis: Error analysis data
            confidence_breakdown: Optional confidence breakdown

        Returns:
            Structured report data
        """
        # Generate summary
        summary = self._generate_summary(results, confidence)

        # Generate overview
        overview = self._generate_overview(results)

        # Generate detailed findings
        detailed_findings = self._generate_findings(results)

        # Generate recommendations
        recommendations = error_analysis.get("recommendations", [])

        # Metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "validator_count": len(results),
            "confidence_score": confidence,
            "total_errors": error_analysis.get("total_errors", 0),
            "total_warnings": error_analysis.get("total_warnings", 0),
        }

        return {
            "summary": summary,
            "overview": overview,
            "detailed_findings": detailed_findings,
            "error_analysis": error_analysis,
            "recommendations": recommendations,
            "confidence_breakdown": confidence_breakdown,
            "metadata": metadata,
        }

    def _generate_summary(
        self, results: list[ValidationResult], confidence: float
    ) -> dict[str, Any]:
        """Generate executive summary."""
        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")

        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)

        # Determine overall status
        if failed == 0 and total_errors == 0:
            overall_status = "PASSED"
            status_description = "All validations passed successfully."
        elif failed > 0:
            overall_status = "FAILED"
            status_description = f"{failed} validator(s) failed with {total_errors} error(s)."
        else:
            overall_status = "PASSED_WITH_WARNINGS"
            status_description = f"Passed with {total_warnings} warning(s)."

        return {
            "overall_status": overall_status,
            "status_description": status_description,
            "confidence_score": confidence,
            "total_validators": len(results),
            "passed_validators": passed,
            "failed_validators": failed,
            "skipped_validators": skipped,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
        }

    def _generate_overview(
        self, results: list[ValidationResult]
    ) -> dict[str, Any]:
        """Generate validation overview with statistics."""
        # Calculate statistics
        total_execution_time = sum(r.execution_time for r in results)
        avg_execution_time = total_execution_time / len(results) if results else 0.0
        avg_coverage = sum(r.coverage for r in results) / len(results) if results else 0.0

        # Validator status breakdown
        status_breakdown = {}
        for result in results:
            status_breakdown[result.validator_name] = {
                "status": result.status,
                "errors": len(result.errors),
                "warnings": len(result.warnings),
                "execution_time": result.execution_time,
                "coverage": result.coverage,
            }

        return {
            "total_execution_time": round(total_execution_time, 3),
            "avg_execution_time": round(avg_execution_time, 3),
            "avg_coverage": round(avg_coverage, 3),
            "validator_breakdown": status_breakdown,
        }

    def _generate_findings(
        self, results: list[ValidationResult]
    ) -> list[dict[str, Any]]:
        """Generate detailed findings for each validator."""
        findings = []

        for result in results:
            finding = {
                "validator": result.validator_name,
                "status": result.status,
                "execution_time": result.execution_time,
                "coverage": result.coverage,
                "errors": [
                    {
                        "severity": e.severity,
                        "message": e.message,
                        "path": e.path,
                        "context": e.context,
                    }
                    for e in result.errors
                ],
                "warnings": [
                    {
                        "severity": w.severity,
                        "message": w.message,
                        "path": w.path,
                    }
                    for w in result.warnings
                ],
                "metadata": result.metadata,
            }
            findings.append(finding)

        return findings

    def _format_as_json(self, report_data: dict[str, Any]) -> str:
        """Format report as JSON."""
        return json.dumps(report_data, indent=2, default=str)

    def _format_as_markdown(self, report_data: dict[str, Any]) -> str:
        """Format report as Markdown."""
        summary = report_data["summary"]
        overview = report_data["overview"]
        findings = report_data["detailed_findings"]
        error_analysis = report_data["error_analysis"]
        recommendations = report_data["recommendations"]
        metadata = report_data["metadata"]

        md = []

        # Title and metadata
        md.append("# Validation Report")
        md.append(f"\nGenerated: {metadata['generated_at']}")
        md.append("\n---\n")

        # Executive Summary
        md.append("## Executive Summary\n")
        md.append(f"**Status:** {summary['overall_status']}")
        md.append(f"\n{summary['status_description']}\n")
        md.append(f"**Confidence Score:** {summary['confidence_score']:.1%}\n")
        md.append("### Statistics")
        md.append(f"- Total Validators: {summary['total_validators']}")
        md.append(f"- Passed: {summary['passed_validators']}")
        md.append(f"- Failed: {summary['failed_validators']}")
        md.append(f"- Errors: {summary['total_errors']}")
        md.append(f"- Warnings: {summary['total_warnings']}")
        md.append("")

        # Confidence Breakdown
        if report_data.get("confidence_breakdown"):
            md.append("### Confidence Breakdown")
            breakdown = report_data["confidence_breakdown"]["breakdown"]
            md.append(f"- Pass Rate: {breakdown['pass_rate']:.1%}")
            md.append(f"- Severity Score: {breakdown['severity']:.1%}")
            md.append(f"- Coverage: {breakdown['coverage']:.1%}")
            md.append(f"- Reliability: {breakdown['reliability']:.1%}")
            md.append("")

        # Validation Overview
        md.append("## Validation Overview\n")
        md.append(f"**Total Execution Time:** {overview['total_execution_time']:.3f}s")
        md.append(f"**Average Coverage:** {overview['avg_coverage']:.1%}\n")

        # Validator Status Table
        md.append("### Validator Status\n")
        md.append("| Validator | Status | Errors | Warnings | Time (s) | Coverage |")
        md.append("|-----------|--------|--------|----------|----------|----------|")
        for validator, data in overview["validator_breakdown"].items():
            md.append(
                f"| {validator} | {data['status']} | {data['errors']} | "
                f"{data['warnings']} | {data['execution_time']:.3f} | "
                f"{data['coverage']:.1%} |"
            )
        md.append("")

        # Error Analysis
        if error_analysis.get("total_errors", 0) > 0 or error_analysis.get("total_warnings", 0) > 0:
            md.append("## Error Analysis\n")

            # By Severity
            if error_analysis.get("by_severity"):
                md.append("### Errors by Severity\n")
                for severity, data in error_analysis["by_severity"].items():
                    md.append(f"- **{severity.upper()}**: {data['count']}")
                md.append("")

            # Top Errors
            if error_analysis.get("top_errors"):
                md.append("### Top Errors\n")
                for i, error in enumerate(error_analysis["top_errors"], 1):
                    md.append(f"{i}. **{error['severity'].upper()}** ({error['count']}x): {error['message']}")
                    md.append(f"   - Validators: {', '.join(error['validators'])}")
                md.append("")

            # Patterns
            if error_analysis.get("patterns"):
                md.append("### Detected Patterns\n")
                for pattern in error_analysis["patterns"]:
                    md.append(f"- {pattern['description']} (Count: {pattern['count']})")
                md.append("")

        # Detailed Findings
        md.append("## Detailed Findings\n")
        for finding in findings:
            md.append(f"### {finding['validator']}\n")
            md.append(f"**Status:** {finding['status']}")
            md.append(f"**Execution Time:** {finding['execution_time']:.3f}s")
            md.append(f"**Coverage:** {finding['coverage']:.1%}\n")

            if finding["errors"]:
                md.append("#### Errors\n")
                for error in finding["errors"]:
                    path_str = f" at `{error['path']}`" if error["path"] else ""
                    md.append(f"- **{error['severity'].upper()}**{path_str}: {error['message']}")
                md.append("")

            if finding["warnings"]:
                md.append("#### Warnings\n")
                for warning in finding["warnings"]:
                    path_str = f" at `{warning['path']}`" if warning["path"] else ""
                    md.append(f"- {warning['message']}{path_str}")
                md.append("")

        # Recommendations
        if recommendations:
            md.append("## Recommendations\n")
            for i, rec in enumerate(recommendations, 1):
                md.append(f"{i}. {rec}")
            md.append("")

        return "\n".join(md)

    def _format_as_html(self, report_data: dict[str, Any]) -> str:
        """Format report as HTML."""
        summary = report_data["summary"]
        overview = report_data["overview"]
        findings = report_data["detailed_findings"]
        error_analysis = report_data["error_analysis"]
        recommendations = report_data["recommendations"]
        metadata = report_data["metadata"]

        # Status color
        status_colors = {
            "PASSED": "#28a745",
            "FAILED": "#dc3545",
            "PASSED_WITH_WARNINGS": "#ffc107",
        }
        status_color = status_colors.get(summary["overall_status"], "#6c757d")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
            background-color: {status_color};
        }}
        .confidence-score {{
            font-size: 2em;
            font-weight: bold;
            color: {status_color};
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid {status_color};
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .error-item {{
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #dc3545;
            background-color: #fff5f5;
        }}
        .warning-item {{
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #ffc107;
            background-color: #fffef5;
        }}
        .recommendation {{
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #17a2b8;
            background-color: #f0f9ff;
        }}
        .severity-critical {{ color: #dc3545; font-weight: bold; }}
        .severity-error {{ color: #fd7e14; font-weight: bold; }}
        .severity-warning {{ color: #ffc107; font-weight: bold; }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Report</h1>
        <p class="metadata">Generated: {metadata['generated_at']}</p>

        <h2>Executive Summary</h2>
        <div class="status-badge">{summary['overall_status']}</div>
        <p>{summary['status_description']}</p>
        <div class="confidence-score">{summary['confidence_score']:.1%}</div>
        <p style="color: #666;">Confidence Score</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{summary['total_validators']}</div>
                <div class="stat-label">Total Validators</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['passed_validators']}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['failed_validators']}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['total_errors']}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary['total_warnings']}</div>
                <div class="stat-label">Warnings</div>
            </div>
        </div>

        <h2>Validation Overview</h2>
        <p><strong>Total Execution Time:</strong> {overview['total_execution_time']:.3f}s</p>
        <p><strong>Average Coverage:</strong> {overview['avg_coverage']:.1%}</p>

        <h3>Validator Status</h3>
        <table>
            <thead>
                <tr>
                    <th>Validator</th>
                    <th>Status</th>
                    <th>Errors</th>
                    <th>Warnings</th>
                    <th>Time (s)</th>
                    <th>Coverage</th>
                </tr>
            </thead>
            <tbody>
"""

        for validator, data in overview["validator_breakdown"].items():
            html += f"""
                <tr>
                    <td>{validator}</td>
                    <td>{data['status']}</td>
                    <td>{data['errors']}</td>
                    <td>{data['warnings']}</td>
                    <td>{data['execution_time']:.3f}</td>
                    <td>{data['coverage']:.1%}</td>
                </tr>
"""

        html += """
            </tbody>
        </table>
"""

        # Error Analysis
        if error_analysis.get("total_errors", 0) > 0:
            html += """
        <h2>Error Analysis</h2>
"""
            if error_analysis.get("top_errors"):
                html += "<h3>Top Errors</h3>"
                for error in error_analysis["top_errors"]:
                    html += f"""
        <div class="error-item">
            <span class="severity-{error['severity']}">{error['severity'].upper()}</span>
            <strong>({error['count']}x)</strong>: {error['message']}
            <br><small>Validators: {', '.join(error['validators'])}</small>
        </div>
"""

        # Recommendations
        if recommendations:
            html += """
        <h2>Recommendations</h2>
"""
            for rec in recommendations:
                html += f"""
        <div class="recommendation">{rec}</div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def _format_as_pdf(
        self, html_content: str, report_data: dict[str, Any]
    ) -> str:
        """
        Format report as PDF (requires HTML to PDF conversion).

        Note: Actual PDF generation would require weasyprint library.
        For now, we return instructions for PDF conversion.

        Args:
            html_content: HTML content to convert
            report_data: Report data

        Returns:
            PDF generation instructions or binary content
        """
        # In a real implementation, we would use weasyprint:
        # from weasyprint import HTML
        # pdf_bytes = HTML(string=html_content).write_pdf()
        # return pdf_bytes

        return (
            "PDF generation requires weasyprint library. "
            "HTML content is available for conversion:\n\n" + html_content
        )
