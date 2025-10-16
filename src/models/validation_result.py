"""Validation result models."""
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Details about a validation error."""

    path: str = Field(description="JSON path to the error location")
    message: str = Field(description="Human-readable error message")
    code: str = Field(description="Error code for programmatic handling")
    severity: str = Field(default="error", description="Error severity: error, warning, info")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ValidationResult(BaseModel):
    """Result from a single validator."""

    validator_name: str = Field(description="Name of the validator that produced this result")
    status: str = Field(description="Status: passed, failed, error, skipped")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    errors: List[ErrorDetail] = Field(default_factory=list, description="List of errors found")
    warnings: List[ErrorDetail] = Field(default_factory=list, description="List of warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this result was created")


class AggregatedResult(BaseModel):
    """Aggregated validation results from multiple validators."""

    overall_status: str = Field(description="Overall status: passed, failed, partial")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Aggregate confidence score")
    validation_results: List[ValidationResult] = Field(default_factory=list)
    summary: str = Field(default="", description="Human-readable summary")
    total_errors: int = Field(default=0, description="Total number of errors")
    total_warnings: int = Field(default=0, description="Total number of warnings")
    execution_time_ms: float = Field(default=0.0, description="Total execution time")
    timestamp: datetime = Field(default_factory=datetime.now)

    def get_all_errors(self) -> List[ErrorDetail]:
        """Get all errors from all validation results."""
        errors = []
        for result in self.validation_results:
            errors.extend(result.errors)
        return errors

    def get_all_warnings(self) -> List[ErrorDetail]:
        """Get all warnings from all validation results."""
        warnings = []
        for result in self.validation_results:
            warnings.extend(result.warnings)
        return warnings

    def get_report(self, format: str = "text") -> str:
        """Generate a formatted report.

        Args:
            format: Report format - 'text', 'markdown', or 'json'

        Returns:
            Formatted report string
        """
        if format == "json":
            return self.model_dump_json(indent=2)
        elif format == "markdown":
            return self._generate_markdown_report()
        else:
            return self._generate_text_report()

    def _generate_text_report(self) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Overall Status: {self.overall_status.upper()}",
            f"Confidence Score: {self.confidence_score:.2%}",
            f"Execution Time: {self.execution_time_ms:.2f}ms",
            f"Timestamp: {self.timestamp.isoformat()}",
            "",
            f"Total Errors: {self.total_errors}",
            f"Total Warnings: {self.total_warnings}",
            "",
            "VALIDATOR RESULTS",
            "-" * 60,
        ]

        for result in self.validation_results:
            lines.extend([
                f"\n{result.validator_name}:",
                f"  Status: {result.status}",
                f"  Confidence: {result.confidence:.2%}",
                f"  Execution Time: {result.execution_time_ms:.2f}ms",
            ])

            if result.errors:
                lines.append(f"  Errors ({len(result.errors)}):")
                for error in result.errors[:5]:  # Show first 5
                    lines.append(f"    - {error.path}: {error.message}")
                if len(result.errors) > 5:
                    lines.append(f"    ... and {len(result.errors) - 5} more")

            if result.warnings:
                lines.append(f"  Warnings ({len(result.warnings)}):")
                for warning in result.warnings[:3]:  # Show first 3
                    lines.append(f"    - {warning.path}: {warning.message}")
                if len(result.warnings) > 3:
                    lines.append(f"    ... and {len(result.warnings) - 3} more")

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _generate_markdown_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Validation Report",
            "",
            f"**Overall Status:** {self.overall_status.upper()}  ",
            f"**Confidence Score:** {self.confidence_score:.2%}  ",
            f"**Execution Time:** {self.execution_time_ms:.2f}ms  ",
            f"**Timestamp:** {self.timestamp.isoformat()}  ",
            "",
            f"**Total Errors:** {self.total_errors}  ",
            f"**Total Warnings:** {self.total_warnings}  ",
            "",
            "## Validator Results",
            "",
        ]

        for result in self.validation_results:
            status_icon = "✅" if result.status == "passed" else "❌" if result.status == "failed" else "⚠️"
            lines.extend([
                f"### {status_icon} {result.validator_name}",
                "",
                f"- **Status:** {result.status}",
                f"- **Confidence:** {result.confidence:.2%}",
                f"- **Execution Time:** {result.execution_time_ms:.2f}ms",
                "",
            ])

            if result.errors:
                lines.append(f"**Errors ({len(result.errors)}):**")
                lines.append("")
                for error in result.errors:
                    lines.append(f"- `{error.path}`: {error.message}")
                lines.append("")

            if result.warnings:
                lines.append(f"**Warnings ({len(result.warnings)}):**")
                lines.append("")
                for warning in result.warnings:
                    lines.append(f"- `{warning.path}`: {warning.message}")
                lines.append("")

        return "\n".join(lines)
