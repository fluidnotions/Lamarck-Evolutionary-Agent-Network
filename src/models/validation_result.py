"""Validation result models."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Details of a validation error."""

    severity: str = Field(
        description="Error severity: critical, error, warning, or info"
    )
    message: str = Field(description="Error message")
    path: Optional[str] = Field(
        default=None, description="Path to the field that failed validation"
    )
    validator: str = Field(description="Name of the validator that produced this error")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Additional context about the error"
    )
    timestamp: datetime = Field(default_factory=datetime.now)

    def __hash__(self) -> int:
        """Make ErrorDetail hashable for deduplication."""
        return hash((self.severity, self.message, self.path, self.validator))

    def __eq__(self, other: object) -> bool:
        """Compare errors for deduplication."""
        if not isinstance(other, ErrorDetail):
            return False
        return (
            self.severity == other.severity
            and self.message == other.message
            and self.path == other.path
            and self.validator == other.validator
        )


class ValidationResult(BaseModel):
    """Result of a validation check."""

    validator_name: str = Field(description="Name of the validator")
    status: str = Field(
        description="Validation status: passed, failed, error, or skipped"
    )
    errors: list[ErrorDetail] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: list[ErrorDetail] = Field(
        default_factory=list, description="List of validation warnings"
    )
    info: list[ErrorDetail] = Field(
        default_factory=list, description="List of informational messages"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the validation"
    )
    execution_time: float = Field(
        default=0.0, description="Execution time in seconds"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    coverage: float = Field(
        default=1.0,
        description="Percentage of data validated (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

    @property
    def total_issues(self) -> int:
        """Total number of issues (errors + warnings)."""
        return len(self.errors) + len(self.warnings)

    @property
    def is_passed(self) -> bool:
        """Check if validation passed."""
        return self.status == "passed" and len(self.errors) == 0

    def get_all_issues(self) -> list[ErrorDetail]:
        """Get all issues (errors, warnings, and info)."""
        return self.errors + self.warnings + self.info
