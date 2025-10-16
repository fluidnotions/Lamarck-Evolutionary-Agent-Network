"""Validation result model."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .error_detail import ErrorDetail


class ValidationResult(BaseModel):
    """Result from a validation check."""

    validator_name: str = Field(description="Name of the validator that produced this result")
    status: str = Field(
        description="Status: passed, failed, skipped, degraded"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this validation"
    )
    errors: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of errors encountered"
    )
    warnings: list[ErrorDetail] = Field(
        default_factory=list,
        description="List of warnings"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the validation"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the validation completed"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="How long the validation took in milliseconds"
    )

    @property
    def is_success(self) -> bool:
        """Check if validation was successful."""
        return self.status in ("passed", "skipped", "degraded") and not self.has_critical_errors

    @property
    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return any(e.severity == "critical" for e in self.errors)

    def add_error(
        self,
        path: str,
        message: str,
        severity: str = "error",
        code: str = "validation_error"
    ) -> None:
        """Add an error to this result."""
        error = ErrorDetail(
            path=path,
            message=message,
            severity=severity,
            code=code
        )
        if severity in ("error", "critical"):
            self.errors.append(error)
        else:
            self.warnings.append(error)
