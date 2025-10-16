"""Validation result models."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ValidationStatus(str, Enum):
    """Status of a validation check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"
    ERROR = "error"


class ValidationResult(BaseModel):
    """Result of a validation check."""

    validator_name: str = Field(..., description="Name of the validator that produced this result")
    status: ValidationStatus = Field(..., description="Status of the validation")
    message: str = Field(..., description="Human-readable description of the result")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional details about the validation"
    )
    error_path: Optional[str] = Field(
        None, description="JSON path to the error location (if applicable)"
    )
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0)"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for fixing issues"
    )

    model_config = ConfigDict(use_enum_values=True)


class DomainValidationResult(BaseModel):
    """Aggregated result from a domain validator."""

    domain: str = Field(..., description="Validation domain (e.g., 'schema', 'business_rules')")
    overall_status: ValidationStatus = Field(..., description="Overall status for this domain")
    individual_results: list[ValidationResult] = Field(
        default_factory=list, description="Individual validation results"
    )
    summary: str = Field(..., description="Summary of validation results")
    passed_count: int = Field(default=0, description="Number of passed validations")
    failed_count: int = Field(default=0, description="Number of failed validations")
    warning_count: int = Field(default=0, description="Number of warnings")
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall confidence score"
    )

    model_config = ConfigDict(use_enum_values=True)


class QualityDimension(BaseModel):
    """Data quality dimension score."""

    dimension: str = Field(..., description="Quality dimension name")
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this dimension (0.0 to 1.0)")
    issues: list[str] = Field(default_factory=list, description="Issues found in this dimension")
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for improvement"
    )
