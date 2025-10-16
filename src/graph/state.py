"""State schema for validation workflow."""

from typing import Any, Optional, TypedDict

from src.models import ErrorDetail, ValidationResult


class ValidationState(TypedDict, total=False):
    """State for the validation workflow.

    This state is passed through the LangGraph workflow and
    accumulates validation results, errors, and metadata.
    """

    # Input data to validate
    input_data: dict[str, Any]

    # Validation request configuration
    validation_request: dict[str, Any]

    # Validators in various stages
    active_validators: list[str]
    completed_validators: list[str]
    failed_validators: list[str]
    skipped_validators: list[str]

    # Results and errors
    validation_results: list[ValidationResult]
    errors: list[ErrorDetail]

    # Overall workflow status
    overall_status: str  # "pending", "in_progress", "completed", "failed", "degraded"
    confidence_score: float

    # Final output
    final_report: Optional[dict[str, Any]]

    # Metadata for error handling and recovery
    metadata: dict[str, Any]
    retry_count: int
    checkpoint_data: Optional[dict[str, Any]]
    degradation_level: int  # 0=none, 1=minor, 2=significant, 3=severe
