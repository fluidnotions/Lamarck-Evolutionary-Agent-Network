"""State management for HVAS-Mini validation workflow."""

from typing import TypedDict, Optional, Any
from datetime import datetime, timezone


class ValidationResult(TypedDict):
    """Result from a single validator."""
    validator_name: str
    status: str  # "passed", "failed", "error"
    messages: list[str]
    confidence_score: float
    timestamp: str
    details: Optional[dict[str, Any]]


class ErrorDetail(TypedDict):
    """Detailed error information."""
    error_type: str
    message: str
    validator: Optional[str]
    timestamp: str
    context: Optional[dict[str, Any]]
    retry_count: int


class ValidationState(TypedDict):
    """
    Central state for the validation workflow.

    This state is passed between all agents in the LangGraph workflow.
    Each agent can read from and write to this state.
    """
    # Input data
    input_data: dict[str, Any]
    validation_request: dict[str, Any]

    # Workflow control
    active_validators: list[str]
    completed_validators: list[str]
    pending_validators: list[str]

    # Results
    validation_results: list[ValidationResult]
    errors: list[ErrorDetail]

    # Overall status
    overall_status: str  # "pending", "in_progress", "completed", "failed"
    confidence_score: float

    # Final output
    final_report: Optional[dict[str, Any]]

    # Metadata
    started_at: str
    completed_at: Optional[str]
    workflow_metadata: dict[str, Any]


def create_initial_state(
    input_data: dict[str, Any],
    validation_request: dict[str, Any]
) -> ValidationState:
    """Create initial validation state."""
    return ValidationState(
        input_data=input_data,
        validation_request=validation_request,
        active_validators=[],
        completed_validators=[],
        pending_validators=[],
        validation_results=[],
        errors=[],
        overall_status="pending",
        confidence_score=0.0,
        final_report=None,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        workflow_metadata={}
    )


def create_validation_result(
    validator_name: str,
    status: str,
    messages: list[str],
    confidence_score: float,
    details: Optional[dict[str, Any]] = None
) -> ValidationResult:
    """Create a validation result."""
    return ValidationResult(
        validator_name=validator_name,
        status=status,
        messages=messages,
        confidence_score=confidence_score,
        timestamp=datetime.now(timezone.utc).isoformat(),
        details=details
    )


def create_error_detail(
    error_type: str,
    message: str,
    validator: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    retry_count: int = 0
) -> ErrorDetail:
    """Create an error detail."""
    return ErrorDetail(
        error_type=error_type,
        message=message,
        validator=validator,
        timestamp=datetime.now(timezone.utc).isoformat(),
        context=context,
        retry_count=retry_count
    )
