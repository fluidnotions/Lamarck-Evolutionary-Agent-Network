"""State management for HVAS-Mini validation workflow.

This module defines the core state schemas used throughout the LangGraph workflow,
including ValidationState, ValidationResult, and ErrorDetail. It implements reducer
patterns for safe concurrent state updates.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict

from langgraph.graph import add_messages


@dataclass
class ErrorDetail:
    """Detailed error information for validation failures.

    Attributes:
        error_type: Classification of the error (e.g., 'api_error', 'validation_error')
        message: Human-readable error message
        validator_name: Name of the validator that encountered the error
        timestamp: When the error occurred
        context: Additional context about the error (stack trace, input data, etc.)
        recoverable: Whether this error can be retried
    """
    error_type: str
    message: str
    validator_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert error detail to dictionary format."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "validator_name": self.validator_name,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recoverable": self.recoverable,
        }


@dataclass
class ValidationResult:
    """Result from a single validation step.

    Attributes:
        validator_name: Name of the validator that produced this result
        status: Status of the validation (passed, failed, warning, error)
        confidence: Confidence score for this validation (0.0 to 1.0)
        findings: List of specific findings or issues identified
        recommendations: Suggested actions or improvements
        timestamp: When the validation completed
        execution_time: Time taken to execute the validation (in seconds)
        metadata: Additional validator-specific metadata
    """
    validator_name: str
    status: Literal["passed", "failed", "warning", "error"]
    confidence: float
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary format."""
        return {
            "validator_name": self.validator_name,
            "status": self.status,
            "confidence": self.confidence,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class ValidationState(TypedDict, total=False):
    """Main state object passed through the LangGraph workflow.

    This state is used by LangGraph reducers to safely merge updates from
    parallel validator executions. Fields support append-only operations
    to prevent data loss during concurrent updates.

    Attributes:
        input_data: Raw input data to be validated
        validation_request: Configuration and requirements for validation
        active_validators: List of validators currently executing
        completed_validators: List of validators that have finished
        validation_results: Results from all completed validators
        errors: List of errors encountered during validation
        overall_status: Current status of the validation workflow
        confidence_score: Aggregated confidence score (0.0 to 1.0)
        final_report: Synthesized final validation report
        metadata: Extensible metadata for custom use cases
    """
    input_data: dict[str, Any]
    validation_request: dict[str, Any]
    active_validators: list[str]
    completed_validators: list[str]
    validation_results: list[ValidationResult]
    errors: list[ErrorDetail]
    overall_status: Literal["pending", "in_progress", "completed", "failed"]
    confidence_score: float
    final_report: Optional[dict[str, Any]]
    metadata: dict[str, Any]


def create_initial_state(
    input_data: dict[str, Any],
    validation_request: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
) -> ValidationState:
    """Create an initial validation state.

    Args:
        input_data: Raw input data to be validated
        validation_request: Configuration for the validation workflow
        metadata: Optional metadata for extensibility

    Returns:
        Initialized ValidationState ready for workflow processing
    """
    return ValidationState(
        input_data=input_data,
        validation_request=validation_request,
        active_validators=[],
        completed_validators=[],
        validation_results=[],
        errors=[],
        overall_status="pending",
        confidence_score=0.0,
        final_report=None,
        metadata=metadata or {},
    )


def add_validation_result(
    state: ValidationState, result: ValidationResult
) -> ValidationState:
    """Add a validation result to the state.

    This function safely updates the state with a new validation result,
    moving the validator from active to completed and appending the result.

    Args:
        state: Current validation state
        result: New validation result to add

    Returns:
        Updated validation state
    """
    new_state = state.copy()

    # Move validator from active to completed
    if result.validator_name in new_state["active_validators"]:
        new_state["active_validators"] = [
            v for v in new_state["active_validators"] if v != result.validator_name
        ]

    if result.validator_name not in new_state["completed_validators"]:
        new_state["completed_validators"] = new_state["completed_validators"] + [
            result.validator_name
        ]

    # Add result
    new_state["validation_results"] = new_state["validation_results"] + [result]

    # Update overall status
    if new_state["active_validators"]:
        new_state["overall_status"] = "in_progress"
    else:
        # All validators completed - determine final status
        has_errors = any(r.status == "error" for r in new_state["validation_results"])
        has_failures = any(r.status == "failed" for r in new_state["validation_results"])

        if has_errors or new_state["errors"]:
            new_state["overall_status"] = "failed"
        elif has_failures:
            new_state["overall_status"] = "completed"
        else:
            new_state["overall_status"] = "completed"

    return new_state


def add_error(state: ValidationState, error: ErrorDetail) -> ValidationState:
    """Add an error to the state.

    Args:
        state: Current validation state
        error: Error detail to add

    Returns:
        Updated validation state
    """
    new_state = state.copy()
    new_state["errors"] = new_state["errors"] + [error]

    # Remove validator from active list if present
    if error.validator_name in new_state["active_validators"]:
        new_state["active_validators"] = [
            v for v in new_state["active_validators"] if v != error.validator_name
        ]

    return new_state


def update_confidence_score(state: ValidationState) -> ValidationState:
    """Recalculate the overall confidence score based on validation results.

    The confidence score is computed as the weighted average of individual
    validator confidence scores, weighted by their status.

    Args:
        state: Current validation state

    Returns:
        Updated validation state with recalculated confidence score
    """
    new_state = state.copy()

    if not new_state["validation_results"]:
        new_state["confidence_score"] = 0.0
        return new_state

    # Weight by status: passed=1.0, warning=0.75, failed=0.5, error=0.0
    weights = {"passed": 1.0, "warning": 0.75, "failed": 0.5, "error": 0.0}

    total_weighted = 0.0
    total_weight = 0.0

    for result in new_state["validation_results"]:
        weight = weights.get(result.status, 0.0)
        total_weighted += result.confidence * weight
        total_weight += weight

    if total_weight > 0:
        new_state["confidence_score"] = total_weighted / total_weight
    else:
        new_state["confidence_score"] = 0.0

    return new_state


def get_active_validators(state: ValidationState) -> list[str]:
    """Get list of currently active validators.

    Args:
        state: Current validation state

    Returns:
        List of active validator names
    """
    return state.get("active_validators", [])


def get_completed_validators(state: ValidationState) -> list[str]:
    """Get list of completed validators.

    Args:
        state: Current validation state

    Returns:
        List of completed validator names
    """
    return state.get("completed_validators", [])


def is_validation_complete(state: ValidationState) -> bool:
    """Check if all validations are complete.

    Args:
        state: Current validation state

    Returns:
        True if no validators are active, False otherwise
    """
    return len(state.get("active_validators", [])) == 0


def get_validation_summary(state: ValidationState) -> dict[str, Any]:
    """Generate a summary of the validation state.

    Args:
        state: Current validation state

    Returns:
        Dictionary containing validation summary statistics
    """
    results = state.get("validation_results", [])
    errors = state.get("errors", [])

    status_counts = {
        "passed": sum(1 for r in results if r.status == "passed"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "warning": sum(1 for r in results if r.status == "warning"),
        "error": sum(1 for r in results if r.status == "error"),
    }

    return {
        "overall_status": state.get("overall_status", "unknown"),
        "confidence_score": state.get("confidence_score", 0.0),
        "total_validators": len(results),
        "status_counts": status_counts,
        "error_count": len(errors),
        "active_validators": len(state.get("active_validators", [])),
        "completed_validators": len(state.get("completed_validators", [])),
    }
