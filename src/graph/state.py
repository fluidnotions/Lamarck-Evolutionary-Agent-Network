"""State management for the validation workflow."""
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator

from src.models.validation_result import ValidationResult, ErrorDetail, AggregatedResult


class ValidationState(TypedDict):
    """State for the validation workflow.

    This state is passed through the LangGraph workflow and tracks
    all validation progress, results, and metadata.
    """

    # Input data and configuration
    input_data: Dict[str, Any]
    validation_request: Dict[str, Any]
    config: Dict[str, Any]

    # Validator tracking
    active_validators: List[str]
    completed_validators: Annotated[List[str], operator.add]
    pending_validators: List[str]

    # Results and errors
    validation_results: Annotated[List[ValidationResult], operator.add]
    errors: Annotated[List[ErrorDetail], operator.add]

    # Status tracking
    overall_status: str  # "pending", "in_progress", "completed", "failed"
    current_step: str
    retry_count: int

    # Aggregation
    confidence_score: float
    final_report: Optional[AggregatedResult]

    # Metadata
    workflow_id: str
    timestamp: datetime
    execution_time_ms: float


def create_initial_state(
    input_data: Dict[str, Any],
    validators: List[str],
    config: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
) -> ValidationState:
    """Create an initial validation state.

    Args:
        input_data: The data to validate
        validators: List of validator names to run
        config: Optional configuration for validators
        workflow_id: Optional workflow identifier

    Returns:
        Initial ValidationState
    """
    import uuid

    return ValidationState(
        input_data=input_data,
        validation_request={"validators": validators, "data": input_data},
        config=config or {},
        active_validators=[],
        completed_validators=[],
        pending_validators=validators.copy(),
        validation_results=[],
        errors=[],
        overall_status="pending",
        current_step="initialized",
        retry_count=0,
        confidence_score=0.0,
        final_report=None,
        workflow_id=workflow_id or str(uuid.uuid4()),
        timestamp=datetime.now(),
        execution_time_ms=0.0,
    )


def update_state_with_result(
    state: ValidationState, result: ValidationResult
) -> Dict[str, Any]:
    """Update state with a validation result.

    Args:
        state: Current state
        result: New validation result to add

    Returns:
        Dictionary of state updates
    """
    updates = {
        "validation_results": [result],
        "completed_validators": [result.validator_name],
    }

    # Add errors if any
    if result.errors:
        updates["errors"] = result.errors

    # Update active validators list
    active = state.get("active_validators", [])
    if result.validator_name in active:
        active = [v for v in active if v != result.validator_name]
        updates["active_validators"] = active

    # Update pending validators
    pending = state.get("pending_validators", [])
    if result.validator_name in pending:
        pending = [v for v in pending if v != result.validator_name]
        updates["pending_validators"] = pending

    return updates


def should_continue_validation(state: ValidationState) -> bool:
    """Determine if validation should continue.

    Args:
        state: Current validation state

    Returns:
        True if there are more validators to run
    """
    pending = state.get("pending_validators", [])
    active = state.get("active_validators", [])
    status = state.get("overall_status", "pending")

    # Continue if there are pending or active validators and status is not failed
    return (len(pending) > 0 or len(active) > 0) and status != "failed"


def calculate_overall_confidence(results: List[ValidationResult]) -> float:
    """Calculate overall confidence score from multiple results.

    Uses weighted average based on validator importance and execution success.

    Args:
        results: List of validation results

    Returns:
        Overall confidence score between 0 and 1
    """
    if not results:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for result in results:
        # Weight by confidence and reduce if there are errors
        weight = 1.0
        if result.errors:
            weight = 0.5  # Reduce weight for failed validators

        weighted_sum += result.confidence * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


def determine_overall_status(results: List[ValidationResult]) -> str:
    """Determine overall validation status.

    Args:
        results: List of validation results

    Returns:
        Overall status: "passed", "failed", or "partial"
    """
    if not results:
        return "pending"

    failed_count = sum(1 for r in results if r.status == "failed")
    error_count = sum(1 for r in results if r.status == "error")
    passed_count = sum(1 for r in results if r.status == "passed")

    total = len(results)

    # If all passed, return passed
    if passed_count == total:
        return "passed"

    # If any failed or errored, check if partial or complete failure
    if failed_count > 0 or error_count > 0:
        if passed_count > 0:
            return "partial"
        else:
            return "failed"

    return "partial"
