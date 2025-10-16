"""LangGraph state definitions for HVAS-Mini."""

from typing import Any, Optional, TypedDict

from src.models.validation_result import DomainValidationResult


class ValidationState(TypedDict, total=False):
    """State for validation workflow in LangGraph.

    This state is passed between agents in the LangGraph workflow.
    """

    # Input data
    input_data: dict[str, Any]
    validation_request: dict[str, Any]

    # Validation tracking
    active_validators: list[str]
    completed_validators: list[str]
    validation_results: list[DomainValidationResult]

    # Error tracking
    errors: list[dict[str, Any]]

    # Overall status
    overall_status: str  # "pending", "in_progress", "completed", "failed"
    confidence_score: float

    # Final report
    final_report: Optional[dict[str, Any]]

    # Metadata
    metadata: dict[str, Any]
