"""LangGraph state definition for HVAS-Mini."""

from typing import TypedDict, Optional
from src.models import ValidationResult, ErrorDetail


class ValidationState(TypedDict):
    """State for the validation workflow."""

    input_data: dict
    validation_request: dict
    active_validators: list[str]
    completed_validators: list[str]
    validation_results: list[ValidationResult]
    errors: list[ErrorDetail]
    overall_status: str  # "pending", "in_progress", "completed", "failed"
    confidence_score: Optional[float]
    final_report: Optional[dict]
