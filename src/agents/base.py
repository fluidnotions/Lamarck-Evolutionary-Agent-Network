"""Base agent class for HVAS-Mini."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel

from src.graph.state import ValidationState
from src.models.validation_result import (
    DomainValidationResult,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents in HVAS-Mini.

    This class provides common functionality for domain validator agents,
    including result aggregation, state management, and error handling.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm: Optional[BaseChatModel] = None,
    ) -> None:
        """Initialize the base agent.

        Args:
            name: Unique name for this agent
            description: Human-readable description of what this agent does
            llm: Optional language model for LLM-powered features
        """
        self.name = name
        self.description = description
        self.llm = llm
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def execute(self, state: ValidationState) -> ValidationState:
        """Execute the agent's validation logic.

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        pass

    def _aggregate_results(
        self,
        results: list[ValidationResult],
        domain: str,
    ) -> DomainValidationResult:
        """Aggregate individual validation results into a domain result.

        Args:
            results: List of individual validation results
            domain: Domain name (e.g., 'schema', 'business_rules')

        Returns:
            Aggregated domain validation result
        """
        if not results:
            return DomainValidationResult(
                domain=domain,
                overall_status=ValidationStatus.SKIPPED,
                summary=f"No {domain} validations were performed",
            )

        # Count results by status
        passed_count = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed_count = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        warning_count = sum(1 for r in results if r.status == ValidationStatus.WARNING)
        error_count = sum(1 for r in results if r.status == ValidationStatus.ERROR)

        # Determine overall status
        if error_count > 0 or failed_count > 0:
            overall_status = ValidationStatus.FAILED
        elif warning_count > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED

        # Calculate average confidence score
        confidence_scores = [r.confidence_score for r in results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Generate summary
        total = len(results)
        summary = (
            f"{domain.replace('_', ' ').title()} validation: "
            f"{passed_count}/{total} passed, "
            f"{failed_count} failed, "
            f"{warning_count} warnings"
        )

        return DomainValidationResult(
            domain=domain,
            overall_status=overall_status,
            individual_results=results,
            summary=summary,
            passed_count=passed_count,
            failed_count=failed_count,
            warning_count=warning_count,
            confidence_score=avg_confidence,
        )

    def _execute_parallel(
        self,
        validators: list[Callable[[Any], ValidationResult]],
        data: Any,
    ) -> list[ValidationResult]:
        """Execute multiple validators in parallel (simulated).

        In a production system, this could use asyncio or threading.
        For now, we execute sequentially but maintain the interface.

        Args:
            validators: List of validator functions
            data: Data to validate

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        for validator in validators:
            try:
                result = validator(data)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Validator {validator.__name__} failed: {e}")
                results.append(
                    ValidationResult(
                        validator_name=validator.__name__,
                        status=ValidationStatus.ERROR,
                        message=f"Validation failed with error: {str(e)}",
                        details={"error": str(e), "error_type": type(e).__name__},
                    )
                )

        return results

    def _update_state(
        self,
        state: ValidationState,
        domain_result: DomainValidationResult,
    ) -> ValidationState:
        """Update the validation state with new results.

        Args:
            state: Current state
            domain_result: Domain validation result to add

        Returns:
            Updated state
        """
        # Create a copy of the state
        new_state = state.copy()

        # Add domain result to validation results
        if "validation_results" not in new_state:
            new_state["validation_results"] = []
        new_state["validation_results"].append(domain_result)

        # Add this validator to completed validators
        if "completed_validators" not in new_state:
            new_state["completed_validators"] = []
        if self.name not in new_state["completed_validators"]:
            new_state["completed_validators"].append(self.name)

        # Remove from active validators if present
        if "active_validators" in new_state and self.name in new_state["active_validators"]:
            new_state["active_validators"] = [
                v for v in new_state["active_validators"] if v != self.name
            ]

        # Update overall status
        if domain_result.overall_status == ValidationStatus.FAILED:
            new_state["overall_status"] = "failed"
        elif "overall_status" not in new_state or new_state["overall_status"] == "pending":
            new_state["overall_status"] = "in_progress"

        return new_state

    def _get_llm_explanation(self, prompt: str) -> str:
        """Get an explanation from the LLM.

        Args:
            prompt: Prompt for the LLM

        Returns:
            LLM-generated explanation
        """
        if not self.llm:
            return "LLM explanation not available (no LLM configured)"

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                return str(response.content)
            return str(response)
        except Exception as e:
            self.logger.error(f"Failed to get LLM explanation: {e}")
            return f"Failed to generate explanation: {str(e)}"
