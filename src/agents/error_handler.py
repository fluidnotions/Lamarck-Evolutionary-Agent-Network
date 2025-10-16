"""Error handler agent for analyzing and recovering from errors.

This agent acts as a LangGraph node that analyzes errors in the validation
workflow and determines the best recovery strategy.
"""

import logging
from typing import Any, Optional

from src.models import ErrorDetail, ValidationResult
from src.resilience import (
    CompositeDegradationStrategy,
    ReturnPartialResults,
    SkipFailedValidators,
    UseCachedResults,
    UseSimplifiedValidation,
)
from src.resilience.error_context import ErrorContextCapture
from src.resilience.recovery import WorkflowRecovery

logger = logging.getLogger(__name__)


class ErrorHandlerAgent:
    """Agent that handles errors and determines recovery strategies.

    This agent analyzes errors that occur during validation and uses
    various strategies to recover or degrade gracefully.
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_client: Optional[Any] = None,
        workflow_recovery: Optional[WorkflowRecovery] = None,
        result_cache: Optional[dict[str, Any]] = None,
    ):
        """Initialize error handler agent.

        Args:
            use_llm: Whether to use LLM for error analysis
            llm_client: LLM client for advanced error analysis
            workflow_recovery: Recovery manager for checkpoints
            result_cache: Cache for degradation strategy
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.workflow_recovery = workflow_recovery or WorkflowRecovery()
        self.error_context_capture = ErrorContextCapture()

        # Setup degradation strategies
        self.degradation_strategy = CompositeDegradationStrategy([
            UseCachedResults(cache=result_cache or {}),
            UseSimplifiedValidation(),
            SkipFailedValidators(),
        ])

    def handle_error(
        self,
        state: dict[str, Any],
        error: Optional[Exception] = None,
    ) -> dict[str, Any]:
        """Handle an error in the validation workflow.

        Args:
            state: Current ValidationState
            error: The exception that occurred (if available)

        Returns:
            Updated state with error handling applied
        """
        logger.info("Error handler analyzing workflow state")

        # Capture comprehensive error context if error provided
        if error:
            error_context = self.error_context_capture.capture(
                exception=error,
                state=state,
            )
            state.setdefault("metadata", {})["last_error_context"] = error_context.to_dict()

        # Identify failed validators
        failed_validators = state.get("failed_validators", [])
        if not failed_validators:
            logger.info("No failed validators detected")
            return state

        logger.warning(f"Handling {len(failed_validators)} failed validators")

        # Analyze error severity
        error_severity = self._analyze_error_severity(state)
        state.setdefault("metadata", {})["error_severity"] = error_severity

        # Determine recovery strategy based on severity and context
        if error_severity == "critical":
            # Critical errors - try recovery first
            logger.info("Critical error detected, attempting recovery")
            state, recovered = self.workflow_recovery.recover(state)

            if recovered:
                logger.info("Recovery successful")
                return state
            else:
                logger.warning("Recovery failed, applying degradation")
                return self._apply_degradation(state, failed_validators)

        elif error_severity == "moderate":
            # Moderate errors - apply degradation
            logger.info("Moderate error detected, applying degradation")
            return self._apply_degradation(state, failed_validators)

        else:
            # Minor errors - skip and continue
            logger.info("Minor error detected, skipping failed validators")
            skip_strategy = SkipFailedValidators()
            return skip_strategy.apply(state, failed_validators)

    def _analyze_error_severity(self, state: dict[str, Any]) -> str:
        """Analyze the severity of errors in the state.

        Args:
            state: Current ValidationState

        Returns:
            Severity level: "critical", "moderate", or "minor"
        """
        errors = state.get("errors", [])
        failed_validators = state.get("failed_validators", [])
        completed_validators = state.get("completed_validators", [])

        # Check for critical errors
        has_critical_errors = any(
            e.severity == "critical" for e in errors
        )

        if has_critical_errors:
            return "critical"

        # Calculate failure rate
        total_validators = len(failed_validators) + len(completed_validators)
        if total_validators > 0:
            failure_rate = len(failed_validators) / total_validators

            if failure_rate > 0.5:
                return "critical"
            elif failure_rate > 0.2:
                return "moderate"

        return "minor"

    def _apply_degradation(
        self,
        state: dict[str, Any],
        failed_validators: list[str],
    ) -> dict[str, Any]:
        """Apply degradation strategy to handle failures.

        Args:
            state: Current ValidationState
            failed_validators: List of failed validator names

        Returns:
            Updated state with degradation applied
        """
        try:
            logger.info("Applying composite degradation strategy")
            state = self.degradation_strategy.apply(state, failed_validators)

            # Add degradation metadata
            state.setdefault("metadata", {})
            state["metadata"]["degradation_applied"] = True
            state["metadata"]["degraded_validators"] = failed_validators

            return state

        except Exception as e:
            logger.error(f"Degradation strategy failed: {e}")

            # Last resort - just skip
            skip_strategy = SkipFailedValidators()
            return skip_strategy.apply(state, failed_validators)

    def suggest_fixes(self, state: dict[str, Any]) -> list[str]:
        """Suggest fixes for common errors using LLM (if available).

        Args:
            state: Current ValidationState

        Returns:
            List of suggested fixes
        """
        if not self.use_llm or not self.llm_client:
            return self._suggest_fixes_heuristic(state)

        # TODO: Implement LLM-based fix suggestions
        logger.info("LLM-based fix suggestions not yet implemented")
        return self._suggest_fixes_heuristic(state)

    def _suggest_fixes_heuristic(self, state: dict[str, Any]) -> list[str]:
        """Suggest fixes using heuristics.

        Args:
            state: Current ValidationState

        Returns:
            List of suggested fixes
        """
        suggestions = []
        errors = state.get("errors", [])

        # Analyze common error patterns
        error_codes = [e.code for e in errors]

        if "timeout" in error_codes or "connection_error" in error_codes:
            suggestions.append(
                "Network issues detected. Consider increasing timeout "
                "or checking network connectivity."
            )

        if "validation_error" in error_codes:
            suggestions.append(
                "Validation errors detected. Check input data format "
                "and schema requirements."
            )

        if "resource_exhausted" in error_codes:
            suggestions.append(
                "Resource exhaustion detected. Consider reducing batch size "
                "or increasing system resources."
            )

        # Check retry history
        metadata = state.get("metadata", {})
        retry_count = metadata.get("retry_count", 0)

        if retry_count > 5:
            suggestions.append(
                "Multiple retries detected. Consider investigating root cause "
                "rather than continuing retries."
            )

        return suggestions

    def get_recovery_recommendations(self, state: dict[str, Any]) -> dict[str, Any]:
        """Get recommendations for recovering from current state.

        Args:
            state: Current ValidationState

        Returns:
            Dictionary with recovery recommendations
        """
        recommendations = {
            "severity": self._analyze_error_severity(state),
            "failed_validators": state.get("failed_validators", []),
            "completed_validators": state.get("completed_validators", []),
            "degradation_level": state.get("degradation_level", 0),
            "suggested_actions": [],
        }

        # Add specific recommendations based on state
        if recommendations["severity"] == "critical":
            recommendations["suggested_actions"].append("Attempt full workflow recovery")
            recommendations["suggested_actions"].append("Check system health")
            recommendations["suggested_actions"].append("Review recent changes")

        elif recommendations["severity"] == "moderate":
            recommendations["suggested_actions"].append("Apply degradation strategies")
            recommendations["suggested_actions"].append("Continue with partial results")
            recommendations["suggested_actions"].append("Monitor for additional failures")

        else:
            recommendations["suggested_actions"].append("Skip failed validators")
            recommendations["suggested_actions"].append("Continue workflow normally")

        # Add fix suggestions
        recommendations["fix_suggestions"] = self.suggest_fixes(state)

        return recommendations


def error_handler_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node function for error handling.

    This function can be used directly as a node in a LangGraph workflow.

    Args:
        state: Current ValidationState

    Returns:
        Updated state after error handling
    """
    # Create error handler agent (using default configuration)
    # In production, this would be configured with proper settings
    agent = ErrorHandlerAgent()

    # Check if there's an error to handle
    has_errors = (
        len(state.get("failed_validators", [])) > 0
        or len(state.get("errors", [])) > 0
    )

    if has_errors:
        logger.info("Error detected, invoking error handler")
        state = agent.handle_error(state)

        # Add recommendations to state
        recommendations = agent.get_recovery_recommendations(state)
        state.setdefault("metadata", {})["error_recommendations"] = recommendations

    else:
        logger.debug("No errors detected, error handler skipped")

    return state


def create_error_handler_node(
    use_llm: bool = False,
    llm_client: Optional[Any] = None,
    workflow_recovery: Optional[WorkflowRecovery] = None,
    result_cache: Optional[dict[str, Any]] = None,
):
    """Create a configured error handler node for LangGraph.

    Args:
        use_llm: Whether to use LLM for error analysis
        llm_client: LLM client for advanced error analysis
        workflow_recovery: Recovery manager for checkpoints
        result_cache: Cache for degradation strategy

    Returns:
        Configured error handler node function
    """
    agent = ErrorHandlerAgent(
        use_llm=use_llm,
        llm_client=llm_client,
        workflow_recovery=workflow_recovery,
        result_cache=result_cache,
    )

    def node_function(state: dict[str, Any]) -> dict[str, Any]:
        """Node function with configured agent."""
        has_errors = (
            len(state.get("failed_validators", [])) > 0
            or len(state.get("errors", [])) > 0
        )

        if has_errors:
            state = agent.handle_error(state)
            recommendations = agent.get_recovery_recommendations(state)
            state.setdefault("metadata", {})["error_recommendations"] = recommendations

        return state

    return node_function
