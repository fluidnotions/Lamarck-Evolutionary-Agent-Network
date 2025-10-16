"""Aggregator agent for collecting and synthesizing validation results."""
from typing import Any, Dict
import logging

from src.agents.base import BaseAgent
from src.graph.state import ValidationState, calculate_overall_confidence, determine_overall_status
from src.models.validation_result import AggregatedResult

logger = logging.getLogger(__name__)


class AggregatorAgent(BaseAgent):
    """Aggregator agent that synthesizes validation results."""

    def __init__(self, **kwargs: Any):
        """Initialize aggregator agent."""
        super().__init__(name="aggregator", **kwargs)

    def process(self, state: ValidationState) -> Dict[str, Any]:
        """Aggregate validation results into final report.

        Args:
            state: Current validation state

        Returns:
            State updates with aggregated result
        """
        logger.info("Aggregator synthesizing results")

        results = state.get("validation_results", [])

        if not results:
            logger.warning("No validation results to aggregate")
            return {
                "overall_status": "failed",
                "confidence_score": 0.0,
                "current_step": "completed",
            }

        # Calculate overall metrics
        overall_status = determine_overall_status(results)
        confidence_score = calculate_overall_confidence(results)

        # Calculate totals
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        total_execution_time = sum(r.execution_time_ms for r in results)

        # Generate summary
        summary = self._generate_summary(results, overall_status, confidence_score)

        # Create aggregated result
        aggregated = AggregatedResult(
            overall_status=overall_status,
            confidence_score=confidence_score,
            validation_results=results,
            summary=summary,
            total_errors=total_errors,
            total_warnings=total_warnings,
            execution_time_ms=total_execution_time,
        )

        logger.info(
            f"Aggregation complete: status={overall_status}, confidence={confidence_score:.2%}"
        )

        return {
            "overall_status": overall_status,
            "confidence_score": confidence_score,
            "final_report": aggregated,
            "current_step": "completed",
            "execution_time_ms": total_execution_time,
        }

    def _generate_summary(
        self, results: list, overall_status: str, confidence_score: float
    ) -> str:
        """Generate human-readable summary.

        Args:
            results: Validation results
            overall_status: Overall validation status
            confidence_score: Confidence score

        Returns:
            Summary string
        """
        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        total = len(results)

        summary = f"Validation {overall_status}: {passed}/{total} validators passed"

        if failed > 0:
            summary += f", {failed} failed"

        summary += f". Overall confidence: {confidence_score:.1%}"

        return summary
