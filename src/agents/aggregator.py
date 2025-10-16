"""Aggregator agent that collects and synthesizes validation results."""

from typing import Optional, Any
from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.aggregator.result_merger import ResultMerger
from src.aggregator.confidence import ConfidenceCalculator
from src.aggregator.error_analysis import ErrorAnalyzer
from src.aggregator.report_generator import ReportGenerator
from src.aggregator.visualization import Visualizer


class AggregatorAgent(BaseAgent):
    """
    Aggregates validation results and generates comprehensive reports.

    This agent:
    1. Collects validation results from state
    2. Merges and deduplicates results
    3. Calculates confidence scores
    4. Analyzes errors and patterns
    5. Generates comprehensive reports
    6. Creates visualizations
    7. Updates state with final results
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        report_format: str = "markdown",
        generate_visualizations: bool = True,
        confidence_weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize aggregator agent.

        Args:
            llm: Optional LLM for enhanced recommendations (future use)
            report_format: Default report format (json, markdown, html, pdf)
            generate_visualizations: Whether to generate visualizations
            confidence_weights: Custom confidence score weights
        """
        super().__init__(
            name="aggregator",
            description="Aggregates validation results and generates reports",
        )
        self.llm = llm
        self.report_format = report_format
        self.generate_visualizations = generate_visualizations

        # Initialize aggregator components
        self.result_merger = ResultMerger()
        self.confidence_calculator = ConfidenceCalculator(
            factor_weights=confidence_weights
        )
        self.error_analyzer = ErrorAnalyzer()
        self.report_generator = ReportGenerator()
        self.visualizer = Visualizer(backend="data")  # Export data format

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Execute aggregation and report generation.

        Args:
            state: Current validation state

        Returns:
            Updated validation state with aggregated results and report
        """
        results = state.get("validation_results", [])

        if not results:
            # No results to aggregate
            return self._handle_empty_results(state)

        # Step 1: Merge and deduplicate results
        merged_results = self.result_merger.merge_results(results)

        # Step 2: Deduplicate across validators
        merged_results = self.result_merger.deduplicate_across_validators(
            merged_results
        )

        # Step 3: Resolve conflicts
        merged_results = self.result_merger.resolve_conflicts(merged_results)

        # Step 4: Calculate confidence score with breakdown
        confidence_breakdown = self.confidence_calculator.calculate_with_breakdown(
            merged_results
        )
        confidence_score = confidence_breakdown["confidence"]

        # Step 5: Analyze errors
        error_analysis = self.error_analyzer.analyze(
            merged_results, use_llm=self.llm is not None
        )

        # Step 6: Generate visualizations (if enabled)
        visualizations = None
        if self.generate_visualizations:
            visualizations = self.visualizer.generate_all_visualizations(
                merged_results, confidence_score, error_analysis
            )

        # Step 7: Generate report
        report = self.report_generator.generate(
            results=merged_results,
            confidence=confidence_score,
            error_analysis=error_analysis,
            format=self.report_format,
            confidence_breakdown=confidence_breakdown,
        )

        # Step 8: Determine overall status
        overall_status = self._determine_overall_status(merged_results)

        # Step 9: Update state
        new_state = state.copy()
        new_state["validation_results"] = merged_results
        new_state["overall_status"] = overall_status
        new_state["confidence_score"] = confidence_score
        new_state["final_report"] = {
            "content": report["content"],
            "format": report["format"],
            "metadata": report["metadata"],
            "confidence_breakdown": confidence_breakdown,
            "error_analysis": error_analysis,
            "visualizations": visualizations,
        }

        # Add aggregation metadata
        if "completed_validators" not in new_state:
            new_state["completed_validators"] = []

        new_state["completed_validators"].append(self.name)

        return new_state

    def _handle_empty_results(self, state: ValidationState) -> ValidationState:
        """
        Handle case where there are no validation results.

        Args:
            state: Current validation state

        Returns:
            Updated state with empty report
        """
        new_state = state.copy()
        new_state["overall_status"] = "error"
        new_state["confidence_score"] = 0.0
        new_state["final_report"] = {
            "content": "No validation results to aggregate.",
            "format": "text",
            "metadata": {
                "validator_count": 0,
                "confidence_score": 0.0,
                "total_errors": 0,
                "total_warnings": 0,
            },
            "error_analysis": {},
            "visualizations": None,
        }
        return new_state

    def _determine_overall_status(self, results: list) -> str:
        """
        Determine overall validation status from merged results.

        Args:
            results: Merged validation results

        Returns:
            Overall status string
        """
        if not results:
            return "error"

        # Check if any validator failed
        failed_count = sum(1 for r in results if r.status == "failed")
        error_count = sum(1 for r in results if r.status == "error")

        # Count total errors
        total_errors = sum(len(r.errors) for r in results)

        if failed_count > 0 or total_errors > 0:
            return "failed"
        elif error_count > 0:
            return "error"
        else:
            return "completed"

    def generate_summary_report(
        self, state: ValidationState
    ) -> dict[str, Any]:
        """
        Generate a quick summary report without full details.

        Args:
            state: Validation state with results

        Returns:
            Summary report dictionary
        """
        results = state.get("validation_results", [])

        if not results:
            return {
                "status": "no_results",
                "message": "No validation results available",
            }

        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)

        confidence = self.confidence_calculator.calculate(results)
        confidence_level = self.confidence_calculator.get_confidence_level(confidence)
        recommendation = self.confidence_calculator.get_recommendation(confidence)

        return {
            "status": state.get("overall_status", "unknown"),
            "confidence": confidence,
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "summary": {
                "total_validators": len(results),
                "passed": passed,
                "failed": failed,
                "errors": total_errors,
                "warnings": total_warnings,
            },
        }

    def export_results_for_analysis(
        self, state: ValidationState, format: str = "json"
    ) -> str:
        """
        Export validation results in a format suitable for external analysis.

        Args:
            state: Validation state
            format: Export format (json, csv, etc.)

        Returns:
            Exported data as string
        """
        results = state.get("validation_results", [])

        if format == "json":
            import json

            export_data = {
                "results": [
                    {
                        "validator": r.validator_name,
                        "status": r.status,
                        "errors": [
                            {
                                "severity": e.severity,
                                "message": e.message,
                                "path": e.path,
                            }
                            for e in r.errors
                        ],
                        "warnings": [
                            {
                                "severity": w.severity,
                                "message": w.message,
                                "path": w.path,
                            }
                            for w in r.warnings
                        ],
                        "execution_time": r.execution_time,
                        "coverage": r.coverage,
                    }
                    for r in results
                ],
                "metadata": {
                    "overall_status": state.get("overall_status"),
                    "confidence_score": state.get("confidence_score"),
                },
            }
            return json.dumps(export_data, indent=2, default=str)

        return "Unsupported export format"

    def get_failed_validators(self, state: ValidationState) -> list[str]:
        """
        Get list of validators that failed.

        Args:
            state: Validation state

        Returns:
            List of failed validator names
        """
        results = state.get("validation_results", [])
        return [r.validator_name for r in results if r.status == "failed"]

    def get_critical_errors(self, state: ValidationState) -> list[dict[str, Any]]:
        """
        Get all critical errors from validation results.

        Args:
            state: Validation state

        Returns:
            List of critical error details
        """
        results = state.get("validation_results", [])
        critical_errors = []

        for result in results:
            for error in result.errors:
                if error.severity == "critical":
                    critical_errors.append({
                        "validator": result.validator_name,
                        "message": error.message,
                        "path": error.path,
                        "context": error.context,
                        "timestamp": error.timestamp,
                    })

        return critical_errors
