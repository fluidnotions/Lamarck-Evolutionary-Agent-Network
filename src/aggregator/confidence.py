"""Multi-factor confidence scoring system."""

from typing import Optional
from src.models import ValidationResult, ErrorDetail


class ConfidenceCalculator:
    """
    Calculates confidence scores for validation results using multiple factors.

    The confidence score is a value between 0.0 and 1.0 that represents
    the overall confidence in the validation results, considering:
    - Pass rate: Percentage of validators that passed
    - Severity: Impact of errors (critical errors reduce confidence more)
    - Coverage: Percentage of data that was validated
    - Reliability: Historical success rate of validators
    """

    # Default severity weights (higher = more severe = lower confidence)
    DEFAULT_SEVERITY_WEIGHTS = {
        "critical": 1.0,
        "error": 0.6,
        "warning": 0.3,
        "info": 0.1,
    }

    # Default factor weights for overall confidence
    DEFAULT_FACTOR_WEIGHTS = {
        "pass_rate": 0.4,
        "severity": 0.3,
        "coverage": 0.2,
        "reliability": 0.1,
    }

    def __init__(
        self,
        factor_weights: Optional[dict[str, float]] = None,
        severity_weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize confidence calculator.

        Args:
            factor_weights: Custom weights for confidence factors
            severity_weights: Custom weights for error severities
        """
        self.factor_weights = factor_weights or self.DEFAULT_FACTOR_WEIGHTS
        self.severity_weights = severity_weights or self.DEFAULT_SEVERITY_WEIGHTS

        # Validate weights sum to 1.0
        total_weight = sum(self.factor_weights.values())
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(
                f"Factor weights must sum to 1.0, got {total_weight}"
            )

    def calculate(
        self,
        results: list[ValidationResult],
        validator_reliability: Optional[dict[str, float]] = None,
    ) -> float:
        """
        Calculate overall confidence score for validation results.

        Args:
            results: List of validation results
            validator_reliability: Optional dict mapping validator names to
                                  reliability scores (0.0 to 1.0)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results:
            return 0.0

        # Calculate individual factor scores
        pass_rate_score = self._calculate_pass_rate(results)
        severity_score = self._calculate_severity_score(results)
        coverage_score = self._calculate_coverage_score(results)
        reliability_score = self._calculate_reliability_score(
            results, validator_reliability
        )

        # Weighted combination
        confidence = (
            pass_rate_score * self.factor_weights["pass_rate"]
            + severity_score * self.factor_weights["severity"]
            + coverage_score * self.factor_weights["coverage"]
            + reliability_score * self.factor_weights["reliability"]
        )

        return round(confidence, 3)

    def calculate_with_breakdown(
        self,
        results: list[ValidationResult],
        validator_reliability: Optional[dict[str, float]] = None,
    ) -> dict:
        """
        Calculate confidence score with detailed breakdown.

        Args:
            results: List of validation results
            validator_reliability: Optional dict of validator reliability scores

        Returns:
            Dictionary with confidence score and breakdown of each factor
        """
        if not results:
            return {
                "confidence": 0.0,
                "breakdown": {
                    "pass_rate": 0.0,
                    "severity": 0.0,
                    "coverage": 0.0,
                    "reliability": 0.0,
                },
                "factor_weights": self.factor_weights,
                "total_validators": 0,
                "passed_validators": 0,
                "failed_validators": 0,
            }

        # Calculate individual scores
        pass_rate_score = self._calculate_pass_rate(results)
        severity_score = self._calculate_severity_score(results)
        coverage_score = self._calculate_coverage_score(results)
        reliability_score = self._calculate_reliability_score(
            results, validator_reliability
        )

        # Calculate overall confidence
        confidence = (
            pass_rate_score * self.factor_weights["pass_rate"]
            + severity_score * self.factor_weights["severity"]
            + coverage_score * self.factor_weights["coverage"]
            + reliability_score * self.factor_weights["reliability"]
        )

        # Count validators by status
        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")

        return {
            "confidence": round(confidence, 3),
            "breakdown": {
                "pass_rate": round(pass_rate_score, 3),
                "severity": round(severity_score, 3),
                "coverage": round(coverage_score, 3),
                "reliability": round(reliability_score, 3),
            },
            "factor_weights": self.factor_weights,
            "total_validators": len(results),
            "passed_validators": passed,
            "failed_validators": failed,
        }

    def _calculate_pass_rate(self, results: list[ValidationResult]) -> float:
        """
        Calculate pass rate score (0.0 to 1.0).

        Pass rate is the percentage of validators that passed.

        Args:
            results: List of validation results

        Returns:
            Pass rate score
        """
        if not results:
            return 0.0

        passed_count = sum(1 for r in results if r.status == "passed")
        return passed_count / len(results)

    def _calculate_severity_score(self, results: list[ValidationResult]) -> float:
        """
        Calculate severity score (0.0 to 1.0) based on error distribution.

        A score of 1.0 means no errors.
        A score of 0.0 means all errors are critical.

        Args:
            results: List of validation results

        Returns:
            Severity score
        """
        total_errors = 0
        weighted_errors = 0.0

        for result in results:
            for error in result.errors:
                total_errors += 1
                weighted_errors += self.severity_weights.get(error.severity, 0.5)

            # Also consider warnings but with lower weight
            for warning in result.warnings:
                total_errors += 1
                weighted_errors += self.severity_weights.get(warning.severity, 0.3)

        if total_errors == 0:
            return 1.0

        # Normalize: fewer weighted errors = higher score
        # Max weighted errors = all critical
        max_weighted = total_errors * 1.0
        score = 1.0 - (weighted_errors / max_weighted)

        return max(0.0, min(1.0, score))

    def _calculate_coverage_score(self, results: list[ValidationResult]) -> float:
        """
        Calculate coverage score (0.0 to 1.0).

        Coverage is the average percentage of data that was validated.

        Args:
            results: List of validation results

        Returns:
            Coverage score
        """
        if not results:
            return 0.0

        total_coverage = sum(r.coverage for r in results)
        return total_coverage / len(results)

    def _calculate_reliability_score(
        self,
        results: list[ValidationResult],
        validator_reliability: Optional[dict[str, float]] = None,
    ) -> float:
        """
        Calculate reliability score (0.0 to 1.0).

        Reliability is based on the historical success rate of validators.
        If no reliability data is provided, assumes all validators are reliable.

        Args:
            results: List of validation results
            validator_reliability: Optional dict of validator reliability scores

        Returns:
            Reliability score
        """
        if not results:
            return 0.0

        if not validator_reliability:
            # If no reliability data, assume all validators are reliable
            return 1.0

        total_reliability = 0.0
        for result in results:
            reliability = validator_reliability.get(result.validator_name, 1.0)
            total_reliability += reliability

        return total_reliability / len(results)

    def get_confidence_level(self, confidence: float) -> str:
        """
        Convert confidence score to a human-readable level.

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            Confidence level string
        """
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.75:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.25:
            return "low"
        else:
            return "very low"

    def get_recommendation(self, confidence: float) -> str:
        """
        Get recommendation based on confidence score.

        Args:
            confidence: Confidence score (0.0 to 1.0)

        Returns:
            Recommendation string
        """
        if confidence >= 0.9:
            return "Data quality is excellent. Safe to proceed."
        elif confidence >= 0.75:
            return "Data quality is good. Review warnings before proceeding."
        elif confidence >= 0.5:
            return "Data quality is acceptable but has issues. Review errors carefully."
        elif confidence >= 0.25:
            return "Data quality is poor. Address critical errors before proceeding."
        else:
            return "Data quality is very poor. Significant remediation required."
