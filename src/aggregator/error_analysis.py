"""Error analysis and grouping with pattern detection."""

from collections import defaultdict
from typing import Any, Optional
from src.models import ValidationResult, ErrorDetail


class ErrorAnalyzer:
    """
    Analyzes validation errors to identify patterns and provide insights.

    Features:
    - Groups errors by type, path, and severity
    - Identifies error patterns across validators
    - Calculates error statistics
    - Prioritizes errors by impact
    - Generates remediation suggestions
    """

    def __init__(self):
        """Initialize error analyzer."""
        pass

    def analyze(
        self, results: list[ValidationResult], use_llm: bool = False
    ) -> dict[str, Any]:
        """
        Perform comprehensive error analysis.

        Args:
            results: List of validation results
            use_llm: Whether to use LLM for generating suggestions (future enhancement)

        Returns:
            Dictionary containing error analysis results
        """
        all_errors = self._collect_all_errors(results)

        if not all_errors:
            return {
                "total_errors": 0,
                "total_warnings": 0,
                "by_severity": {},
                "by_validator": {},
                "by_path": {},
                "patterns": [],
                "top_errors": [],
                "recommendations": ["No critical issues found. Data quality is good."],
                "statistics": {
                    "unique_errors": 0,
                    "affected_validators": 0,
                    "error_rate": 0.0,
                },
            }

        grouped_by_severity = self._group_by_severity(all_errors)
        grouped_by_validator = self._group_by_validator(results)
        grouped_by_path = self._group_by_path(all_errors)
        patterns = self._detect_patterns(all_errors)
        top_errors = self._get_top_errors(all_errors)
        statistics = self._calculate_statistics(results, all_errors)
        recommendations = self._generate_recommendations(
            all_errors, patterns, use_llm
        )

        return {
            "total_errors": len([e for e in all_errors if e.severity in ["critical", "error"]]),
            "total_warnings": len([e for e in all_errors if e.severity == "warning"]),
            "by_severity": grouped_by_severity,
            "by_validator": grouped_by_validator,
            "by_path": grouped_by_path,
            "patterns": patterns,
            "top_errors": top_errors,
            "recommendations": recommendations,
            "statistics": statistics,
        }

    def _collect_all_errors(
        self, results: list[ValidationResult]
    ) -> list[ErrorDetail]:
        """
        Collect all errors and warnings from results.

        Args:
            results: List of validation results

        Returns:
            List of all errors and warnings
        """
        all_errors = []
        for result in results:
            all_errors.extend(result.errors)
            all_errors.extend(result.warnings)
        return all_errors

    def _group_by_severity(
        self, errors: list[ErrorDetail]
    ) -> dict[str, list[ErrorDetail]]:
        """
        Group errors by severity level.

        Args:
            errors: List of errors

        Returns:
            Dictionary mapping severity to list of errors
        """
        grouped = defaultdict(list)
        for error in errors:
            grouped[error.severity].append(error)

        # Convert to regular dict with counts
        return {
            severity: {
                "count": len(error_list),
                "errors": error_list,
            }
            for severity, error_list in grouped.items()
        }

    def _group_by_validator(
        self, results: list[ValidationResult]
    ) -> dict[str, dict[str, Any]]:
        """
        Group errors by validator.

        Args:
            results: List of validation results

        Returns:
            Dictionary mapping validator name to error summary
        """
        grouped = {}
        for result in results:
            total_issues = len(result.errors) + len(result.warnings)
            grouped[result.validator_name] = {
                "status": result.status,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "total_issues": total_issues,
                "execution_time": result.execution_time,
                "coverage": result.coverage,
            }
        return grouped

    def _group_by_path(
        self, errors: list[ErrorDetail]
    ) -> dict[str, dict[str, Any]]:
        """
        Group errors by path (field location).

        Args:
            errors: List of errors

        Returns:
            Dictionary mapping path to error summary
        """
        grouped = defaultdict(list)
        for error in errors:
            path = error.path or "root"
            grouped[path].append(error)

        # Convert to summary format
        return {
            path: {
                "count": len(error_list),
                "severities": self._count_severities(error_list),
                "validators": list(set(e.validator for e in error_list)),
            }
            for path, error_list in grouped.items()
        }

    def _count_severities(self, errors: list[ErrorDetail]) -> dict[str, int]:
        """Count errors by severity."""
        counts = defaultdict(int)
        for error in errors:
            counts[error.severity] += 1
        return dict(counts)

    def _detect_patterns(self, errors: list[ErrorDetail]) -> list[dict[str, Any]]:
        """
        Detect patterns in errors.

        Patterns include:
        - Multiple errors with similar messages
        - Errors affecting multiple paths with same root
        - Errors reported by multiple validators

        Args:
            errors: List of errors

        Returns:
            List of detected patterns
        """
        patterns = []

        # Pattern 1: Similar messages
        message_groups = defaultdict(list)
        for error in errors:
            # Extract key words from message (simple pattern matching)
            key_words = self._extract_keywords(error.message)
            message_groups[key_words].append(error)

        for key_words, error_list in message_groups.items():
            if len(error_list) >= 2:  # At least 2 similar errors
                max_severity = max((e.severity for e in error_list), key=lambda s: self._severity_rank(s))
                patterns.append({
                    "type": "similar_messages",
                    "description": f"Multiple errors with similar message: {key_words}",
                    "count": len(error_list),
                    "affected_paths": list(set(e.path for e in error_list if e.path)),
                    "severity": max_severity,
                })

        # Pattern 2: Common path prefix (errors in same section of data)
        path_prefixes = defaultdict(list)
        for error in errors:
            if error.path:
                # Get first segment of path
                prefix = error.path.split(".")[0] if "." in error.path else error.path
                path_prefixes[prefix].append(error)

        for prefix, error_list in path_prefixes.items():
            if len(error_list) >= 3:  # At least 3 errors in same section
                patterns.append({
                    "type": "common_path_prefix",
                    "description": f"Multiple errors in section: {prefix}",
                    "count": len(error_list),
                    "path_prefix": prefix,
                    "validators": list(set(e.validator for e in error_list)),
                })

        return patterns

    def _extract_keywords(self, message: str) -> str:
        """Extract key words from error message for pattern matching."""
        # Simple keyword extraction - take first 3 words
        words = message.lower().split()[:3]
        return " ".join(words)

    def _severity_rank(self, severity: str) -> int:
        """Rank severity for comparison."""
        ranks = {"critical": 4, "error": 3, "warning": 2, "info": 1}
        return ranks.get(severity, 0)

    def _get_top_errors(
        self, errors: list[ErrorDetail], limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Get top errors by severity and frequency.

        Args:
            errors: List of errors
            limit: Maximum number of top errors to return

        Returns:
            List of top error summaries
        """
        # Count errors by message and severity
        error_counts = defaultdict(lambda: {"count": 0, "severity": "", "validators": set()})

        for error in errors:
            key = (error.message, error.severity)
            error_counts[key]["count"] += 1
            error_counts[key]["severity"] = error.severity
            error_counts[key]["validators"].add(error.validator)

        # Sort by severity rank then count
        sorted_errors = sorted(
            [
                {
                    "message": msg,
                    "severity": info["severity"],
                    "count": info["count"],
                    "validators": list(info["validators"]),
                }
                for (msg, sev), info in error_counts.items()
            ],
            key=lambda x: (self._severity_rank(x["severity"]), x["count"]),
            reverse=True,
        )

        return sorted_errors[:limit]

    def _calculate_statistics(
        self, results: list[ValidationResult], errors: list[ErrorDetail]
    ) -> dict[str, Any]:
        """
        Calculate error statistics.

        Args:
            results: List of validation results
            errors: List of errors

        Returns:
            Dictionary of statistics
        """
        # Count unique errors (by message + path)
        unique_errors = set((e.message, e.path) for e in errors)

        # Count validators with errors
        validators_with_errors = set(
            r.validator_name for r in results if len(r.errors) > 0
        )

        # Calculate error rate
        total_validators = len(results)
        error_rate = len(validators_with_errors) / total_validators if total_validators > 0 else 0.0

        # Average errors per validator
        avg_errors_per_validator = len(errors) / total_validators if total_validators > 0 else 0.0

        return {
            "unique_errors": len(unique_errors),
            "affected_validators": len(validators_with_errors),
            "error_rate": round(error_rate, 3),
            "avg_errors_per_validator": round(avg_errors_per_validator, 2),
            "total_validators": total_validators,
        }

    def _generate_recommendations(
        self,
        errors: list[ErrorDetail],
        patterns: list[dict[str, Any]],
        use_llm: bool = False,
    ) -> list[str]:
        """
        Generate remediation recommendations.

        Args:
            errors: List of errors
            patterns: Detected patterns
            use_llm: Whether to use LLM for suggestions (future enhancement)

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Recommendation based on severity
        critical_count = len([e for e in errors if e.severity == "critical"])
        error_count = len([e for e in errors if e.severity == "error"])
        warning_count = len([e for e in errors if e.severity == "warning"])

        if critical_count > 0:
            recommendations.append(
                f"Address {critical_count} critical error(s) immediately - these prevent validation from passing."
            )

        if error_count > 0:
            recommendations.append(
                f"Fix {error_count} error(s) to improve data quality and validation results."
            )

        if warning_count > 5:
            recommendations.append(
                f"Review {warning_count} warning(s) - while not critical, they indicate potential issues."
            )

        # Recommendations based on patterns
        for pattern in patterns:
            if pattern["type"] == "similar_messages":
                recommendations.append(
                    f"Pattern detected: {pattern['description']} - "
                    f"consider fixing the root cause to resolve multiple errors at once."
                )
            elif pattern["type"] == "common_path_prefix":
                recommendations.append(
                    f"Multiple errors in {pattern['path_prefix']} section - "
                    f"focus remediation efforts on this area."
                )

        # Future: Use LLM to generate more sophisticated recommendations
        if use_llm:
            recommendations.append(
                "Note: LLM-based recommendations are not yet implemented."
            )

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append("No critical issues found. Data quality is good.")

        return recommendations
