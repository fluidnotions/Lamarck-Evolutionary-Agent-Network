"""Result merging and deduplication logic."""

from typing import Any
from collections import defaultdict
from src.models import ValidationResult, ErrorDetail


class ResultMerger:
    """Merges and deduplicates validation results from multiple validators."""

    def __init__(self):
        """Initialize result merger."""
        self.provenance_tracking = True

    def merge_results(
        self, results: list[ValidationResult]
    ) -> list[ValidationResult]:
        """
        Merge validation results from parallel executions.

        This method:
        1. Groups results by validator name
        2. Deduplicates errors within each validator
        3. Preserves result provenance

        Args:
            results: List of validation results to merge

        Returns:
            Merged list of validation results
        """
        if not results:
            return []

        # Group results by validator name
        validator_groups = defaultdict(list)
        for result in results:
            validator_groups[result.validator_name].append(result)

        # Merge results for each validator
        merged_results = []
        for validator_name, validator_results in validator_groups.items():
            merged_result = self._merge_validator_results(
                validator_name, validator_results
            )
            merged_results.append(merged_result)

        return merged_results

    def _merge_validator_results(
        self, validator_name: str, results: list[ValidationResult]
    ) -> ValidationResult:
        """
        Merge multiple results from the same validator.

        Args:
            validator_name: Name of the validator
            results: List of results from this validator

        Returns:
            Single merged ValidationResult
        """
        if len(results) == 1:
            return results[0]

        # Deduplicate errors, warnings, and info messages
        all_errors = []
        all_warnings = []
        all_info = []

        for result in results:
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            all_info.extend(result.info)

        # Deduplicate using set (ErrorDetail has __hash__ and __eq__)
        unique_errors = list(set(all_errors))
        unique_warnings = list(set(all_warnings))
        unique_info = list(set(all_info))

        # Sort by timestamp to maintain order
        unique_errors.sort(key=lambda x: x.timestamp)
        unique_warnings.sort(key=lambda x: x.timestamp)
        unique_info.sort(key=lambda x: x.timestamp)

        # Determine overall status
        statuses = [r.status for r in results]
        if "failed" in statuses:
            overall_status = "failed"
        elif "error" in statuses:
            overall_status = "error"
        elif "skipped" in statuses and all(s == "skipped" for s in statuses):
            overall_status = "skipped"
        else:
            overall_status = "passed"

        # Merge metadata
        merged_metadata = self._merge_metadata([r.metadata for r in results])

        # Calculate total execution time and average coverage
        total_execution_time = sum(r.execution_time for r in results)
        avg_coverage = sum(r.coverage for r in results) / len(results)

        return ValidationResult(
            validator_name=validator_name,
            status=overall_status,
            errors=unique_errors,
            warnings=unique_warnings,
            info=unique_info,
            metadata=merged_metadata,
            execution_time=total_execution_time,
            coverage=avg_coverage,
            timestamp=results[0].timestamp,  # Use first result's timestamp
        )

    def _merge_metadata(self, metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Merge metadata from multiple results.

        Args:
            metadata_list: List of metadata dictionaries

        Returns:
            Merged metadata dictionary
        """
        if not metadata_list:
            return {}

        merged = {}
        for metadata in metadata_list:
            for key, value in metadata.items():
                if key not in merged:
                    merged[key] = value
                elif isinstance(value, (int, float)) and isinstance(
                    merged[key], (int, float)
                ):
                    # Sum numeric values
                    merged[key] = merged[key] + value
                elif isinstance(value, list):
                    # Extend lists
                    if not isinstance(merged[key], list):
                        merged[key] = [merged[key]]
                    merged[key].extend(value)

        return merged

    def deduplicate_across_validators(
        self, results: list[ValidationResult]
    ) -> list[ValidationResult]:
        """
        Deduplicate errors that appear across multiple validators.

        Some errors may be detected by multiple validators. This method
        identifies and marks such duplicates while preserving provenance.

        Args:
            results: List of validation results

        Returns:
            Results with cross-validator duplicates marked
        """
        # Build error signature -> list of (validator, error) pairs
        error_map = defaultdict(list)

        for result in results:
            for error in result.errors:
                # Create a signature based on message and path (not validator)
                signature = (error.severity, error.message, error.path)
                error_map[signature].append((result.validator_name, error))

        # Mark errors that appear in multiple validators
        for signature, error_list in error_map.items():
            if len(error_list) > 1:
                # This error appears in multiple validators
                validators = [v for v, _ in error_list]
                for validator_name, error in error_list:
                    # Add provenance info to context
                    error.context["detected_by"] = validators
                    error.context["is_duplicate"] = True

        return results

    def resolve_conflicts(
        self, results: list[ValidationResult]
    ) -> list[ValidationResult]:
        """
        Resolve conflicting results from different validators.

        Some validators may disagree on the status of the same data.
        This method applies resolution rules to handle conflicts.

        Resolution strategy:
        - If any validator reports a failure, consider it a failure
        - Aggregate all unique errors
        - Keep the most severe status

        Args:
            results: List of validation results

        Returns:
            Results with conflicts resolved
        """
        # For now, we don't modify the results but we could implement
        # more sophisticated conflict resolution logic here, such as:
        # - Voting mechanisms
        # - Confidence-weighted decisions
        # - Domain-specific priority rules

        # Mark potential conflicts in metadata
        validator_names = [r.validator_name for r in results]
        for result in results:
            result.metadata["total_validators"] = len(validator_names)
            result.metadata["concurrent_validators"] = validator_names

        return results
