"""Unit tests for result merger."""

import pytest
from src.aggregator.result_merger import ResultMerger
from src.models import ValidationResult, ErrorDetail


class TestResultMerger:
    """Test cases for ResultMerger."""

    def test_merge_empty_results(self):
        """Test merging empty results list."""
        merger = ResultMerger()
        results = merger.merge_results([])
        assert results == []

    def test_merge_single_result(self, passing_result):
        """Test merging single result."""
        merger = ResultMerger()
        results = merger.merge_results([passing_result])
        assert len(results) == 1
        assert results[0].validator_name == "passing_validator"

    def test_merge_different_validators(self, passing_result, failing_result):
        """Test merging results from different validators."""
        merger = ResultMerger()
        results = merger.merge_results([passing_result, failing_result])
        assert len(results) == 2

    def test_merge_same_validator_deduplicates_errors(self):
        """Test that duplicate errors from same validator are removed."""
        merger = ResultMerger()

        # Create two results with duplicate errors
        error1 = ErrorDetail(
            severity="error",
            message="Duplicate error",
            path="field",
            validator="test_validator",
        )
        error2 = ErrorDetail(
            severity="error",
            message="Duplicate error",
            path="field",
            validator="test_validator",
        )

        result1 = ValidationResult(
            validator_name="test_validator",
            status="failed",
            errors=[error1],
            execution_time=0.5,
        )
        result2 = ValidationResult(
            validator_name="test_validator",
            status="failed",
            errors=[error2],
            execution_time=0.5,
        )

        merged = merger.merge_results([result1, result2])
        assert len(merged) == 1
        # Should have only one error after deduplication
        assert len(merged[0].errors) == 1

    def test_merge_calculates_total_execution_time(self):
        """Test that execution times are summed."""
        merger = ResultMerger()

        result1 = ValidationResult(
            validator_name="test_validator",
            status="passed",
            execution_time=0.5,
        )
        result2 = ValidationResult(
            validator_name="test_validator",
            status="passed",
            execution_time=1.0,
        )

        merged = merger.merge_results([result1, result2])
        assert merged[0].execution_time == 1.5

    def test_merge_determines_worst_status(self):
        """Test that the worst status is preserved."""
        merger = ResultMerger()

        result1 = ValidationResult(
            validator_name="test_validator",
            status="passed",
            execution_time=0.5,
        )
        result2 = ValidationResult(
            validator_name="test_validator",
            status="failed",
            errors=[ErrorDetail(
                severity="error",
                message="Error",
                validator="test_validator",
            )],
            execution_time=0.5,
        )

        merged = merger.merge_results([result1, result2])
        assert merged[0].status == "failed"

    def test_deduplicate_across_validators(self):
        """Test deduplication across different validators."""
        merger = ResultMerger()

        error1 = ErrorDetail(
            severity="error",
            message="Same error",
            path="field",
            validator="validator1",
        )
        error2 = ErrorDetail(
            severity="error",
            message="Same error",
            path="field",
            validator="validator2",
        )

        result1 = ValidationResult(
            validator_name="validator1",
            status="failed",
            errors=[error1],
        )
        result2 = ValidationResult(
            validator_name="validator2",
            status="failed",
            errors=[error2],
        )

        results = merger.deduplicate_across_validators([result1, result2])

        # Both errors should have is_duplicate marked
        assert results[0].errors[0].context.get("is_duplicate") == True
        assert results[1].errors[0].context.get("is_duplicate") == True

        # Both should list both validators
        detected_by = results[0].errors[0].context.get("detected_by", [])
        assert "validator1" in detected_by
        assert "validator2" in detected_by

    def test_resolve_conflicts_adds_metadata(self, mixed_results):
        """Test that conflict resolution adds metadata."""
        merger = ResultMerger()
        results = merger.resolve_conflicts(mixed_results)

        for result in results:
            assert "total_validators" in result.metadata
            assert "concurrent_validators" in result.metadata
            assert result.metadata["total_validators"] == 3

    def test_merge_metadata(self):
        """Test metadata merging logic."""
        merger = ResultMerger()

        result1 = ValidationResult(
            validator_name="test",
            status="passed",
            metadata={"count": 10, "items": ["a"]},
        )
        result2 = ValidationResult(
            validator_name="test",
            status="passed",
            metadata={"count": 20, "items": ["b"]},
        )

        merged = merger.merge_results([result1, result2])
        # Numeric values should be summed
        assert merged[0].metadata["count"] == 30
        # Lists should be extended
        assert "a" in merged[0].metadata["items"]
        assert "b" in merged[0].metadata["items"]
