"""Tests for graceful degradation strategies."""

import pytest

from src.models import ValidationResult
from src.resilience.degradation import (
    CompositeDegradationStrategy,
    ReturnPartialResults,
    SkipFailedValidators,
    UseCachedResults,
    UseSimplifiedValidation,
)


class TestSkipFailedValidators:
    """Tests for SkipFailedValidators strategy."""

    def test_skip_failed_validators(self):
        """Test skipping failed validators."""
        strategy = SkipFailedValidators()
        state = {"validation_results": []}
        failed = ["validator1", "validator2"]

        result_state = strategy.apply(state, failed)

        assert "skipped_validators" in result_state
        assert result_state["skipped_validators"] == failed
        assert result_state["degradation_level"] == 1
        assert result_state["overall_status"] == "degraded"

        # Check validation results
        results = result_state["validation_results"]
        assert len(results) == 2
        assert all(r.status == "skipped" for r in results)

    def test_skip_adds_warnings(self):
        """Test that skip strategy adds warning messages."""
        strategy = SkipFailedValidators()
        state = {"validation_results": []}
        failed = ["test_validator"]

        result_state = strategy.apply(state, failed)

        result = result_state["validation_results"][0]
        assert len(result.warnings) > 0
        assert "skipped" in result.warnings[0].message.lower()


class TestUseCachedResults:
    """Tests for UseCachedResults strategy."""

    def test_use_valid_cache(self):
        """Test using valid cached results."""
        # Create cached result
        cached_result = ValidationResult(
            validator_name="test_validator",
            status="passed",
            confidence=1.0,
        )

        cache = {
            "test_validator": {
                "timestamp": "2024-01-01T00:00:00",
                "result": cached_result,
                "input_hash": "abc123",
            }
        }

        strategy = UseCachedResults(cache=cache, max_age_seconds=3600)
        state = {
            "input_data": {"test": "data"},
            "validation_results": [],
        }
        failed = ["test_validator"]

        result_state = strategy.apply(state, failed)

        # Should have used cache
        assert len(result_state["validation_results"]) == 1
        result = result_state["validation_results"][0]
        assert result.status == "degraded"
        assert result.confidence == 0.8  # Reduced from cached

    def test_missing_cache_falls_back(self):
        """Test fallback when cache is missing."""
        strategy = UseCachedResults(cache={})
        state = {"input_data": {}, "validation_results": []}
        failed = ["test_validator"]

        result_state = strategy.apply(state, failed)

        # Should have skipped (fallback)
        assert "skipped_validators" in result_state

    def test_cache_result(self):
        """Test caching a validation result."""
        strategy = UseCachedResults()

        result = ValidationResult(
            validator_name="test",
            status="passed",
            confidence=1.0,
        )
        input_data = {"key": "value"}

        strategy.cache_result("test", result, input_data)

        assert "test" in strategy.cache
        assert strategy.cache["test"]["result"] == result


class TestUseSimplifiedValidation:
    """Tests for UseSimplifiedValidation strategy."""

    def test_use_fallback_validator(self):
        """Test using fallback validator."""
        fallback_map = {"complex_validator": "simple_validator"}
        strategy = UseSimplifiedValidation(fallback_validators=fallback_map)

        state = {
            "active_validators": [],
            "validation_results": [],
        }
        failed = ["complex_validator"]

        result_state = strategy.apply(state, failed)

        # Should have added fallback to active validators
        assert "simple_validator" in result_state["active_validators"]
        assert result_state["degradation_level"] == 2

    def test_no_fallback_available(self):
        """Test when no fallback is available."""
        strategy = UseSimplifiedValidation(fallback_validators={})
        state = {"validation_results": []}
        failed = ["validator_with_no_fallback"]

        result_state = strategy.apply(state, failed)

        # Should have skipped
        assert "skipped_validators" in result_state


class TestReturnPartialResults:
    """Tests for ReturnPartialResults strategy."""

    def test_reduce_confidence(self):
        """Test confidence reduction for partial results."""
        result1 = ValidationResult(
            validator_name="test1",
            status="passed",
            confidence=1.0,
            errors=[],
            warnings=[],
        )

        strategy = ReturnPartialResults(confidence_penalty=0.5)
        state = {
            "validation_results": [result1],
            "confidence_score": 1.0,
        }
        failed = ["test2"]

        result_state = strategy.apply(state, failed)

        # Confidence should be reduced
        assert result_state["validation_results"][0].confidence == 0.5
        assert result_state["confidence_score"] == 0.5
        assert result_state["degradation_level"] == 2

    def test_adds_warning_messages(self):
        """Test that partial results add warnings."""
        result1 = ValidationResult(
            validator_name="test1",
            status="passed",
            confidence=1.0,
            errors=[],
            warnings=[],
        )

        strategy = ReturnPartialResults()
        state = {"validation_results": [result1]}
        failed = ["test2"]

        result_state = strategy.apply(state, failed)

        # Should have added warning
        assert len(result_state["validation_results"][0].warnings) > 0


class TestCompositeDegradationStrategy:
    """Tests for CompositeDegradationStrategy."""

    def test_tries_strategies_in_order(self):
        """Test that composite tries strategies in order."""
        # First strategy will be used
        cache = {
            "test": {
                "timestamp": "2024-01-01T00:00:00",
                "result": ValidationResult(
                    validator_name="test",
                    status="passed",
                    confidence=1.0,
                ),
                "input_hash": "abc",
            }
        }

        composite = CompositeDegradationStrategy([
            UseCachedResults(cache=cache),
            SkipFailedValidators(),
        ])

        state = {
            "input_data": {},
            "validation_results": [],
        }
        failed = ["test"]

        result_state = composite.apply(state, failed)

        # Should have used cache (first strategy)
        assert len(result_state["validation_results"]) == 1
        assert result_state["validation_results"][0].status == "degraded"

    def test_falls_back_on_failure(self):
        """Test fallback when first strategy fails."""

        class FailingStrategy:
            """A strategy that always fails."""

            name = "failing"

            def apply(self, state, failed):
                raise Exception("Strategy failed")

        composite = CompositeDegradationStrategy([
            FailingStrategy(),
            SkipFailedValidators(),
        ])

        state = {"validation_results": []}
        failed = ["test"]

        result_state = composite.apply(state, failed)

        # Should have fallen back to skip
        assert "skipped_validators" in result_state

    def test_all_strategies_fail(self):
        """Test when all strategies fail."""

        class FailingStrategy:
            """A strategy that always fails."""

            name = "failing"

            def apply(self, state, failed):
                raise Exception("Strategy failed")

        composite = CompositeDegradationStrategy([
            FailingStrategy(),
            FailingStrategy(),
        ])

        state = {"validation_results": []}
        failed = ["test"]

        # Should use skip as last resort
        result_state = composite.apply(state, failed)
        assert "skipped_validators" in result_state


class TestDegradationLevels:
    """Tests for degradation level tracking."""

    def test_skip_is_level_1(self):
        """Test skip degradation is level 1 (minor)."""
        strategy = SkipFailedValidators()
        state = {"validation_results": []}

        result = strategy.apply(state, ["test"])

        assert result["degradation_level"] == 1

    def test_simplified_is_level_2(self):
        """Test simplified validation is level 2 (moderate)."""
        strategy = UseSimplifiedValidation({"test": "simple"})
        state = {"validation_results": [], "active_validators": []}

        result = strategy.apply(state, ["test"])

        assert result["degradation_level"] == 2

    def test_partial_is_level_2(self):
        """Test partial results is level 2 (moderate)."""
        strategy = ReturnPartialResults()
        state = {"validation_results": []}

        result = strategy.apply(state, ["test"])

        assert result["degradation_level"] == 2

    def test_degradation_level_increases(self):
        """Test degradation level doesn't decrease."""
        state = {"validation_results": [], "degradation_level": 2}
        strategy = SkipFailedValidators()  # Level 1

        result = strategy.apply(state, ["test"])

        # Should stay at 2, not decrease to 1
        assert result["degradation_level"] == 2
