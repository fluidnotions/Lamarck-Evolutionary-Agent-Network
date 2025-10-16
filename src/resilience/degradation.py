"""Graceful degradation strategies for handling validator failures."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Optional

from src.models import ErrorDetail, ValidationResult

logger = logging.getLogger(__name__)


class DegradationStrategy(ABC):
    """Base class for degradation strategies.

    Degradation strategies define how the system should behave
    when validators fail, allowing graceful continuation rather
    than complete failure.
    """

    def __init__(self, name: str):
        """Initialize degradation strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name

    @abstractmethod
    def apply(
        self,
        state: dict[str, Any],
        failed_validators: list[str]
    ) -> dict[str, Any]:
        """Apply degradation strategy to state.

        Args:
            state: Current ValidationState
            failed_validators: List of validator names that failed

        Returns:
            Modified state with degradation applied
        """
        pass

    def _mark_as_degraded(self, state: dict[str, Any], level: int = 1) -> None:
        """Mark state as degraded.

        Args:
            state: State to mark
            level: Degradation level (1=minor, 2=significant, 3=severe)
        """
        current_level = state.get("degradation_level", 0)
        state["degradation_level"] = max(current_level, level)
        state["overall_status"] = "degraded"


class SkipFailedValidators(DegradationStrategy):
    """Continue with successful validators, skip failed ones.

    This is the simplest degradation strategy - just mark failed
    validators as skipped and continue with the workflow.
    """

    def __init__(self):
        """Initialize skip strategy."""
        super().__init__("skip_failed")

    def apply(
        self,
        state: dict[str, Any],
        failed_validators: list[str]
    ) -> dict[str, Any]:
        """Apply skip strategy.

        Args:
            state: Current ValidationState
            failed_validators: List of validator names that failed

        Returns:
            Modified state
        """
        logger.info(f"Skipping {len(failed_validators)} failed validators")

        # Mark validators as skipped in state
        state.setdefault("skipped_validators", []).extend(failed_validators)

        # Add skipped validation results
        for validator_name in failed_validators:
            result = ValidationResult(
                validator_name=validator_name,
                status="skipped",
                confidence=0.0,
                errors=[],
                warnings=[
                    ErrorDetail(
                        path="",
                        message=f"Validator '{validator_name}' skipped due to failure",
                        severity="warning",
                        code="degraded_skipped",
                    )
                ],
                metadata={
                    "degradation_strategy": self.name,
                    "reason": "validator_failed",
                }
            )
            state.setdefault("validation_results", []).append(result)

        # Mark as degraded
        self._mark_as_degraded(state, level=1)

        return state


class UseCachedResults(DegradationStrategy):
    """Use cached results from previous successful runs.

    When a validator fails, try to use results from a previous
    successful run. This works best for validators that don't
    change frequently.
    """

    def __init__(self, cache: Optional[dict[str, Any]] = None, max_age_seconds: float = 3600):
        """Initialize cached results strategy.

        Args:
            cache: Cache storage (validator_name -> {timestamp, result, input_hash})
            max_age_seconds: Maximum age of cached results to use
        """
        super().__init__("use_cached")
        self.cache = cache or {}
        self.max_age_seconds = max_age_seconds

    def apply(
        self,
        state: dict[str, Any],
        failed_validators: list[str]
    ) -> dict[str, Any]:
        """Apply cached results strategy.

        Args:
            state: Current ValidationState
            failed_validators: List of validator names that failed

        Returns:
            Modified state
        """
        input_data = state.get("input_data", {})
        input_hash = self._hash_input(input_data)

        cached_count = 0
        skipped_count = 0

        for validator_name in failed_validators:
            cached_entry = self.cache.get(validator_name)

            if cached_entry and self._is_cache_valid(cached_entry, input_hash):
                # Use cached result
                cached_result = cached_entry["result"]
                logger.info(f"Using cached result for {validator_name}")

                # Create result with cached data
                result = ValidationResult(
                    validator_name=validator_name,
                    status="degraded",
                    confidence=cached_result.confidence * 0.8,  # Reduce confidence
                    errors=cached_result.errors,
                    warnings=cached_result.warnings + [
                        ErrorDetail(
                            path="",
                            message=f"Using cached result from {cached_entry['timestamp']}",
                            severity="info",
                            code="degraded_cached",
                        )
                    ],
                    metadata={
                        "degradation_strategy": self.name,
                        "cached_from": cached_entry["timestamp"],
                        "cache_age_seconds": (
                            datetime.now() - datetime.fromisoformat(cached_entry["timestamp"])
                        ).total_seconds(),
                    }
                )
                state.setdefault("validation_results", []).append(result)
                cached_count += 1
            else:
                # No valid cache, skip
                logger.warning(f"No valid cache for {validator_name}, skipping")
                skip_strategy = SkipFailedValidators()
                state = skip_strategy.apply(state, [validator_name])
                skipped_count += 1

        if cached_count > 0:
            logger.info(f"Used {cached_count} cached results, skipped {skipped_count}")
            self._mark_as_degraded(state, level=1)
        else:
            self._mark_as_degraded(state, level=2)

        return state

    def cache_result(
        self,
        validator_name: str,
        result: ValidationResult,
        input_data: dict[str, Any]
    ) -> None:
        """Cache a successful validation result.

        Args:
            validator_name: Name of the validator
            result: Validation result to cache
            input_data: Input data that was validated
        """
        self.cache[validator_name] = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "input_hash": self._hash_input(input_data),
        }

    def _is_cache_valid(self, cached_entry: dict[str, Any], input_hash: str) -> bool:
        """Check if cached entry is still valid.

        Args:
            cached_entry: Cached entry to check
            input_hash: Hash of current input data

        Returns:
            True if cache is valid
        """
        # Check age
        timestamp = datetime.fromisoformat(cached_entry["timestamp"])
        age = (datetime.now() - timestamp).total_seconds()
        if age > self.max_age_seconds:
            return False

        # Check if input matches (optional)
        # For now, we allow different inputs (could make this stricter)
        return True

    def _hash_input(self, input_data: dict[str, Any]) -> str:
        """Create hash of input data.

        Args:
            input_data: Input data to hash

        Returns:
            Hash string
        """
        import hashlib
        import json

        # Simple hash based on JSON representation
        json_str = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class UseSimplifiedValidation(DegradationStrategy):
    """Fall back to simpler validation methods.

    When complex validators fail, use simpler alternatives that
    are more likely to succeed but provide less thorough validation.
    """

    def __init__(self, fallback_validators: Optional[dict[str, str]] = None):
        """Initialize simplified validation strategy.

        Args:
            fallback_validators: Mapping of validator -> simpler fallback validator
        """
        super().__init__("use_simplified")
        self.fallback_validators = fallback_validators or {}

    def apply(
        self,
        state: dict[str, Any],
        failed_validators: list[str]
    ) -> dict[str, Any]:
        """Apply simplified validation strategy.

        Args:
            state: Current ValidationState
            failed_validators: List of validator names that failed

        Returns:
            Modified state
        """
        for validator_name in failed_validators:
            fallback_name = self.fallback_validators.get(validator_name)

            if fallback_name:
                logger.info(
                    f"Using simplified validator '{fallback_name}' "
                    f"instead of '{validator_name}'"
                )

                # Add the fallback validator to active list
                state.setdefault("active_validators", []).append(fallback_name)

                # Add note about degradation
                result = ValidationResult(
                    validator_name=validator_name,
                    status="degraded",
                    confidence=0.5,
                    errors=[],
                    warnings=[
                        ErrorDetail(
                            path="",
                            message=(
                                f"Using simplified validator '{fallback_name}' "
                                f"due to failure of '{validator_name}'"
                            ),
                            severity="warning",
                            code="degraded_simplified",
                        )
                    ],
                    metadata={
                        "degradation_strategy": self.name,
                        "fallback_validator": fallback_name,
                    }
                )
                state.setdefault("validation_results", []).append(result)
            else:
                # No fallback available, skip
                logger.warning(f"No fallback validator for {validator_name}, skipping")
                skip_strategy = SkipFailedValidators()
                state = skip_strategy.apply(state, [validator_name])

        self._mark_as_degraded(state, level=2)
        return state


class ReturnPartialResults(DegradationStrategy):
    """Return partial results with lower confidence.

    Collect results from successful validators and return them
    with reduced confidence, acknowledging incompleteness.
    """

    def __init__(self, confidence_penalty: float = 0.5):
        """Initialize partial results strategy.

        Args:
            confidence_penalty: Multiplier for confidence scores (0-1)
        """
        super().__init__("partial_results")
        self.confidence_penalty = confidence_penalty

    def apply(
        self,
        state: dict[str, Any],
        failed_validators: list[str]
    ) -> dict[str, Any]:
        """Apply partial results strategy.

        Args:
            state: Current ValidationState
            failed_validators: List of validator names that failed

        Returns:
            Modified state
        """
        logger.info(
            f"Returning partial results with {len(failed_validators)} "
            f"validators failed"
        )

        # Reduce confidence of all existing results
        for result in state.get("validation_results", []):
            result.confidence *= self.confidence_penalty
            result.warnings.append(
                ErrorDetail(
                    path="",
                    message="Confidence reduced due to partial validation",
                    severity="warning",
                    code="degraded_partial",
                )
            )

        # Reduce overall confidence
        current_confidence = state.get("confidence_score", 1.0)
        state["confidence_score"] = current_confidence * self.confidence_penalty

        # Add note about failed validators
        state.setdefault("metadata", {})["failed_validators"] = failed_validators
        state["metadata"]["partial_validation"] = True

        self._mark_as_degraded(state, level=2)
        return state


class CompositeDegradationStrategy(DegradationStrategy):
    """Composite strategy that tries multiple strategies in sequence.

    This strategy attempts multiple degradation approaches in order,
    using the first one that succeeds.
    """

    def __init__(self, strategies: list[DegradationStrategy]):
        """Initialize composite strategy.

        Args:
            strategies: List of strategies to try in order
        """
        super().__init__("composite")
        self.strategies = strategies

    def apply(
        self,
        state: dict[str, Any],
        failed_validators: list[str]
    ) -> dict[str, Any]:
        """Apply composite strategy.

        Args:
            state: Current ValidationState
            failed_validators: List of validator names that failed

        Returns:
            Modified state
        """
        logger.info(f"Trying {len(self.strategies)} degradation strategies")

        for strategy in self.strategies:
            try:
                logger.debug(f"Attempting degradation strategy: {strategy.name}")
                state = strategy.apply(state, failed_validators)
                logger.info(f"Successfully applied strategy: {strategy.name}")
                return state
            except Exception as e:
                logger.warning(
                    f"Degradation strategy {strategy.name} failed: {e}, "
                    "trying next strategy"
                )
                continue

        # All strategies failed, use skip as last resort
        logger.error("All degradation strategies failed, skipping validators")
        skip_strategy = SkipFailedValidators()
        return skip_strategy.apply(state, failed_validators)
