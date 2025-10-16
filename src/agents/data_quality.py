"""Data quality validator agent."""
from typing import Any, Dict, List
import logging

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.validators import quality_checks
from src.models.validation_result import ErrorDetail

logger = logging.getLogger(__name__)


class DataQualityAgent(BaseAgent):
    """Agent that validates data quality."""

    def __init__(self, **kwargs: Any):
        """Initialize data quality agent."""
        super().__init__(name="data_quality", **kwargs)

    def process(self, state: ValidationState) -> Dict[str, Any]:
        """Validate data quality.

        Args:
            state: Current validation state

        Returns:
            State updates with validation result
        """
        logger.info("Data quality validator processing")

        data = state["input_data"]
        config = state.get("config", {})
        quality_config = config.get("data_quality", {})

        all_errors: List[ErrorDetail] = []

        # Completeness check
        required_fields = quality_config.get("required_fields", [])
        if required_fields:
            errors = quality_checks.check_completeness(data, required_fields)
            all_errors.extend(errors)

        # Type check
        type_specs = quality_config.get("types", {})
        if type_specs:
            type_map = {}
            for field, type_name in type_specs.items():
                if type_name == "str":
                    type_map[field] = str
                elif type_name == "int":
                    type_map[field] = int
                elif type_name == "float":
                    type_map[field] = float
                elif type_name == "bool":
                    type_map[field] = bool

            if type_map:
                errors = quality_checks.check_data_types(data, type_map)
                all_errors.extend(errors)

        # Pattern check
        patterns = quality_config.get("patterns", {})
        if patterns:
            errors = quality_checks.check_string_patterns(data, patterns)
            all_errors.extend(errors)

        # Range check
        ranges = quality_config.get("ranges", {})
        if ranges:
            # Convert ranges to tuples
            range_tuples = {k: (v.get("min"), v.get("max")) for k, v in ranges.items()}
            errors = quality_checks.check_value_ranges(data, range_tuples)
            all_errors.extend(errors)

        # Consistency check
        consistency_rules = quality_config.get("consistency", [])
        if consistency_rules:
            errors = quality_checks.check_consistency(data, consistency_rules)
            all_errors.extend(errors)

        # Calculate confidence based on errors
        is_valid = len(all_errors) == 0
        confidence = 1.0 if is_valid else max(0.0, 1.0 - (len(all_errors) * 0.15))

        # Create result
        result = self.create_result(
            status="passed" if is_valid else "failed",
            confidence=confidence,
            errors=all_errors,
            metadata={"checks_performed": ["completeness", "types", "patterns", "ranges", "consistency"]},
        )

        # Remove self from pending validators
        pending = state.get("pending_validators", []).copy()
        if self.name in pending:
            pending.remove(self.name)

        return {
            "validation_results": [result],
            "completed_validators": [self.name],
            "errors": all_errors,
            "pending_validators": pending,
        }
