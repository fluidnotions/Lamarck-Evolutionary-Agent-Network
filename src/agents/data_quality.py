"""Data quality validation agent for HVAS-Mini."""

import re
from datetime import datetime
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.models.validation_result import QualityDimension, ValidationResult, ValidationStatus


class DataQualityAgent(BaseAgent):
    """Validates data quality across multiple dimensions."""

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        """Initialize the data quality agent.

        Args:
            llm: Optional language model for enhanced suggestions
        """
        super().__init__(
            name="data_quality",
            description="Validates data completeness, consistency, and accuracy",
            llm=llm,
        )

    def execute(self, state: ValidationState) -> ValidationState:
        """Assess data quality across dimensions.

        Quality Dimensions:
        - Completeness: Missing values, required fields
        - Consistency: Cross-field validation, logical consistency
        - Accuracy: Format, range, domain validity
        - Timeliness: Freshness checks (if timestamps available)

        Returns quality scores per dimension and overall.

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        input_data = state.get("input_data", {})
        validation_request = state.get("validation_request", {})

        # Get quality configuration
        quality_config = validation_request.get("quality_config", {})

        # Check each quality dimension
        results: list[ValidationResult] = []

        # Completeness checks
        completeness_result = self._check_completeness(input_data, quality_config)
        results.append(completeness_result)

        # Consistency checks
        consistency_result = self._check_consistency(input_data, quality_config)
        results.append(consistency_result)

        # Accuracy checks
        accuracy_result = self._check_accuracy(input_data, quality_config)
        results.append(accuracy_result)

        # Timeliness checks (if applicable)
        if quality_config.get("check_timeliness", False):
            timeliness_result = self._check_timeliness(input_data, quality_config)
            results.append(timeliness_result)

        # Aggregate results
        domain_result = self._aggregate_results(results, "data_quality")

        # Update state
        new_state = self._update_state(state, domain_result)

        return new_state

    def _check_completeness(
        self, data: dict[str, Any], config: dict[str, Any]
    ) -> ValidationResult:
        """Check data completeness.

        Args:
            data: Data to check
            config: Configuration for completeness checks

        Returns:
            Validation result for completeness
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Get required fields from config
        required_fields = config.get("required_fields", [])
        allow_null = config.get("allow_null", False)

        # Check for missing required fields
        missing_fields = []
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif not allow_null and data[field] is None:
                issues.append(f"Field '{field}' is null")

        if missing_fields:
            issues.append(f"Missing required fields: {', '.join(missing_fields)}")
            suggestions.append(f"Add the required fields: {', '.join(missing_fields)}")

        # Check for empty strings in string fields
        empty_string_fields = []
        for key, value in data.items():
            if isinstance(value, str) and value.strip() == "":
                empty_string_fields.append(key)

        if empty_string_fields:
            issues.append(f"Empty string values in fields: {', '.join(empty_string_fields)}")
            suggestions.append("Provide non-empty values for string fields")

        # Calculate completeness score
        total_fields = len(required_fields) if required_fields else len(data)
        missing_count = len(missing_fields) + len(empty_string_fields)
        score = max(0.0, (total_fields - missing_count) / total_fields) if total_fields > 0 else 1.0

        # Determine status
        if missing_fields:
            status = ValidationStatus.FAILED
        elif issues:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="completeness",
            status=status,
            message=f"Completeness score: {score:.2%}",
            details={
                "dimension": "completeness",
                "score": score,
                "issues": issues,
                "missing_fields": missing_fields,
                "empty_string_fields": empty_string_fields,
            },
            suggestions=suggestions,
            confidence_score=score,
        )

    def _check_consistency(
        self, data: dict[str, Any], config: dict[str, Any]
    ) -> ValidationResult:
        """Check data consistency.

        Args:
            data: Data to check
            config: Configuration for consistency checks

        Returns:
            Validation result for consistency
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Get consistency rules from config
        consistency_rules = config.get("consistency_rules", [])

        for rule in consistency_rules:
            rule_type = rule.get("type")
            fields = rule.get("fields", [])

            if rule_type == "mutual_exclusivity":
                # Check that only one of the fields is present
                present_fields = [f for f in fields if f in data and data[f] is not None]
                if len(present_fields) > 1:
                    issues.append(f"Mutually exclusive fields present: {', '.join(present_fields)}")
                    suggestions.append(f"Only one of {', '.join(fields)} should be provided")

            elif rule_type == "conditional_required":
                # If field A is present, field B must be present
                condition_field = rule.get("if_field")
                required_field = rule.get("then_field")

                if condition_field in data and data[condition_field] is not None:
                    if required_field not in data or data[required_field] is None:
                        issues.append(
                            f"Field '{required_field}' is required when '{condition_field}' is present"
                        )
                        suggestions.append(f"Provide '{required_field}' field")

            elif rule_type == "field_relationship":
                # Check relationships between fields (e.g., start_date < end_date)
                field1 = rule.get("field1")
                field2 = rule.get("field2")
                operator = rule.get("operator", "<")

                if field1 in data and field2 in data:
                    val1 = data[field1]
                    val2 = data[field2]

                    if val1 is not None and val2 is not None:
                        if not self._compare_values(val1, val2, operator):
                            issues.append(
                                f"Field relationship violated: {field1} {operator} {field2}"
                            )
                            suggestions.append(
                                f"Ensure {field1} {operator} {field2}"
                            )

        # Calculate consistency score
        score = 1.0 if not issues else max(0.0, 1.0 - (len(issues) * 0.2))

        # Determine status
        if issues:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="consistency",
            status=status,
            message=f"Consistency score: {score:.2%}",
            details={
                "dimension": "consistency",
                "score": score,
                "issues": issues,
            },
            suggestions=suggestions,
            confidence_score=score,
        )

    def _check_accuracy(self, data: dict[str, Any], config: dict[str, Any]) -> ValidationResult:
        """Check data accuracy.

        Args:
            data: Data to check
            config: Configuration for accuracy checks

        Returns:
            Validation result for accuracy
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Get field specifications from config
        field_specs = config.get("field_specs", {})

        for field_name, spec in field_specs.items():
            if field_name not in data:
                continue

            value = data[field_name]
            if value is None:
                continue

            # Check data type
            expected_type = spec.get("type")
            if expected_type and not self._check_type(value, expected_type):
                issues.append(f"Field '{field_name}' has incorrect type (expected {expected_type})")
                suggestions.append(f"Ensure '{field_name}' is of type {expected_type}")

            # Check format (regex pattern)
            pattern = spec.get("pattern")
            if pattern and isinstance(value, str):
                if not re.match(pattern, value):
                    issues.append(f"Field '{field_name}' does not match expected pattern")
                    suggestions.append(f"Ensure '{field_name}' matches pattern: {pattern}")

            # Check range for numeric values
            if isinstance(value, (int, float)):
                min_value = spec.get("min")
                max_value = spec.get("max")

                if min_value is not None and value < min_value:
                    issues.append(f"Field '{field_name}' is below minimum value {min_value}")
                    suggestions.append(f"Ensure '{field_name}' is at least {min_value}")

                if max_value is not None and value > max_value:
                    issues.append(f"Field '{field_name}' exceeds maximum value {max_value}")
                    suggestions.append(f"Ensure '{field_name}' is at most {max_value}")

            # Check enum values
            enum_values = spec.get("enum")
            if enum_values and value not in enum_values:
                issues.append(
                    f"Field '{field_name}' has invalid value (not in {enum_values})"
                )
                suggestions.append(f"Use one of the valid values: {', '.join(map(str, enum_values))}")

        # Calculate accuracy score
        checked_fields = len([f for f in field_specs if f in data])
        score = 1.0 if not issues else max(0.0, 1.0 - (len(issues) / max(checked_fields, 1)))

        # Determine status
        if issues:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="accuracy",
            status=status,
            message=f"Accuracy score: {score:.2%}",
            details={
                "dimension": "accuracy",
                "score": score,
                "issues": issues,
            },
            suggestions=suggestions,
            confidence_score=score,
        )

    def _check_timeliness(self, data: dict[str, Any], config: dict[str, Any]) -> ValidationResult:
        """Check data timeliness.

        Args:
            data: Data to check
            config: Configuration for timeliness checks

        Returns:
            Validation result for timeliness
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Get timestamp fields from config
        timestamp_fields = config.get("timestamp_fields", [])
        max_age_days = config.get("max_age_days")

        if not timestamp_fields:
            return ValidationResult(
                validator_name="timeliness",
                status=ValidationStatus.SKIPPED,
                message="No timestamp fields configured",
            )

        for field in timestamp_fields:
            if field not in data:
                continue

            value = data[field]
            if value is None:
                continue

            # Try to parse timestamp
            try:
                if isinstance(value, str):
                    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                elif isinstance(value, (int, float)):
                    timestamp = datetime.fromtimestamp(value)
                else:
                    continue

                # Check age
                if max_age_days is not None:
                    age_days = (datetime.now() - timestamp).days
                    if age_days > max_age_days:
                        issues.append(
                            f"Field '{field}' is {age_days} days old (max: {max_age_days} days)"
                        )
                        suggestions.append("Update with more recent data")

            except Exception as e:
                issues.append(f"Failed to parse timestamp in field '{field}': {str(e)}")
                suggestions.append(f"Ensure '{field}' is a valid timestamp")

        # Calculate timeliness score
        score = 1.0 if not issues else 0.5

        # Determine status
        if issues:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="timeliness",
            status=status,
            message=f"Timeliness score: {score:.2%}",
            details={
                "dimension": "timeliness",
                "score": score,
                "issues": issues,
            },
            suggestions=suggestions,
            confidence_score=score,
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type.

        Args:
            value: Value to check
            expected_type: Expected type name

        Returns:
            True if type matches
        """
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected = type_map.get(expected_type)
        if expected is None:
            return True

        return isinstance(value, expected)

    def _compare_values(self, val1: Any, val2: Any, operator: str) -> bool:
        """Compare two values with the given operator.

        Args:
            val1: First value
            val2: Second value
            operator: Comparison operator

        Returns:
            True if comparison is true
        """
        try:
            if operator == "<":
                return val1 < val2
            elif operator == "<=":
                return val1 <= val2
            elif operator == ">":
                return val1 > val2
            elif operator == ">=":
                return val1 >= val2
            elif operator == "==":
                return val1 == val2
            elif operator == "!=":
                return val1 != val2
            else:
                return True
        except Exception:
            return False
