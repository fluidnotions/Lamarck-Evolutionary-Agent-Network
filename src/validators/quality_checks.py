"""Data quality validation checks."""
from typing import Any, Dict, List
import re

from src.models.validation_result import ErrorDetail


def check_completeness(data: Dict[str, Any], required_fields: List[str]) -> List[ErrorDetail]:
    """Check if all required fields are present and non-empty.

    Args:
        data: Data to check
        required_fields: List of required field names

    Returns:
        List of errors found
    """
    errors = []

    for field in required_fields:
        if field not in data:
            errors.append(
                ErrorDetail(
                    path=field,
                    message=f"Required field '{field}' is missing",
                    code="MISSING_FIELD",
                    severity="error",
                )
            )
        elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
            errors.append(
                ErrorDetail(
                    path=field,
                    message=f"Required field '{field}' is empty",
                    code="EMPTY_FIELD",
                    severity="error",
                )
            )

    return errors


def check_data_types(data: Dict[str, Any], type_specs: Dict[str, type]) -> List[ErrorDetail]:
    """Check if fields have expected data types.

    Args:
        data: Data to check
        type_specs: Dictionary mapping field names to expected types

    Returns:
        List of errors found
    """
    errors = []

    for field, expected_type in type_specs.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Field '{field}' has type {type(data[field]).__name__}, expected {expected_type.__name__}",
                        code="INVALID_TYPE",
                        severity="error",
                        context={"expected": expected_type.__name__, "actual": type(data[field]).__name__},
                    )
                )

    return errors


def check_string_patterns(data: Dict[str, Any], patterns: Dict[str, str]) -> List[ErrorDetail]:
    """Check if string fields match expected patterns.

    Args:
        data: Data to check
        patterns: Dictionary mapping field names to regex patterns

    Returns:
        List of errors found
    """
    errors = []

    for field, pattern in patterns.items():
        if field in data and isinstance(data[field], str):
            if not re.match(pattern, data[field]):
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Field '{field}' does not match expected pattern",
                        code="PATTERN_MISMATCH",
                        severity="error",
                        context={"pattern": pattern, "value": data[field]},
                    )
                )

    return errors


def check_value_ranges(
    data: Dict[str, Any], ranges: Dict[str, tuple[float | None, float | None]]
) -> List[ErrorDetail]:
    """Check if numeric fields are within expected ranges.

    Args:
        data: Data to check
        ranges: Dictionary mapping field names to (min, max) tuples

    Returns:
        List of errors found
    """
    errors = []

    for field, (min_val, max_val) in ranges.items():
        if field in data and isinstance(data[field], (int, float)):
            value = data[field]

            if min_val is not None and value < min_val:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Field '{field}' value {value} is below minimum {min_val}",
                        code="VALUE_TOO_LOW",
                        severity="error",
                        context={"value": value, "min": min_val},
                    )
                )

            if max_val is not None and value > max_val:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Field '{field}' value {value} exceeds maximum {max_val}",
                        code="VALUE_TOO_HIGH",
                        severity="error",
                        context={"value": value, "max": max_val},
                    )
                )

    return errors


def check_consistency(
    data: Dict[str, Any], consistency_rules: List[tuple[str, str, str]]
) -> List[ErrorDetail]:
    """Check consistency between related fields.

    Args:
        data: Data to check
        consistency_rules: List of (field1, operator, field2) tuples
            where operator is one of: '<', '>', '<=', '>=', '==', '!='

    Returns:
        List of errors found
    """
    errors = []

    for field1, op, field2 in consistency_rules:
        if field1 not in data or field2 not in data:
            continue

        val1, val2 = data[field1], data[field2]

        if val1 is None or val2 is None:
            continue

        valid = False
        try:
            if op == "<":
                valid = val1 < val2
            elif op == ">":
                valid = val1 > val2
            elif op == "<=":
                valid = val1 <= val2
            elif op == ">=":
                valid = val1 >= val2
            elif op == "==":
                valid = val1 == val2
            elif op == "!=":
                valid = val1 != val2
        except Exception:
            errors.append(
                ErrorDetail(
                    path=f"{field1}, {field2}",
                    message=f"Cannot compare {field1} and {field2}",
                    code="COMPARISON_ERROR",
                    severity="error",
                )
            )
            continue

        if not valid:
            errors.append(
                ErrorDetail(
                    path=f"{field1}, {field2}",
                    message=f"Consistency check failed: {field1} {op} {field2}",
                    code="CONSISTENCY_VIOLATION",
                    severity="error",
                    context={"field1": field1, "field2": field2, "operator": op},
                )
            )

    return errors


def check_uniqueness(
    data: Dict[str, Any], unique_fields: List[str], known_values: Dict[str, set]
) -> List[ErrorDetail]:
    """Check if fields that should be unique are actually unique.

    Args:
        data: Data to check
        unique_fields: List of fields that should have unique values
        known_values: Dictionary mapping field names to sets of known values

    Returns:
        List of errors found
    """
    errors = []

    for field in unique_fields:
        if field in data and data[field] is not None:
            value = data[field]

            if field in known_values and value in known_values[field]:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Duplicate value '{value}' found for unique field '{field}'",
                        code="DUPLICATE_VALUE",
                        severity="error",
                        context={"field": field, "value": str(value)},
                    )
                )

    return errors
