"""Quality check validators for data validation."""

import re
import time
from collections import Counter
from typing import Any, Callable

import numpy as np

from ..models.error_detail import ErrorDetail
from ..models.validation_result import ValidationResult


class QualityChecker:
    """
    Performs various data quality checks.

    Includes completeness, consistency, accuracy, and statistical checks.
    """

    @staticmethod
    def check_completeness(
        data: dict[str, Any],
        required_fields: list[str],
        allow_empty_strings: bool = False,
    ) -> ValidationResult:
        """
        Check for missing or null values in required fields.

        Args:
            data: Data to check
            required_fields: List of required field names
            allow_empty_strings: Whether to allow empty strings

        Returns:
            ValidationResult with missing field errors
        """
        start_time = time.time()
        errors = []

        for field in required_fields:
            if field not in data:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Required field '{field}' is missing",
                        severity="error",
                        code="missing_field",
                    )
                )
            elif data[field] is None:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Required field '{field}' is null",
                        severity="error",
                        code="null_value",
                    )
                )
            elif not allow_empty_strings and isinstance(data[field], str) and not data[field]:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Required field '{field}' is empty",
                        severity="error",
                        code="empty_value",
                    )
                )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="completeness_check",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={"required_fields": required_fields, "fields_checked": len(required_fields)},
        )

    @staticmethod
    def check_consistency(
        data: dict[str, Any], rules: list[tuple[str, Callable[[Any], bool], str]]
    ) -> ValidationResult:
        """
        Check logical consistency across fields.

        Args:
            data: Data to check
            rules: List of (field, check_function, error_message) tuples

        Returns:
            ValidationResult with consistency violations
        """
        start_time = time.time()
        errors = []

        for field, check_func, error_msg in rules:
            try:
                if not check_func(data):
                    errors.append(
                        ErrorDetail(
                            path=field,
                            message=error_msg,
                            severity="error",
                            code="consistency_violation",
                        )
                    )
            except Exception as e:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Consistency check failed: {str(e)}",
                        severity="error",
                        code="consistency_check_error",
                        context={"exception": str(e)},
                    )
                )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="consistency_check",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={"rules_checked": len(rules)},
        )

    @staticmethod
    def check_format(
        data: dict[str, Any], field: str, format_type: str
    ) -> ValidationResult:
        """
        Check if field value matches expected format.

        Args:
            data: Data to check
            field: Field name to check
            format_type: Format type (email, phone, url, date, uuid, etc.)

        Returns:
            ValidationResult with format violations
        """
        start_time = time.time()
        errors = []

        if field not in data:
            errors.append(
                ErrorDetail(
                    path=field,
                    message=f"Field '{field}' not found",
                    severity="error",
                    code="field_not_found",
                )
            )
        else:
            value = data[field]
            if value is None:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Field '{field}' is null",
                        severity="error",
                        code="null_value",
                    )
                )
            else:
                is_valid = False
                error_message = ""

                if format_type == "email":
                    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                    is_valid = bool(re.match(email_pattern, str(value)))
                    error_message = "Invalid email format"

                elif format_type == "phone":
                    phone_pattern = r"^\+?1?\d{9,15}$"
                    cleaned = re.sub(r"[\s\-\(\)]", "", str(value))
                    is_valid = bool(re.match(phone_pattern, cleaned))
                    error_message = "Invalid phone number format"

                elif format_type == "url":
                    url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
                    is_valid = bool(re.match(url_pattern, str(value), re.IGNORECASE))
                    error_message = "Invalid URL format"

                elif format_type == "uuid":
                    uuid_pattern = (
                        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
                    )
                    is_valid = bool(re.match(uuid_pattern, str(value), re.IGNORECASE))
                    error_message = "Invalid UUID format"

                elif format_type == "date":
                    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
                    is_valid = bool(re.match(date_pattern, str(value)))
                    error_message = "Invalid date format (expected YYYY-MM-DD)"

                elif format_type == "time":
                    time_pattern = r"^\d{2}:\d{2}:\d{2}$"
                    is_valid = bool(re.match(time_pattern, str(value)))
                    error_message = "Invalid time format (expected HH:MM:SS)"

                else:
                    errors.append(
                        ErrorDetail(
                            path=field,
                            message=f"Unknown format type: {format_type}",
                            severity="error",
                            code="unknown_format",
                        )
                    )

                if not is_valid and not errors:
                    errors.append(
                        ErrorDetail(
                            path=field,
                            message=error_message,
                            severity="error",
                            code="format_violation",
                            context={"format_type": format_type, "value": str(value)[:100]},
                        )
                    )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="format_check",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={"field": field, "format_type": format_type},
        )

    @staticmethod
    def check_range(
        data: dict[str, Any],
        field: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> ValidationResult:
        """
        Check if numeric field is within valid range.

        Args:
            data: Data to check
            field: Field name to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            ValidationResult with range violations
        """
        start_time = time.time()
        errors = []

        if field not in data:
            errors.append(
                ErrorDetail(
                    path=field,
                    message=f"Field '{field}' not found",
                    severity="error",
                    code="field_not_found",
                )
            )
        else:
            value = data[field]
            if value is None:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Field '{field}' is null",
                        severity="error",
                        code="null_value",
                    )
                )
            else:
                try:
                    numeric_value = float(value)

                    if min_value is not None and numeric_value < min_value:
                        errors.append(
                            ErrorDetail(
                                path=field,
                                message=f"Value {numeric_value} is below minimum {min_value}",
                                severity="error",
                                code="below_minimum",
                                context={"value": numeric_value, "min": min_value},
                            )
                        )

                    if max_value is not None and numeric_value > max_value:
                        errors.append(
                            ErrorDetail(
                                path=field,
                                message=f"Value {numeric_value} exceeds maximum {max_value}",
                                severity="error",
                                code="above_maximum",
                                context={"value": numeric_value, "max": max_value},
                            )
                        )

                except (ValueError, TypeError):
                    errors.append(
                        ErrorDetail(
                            path=field,
                            message=f"Value '{value}' is not numeric",
                            severity="error",
                            code="not_numeric",
                        )
                    )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="range_check",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={"field": field, "min": min_value, "max": max_value},
        )

    @staticmethod
    def check_enum(data: dict[str, Any], field: str, allowed_values: list[Any]) -> ValidationResult:
        """
        Check if field value is in allowed set.

        Args:
            data: Data to check
            field: Field name to check
            allowed_values: List of allowed values

        Returns:
            ValidationResult with enum violations
        """
        start_time = time.time()
        errors = []

        if field not in data:
            errors.append(
                ErrorDetail(
                    path=field,
                    message=f"Field '{field}' not found",
                    severity="error",
                    code="field_not_found",
                )
            )
        else:
            value = data[field]
            if value not in allowed_values:
                errors.append(
                    ErrorDetail(
                        path=field,
                        message=f"Value '{value}' not in allowed values: {allowed_values}",
                        severity="error",
                        code="enum_violation",
                        context={"value": value, "allowed_values": allowed_values},
                    )
                )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="enum_check",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={"field": field, "allowed_values": allowed_values},
        )

    @staticmethod
    def detect_outliers(
        values: list[float], threshold: float = 3.0, method: str = "zscore"
    ) -> ValidationResult:
        """
        Detect statistical outliers in numeric data.

        Args:
            values: List of numeric values
            threshold: Z-score threshold for outlier detection
            method: Detection method ("zscore" or "iqr")

        Returns:
            ValidationResult with outlier warnings
        """
        start_time = time.time()
        warnings = []

        if not values:
            return ValidationResult(
                validator_name="outlier_detection",
                status="passed",
                timing=time.time() - start_time,
                metadata={"method": method, "values_count": 0},
            )

        try:
            arr = np.array(values)

            if method == "zscore":
                mean = np.mean(arr)
                std = np.std(arr)

                if std == 0:
                    # All values are the same, no outliers
                    pass
                else:
                    z_scores = np.abs((arr - mean) / std)
                    outlier_indices = np.where(z_scores > threshold)[0]

                    for idx in outlier_indices:
                        warnings.append(
                            ErrorDetail(
                                path=f"values[{idx}]",
                                message=f"Outlier detected: {values[idx]} (z-score: {z_scores[idx]:.2f})",
                                severity="warning",
                                code="outlier_zscore",
                                context={
                                    "value": values[idx],
                                    "z_score": float(z_scores[idx]),
                                    "mean": float(mean),
                                    "std": float(std),
                                },
                            )
                        )

            elif method == "iqr":
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                for idx, value in enumerate(values):
                    if value < lower_bound or value > upper_bound:
                        warnings.append(
                            ErrorDetail(
                                path=f"values[{idx}]",
                                message=f"Outlier detected: {value} (outside [{lower_bound:.2f}, {upper_bound:.2f}])",
                                severity="warning",
                                code="outlier_iqr",
                                context={
                                    "value": value,
                                    "q1": float(q1),
                                    "q3": float(q3),
                                    "iqr": float(iqr),
                                },
                            )
                        )

        except Exception as e:
            warnings.append(
                ErrorDetail(
                    path="values",
                    message=f"Outlier detection failed: {str(e)}",
                    severity="warning",
                    code="outlier_detection_error",
                    context={"exception": str(e)},
                )
            )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="outlier_detection",
            status="passed",
            warnings=warnings,
            timing=execution_time,
            metadata={
                "method": method,
                "threshold": threshold,
                "values_count": len(values),
                "outliers_found": len(warnings),
            },
        )

    @staticmethod
    def detect_duplicates(values: list[Any], field_name: str = "values") -> ValidationResult:
        """
        Detect duplicate values in a list.

        Args:
            values: List of values to check
            field_name: Name of the field for error reporting

        Returns:
            ValidationResult with duplicate warnings
        """
        start_time = time.time()
        warnings = []

        # Count occurrences
        counter = Counter(values)
        duplicates = {value: count for value, count in counter.items() if count > 1}

        for value, count in duplicates.items():
            # Find indices of duplicates
            indices = [i for i, v in enumerate(values) if v == value]
            warnings.append(
                ErrorDetail(
                    path=field_name,
                    message=f"Duplicate value '{value}' found {count} times at indices: {indices}",
                    severity="warning",
                    code="duplicate_value",
                    context={"value": value, "count": count, "indices": indices},
                )
            )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="duplicate_detection",
            status="passed",
            warnings=warnings,
            timing=execution_time,
            metadata={
                "field_name": field_name,
                "values_count": len(values),
                "unique_count": len(counter),
                "duplicate_values": len(duplicates),
            },
        )
