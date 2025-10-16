"""JSON Schema validator implementation."""

import time
from typing import Any

import jsonschema
from jsonschema import Draft7Validator, Draft202012Validator, validators

from ..models.error_detail import ErrorDetail
from ..models.validation_result import ValidationResult


class JSONSchemaValidator:
    """
    Validates data against JSON Schema.

    Supports Draft 7 and Draft 2020-12 schemas with custom format validators.
    """

    def __init__(self, schema: dict[str, Any], draft: str = "draft7") -> None:
        """
        Initialize JSON Schema validator.

        Args:
            schema: JSON schema definition
            draft: Schema draft version ("draft7" or "draft2020-12")
        """
        self.schema = schema
        self.draft = draft

        # Select appropriate validator class
        if draft == "draft7":
            self.validator_class = Draft7Validator
        elif draft == "draft2020-12":
            self.validator_class = Draft202012Validator
        else:
            raise ValueError(f"Unsupported draft: {draft}. Use 'draft7' or 'draft2020-12'")

        # Create validator with custom format checker
        self.format_checker = jsonschema.FormatChecker()
        self.validator = self.validator_class(schema, format_checker=self.format_checker)

    def add_format(self, format_name: str, check_func: Any) -> None:
        """
        Add custom format validator.

        Args:
            format_name: Name of the format
            check_func: Function to check format validity
        """
        self.format_checker.checks(format_name)(check_func)

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data against JSON schema.

        Args:
            data: Data to validate

        Returns:
            ValidationResult with detailed error paths
        """
        start_time = time.time()
        errors = []

        try:
            # Collect all validation errors
            for error in self.validator.iter_errors(data):
                # Build path string from error path
                path_parts = []
                for part in error.absolute_path:
                    if isinstance(part, int):
                        path_parts.append(f"[{part}]")
                    else:
                        if path_parts:
                            path_parts.append(f".{part}")
                        else:
                            path_parts.append(str(part))

                path_str = "".join(path_parts) if path_parts else "root"

                # Build schema path for context
                schema_path_parts = [str(p) for p in error.absolute_schema_path]
                schema_path_str = ".".join(schema_path_parts) if schema_path_parts else "schema"

                # Create error detail
                error_detail = ErrorDetail(
                    path=path_str,
                    message=error.message,
                    severity="error",
                    code="schema_validation",
                    context={
                        "schema_path": schema_path_str,
                        "validator": error.validator,
                        "validator_value": error.validator_value,
                        "failed_value": str(error.instance)[:100],  # Limit length
                    },
                )
                errors.append(error_detail)

        except Exception as e:
            # Handle unexpected errors during validation
            errors.append(
                ErrorDetail(
                    path="root",
                    message=f"Validation failed with exception: {str(e)}",
                    severity="error",
                    code="validation_exception",
                    context={"exception_type": type(e).__name__},
                )
            )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="json_schema",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={
                "schema_draft": self.draft,
                "error_count": len(errors),
            },
        )

    def validate_partial(self, data: Any, path: str) -> ValidationResult:
        """
        Validate a partial section of data.

        Args:
            data: Complete data object
            path: JSON path to validate (e.g., "users.0.email")

        Returns:
            ValidationResult for the specific path
        """
        # Navigate to the specified path
        parts = path.split(".")
        current = data

        try:
            for part in parts:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = current[part]
        except (KeyError, IndexError, TypeError) as e:
            return ValidationResult(
                validator_name="json_schema_partial",
                status="failed",
                errors=[
                    ErrorDetail(
                        path=path,
                        message=f"Path not found: {str(e)}",
                        severity="error",
                        code="path_not_found",
                    )
                ],
            )

        # Validate the partial data
        return self.validate(current)


def validate_json_schema(
    data: Any, schema: dict[str, Any], draft: str = "draft7"
) -> ValidationResult:
    """
    Convenience function to validate data against JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema definition
        draft: Schema draft version

    Returns:
        ValidationResult with detailed error paths
    """
    validator = JSONSchemaValidator(schema, draft)
    return validator.validate(data)
