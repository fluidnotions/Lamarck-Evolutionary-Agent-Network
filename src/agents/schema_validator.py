"""Schema validation agent for HVAS-Mini."""

import json
from typing import Any, Optional

import jsonschema
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationResult, ValidationStatus


class SchemaValidatorAgent(BaseAgent):
    """Validates data against schemas (JSON Schema, Pydantic, etc.)."""

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        """Initialize the schema validator agent.

        Args:
            llm: Optional language model for enhanced explanations
        """
        super().__init__(
            name="schema_validator",
            description="Validates data structure and types against schemas",
            llm=llm,
        )

    def execute(self, state: ValidationState) -> ValidationState:
        """Validate input data against detected or provided schema.

        Steps:
        1. Detect or load schema definition
        2. Validate data structure
        3. Validate data types
        4. Validate constraints (required fields, patterns, etc.)
        5. Generate detailed error paths
        6. Update state with results

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        input_data = state.get("input_data", {})
        validation_request = state.get("validation_request", {})

        # Get schema from validation request
        schema = validation_request.get("schema")
        schema_type = validation_request.get("schema_type", "auto")

        # Detect schema type if not specified
        if schema_type == "auto":
            schema_type = self._detect_schema_type(schema)

        # Coordinate atomic validators
        results: list[ValidationResult] = []

        if schema_type == "json_schema":
            results.extend(self._validate_json_schema(input_data, schema))
        elif schema_type == "pydantic":
            results.extend(self._validate_pydantic(input_data, schema))
        elif schema_type == "unknown":
            results.append(
                ValidationResult(
                    validator_name="schema_validation",
                    status=ValidationStatus.SKIPPED,
                    message="No schema provided or schema type could not be detected",
                )
            )
        else:
            results.append(
                ValidationResult(
                    validator_name="schema_detection",
                    status=ValidationStatus.ERROR,
                    message=f"Unsupported schema type: {schema_type}",
                )
            )

        # Aggregate results
        domain_result = self._aggregate_results(results, "schema")

        # Update state
        new_state = self._update_state(state, domain_result)

        return new_state

    def _detect_schema_type(self, schema: Any) -> str:
        """Detect the type of schema provided.

        Args:
            schema: Schema definition

        Returns:
            Schema type: 'json_schema', 'pydantic', or 'unknown'
        """
        if schema is None:
            return "unknown"

        # Check if it's a Pydantic model class
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return "pydantic"

        # Check if it's a dict that looks like JSON Schema
        if isinstance(schema, dict):
            if "$schema" in schema or "type" in schema or "properties" in schema:
                return "json_schema"

        return "unknown"

    def _validate_json_schema(
        self, data: dict[str, Any], schema: Optional[dict[str, Any]]
    ) -> list[ValidationResult]:
        """Validate data against JSON Schema.

        Args:
            data: Data to validate
            schema: JSON Schema definition

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        if schema is None:
            results.append(
                ValidationResult(
                    validator_name="json_schema",
                    status=ValidationStatus.SKIPPED,
                    message="No JSON Schema provided",
                )
            )
            return results

        try:
            # Validate against schema
            jsonschema.validate(instance=data, schema=schema)

            results.append(
                ValidationResult(
                    validator_name="json_schema",
                    status=ValidationStatus.PASSED,
                    message="Data conforms to JSON Schema",
                    confidence_score=1.0,
                )
            )

        except jsonschema.ValidationError as e:
            # Extract error path
            error_path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"

            # Generate detailed error message
            error_details = {
                "path": error_path,
                "message": e.message,
                "validator": e.validator,
                "validator_value": e.validator_value,
                "schema_path": list(e.absolute_schema_path),
            }

            # Get suggestions
            suggestions = self._generate_schema_suggestions(e)

            results.append(
                ValidationResult(
                    validator_name="json_schema",
                    status=ValidationStatus.FAILED,
                    message=f"Schema validation failed at '{error_path}': {e.message}",
                    details=error_details,
                    error_path=error_path,
                    suggestions=suggestions,
                )
            )

        except jsonschema.SchemaError as e:
            results.append(
                ValidationResult(
                    validator_name="json_schema",
                    status=ValidationStatus.ERROR,
                    message=f"Invalid JSON Schema: {e.message}",
                    details={"error": str(e)},
                )
            )

        except Exception as e:
            results.append(
                ValidationResult(
                    validator_name="json_schema",
                    status=ValidationStatus.ERROR,
                    message=f"Unexpected error during schema validation: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                )
            )

        return results

    def _validate_pydantic(
        self, data: dict[str, Any], model_class: Optional[type[BaseModel]]
    ) -> list[ValidationResult]:
        """Validate data against Pydantic model.

        Args:
            data: Data to validate
            model_class: Pydantic model class

        Returns:
            List of validation results
        """
        results: list[ValidationResult] = []

        if model_class is None:
            results.append(
                ValidationResult(
                    validator_name="pydantic",
                    status=ValidationStatus.SKIPPED,
                    message="No Pydantic model provided",
                )
            )
            return results

        try:
            # Validate using Pydantic
            model_class.model_validate(data)

            results.append(
                ValidationResult(
                    validator_name="pydantic",
                    status=ValidationStatus.PASSED,
                    message=f"Data conforms to Pydantic model {model_class.__name__}",
                    confidence_score=1.0,
                )
            )

        except ValidationError as e:
            # Process each validation error
            for error in e.errors():
                error_path = ".".join(str(p) for p in error["loc"])
                error_type = error["type"]
                error_msg = error["msg"]

                suggestions = self._generate_pydantic_suggestions(error)

                results.append(
                    ValidationResult(
                        validator_name="pydantic",
                        status=ValidationStatus.FAILED,
                        message=f"Validation failed at '{error_path}': {error_msg}",
                        details={
                            "path": error_path,
                            "type": error_type,
                            "message": error_msg,
                            "input": error.get("input"),
                        },
                        error_path=error_path,
                        suggestions=suggestions,
                    )
                )

        except Exception as e:
            results.append(
                ValidationResult(
                    validator_name="pydantic",
                    status=ValidationStatus.ERROR,
                    message=f"Unexpected error during Pydantic validation: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                )
            )

        return results

    def _generate_schema_suggestions(self, error: jsonschema.ValidationError) -> list[str]:
        """Generate suggestions for fixing JSON Schema validation errors.

        Args:
            error: JSON Schema validation error

        Returns:
            List of suggestions
        """
        suggestions: list[str] = []

        if error.validator == "required":
            missing_props = error.validator_value
            suggestions.append(f"Add required properties: {', '.join(missing_props)}")

        elif error.validator == "type":
            expected_type = error.validator_value
            suggestions.append(f"Ensure the value is of type: {expected_type}")

        elif error.validator == "enum":
            valid_values = error.validator_value
            suggestions.append(f"Use one of the valid values: {', '.join(map(str, valid_values))}")

        elif error.validator == "pattern":
            pattern = error.validator_value
            suggestions.append(f"Ensure the value matches the pattern: {pattern}")

        elif error.validator == "minLength":
            min_len = error.validator_value
            suggestions.append(f"Ensure the string has at least {min_len} characters")

        elif error.validator == "maxLength":
            max_len = error.validator_value
            suggestions.append(f"Ensure the string has at most {max_len} characters")

        elif error.validator == "minimum":
            minimum = error.validator_value
            suggestions.append(f"Ensure the value is at least {minimum}")

        elif error.validator == "maximum":
            maximum = error.validator_value
            suggestions.append(f"Ensure the value is at most {maximum}")

        else:
            suggestions.append(f"Fix the validation error: {error.message}")

        return suggestions

    def _generate_pydantic_suggestions(self, error: dict[str, Any]) -> list[str]:
        """Generate suggestions for fixing Pydantic validation errors.

        Args:
            error: Pydantic validation error dict

        Returns:
            List of suggestions
        """
        suggestions: list[str] = []
        error_type = error["type"]

        if error_type == "missing":
            suggestions.append("Provide the required field")

        elif error_type.startswith("type_error"):
            suggestions.append(f"Ensure the value is of the correct type")

        elif error_type == "value_error.missing":
            suggestions.append("This field is required and cannot be null")

        elif error_type.startswith("value_error"):
            suggestions.append("Ensure the value meets the specified constraints")

        elif error_type == "extra_forbidden":
            suggestions.append("Remove this extra field that is not defined in the model")

        else:
            suggestions.append(f"Fix the validation error: {error['msg']}")

        return suggestions
