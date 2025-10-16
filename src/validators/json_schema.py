"""JSON Schema validation."""
import jsonschema
from typing import Any, Dict, List

from src.models.validation_result import ErrorDetail


def validate_json_schema(
    data: Dict[str, Any], schema: Dict[str, Any]
) -> tuple[bool, List[ErrorDetail]]:
    """Validate data against JSON schema.

    Args:
        data: Data to validate
        schema: JSON schema

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.ValidationError as e:
        error = ErrorDetail(
            path=".".join(str(p) for p in e.path) or "root",
            message=e.message,
            code="SCHEMA_VALIDATION_ERROR",
            severity="error",
            context={"schema_path": list(e.schema_path), "validator": e.validator},
        )
        errors.append(error)
        return False, errors
    except jsonschema.SchemaError as e:
        error = ErrorDetail(
            path="schema",
            message=f"Invalid schema: {e.message}",
            code="SCHEMA_ERROR",
            severity="error",
        )
        errors.append(error)
        return False, errors
    except Exception as e:
        error = ErrorDetail(
            path="root",
            message=f"Unexpected error: {str(e)}",
            code="UNEXPECTED_ERROR",
            severity="error",
        )
        errors.append(error)
        return False, errors
