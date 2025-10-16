"""Tests for JSON Schema validator."""

import pytest

from src.validators.json_schema import JSONSchemaValidator, validate_json_schema


class TestJSONSchemaValidator:
    """Test JSON Schema validator."""

    def test_valid_schema_draft7(self) -> None:
        """Test validation with valid data using Draft 7."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }

        data = {"name": "John", "age": 30}

        validator = JSONSchemaValidator(schema, draft="draft7")
        result = validator.validate(data)

        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_invalid_schema(self) -> None:
        """Test validation with invalid data."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["age"],
        }

        data = {"age": -5}  # Age is negative

        validator = JSONSchemaValidator(schema)
        result = validator.validate(data)

        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_missing_required_field(self) -> None:
        """Test validation with missing required field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }

        data = {}  # Missing name

        validator = JSONSchemaValidator(schema)
        result = validator.validate(data)

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "required" in result.errors[0].message.lower()

    def test_nested_schema(self) -> None:
        """Test validation with nested schema."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                    "required": ["name", "email"],
                }
            },
        }

        data = {"user": {"name": "John"}}  # Missing email

        validator = JSONSchemaValidator(schema)
        result = validator.validate(data)

        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_array_validation(self) -> None:
        """Test validation with array items."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                },
                "required": ["id"],
            },
        }

        data = [{"id": 1}, {"id": 2}, {}]  # Third item missing id

        validator = JSONSchemaValidator(schema)
        result = validator.validate(data)

        assert result.status == "failed"
        assert len(result.errors) > 0
        # Check that error path includes array index
        assert "[2]" in result.errors[0].path or "2" in result.errors[0].path

    def test_error_path_generation(self) -> None:
        """Test that error paths are correctly generated."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string"},
                        },
                        "required": ["email"],
                    },
                }
            },
        }

        data = {"users": [{"email": "test@example.com"}, {}]}  # Second user missing email

        validator = JSONSchemaValidator(schema)
        result = validator.validate(data)

        assert result.status == "failed"
        assert len(result.errors) > 0
        # Path should indicate nested location
        assert "users" in result.errors[0].path

    def test_unsupported_draft(self) -> None:
        """Test that unsupported draft raises error."""
        schema = {"type": "object"}

        with pytest.raises(ValueError):
            JSONSchemaValidator(schema, draft="draft99")

    def test_convenience_function(self) -> None:
        """Test convenience function."""
        schema = {"type": "string"}
        data = "test"

        result = validate_json_schema(data, schema)

        assert result.status == "passed"
        assert result.validator_name == "json_schema"

    def test_timing_metadata(self) -> None:
        """Test that timing is recorded."""
        schema = {"type": "object"}
        data = {}

        validator = JSONSchemaValidator(schema)
        result = validator.validate(data)

        assert result.timing >= 0
        assert "schema_draft" in result.metadata
