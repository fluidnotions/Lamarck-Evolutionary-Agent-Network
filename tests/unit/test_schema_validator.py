"""Unit tests for SchemaValidatorAgent."""

import pytest
from pydantic import BaseModel, Field

from src.agents.schema_validator import SchemaValidatorAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationStatus


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    email: str


class TestSchemaValidatorAgent:
    """Test suite for SchemaValidatorAgent."""

    def test_init(self) -> None:
        """Test agent initialization."""
        agent = SchemaValidatorAgent()
        assert agent.name == "schema_validator"
        assert agent.description == "Validates data structure and types against schemas"

    def test_detect_json_schema(self) -> None:
        """Test JSON Schema detection."""
        agent = SchemaValidatorAgent()

        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        assert agent._detect_schema_type(schema) == "json_schema"

    def test_detect_pydantic_schema(self) -> None:
        """Test Pydantic model detection."""
        agent = SchemaValidatorAgent()
        assert agent._detect_schema_type(SampleModel) == "pydantic"

    def test_detect_unknown_schema(self) -> None:
        """Test unknown schema detection."""
        agent = SchemaValidatorAgent()
        assert agent._detect_schema_type("not a schema") == "unknown"
        assert agent._detect_schema_type(None) == "unknown"

    def test_json_schema_valid_data(self) -> None:
        """Test JSON Schema validation with valid data."""
        agent = SchemaValidatorAgent()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }

        data = {"name": "John Doe", "age": 30}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": schema, "schema_type": "json_schema"},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.domain == "schema"
        assert domain_result.overall_status == ValidationStatus.PASSED

    def test_json_schema_invalid_data(self) -> None:
        """Test JSON Schema validation with invalid data."""
        agent = SchemaValidatorAgent()

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name", "age"],
        }

        data = {"name": "John Doe"}  # Missing 'age'

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": schema, "schema_type": "json_schema"},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.domain == "schema"
        assert domain_result.overall_status == ValidationStatus.FAILED
        assert domain_result.failed_count > 0

    def test_json_schema_type_error(self) -> None:
        """Test JSON Schema validation with type error."""
        agent = SchemaValidatorAgent()

        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
        }

        data = {"age": "not an integer"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": schema, "schema_type": "json_schema"},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.FAILED

    def test_pydantic_valid_data(self) -> None:
        """Test Pydantic validation with valid data."""
        agent = SchemaValidatorAgent()

        data = {"name": "John Doe", "age": 30, "email": "john@example.com"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": SampleModel, "schema_type": "pydantic"},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.domain == "schema"
        assert domain_result.overall_status == ValidationStatus.PASSED

    def test_pydantic_invalid_data(self) -> None:
        """Test Pydantic validation with invalid data."""
        agent = SchemaValidatorAgent()

        data = {"name": "", "age": -5, "email": "john@example.com"}  # Invalid name and age

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": SampleModel, "schema_type": "pydantic"},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.FAILED
        assert domain_result.failed_count > 0

    def test_pydantic_missing_field(self) -> None:
        """Test Pydantic validation with missing field."""
        agent = SchemaValidatorAgent()

        data = {"name": "John Doe", "age": 30}  # Missing email

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": SampleModel, "schema_type": "pydantic"},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.FAILED

    def test_no_schema_provided(self) -> None:
        """Test validation when no schema is provided."""
        agent = SchemaValidatorAgent()

        data = {"name": "John Doe"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        # Should have at least one skipped result
        assert any(r.status == ValidationStatus.SKIPPED for r in domain_result.individual_results)

    def test_auto_detect_json_schema(self) -> None:
        """Test auto-detection of JSON Schema."""
        agent = SchemaValidatorAgent()

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        data = {"name": "John Doe"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": schema},  # No schema_type specified
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.PASSED

    def test_auto_detect_pydantic(self) -> None:
        """Test auto-detection of Pydantic model."""
        agent = SchemaValidatorAgent()

        data = {"name": "John Doe", "age": 30, "email": "john@example.com"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": SampleModel},  # No schema_type specified
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        assert len(result_state["validation_results"]) == 1
        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.PASSED

    def test_state_updates(self) -> None:
        """Test that state is properly updated."""
        agent = SchemaValidatorAgent()

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": "John Doe"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": schema},
            "validation_results": [],
            "completed_validators": [],
            "active_validators": ["schema_validator"],
        }

        result_state = agent.execute(state)

        # Check that validator was added to completed
        assert "schema_validator" in result_state["completed_validators"]

        # Check that validator was removed from active
        assert "schema_validator" not in result_state.get("active_validators", [])

        # Check that validation results were added
        assert len(result_state["validation_results"]) == 1

    def test_generate_suggestions_required(self) -> None:
        """Test suggestion generation for required field violations."""
        agent = SchemaValidatorAgent()

        schema = {
            "type": "object",
            "required": ["name", "email"],
        }

        data = {}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"schema": schema},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        # Check that suggestions were generated
        assert any(r.suggestions for r in domain_result.individual_results)
