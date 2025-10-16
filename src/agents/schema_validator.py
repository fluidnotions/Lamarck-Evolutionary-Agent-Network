"""Schema validator agent."""
from typing import Any, Dict
import logging

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.validators.json_schema import validate_json_schema

logger = logging.getLogger(__name__)


class SchemaValidatorAgent(BaseAgent):
    """Agent that validates data against JSON schemas."""

    def __init__(self, **kwargs: Any):
        """Initialize schema validator agent."""
        super().__init__(name="schema_validator", **kwargs)

    def process(self, state: ValidationState) -> Dict[str, Any]:
        """Validate data against schema.

        Args:
            state: Current validation state

        Returns:
            State updates with validation result
        """
        logger.info("Schema validator processing")

        data = state["input_data"]
        config = state.get("config", {})
        schema_config = config.get("schema", {})

        # Get schema from config or use default
        schema = schema_config.get("schema", self._get_default_schema())

        # Validate
        is_valid, errors = validate_json_schema(data, schema)

        # Create result
        result = self.create_result(
            status="passed" if is_valid else "failed",
            confidence=1.0 if is_valid else 0.0,
            errors=errors,
            metadata={"schema": schema},
        )

        # Remove self from pending validators
        pending = state.get("pending_validators", []).copy()
        if self.name in pending:
            pending.remove(self.name)

        return {
            "validation_results": [result],
            "completed_validators": [self.name],
            "errors": errors,
            "pending_validators": pending,
        }

    def _get_default_schema(self) -> Dict[str, Any]:
        """Get default JSON schema.

        Returns:
            Default schema
        """
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        }
