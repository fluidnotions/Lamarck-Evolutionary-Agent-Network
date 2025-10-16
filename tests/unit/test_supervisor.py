"""Unit tests for supervisor agent."""

import json
import pytest
from unittest.mock import Mock, MagicMock

from src.agents.supervisor import SupervisorAgent
from src.agents.registry import ValidatorRegistry
from src.agents.base import BaseAgent
from src.graph.state import ValidationState, create_initial_state


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str):
        self.response = response
        self.invocations: list[list] = []

    def invoke(self, messages: list) -> Mock:
        """Mock invoke method."""
        self.invocations.append(messages)
        mock_response = Mock()
        mock_response.content = self.response
        return mock_response


class MockValidator(BaseAgent):
    """Mock validator for testing."""

    def __init__(self, name: str, capabilities: list[str]):
        super().__init__(name=name, description=f"Mock {name}", capabilities=capabilities)

    def execute(self, state: ValidationState) -> ValidationState:
        return state


class TestSupervisorAgent:
    """Test supervisor agent functionality."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.registry = ValidatorRegistry()

        # Register some mock validators
        self.schema_validator = MockValidator("schema_validator", ["schema", "json"])
        self.business_validator = MockValidator("business_rules", ["business", "rules"])
        self.quality_validator = MockValidator("data_quality", ["quality", "completeness"])

        self.registry.register(self.schema_validator)
        self.registry.register(self.business_validator)
        self.registry.register(self.quality_validator)

    def test_supervisor_initialization(self) -> None:
        """Test supervisor initializes correctly."""
        llm = MockLLM("{}")
        supervisor = SupervisorAgent(llm, self.registry)

        assert supervisor.name == "supervisor"
        assert "task_analysis" in supervisor.capabilities
        assert supervisor.validator_registry is self.registry

    def test_supervisor_analyzes_task_successfully(self) -> None:
        """Test supervisor successfully analyzes a task."""
        llm_response = json.dumps({
            "validators": ["schema_validator", "business_rules"],
            "execution_mode": "sequential",
            "reasoning": "Schema first, then business rules",
            "priority_order": ["schema_validator", "business_rules"]
        })
        llm = MockLLM(llm_response)
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"user": "test", "email": "test@test.com"},
            validation_request={"type": "user_registration"}
        )

        result = supervisor.execute(state)

        assert result["overall_status"] == "in_progress"
        assert len(result["active_validators"]) == 1
        assert result["active_validators"][0] == "schema_validator"
        assert "schema_validator" in result["pending_validators"]
        assert "business_rules" in result["pending_validators"]

    def test_supervisor_handles_parallel_execution(self) -> None:
        """Test supervisor sets up parallel execution."""
        llm_response = json.dumps({
            "validators": ["schema_validator", "data_quality"],
            "execution_mode": "parallel",
            "reasoning": "Independent validators can run in parallel",
            "priority_order": ["schema_validator", "data_quality"]
        })
        llm = MockLLM(llm_response)
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        assert result["overall_status"] == "in_progress"
        assert len(result["active_validators"]) == 2
        assert "schema_validator" in result["active_validators"]
        assert "data_quality" in result["active_validators"]
        assert result["workflow_metadata"]["execution_mode"] == "parallel"

    def test_supervisor_filters_invalid_validators(self) -> None:
        """Test supervisor filters out validators not in registry."""
        llm_response = json.dumps({
            "validators": ["schema_validator", "nonexistent_validator", "business_rules"],
            "execution_mode": "sequential",
            "reasoning": "Test with invalid validator",
            "priority_order": ["schema_validator", "nonexistent_validator", "business_rules"]
        })
        llm = MockLLM(llm_response)
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        # Should filter out nonexistent_validator
        assert "nonexistent_validator" not in result["pending_validators"]
        assert "schema_validator" in result["pending_validators"]
        assert "business_rules" in result["pending_validators"]

    def test_supervisor_handles_json_with_markdown(self) -> None:
        """Test supervisor parses JSON wrapped in markdown code blocks."""
        llm_response = """```json
{
    "validators": ["schema_validator"],
    "execution_mode": "sequential",
    "reasoning": "Only schema needed",
    "priority_order": ["schema_validator"]
}
```"""
        llm = MockLLM(llm_response)
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        assert result["overall_status"] == "in_progress"
        assert "schema_validator" in result["pending_validators"]

    def test_supervisor_handles_empty_registry(self) -> None:
        """Test supervisor handles empty registry gracefully."""
        empty_registry = ValidatorRegistry()
        llm = MockLLM("{}")
        supervisor = SupervisorAgent(llm, empty_registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        # Should still set status to in_progress but with no validators
        assert result["overall_status"] == "in_progress"
        assert len(result["pending_validators"]) == 0

    def test_supervisor_fallback_on_llm_error(self) -> None:
        """Test supervisor uses fallback when LLM fails."""
        llm = Mock()
        llm.invoke.side_effect = Exception("LLM error")
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        # Should use fallback decision with all validators
        assert result["overall_status"] == "in_progress"
        assert len(result["pending_validators"]) == 3

    def test_supervisor_fallback_on_invalid_json(self) -> None:
        """Test supervisor handles invalid JSON response."""
        llm = MockLLM("This is not valid JSON {invalid")
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        # Should use fallback
        assert result["overall_status"] == "in_progress"
        assert len(result["pending_validators"]) == 3

    def test_supervisor_stores_decision_metadata(self) -> None:
        """Test supervisor stores decision in workflow metadata."""
        llm_response = json.dumps({
            "validators": ["schema_validator"],
            "execution_mode": "sequential",
            "reasoning": "Only schema validation needed",
            "priority_order": ["schema_validator"]
        })
        llm = MockLLM(llm_response)
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        assert "execution_mode" in result["workflow_metadata"]
        assert result["workflow_metadata"]["execution_mode"] == "sequential"
        assert "reasoning" in result["workflow_metadata"]
        assert "supervisor_decision" in result["workflow_metadata"]

    def test_supervisor_creates_data_sample_for_large_input(self) -> None:
        """Test supervisor creates sample for large input data."""
        llm = MockLLM(json.dumps({
            "validators": ["schema_validator"],
            "execution_mode": "sequential",
            "reasoning": "Test",
            "priority_order": ["schema_validator"]
        }))
        supervisor = SupervisorAgent(llm, self.registry)

        # Create large input data
        large_data = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        state = create_initial_state(
            input_data=large_data,
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        # Check that LLM was invoked
        assert len(llm.invocations) == 1

        # The prompt should not contain the full large_data
        prompt = llm.invocations[0][1]["content"]
        assert len(prompt) < len(json.dumps(large_data))

    def test_supervisor_validates_execution_mode(self) -> None:
        """Test supervisor validates and corrects invalid execution mode."""
        llm_response = json.dumps({
            "validators": ["schema_validator"],
            "execution_mode": "invalid_mode",
            "reasoning": "Test",
            "priority_order": ["schema_validator"]
        })
        llm = MockLLM(llm_response)
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        result = supervisor.execute(state)

        # Should default to sequential
        assert result["workflow_metadata"]["execution_mode"] == "sequential"

    def test_supervisor_callable_interface(self) -> None:
        """Test supervisor can be called directly (for LangGraph)."""
        llm = MockLLM(json.dumps({
            "validators": ["schema_validator"],
            "execution_mode": "sequential",
            "reasoning": "Test",
            "priority_order": ["schema_validator"]
        }))
        supervisor = SupervisorAgent(llm, self.registry)

        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "validation"}
        )

        # Call supervisor directly
        result = supervisor(state)

        assert result["overall_status"] == "in_progress"
        assert "schema_validator" in result["pending_validators"]
