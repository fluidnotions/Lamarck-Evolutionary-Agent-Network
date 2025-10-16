"""Integration tests for supervisor with LangGraph workflow."""

import json
import pytest
from unittest.mock import Mock

from langgraph.graph import StateGraph, END

from src.agents.supervisor import SupervisorAgent
from src.agents.base import BaseAgent
from src.agents.registry import ValidatorRegistry
from src.graph.state import ValidationState, create_initial_state, create_validation_result
from src.graph.routing import route_to_validators, update_state_after_validator


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str):
        self.response = response

    def invoke(self, messages: list) -> Mock:
        """Mock invoke method."""
        mock_response = Mock()
        mock_response.content = self.response
        return mock_response


class MockValidator(BaseAgent):
    """Mock validator that actually validates."""

    def __init__(self, name: str, capabilities: list[str]):
        super().__init__(name=name, description=f"Mock {name}", capabilities=capabilities)

    def execute(self, state: ValidationState) -> ValidationState:
        """Execute validation and add result."""
        # Add validation result
        result = create_validation_result(
            validator_name=self.name,
            status="passed",
            messages=[f"{self.name} validation passed"],
            confidence_score=0.95
        )
        state["validation_results"].append(result)

        # Update state after completion
        state = update_state_after_validator(state, self.name)

        return state


class MockAggregator(BaseAgent):
    """Mock aggregator for testing."""

    def __init__(self) -> None:
        super().__init__(name="aggregator", description="Aggregates results")

    def execute(self, state: ValidationState) -> ValidationState:
        """Aggregate results."""
        state["overall_status"] = "completed"
        state["final_report"] = {
            "total_validators": len(state["completed_validators"]),
            "status": "success"
        }
        return state


class TestSupervisorIntegration:
    """Test supervisor integration with LangGraph."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.registry = ValidatorRegistry()

        # Create and register validators
        self.schema_validator = MockValidator("schema_validator", ["schema"])
        self.business_validator = MockValidator("business_rules", ["business"])

        self.registry.register(self.schema_validator)
        self.registry.register(self.business_validator)

    def test_supervisor_in_simple_workflow(self) -> None:
        """Test supervisor in a simple LangGraph workflow."""
        # Setup LLM to select validators
        llm_response = json.dumps({
            "validators": ["schema_validator"],
            "execution_mode": "sequential",
            "reasoning": "Only schema validation needed",
            "priority_order": ["schema_validator"]
        })
        llm = MockLLM(llm_response)

        # Create supervisor
        supervisor = SupervisorAgent(llm, self.registry)
        aggregator = MockAggregator()

        # Create workflow
        workflow = StateGraph(ValidationState)
        workflow.add_node("supervisor", supervisor)
        workflow.add_node("schema_validator", self.schema_validator)
        workflow.add_node("aggregator", aggregator)

        # Add edges
        workflow.add_conditional_edges(
            "supervisor",
            route_to_validators,
            {
                "schema_validator": "schema_validator",
                "aggregator": "aggregator"
            }
        )

        workflow.add_conditional_edges(
            "schema_validator",
            route_to_validators,
            {
                "aggregator": "aggregator",
                END: END
            }
        )

        workflow.add_edge("aggregator", END)
        workflow.set_entry_point("supervisor")

        # Compile and run
        app = workflow.compile()

        initial_state = create_initial_state(
            input_data={"user": "test"},
            validation_request={"type": "user_validation"}
        )

        result = app.invoke(initial_state)

        # Verify workflow completed
        assert result["overall_status"] == "completed"
        assert "schema_validator" in result["completed_validators"]
        assert len(result["validation_results"]) == 1
        assert result["final_report"] is not None

    def test_supervisor_sequential_workflow(self) -> None:
        """Test supervisor orchestrating sequential validators."""
        # Setup LLM to select multiple validators
        llm_response = json.dumps({
            "validators": ["schema_validator", "business_rules"],
            "execution_mode": "sequential",
            "reasoning": "Sequential validation required",
            "priority_order": ["schema_validator", "business_rules"]
        })
        llm = MockLLM(llm_response)

        # Create agents
        supervisor = SupervisorAgent(llm, self.registry)
        aggregator = MockAggregator()

        # Create workflow
        workflow = StateGraph(ValidationState)
        workflow.add_node("supervisor", supervisor)
        workflow.add_node("schema_validator", self.schema_validator)
        workflow.add_node("business_rules", self.business_validator)
        workflow.add_node("aggregator", aggregator)

        # Add conditional routing
        workflow.add_conditional_edges(
            "supervisor",
            route_to_validators,
            {
                "schema_validator": "schema_validator",
                "business_rules": "business_rules",
                "aggregator": "aggregator"
            }
        )

        workflow.add_conditional_edges(
            "schema_validator",
            route_to_validators,
            {
                "business_rules": "business_rules",
                "aggregator": "aggregator",
                END: END
            }
        )

        workflow.add_conditional_edges(
            "business_rules",
            route_to_validators,
            {
                "aggregator": "aggregator",
                END: END
            }
        )

        workflow.add_edge("aggregator", END)
        workflow.set_entry_point("supervisor")

        # Compile and run
        app = workflow.compile()

        initial_state = create_initial_state(
            input_data={"user": "test", "age": 25},
            validation_request={"type": "user_validation"}
        )

        result = app.invoke(initial_state)

        # Verify both validators ran in sequence
        assert result["overall_status"] == "completed"
        assert "schema_validator" in result["completed_validators"]
        assert "business_rules" in result["completed_validators"]
        assert len(result["validation_results"]) == 2
        assert result["workflow_metadata"]["execution_mode"] == "sequential"

    def test_supervisor_parallel_workflow(self) -> None:
        """Test supervisor sets up parallel execution mode."""
        # Setup LLM for parallel execution
        llm_response = json.dumps({
            "validators": ["schema_validator", "business_rules"],
            "execution_mode": "parallel",
            "reasoning": "Independent validators can run in parallel",
            "priority_order": ["schema_validator", "business_rules"]
        })
        llm = MockLLM(llm_response)

        # Create supervisor
        supervisor = SupervisorAgent(llm, self.registry)

        # Test state setup
        initial_state = create_initial_state(
            input_data={"user": "test", "age": 25},
            validation_request={"type": "user_validation"}
        )

        # Execute supervisor only
        result = supervisor.execute(initial_state)

        # Verify supervisor configured parallel execution
        assert result["workflow_metadata"]["execution_mode"] == "parallel"
        assert "schema_validator" in result["active_validators"]
        assert "business_rules" in result["active_validators"]
        assert len(result["active_validators"]) == 2

        # Note: True parallel execution in LangGraph requires using Send() API
        # which is beyond the scope of this MVP. The supervisor correctly
        # identifies and marks validators for parallel execution.

    def test_supervisor_state_transitions(self) -> None:
        """Test state transitions through supervisor workflow."""
        llm_response = json.dumps({
            "validators": ["schema_validator"],
            "execution_mode": "sequential",
            "reasoning": "Test",
            "priority_order": ["schema_validator"]
        })
        llm = MockLLM(llm_response)

        supervisor = SupervisorAgent(llm, self.registry)

        # Test initial state
        state = create_initial_state(
            input_data={"data": "test"},
            validation_request={"type": "test"}
        )
        assert state["overall_status"] == "pending"
        assert len(state["active_validators"]) == 0

        # After supervisor executes
        state = supervisor.execute(state)
        assert state["overall_status"] == "in_progress"
        assert len(state["active_validators"]) == 1
        assert "schema_validator" in state["pending_validators"]

        # After validator executes
        state = self.schema_validator.execute(state)
        assert "schema_validator" in state["completed_validators"]
        assert len(state["validation_results"]) == 1

    def test_supervisor_with_empty_validation_request(self) -> None:
        """Test supervisor handles empty validation request."""
        llm_response = json.dumps({
            "validators": [],
            "execution_mode": "sequential",
            "reasoning": "No validation needed",
            "priority_order": []
        })
        llm = MockLLM(llm_response)

        supervisor = SupervisorAgent(llm, self.registry)
        aggregator = MockAggregator()

        workflow = StateGraph(ValidationState)
        workflow.add_node("supervisor", supervisor)
        workflow.add_node("aggregator", aggregator)

        workflow.add_conditional_edges(
            "supervisor",
            route_to_validators,
            {
                "aggregator": "aggregator",
                END: END
            }
        )

        workflow.add_edge("aggregator", END)
        workflow.set_entry_point("supervisor")

        app = workflow.compile()

        initial_state = create_initial_state(
            input_data={},
            validation_request={}
        )

        result = app.invoke(initial_state)

        # Should complete with no validators
        assert result["overall_status"] == "completed"
        assert len(result["completed_validators"]) == 0
        assert len(result["validation_results"]) == 0
