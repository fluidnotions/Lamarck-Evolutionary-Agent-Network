"""Integration tests for domain validators with LangGraph."""

import pytest
from langgraph.graph import StateGraph, END

from src.agents.schema_validator import SchemaValidatorAgent
from src.agents.data_quality import DataQualityAgent
from src.agents.business_rules import BusinessRulesAgent, RuleEngine, Rule
from src.agents.cross_reference import CrossReferenceAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationStatus


class TestDomainValidatorsIntegration:
    """Integration tests for domain validators in LangGraph workflow."""

    def test_single_validator_in_graph(self) -> None:
        """Test a single validator in a LangGraph workflow."""
        # Create a simple graph with schema validator
        graph = StateGraph(ValidationState)

        schema_validator = SchemaValidatorAgent()
        graph.add_node("schema_validator", schema_validator.execute)

        graph.set_entry_point("schema_validator")
        graph.add_edge("schema_validator", END)

        workflow = graph.compile()

        # Execute workflow
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        initial_state: ValidationState = {
            "input_data": {"name": "Test"},
            "validation_request": {"schema": schema},
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Verify results
        assert len(result["validation_results"]) == 1
        assert result["validation_results"][0].domain == "schema"
        assert "schema_validator" in result["completed_validators"]

    def test_multiple_validators_sequential(self) -> None:
        """Test multiple validators running sequentially."""
        graph = StateGraph(ValidationState)

        schema_validator = SchemaValidatorAgent()
        data_quality = DataQualityAgent()

        graph.add_node("schema_validator", schema_validator.execute)
        graph.add_node("data_quality", data_quality.execute)

        graph.set_entry_point("schema_validator")
        graph.add_edge("schema_validator", "data_quality")
        graph.add_edge("data_quality", END)

        workflow = graph.compile()

        # Execute workflow
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        initial_state: ValidationState = {
            "input_data": {"name": "John Doe", "age": 30},
            "validation_request": {
                "schema": schema,
                "quality_config": {"required_fields": ["name", "age"]},
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Verify results
        assert len(result["validation_results"]) == 2
        assert result["validation_results"][0].domain == "schema"
        assert result["validation_results"][1].domain == "data_quality"
        assert "schema_validator" in result["completed_validators"]
        assert "data_quality" in result["completed_validators"]

    def test_all_validators_pipeline(self) -> None:
        """Test all domain validators in a complete pipeline."""
        graph = StateGraph(ValidationState)

        # Create validators
        schema_validator = SchemaValidatorAgent()
        data_quality = DataQualityAgent()

        # Create rule engine with a rule
        rule_engine = RuleEngine()
        rule_engine.add_rule(
            Rule(
                rule_id="age_check",
                name="Age Check",
                description="Age must be between 0 and 150",
                rule_type="constraint",
                condition=lambda data: 0 <= data.get("age", 0) <= 150,
            )
        )
        business_rules = BusinessRulesAgent(rule_engine=rule_engine)

        cross_reference = CrossReferenceAgent()

        # Add nodes
        graph.add_node("schema_validator", schema_validator.execute)
        graph.add_node("data_quality", data_quality.execute)
        graph.add_node("business_rules", business_rules.execute)
        graph.add_node("cross_reference", cross_reference.execute)

        # Create pipeline
        graph.set_entry_point("schema_validator")
        graph.add_edge("schema_validator", "data_quality")
        graph.add_edge("data_quality", "business_rules")
        graph.add_edge("business_rules", "cross_reference")
        graph.add_edge("cross_reference", END)

        workflow = graph.compile()

        # Execute workflow
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        }

        reference_data = {"users": [{"id": 1, "name": "Alice"}]}

        initial_state: ValidationState = {
            "input_data": {"name": "John Doe", "age": 30, "user_id": 1},
            "validation_request": {
                "schema": schema,
                "quality_config": {"required_fields": ["name", "age"]},
                "cross_reference_config": {
                    "foreign_keys": [
                        {
                            "field": "user_id",
                            "reference_table": "users",
                            "reference_field": "id",
                        }
                    ]
                },
            },
            "metadata": {"reference_data": reference_data},
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Verify results
        assert len(result["validation_results"]) == 4
        domains = [r.domain for r in result["validation_results"]]
        assert "schema" in domains
        assert "data_quality" in domains
        assert "business_rules" in domains
        assert "cross_reference" in domains

        # Verify all validators completed
        assert len(result["completed_validators"]) == 4

    def test_validator_with_failures(self) -> None:
        """Test validators with validation failures."""
        graph = StateGraph(ValidationState)

        schema_validator = SchemaValidatorAgent()
        graph.add_node("schema_validator", schema_validator.execute)

        graph.set_entry_point("schema_validator")
        graph.add_edge("schema_validator", END)

        workflow = graph.compile()

        # Execute with invalid data
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer"}},
            "required": ["age"],
        }

        initial_state: ValidationState = {
            "input_data": {"age": "not an integer"},
            "validation_request": {"schema": schema},
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Verify failure is recorded
        domain_result = result["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.FAILED
        assert domain_result.failed_count > 0

    def test_state_preservation_across_validators(self) -> None:
        """Test that state is preserved and accumulated across validators."""
        graph = StateGraph(ValidationState)

        schema_validator = SchemaValidatorAgent()
        data_quality = DataQualityAgent()

        graph.add_node("schema_validator", schema_validator.execute)
        graph.add_node("data_quality", data_quality.execute)

        graph.set_entry_point("schema_validator")
        graph.add_edge("schema_validator", "data_quality")
        graph.add_edge("data_quality", END)

        workflow = graph.compile()

        # Execute workflow
        initial_state: ValidationState = {
            "input_data": {"name": "Test"},
            "validation_request": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "quality_config": {},
            },
            "validation_results": [],
            "completed_validators": [],
            "metadata": {"custom_field": "custom_value"},
        }

        result = workflow.invoke(initial_state)

        # Verify state preservation
        assert "metadata" in result
        assert result["metadata"]["custom_field"] == "custom_value"

        # Verify state accumulation
        assert len(result["validation_results"]) == 2
        assert len(result["completed_validators"]) == 2

    def test_conditional_routing(self) -> None:
        """Test conditional routing based on validation results."""

        def route_after_schema(state: ValidationState) -> str:
            """Route to data quality if schema passed, otherwise end."""
            results = state.get("validation_results", [])
            if results and results[-1].overall_status == ValidationStatus.PASSED:
                return "data_quality"
            return "end"

        graph = StateGraph(ValidationState)

        schema_validator = SchemaValidatorAgent()
        data_quality = DataQualityAgent()

        graph.add_node("schema_validator", schema_validator.execute)
        graph.add_node("data_quality", data_quality.execute)

        graph.set_entry_point("schema_validator")
        graph.add_conditional_edges(
            "schema_validator",
            route_after_schema,
            {"data_quality": "data_quality", "end": END},
        )
        graph.add_edge("data_quality", END)

        workflow = graph.compile()

        # Test with passing schema validation
        initial_state: ValidationState = {
            "input_data": {"name": "Test"},
            "validation_request": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                "quality_config": {},
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Should have run both validators
        assert len(result["validation_results"]) == 2
        assert "data_quality" in result["completed_validators"]

    def test_parallel_execution_simulation(self) -> None:
        """Test parallel execution pattern (simulated through sequential execution)."""

        def run_parallel_validators(state: ValidationState) -> ValidationState:
            """Run multiple validators and combine results."""
            # In a real parallel setup, these would run concurrently
            schema_validator = SchemaValidatorAgent()
            data_quality = DataQualityAgent()

            # Run schema validation
            state = schema_validator.execute(state)

            # Run data quality validation
            state = data_quality.execute(state)

            return state

        graph = StateGraph(ValidationState)

        graph.add_node("parallel_validators", run_parallel_validators)
        graph.set_entry_point("parallel_validators")
        graph.add_edge("parallel_validators", END)

        workflow = graph.compile()

        # Execute workflow
        initial_state: ValidationState = {
            "input_data": {"name": "Test", "age": 30},
            "validation_request": {
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                },
                "quality_config": {"required_fields": ["name", "age"]},
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Verify both validators ran
        assert len(result["validation_results"]) == 2
        assert len(result["completed_validators"]) == 2

    def test_error_handling_in_workflow(self) -> None:
        """Test error handling when a validator encounters an error."""
        graph = StateGraph(ValidationState)

        schema_validator = SchemaValidatorAgent()
        graph.add_node("schema_validator", schema_validator.execute)

        graph.set_entry_point("schema_validator")
        graph.add_edge("schema_validator", END)

        workflow = graph.compile()

        # Execute with malformed schema (should handle gracefully)
        initial_state: ValidationState = {
            "input_data": {"name": "Test"},
            "validation_request": {"schema": "not a valid schema"},
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Should still complete (with error/skipped status)
        assert len(result["validation_results"]) >= 1
        assert "schema_validator" in result["completed_validators"]

    def test_empty_data_handling(self) -> None:
        """Test validators with empty data."""
        graph = StateGraph(ValidationState)

        data_quality = DataQualityAgent()
        graph.add_node("data_quality", data_quality.execute)

        graph.set_entry_point("data_quality")
        graph.add_edge("data_quality", END)

        workflow = graph.compile()

        # Execute with empty data
        initial_state: ValidationState = {
            "input_data": {},
            "validation_request": {"quality_config": {"required_fields": ["name"]}},
            "validation_results": [],
            "completed_validators": [],
        }

        result = workflow.invoke(initial_state)

        # Should handle empty data gracefully
        assert len(result["validation_results"]) == 1
        domain_result = result["validation_results"][0]
        # Should detect missing fields
        completeness = next(
            r for r in domain_result.individual_results if r.validator_name == "completeness"
        )
        assert completeness.status == ValidationStatus.FAILED
