"""Unit tests for CrossReferenceAgent."""

import pytest

from src.agents.cross_reference import CrossReferenceAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationStatus


class TestCrossReferenceAgent:
    """Test suite for CrossReferenceAgent."""

    def test_init(self) -> None:
        """Test agent initialization."""
        agent = CrossReferenceAgent()
        assert agent.name == "cross_reference"
        assert agent.description == "Validates relationships and referential integrity"

    def test_no_config(self) -> None:
        """Test execution with no cross-reference configuration."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {"id": 1, "name": "Test"},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        # Should have a skipped result
        assert any(
            r.status == ValidationStatus.SKIPPED for r in domain_result.individual_results
        )

    def test_foreign_key_valid(self) -> None:
        """Test foreign key validation with valid references."""
        agent = CrossReferenceAgent()

        # Setup reference data
        reference_data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }

        state: ValidationState = {
            "input_data": {"order_id": 100, "user_id": 1},
            "validation_request": {
                "cross_reference_config": {
                    "foreign_keys": [
                        {
                            "field": "user_id",
                            "reference_table": "users",
                            "reference_field": "id",
                        }
                    ]
                }
            },
            "metadata": {"reference_data": reference_data},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        fk_result = next(
            r for r in domain_result.individual_results if r.validator_name == "foreign_key_validation"
        )
        assert fk_result.status == ValidationStatus.PASSED

    def test_foreign_key_invalid(self) -> None:
        """Test foreign key validation with invalid reference."""
        agent = CrossReferenceAgent()

        reference_data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }

        state: ValidationState = {
            "input_data": {"order_id": 100, "user_id": 999},  # Non-existent user
            "validation_request": {
                "cross_reference_config": {
                    "foreign_keys": [
                        {
                            "field": "user_id",
                            "reference_table": "users",
                            "reference_field": "id",
                        }
                    ]
                }
            },
            "metadata": {"reference_data": reference_data},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        fk_result = next(
            r for r in domain_result.individual_results if r.validator_name == "foreign_key_validation"
        )
        assert fk_result.status == ValidationStatus.FAILED
        assert fk_result.suggestions

    def test_foreign_key_null_allowed(self) -> None:
        """Test foreign key validation with null value (allowed)."""
        agent = CrossReferenceAgent()

        reference_data = {"users": [{"id": 1, "name": "Alice"}]}

        state: ValidationState = {
            "input_data": {"order_id": 100, "user_id": None},
            "validation_request": {
                "cross_reference_config": {
                    "foreign_keys": [
                        {
                            "field": "user_id",
                            "reference_table": "users",
                            "reference_field": "id",
                            "allow_null": True,
                        }
                    ]
                }
            },
            "metadata": {"reference_data": reference_data},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        fk_result = next(
            r for r in domain_result.individual_results if r.validator_name == "foreign_key_validation"
        )
        assert fk_result.status == ValidationStatus.PASSED

    def test_foreign_key_null_not_allowed(self) -> None:
        """Test foreign key validation with null value (not allowed)."""
        agent = CrossReferenceAgent()

        reference_data = {"users": [{"id": 1, "name": "Alice"}]}

        state: ValidationState = {
            "input_data": {"order_id": 100, "user_id": None},
            "validation_request": {
                "cross_reference_config": {
                    "foreign_keys": [
                        {
                            "field": "user_id",
                            "reference_table": "users",
                            "reference_field": "id",
                            "allow_null": False,
                        }
                    ]
                }
            },
            "metadata": {"reference_data": reference_data},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        fk_result = next(
            r for r in domain_result.individual_results if r.validator_name == "foreign_key_validation"
        )
        assert fk_result.status == ValidationStatus.FAILED

    def test_cardinality_valid(self) -> None:
        """Test cardinality validation with valid counts."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {"tags": ["python", "testing"]},
            "validation_request": {
                "cross_reference_config": {
                    "cardinality": [
                        {
                            "field": "tags",
                            "min": 1,
                            "max": 5,
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        card_result = next(
            r for r in domain_result.individual_results if r.validator_name == "cardinality_validation"
        )
        assert card_result.status == ValidationStatus.PASSED

    def test_cardinality_below_minimum(self) -> None:
        """Test cardinality validation below minimum."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {"tags": []},
            "validation_request": {
                "cross_reference_config": {
                    "cardinality": [
                        {
                            "field": "tags",
                            "min": 1,
                            "max": 5,
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        card_result = next(
            r for r in domain_result.individual_results if r.validator_name == "cardinality_validation"
        )
        assert card_result.status == ValidationStatus.FAILED

    def test_cardinality_above_maximum(self) -> None:
        """Test cardinality validation above maximum."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {"tags": ["a", "b", "c", "d", "e", "f"]},
            "validation_request": {
                "cross_reference_config": {
                    "cardinality": [
                        {
                            "field": "tags",
                            "min": 1,
                            "max": 5,
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        card_result = next(
            r for r in domain_result.individual_results if r.validator_name == "cardinality_validation"
        )
        assert card_result.status == ValidationStatus.FAILED

    def test_cyclic_dependency_no_cycles(self) -> None:
        """Test cyclic dependency check with no cycles."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {
                "items": [
                    {"id": 1, "parent_id": None},
                    {"id": 2, "parent_id": 1},
                    {"id": 3, "parent_id": 1},
                ]
            },
            "validation_request": {
                "cross_reference_config": {
                    "check_cycles": True,
                    "relationship_field": "parent_id",
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        cycle_result = next(
            r for r in domain_result.individual_results if r.validator_name == "cyclic_dependency_check"
        )
        assert cycle_result.status == ValidationStatus.PASSED

    def test_cyclic_dependency_with_cycle(self) -> None:
        """Test cyclic dependency check with a cycle."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {
                "items": [
                    {"id": 1, "parent_id": 3},
                    {"id": 2, "parent_id": 1},
                    {"id": 3, "parent_id": 2},
                ]
            },
            "validation_request": {
                "cross_reference_config": {
                    "check_cycles": True,
                    "relationship_field": "parent_id",
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        cycle_result = next(
            r for r in domain_result.individual_results if r.validator_name == "cyclic_dependency_check"
        )
        assert cycle_result.status == ValidationStatus.FAILED

    def test_relationship_one_to_many(self) -> None:
        """Test one-to-many relationship validation."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {
                "user_id": 1,
                "orders": [100, 101, 102],
            },
            "validation_request": {
                "cross_reference_config": {
                    "relationship_rules": [
                        {
                            "type": "one_to_many",
                            "parent_field": "user_id",
                            "child_field": "orders",
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        rel_result = next(
            r for r in domain_result.individual_results if r.validator_name == "relationship_validation"
        )
        assert rel_result.status == ValidationStatus.PASSED

    def test_relationship_one_to_many_invalid(self) -> None:
        """Test one-to-many relationship validation with invalid structure."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {
                "user_id": 1,
                "orders": 100,  # Should be a list
            },
            "validation_request": {
                "cross_reference_config": {
                    "relationship_rules": [
                        {
                            "type": "one_to_many",
                            "parent_field": "user_id",
                            "child_field": "orders",
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        rel_result = next(
            r for r in domain_result.individual_results if r.validator_name == "relationship_validation"
        )
        assert rel_result.status == ValidationStatus.FAILED

    def test_relationship_unique_reference(self) -> None:
        """Test unique reference validation."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {
                "references": [1, 2, 3],
            },
            "validation_request": {
                "cross_reference_config": {
                    "relationship_rules": [
                        {
                            "type": "unique_reference",
                            "field": "references",
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        rel_result = next(
            r for r in domain_result.individual_results if r.validator_name == "relationship_validation"
        )
        assert rel_result.status == ValidationStatus.PASSED

    def test_relationship_unique_reference_with_duplicates(self) -> None:
        """Test unique reference validation with duplicates."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {
                "references": [1, 2, 2, 3],
            },
            "validation_request": {
                "cross_reference_config": {
                    "relationship_rules": [
                        {
                            "type": "unique_reference",
                            "field": "references",
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        rel_result = next(
            r for r in domain_result.individual_results if r.validator_name == "relationship_validation"
        )
        assert rel_result.status == ValidationStatus.FAILED

    def test_state_updates(self) -> None:
        """Test that state is properly updated."""
        agent = CrossReferenceAgent()

        state: ValidationState = {
            "input_data": {},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
            "active_validators": ["cross_reference"],
        }

        result_state = agent.execute(state)

        # Check that validator was added to completed
        assert "cross_reference" in result_state["completed_validators"]

        # Check that validator was removed from active
        assert "cross_reference" not in result_state.get("active_validators", [])

    def test_multiple_checks(self) -> None:
        """Test execution with multiple check types."""
        agent = CrossReferenceAgent()

        reference_data = {"users": [{"id": 1, "name": "Alice"}]}

        state: ValidationState = {
            "input_data": {
                "user_id": 1,
                "tags": ["python", "testing"],
            },
            "validation_request": {
                "cross_reference_config": {
                    "foreign_keys": [
                        {
                            "field": "user_id",
                            "reference_table": "users",
                            "reference_field": "id",
                        }
                    ],
                    "cardinality": [
                        {
                            "field": "tags",
                            "min": 1,
                            "max": 10,
                        }
                    ],
                }
            },
            "metadata": {"reference_data": reference_data},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        # Should have results from both checks
        assert len(domain_result.individual_results) >= 2
        assert domain_result.overall_status == ValidationStatus.PASSED
