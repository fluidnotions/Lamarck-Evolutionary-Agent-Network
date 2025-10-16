"""Unit tests for DataQualityAgent."""

import pytest
from datetime import datetime, timedelta

from src.agents.data_quality import DataQualityAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationStatus


class TestDataQualityAgent:
    """Test suite for DataQualityAgent."""

    def test_init(self) -> None:
        """Test agent initialization."""
        agent = DataQualityAgent()
        assert agent.name == "data_quality"
        assert agent.description == "Validates data completeness, consistency, and accuracy"

    def test_completeness_all_fields_present(self) -> None:
        """Test completeness check with all required fields present."""
        agent = DataQualityAgent()

        data = {"name": "John Doe", "email": "john@example.com", "age": 30}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "required_fields": ["name", "email", "age"],
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        completeness_result = next(
            r for r in domain_result.individual_results if r.validator_name == "completeness"
        )
        assert completeness_result.status == ValidationStatus.PASSED
        assert completeness_result.confidence_score == 1.0

    def test_completeness_missing_fields(self) -> None:
        """Test completeness check with missing required fields."""
        agent = DataQualityAgent()

        data = {"name": "John Doe"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "required_fields": ["name", "email", "age"],
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        completeness_result = next(
            r for r in domain_result.individual_results if r.validator_name == "completeness"
        )
        assert completeness_result.status == ValidationStatus.FAILED
        assert completeness_result.confidence_score < 1.0
        assert completeness_result.suggestions

    def test_completeness_null_values(self) -> None:
        """Test completeness check with null values."""
        agent = DataQualityAgent()

        data = {"name": "John Doe", "email": None, "age": 30}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "required_fields": ["name", "email", "age"],
                    "allow_null": False,
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        completeness_result = next(
            r for r in domain_result.individual_results if r.validator_name == "completeness"
        )
        assert completeness_result.status == ValidationStatus.WARNING

    def test_completeness_empty_strings(self) -> None:
        """Test completeness check with empty strings."""
        agent = DataQualityAgent()

        data = {"name": "", "email": "john@example.com"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"quality_config": {}},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        completeness_result = next(
            r for r in domain_result.individual_results if r.validator_name == "completeness"
        )
        assert completeness_result.status == ValidationStatus.WARNING

    def test_consistency_mutual_exclusivity(self) -> None:
        """Test consistency check with mutual exclusivity rule."""
        agent = DataQualityAgent()

        data = {"home_phone": "123-456-7890", "mobile_phone": "098-765-4321"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "consistency_rules": [
                        {
                            "type": "mutual_exclusivity",
                            "fields": ["home_phone", "mobile_phone"],
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        consistency_result = next(
            r for r in domain_result.individual_results if r.validator_name == "consistency"
        )
        assert consistency_result.status == ValidationStatus.FAILED

    def test_consistency_conditional_required(self) -> None:
        """Test consistency check with conditional required rule."""
        agent = DataQualityAgent()

        data = {"has_spouse": True}  # Missing spouse_name

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "consistency_rules": [
                        {
                            "type": "conditional_required",
                            "if_field": "has_spouse",
                            "then_field": "spouse_name",
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        consistency_result = next(
            r for r in domain_result.individual_results if r.validator_name == "consistency"
        )
        assert consistency_result.status == ValidationStatus.FAILED

    def test_consistency_field_relationship(self) -> None:
        """Test consistency check with field relationship rule."""
        agent = DataQualityAgent()

        data = {"start_date": "2024-01-01", "end_date": "2023-01-01"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "consistency_rules": [
                        {
                            "type": "field_relationship",
                            "field1": "start_date",
                            "field2": "end_date",
                            "operator": "<",
                        }
                    ]
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        consistency_result = next(
            r for r in domain_result.individual_results if r.validator_name == "consistency"
        )
        assert consistency_result.status == ValidationStatus.FAILED

    def test_accuracy_correct_types(self) -> None:
        """Test accuracy check with correct types."""
        agent = DataQualityAgent()

        data = {"name": "John Doe", "age": 30, "is_active": True}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "field_specs": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "is_active": {"type": "boolean"},
                    }
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        accuracy_result = next(
            r for r in domain_result.individual_results if r.validator_name == "accuracy"
        )
        assert accuracy_result.status == ValidationStatus.PASSED

    def test_accuracy_wrong_types(self) -> None:
        """Test accuracy check with wrong types."""
        agent = DataQualityAgent()

        data = {"age": "thirty"}  # Should be integer

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "field_specs": {
                        "age": {"type": "integer"},
                    }
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        accuracy_result = next(
            r for r in domain_result.individual_results if r.validator_name == "accuracy"
        )
        assert accuracy_result.status == ValidationStatus.FAILED

    def test_accuracy_pattern_matching(self) -> None:
        """Test accuracy check with pattern matching."""
        agent = DataQualityAgent()

        data = {"email": "invalid-email"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "field_specs": {
                        "email": {"type": "string", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                    }
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        accuracy_result = next(
            r for r in domain_result.individual_results if r.validator_name == "accuracy"
        )
        assert accuracy_result.status == ValidationStatus.FAILED

    def test_accuracy_range_checks(self) -> None:
        """Test accuracy check with range validation."""
        agent = DataQualityAgent()

        data = {"age": 200}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "field_specs": {
                        "age": {"type": "integer", "min": 0, "max": 150},
                    }
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        accuracy_result = next(
            r for r in domain_result.individual_results if r.validator_name == "accuracy"
        )
        assert accuracy_result.status == ValidationStatus.FAILED

    def test_accuracy_enum_values(self) -> None:
        """Test accuracy check with enum validation."""
        agent = DataQualityAgent()

        data = {"status": "invalid_status"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "field_specs": {
                        "status": {"enum": ["active", "inactive", "pending"]},
                    }
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        accuracy_result = next(
            r for r in domain_result.individual_results if r.validator_name == "accuracy"
        )
        assert accuracy_result.status == ValidationStatus.FAILED

    def test_timeliness_recent_data(self) -> None:
        """Test timeliness check with recent data."""
        agent = DataQualityAgent()

        recent_timestamp = datetime.now().isoformat()
        data = {"created_at": recent_timestamp}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "check_timeliness": True,
                    "timestamp_fields": ["created_at"],
                    "max_age_days": 30,
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        timeliness_result = next(
            (r for r in domain_result.individual_results if r.validator_name == "timeliness"),
            None,
        )
        assert timeliness_result is not None
        assert timeliness_result.status == ValidationStatus.PASSED

    def test_timeliness_old_data(self) -> None:
        """Test timeliness check with old data."""
        agent = DataQualityAgent()

        old_timestamp = (datetime.now() - timedelta(days=60)).isoformat()
        data = {"created_at": old_timestamp}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {
                "quality_config": {
                    "check_timeliness": True,
                    "timestamp_fields": ["created_at"],
                    "max_age_days": 30,
                }
            },
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        timeliness_result = next(
            r for r in domain_result.individual_results if r.validator_name == "timeliness"
        )
        assert timeliness_result.status == ValidationStatus.WARNING

    def test_no_quality_config(self) -> None:
        """Test with no quality configuration."""
        agent = DataQualityAgent()

        data = {"name": "John Doe"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        # Should still complete with default checks
        assert len(result_state["validation_results"]) == 1
        assert "data_quality" in result_state["completed_validators"]

    def test_state_updates(self) -> None:
        """Test that state is properly updated."""
        agent = DataQualityAgent()

        data = {"name": "John Doe"}

        state: ValidationState = {
            "input_data": data,
            "validation_request": {"quality_config": {}},
            "validation_results": [],
            "completed_validators": [],
            "active_validators": ["data_quality"],
        }

        result_state = agent.execute(state)

        # Check that validator was added to completed
        assert "data_quality" in result_state["completed_validators"]

        # Check that validator was removed from active
        assert "data_quality" not in result_state.get("active_validators", [])
