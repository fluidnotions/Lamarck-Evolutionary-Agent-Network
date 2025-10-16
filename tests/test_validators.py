"""Tests for validator modules."""
import pytest
from src.validators.json_schema import validate_json_schema
from src.validators.rule_engine import Rule, RuleEngine, create_range_rule, create_required_field_rule
from src.validators.quality_checks import (
    check_completeness,
    check_data_types,
    check_string_patterns,
    check_value_ranges,
    check_consistency,
)


class TestJsonSchemaValidator:
    """Tests for JSON schema validation."""

    def test_validates_valid_data(self, sample_data, sample_schema):
        """Test that valid data passes schema validation."""
        is_valid, errors = validate_json_schema(sample_data, sample_schema)
        assert is_valid is True
        assert len(errors) == 0

    def test_detects_missing_required_field(self, sample_schema):
        """Test that missing required fields are detected."""
        invalid_data = {"user": {"name": "John"}}  # Missing email and age
        is_valid, errors = validate_json_schema(invalid_data, sample_schema)

        assert is_valid is False
        assert len(errors) > 0

    def test_detects_type_mismatch(self, sample_schema):
        """Test that type mismatches are detected."""
        invalid_data = {
            "user": {"name": "John", "email": "john@example.com", "age": "thirty"},
            "order": {"id": "ORD-1", "total": 99.99},
        }
        is_valid, errors = validate_json_schema(invalid_data, sample_schema)

        assert is_valid is False
        assert len(errors) > 0

    def test_handles_invalid_schema(self):
        """Test handling of invalid schema."""
        invalid_schema = {"type": "invalid_type"}
        is_valid, errors = validate_json_schema({"test": "data"}, invalid_schema)

        assert is_valid is False
        assert len(errors) > 0


class TestRuleEngine:
    """Tests for rule engine."""

    def test_add_and_remove_rule(self):
        """Test adding and removing rules."""
        engine = RuleEngine()

        rule = Rule(
            name="test_rule",
            condition=lambda data: True,
            error_message="Test error",
        )

        engine.add_rule(rule)
        assert "test_rule" in engine.rules

        engine.remove_rule("test_rule")
        assert "test_rule" not in engine.rules

    def test_validates_passing_rule(self):
        """Test validation with passing rule."""
        engine = RuleEngine()

        rule = Rule(
            name="always_pass",
            condition=lambda data: True,
            error_message="Should not see this",
        )
        engine.add_rule(rule)

        passed, errors = engine.validate({"test": "data"})
        assert passed is True
        assert len(errors) == 0

    def test_validates_failing_rule(self):
        """Test validation with failing rule."""
        engine = RuleEngine()

        rule = Rule(
            name="always_fail",
            condition=lambda data: False,
            error_message="This rule always fails",
        )
        engine.add_rule(rule)

        passed, errors = engine.validate({"test": "data"})
        assert passed is False
        assert len(errors) == 1
        assert "always fails" in errors[0].message

    def test_validates_multiple_rules(self):
        """Test validation with multiple rules."""
        engine = RuleEngine()

        engine.add_rule(
            Rule("pass1", lambda data: True, "Error 1")
        )
        engine.add_rule(
            Rule("fail1", lambda data: False, "Error 2")
        )
        engine.add_rule(
            Rule("pass2", lambda data: True, "Error 3")
        )

        passed, errors = engine.validate({"test": "data"})
        assert passed is False
        assert len(errors) == 1

    def test_validates_specific_rules(self):
        """Test validating only specific rules."""
        engine = RuleEngine()

        engine.add_rule(Rule("rule1", lambda data: False, "Error 1"))
        engine.add_rule(Rule("rule2", lambda data: True, "Error 2"))

        # Only validate rule2
        passed, errors = engine.validate({"test": "data"}, ["rule2"])
        assert passed is True
        assert len(errors) == 0


class TestRuleBuilders:
    """Tests for rule builder functions."""

    def test_range_rule_within_range(self):
        """Test range rule with value in range."""
        rule = create_range_rule("age", min_val=0, max_val=120)
        passed, error = rule.validate({"age": 30})

        assert passed is True
        assert error is None

    def test_range_rule_below_minimum(self):
        """Test range rule with value below minimum."""
        rule = create_range_rule("age", min_val=0, max_val=120)
        passed, error = rule.validate({"age": -1})

        assert passed is False
        assert error is not None

    def test_range_rule_above_maximum(self):
        """Test range rule with value above maximum."""
        rule = create_range_rule("age", min_val=0, max_val=120)
        passed, error = rule.validate({"age": 150})

        assert passed is False
        assert error is not None

    def test_required_field_rule_present(self):
        """Test required field rule with field present."""
        rule = create_required_field_rule("name")
        passed, error = rule.validate({"name": "John"})

        assert passed is True
        assert error is None

    def test_required_field_rule_missing(self):
        """Test required field rule with field missing."""
        rule = create_required_field_rule("name")
        passed, error = rule.validate({"other": "value"})

        assert passed is False
        assert error is not None

    def test_required_field_rule_empty_string(self):
        """Test required field rule with empty string."""
        rule = create_required_field_rule("name")
        passed, error = rule.validate({"name": "   "})

        assert passed is False
        assert error is not None


class TestQualityChecks:
    """Tests for data quality checks."""

    def test_check_completeness_all_present(self):
        """Test completeness check with all fields present."""
        data = {"name": "John", "email": "john@example.com"}
        errors = check_completeness(data, ["name", "email"])

        assert len(errors) == 0

    def test_check_completeness_missing_field(self):
        """Test completeness check with missing field."""
        data = {"name": "John"}
        errors = check_completeness(data, ["name", "email"])

        assert len(errors) == 1
        assert "email" in errors[0].path

    def test_check_completeness_empty_field(self):
        """Test completeness check with empty field."""
        data = {"name": "", "email": "john@example.com"}
        errors = check_completeness(data, ["name", "email"])

        assert len(errors) == 1
        assert "empty" in errors[0].message.lower()

    def test_check_data_types_correct(self):
        """Test type check with correct types."""
        data = {"name": "John", "age": 30}
        errors = check_data_types(data, {"name": str, "age": int})

        assert len(errors) == 0

    def test_check_data_types_incorrect(self):
        """Test type check with incorrect types."""
        data = {"name": "John", "age": "thirty"}
        errors = check_data_types(data, {"name": str, "age": int})

        assert len(errors) == 1
        assert "age" in errors[0].path

    def test_check_string_patterns_match(self):
        """Test pattern check with matching pattern."""
        data = {"email": "john@example.com"}
        errors = check_string_patterns(data, {"email": r"^[\w\.-]+@[\w\.-]+\.\w+$"})

        assert len(errors) == 0

    def test_check_string_patterns_no_match(self):
        """Test pattern check with non-matching pattern."""
        data = {"email": "invalid-email"}
        errors = check_string_patterns(data, {"email": r"^[\w\.-]+@[\w\.-]+\.\w+$"})

        assert len(errors) == 1

    def test_check_value_ranges_within(self):
        """Test range check with value in range."""
        data = {"age": 30, "score": 85.5}
        errors = check_value_ranges(data, {"age": (0, 120), "score": (0.0, 100.0)})

        assert len(errors) == 0

    def test_check_value_ranges_outside(self):
        """Test range check with value out of range."""
        data = {"age": 150}
        errors = check_value_ranges(data, {"age": (0, 120)})

        assert len(errors) == 1

    def test_check_consistency_valid(self):
        """Test consistency check with valid data."""
        data = {"start_date": "2024-01-01", "end_date": "2024-12-31"}
        errors = check_consistency(data, [("start_date", "<", "end_date")])

        assert len(errors) == 0

    def test_check_consistency_invalid(self):
        """Test consistency check with invalid data."""
        data = {"min": 100, "max": 50}
        errors = check_consistency(data, [("min", "<", "max")])

        assert len(errors) == 1
