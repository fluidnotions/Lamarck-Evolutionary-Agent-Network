"""Tests for rule engine."""

import pytest

from src.validators.rule_engine import (
    Rule,
    RuleEngine,
    create_date_range_rule,
    create_enum_rule,
    create_positive_value_rule,
    create_required_field_rule,
)


class TestRule:
    """Test Rule class."""

    def test_create_rule(self) -> None:
        """Test creating a rule."""
        rule = Rule(
            name="test_rule",
            condition=lambda data: data.get("value", 0) > 0,
            message="Value must be positive",
        )

        assert rule.name == "test_rule"
        assert rule.severity == "error"
        assert rule.rule_type == "constraint"

    def test_evaluate_rule_pass(self) -> None:
        """Test evaluating a passing rule."""
        rule = Rule(
            name="test_rule",
            condition=lambda data: data.get("value", 0) > 0,
            message="Value must be positive",
        )

        assert rule.evaluate({"value": 10}) is True

    def test_evaluate_rule_fail(self) -> None:
        """Test evaluating a failing rule."""
        rule = Rule(
            name="test_rule",
            condition=lambda data: data.get("value", 0) > 0,
            message="Value must be positive",
        )

        assert rule.evaluate({"value": -5}) is False

    def test_rule_with_exception(self) -> None:
        """Test rule that raises exception."""
        rule = Rule(
            name="test_rule",
            condition=lambda data: data["nonexistent"] > 0,  # Will raise KeyError
            message="Test",
        )

        # Should return False on exception
        assert rule.evaluate({}) is False

    def test_invalid_severity(self) -> None:
        """Test creating rule with invalid severity."""
        with pytest.raises(ValueError):
            Rule(
                name="test",
                condition=lambda d: True,
                message="test",
                severity="critical",  # Invalid
            )

    def test_invalid_rule_type(self) -> None:
        """Test creating rule with invalid type."""
        with pytest.raises(ValueError):
            Rule(
                name="test",
                condition=lambda d: True,
                message="test",
                rule_type="unknown",  # Invalid
            )


class TestRuleEngine:
    """Test RuleEngine class."""

    def test_create_engine(self) -> None:
        """Test creating a rule engine."""
        rules = [
            Rule("rule1", lambda d: True, "Test 1"),
            Rule("rule2", lambda d: True, "Test 2"),
        ]

        engine = RuleEngine(rules)

        assert len(engine.rules) == 2

    def test_add_rule(self) -> None:
        """Test adding a rule to engine."""
        engine = RuleEngine()

        rule = Rule("test_rule", lambda d: True, "Test")
        engine.add_rule(rule)

        assert len(engine.rules) == 1

    def test_add_duplicate_rule(self) -> None:
        """Test adding duplicate rule raises error."""
        engine = RuleEngine()

        rule1 = Rule("test_rule", lambda d: True, "Test")
        engine.add_rule(rule1)

        rule2 = Rule("test_rule", lambda d: False, "Test 2")
        with pytest.raises(ValueError):
            engine.add_rule(rule2)

    def test_evaluate_all_pass(self) -> None:
        """Test evaluating when all rules pass."""
        rules = [
            Rule("rule1", lambda d: d.get("value", 0) > 0, "Value must be positive"),
            Rule("rule2", lambda d: d.get("value", 0) < 100, "Value must be less than 100"),
        ]

        engine = RuleEngine(rules)
        result = engine.evaluate({"value": 50})

        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_evaluate_with_failures(self) -> None:
        """Test evaluating when rules fail."""
        rules = [
            Rule("rule1", lambda d: d.get("value", 0) > 0, "Value must be positive"),
            Rule("rule2", lambda d: d.get("value", 0) < 100, "Value must be less than 100"),
        ]

        engine = RuleEngine(rules)
        result = engine.evaluate({"value": -5})

        assert result.status == "failed"
        assert len(result.errors) == 1

    def test_evaluate_with_dependencies(self) -> None:
        """Test evaluating rules with dependencies."""
        rules = [
            Rule("rule1", lambda d: "value" in d, "Value must exist"),
            Rule(
                "rule2",
                lambda d: d.get("value", 0) > 0,
                "Value must be positive",
                depends_on=["rule1"],
            ),
        ]

        engine = RuleEngine(rules)

        # Should pass both rules
        result = engine.evaluate({"value": 10})
        assert result.status == "passed"

        # Should fail rule1, skip rule2
        result = engine.evaluate({})
        assert result.status == "failed"
        assert len(result.errors) == 1  # Only rule1 fails

    def test_circular_dependency_detection(self) -> None:
        """Test that circular dependencies are detected."""
        rules = [
            Rule("rule1", lambda d: True, "Test", depends_on=["rule2"]),
            Rule("rule2", lambda d: True, "Test", depends_on=["rule1"]),
        ]

        with pytest.raises(ValueError, match="Circular dependency"):
            RuleEngine(rules)

    def test_evaluate_single_rule(self) -> None:
        """Test evaluating a single rule by name."""
        rules = [
            Rule("rule1", lambda d: d.get("value", 0) > 0, "Value must be positive"),
        ]

        engine = RuleEngine(rules)

        assert engine.evaluate_rule("rule1", {"value": 10}) is True
        assert engine.evaluate_rule("rule1", {"value": -5}) is False

    def test_evaluate_nonexistent_rule(self) -> None:
        """Test evaluating nonexistent rule raises error."""
        engine = RuleEngine()

        with pytest.raises(ValueError):
            engine.evaluate_rule("nonexistent", {})

    def test_get_rule_dependencies(self) -> None:
        """Test getting rule dependencies."""
        rules = [
            Rule("rule1", lambda d: True, "Test"),
            Rule("rule2", lambda d: True, "Test", depends_on=["rule1"]),
        ]

        engine = RuleEngine(rules)

        assert engine.get_rule_dependencies("rule1") == []
        assert engine.get_rule_dependencies("rule2") == ["rule1"]

    def test_clear_cache(self) -> None:
        """Test clearing evaluation cache."""
        rules = [
            Rule("rule1", lambda d: d.get("value", 0) > 0, "Value must be positive"),
        ]

        engine = RuleEngine(rules)
        engine.evaluate({"value": 10})

        # Cache should have results
        assert len(engine._result_cache) > 0

        engine.clear_cache()

        # Cache should be empty
        assert len(engine._result_cache) == 0

    def test_rule_severity_levels(self) -> None:
        """Test rules with different severity levels."""
        rules = [
            Rule("error_rule", lambda d: False, "Error", severity="error"),
            Rule("warning_rule", lambda d: False, "Warning", severity="warning"),
            Rule("info_rule", lambda d: False, "Info", severity="info"),
        ]

        engine = RuleEngine(rules)
        result = engine.evaluate({})

        assert len(result.errors) == 1
        assert len(result.warnings) == 1
        assert len(result.info) == 1


class TestRuleHelpers:
    """Test rule helper functions."""

    def test_create_required_field_rule(self) -> None:
        """Test creating required field rule."""
        rule = create_required_field_rule("name")

        assert rule.evaluate({"name": "John"}) is True
        assert rule.evaluate({}) is False
        assert rule.evaluate({"name": None}) is False

    def test_create_positive_value_rule(self) -> None:
        """Test creating positive value rule."""
        rule = create_positive_value_rule("age")

        assert rule.evaluate({"age": 25}) is True
        assert rule.evaluate({"age": -5}) is False
        assert rule.evaluate({"age": 0}) is False

    def test_create_date_range_rule(self) -> None:
        """Test creating date range rule."""
        rule = create_date_range_rule("start_date", "end_date")

        assert rule.evaluate({"start_date": "2024-01-01", "end_date": "2024-12-31"}) is True
        assert rule.evaluate({"start_date": "2024-12-31", "end_date": "2024-01-01"}) is False

    def test_create_enum_rule(self) -> None:
        """Test creating enum rule."""
        rule = create_enum_rule("status", ["active", "inactive"])

        assert rule.evaluate({"status": "active"}) is True
        assert rule.evaluate({"status": "unknown"}) is False
