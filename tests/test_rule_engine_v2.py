"""Tests for advanced rule engine (rule_engine_v2.py)."""
import pytest
from src.validators.rule_engine_v2 import (
    RuleV2,
    RuleEngineV2,
    Condition,
    CompositeCondition,
    BooleanOperator,
    ComparisonOperator,
    create_field_condition,
    create_and_condition,
    create_or_condition,
    create_not_condition,
    create_xor_condition,
)


class TestCondition:
    """Test Condition class."""

    def test_simple_equality_condition(self):
        """Test simple equality condition."""
        condition = Condition(
            field="age",
            operator=ComparisonOperator.EQ,
            value=25
        )

        assert condition.evaluate({"age": 25}) is True
        assert condition.evaluate({"age": 30}) is False

    def test_comparison_operators(self):
        """Test all comparison operators."""
        data = {"value": 50}

        # Less than
        assert Condition("value", ComparisonOperator.LT, 100).evaluate(data) is True
        assert Condition("value", ComparisonOperator.LT, 30).evaluate(data) is False

        # Greater than
        assert Condition("value", ComparisonOperator.GT, 30).evaluate(data) is True
        assert Condition("value", ComparisonOperator.GT, 100).evaluate(data) is False

        # Less than or equal
        assert Condition("value", ComparisonOperator.LE, 50).evaluate(data) is True
        assert Condition("value", ComparisonOperator.LE, 49).evaluate(data) is False

        # Greater than or equal
        assert Condition("value", ComparisonOperator.GE, 50).evaluate(data) is True
        assert Condition("value", ComparisonOperator.GE, 51).evaluate(data) is False

        # Not equal
        assert Condition("value", ComparisonOperator.NE, 100).evaluate(data) is True
        assert Condition("value", ComparisonOperator.NE, 50).evaluate(data) is False

    def test_string_operators(self):
        """Test string-specific operators."""
        data = {"name": "John Doe"}

        # Contains
        assert Condition("name", ComparisonOperator.CONTAINS, "John").evaluate(data) is True
        assert Condition("name", ComparisonOperator.CONTAINS, "Jane").evaluate(data) is False

        # Starts with
        assert Condition("name", ComparisonOperator.STARTS_WITH, "John").evaluate(data) is True
        assert Condition("name", ComparisonOperator.STARTS_WITH, "Doe").evaluate(data) is False

        # Ends with
        assert Condition("name", ComparisonOperator.ENDS_WITH, "Doe").evaluate(data) is True
        assert Condition("name", ComparisonOperator.ENDS_WITH, "John").evaluate(data) is False

    def test_membership_operators(self):
        """Test in/not_in operators."""
        data = {"status": "active", "role": "admin"}

        # In
        condition_in = Condition("status", ComparisonOperator.IN, ["active", "pending"])
        assert condition_in.evaluate(data) is True

        condition_in2 = Condition("status", ComparisonOperator.IN, ["inactive", "deleted"])
        assert condition_in2.evaluate(data) is False

        # Not in
        condition_not_in = Condition("status", ComparisonOperator.NOT_IN, ["inactive", "deleted"])
        assert condition_not_in.evaluate(data) is True

    def test_regex_match(self):
        """Test regex matching."""
        data = {"email": "test@example.com"}

        condition = Condition(
            "email",
            ComparisonOperator.MATCHES,
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        assert condition.evaluate(data) is True

        invalid_data = {"email": "not-an-email"}
        assert condition.evaluate(invalid_data) is False

    def test_nested_field_access(self):
        """Test accessing nested fields."""
        data = {
            "user": {
                "profile": {
                    "age": 30
                }
            }
        }

        condition = Condition("user.profile.age", ComparisonOperator.GE, 18)
        assert condition.evaluate(data) is True

    def test_negated_condition(self):
        """Test negated conditions."""
        condition = Condition("age", ComparisonOperator.LT, 18, negate=True)

        assert condition.evaluate({"age": 25}) is True
        assert condition.evaluate({"age": 15}) is False

    def test_missing_field(self):
        """Test behavior with missing fields."""
        condition = Condition("age", ComparisonOperator.GE, 18)

        # Missing field should return False
        assert condition.evaluate({}) is False
        assert condition.evaluate({"other_field": 25}) is False


class TestCompositeCondition:
    """Test CompositeCondition class."""

    def test_and_operator(self):
        """Test AND logic."""
        composite = CompositeCondition(
            operator=BooleanOperator.AND,
            conditions=[
                Condition("age", ComparisonOperator.GE, 18),
                Condition("status", ComparisonOperator.EQ, "active")
            ]
        )

        assert composite.evaluate({"age": 25, "status": "active"}) is True
        assert composite.evaluate({"age": 15, "status": "active"}) is False
        assert composite.evaluate({"age": 25, "status": "inactive"}) is False

    def test_or_operator(self):
        """Test OR logic."""
        composite = CompositeCondition(
            operator=BooleanOperator.OR,
            conditions=[
                Condition("is_admin", ComparisonOperator.EQ, True),
                Condition("is_moderator", ComparisonOperator.EQ, True)
            ]
        )

        assert composite.evaluate({"is_admin": True, "is_moderator": False}) is True
        assert composite.evaluate({"is_admin": False, "is_moderator": True}) is True
        assert composite.evaluate({"is_admin": False, "is_moderator": False}) is False

    def test_not_operator(self):
        """Test NOT logic."""
        composite = CompositeCondition(
            operator=BooleanOperator.NOT,
            conditions=[
                Condition("is_banned", ComparisonOperator.EQ, True)
            ]
        )

        assert composite.evaluate({"is_banned": False}) is True
        assert composite.evaluate({"is_banned": True}) is False

    def test_xor_operator(self):
        """Test XOR logic."""
        composite = CompositeCondition(
            operator=BooleanOperator.XOR,
            conditions=[
                Condition("has_email", ComparisonOperator.EQ, True),
                Condition("has_phone", ComparisonOperator.EQ, True)
            ]
        )

        # Exactly one should be true
        assert composite.evaluate({"has_email": True, "has_phone": False}) is True
        assert composite.evaluate({"has_email": False, "has_phone": True}) is True
        assert composite.evaluate({"has_email": True, "has_phone": True}) is False
        assert composite.evaluate({"has_email": False, "has_phone": False}) is False

    def test_nested_composite(self):
        """Test nested composite conditions."""
        # (age >= 18 AND status == 'active') OR is_admin == True
        composite = CompositeCondition(
            operator=BooleanOperator.OR,
            conditions=[
                CompositeCondition(
                    operator=BooleanOperator.AND,
                    conditions=[
                        Condition("age", ComparisonOperator.GE, 18),
                        Condition("status", ComparisonOperator.EQ, "active")
                    ]
                ),
                Condition("is_admin", ComparisonOperator.EQ, True)
            ]
        )

        # First path: age and status
        assert composite.evaluate({"age": 25, "status": "active", "is_admin": False}) is True

        # Second path: is_admin
        assert composite.evaluate({"age": 15, "status": "inactive", "is_admin": True}) is True

        # Neither path
        assert composite.evaluate({"age": 15, "status": "inactive", "is_admin": False}) is False


class TestRuleV2:
    """Test RuleV2 class."""

    def test_basic_rule_evaluation(self):
        """Test basic rule evaluation."""
        rule = RuleV2(
            name="age_check",
            condition=Condition("age", ComparisonOperator.GE, 18),
            error_message="Must be 18 or older",
            severity="error"
        )

        passed, error = rule.evaluate({"age": 25})
        assert passed is True
        assert error is None

        passed, error = rule.evaluate({"age": 15})
        assert passed is False
        assert error is not None
        assert error.message == "Must be 18 or older"

    def test_rule_with_composite_condition(self):
        """Test rule with composite condition."""
        rule = RuleV2(
            name="user_valid",
            condition=create_and_condition(
                Condition("age", ComparisonOperator.GE, 18),
                Condition("email", ComparisonOperator.NE, None)
            ),
            error_message="User must be 18+ with valid email",
            severity="error"
        )

        passed, _ = rule.evaluate({"age": 25, "email": "test@example.com"})
        assert passed is True

        passed, _ = rule.evaluate({"age": 25, "email": None})
        assert passed is False

    def test_rule_priority_and_tags(self):
        """Test rule priority and tags."""
        rule = RuleV2(
            name="premium_check",
            condition=Condition("is_premium", ComparisonOperator.EQ, True),
            error_message="Premium required",
            severity="warning",
            priority=10,
            tags={"premium", "access_control"}
        )

        assert rule.priority == 10
        assert "premium" in rule.tags
        assert "access_control" in rule.tags

    def test_rule_dependencies(self):
        """Test rule dependencies."""
        rule = RuleV2(
            name="final_check",
            condition=Condition("valid", ComparisonOperator.EQ, True),
            error_message="Final validation failed",
            dependencies={"initial_check", "secondary_check"}
        )

        assert "initial_check" in rule.dependencies
        assert "secondary_check" in rule.dependencies

    def test_rule_metrics(self):
        """Test rule evaluation metrics."""
        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Value must be positive"
        )

        # Run multiple evaluations
        for i in range(10):
            rule.evaluate({"value": i})

        assert rule._evaluation_count == 10
        assert rule.average_time_ms > 0
        assert 0 <= rule.failure_rate <= 1

    def test_disabled_rule(self):
        """Test disabled rules."""
        rule = RuleV2(
            name="disabled_rule",
            condition=Condition("value", ComparisonOperator.GT, 100),
            error_message="Should never trigger",
            enabled=False
        )

        # Should pass even with invalid data
        passed, error = rule.evaluate({"value": 50})
        assert passed is True
        assert error is None

    def test_lambda_condition(self):
        """Test backward compatibility with lambda conditions."""
        rule = RuleV2(
            name="lambda_rule",
            condition=lambda data: data.get("age", 0) >= 18,
            error_message="Age check failed"
        )

        passed, _ = rule.evaluate({"age": 25})
        assert passed is True

        passed, _ = rule.evaluate({"age": 15})
        assert passed is False


class TestRuleEngineV2:
    """Test RuleEngineV2 class."""

    def test_add_remove_rule(self):
        """Test adding and removing rules."""
        engine = RuleEngineV2()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Value must be positive"
        )

        engine.add_rule(rule)
        assert "test_rule" in engine.rules

        engine.remove_rule("test_rule")
        assert "test_rule" not in engine.rules

    def test_enable_disable_rule(self):
        """Test enabling/disabling rules."""
        engine = RuleEngineV2()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Value must be positive"
        )

        engine.add_rule(rule)
        assert engine.rules["test_rule"].enabled is True

        engine.disable_rule("test_rule")
        assert engine.rules["test_rule"].enabled is False

        engine.enable_rule("test_rule")
        assert engine.rules["test_rule"].enabled is True

    def test_validate_all_rules(self):
        """Test validating against all rules."""
        engine = RuleEngineV2()

        engine.add_rule(RuleV2(
            name="positive",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        ))

        engine.add_rule(RuleV2(
            name="less_than_100",
            condition=Condition("value", ComparisonOperator.LT, 100),
            error_message="Must be less than 100"
        ))

        # Valid data
        passed, errors = engine.validate({"value": 50})
        assert passed is True
        assert len(errors) == 0

        # Invalid data (fails both)
        passed, errors = engine.validate({"value": -10})
        assert passed is False
        assert len(errors) == 1

        # Invalid data (fails second rule)
        passed, errors = engine.validate({"value": 150})
        assert passed is False
        assert len(errors) == 1

    def test_validate_specific_rules(self):
        """Test validating specific rules."""
        engine = RuleEngineV2()

        engine.add_rule(RuleV2(
            name="rule1",
            condition=Condition("a", ComparisonOperator.GT, 0),
            error_message="A must be positive"
        ))

        engine.add_rule(RuleV2(
            name="rule2",
            condition=Condition("b", ComparisonOperator.GT, 0),
            error_message="B must be positive"
        ))

        # Validate only rule1
        passed, errors = engine.validate({"a": -1, "b": 5}, rule_names=["rule1"])
        assert passed is False
        assert len(errors) == 1

        # Validate only rule2
        passed, errors = engine.validate({"a": -1, "b": 5}, rule_names=["rule2"])
        assert passed is True
        assert len(errors) == 0

    def test_validate_by_tags(self):
        """Test validating rules by tags."""
        engine = RuleEngineV2()

        engine.add_rule(RuleV2(
            name="age_rule",
            condition=Condition("age", ComparisonOperator.GE, 18),
            error_message="Age check",
            tags={"user_validation"}
        ))

        engine.add_rule(RuleV2(
            name="email_rule",
            condition=Condition("email", ComparisonOperator.NE, None),
            error_message="Email required",
            tags={"user_validation"}
        ))

        engine.add_rule(RuleV2(
            name="payment_rule",
            condition=Condition("payment", ComparisonOperator.GT, 0),
            error_message="Payment required",
            tags={"payment_validation"}
        ))

        # Validate only user_validation rules
        passed, errors = engine.validate(
            {"age": 25, "email": "test@example.com", "payment": -1},
            tags=["user_validation"]
        )
        assert passed is True
        assert len(errors) == 0

    def test_rule_priority_ordering(self):
        """Test rules are executed in priority order."""
        engine = RuleEngineV2()

        execution_order = []

        def make_condition(name):
            def condition(data):
                execution_order.append(name)
                return True
            return condition

        engine.add_rule(RuleV2(
            name="low",
            condition=make_condition("low"),
            error_message="Low priority",
            priority=1
        ))

        engine.add_rule(RuleV2(
            name="high",
            condition=make_condition("high"),
            error_message="High priority",
            priority=10
        ))

        engine.add_rule(RuleV2(
            name="medium",
            condition=make_condition("medium"),
            error_message="Medium priority",
            priority=5
        ))

        engine.validate({})

        # Should execute in priority order (high to low)
        assert execution_order == ["high", "medium", "low"]

    def test_fail_fast(self):
        """Test fail-fast mode."""
        engine = RuleEngineV2()

        engine.add_rule(RuleV2(
            name="rule1",
            condition=Condition("a", ComparisonOperator.GT, 0),
            error_message="Rule 1 failed"
        ))

        engine.add_rule(RuleV2(
            name="rule2",
            condition=Condition("b", ComparisonOperator.GT, 0),
            error_message="Rule 2 failed"
        ))

        # Without fail_fast - should report all errors
        passed, errors = engine.validate({"a": -1, "b": -1}, fail_fast=False)
        assert len(errors) == 2

        # With fail_fast - should stop at first error
        passed, errors = engine.validate({"a": -1, "b": -1}, fail_fast=True)
        assert len(errors) == 1

    def test_dependency_resolution(self):
        """Test dependency resolution."""
        engine = RuleEngineV2()

        execution_order = []

        def make_condition(name):
            def condition(data):
                execution_order.append(name)
                return True
            return condition

        engine.add_rule(RuleV2(
            name="final",
            condition=make_condition("final"),
            error_message="Final",
            dependencies={"middle"}
        ))

        engine.add_rule(RuleV2(
            name="middle",
            condition=make_condition("middle"),
            error_message="Middle",
            dependencies={"initial"}
        ))

        engine.add_rule(RuleV2(
            name="initial",
            condition=make_condition("initial"),
            error_message="Initial"
        ))

        engine.validate({})

        # Should execute in dependency order
        assert execution_order.index("initial") < execution_order.index("middle")
        assert execution_order.index("middle") < execution_order.index("final")

    def test_get_analytics(self):
        """Test getting analytics."""
        engine = RuleEngineV2()

        engine.add_rule(RuleV2(
            name="rule1",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Rule 1"
        ))

        # Run some evaluations
        for i in range(10):
            engine.validate({"value": i})

        analytics = engine.get_analytics()

        assert analytics["total_rules"] == 1
        assert analytics["enabled_rules"] == 1
        assert "slowest_rules" in analytics
        assert "most_failing_rules" in analytics

    def test_detect_conflicts(self):
        """Test conflict detection."""
        engine = RuleEngineV2()

        # Add rules with same priority
        engine.add_rule(RuleV2(
            name="rule1",
            condition=Condition("a", ComparisonOperator.GT, 0),
            error_message="Rule 1",
            priority=5
        ))

        engine.add_rule(RuleV2(
            name="rule2",
            condition=Condition("b", ComparisonOperator.GT, 0),
            error_message="Rule 2",
            priority=5
        ))

        conflicts = engine.detect_conflicts()
        assert len(conflicts) > 0
        assert conflicts[0]["type"] == "priority_conflict"


class TestRuleBuilders:
    """Test rule builder functions."""

    def test_create_field_condition(self):
        """Test create_field_condition builder."""
        condition = create_field_condition("age", ">=", 18)

        assert condition.field == "age"
        assert condition.operator == ComparisonOperator.GE
        assert condition.value == 18

    def test_create_and_condition(self):
        """Test create_and_condition builder."""
        c1 = Condition("a", ComparisonOperator.GT, 0)
        c2 = Condition("b", ComparisonOperator.GT, 0)

        composite = create_and_condition(c1, c2)

        assert composite.operator == BooleanOperator.AND
        assert len(composite.conditions) == 2

    def test_create_or_condition(self):
        """Test create_or_condition builder."""
        c1 = Condition("a", ComparisonOperator.GT, 0)
        c2 = Condition("b", ComparisonOperator.GT, 0)

        composite = create_or_condition(c1, c2)

        assert composite.operator == BooleanOperator.OR
        assert len(composite.conditions) == 2

    def test_create_not_condition(self):
        """Test create_not_condition builder."""
        c1 = Condition("banned", ComparisonOperator.EQ, True)

        composite = create_not_condition(c1)

        assert composite.operator == BooleanOperator.NOT
        assert len(composite.conditions) == 1

    def test_create_xor_condition(self):
        """Test create_xor_condition builder."""
        c1 = Condition("has_email", ComparisonOperator.EQ, True)
        c2 = Condition("has_phone", ComparisonOperator.EQ, True)

        composite = create_xor_condition(c1, c2)

        assert composite.operator == BooleanOperator.XOR
        assert len(composite.conditions) == 2
