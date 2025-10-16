"""Tests for rule DSL (rule_dsl.py)."""
import pytest
from src.validators.rule_dsl import (
    rule,
    data,
    AND,
    OR,
    NOT,
    ERROR,
    WARNING,
    INFO,
    RuleDSLParser,
    FieldAccessor,
)
from src.validators.rule_engine_v2 import ComparisonOperator, BooleanOperator


class TestFieldAccessor:
    """Test FieldAccessor class."""

    def test_equality_operator(self):
        """Test equality operator."""
        accessor = FieldAccessor("age")
        condition = accessor == 25

        assert condition.field == "age"
        assert condition.operator == ComparisonOperator.EQ
        assert condition.value == 25

    def test_inequality_operator(self):
        """Test inequality operator."""
        accessor = FieldAccessor("status")
        condition = accessor != "inactive"

        assert condition.field == "status"
        assert condition.operator == ComparisonOperator.NE
        assert condition.value == "inactive"

    def test_comparison_operators(self):
        """Test comparison operators."""
        accessor = FieldAccessor("score")

        # Less than
        condition = accessor < 100
        assert condition.operator == ComparisonOperator.LT

        # Less than or equal
        condition = accessor <= 100
        assert condition.operator == ComparisonOperator.LE

        # Greater than
        condition = accessor > 0
        assert condition.operator == ComparisonOperator.GT

        # Greater than or equal
        condition = accessor >= 0
        assert condition.operator == ComparisonOperator.GE

    def test_string_methods(self):
        """Test string-specific methods."""
        accessor = FieldAccessor("name")

        # Contains
        condition = accessor.contains("John")
        assert condition.operator == ComparisonOperator.CONTAINS

        # Starts with
        condition = accessor.starts_with("Mr")
        assert condition.operator == ComparisonOperator.STARTS_WITH

        # Ends with
        condition = accessor.ends_with("Jr")
        assert condition.operator == ComparisonOperator.ENDS_WITH

    def test_membership_methods(self):
        """Test membership methods."""
        accessor = FieldAccessor("status")

        # In
        condition = accessor.is_in(["active", "pending"])
        assert condition.operator == ComparisonOperator.IN
        assert condition.value == ["active", "pending"]

        # Not in
        condition = accessor.not_in(["deleted", "banned"])
        assert condition.operator == ComparisonOperator.NOT_IN

    def test_existence_methods(self):
        """Test existence check methods."""
        accessor = FieldAccessor("email")

        # Exists
        condition = accessor.exists()
        assert condition.operator == ComparisonOperator.NE
        assert condition.value is None

        # Is null
        condition = accessor.is_null()
        assert condition.operator == ComparisonOperator.EQ
        assert condition.value is None

        # Is empty
        condition = accessor.is_empty()
        assert condition.operator == ComparisonOperator.EQ
        assert condition.value == ""

    def test_regex_match(self):
        """Test regex matching."""
        accessor = FieldAccessor("email")

        condition = accessor.matches(r"^[a-z]+@[a-z]+\.[a-z]+$")
        assert condition.operator == ComparisonOperator.MATCHES


class TestDataProxy:
    """Test DataProxy class."""

    def test_attribute_access(self):
        """Test accessing fields via attributes."""
        condition = data.age >= 18

        assert condition.field == "age"
        assert condition.operator == ComparisonOperator.GE
        assert condition.value == 18

    def test_item_access(self):
        """Test accessing nested fields via items."""
        condition = data["user.age"] >= 18

        assert condition.field == "user.age"
        assert condition.operator == ComparisonOperator.GE


class TestRuleBuilder:
    """Test RuleBuilder class."""

    def test_basic_rule_creation(self):
        """Test creating a basic rule."""
        my_rule = (
            rule("age_check")
            .when(data.age >= 18)
            .then("Must be 18 or older")
            .build()
        )

        assert my_rule.name == "age_check"
        assert my_rule.error_message == "Must be 18 or older"
        assert my_rule.severity == "error"

    def test_rule_with_severity(self):
        """Test rule with custom severity."""
        my_rule = (
            rule("warning_rule")
            .when(data.value < 10)
            .then("Value is low")
            .severity(WARNING)
            .build()
        )

        assert my_rule.severity == "warning"

    def test_rule_with_priority(self):
        """Test rule with priority."""
        my_rule = (
            rule("important_rule")
            .when(data.critical == True)
            .then("Critical check")
            .priority(100)
            .build()
        )

        assert my_rule.priority == 100

    def test_rule_with_tags(self):
        """Test rule with tags."""
        my_rule = (
            rule("tagged_rule")
            .when(data.value > 0)
            .then("Must be positive")
            .tag("validation", "numeric")
            .build()
        )

        assert "validation" in my_rule.tags
        assert "numeric" in my_rule.tags

    def test_rule_with_metadata(self):
        """Test rule with metadata."""
        my_rule = (
            rule("metadata_rule")
            .when(data.value > 0)
            .then("Test")
            .with_metadata(author="test", version="1.0")
            .build()
        )

        assert my_rule.metadata["author"] == "test"
        assert my_rule.metadata["version"] == "1.0"

    def test_rule_with_dependencies(self):
        """Test rule with dependencies."""
        my_rule = (
            rule("dependent_rule")
            .when(data.value > 0)
            .then("Test")
            .depends_on("rule1", "rule2")
            .build()
        )

        assert "rule1" in my_rule.dependencies
        assert "rule2" in my_rule.dependencies

    def test_rule_enabled_flag(self):
        """Test setting enabled flag."""
        my_rule = (
            rule("disabled_rule")
            .when(data.value > 0)
            .then("Test")
            .enabled(False)
            .build()
        )

        assert my_rule.enabled is False

    def test_reject_alias(self):
        """Test reject() is alias for then()."""
        my_rule = (
            rule("test_rule")
            .when(data.age < 18)
            .reject("Too young")
            .build()
        )

        assert my_rule.error_message == "Too young"

    def test_complex_rule(self):
        """Test creating a complex rule with all features."""
        my_rule = (
            rule("complex_validation")
            .when(
                AND(
                    data.age >= 18,
                    data.email.exists(),
                    OR(
                        data.role == "admin",
                        data.verified == True
                    )
                )
            )
            .then("User validation failed")
            .severity(ERROR)
            .priority(10)
            .tag("user", "authentication")
            .depends_on("email_format_check")
            .with_metadata(version="2.0")
            .build()
        )

        assert my_rule.name == "complex_validation"
        assert my_rule.priority == 10
        assert "user" in my_rule.tags

    def test_missing_condition_error(self):
        """Test error when building without condition."""
        with pytest.raises(ValueError, match="must have a condition"):
            rule("incomplete").then("Error message").build()

    def test_missing_error_message_error(self):
        """Test error when building without error message."""
        with pytest.raises(ValueError, match="must have an error message"):
            rule("incomplete").when(data.value > 0).build()


class TestLogicalOperators:
    """Test logical operator functions."""

    def test_and_operator(self):
        """Test AND operator."""
        condition = AND(
            data.age >= 18,
            data.status == "active"
        )

        assert condition.operator == BooleanOperator.AND
        assert len(condition.conditions) == 2

    def test_or_operator(self):
        """Test OR operator."""
        condition = OR(
            data.is_admin == True,
            data.is_moderator == True
        )

        assert condition.operator == BooleanOperator.OR
        assert len(condition.conditions) == 2

    def test_not_operator(self):
        """Test NOT operator."""
        condition = NOT(data.is_banned == True)

        assert condition.operator == BooleanOperator.NOT
        assert len(condition.conditions) == 1

    def test_nested_operators(self):
        """Test nested logical operators."""
        condition = AND(
            data.age >= 18,
            OR(
                data.verified == True,
                data.trusted == True
            ),
            NOT(data.banned == True)
        )

        assert condition.operator == BooleanOperator.AND
        assert len(condition.conditions) == 3

        # Check nested OR
        or_condition = condition.conditions[1]
        assert or_condition.operator == BooleanOperator.OR

        # Check nested NOT
        not_condition = condition.conditions[2]
        assert not_condition.operator == BooleanOperator.NOT


class TestRuleDSLParser:
    """Test RuleDSLParser class."""

    def test_parse_simple_condition(self):
        """Test parsing simple condition."""
        parser = RuleDSLParser()

        condition = parser.parse("age >= 18")

        assert condition.field == "age"
        assert condition.operator == ComparisonOperator.GE
        assert condition.value == 18

    def test_parse_string_value(self):
        """Test parsing string values."""
        parser = RuleDSLParser()

        condition = parser.parse("status == 'active'")

        assert condition.field == "status"
        assert condition.operator == ComparisonOperator.EQ
        assert condition.value == "active"

    def test_parse_boolean_value(self):
        """Test parsing boolean values."""
        parser = RuleDSLParser()

        condition = parser.parse("is_admin == true")

        assert condition.field == "is_admin"
        assert condition.value is True

    def test_parse_null_value(self):
        """Test parsing null values."""
        parser = RuleDSLParser()

        condition = parser.parse("email != null")

        assert condition.field == "email"
        assert condition.value is None

    def test_parse_and_operator(self):
        """Test parsing AND operator."""
        parser = RuleDSLParser()

        condition = parser.parse("age >= 18 AND status == 'active'")

        assert condition.operator == BooleanOperator.AND
        assert len(condition.conditions) == 2

    def test_parse_or_operator(self):
        """Test parsing OR operator."""
        parser = RuleDSLParser()

        condition = parser.parse("is_admin == true OR is_moderator == true")

        assert condition.operator == BooleanOperator.OR
        assert len(condition.conditions) == 2

    def test_parse_not_operator(self):
        """Test parsing NOT operator."""
        parser = RuleDSLParser()

        condition = parser.parse("NOT is_banned == true")

        assert condition.operator == BooleanOperator.NOT
        assert len(condition.conditions) == 1

    def test_parse_complex_expression(self):
        """Test parsing complex expression."""
        parser = RuleDSLParser()

        condition = parser.parse("age >= 18 AND status == 'active' OR is_admin == true")

        # Parser processes AND first, then OR - so it becomes:
        # age >= 18 AND (status == 'active' OR is_admin == true)
        assert condition.operator == BooleanOperator.AND

    def test_parse_in_operator(self):
        """Test parsing 'in' operator."""
        parser = RuleDSLParser()

        condition = parser.parse("status in ['active', 'pending']")

        assert condition.field == "status"
        assert condition.operator == ComparisonOperator.IN
        assert condition.value == ["active", "pending"]

    def test_parse_comparison_operators(self):
        """Test parsing various comparison operators."""
        parser = RuleDSLParser()

        # Greater than
        cond = parser.parse("value > 100")
        assert cond.operator == ComparisonOperator.GT

        # Less than
        cond = parser.parse("value < 10")
        assert cond.operator == ComparisonOperator.LT

        # Not equal
        cond = parser.parse("status != 'inactive'")
        assert cond.operator == ComparisonOperator.NE

    def test_parse_string_operators(self):
        """Test parsing string operators."""
        parser = RuleDSLParser()

        # Contains
        cond = parser.parse("name contains 'John'")
        assert cond.operator == ComparisonOperator.CONTAINS

        # Starts with
        cond = parser.parse("name starts with 'Mr'")
        assert cond.operator == ComparisonOperator.STARTS_WITH

        # Ends with
        cond = parser.parse("name ends with 'Jr'")
        assert cond.operator == ComparisonOperator.ENDS_WITH

    def test_parse_invalid_condition(self):
        """Test parsing invalid condition."""
        parser = RuleDSLParser()

        # Parser will raise ValueError for truly invalid syntax
        with pytest.raises(ValueError):
            parser.parse("@#$%^&*()")


class TestSeverityConstants:
    """Test severity level constants."""

    def test_severity_constants(self):
        """Test severity constants are defined."""
        assert ERROR == "error"
        assert WARNING == "warning"
        assert INFO == "info"


class TestIntegration:
    """Integration tests for DSL."""

    def test_create_and_evaluate_rule(self):
        """Test creating and evaluating a rule using DSL."""
        my_rule = (
            rule("user_age_check")
            .when(data.age >= 18)
            .then("User must be at least 18 years old")
            .severity(ERROR)
            .build()
        )

        # Test with valid data
        passed, error = my_rule.evaluate({"age": 25})
        assert passed is True
        assert error is None

        # Test with invalid data
        passed, error = my_rule.evaluate({"age": 15})
        assert passed is False
        assert error is not None
        assert "18 years old" in error.message

    def test_complex_rule_evaluation(self):
        """Test complex rule with multiple conditions."""
        my_rule = (
            rule("premium_access")
            .when(
                AND(
                    data.age >= 18,
                    OR(
                        data.is_premium == True,
                        data.is_trial == True
                    ),
                    NOT(data.is_banned == True)
                )
            )
            .then("Premium access requirements not met")
            .build()
        )

        # Valid: adult, premium, not banned
        passed, _ = my_rule.evaluate({
            "age": 25,
            "is_premium": True,
            "is_banned": False
        })
        assert passed is True

        # Invalid: underage
        passed, _ = my_rule.evaluate({
            "age": 15,
            "is_premium": True,
            "is_banned": False
        })
        assert passed is False

        # Invalid: banned
        passed, _ = my_rule.evaluate({
            "age": 25,
            "is_premium": True,
            "is_banned": True
        })
        assert passed is False

        # Valid: trial user
        passed, _ = my_rule.evaluate({
            "age": 25,
            "is_trial": True,
            "is_banned": False
        })
        assert passed is True

    def test_string_operations(self):
        """Test string operation conditions."""
        my_rule = (
            rule("email_validation")
            .when(
                AND(
                    data.email.exists(),
                    data.email.contains("@"),
                    data.email.ends_with(".com")
                )
            )
            .then("Invalid email format")
            .build()
        )

        # Valid email
        passed, _ = my_rule.evaluate({"email": "user@example.com"})
        assert passed is True

        # Invalid: no @
        passed, _ = my_rule.evaluate({"email": "userexample.com"})
        assert passed is False

        # Invalid: doesn't end with .com
        passed, _ = my_rule.evaluate({"email": "user@example.org"})
        assert passed is False
