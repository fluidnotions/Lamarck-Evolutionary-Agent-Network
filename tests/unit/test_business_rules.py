"""Unit tests for BusinessRulesAgent."""

import pytest

from src.agents.business_rules import BusinessRulesAgent, Rule, RuleEngine, create_common_rules
from src.graph.state import ValidationState
from src.models.validation_result import ValidationStatus


class TestRule:
    """Test suite for Rule class."""

    def test_rule_creation(self) -> None:
        """Test creating a rule."""
        rule = Rule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            rule_type="constraint",
            condition=lambda data: data.get("value", 0) > 0,
        )

        assert rule.rule_id == "test_rule"
        assert rule.name == "Test Rule"
        assert rule.severity == "error"  # Default

    def test_rule_with_severity(self) -> None:
        """Test creating a rule with custom severity."""
        rule = Rule(
            rule_id="warning_rule",
            name="Warning Rule",
            description="A warning rule",
            rule_type="constraint",
            condition=lambda data: True,
            severity="warning",
        )

        assert rule.severity == "warning"


class TestRuleEngine:
    """Test suite for RuleEngine class."""

    def test_add_rule(self) -> None:
        """Test adding a rule to the engine."""
        engine = RuleEngine()
        rule = Rule(
            rule_id="rule1",
            name="Rule 1",
            description="Test rule",
            rule_type="constraint",
            condition=lambda data: True,
        )

        engine.add_rule(rule)
        assert "rule1" in engine.rules

    def test_evaluate_all_rules(self) -> None:
        """Test evaluating all rules."""
        engine = RuleEngine()

        rule1 = Rule(
            rule_id="rule1",
            name="Rule 1",
            description="Always passes",
            rule_type="constraint",
            condition=lambda data: True,
        )

        rule2 = Rule(
            rule_id="rule2",
            name="Rule 2",
            description="Always fails",
            rule_type="constraint",
            condition=lambda data: False,
        )

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        results = engine.evaluate({})

        assert len(results) == 2
        assert results[0][1] is True  # rule1 passed
        assert results[1][1] is False  # rule2 failed

    def test_evaluate_specific_rules(self) -> None:
        """Test evaluating specific rules only."""
        engine = RuleEngine()

        rule1 = Rule(
            rule_id="rule1",
            name="Rule 1",
            description="Test",
            rule_type="constraint",
            condition=lambda data: True,
        )

        rule2 = Rule(
            rule_id="rule2",
            name="Rule 2",
            description="Test",
            rule_type="constraint",
            condition=lambda data: False,
        )

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        results = engine.evaluate({}, rule_ids=["rule1"])

        assert len(results) == 1
        assert results[0][0].rule_id == "rule1"

    def test_evaluate_with_exception(self) -> None:
        """Test evaluating a rule that raises an exception."""
        engine = RuleEngine()

        def failing_condition(data: dict) -> bool:
            raise ValueError("Test error")

        rule = Rule(
            rule_id="failing_rule",
            name="Failing Rule",
            description="This rule throws an error",
            rule_type="constraint",
            condition=failing_condition,
        )

        engine.add_rule(rule)
        results = engine.evaluate({})

        assert len(results) == 1
        assert results[0][1] is False  # Should be treated as failure


class TestBusinessRulesAgent:
    """Test suite for BusinessRulesAgent."""

    def test_init(self) -> None:
        """Test agent initialization."""
        agent = BusinessRulesAgent()
        assert agent.name == "business_rules"
        assert agent.description == "Validates domain-specific business rules"
        assert agent.rule_engine is not None

    def test_init_with_rule_engine(self) -> None:
        """Test agent initialization with custom rule engine."""
        engine = RuleEngine()
        agent = BusinessRulesAgent(rule_engine=engine)
        assert agent.rule_engine is engine

    def test_execute_no_rules(self) -> None:
        """Test execution with no rules configured."""
        agent = BusinessRulesAgent()

        state: ValidationState = {
            "input_data": {"name": "John Doe"},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        assert domain_result.domain == "business_rules"
        # Should have a skipped result
        assert any(
            r.status == ValidationStatus.SKIPPED for r in domain_result.individual_results
        )

    def test_execute_passing_rules(self) -> None:
        """Test execution with passing rules."""
        engine = RuleEngine()

        rule = Rule(
            rule_id="age_check",
            name="Age Check",
            description="Age must be positive",
            rule_type="constraint",
            condition=lambda data: data.get("age", 0) > 0,
        )

        engine.add_rule(rule)
        agent = BusinessRulesAgent(rule_engine=engine)

        state: ValidationState = {
            "input_data": {"age": 30},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.PASSED
        assert domain_result.passed_count == 1
        assert domain_result.failed_count == 0

    def test_execute_failing_rules(self) -> None:
        """Test execution with failing rules."""
        engine = RuleEngine()

        rule = Rule(
            rule_id="age_check",
            name="Age Check",
            description="Age must be positive",
            rule_type="constraint",
            condition=lambda data: data.get("age", 0) > 0,
        )

        engine.add_rule(rule)
        agent = BusinessRulesAgent(rule_engine=engine)

        state: ValidationState = {
            "input_data": {"age": -5},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        assert domain_result.overall_status == ValidationStatus.FAILED
        assert domain_result.failed_count == 1
        assert domain_result.passed_count == 0

    def test_execute_warning_severity(self) -> None:
        """Test execution with warning severity rule."""
        engine = RuleEngine()

        rule = Rule(
            rule_id="recommendation",
            name="Recommendation",
            description="Email should be provided",
            rule_type="constraint",
            condition=lambda data: "email" in data,
            severity="warning",
        )

        engine.add_rule(rule)
        agent = BusinessRulesAgent(rule_engine=engine)

        state: ValidationState = {
            "input_data": {"name": "John Doe"},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        # Should have warning status (not failed)
        rule_result = domain_result.individual_results[0]
        assert rule_result.status == ValidationStatus.WARNING

    def test_execute_specific_rules(self) -> None:
        """Test execution with specific rule IDs."""
        engine = RuleEngine()

        rule1 = Rule(
            rule_id="rule1",
            name="Rule 1",
            description="Test",
            rule_type="constraint",
            condition=lambda data: True,
        )

        rule2 = Rule(
            rule_id="rule2",
            name="Rule 2",
            description="Test",
            rule_type="constraint",
            condition=lambda data: False,
        )

        engine.add_rule(rule1)
        engine.add_rule(rule2)
        agent = BusinessRulesAgent(rule_engine=engine)

        state: ValidationState = {
            "input_data": {},
            "validation_request": {"rule_ids": ["rule1"]},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        # Should only have result for rule1
        assert len(domain_result.individual_results) == 1
        assert domain_result.individual_results[0].validator_name == "rule_rule1"

    def test_create_common_rules(self) -> None:
        """Test creating common rules."""
        rules = create_common_rules()

        assert len(rules) > 0
        assert all(isinstance(r, Rule) for r in rules)

        # Check specific rules exist
        rule_ids = [r.rule_id for r in rules]
        assert "age_range" in rule_ids
        assert "email_format" in rule_ids
        assert "positive_price" in rule_ids

    def test_age_range_rule(self) -> None:
        """Test the age range common rule."""
        rules = create_common_rules()
        age_rule = next(r for r in rules if r.rule_id == "age_range")

        engine = RuleEngine()
        engine.add_rule(age_rule)

        # Valid age
        results = engine.evaluate({"age": 30})
        assert results[0][1] is True

        # Invalid age (too old)
        results = engine.evaluate({"age": 200})
        assert results[0][1] is False

        # Invalid age (negative)
        results = engine.evaluate({"age": -5})
        assert results[0][1] is False

    def test_email_format_rule(self) -> None:
        """Test the email format common rule."""
        rules = create_common_rules()
        email_rule = next(r for r in rules if r.rule_id == "email_format")

        engine = RuleEngine()
        engine.add_rule(email_rule)

        # Valid email
        results = engine.evaluate({"email": "john@example.com"})
        assert results[0][1] is True

        # Invalid email
        results = engine.evaluate({"email": "invalid-email"})
        assert results[0][1] is False

    def test_positive_price_rule(self) -> None:
        """Test the positive price common rule."""
        rules = create_common_rules()
        price_rule = next(r for r in rules if r.rule_id == "positive_price")

        engine = RuleEngine()
        engine.add_rule(price_rule)

        # Valid price
        results = engine.evaluate({"price": 10.99})
        assert results[0][1] is True

        # Invalid price
        results = engine.evaluate({"price": -5.0})
        assert results[0][1] is False

        # Zero price
        results = engine.evaluate({"price": 0})
        assert results[0][1] is False

    def test_date_order_rule(self) -> None:
        """Test the date order common rule."""
        rules = create_common_rules()
        date_rule = next(r for r in rules if r.rule_id == "date_order")

        engine = RuleEngine()
        engine.add_rule(date_rule)

        # Valid date order
        results = engine.evaluate({"start_date": "2024-01-01", "end_date": "2024-12-31"})
        assert results[0][1] is True

        # Invalid date order
        results = engine.evaluate({"start_date": "2024-12-31", "end_date": "2024-01-01"})
        assert results[0][1] is False

    def test_state_updates(self) -> None:
        """Test that state is properly updated."""
        agent = BusinessRulesAgent()

        state: ValidationState = {
            "input_data": {},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
            "active_validators": ["business_rules"],
        }

        result_state = agent.execute(state)

        # Check that validator was added to completed
        assert "business_rules" in result_state["completed_validators"]

        # Check that validator was removed from active
        assert "business_rules" not in result_state.get("active_validators", [])

    def test_violation_details(self) -> None:
        """Test that violation details are included."""
        engine = RuleEngine()

        rule = Rule(
            rule_id="test_rule",
            name="Test Rule",
            description="Test rule description",
            rule_type="constraint",
            condition=lambda data: False,
        )

        engine.add_rule(rule)
        agent = BusinessRulesAgent(rule_engine=engine)

        state: ValidationState = {
            "input_data": {},
            "validation_request": {},
            "validation_results": [],
            "completed_validators": [],
        }

        result_state = agent.execute(state)

        domain_result = result_state["validation_results"][0]
        rule_result = domain_result.individual_results[0]

        # Check that details include rule information
        assert "rule_id" in rule_result.details
        assert "rule_name" in rule_result.details
        assert "rule_type" in rule_result.details
        assert "description" in rule_result.details
        assert rule_result.details["rule_id"] == "test_rule"
