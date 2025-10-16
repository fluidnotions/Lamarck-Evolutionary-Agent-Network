"""Integration tests for business rules engine."""
import pytest
from unittest.mock import Mock, MagicMock
from src.validators.rule_engine_v2 import RuleV2, RuleEngineV2
from src.validators.rule_dsl import rule, data, AND, OR, NOT, ERROR, WARNING
from src.validators.rule_manager import RuleManager, RuleStatus
from src.validators.rule_testing import RuleTestFramework, TestCase
from src.agents.business_rules import BusinessRulesAgent
from src.graph.state import ValidationState


class TestBusinessRulesIntegration:
    """Integration tests for complete business rules system."""

    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow with business rules."""
        # Create business rules agent
        agent = BusinessRulesAgent()

        # Add custom rules using DSL
        agent.add_rule(
            rule("user_age_minimum")
            .when(data.age >= 18)
            .then("User must be at least 18 years old")
            .severity(ERROR)
            .priority(10)
            .tag("user_validation", "age")
            .build()
        )

        agent.add_rule(
            rule("user_email_required")
            .when(data.email.exists())
            .then("Email address is required")
            .severity(ERROR)
            .priority(9)
            .tag("user_validation", "contact")
            .build()
        )

        agent.add_rule(
            rule("user_verified_or_trial")
            .when(
                OR(
                    data.is_verified == True,
                    data.is_trial == True
                )
            )
            .then("User must be verified or in trial period")
            .severity(WARNING)
            .priority(5)
            .tag("user_validation", "access")
            .build()
        )

        # Create state with valid data
        valid_state = ValidationState(
            input_data={
                "age": 25,
                "email": "user@example.com",
                "is_verified": True,
                "is_trial": False
            },
            config={},
            pending_validators=["business_rules"],
            validation_results=[],
            completed_validators=[],
            errors=[],
            metadata={}
        )

        # Process valid data
        result = agent.process(valid_state)

        assert len(result["validation_results"]) == 1
        assert result["validation_results"][0].status == "passed"
        assert len(result["errors"]) == 0

        # Create state with invalid data
        invalid_state = ValidationState(
            input_data={
                "age": 15,  # Too young
                "email": None,  # Missing email
                "is_verified": False,
                "is_trial": False
            },
            config={},
            pending_validators=["business_rules"],
            validation_results=[],
            completed_validators=[],
            errors=[],
            metadata={}
        )

        # Process invalid data
        result = agent.process(invalid_state)

        assert result["validation_results"][0].status == "failed"
        assert len(result["errors"]) > 0

    def test_rule_management_lifecycle(self):
        """Test complete rule management lifecycle."""
        # Create manager
        manager = RuleManager()

        # Add initial rule
        rule_v1 = (
            rule("payment_amount")
            .when(data.amount > 0)
            .then("Payment amount must be positive")
            .severity(ERROR)
            .priority(10)
            .build()
        )

        version1 = manager.add_rule(
            rule_v1,
            created_by="developer1",
            changelog="Initial version"
        )

        assert version1.version == 1
        assert version1.status == RuleStatus.ACTIVE

        # Validate data with v1
        passed, errors = manager.engine.validate({"amount": 100})
        assert passed is True

        passed, errors = manager.engine.validate({"amount": -10})
        assert passed is False

        # Add new version with stricter requirements
        rule_v2 = (
            rule("payment_amount")
            .when(
                AND(
                    data.amount > 0,
                    data.amount <= 10000
                )
            )
            .then("Payment must be between 0 and 10,000")
            .severity(ERROR)
            .priority(10)
            .build()
        )

        version2 = manager.add_rule(
            rule_v2,
            created_by="developer2",
            changelog="Added maximum limit"
        )

        assert version2.version == 2
        assert version1.status == RuleStatus.DEPRECATED

        # Test with new version
        passed, errors = manager.engine.validate({"amount": 100})
        assert passed is True

        passed, errors = manager.engine.validate({"amount": 15000})
        assert passed is False

        # Check version history
        versions = manager.get_all_versions("payment_amount")
        assert len(versions) == 2

        # Export rules
        export = manager.export_rules()
        assert "payment_amount" in export["rules"]

        # Get statistics
        stats = manager.get_statistics()
        assert stats["total_rules"] == 1
        assert stats["total_versions"] == 2

    def test_rule_conflict_detection(self):
        """Test detecting conflicts between rules."""
        manager = RuleManager()

        # Add rules with same priority (conflict)
        manager.add_rule(
            rule("rule1")
            .when(data.value > 0)
            .then("Rule 1")
            .priority(10)
            .build()
        )

        manager.add_rule(
            rule("rule2")
            .when(data.value < 100)
            .then("Rule 2")
            .priority(10)  # Same priority
            .build()
        )

        conflicts = manager.detect_conflicts()

        assert len(conflicts) > 0
        priority_conflicts = [c for c in conflicts if c.conflict_type.value == "priority_conflict"]
        assert len(priority_conflicts) > 0

    def test_rule_dependencies_execution_order(self):
        """Test rules execute in dependency order."""
        engine = RuleEngineV2()
        execution_order = []

        def make_rule_with_tracking(name, depends_on=None):
            def condition(data):
                execution_order.append(name)
                return True

            return RuleV2(
                name=name,
                condition=condition,
                error_message=f"Rule {name}",
                dependencies=set(depends_on) if depends_on else set()
            )

        # Create dependency chain: final -> middle -> initial
        engine.add_rule(make_rule_with_tracking("initial"))
        engine.add_rule(make_rule_with_tracking("middle", ["initial"]))
        engine.add_rule(make_rule_with_tracking("final", ["middle"]))

        engine.validate({})

        # Check execution order
        assert execution_order.index("initial") < execution_order.index("middle")
        assert execution_order.index("middle") < execution_order.index("final")

    def test_comprehensive_testing_workflow(self):
        """Test complete testing workflow."""
        engine = RuleEngineV2()

        # Add rules
        engine.add_rule(
            rule("age_minimum")
            .when(data.age >= 18)
            .then("Must be 18 or older")
            .build()
        )

        engine.add_rule(
            rule("email_required")
            .when(data.email.exists())
            .then("Email required")
            .build()
        )

        # Create test framework
        framework = RuleTestFramework(engine)

        # Add test cases
        framework.add_test_case("age_minimum", TestCase(
            name="valid_age",
            description="Valid age",
            data={"age": 25},
            should_pass=True
        ))

        framework.add_test_case("age_minimum", TestCase(
            name="invalid_age",
            description="Underage",
            data={"age": 15},
            should_pass=False
        ))

        framework.add_test_case("email_required", TestCase(
            name="with_email",
            description="Has email",
            data={"email": "test@example.com"},
            should_pass=True
        ))

        # Run all tests
        results = framework.run_all_tests()
        assert len(results) == 3
        assert all(r.passed for r in results)

        # Get coverage report
        report = framework.get_coverage_report()
        assert report.total_rules == 2
        assert report.tested_rules == 2
        assert report.coverage_percentage == 100.0

        # Export results
        export = framework.export_results()
        assert export["coverage"]["coverage_percentage"] == 100.0

    def test_performance_benchmarking(self):
        """Test performance benchmarking meets requirements."""
        engine = RuleEngineV2()

        # Add many rules
        for i in range(50):
            engine.add_rule(
                rule(f"rule_{i}")
                .when(data.value > i)
                .then(f"Rule {i} failed")
                .build()
            )

        framework = RuleTestFramework(engine)

        # Generate test data
        test_data = [{"value": i} for i in range(100)]

        # Benchmark engine
        benchmark = framework.benchmark_engine(test_data, num_iterations=1000)

        # Should meet <10ms target for 50 rules
        assert benchmark["avg_time_ms"] < 10.0
        assert benchmark["meets_10ms_target"] is True

    def test_complex_nested_conditions(self):
        """Test complex nested conditions."""
        engine = RuleEngineV2()

        # Create complex rule:
        # (age >= 18 AND has_account) AND
        # ((is_verified OR has_phone) AND NOT is_banned)
        complex_rule = (
            rule("account_access")
            .when(
                AND(
                    AND(
                        data.age >= 18,
                        data.has_account == True
                    ),
                    AND(
                        OR(
                            data.is_verified == True,
                            data.has_phone == True
                        ),
                        NOT(data.is_banned == True)
                    )
                )
            )
            .then("Account access requirements not met")
            .build()
        )

        engine.add_rule(complex_rule)

        # Test valid scenario
        passed, _ = engine.validate({
            "age": 25,
            "has_account": True,
            "is_verified": True,
            "has_phone": False,
            "is_banned": False
        })
        assert passed is True

        # Test: age OK but banned
        passed, _ = engine.validate({
            "age": 25,
            "has_account": True,
            "is_verified": True,
            "has_phone": False,
            "is_banned": True
        })
        assert passed is False

        # Test: verified but no phone (OR should pass)
        passed, _ = engine.validate({
            "age": 25,
            "has_account": True,
            "is_verified": False,
            "has_phone": True,
            "is_banned": False
        })
        assert passed is True

    def test_rule_analytics(self):
        """Test rule engine analytics."""
        engine = RuleEngineV2()

        # Add rules with tags
        engine.add_rule(
            rule("rule1")
            .when(data.value > 0)
            .then("Rule 1")
            .tag("validation", "numeric")
            .build()
        )

        engine.add_rule(
            rule("rule2")
            .when(data.status == "active")
            .then("Rule 2")
            .tag("validation", "status")
            .build()
        )

        # Run multiple validations
        for i in range(20):
            engine.validate({"value": i % 10 - 5, "status": "active" if i % 2 == 0 else "inactive"})

        # Get analytics
        analytics = engine.get_analytics()

        assert analytics["total_rules"] == 2
        assert analytics["enabled_rules"] == 2
        assert len(analytics["slowest_rules"]) > 0
        assert len(analytics["most_failing_rules"]) > 0

    def test_rule_tagging_and_filtering(self):
        """Test filtering rules by tags."""
        engine = RuleEngineV2()

        # Add rules with different tags
        engine.add_rule(
            rule("user_age")
            .when(data.age >= 18)
            .then("Age check")
            .tag("user", "age")
            .build()
        )

        engine.add_rule(
            rule("user_email")
            .when(data.email.exists())
            .then("Email check")
            .tag("user", "contact")
            .build()
        )

        engine.add_rule(
            rule("payment_amount")
            .when(data.amount > 0)
            .then("Payment check")
            .tag("payment", "amount")
            .build()
        )

        # Validate only user rules
        passed, errors = engine.validate(
            {"age": 25, "email": "test@example.com", "amount": -10},
            tags=["user"]
        )
        # Should pass user validation even though payment would fail
        assert passed is True

        # Validate only payment rules
        passed, errors = engine.validate(
            {"age": 15, "email": None, "amount": -10},
            tags=["payment"]
        )
        # Should fail payment validation
        assert passed is False

    def test_fail_fast_optimization(self):
        """Test fail-fast mode stops at first error."""
        engine = RuleEngineV2()

        evaluation_count = []

        def make_counting_rule(name):
            def condition(data):
                evaluation_count.append(name)
                return data.get(name, False)
            return RuleV2(
                name=name,
                condition=condition,
                error_message=f"{name} failed"
            )

        # Add multiple rules that will fail
        for i in range(10):
            engine.add_rule(make_counting_rule(f"rule_{i}"))

        # Run with fail_fast
        evaluation_count.clear()
        passed, errors = engine.validate({}, fail_fast=True)

        # Should stop after first failure
        assert len(evaluation_count) == 1
        assert len(errors) == 1

        # Run without fail_fast
        evaluation_count.clear()
        passed, errors = engine.validate({}, fail_fast=False)

        # Should evaluate all rules
        assert len(evaluation_count) == 10
        assert len(errors) == 10

    def test_real_world_user_registration_scenario(self):
        """Test real-world user registration validation."""
        manager = RuleManager()

        # Define comprehensive user registration rules
        manager.add_rule(
            rule("age_requirement")
            .when(data.age >= 18)
            .then("You must be at least 18 years old to register")
            .severity(ERROR)
            .priority(100)
            .tag("user_registration", "age")
            .build(),
            created_by="security_team"
        )

        manager.add_rule(
            rule("email_format")
            .when(
                AND(
                    data.email.exists(),
                    data.email.contains("@"),
                    data.email.contains(".")
                )
            )
            .then("Please provide a valid email address")
            .severity(ERROR)
            .priority(90)
            .tag("user_registration", "contact")
            .build(),
            created_by="backend_team"
        )

        manager.add_rule(
            rule("username_length")
            .when(
                AND(
                    data.username.exists(),
                    data["username_length"] >= 3
                )
            )
            .then("Username must be at least 3 characters long")
            .severity(ERROR)
            .priority(80)
            .tag("user_registration", "username")
            .build(),
            created_by="product_team"
        )

        manager.add_rule(
            rule("terms_acceptance")
            .when(data.accepted_terms == True)
            .then("You must accept the terms and conditions")
            .severity(ERROR)
            .priority(70)
            .tag("user_registration", "legal")
            .build(),
            created_by="legal_team"
        )

        # Test valid registration
        valid_user = {
            "age": 25,
            "email": "john.doe@example.com",
            "username": "johndoe",
            "username_length": 7,
            "accepted_terms": True
        }

        passed, errors = manager.engine.validate(valid_user)
        assert passed is True
        assert len(errors) == 0

        # Test invalid registration (multiple failures)
        invalid_user = {
            "age": 16,  # Too young
            "email": "invalid-email",  # Invalid format
            "username": "ab",  # Too short
            "username_length": 2,
            "accepted_terms": False  # Not accepted
        }

        passed, errors = manager.engine.validate(invalid_user)
        assert passed is False
        assert len(errors) == 4  # All rules should fail

        # Get statistics
        stats = manager.get_statistics()
        assert stats["total_rules"] == 4
