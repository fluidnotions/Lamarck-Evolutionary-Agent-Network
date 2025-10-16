"""Tests for rule testing framework (rule_testing.py)."""
import pytest
from src.validators.rule_testing import (
    RuleTestFramework,
    TestCase,
    TestResult,
)
from src.validators.rule_engine_v2 import RuleV2, RuleEngineV2, Condition, ComparisonOperator


class TestRuleTestFramework:
    """Test RuleTestFramework class."""

    def test_add_test_case(self):
        """Test adding test cases."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        test_case = TestCase(
            name="test_positive",
            description="Value should be positive",
            data={"value": 10},
            should_pass=True
        )

        framework.add_test_case("test_rule", test_case)

        assert "test_rule" in framework.test_cases
        assert len(framework.test_cases["test_rule"]) == 1

    def test_run_single_test(self):
        """Test running a single test case."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        rule = RuleV2(
            name="positive_check",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        engine.add_rule(rule)

        test_case = TestCase(
            name="test_positive",
            description="Positive value",
            data={"value": 10},
            should_pass=True
        )

        result = framework.run_test(rule, test_case)

        assert result.passed is True
        assert result.actual_pass is True
        assert result.execution_time_ms > 0

    def test_run_failing_test(self):
        """Test running a failing test case."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        rule = RuleV2(
            name="positive_check",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        engine.add_rule(rule)

        test_case = TestCase(
            name="test_negative",
            description="Negative value should fail",
            data={"value": -5},
            should_pass=False
        )

        result = framework.run_test(rule, test_case)

        assert result.passed is True  # Test expects failure and got it
        assert result.actual_pass is False  # Rule failed as expected

    def test_run_all_tests(self):
        """Test running all tests."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        rule = RuleV2(
            name="positive_check",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        engine.add_rule(rule)

        # Add multiple test cases
        framework.add_test_case("positive_check", TestCase(
            name="test1",
            description="Positive",
            data={"value": 10},
            should_pass=True
        ))

        framework.add_test_case("positive_check", TestCase(
            name="test2",
            description="Negative",
            data={"value": -5},
            should_pass=False
        ))

        results = framework.run_all_tests(["positive_check"])

        assert len(results) == 2

    def test_coverage_report(self):
        """Test generating coverage report."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        # Add rules
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

        # Add tests only for rule1
        framework.add_test_case("rule1", TestCase(
            name="test1",
            description="Test",
            data={"a": 5},
            should_pass=True
        ))

        framework.run_all_tests(["rule1"])

        report = framework.get_coverage_report()

        assert report.total_rules == 2
        assert report.tested_rules == 1
        assert "rule2" in report.untested_rules
        assert report.coverage_percentage == 50.0

    def test_benchmark_rule(self):
        """Test benchmarking a rule."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        engine.add_rule(rule)

        test_data = [{"value": i} for i in range(10)]

        benchmark = framework.benchmark_rule("test_rule", test_data, num_iterations=100)

        assert benchmark.rule_name == "test_rule"
        assert benchmark.num_evaluations == 100
        assert benchmark.avg_time_ms > 0
        assert benchmark.evaluations_per_second > 0

    def test_benchmark_engine(self):
        """Test benchmarking entire engine."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        # Add multiple rules
        for i in range(5):
            engine.add_rule(RuleV2(
                name=f"rule{i}",
                condition=Condition("value", ComparisonOperator.GT, i),
                error_message=f"Rule {i}"
            ))

        test_data = [{"value": i} for i in range(10)]

        benchmark = framework.benchmark_engine(test_data, num_iterations=100)

        assert benchmark["num_rules"] == 5
        assert benchmark["num_evaluations"] == 100
        assert benchmark["avg_time_ms"] > 0
        assert "meets_10ms_target" in benchmark

    def test_generate_edge_cases(self):
        """Test generating edge cases."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        edge_cases = framework.generate_edge_cases("test_rule")

        assert len(edge_cases) > 0
        assert any("empty_dict" in tc.name for tc in edge_cases)
        assert any("null_value" in tc.name for tc in edge_cases)

    def test_analyze_failures(self):
        """Test analyzing test failures."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        engine.add_rule(rule)

        # Add tests that will fail expectation
        framework.add_test_case("test_rule", TestCase(
            name="wrong_expectation",
            description="Expected pass but will fail",
            data={"value": -5},
            should_pass=True  # Wrong expectation
        ))

        framework.run_all_tests(["test_rule"])

        analysis = framework.analyze_failures()

        assert analysis["total_failures"] > 0
        assert "failures_by_rule" in analysis

    def test_export_results(self):
        """Test exporting test results."""
        engine = RuleEngineV2()
        framework = RuleTestFramework(engine)

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        engine.add_rule(rule)

        framework.add_test_case("test_rule", TestCase(
            name="test1",
            description="Test",
            data={"value": 5},
            should_pass=True
        ))

        framework.run_all_tests(["test_rule"])

        export = framework.export_results()

        assert "coverage" in export
        assert "test_results" in export
        assert "failure_analysis" in export
