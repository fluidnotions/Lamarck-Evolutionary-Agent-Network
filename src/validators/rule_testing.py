"""Rule testing framework with test generation and coverage analysis."""
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
import random

from src.validators.rule_engine_v2 import RuleV2, RuleEngineV2

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a test case for a rule."""

    name: str
    description: str
    data: Dict[str, Any]
    should_pass: bool
    reason: str = ""
    tags: Set[str] = field(default_factory=set)


@dataclass
class TestResult:
    """Result of running a test case."""

    test_case: TestCase
    passed: bool
    actual_pass: bool
    execution_time_ms: float
    error_message: Optional[str] = None


@dataclass
class CoverageReport:
    """Coverage report for rule testing."""

    total_rules: int
    tested_rules: int
    untested_rules: List[str]
    coverage_percentage: float
    test_results: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""

    rule_name: str
    num_evaluations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    evaluations_per_second: float


class RuleTestFramework:
    """Framework for testing and analyzing rules."""

    def __init__(self, engine: RuleEngineV2):
        """Initialize test framework.

        Args:
            engine: Rule engine to test
        """
        self.engine = engine
        self.test_cases: Dict[str, List[TestCase]] = {}  # rule_name -> test cases
        self.test_results: List[TestResult] = []

    def add_test_case(self, rule_name: str, test_case: TestCase) -> None:
        """Add a test case for a rule.

        Args:
            rule_name: Name of rule to test
            test_case: Test case to add
        """
        if rule_name not in self.test_cases:
            self.test_cases[rule_name] = []
        self.test_cases[rule_name].append(test_case)

    def generate_test_cases(
        self,
        rule_name: str,
        data_generator: Callable[[], Dict[str, Any]],
        num_cases: int = 10,
    ) -> List[TestCase]:
        """Generate test cases using a data generator.

        Args:
            rule_name: Rule to generate tests for
            data_generator: Function that generates test data
            num_cases: Number of cases to generate

        Returns:
            List of generated test cases
        """
        generated_cases = []

        for i in range(num_cases):
            data = data_generator()
            test_case = TestCase(
                name=f"generated_{rule_name}_{i}",
                description=f"Generated test case {i}",
                data=data,
                should_pass=True,  # Will be determined by running
            )
            generated_cases.append(test_case)
            self.add_test_case(rule_name, test_case)

        return generated_cases

    def run_test(self, rule: RuleV2, test_case: TestCase) -> TestResult:
        """Run a single test case against a rule.

        Args:
            rule: Rule to test
            test_case: Test case to run

        Returns:
            TestResult
        """
        start_time = time.perf_counter()

        try:
            passed, error = rule.evaluate(test_case.data)
            execution_time = (time.perf_counter() - start_time) * 1000

            # Check if result matches expectation
            test_passed = (passed == test_case.should_pass)

            result = TestResult(
                test_case=test_case,
                passed=test_passed,
                actual_pass=passed,
                execution_time_ms=execution_time,
                error_message=error.message if error else None,
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            result = TestResult(
                test_case=test_case,
                passed=False,
                actual_pass=False,
                execution_time_ms=execution_time,
                error_message=str(e),
            )

        self.test_results.append(result)
        return result

    def run_all_tests(self, rule_names: Optional[List[str]] = None) -> List[TestResult]:
        """Run all test cases for specified rules.

        Args:
            rule_names: Specific rules to test (None for all)

        Returns:
            List of test results
        """
        results = []
        rules_to_test = rule_names or list(self.test_cases.keys())

        for rule_name in rules_to_test:
            if rule_name not in self.engine.rules:
                logger.warning(f"Rule {rule_name} not found in engine")
                continue

            rule = self.engine.rules[rule_name]
            test_cases = self.test_cases.get(rule_name, [])

            for test_case in test_cases:
                result = self.run_test(rule, test_case)
                results.append(result)

        return results

    def get_coverage_report(self) -> CoverageReport:
        """Generate coverage report showing which rules have tests.

        Returns:
            CoverageReport
        """
        all_rules = set(self.engine.rules.keys())
        tested_rules = set(self.test_cases.keys())
        untested_rules = list(all_rules - tested_rules)

        total_rules = len(all_rules)
        tested_count = len(tested_rules)
        coverage_pct = (tested_count / total_rules * 100) if total_rules > 0 else 0

        total_tests = len(self.test_results)
        passed = sum(1 for r in self.test_results if r.passed)
        failed = total_tests - passed

        return CoverageReport(
            total_rules=total_rules,
            tested_rules=tested_count,
            untested_rules=untested_rules,
            coverage_percentage=coverage_pct,
            test_results=self.test_results,
            total_tests=total_tests,
            passed_tests=passed,
            failed_tests=failed,
        )

    def benchmark_rule(
        self,
        rule_name: str,
        test_data: List[Dict[str, Any]],
        num_iterations: int = 1000,
    ) -> PerformanceBenchmark:
        """Benchmark a rule's performance.

        Args:
            rule_name: Rule to benchmark
            test_data: List of test data to use
            num_iterations: Number of times to run each test

        Returns:
            PerformanceBenchmark results
        """
        if rule_name not in self.engine.rules:
            raise ValueError(f"Rule {rule_name} not found")

        rule = self.engine.rules[rule_name]
        times = []

        for _ in range(num_iterations):
            # Pick random test data
            data = random.choice(test_data) if test_data else {}

            start_time = time.perf_counter()
            rule.evaluate(data)
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)

        # Calculate statistics
        total_time = sum(times)
        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        # Standard deviation
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5

        # Evaluations per second
        eps = 1000 / avg_time if avg_time > 0 else 0

        return PerformanceBenchmark(
            rule_name=rule_name,
            num_evaluations=len(times),
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            std_dev_ms=std_dev,
            evaluations_per_second=eps,
        )

    def benchmark_engine(
        self,
        test_data: List[Dict[str, Any]],
        num_iterations: int = 1000,
    ) -> Dict[str, Any]:
        """Benchmark the entire engine's performance.

        Args:
            test_data: List of test data to use
            num_iterations: Number of iterations

        Returns:
            Dictionary with benchmark results
        """
        times = []

        for _ in range(num_iterations):
            data = random.choice(test_data) if test_data else {}

            start_time = time.perf_counter()
            self.engine.validate(data)
            elapsed = (time.perf_counter() - start_time) * 1000
            times.append(elapsed)

        total_time = sum(times)
        avg_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        std_dev = variance ** 0.5

        validations_per_second = 1000 / avg_time if avg_time > 0 else 0

        return {
            "num_rules": len(self.engine.rules),
            "num_evaluations": len(times),
            "total_time_ms": total_time,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "std_dev_ms": std_dev,
            "validations_per_second": validations_per_second,
            "meets_10ms_target": avg_time < 10.0,
        }

    def mutation_test_rule(
        self,
        rule_name: str,
        mutations: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Perform mutation testing on a rule.

        Args:
            rule_name: Rule to test
            mutations: List of mutation functions to apply to test data

        Returns:
            Dictionary with mutation test results
        """
        if rule_name not in self.engine.rules:
            raise ValueError(f"Rule {rule_name} not found")

        if rule_name not in self.test_cases:
            return {
                "rule_name": rule_name,
                "error": "No test cases found",
            }

        rule = self.engine.rules[rule_name]
        test_cases = self.test_cases[rule_name]

        results = {
            "rule_name": rule_name,
            "mutations_tested": 0,
            "mutations_detected": 0,
            "mutations_survived": 0,
            "mutation_score": 0.0,
        }

        for test_case in test_cases:
            original_pass, _ = rule.evaluate(test_case.data)

            for mutation in mutations:
                results["mutations_tested"] += 1

                # Apply mutation
                mutated_data = mutation(test_case.data.copy())

                # Evaluate mutated data
                mutated_pass, _ = rule.evaluate(mutated_data)

                # Check if mutation was detected (result changed)
                if mutated_pass != original_pass:
                    results["mutations_detected"] += 1
                else:
                    results["mutations_survived"] += 1

        # Calculate mutation score
        if results["mutations_tested"] > 0:
            results["mutation_score"] = (
                results["mutations_detected"] / results["mutations_tested"]
            )

        return results

    def generate_edge_cases(self, rule_name: str) -> List[TestCase]:
        """Generate edge case test data for a rule.

        Args:
            rule_name: Rule to generate edge cases for

        Returns:
            List of edge case test cases
        """
        edge_cases = []

        # Common edge cases
        edge_data = [
            ({}, "empty_dict"),
            ({"field": None}, "null_value"),
            ({"field": ""}, "empty_string"),
            ({"field": []}, "empty_list"),
            ({"field": 0}, "zero_value"),
            ({"field": -1}, "negative_value"),
            ({"field": float('inf')}, "infinity"),
            ({"field": float('nan')}, "nan"),
        ]

        for data, name in edge_data:
            test_case = TestCase(
                name=f"{rule_name}_edge_{name}",
                description=f"Edge case: {name}",
                data=data,
                should_pass=False,  # Edge cases typically fail
                tags={"edge_case"},
            )
            edge_cases.append(test_case)
            self.add_test_case(rule_name, test_case)

        return edge_cases

    def analyze_failures(self) -> Dict[str, Any]:
        """Analyze failed test results to find patterns.

        Returns:
            Dictionary with failure analysis
        """
        failed_results = [r for r in self.test_results if not r.passed]

        if not failed_results:
            return {
                "total_failures": 0,
                "message": "No failures to analyze",
            }

        # Group by rule
        failures_by_rule: Dict[str, List[TestResult]] = {}
        for result in failed_results:
            rule_name = result.test_case.name.split('_')[0]
            if rule_name not in failures_by_rule:
                failures_by_rule[rule_name] = []
            failures_by_rule[rule_name].append(result)

        # Most common failure reasons
        failure_reasons: Dict[str, int] = {}
        for result in failed_results:
            reason = result.error_message or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        return {
            "total_failures": len(failed_results),
            "failures_by_rule": {
                name: len(results) for name, results in failures_by_rule.items()
            },
            "common_failure_reasons": sorted(
                failure_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "slowest_failures": sorted(
                failed_results,
                key=lambda r: r.execution_time_ms,
                reverse=True
            )[:5],
        }

    def export_results(self) -> Dict[str, Any]:
        """Export all test results to a dictionary.

        Returns:
            Dictionary with all test data
        """
        coverage = self.get_coverage_report()

        return {
            "exported_at": datetime.now().isoformat(),
            "coverage": {
                "total_rules": coverage.total_rules,
                "tested_rules": coverage.tested_rules,
                "coverage_percentage": coverage.coverage_percentage,
                "untested_rules": coverage.untested_rules,
            },
            "test_results": {
                "total": coverage.total_tests,
                "passed": coverage.passed_tests,
                "failed": coverage.failed_tests,
                "pass_rate": (coverage.passed_tests / coverage.total_tests * 100)
                if coverage.total_tests > 0 else 0,
            },
            "failure_analysis": self.analyze_failures(),
        }
