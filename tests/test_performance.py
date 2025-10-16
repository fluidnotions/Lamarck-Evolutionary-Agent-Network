"""Performance benchmarks for validators."""

import asyncio
import time

import pytest

from src.validators.executor import execute_validators_parallel
from src.validators.json_schema import JSONSchemaValidator
from src.validators.quality_checks import QualityChecker
from src.validators.rule_engine import Rule, RuleEngine


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_json_schema_performance(self) -> None:
        """Test JSON schema validator performance."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "email": {"type": "string", "format": "email"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
            },
            "required": ["id", "name", "email"],
        }

        data = {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        }

        validator = JSONSchemaValidator(schema)

        start_time = time.time()
        for _ in range(100):
            result = validator.validate(data)
            assert result.status == "passed"
        elapsed = time.time() - start_time

        # Should complete 100 validations in less than 1 second
        assert elapsed < 1.0, f"JSON schema validation too slow: {elapsed:.3f}s"
        print(f"\nJSON Schema: 100 validations in {elapsed:.3f}s ({elapsed*10:.3f}ms per validation)")

    def test_rule_engine_performance(self) -> None:
        """Test rule engine performance with 100+ rules."""
        # Create 100 rules
        rules = []
        for i in range(100):
            rules.append(
                Rule(
                    name=f"rule_{i}",
                    condition=lambda d, idx=i: d.get(f"field_{idx}", 0) >= 0,
                    message=f"Field {i} must be non-negative",
                )
            )

        engine = RuleEngine(rules)

        # Create test data
        data = {f"field_{i}": i for i in range(100)}

        start_time = time.time()
        result = engine.evaluate(data)
        elapsed = time.time() - start_time

        assert result.status == "passed"
        # Should evaluate 100 rules in less than 1 second
        assert elapsed < 1.0, f"Rule engine too slow: {elapsed:.3f}s"
        print(f"\nRule Engine: 100 rules evaluated in {elapsed:.3f}s ({elapsed*10:.3f}ms per rule)")

    def test_quality_checks_performance(self) -> None:
        """Test quality check performance."""
        data = {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
            "status": "active",
        }

        start_time = time.time()

        for _ in range(100):
            # Run multiple quality checks
            QualityChecker.check_completeness(data, ["id", "name", "email", "age"])
            QualityChecker.check_format(data, "email", "email")
            QualityChecker.check_range(data, "age", min_value=0, max_value=150)
            QualityChecker.check_enum(data, "status", ["active", "inactive"])

        elapsed = time.time() - start_time

        # Should complete 400 checks (100 iterations × 4 checks) in less than 1 second
        assert elapsed < 1.0, f"Quality checks too slow: {elapsed:.3f}s"
        print(f"\nQuality Checks: 400 checks in {elapsed:.3f}s ({elapsed*2.5:.3f}ms per check)")

    def test_outlier_detection_performance(self) -> None:
        """Test outlier detection performance with large dataset."""
        # Generate 1000 values
        values = [float(i) for i in range(1000)]
        values.extend([10000.0, 10001.0])  # Add outliers

        start_time = time.time()
        result = QualityChecker.detect_outliers(values, method="zscore")
        elapsed = time.time() - start_time

        assert result.status == "passed"
        # Should detect outliers in 1000+ values in less than 0.1 seconds
        assert elapsed < 0.1, f"Outlier detection too slow: {elapsed:.3f}s"
        print(f"\nOutlier Detection: 1002 values analyzed in {elapsed:.3f}s")

    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self) -> None:
        """Test parallel execution speedup."""
        def slow_validator(data: dict):
            """Validator that takes time to complete."""
            time.sleep(0.1)  # Simulate work
            from src.models.validation_result import ValidationResult
            return ValidationResult(
                validator_name="slow_validator",
                status="passed",
                timing=0.1,
            )

        # Create 10 validators
        validators = [(slow_validator, ({},), {}) for _ in range(10)]

        # Test parallel execution
        start_time = time.time()
        results = await execute_validators_parallel(validators, timeout=5.0)
        parallel_elapsed = time.time() - start_time

        assert len(results) == 10
        assert all(r.status == "passed" for r in results)

        # Parallel execution should be significantly faster than sequential (10 × 0.1s = 1s)
        # With parallelism, should complete in around 0.1-0.2s
        assert parallel_elapsed < 0.5, f"Parallel execution not efficient: {parallel_elapsed:.3f}s"
        print(f"\nParallel Execution: 10 validators in {parallel_elapsed:.3f}s (expected ~0.1s with parallelism)")

        # Calculate speedup
        expected_sequential_time = 1.0  # 10 validators × 0.1s each
        speedup = expected_sequential_time / parallel_elapsed
        print(f"Speedup factor: {speedup:.1f}x")

    def test_large_dataset_validation(self) -> None:
        """Test validation of large dataset."""
        # Create a large dataset with 1000 records
        data = [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "age": 20 + (i % 50),
            }
            for i in range(1000)
        ]

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0, "maximum": 150},
                },
                "required": ["id", "name", "email", "age"],
            },
        }

        validator = JSONSchemaValidator(schema)

        start_time = time.time()
        result = validator.validate(data)
        elapsed = time.time() - start_time

        assert result.status == "passed"
        # Should validate 1000 records in less than 1 second
        assert elapsed < 1.0, f"Large dataset validation too slow: {elapsed:.3f}s"
        print(f"\nLarge Dataset: 1000 records validated in {elapsed:.3f}s ({elapsed:.3f}ms per record)")

    def test_complex_rule_dependencies_performance(self) -> None:
        """Test performance with complex rule dependencies."""
        # Create rules with dependencies
        rules = [
            Rule("base_rule", lambda d: True, "Base rule"),
        ]

        # Create 50 rules that depend on base_rule
        for i in range(50):
            rules.append(
                Rule(
                    name=f"dependent_rule_{i}",
                    condition=lambda d, idx=i: d.get(f"value_{idx}", 0) >= 0,
                    message=f"Rule {i}",
                    depends_on=["base_rule"],
                )
            )

        engine = RuleEngine(rules)
        data = {f"value_{i}": i for i in range(50)}

        start_time = time.time()
        result = engine.evaluate(data)
        elapsed = time.time() - start_time

        assert result.status == "passed"
        # Should handle dependencies efficiently
        assert elapsed < 0.5, f"Dependency resolution too slow: {elapsed:.3f}s"
        print(f"\nComplex Dependencies: 51 rules with dependencies evaluated in {elapsed:.3f}s")


if __name__ == "__main__":
    """Run benchmarks directly."""
    print("Running Performance Benchmarks...")
    print("=" * 60)

    benchmarks = TestPerformanceBenchmarks()

    print("\n1. JSON Schema Validator Performance")
    benchmarks.test_json_schema_performance()

    print("\n2. Rule Engine Performance (100+ rules)")
    benchmarks.test_rule_engine_performance()

    print("\n3. Quality Checks Performance")
    benchmarks.test_quality_checks_performance()

    print("\n4. Outlier Detection Performance")
    benchmarks.test_outlier_detection_performance()

    print("\n5. Parallel Execution Performance")
    asyncio.run(benchmarks.test_parallel_execution_performance())

    print("\n6. Large Dataset Validation")
    benchmarks.test_large_dataset_validation()

    print("\n7. Complex Rule Dependencies")
    benchmarks.test_complex_rule_dependencies_performance()

    print("\n" + "=" * 60)
    print("All benchmarks completed successfully!")
