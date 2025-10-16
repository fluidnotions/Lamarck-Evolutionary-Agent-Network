#!/usr/bin/env python3
"""Performance benchmark for rule engine."""
import time
import random
from src.validators.rule_engine_v2 import RuleV2, RuleEngineV2
from src.validators.rule_dsl import rule, data, AND, OR, NOT
from src.validators.rule_testing import RuleTestFramework


def create_test_rules(num_rules: int) -> RuleEngineV2:
    """Create a large set of test rules."""
    engine = RuleEngineV2()

    # Add various types of rules
    for i in range(num_rules):
        if i % 5 == 0:
            # Simple comparison rules
            engine.add_rule(
                rule(f"rule_{i}")
                .when(data.value > i)
                .then(f"Value must be greater than {i}")
                .build()
            )
        elif i % 5 == 1:
            # AND conditions
            engine.add_rule(
                rule(f"rule_{i}")
                .when(
                    AND(
                        data.value > i,
                        data.status == "active"
                    )
                )
                .then(f"Rule {i} failed")
                .build()
            )
        elif i % 5 == 2:
            # OR conditions
            engine.add_rule(
                rule(f"rule_{i}")
                .when(
                    OR(
                        data.value > i,
                        data.alternative == True
                    )
                )
                .then(f"Rule {i} failed")
                .build()
            )
        elif i % 5 == 3:
            # NOT conditions
            engine.add_rule(
                rule(f"rule_{i}")
                .when(NOT(data.disabled == True))
                .then(f"Rule {i} failed")
                .build()
            )
        else:
            # Complex nested conditions
            engine.add_rule(
                rule(f"rule_{i}")
                .when(
                    AND(
                        data.value > i,
                        OR(
                            data.status == "active",
                            data.priority >= 5
                        ),
                        NOT(data.disabled == True)
                    )
                )
                .then(f"Rule {i} failed")
                .build()
            )

    return engine


def generate_test_data(num_samples: int) -> list:
    """Generate test data samples."""
    data_samples = []

    for _ in range(num_samples):
        sample = {
            "value": random.randint(0, 1000),
            "status": random.choice(["active", "inactive", "pending"]),
            "alternative": random.choice([True, False]),
            "disabled": random.choice([True, False]),
            "priority": random.randint(0, 10),
        }
        data_samples.append(sample)

    return data_samples


def benchmark_engine(engine: RuleEngineV2, test_data: list, num_iterations: int) -> dict:
    """Benchmark engine performance."""
    times = []

    print(f"\nRunning {num_iterations} iterations with {len(engine.rules)} rules...")

    for i in range(num_iterations):
        sample = random.choice(test_data)

        start = time.perf_counter()
        engine.validate(sample)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms

        times.append(elapsed)

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{num_iterations} iterations...")

    # Calculate statistics
    total_time = sum(times)
    avg_time = total_time / len(times)
    min_time = min(times)
    max_time = max(times)

    # Standard deviation
    variance = sum((t - avg_time) ** 2 for t in times) / len(times)
    std_dev = variance ** 0.5

    # Percentiles
    sorted_times = sorted(times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]

    # Validations per second
    vps = 1000 / avg_time if avg_time > 0 else 0

    return {
        "num_rules": len(engine.rules),
        "num_iterations": num_iterations,
        "total_time_ms": total_time,
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "std_dev_ms": std_dev,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "validations_per_second": vps,
        "meets_10ms_target": avg_time < 10.0,
    }


def print_results(results: dict) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Total Rules:     {results['num_rules']}")
    print(f"  Iterations:      {results['num_iterations']}")

    print(f"\nTiming Statistics:")
    print(f"  Average:         {results['avg_time_ms']:.4f} ms")
    print(f"  Minimum:         {results['min_time_ms']:.4f} ms")
    print(f"  Maximum:         {results['max_time_ms']:.4f} ms")
    print(f"  Std Deviation:   {results['std_dev_ms']:.4f} ms")

    print(f"\nPercentiles:")
    print(f"  50th (median):   {results['p50_ms']:.4f} ms")
    print(f"  95th:            {results['p95_ms']:.4f} ms")
    print(f"  99th:            {results['p99_ms']:.4f} ms")

    print(f"\nThroughput:")
    print(f"  Validations/sec: {results['validations_per_second']:.2f}")

    print(f"\nTarget Performance:")
    target_status = "✓ PASS" if results['meets_10ms_target'] else "✗ FAIL"
    print(f"  <10ms target:    {target_status}")

    print("=" * 70)


def main():
    """Run performance benchmarks."""
    print("Business Rules Engine Performance Benchmark")
    print("=" * 70)

    # Test with different rule counts
    test_configs = [
        (50, 1000),
        (100, 1000),
        (500, 1000),
        (1000, 1000),
    ]

    for num_rules, num_iterations in test_configs:
        print(f"\n\nBenchmark: {num_rules} rules")
        print("-" * 70)

        # Create engine with rules
        engine = create_test_rules(num_rules)

        # Generate test data
        test_data = generate_test_data(100)

        # Run benchmark
        results = benchmark_engine(engine, test_data, num_iterations)

        # Print results
        print_results(results)

        # Show top 5 slowest rules
        analytics = engine.get_analytics()
        slowest = analytics['slowest_rules'][:5]

        if slowest and any(r['evaluations'] > 0 for r in slowest):
            print("\nSlowest Rules:")
            for r in slowest:
                if r['evaluations'] > 0:
                    print(f"  {r['name']}: {r['avg_time_ms']:.4f} ms avg ({r['evaluations']} evals)")


if __name__ == "__main__":
    main()
