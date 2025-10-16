"""Streaming validation example.

This example demonstrates:
1. Streaming validation execution
2. Real-time progress monitoring
3. Incremental result processing
"""

from src.graph.workflow import ValidationWorkflow


def main():
    """Run streaming validation example."""
    print("=" * 60)
    print("HVAS-Mini: Streaming Validation Example")
    print("=" * 60)

    # Sample data
    data = {
        "records": [
            {"id": i, "name": f"Record-{i}", "value": i * 10}
            for i in range(50)
        ],
        "metadata": {
            "total_count": 50,
            "source": "batch-processor",
            "timestamp": "2024-01-15T10:00:00Z",
        },
    }

    # Configuration
    config = {
        "data_quality": {
            "required_fields": ["records", "metadata"],
        }
    }

    # Create workflow
    print("\n1. Creating streaming workflow...")
    workflow = ValidationWorkflow()

    # Stream validation execution
    print("2. Starting streaming validation...")
    print("   Monitoring real-time progress:\n")

    step_count = 0
    for state_update in workflow.stream(
        data=data,
        validators=["schema", "quality"],
        config=config,
    ):
        step_count += 1
        node_name = list(state_update.keys())[0] if state_update else "unknown"

        # Display progress
        print(f"   Step {step_count}: Processing {node_name}")

        # Check for completed validators
        if node_name in state_update:
            step_state = state_update[node_name]
            if "validation_results" in step_state:
                results = step_state.get("validation_results", [])
                if results:
                    latest_result = results[-1] if isinstance(results, list) else results
                    if hasattr(latest_result, "validator_name"):
                        status_icon = "✅" if latest_result.status == "passed" else "❌"
                        print(f"      {status_icon} {latest_result.validator_name}: {latest_result.status}")

    print(f"\n   Total steps: {step_count}")

    print("\n3. Streaming validation complete!")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
