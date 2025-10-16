"""Basic validation example showing HVAS-Mini usage.

This example demonstrates how to:
1. Create a validation workflow
2. Run validation on sample data
3. Review validation results
4. Generate validation reports
"""

from src.graph.workflow import ValidationWorkflow


def main():
    """Run basic validation example."""
    print("=" * 60)
    print("HVAS-Mini: Basic Validation Example")
    print("=" * 60)

    # Sample data to validate
    data = {
        "user": {
            "name": "John Doe",
            "email": "john@example.com",
            "age": 30,
        },
        "order": {
            "id": "ORD-123",
            "total": 99.99,
            "items": [
                {"product": "Widget", "quantity": 2, "price": 49.99}
            ],
        },
    }

    # Configuration for validators
    config = {
        "schema": {
            "schema": {
                "type": "object",
                "required": ["user", "order"],
                "properties": {
                    "user": {
                        "type": "object",
                        "required": ["name", "email", "age"],
                    },
                    "order": {
                        "type": "object",
                        "required": ["id", "total"],
                    },
                },
            }
        },
        "data_quality": {
            "required_fields": ["user", "order"],
            "types": {
                "user.name": "str",
                "user.age": "int",
            },
        },
    }

    # Create workflow
    print("\n1. Creating validation workflow...")
    workflow = ValidationWorkflow()

    # Run validation
    print("2. Running validation...")
    result = workflow.run(
        data=data,
        validators=["schema", "quality"],
        config=config,
    )

    # Print results
    print("\n3. Validation Results:")
    print("-" * 60)
    print(f"Overall Status: {result.overall_status.upper()}")
    print(f"Confidence Score: {result.confidence_score:.2%}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms")
    print(f"Total Errors: {result.total_errors}")
    print(f"Total Warnings: {result.total_warnings}")

    print("\nValidator Results:")
    for val_result in result.validation_results:
        status_icon = "✅" if val_result.status == "passed" else "❌"
        print(f"  {status_icon} {val_result.validator_name}: {val_result.status}")

        if val_result.errors:
            for error in val_result.errors[:3]:
                print(f"      - {error.path}: {error.message}")

    # Generate and save report
    print("\n4. Generating reports...")

    # Text report
    text_report = result.get_report("text")
    with open("validation_report.txt", "w") as f:
        f.write(text_report)
    print("  - Text report saved to: validation_report.txt")

    # Markdown report
    markdown_report = result.get_report("markdown")
    with open("validation_report.md", "w") as f:
        f.write(markdown_report)
    print("  - Markdown report saved to: validation_report.md")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
