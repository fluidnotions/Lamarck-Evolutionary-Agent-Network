"""Advanced workflow example with multiple validators and complex data.

This example demonstrates:
1. Complex nested data validation
2. Running multiple validators together
3. Comprehensive error reporting
4. Advanced configuration options
"""

from src.graph.workflow import ValidationWorkflow


def main():
    """Run advanced workflow example."""
    print("=" * 70)
    print("HVAS-Mini: Advanced Workflow Example")
    print("=" * 70)

    # Complex nested data structure
    data = {
        "transaction": {
            "id": "TXN-2024-12345",
            "type": "purchase",
            "timestamp": "2024-01-15T14:30:00Z",
            "amount": {
                "currency": "USD",
                "value": 1250.00,
                "tax": 100.00,
                "total": 1350.00,
            },
        },
        "customer": {
            "id": "CUST-67890",
            "profile": {
                "first_name": "Alice",
                "last_name": "Johnson",
                "email": "alice.johnson@example.com",
                "phone": "+1-555-0123",
                "age": 32,
            },
            "address": {
                "street": "789 Pine Street",
                "city": "Seattle",
                "state": "WA",
                "zip": "98101",
                "country": "USA",
            },
            "loyalty": {
                "tier": "gold",
                "points": 5420,
                "member_since": "2020-03-15",
            },
        },
        "items": [
            {
                "id": "ITEM-001",
                "name": "Premium Widget",
                "category": "electronics",
                "quantity": 2,
                "unit_price": 500.00,
                "subtotal": 1000.00,
                "discount": 0.00,
            },
            {
                "id": "ITEM-002",
                "name": "Extended Warranty",
                "category": "services",
                "quantity": 1,
                "unit_price": 250.00,
                "subtotal": 250.00,
                "discount": 0.00,
            },
        ],
        "payment": {
            "method": "credit_card",
            "processor": "stripe",
            "status": "authorized",
            "card": {
                "brand": "visa",
                "last_four": "4242",
                "expiry": "12/2025",
            },
        },
        "shipping": {
            "method": "express",
            "carrier": "FedEx",
            "tracking": "1234567890",
            "estimated_delivery": "2024-01-17",
        },
    }

    # Comprehensive validation configuration
    config = {
        "schema": {
            "schema": {
                "type": "object",
                "required": ["transaction", "customer", "items", "payment"],
                "properties": {
                    "transaction": {
                        "type": "object",
                        "required": ["id", "type", "amount"],
                    },
                    "customer": {
                        "type": "object",
                        "required": ["id", "profile", "address"],
                    },
                    "items": {
                        "type": "array",
                        "minItems": 1,
                    },
                    "payment": {
                        "type": "object",
                        "required": ["method", "status"],
                    },
                },
            }
        },
        "data_quality": {
            "required_fields": [
                "transaction",
                "customer",
                "items",
                "payment",
            ],
            "patterns": {
                "customer.profile.email": r"^[\w\.-]+@[\w\.-]+\.\w+$",
                "customer.address.zip": r"^\d{5}$",
            },
            "ranges": {
                "customer.profile.age": {"min": 18, "max": 120},
                "transaction.amount.value": {"min": 0, "max": 1000000},
            },
        },
    }

    # Create and run workflow
    print("\n1. Creating workflow with multiple validators...")
    workflow = ValidationWorkflow()

    print("2. Running comprehensive validation...")
    print("   - Schema validation")
    print("   - Business rules validation")
    print("   - Data quality validation")

    result = workflow.run(
        data=data,
        validators=["schema", "business", "quality"],
        config=config,
    )

    # Detailed results
    print("\n3. Validation Results:")
    print("=" * 70)
    print(f"Overall Status: {result.overall_status.upper()}")
    print(f"Confidence Score: {result.confidence_score:.2%}")
    print(f"Execution Time: {result.execution_time_ms:.2f}ms")
    print()
    print(f"Summary: {result.summary}")
    print()

    # Validator breakdown
    print("Validator Breakdown:")
    print("-" * 70)
    for val_result in result.validation_results:
        status_icon = "✅" if val_result.status == "passed" else "❌"
        print(f"\n{status_icon} {val_result.validator_name.upper()}")
        print(f"   Status: {val_result.status}")
        print(f"   Confidence: {val_result.confidence:.2%}")
        print(f"   Execution Time: {val_result.execution_time_ms:.2f}ms")

        if val_result.errors:
            print(f"   Errors: {len(val_result.errors)}")
            for i, error in enumerate(val_result.errors[:5], 1):
                print(f"      {i}. [{error.code}] {error.path}: {error.message}")
            if len(val_result.errors) > 5:
                print(f"      ... and {len(val_result.errors) - 5} more errors")

        if val_result.warnings:
            print(f"   Warnings: {len(val_result.warnings)}")
            for i, warning in enumerate(val_result.warnings[:3], 1):
                print(f"      {i}. {warning.path}: {warning.message}")

    # Error summary
    if result.total_errors > 0:
        print("\n" + "=" * 70)
        print("ERROR SUMMARY")
        print("=" * 70)
        all_errors = result.get_all_errors()
        error_by_severity = {}
        for error in all_errors:
            severity = error.severity
            if severity not in error_by_severity:
                error_by_severity[severity] = []
            error_by_severity[severity].append(error)

        for severity, errors in error_by_severity.items():
            print(f"\n{severity.upper()}: {len(errors)} issues")
            for error in errors[:5]:
                print(f"  - {error.path}: {error.message}")

    # Generate comprehensive report
    print("\n4. Generating comprehensive report...")
    markdown_report = result.get_report("markdown")
    with open("advanced_validation_report.md", "w") as f:
        f.write(markdown_report)
    print("   ✓ Report saved to: advanced_validation_report.md")

    print("\n" + "=" * 70)
    print("Advanced workflow example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
