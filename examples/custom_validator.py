"""Example showing how to create custom validators.

This example demonstrates:
1. Creating custom validation rules
2. Adding rules to the rule engine
3. Running validation with custom rules
"""

from src.graph.workflow import ValidationWorkflow
from src.validators.rule_engine import Rule


def main():
    """Run custom validator example."""
    print("=" * 60)
    print("HVAS-Mini: Custom Validator Example")
    print("=" * 60)

    # Sample e-commerce order data
    data = {
        "order_id": "ORD-2024-001",
        "customer_age": 25,
        "total_amount": 150.00,
        "discount": 15.00,
        "final_amount": 135.00,
        "items_count": 3,
        "shipping_method": "express",
        "payment_status": "pending",
    }

    print("\nData to validate:")
    for key, value in data.items():
        print(f"  {key}: {value}")

    # Create workflow
    workflow = ValidationWorkflow()

    # Add custom business rules
    print("\nAdding custom business rules...")

    # Rule 1: Customer must be 18 or older
    age_rule = Rule(
        name="minimum_age",
        condition=lambda d: d.get("customer_age", 0) >= 18,
        error_message="Customer must be at least 18 years old",
        severity="error",
    )
    workflow.business_rules.add_rule(age_rule)
    print("  ✓ Added minimum age rule")

    # Rule 2: Discount cannot exceed 20% of total
    discount_rule = Rule(
        name="discount_limit",
        condition=lambda d: d.get("discount", 0) <= d.get("total_amount", 0) * 0.2,
        error_message="Discount cannot exceed 20% of total amount",
        severity="error",
    )
    workflow.business_rules.add_rule(discount_rule)
    print("  ✓ Added discount limit rule")

    # Rule 3: Final amount should equal total minus discount
    calculation_rule = Rule(
        name="amount_calculation",
        condition=lambda d: abs(
            d.get("final_amount", 0) - (d.get("total_amount", 0) - d.get("discount", 0))
        ) < 0.01,
        error_message="Final amount calculation is incorrect",
        severity="error",
    )
    workflow.business_rules.add_rule(calculation_rule)
    print("  ✓ Added amount calculation rule")

    # Rule 4: Express shipping requires minimum order
    shipping_rule = Rule(
        name="express_shipping_minimum",
        condition=lambda d: (
            d.get("shipping_method") != "express" or d.get("final_amount", 0) >= 50.00
        ),
        error_message="Express shipping requires minimum order of $50",
        severity="warning",
    )
    workflow.business_rules.add_rule(shipping_rule)
    print("  ✓ Added express shipping rule")

    # Configure validation
    config = {
        "business_rules": {
            "rules": ["minimum_age", "discount_limit", "amount_calculation", "express_shipping_minimum"]
        }
    }

    # Run validation
    print("\nRunning validation with custom rules...")
    result = workflow.run(
        data=data,
        validators=["business"],
        config=config,
    )

    # Display results
    print("\nValidation Results:")
    print("-" * 60)
    print(f"Overall Status: {result.overall_status.upper()}")
    print(f"Confidence Score: {result.confidence_score:.2%}")

    for val_result in result.validation_results:
        print(f"\n{val_result.validator_name}:")

        if val_result.errors:
            print(f"  Errors ({len(val_result.errors)}):")
            for error in val_result.errors:
                print(f"    ❌ {error.message}")

        if val_result.warnings:
            print(f"  Warnings ({len(val_result.warnings)}):")
            for warning in val_result.warnings:
                print(f"    ⚠️  {warning.message}")

        if not val_result.errors and not val_result.warnings:
            print("  ✅ All rules passed!")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
