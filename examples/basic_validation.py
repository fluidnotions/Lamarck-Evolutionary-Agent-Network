"""Basic example of using HVAS-Mini domain validators."""

from pydantic import BaseModel, Field

from src.agents.schema_validator import SchemaValidatorAgent
from src.agents.data_quality import DataQualityAgent
from src.agents.business_rules import BusinessRulesAgent, RuleEngine, Rule, create_common_rules
from src.agents.cross_reference import CrossReferenceAgent
from src.graph.state import ValidationState


class User(BaseModel):
    """Example Pydantic model for a user."""

    name: str = Field(..., min_length=1)
    email: str
    age: int = Field(..., ge=0, le=150)


def main() -> None:
    """Run basic validation examples."""
    print("=" * 60)
    print("HVAS-Mini Domain Validators Example")
    print("=" * 60)

    # Example 1: Schema Validation with Pydantic
    print("\n1. Schema Validation (Pydantic)")
    print("-" * 60)

    schema_validator = SchemaValidatorAgent()

    user_data = {"name": "Alice Smith", "email": "alice@example.com", "age": 30}

    state: ValidationState = {
        "input_data": user_data,
        "validation_request": {"schema": User},
        "validation_results": [],
        "completed_validators": [],
    }

    result = schema_validator.execute(state)
    domain_result = result["validation_results"][0]

    print(f"Status: {domain_result.overall_status}")
    print(f"Summary: {domain_result.summary}")
    print(f"Confidence: {domain_result.confidence_score:.2%}")

    # Example 2: Data Quality Validation
    print("\n2. Data Quality Validation")
    print("-" * 60)

    data_quality = DataQualityAgent()

    state2: ValidationState = {
        "input_data": user_data,
        "validation_request": {
            "quality_config": {
                "required_fields": ["name", "email", "age"],
                "field_specs": {
                    "email": {"type": "string", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                    "age": {"type": "integer", "min": 0, "max": 150},
                },
            }
        },
        "validation_results": [],
        "completed_validators": [],
    }

    result2 = data_quality.execute(state2)
    domain_result2 = result2["validation_results"][0]

    print(f"Status: {domain_result2.overall_status}")
    print(f"Summary: {domain_result2.summary}")

    for individual_result in domain_result2.individual_results:
        print(f"  - {individual_result.validator_name}: {individual_result.status}")

    # Example 3: Business Rules Validation
    print("\n3. Business Rules Validation")
    print("-" * 60)

    rule_engine = RuleEngine()

    # Add common rules
    for rule in create_common_rules():
        rule_engine.add_rule(rule)

    # Add a custom rule
    rule_engine.add_rule(
        Rule(
            rule_id="valid_name_length",
            name="Valid Name Length",
            description="Name should have at least 2 characters",
            rule_type="constraint",
            condition=lambda data: len(data.get("name", "")) >= 2,
        )
    )

    business_rules = BusinessRulesAgent(rule_engine=rule_engine)

    state3: ValidationState = {
        "input_data": user_data,
        "validation_request": {},
        "validation_results": [],
        "completed_validators": [],
    }

    result3 = business_rules.execute(state3)
    domain_result3 = result3["validation_results"][0]

    print(f"Status: {domain_result3.overall_status}")
    print(f"Summary: {domain_result3.summary}")
    print(f"Rules evaluated: {len(domain_result3.individual_results)}")

    # Example 4: Cross-Reference Validation
    print("\n4. Cross-Reference Validation")
    print("-" * 60)

    cross_reference = CrossReferenceAgent()

    # Simulated data with references
    order_data = {
        "order_id": 100,
        "user_id": 1,
        "product_ids": [10, 20, 30],
    }

    reference_data = {
        "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        "products": [{"id": 10, "name": "Item A"}, {"id": 20, "name": "Item B"}],
    }

    state4: ValidationState = {
        "input_data": order_data,
        "validation_request": {
            "cross_reference_config": {
                "foreign_keys": [
                    {
                        "field": "user_id",
                        "reference_table": "users",
                        "reference_field": "id",
                    }
                ],
                "cardinality": [{"field": "product_ids", "min": 1, "max": 10}],
            }
        },
        "metadata": {"reference_data": reference_data},
        "validation_results": [],
        "completed_validators": [],
    }

    result4 = cross_reference.execute(state4)
    domain_result4 = result4["validation_results"][0]

    print(f"Status: {domain_result4.overall_status}")
    print(f"Summary: {domain_result4.summary}")

    for individual_result in domain_result4.individual_results:
        print(f"  - {individual_result.validator_name}: {individual_result.status}")
        if individual_result.status != "passed":
            print(f"    Message: {individual_result.message}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
