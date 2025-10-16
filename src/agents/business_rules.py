"""Business rules validation agent for HVAS-Mini."""

from typing import Any, Callable, Optional

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationResult, ValidationStatus


class Rule:
    """Represents a business rule."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        rule_type: str,
        condition: Callable[[dict[str, Any]], bool],
        severity: str = "error",
    ) -> None:
        """Initialize a business rule.

        Args:
            rule_id: Unique identifier for the rule
            name: Human-readable name
            description: Description of what the rule checks
            rule_type: Type of rule (constraint, derivation, inference)
            condition: Function that returns True if rule passes, False if violated
            severity: Severity level (error, warning, info)
        """
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.rule_type = rule_type
        self.condition = condition
        self.severity = severity


class RuleEngine:
    """Simple rule engine for business rule evaluation."""

    def __init__(self) -> None:
        """Initialize the rule engine."""
        self.rules: dict[str, Rule] = {}

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine.

        Args:
            rule: Rule to add
        """
        self.rules[rule.rule_id] = rule

    def evaluate(self, data: dict[str, Any], rule_ids: Optional[list[str]] = None) -> list[tuple[Rule, bool]]:
        """Evaluate rules against data.

        Args:
            data: Data to evaluate
            rule_ids: Optional list of specific rule IDs to evaluate

        Returns:
            List of (rule, passed) tuples
        """
        results: list[tuple[Rule, bool]] = []

        # Determine which rules to evaluate
        rules_to_eval = []
        if rule_ids:
            rules_to_eval = [self.rules[rid] for rid in rule_ids if rid in self.rules]
        else:
            rules_to_eval = list(self.rules.values())

        # Evaluate each rule
        for rule in rules_to_eval:
            try:
                passed = rule.condition(data)
                results.append((rule, passed))
            except Exception as e:
                # If evaluation fails, consider it a failure
                results.append((rule, False))

        return results


class BusinessRulesAgent(BaseAgent):
    """Validates business rules and constraints."""

    def __init__(
        self, llm: Optional[BaseChatModel] = None, rule_engine: Optional[RuleEngine] = None
    ) -> None:
        """Initialize the business rules agent.

        Args:
            llm: Language model for generating explanations
            rule_engine: Rule engine with loaded business rules
        """
        super().__init__(
            name="business_rules",
            description="Validates domain-specific business rules",
            llm=llm,
        )
        self.rule_engine = rule_engine or RuleEngine()

    def execute(self, state: ValidationState) -> ValidationState:
        """Evaluate business rules against input data.

        Steps:
        1. Load applicable rules from rule engine
        2. Evaluate each rule
        3. For violations, generate LLM explanations
        4. Calculate overall compliance score
        5. Update state with results

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        input_data = state.get("input_data", {})
        validation_request = state.get("validation_request", {})

        # Get specific rules to evaluate (if specified)
        rule_ids = validation_request.get("rule_ids")

        # Evaluate rules
        rule_results = self.rule_engine.evaluate(input_data, rule_ids)

        # Convert to validation results
        results: list[ValidationResult] = []
        for rule, passed in rule_results:
            result = self._create_validation_result(rule, passed, input_data)
            results.append(result)

        # If no rules were evaluated, add a skipped result
        if not results:
            results.append(
                ValidationResult(
                    validator_name="business_rules",
                    status=ValidationStatus.SKIPPED,
                    message="No business rules configured or evaluated",
                )
            )

        # Aggregate results
        domain_result = self._aggregate_results(results, "business_rules")

        # Update state
        new_state = self._update_state(state, domain_result)

        return new_state

    def _create_validation_result(
        self, rule: Rule, passed: bool, data: dict[str, Any]
    ) -> ValidationResult:
        """Create a validation result for a rule evaluation.

        Args:
            rule: The rule that was evaluated
            passed: Whether the rule passed
            data: The data that was evaluated

        Returns:
            Validation result
        """
        if passed:
            return ValidationResult(
                validator_name=f"rule_{rule.rule_id}",
                status=ValidationStatus.PASSED,
                message=f"Rule '{rule.name}' passed",
                details={
                    "rule_id": rule.rule_id,
                    "rule_name": rule.name,
                    "rule_type": rule.rule_type,
                    "description": rule.description,
                },
                confidence_score=1.0,
            )

        # Rule failed - determine status based on severity
        if rule.severity == "error":
            status = ValidationStatus.FAILED
        elif rule.severity == "warning":
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.WARNING

        # Generate explanation using LLM if available
        explanation = self._generate_violation_explanation(rule, data)

        return ValidationResult(
            validator_name=f"rule_{rule.rule_id}",
            status=status,
            message=f"Rule '{rule.name}' violated: {explanation}",
            details={
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "rule_type": rule.rule_type,
                "description": rule.description,
                "severity": rule.severity,
                "explanation": explanation,
            },
            suggestions=self._generate_suggestions(rule, data),
            confidence_score=0.8 if self.llm else 1.0,
        )

    def _generate_violation_explanation(self, rule: Rule, data: dict[str, Any]) -> str:
        """Generate a human-readable explanation for a rule violation.

        Uses LLM if available, otherwise returns the rule description.

        Args:
            rule: The rule that was violated
            data: The data that violated the rule

        Returns:
            Explanation of the violation
        """
        if not self.llm:
            return rule.description

        # Create a prompt for the LLM
        prompt = f"""You are a data validation expert. Explain why the following business rule was violated.

Rule Name: {rule.name}
Rule Description: {rule.description}
Rule Type: {rule.rule_type}

Data being validated:
{self._format_data_for_prompt(data)}

Please provide a clear, concise explanation of why this rule was violated and what the issue is.
Keep your explanation under 100 words and focus on the specific violation."""

        return self._get_llm_explanation(prompt)

    def _generate_suggestions(self, rule: Rule, data: dict[str, Any]) -> list[str]:
        """Generate suggestions for fixing a rule violation.

        Args:
            rule: The rule that was violated
            data: The data that violated the rule

        Returns:
            List of suggestions
        """
        suggestions: list[str] = []

        if not self.llm:
            # Provide generic suggestion
            suggestions.append(f"Ensure data complies with rule: {rule.description}")
            return suggestions

        # Use LLM to generate specific suggestions
        prompt = f"""You are a data validation expert. Provide specific, actionable suggestions for fixing a business rule violation.

Rule Name: {rule.name}
Rule Description: {rule.description}
Rule Type: {rule.rule_type}

Data being validated:
{self._format_data_for_prompt(data)}

Please provide 2-3 specific, actionable suggestions for fixing this violation.
Format as a simple list without numbering or bullets."""

        llm_response = self._get_llm_explanation(prompt)

        # Split the response into individual suggestions
        suggestion_lines = [line.strip() for line in llm_response.split("\n") if line.strip()]
        suggestions.extend(suggestion_lines[:3])  # Take up to 3 suggestions

        return suggestions

    def _format_data_for_prompt(self, data: dict[str, Any]) -> str:
        """Format data for inclusion in LLM prompt.

        Args:
            data: Data to format

        Returns:
            Formatted string representation
        """
        # Limit data size for prompt
        max_items = 10
        items = list(data.items())[:max_items]

        lines = []
        for key, value in items:
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            lines.append(f"  {key}: {value_str}")

        if len(data) > max_items:
            lines.append(f"  ... and {len(data) - max_items} more fields")

        return "\n".join(lines)


# Example rule definitions for common business rules
def create_common_rules() -> list[Rule]:
    """Create a set of common business rules.

    Returns:
        List of common business rules
    """
    rules = []

    # Age must be within valid range
    rules.append(
        Rule(
            rule_id="age_range",
            name="Age Range",
            description="Age must be between 0 and 150",
            rule_type="constraint",
            condition=lambda data: (
                "age" not in data or data["age"] is None or (0 <= data["age"] <= 150)
            ),
        )
    )

    # Email format validation
    rules.append(
        Rule(
            rule_id="email_format",
            name="Email Format",
            description="Email must contain @ symbol and domain",
            rule_type="constraint",
            condition=lambda data: (
                "email" not in data
                or data["email"] is None
                or ("@" in str(data["email"]) and "." in str(data["email"]))
            ),
        )
    )

    # Price must be positive
    rules.append(
        Rule(
            rule_id="positive_price",
            name="Positive Price",
            description="Price must be greater than 0",
            rule_type="constraint",
            condition=lambda data: (
                "price" not in data or data["price"] is None or data["price"] > 0
            ),
        )
    )

    # Start date must be before end date
    rules.append(
        Rule(
            rule_id="date_order",
            name="Date Order",
            description="Start date must be before end date",
            rule_type="constraint",
            condition=lambda data: (
                "start_date" not in data
                or "end_date" not in data
                or data["start_date"] is None
                or data["end_date"] is None
                or data["start_date"] < data["end_date"]
            ),
        )
    )

    # Quantity must be integer
    rules.append(
        Rule(
            rule_id="integer_quantity",
            name="Integer Quantity",
            description="Quantity must be a whole number",
            rule_type="constraint",
            condition=lambda data: (
                "quantity" not in data
                or data["quantity"] is None
                or isinstance(data["quantity"], int)
            ),
        )
    )

    return rules
