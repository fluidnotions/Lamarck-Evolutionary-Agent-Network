"""Business rules validator agent."""
from typing import Any, Dict
import logging

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.validators.rule_engine import RuleEngine, Rule

logger = logging.getLogger(__name__)


class BusinessRulesAgent(BaseAgent):
    """Agent that validates business rules."""

    def __init__(self, **kwargs: Any):
        """Initialize business rules agent."""
        super().__init__(name="business_rules", **kwargs)
        self.rule_engine = RuleEngine()
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default business rules."""
        # Example rules - these would be customized per use case
        self.rule_engine.add_rule(
            Rule(
                name="data_not_empty",
                condition=lambda data: bool(data),
                error_message="Data cannot be empty",
            )
        )

    def add_rule(self, rule: Rule) -> None:
        """Add a custom rule.

        Args:
            rule: Rule to add
        """
        self.rule_engine.add_rule(rule)

    def process(self, state: ValidationState) -> Dict[str, Any]:
        """Validate data against business rules.

        Args:
            state: Current validation state

        Returns:
            State updates with validation result
        """
        logger.info("Business rules validator processing")

        data = state["input_data"]
        config = state.get("config", {})
        rules_config = config.get("business_rules", {})

        # Get specific rules to run or run all
        rule_names = rules_config.get("rules")

        # Validate
        all_passed, errors = self.rule_engine.validate(data, rule_names)

        # Create result
        result = self.create_result(
            status="passed" if all_passed else "failed",
            confidence=1.0 if all_passed else max(0.0, 1.0 - (len(errors) * 0.2)),
            errors=errors,
            metadata={"rules_evaluated": len(self.rule_engine.rules)},
        )

        # Remove self from pending validators
        pending = state.get("pending_validators", []).copy()
        if self.name in pending:
            pending.remove(self.name)

        return {
            "validation_results": [result],
            "completed_validators": [self.name],
            "errors": errors,
            "pending_validators": pending,
        }
