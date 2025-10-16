"""Business rule validation engine."""
from typing import Any, Dict, List, Callable
import operator
import logging

from src.models.validation_result import ErrorDetail

logger = logging.getLogger(__name__)


class Rule:
    """Represents a business rule."""

    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        error_message: str,
        severity: str = "error",
    ):
        """Initialize rule.

        Args:
            name: Rule name/identifier
            condition: Function that returns True if rule passes
            error_message: Error message if rule fails
            severity: Error severity level
        """
        self.name = name
        self.condition = condition
        self.error_message = error_message
        self.severity = severity

    def validate(self, data: Dict[str, Any]) -> tuple[bool, ErrorDetail | None]:
        """Validate data against this rule.

        Args:
            data: Data to validate

        Returns:
            Tuple of (passed, error_detail)
        """
        try:
            passed = self.condition(data)
            if passed:
                return True, None
            else:
                error = ErrorDetail(
                    path="root",
                    message=self.error_message,
                    code=f"RULE_{self.name.upper().replace(' ', '_')}",
                    severity=self.severity,
                    context={"rule": self.name},
                )
                return False, error
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            error = ErrorDetail(
                path="root",
                message=f"Rule evaluation error: {str(e)}",
                code="RULE_EVALUATION_ERROR",
                severity="error",
                context={"rule": self.name, "exception": str(e)},
            )
            return False, error


class RuleEngine:
    """Engine for evaluating business rules."""

    def __init__(self):
        """Initialize rule engine."""
        self.rules: Dict[str, Rule] = {}

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine.

        Args:
            rule: Rule to add
        """
        self.rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove a rule from the engine.

        Args:
            name: Name of rule to remove
        """
        if name in self.rules:
            del self.rules[name]

    def validate(self, data: Dict[str, Any], rule_names: List[str] | None = None) -> tuple[bool, List[ErrorDetail]]:
        """Validate data against rules.

        Args:
            data: Data to validate
            rule_names: Optional list of specific rules to run (runs all if None)

        Returns:
            Tuple of (all_passed, errors)
        """
        errors = []
        rules_to_run = rule_names or list(self.rules.keys())

        for rule_name in rules_to_run:
            if rule_name not in self.rules:
                logger.warning(f"Rule {rule_name} not found in engine")
                continue

            rule = self.rules[rule_name]
            passed, error = rule.validate(data)

            if not passed and error:
                errors.append(error)

        return len(errors) == 0, errors


# Common rule builders
def create_range_rule(
    field: str, min_val: float | None = None, max_val: float | None = None
) -> Rule:
    """Create a rule that checks if a numeric field is in range.

    Args:
        field: Field name to check
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Rule instance
    """

    def condition(data: Dict[str, Any]) -> bool:
        value = data.get(field)
        if value is None:
            return False

        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True

    error_msg = f"{field} must be"
    if min_val is not None:
        error_msg += f" >= {min_val}"
    if max_val is not None:
        if min_val is not None:
            error_msg += " and"
        error_msg += f" <= {max_val}"

    return Rule(
        name=f"{field}_range",
        condition=condition,
        error_message=error_msg,
    )


def create_required_field_rule(field: str) -> Rule:
    """Create a rule that checks if a field exists and is not empty.

    Args:
        field: Field name to check

    Returns:
        Rule instance
    """

    def condition(data: Dict[str, Any]) -> bool:
        value = data.get(field)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
        return True

    return Rule(
        name=f"{field}_required",
        condition=condition,
        error_message=f"Field '{field}' is required and must not be empty",
    )


def create_comparison_rule(
    field1: str, op_str: str, field2: str
) -> Rule:
    """Create a rule that compares two fields.

    Args:
        field1: First field name
        op_str: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        field2: Second field name

    Returns:
        Rule instance
    """
    ops = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if op_str not in ops:
        raise ValueError(f"Invalid operator: {op_str}")

    op_func = ops[op_str]

    def condition(data: Dict[str, Any]) -> bool:
        val1 = data.get(field1)
        val2 = data.get(field2)
        if val1 is None or val2 is None:
            return False
        return op_func(val1, val2)

    return Rule(
        name=f"{field1}_{op_str}_{field2}",
        condition=condition,
        error_message=f"{field1} must be {op_str} {field2}",
    )
