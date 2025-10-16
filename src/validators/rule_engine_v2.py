"""Advanced rule engine with complex boolean logic and performance optimization."""
from typing import Any, Dict, List, Callable, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import operator
import logging
import time
from collections import defaultdict

from src.models.validation_result import ErrorDetail

logger = logging.getLogger(__name__)


class BooleanOperator(Enum):
    """Boolean operators for combining conditions."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    XOR = "XOR"


class ComparisonOperator(Enum):
    """Comparison operators for conditions."""
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # regex


@dataclass
class Condition:
    """Represents a single condition in a rule."""

    field: str
    operator: ComparisonOperator
    value: Any
    negate: bool = False

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate this condition against data.

        Args:
            data: Data to evaluate

        Returns:
            True if condition passes
        """
        try:
            # Get field value (supports nested paths like "user.age")
            field_value = self._get_field_value(data, self.field)

            # Evaluate based on operator
            result = self._apply_operator(field_value, self.value, self.operator)

            # Apply negation if needed
            return not result if self.negate else result

        except Exception as e:
            logger.warning(f"Error evaluating condition {self.field} {self.operator}: {e}")
            return False

    def _get_field_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested path like 'user.address.city'."""
        parts = path.split('.')
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def _apply_operator(self, field_value: Any, compare_value: Any, op: ComparisonOperator) -> bool:
        """Apply comparison operator."""
        if op == ComparisonOperator.EQ:
            return field_value == compare_value
        elif op == ComparisonOperator.NE:
            return field_value != compare_value
        elif op == ComparisonOperator.LT:
            return field_value < compare_value
        elif op == ComparisonOperator.LE:
            return field_value <= compare_value
        elif op == ComparisonOperator.GT:
            return field_value > compare_value
        elif op == ComparisonOperator.GE:
            return field_value >= compare_value
        elif op == ComparisonOperator.IN:
            return field_value in compare_value
        elif op == ComparisonOperator.NOT_IN:
            return field_value not in compare_value
        elif op == ComparisonOperator.CONTAINS:
            return compare_value in field_value
        elif op == ComparisonOperator.STARTS_WITH:
            return str(field_value).startswith(str(compare_value))
        elif op == ComparisonOperator.ENDS_WITH:
            return str(field_value).endswith(str(compare_value))
        elif op == ComparisonOperator.MATCHES:
            import re
            return bool(re.match(compare_value, str(field_value)))
        return False


@dataclass
class CompositeCondition:
    """Represents a composite condition with boolean logic."""

    operator: BooleanOperator
    conditions: List[Union['CompositeCondition', Condition]] = field(default_factory=list)

    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate this composite condition.

        Args:
            data: Data to evaluate

        Returns:
            True if condition passes
        """
        if not self.conditions:
            return True

        # Evaluate all sub-conditions
        results = [c.evaluate(data) for c in self.conditions]

        # Apply boolean operator
        if self.operator == BooleanOperator.AND:
            return all(results)
        elif self.operator == BooleanOperator.OR:
            return any(results)
        elif self.operator == BooleanOperator.NOT:
            # NOT should have exactly one condition
            return not results[0] if results else True
        elif self.operator == BooleanOperator.XOR:
            # XOR: exactly one should be true
            return sum(results) == 1

        return False

    def add_condition(self, condition: Union['CompositeCondition', Condition]) -> 'CompositeCondition':
        """Add a condition to this composite."""
        self.conditions.append(condition)
        return self


@dataclass
class RuleV2:
    """Advanced rule with complex conditions and metadata."""

    name: str
    condition: Union[CompositeCondition, Condition, Callable]
    error_message: str
    severity: str = "error"
    enabled: bool = True
    priority: int = 0  # Higher priority rules run first
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # Rule names this depends on

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        self._evaluation_count = 0
        self._total_time_ms = 0.0
        self._failure_count = 0

    def evaluate(self, data: Dict[str, Any]) -> tuple[bool, Optional[ErrorDetail]]:
        """Evaluate this rule against data.

        Args:
            data: Data to validate

        Returns:
            Tuple of (passed, error_detail)
        """
        if not self.enabled:
            return True, None

        start_time = time.perf_counter()

        try:
            # Evaluate condition
            if callable(self.condition):
                # Legacy lambda-based condition
                passed = self.condition(data)
            else:
                # New structured condition
                passed = self.condition.evaluate(data)

            # Track metrics
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._evaluation_count += 1
            self._total_time_ms += elapsed_ms

            if not passed:
                self._failure_count += 1
                error = ErrorDetail(
                    path="root",
                    message=self.error_message,
                    code=f"RULE_{self.name.upper().replace(' ', '_').replace('-', '_')}",
                    severity=self.severity,
                    context={
                        "rule": self.name,
                        "tags": list(self.tags),
                        "priority": self.priority,
                    },
                )
                return False, error

            return True, None

        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            self._failure_count += 1
            error = ErrorDetail(
                path="root",
                message=f"Rule evaluation error: {str(e)}",
                code="RULE_EVALUATION_ERROR",
                severity="error",
                context={"rule": self.name, "exception": str(e)},
            )
            return False, error

    @property
    def average_time_ms(self) -> float:
        """Get average evaluation time in milliseconds."""
        if self._evaluation_count == 0:
            return 0.0
        return self._total_time_ms / self._evaluation_count

    @property
    def failure_rate(self) -> float:
        """Get failure rate (0.0 to 1.0)."""
        if self._evaluation_count == 0:
            return 0.0
        return self._failure_count / self._evaluation_count


class RuleEngineV2:
    """Advanced rule engine with performance optimization and analytics."""

    def __init__(self) -> None:
        """Initialize advanced rule engine."""
        self.rules: Dict[str, RuleV2] = {}
        self._rule_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> rule names
        self._compiled_rules: Dict[str, Any] = {}  # For future compilation optimization

    def add_rule(self, rule: RuleV2) -> None:
        """Add a rule to the engine.

        Args:
            rule: Rule to add
        """
        self.rules[rule.name] = rule

        # Index by tags
        for tag in rule.tags:
            self._rule_index[tag].add(rule.name)

    def remove_rule(self, name: str) -> None:
        """Remove a rule from the engine.

        Args:
            name: Name of rule to remove
        """
        if name in self.rules:
            rule = self.rules[name]
            # Remove from tag index
            for tag in rule.tags:
                self._rule_index[tag].discard(name)
            del self.rules[name]

    def enable_rule(self, name: str) -> None:
        """Enable a rule."""
        if name in self.rules:
            self.rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        """Disable a rule."""
        if name in self.rules:
            self.rules[name].enabled = False

    def get_rules_by_tag(self, tag: str) -> List[RuleV2]:
        """Get all rules with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of rules with that tag
        """
        rule_names = self._rule_index.get(tag, set())
        return [self.rules[name] for name in rule_names if name in self.rules]

    def validate(
        self,
        data: Dict[str, Any],
        rule_names: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        fail_fast: bool = False,
    ) -> tuple[bool, List[ErrorDetail]]:
        """Validate data against rules.

        Args:
            data: Data to validate
            rule_names: Optional list of specific rules to run
            tags: Optional list of tags to filter rules
            fail_fast: Stop on first failure if True

        Returns:
            Tuple of (all_passed, errors)
        """
        errors: List[ErrorDetail] = []

        # Determine which rules to run
        if rule_names:
            rules_to_run = [self.rules[name] for name in rule_names if name in self.rules]
        elif tags:
            # Get rules matching any of the tags
            rule_set = set()
            for tag in tags:
                rule_set.update(self._rule_index.get(tag, set()))
            rules_to_run = [self.rules[name] for name in rule_set if name in self.rules]
        else:
            rules_to_run = list(self.rules.values())

        # Filter enabled rules and sort by priority
        rules_to_run = [r for r in rules_to_run if r.enabled]
        rules_to_run.sort(key=lambda r: r.priority, reverse=True)

        # Check dependencies
        rules_to_run = self._resolve_dependencies(rules_to_run)

        # Evaluate rules
        for rule in rules_to_run:
            passed, error = rule.evaluate(data)

            if not passed and error:
                errors.append(error)
                if fail_fast:
                    break

        return len(errors) == 0, errors

    def _resolve_dependencies(self, rules: List[RuleV2]) -> List[RuleV2]:
        """Resolve rule dependencies and return in execution order.

        Args:
            rules: List of rules to order

        Returns:
            Ordered list of rules
        """
        # Build dependency graph
        rule_map = {r.name: r for r in rules}
        ordered: List[RuleV2] = []
        visited: Set[str] = set()

        def visit(rule_name: str) -> None:
            if rule_name in visited:
                return

            if rule_name not in rule_map:
                return

            visited.add(rule_name)
            rule = rule_map[rule_name]

            # Visit dependencies first
            for dep in rule.dependencies:
                visit(dep)

            ordered.append(rule)

        # Visit all rules
        for rule in rules:
            visit(rule.name)

        return ordered

    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics about rule performance.

        Returns:
            Dictionary with analytics data
        """
        total_rules = len(self.rules)
        enabled_rules = sum(1 for r in self.rules.values() if r.enabled)

        # Performance stats
        slowest_rules = sorted(
            self.rules.values(),
            key=lambda r: r.average_time_ms,
            reverse=True
        )[:10]

        # Most failing rules
        failing_rules = sorted(
            self.rules.values(),
            key=lambda r: r.failure_rate,
            reverse=True
        )[:10]

        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "disabled_rules": total_rules - enabled_rules,
            "slowest_rules": [
                {
                    "name": r.name,
                    "avg_time_ms": r.average_time_ms,
                    "evaluations": r._evaluation_count,
                }
                for r in slowest_rules
            ],
            "most_failing_rules": [
                {
                    "name": r.name,
                    "failure_rate": r.failure_rate,
                    "failures": r._failure_count,
                    "evaluations": r._evaluation_count,
                }
                for r in failing_rules
            ],
            "tags": dict(self._rule_index),
        }

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detect potentially conflicting rules.

        Returns:
            List of conflict descriptions
        """
        conflicts: List[Dict[str, Any]] = []

        # Check for rules with same priority and conflicting conditions
        # This is a simplified implementation - real conflict detection
        # would need semantic analysis of conditions

        priority_groups = defaultdict(list)
        for rule in self.rules.values():
            priority_groups[rule.priority].append(rule)

        for priority, rules in priority_groups.items():
            if len(rules) > 1:
                conflicts.append({
                    "type": "priority_conflict",
                    "priority": priority,
                    "rules": [r.name for r in rules],
                    "message": f"{len(rules)} rules have the same priority {priority}",
                })

        return conflicts


# Rule builders for common patterns
def create_field_condition(
    field: str,
    operator: Union[ComparisonOperator, str],
    value: Any,
    negate: bool = False,
) -> Condition:
    """Create a simple field condition.

    Args:
        field: Field path (supports nested like 'user.age')
        operator: Comparison operator
        value: Value to compare against
        negate: Whether to negate the condition

    Returns:
        Condition instance
    """
    if isinstance(operator, str):
        # Convert string to enum
        op_map = {
            "==": ComparisonOperator.EQ,
            "!=": ComparisonOperator.NE,
            "<": ComparisonOperator.LT,
            "<=": ComparisonOperator.LE,
            ">": ComparisonOperator.GT,
            ">=": ComparisonOperator.GE,
            "in": ComparisonOperator.IN,
            "not_in": ComparisonOperator.NOT_IN,
            "contains": ComparisonOperator.CONTAINS,
            "starts_with": ComparisonOperator.STARTS_WITH,
            "ends_with": ComparisonOperator.ENDS_WITH,
            "matches": ComparisonOperator.MATCHES,
        }
        operator = op_map.get(operator, ComparisonOperator.EQ)

    return Condition(field=field, operator=operator, value=value, negate=negate)


def create_and_condition(*conditions: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Create an AND composite condition.

    Args:
        *conditions: Conditions to combine with AND

    Returns:
        CompositeCondition with AND operator
    """
    return CompositeCondition(operator=BooleanOperator.AND, conditions=list(conditions))


def create_or_condition(*conditions: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Create an OR composite condition.

    Args:
        *conditions: Conditions to combine with OR

    Returns:
        CompositeCondition with OR operator
    """
    return CompositeCondition(operator=BooleanOperator.OR, conditions=list(conditions))


def create_not_condition(condition: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Create a NOT composite condition.

    Args:
        condition: Condition to negate

    Returns:
        CompositeCondition with NOT operator
    """
    return CompositeCondition(operator=BooleanOperator.NOT, conditions=[condition])


def create_xor_condition(*conditions: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Create an XOR composite condition.

    Args:
        *conditions: Conditions to combine with XOR

    Returns:
        CompositeCondition with XOR operator
    """
    return CompositeCondition(operator=BooleanOperator.XOR, conditions=list(conditions))
