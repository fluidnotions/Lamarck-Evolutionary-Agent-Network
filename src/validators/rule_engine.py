"""Rule engine for business logic validation."""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from ..models.error_detail import ErrorDetail
from ..models.validation_result import ValidationResult


@dataclass
class Rule:
    """
    Represents a validation rule.

    Attributes:
        name: Unique rule identifier
        condition: Function that returns True if rule passes
        message: Error message when rule fails
        severity: Error severity ("error", "warning", "info")
        depends_on: List of rule names this rule depends on
        rule_type: Type of rule ("constraint", "derivation", "inference")
    """

    name: str
    condition: Callable[[Any], bool]
    message: str
    severity: str = "error"
    depends_on: list[str] | None = None
    rule_type: str = "constraint"

    def __post_init__(self) -> None:
        """Validate rule after initialization."""
        if self.severity not in ("error", "warning", "info"):
            raise ValueError(f"Invalid severity: {self.severity}")

        if self.rule_type not in ("constraint", "derivation", "inference"):
            raise ValueError(f"Invalid rule_type: {self.rule_type}")

        if self.depends_on is None:
            self.depends_on = []

    def evaluate(self, data: Any) -> bool:
        """
        Evaluate rule against data.

        Args:
            data: Data to validate

        Returns:
            True if rule passes, False otherwise
        """
        try:
            return self.condition(data)
        except Exception:
            # If condition raises exception, rule fails
            return False

    def to_error_detail(self, path: str = "") -> ErrorDetail:
        """
        Convert rule violation to error detail.

        Args:
            path: Path to the violating data

        Returns:
            ErrorDetail for the violation
        """
        return ErrorDetail(
            path=path,
            message=self.message,
            severity=self.severity,
            code=f"rule_{self.name}",
            context={"rule_name": self.name, "rule_type": self.rule_type},
        )


class RuleEngine:
    """
    Evaluates business rules against data.

    Supports rule dependencies, conflict detection, and result caching.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        """
        Initialize rule engine.

        Args:
            rules: List of rules to evaluate
        """
        self.rules: list[Rule] = rules or []
        self._rule_map: dict[str, Rule] = {rule.name: rule for rule in self.rules}
        self._result_cache: dict[str, bool] = {}
        self._detect_conflicts()

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the engine.

        Args:
            rule: Rule to add
        """
        if rule.name in self._rule_map:
            raise ValueError(f"Rule with name '{rule.name}' already exists")

        self.rules.append(rule)
        self._rule_map[rule.name] = rule
        self._detect_conflicts()

    def _detect_conflicts(self) -> None:
        """Detect circular dependencies in rules."""
        def has_cycle(rule_name: str, visited: set[str], rec_stack: set[str]) -> bool:
            visited.add(rule_name)
            rec_stack.add(rule_name)

            rule = self._rule_map.get(rule_name)
            if rule and rule.depends_on:
                for dep in rule.depends_on:
                    if dep not in visited:
                        if has_cycle(dep, visited, rec_stack):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(rule_name)
            return False

        visited: set[str] = set()
        rec_stack: set[str] = set()

        for rule_name in self._rule_map:
            if rule_name not in visited:
                if has_cycle(rule_name, visited, rec_stack):
                    raise ValueError(f"Circular dependency detected involving rule: {rule_name}")

    def _topological_sort(self) -> list[Rule]:
        """
        Sort rules based on dependencies.

        Returns:
            List of rules in dependency order
        """
        in_degree = defaultdict(int)
        adj_list = defaultdict(list)

        # Build graph
        for rule in self.rules:
            if rule.depends_on:
                for dep in rule.depends_on:
                    adj_list[dep].append(rule.name)
                    in_degree[rule.name] += 1
            else:
                # Ensure rule is in in_degree dict
                if rule.name not in in_degree:
                    in_degree[rule.name] = 0

        # Kahn's algorithm
        queue = [name for name in self._rule_map if in_degree[name] == 0]
        sorted_rules = []

        while queue:
            rule_name = queue.pop(0)
            sorted_rules.append(self._rule_map[rule_name])

            for neighbor in adj_list[rule_name]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return sorted_rules

    def evaluate(self, data: Any, use_cache: bool = True) -> ValidationResult:
        """
        Evaluate all rules against data.

        Args:
            data: Data to validate
            use_cache: Whether to use cached results

        Returns:
            ValidationResult with rule violations
        """
        start_time = time.time()

        if not use_cache:
            self._result_cache.clear()

        # Sort rules by dependencies
        sorted_rules = self._topological_sort()

        errors = []
        warnings = []
        info = []

        for rule in sorted_rules:
            # Check if dependencies are satisfied
            deps_satisfied = True
            if rule.depends_on:
                for dep in rule.depends_on:
                    if dep in self._result_cache and not self._result_cache[dep]:
                        deps_satisfied = False
                        break

            if not deps_satisfied:
                # Skip rule if dependencies failed
                self._result_cache[rule.name] = False
                continue

            # Evaluate rule
            passed = rule.evaluate(data)
            self._result_cache[rule.name] = passed

            if not passed:
                error_detail = rule.to_error_detail()

                if rule.severity == "error":
                    errors.append(error_detail)
                elif rule.severity == "warning":
                    warnings.append(error_detail)
                else:
                    info.append(error_detail)

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="rule_engine",
            status="passed" if not errors else "failed",
            errors=errors,
            warnings=warnings,
            info=info,
            timing=execution_time,
            metadata={
                "total_rules": len(self.rules),
                "rules_evaluated": len(sorted_rules),
                "violations": len(errors) + len(warnings),
            },
        )

    def evaluate_rule(self, rule_name: str, data: Any) -> bool:
        """
        Evaluate a single rule by name.

        Args:
            rule_name: Name of rule to evaluate
            data: Data to validate

        Returns:
            True if rule passes, False otherwise
        """
        if rule_name not in self._rule_map:
            raise ValueError(f"Rule not found: {rule_name}")

        rule = self._rule_map[rule_name]
        return rule.evaluate(data)

    def get_rule_dependencies(self, rule_name: str) -> list[str]:
        """
        Get dependencies for a rule.

        Args:
            rule_name: Name of rule

        Returns:
            List of dependent rule names
        """
        if rule_name not in self._rule_map:
            raise ValueError(f"Rule not found: {rule_name}")

        return self._rule_map[rule_name].depends_on or []

    def clear_cache(self) -> None:
        """Clear the evaluation result cache."""
        self._result_cache.clear()


# Convenience functions for common rules
def create_required_field_rule(field_name: str) -> Rule:
    """Create a rule that checks if a field is present and not null."""
    return Rule(
        name=f"required_{field_name}",
        condition=lambda data: field_name in data and data[field_name] is not None,
        message=f"Required field '{field_name}' is missing or null",
        severity="error",
        rule_type="constraint",
    )


def create_positive_value_rule(field_name: str) -> Rule:
    """Create a rule that checks if a numeric field is positive."""
    return Rule(
        name=f"positive_{field_name}",
        condition=lambda data: field_name in data and data[field_name] > 0,
        message=f"Field '{field_name}' must be positive",
        severity="error",
        rule_type="constraint",
    )


def create_date_range_rule(start_field: str, end_field: str) -> Rule:
    """Create a rule that checks if end date is after start date."""
    return Rule(
        name=f"date_range_{start_field}_{end_field}",
        condition=lambda data: (
            start_field in data
            and end_field in data
            and data[end_field] > data[start_field]
        ),
        message=f"'{end_field}' must be after '{start_field}'",
        severity="error",
        rule_type="constraint",
    )


def create_enum_rule(field_name: str, allowed_values: list[Any]) -> Rule:
    """Create a rule that checks if a field value is in allowed set."""
    return Rule(
        name=f"enum_{field_name}",
        condition=lambda data: field_name in data and data[field_name] in allowed_values,
        message=f"Field '{field_name}' must be one of: {allowed_values}",
        severity="error",
        rule_type="constraint",
    )
