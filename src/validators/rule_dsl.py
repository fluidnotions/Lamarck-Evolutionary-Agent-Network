"""Natural language-like DSL for defining validation rules."""
from typing import Any, Dict, Optional, Set, Callable, Union
from dataclasses import dataclass, field
import re

from src.validators.rule_engine_v2 import (
    RuleV2,
    Condition,
    CompositeCondition,
    BooleanOperator,
    ComparisonOperator,
    create_field_condition,
    create_and_condition,
    create_or_condition,
    create_not_condition,
)


class FieldAccessor:
    """Provides fluent API for field access in rules."""

    def __init__(self, field_path: str):
        """Initialize field accessor.

        Args:
            field_path: Path to field (e.g., 'user.age')
        """
        self.field_path = field_path

    def __eq__(self, value: Any) -> Condition:  # type: ignore
        """Create equality condition."""
        return create_field_condition(self.field_path, ComparisonOperator.EQ, value)

    def __ne__(self, value: Any) -> Condition:  # type: ignore
        """Create not-equal condition."""
        return create_field_condition(self.field_path, ComparisonOperator.NE, value)

    def __lt__(self, value: Any) -> Condition:
        """Create less-than condition."""
        return create_field_condition(self.field_path, ComparisonOperator.LT, value)

    def __le__(self, value: Any) -> Condition:
        """Create less-than-or-equal condition."""
        return create_field_condition(self.field_path, ComparisonOperator.LE, value)

    def __gt__(self, value: Any) -> Condition:
        """Create greater-than condition."""
        return create_field_condition(self.field_path, ComparisonOperator.GT, value)

    def __ge__(self, value: Any) -> Condition:
        """Create greater-than-or-equal condition."""
        return create_field_condition(self.field_path, ComparisonOperator.GE, value)

    def is_in(self, values: list) -> Condition:
        """Create 'in' condition."""
        return create_field_condition(self.field_path, ComparisonOperator.IN, values)

    def not_in(self, values: list) -> Condition:
        """Create 'not in' condition."""
        return create_field_condition(self.field_path, ComparisonOperator.NOT_IN, values)

    def contains(self, value: Any) -> Condition:
        """Create 'contains' condition."""
        return create_field_condition(self.field_path, ComparisonOperator.CONTAINS, value)

    def starts_with(self, value: str) -> Condition:
        """Create 'starts with' condition."""
        return create_field_condition(self.field_path, ComparisonOperator.STARTS_WITH, value)

    def ends_with(self, value: str) -> Condition:
        """Create 'ends with' condition."""
        return create_field_condition(self.field_path, ComparisonOperator.ENDS_WITH, value)

    def matches(self, pattern: str) -> Condition:
        """Create regex match condition."""
        return create_field_condition(self.field_path, ComparisonOperator.MATCHES, pattern)

    def exists(self) -> Condition:
        """Create condition checking if field exists."""
        return create_field_condition(self.field_path, ComparisonOperator.NE, None)

    def is_empty(self) -> Condition:
        """Create condition checking if field is empty."""
        return create_field_condition(self.field_path, ComparisonOperator.EQ, "")

    def is_null(self) -> Condition:
        """Create condition checking if field is null."""
        return create_field_condition(self.field_path, ComparisonOperator.EQ, None)


class DataProxy:
    """Proxy object for accessing data fields with dot notation."""

    def __getattr__(self, name: str) -> FieldAccessor:
        """Get field accessor for attribute.

        Args:
            name: Field name

        Returns:
            FieldAccessor for this field
        """
        return FieldAccessor(name)

    def __getitem__(self, path: str) -> FieldAccessor:
        """Get field accessor for nested path.

        Args:
            path: Field path like 'user.age'

        Returns:
            FieldAccessor for this path
        """
        return FieldAccessor(path)


# Global data proxy for use in rule definitions
data = DataProxy()


class RuleBuilder:
    """Fluent builder for creating rules using DSL."""

    def __init__(self, name: str):
        """Initialize rule builder.

        Args:
            name: Rule name
        """
        self._name = name
        self._condition: Optional[Union[Condition, CompositeCondition, Callable]] = None
        self._error_message: str = ""
        self._severity: str = "error"
        self._priority: int = 0
        self._tags: Set[str] = set()
        self._metadata: Dict[str, Any] = {}
        self._dependencies: Set[str] = set()
        self._enabled: bool = True

    def when(self, condition: Union[Condition, CompositeCondition, Callable]) -> 'RuleBuilder':
        """Set the rule condition.

        Args:
            condition: Condition to evaluate

        Returns:
            Self for chaining
        """
        self._condition = condition
        return self

    def then(self, error_message: str) -> 'RuleBuilder':
        """Set the error message for rule failure.

        Args:
            error_message: Error message

        Returns:
            Self for chaining
        """
        self._error_message = error_message
        return self

    def reject(self, error_message: str) -> 'RuleBuilder':
        """Alias for then() - set error message for rejection.

        Args:
            error_message: Error message

        Returns:
            Self for chaining
        """
        return self.then(error_message)

    def severity(self, level: str) -> 'RuleBuilder':
        """Set the severity level.

        Args:
            level: Severity level (error, warning, info)

        Returns:
            Self for chaining
        """
        self._severity = level
        return self

    def priority(self, value: int) -> 'RuleBuilder':
        """Set the rule priority.

        Args:
            value: Priority value (higher = runs first)

        Returns:
            Self for chaining
        """
        self._priority = value
        return self

    def tag(self, *tags: str) -> 'RuleBuilder':
        """Add tags to the rule.

        Args:
            *tags: Tags to add

        Returns:
            Self for chaining
        """
        self._tags.update(tags)
        return self

    def with_metadata(self, **metadata: Any) -> 'RuleBuilder':
        """Add metadata to the rule.

        Args:
            **metadata: Metadata key-value pairs

        Returns:
            Self for chaining
        """
        self._metadata.update(metadata)
        return self

    def depends_on(self, *rule_names: str) -> 'RuleBuilder':
        """Declare rule dependencies.

        Args:
            *rule_names: Names of rules this depends on

        Returns:
            Self for chaining
        """
        self._dependencies.update(rule_names)
        return self

    def enabled(self, is_enabled: bool = True) -> 'RuleBuilder':
        """Set whether rule is enabled.

        Args:
            is_enabled: Enable/disable flag

        Returns:
            Self for chaining
        """
        self._enabled = is_enabled
        return self

    def build(self) -> RuleV2:
        """Build the rule.

        Returns:
            RuleV2 instance

        Raises:
            ValueError: If required fields are missing
        """
        if not self._condition:
            raise ValueError(f"Rule {self._name} must have a condition")
        if not self._error_message:
            raise ValueError(f"Rule {self._name} must have an error message")

        return RuleV2(
            name=self._name,
            condition=self._condition,
            error_message=self._error_message,
            severity=self._severity,
            priority=self._priority,
            tags=self._tags,
            metadata=self._metadata,
            dependencies=self._dependencies,
            enabled=self._enabled,
        )


def rule(name: str) -> RuleBuilder:
    """Create a new rule using DSL.

    Args:
        name: Rule name

    Returns:
        RuleBuilder for fluent API

    Example:
        ```python
        rule("minimum_age")
            .when(data.age >= 18)
            .then("Must be 18 or older")
            .severity("error")
            .tag("age_validation")
            .build()
        ```
    """
    return RuleBuilder(name)


# Logical operators for combining conditions
def AND(*conditions: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Combine conditions with AND logic.

    Args:
        *conditions: Conditions to combine

    Returns:
        CompositeCondition with AND operator
    """
    return create_and_condition(*conditions)


def OR(*conditions: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Combine conditions with OR logic.

    Args:
        *conditions: Conditions to combine

    Returns:
        CompositeCondition with OR operator
    """
    return create_or_condition(*conditions)


def NOT(condition: Union[Condition, CompositeCondition]) -> CompositeCondition:
    """Negate a condition.

    Args:
        condition: Condition to negate

    Returns:
        CompositeCondition with NOT operator
    """
    return create_not_condition(condition)


class RuleDSLParser:
    """Parser for text-based rule DSL."""

    def __init__(self) -> None:
        """Initialize DSL parser."""
        self.operators = {
            '==': ComparisonOperator.EQ,
            '!=': ComparisonOperator.NE,
            '<': ComparisonOperator.LT,
            '<=': ComparisonOperator.LE,
            '>': ComparisonOperator.GT,
            '>=': ComparisonOperator.GE,
            'in': ComparisonOperator.IN,
            'not in': ComparisonOperator.NOT_IN,
            'contains': ComparisonOperator.CONTAINS,
            'starts with': ComparisonOperator.STARTS_WITH,
            'ends with': ComparisonOperator.ENDS_WITH,
            'matches': ComparisonOperator.MATCHES,
        }

    def parse(self, text: str) -> Union[Condition, CompositeCondition]:
        """Parse text-based rule DSL.

        Args:
            text: Rule text like "age >= 18 AND status == 'active'"

        Returns:
            Condition or CompositeCondition

        Example:
            ```python
            parser = RuleDSLParser()
            condition = parser.parse("age >= 18 AND status == 'active'")
            ```
        """
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Check for boolean operators at top level
        if ' AND ' in text:
            parts = text.split(' AND ')
            conditions = [self.parse(part.strip()) for part in parts]
            return create_and_condition(*conditions)

        if ' OR ' in text:
            parts = text.split(' OR ')
            conditions = [self.parse(part.strip()) for part in parts]
            return create_or_condition(*conditions)

        if text.startswith('NOT '):
            inner = text[4:].strip()
            condition = self.parse(inner)
            return create_not_condition(condition)

        # Parse simple condition
        return self._parse_simple_condition(text)

    def _parse_simple_condition(self, text: str) -> Condition:
        """Parse a simple condition without boolean operators.

        Args:
            text: Condition text

        Returns:
            Condition instance
        """
        # Try to match pattern: field operator value
        for op_str, op_enum in sorted(self.operators.items(), key=lambda x: -len(x[0])):
            if op_str in text:
                parts = text.split(op_str, 1)
                if len(parts) == 2:
                    field = parts[0].strip()
                    value_str = parts[1].strip()

                    # Parse value
                    value = self._parse_value(value_str)

                    return Condition(
                        field=field,
                        operator=op_enum,
                        value=value,
                    )

        raise ValueError(f"Could not parse condition: {text}")

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value from string.

        Args:
            value_str: Value string

        Returns:
            Parsed value
        """
        value_str = value_str.strip()

        # String literal
        if (value_str.startswith("'") and value_str.endswith("'")) or \
           (value_str.startswith('"') and value_str.endswith('"')):
            return value_str[1:-1]

        # Boolean
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False

        # None/null
        if value_str.lower() in ('none', 'null'):
            return None

        # List
        if value_str.startswith('[') and value_str.endswith(']'):
            # Simple list parsing
            items = value_str[1:-1].split(',')
            return [self._parse_value(item.strip()) for item in items]

        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Return as string if all else fails
        return value_str


# Convenience constants for severity levels
ERROR = "error"
WARNING = "warning"
INFO = "info"
