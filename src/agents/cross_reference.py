"""Cross-reference validation agent for HVAS-Mini."""

from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from src.agents.base import BaseAgent
from src.graph.state import ValidationState
from src.models.validation_result import ValidationResult, ValidationStatus


class CrossReferenceAgent(BaseAgent):
    """Validates relationships and referential integrity."""

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        """Initialize the cross-reference agent.

        Args:
            llm: Optional language model for enhanced explanations
        """
        super().__init__(
            name="cross_reference",
            description="Validates relationships and referential integrity",
            llm=llm,
        )

    def execute(self, state: ValidationState) -> ValidationState:
        """Validate relationships and referential integrity.

        Steps:
        1. Check foreign key references
        2. Validate cardinality constraints
        3. Check for cyclic dependencies
        4. Validate relationship rules
        5. Update state with results

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        input_data = state.get("input_data", {})
        validation_request = state.get("validation_request", {})

        # Get cross-reference configuration
        cross_ref_config = validation_request.get("cross_reference_config", {})

        # Coordinate atomic validators
        results: list[ValidationResult] = []

        # Foreign key validation
        if "foreign_keys" in cross_ref_config:
            fk_result = self._validate_foreign_keys(
                input_data, cross_ref_config["foreign_keys"], state
            )
            results.append(fk_result)

        # Cardinality validation
        if "cardinality" in cross_ref_config:
            card_result = self._validate_cardinality(
                input_data, cross_ref_config["cardinality"]
            )
            results.append(card_result)

        # Cyclic dependency check
        if "check_cycles" in cross_ref_config and cross_ref_config["check_cycles"]:
            cycle_result = self._check_cyclic_dependencies(
                input_data, cross_ref_config.get("relationship_field", "parent_id")
            )
            results.append(cycle_result)

        # Relationship rules
        if "relationship_rules" in cross_ref_config:
            rel_result = self._validate_relationships(
                input_data, cross_ref_config["relationship_rules"]
            )
            results.append(rel_result)

        # If no checks configured, add skipped result
        if not results:
            results.append(
                ValidationResult(
                    validator_name="cross_reference",
                    status=ValidationStatus.SKIPPED,
                    message="No cross-reference checks configured",
                )
            )

        # Aggregate results
        domain_result = self._aggregate_results(results, "cross_reference")

        # Update state
        new_state = self._update_state(state, domain_result)

        return new_state

    def _validate_foreign_keys(
        self,
        data: dict[str, Any],
        foreign_keys: list[dict[str, Any]],
        state: ValidationState,
    ) -> ValidationResult:
        """Validate foreign key references.

        Args:
            data: Data to validate
            foreign_keys: List of foreign key definitions
            state: Current state (may contain reference data)

        Returns:
            Validation result
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Get reference data from state metadata
        metadata = state.get("metadata", {})
        reference_data = metadata.get("reference_data", {})

        for fk_def in foreign_keys:
            field = fk_def.get("field")
            reference_table = fk_def.get("reference_table")
            reference_field = fk_def.get("reference_field", "id")
            allow_null = fk_def.get("allow_null", False)

            if field not in data:
                continue

            value = data[field]

            # Check if null is allowed
            if value is None:
                if not allow_null:
                    issues.append(f"Foreign key '{field}' cannot be null")
                    suggestions.append(f"Provide a valid value for '{field}'")
                continue

            # Check if reference table data is available
            if reference_table not in reference_data:
                issues.append(
                    f"Cannot validate foreign key '{field}': reference table '{reference_table}' not available"
                )
                suggestions.append(
                    f"Provide reference data for table '{reference_table}' in state metadata"
                )
                continue

            # Check if value exists in reference table
            ref_table = reference_data[reference_table]
            ref_values = [item.get(reference_field) for item in ref_table if isinstance(item, dict)]

            if value not in ref_values:
                issues.append(
                    f"Foreign key '{field}' references non-existent value '{value}' in '{reference_table}.{reference_field}'"
                )
                suggestions.append(
                    f"Use a valid reference from '{reference_table}' or create the referenced record first"
                )

        # Determine status
        if issues:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="foreign_key_validation",
            status=status,
            message=f"Foreign key validation: {len(issues)} issues found" if issues else "All foreign keys are valid",
            details={"issues": issues, "foreign_keys_checked": len(foreign_keys)},
            suggestions=suggestions,
        )

    def _validate_cardinality(
        self, data: dict[str, Any], cardinality_rules: list[dict[str, Any]]
    ) -> ValidationResult:
        """Validate cardinality constraints.

        Args:
            data: Data to validate
            cardinality_rules: List of cardinality constraint definitions

        Returns:
            Validation result
        """
        issues: list[str] = []
        suggestions: list[str] = []

        for rule in cardinality_rules:
            field = rule.get("field")
            min_count = rule.get("min", 0)
            max_count = rule.get("max")

            if field not in data:
                if min_count > 0:
                    issues.append(f"Required field '{field}' is missing")
                    suggestions.append(f"Add field '{field}' with at least {min_count} items")
                continue

            value = data[field]

            # Check if value is a collection
            if not isinstance(value, (list, tuple, set)):
                if min_count > 1:
                    issues.append(f"Field '{field}' should be a collection with {min_count} items")
                    suggestions.append(f"Ensure '{field}' is a list or array")
                continue

            count = len(value)

            # Check minimum cardinality
            if count < min_count:
                issues.append(
                    f"Field '{field}' has {count} items but requires at least {min_count}"
                )
                suggestions.append(f"Add more items to '{field}' (need at least {min_count})")

            # Check maximum cardinality
            if max_count is not None and count > max_count:
                issues.append(
                    f"Field '{field}' has {count} items but allows at most {max_count}"
                )
                suggestions.append(f"Remove items from '{field}' (max allowed: {max_count})")

        # Determine status
        if issues:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="cardinality_validation",
            status=status,
            message=f"Cardinality validation: {len(issues)} issues found" if issues else "All cardinality constraints satisfied",
            details={"issues": issues, "rules_checked": len(cardinality_rules)},
            suggestions=suggestions,
        )

    def _check_cyclic_dependencies(
        self, data: dict[str, Any], relationship_field: str
    ) -> ValidationResult:
        """Check for cyclic dependencies in hierarchical data.

        Args:
            data: Data to validate (should contain an 'id' field and a relationship field)
            relationship_field: Field name that contains the parent reference

        Returns:
            Validation result
        """
        issues: list[str] = []
        suggestions: list[str] = []

        # Check if we have list of items with relationships
        if not isinstance(data, dict):
            return ValidationResult(
                validator_name="cyclic_dependency_check",
                status=ValidationStatus.SKIPPED,
                message="Data structure not suitable for cyclic dependency check",
            )

        # If data contains a list of items, check each
        items_field = None
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                items_field = key
                break

        if not items_field:
            return ValidationResult(
                validator_name="cyclic_dependency_check",
                status=ValidationStatus.SKIPPED,
                message="No list of items found for cyclic dependency check",
            )

        items = data[items_field]

        # Build a graph of relationships
        graph: dict[Any, Any] = {}
        for item in items:
            if "id" in item and relationship_field in item:
                node_id = item["id"]
                parent_id = item[relationship_field]
                if parent_id is not None:
                    graph[node_id] = parent_id

        # Check for cycles using DFS
        def has_cycle(node: Any, visited: set[Any], rec_stack: set[Any]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            if node in graph:
                neighbor = graph[node]
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited: set[Any] = set()
        cycles_found: list[Any] = []

        for node in graph:
            if node not in visited:
                if has_cycle(node, visited, set()):
                    cycles_found.append(node)

        if cycles_found:
            issues.append(
                f"Cyclic dependencies detected involving nodes: {', '.join(map(str, cycles_found))}"
            )
            suggestions.append("Remove circular references in the hierarchy")

        # Determine status
        if issues:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="cyclic_dependency_check",
            status=status,
            message=f"Cyclic dependency check: {'cycles found' if issues else 'no cycles detected'}",
            details={"issues": issues, "nodes_checked": len(graph)},
            suggestions=suggestions,
        )

    def _validate_relationships(
        self, data: dict[str, Any], relationship_rules: list[dict[str, Any]]
    ) -> ValidationResult:
        """Validate custom relationship rules.

        Args:
            data: Data to validate
            relationship_rules: List of relationship rule definitions

        Returns:
            Validation result
        """
        issues: list[str] = []
        suggestions: list[str] = []

        for rule in relationship_rules:
            rule_type = rule.get("type")

            if rule_type == "one_to_many":
                # Check that a parent field maps to multiple children
                parent_field = rule.get("parent_field")
                child_field = rule.get("child_field")

                if parent_field in data and child_field in data:
                    parent_val = data[parent_field]
                    child_val = data[child_field]

                    if not isinstance(child_val, list):
                        issues.append(
                            f"One-to-many relationship: '{child_field}' should be a list"
                        )
                        suggestions.append(f"Ensure '{child_field}' is a list or array")

            elif rule_type == "many_to_many":
                # Check that both fields are lists
                field1 = rule.get("field1")
                field2 = rule.get("field2")

                if field1 in data and not isinstance(data[field1], list):
                    issues.append(
                        f"Many-to-many relationship: '{field1}' should be a list"
                    )
                    suggestions.append(f"Ensure '{field1}' is a list or array")

                if field2 in data and not isinstance(data[field2], list):
                    issues.append(
                        f"Many-to-many relationship: '{field2}' should be a list"
                    )
                    suggestions.append(f"Ensure '{field2}' is a list or array")

            elif rule_type == "unique_reference":
                # Check that referenced values are unique
                field = rule.get("field")

                if field in data:
                    value = data[field]
                    if isinstance(value, list):
                        if len(value) != len(set(value)):
                            issues.append(
                                f"Field '{field}' contains duplicate references"
                            )
                            suggestions.append(
                                f"Ensure all values in '{field}' are unique"
                            )

        # Determine status
        if issues:
            status = ValidationStatus.FAILED
        else:
            status = ValidationStatus.PASSED

        return ValidationResult(
            validator_name="relationship_validation",
            status=status,
            message=f"Relationship validation: {len(issues)} issues found" if issues else "All relationships are valid",
            details={"issues": issues, "rules_checked": len(relationship_rules)},
            suggestions=suggestions,
        )
