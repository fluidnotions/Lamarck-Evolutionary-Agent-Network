"""Cross-reference validators for referential integrity checks."""

import time
from collections import defaultdict
from typing import Any, Callable

from ..models.error_detail import ErrorDetail
from ..models.validation_result import ValidationResult


class CrossReferenceValidator:
    """
    Validates referential integrity and cross-references.

    Supports foreign key validation, cardinality checks, and cycle detection.
    """

    def __init__(self) -> None:
        """Initialize cross-reference validator."""
        self._reference_cache: dict[str, set[Any]] = {}

    def validate_foreign_key(
        self,
        data: list[dict[str, Any]],
        source_field: str,
        target_data: list[dict[str, Any]],
        target_field: str,
        allow_null: bool = False,
    ) -> ValidationResult:
        """
        Validate foreign key references.

        Args:
            data: Source data containing foreign keys
            source_field: Field name in source data
            target_data: Target data containing primary keys
            target_field: Field name in target data
            allow_null: Whether null foreign keys are allowed

        Returns:
            ValidationResult with foreign key violations
        """
        start_time = time.time()
        errors = []

        # Build set of valid target keys
        valid_keys = {item[target_field] for item in target_data if target_field in item}

        # Check each source reference
        for idx, item in enumerate(data):
            if source_field not in item:
                errors.append(
                    ErrorDetail(
                        path=f"[{idx}].{source_field}",
                        message=f"Foreign key field '{source_field}' is missing",
                        severity="error",
                        code="missing_foreign_key",
                    )
                )
            elif item[source_field] is None:
                if not allow_null:
                    errors.append(
                        ErrorDetail(
                            path=f"[{idx}].{source_field}",
                            message=f"Foreign key '{source_field}' cannot be null",
                            severity="error",
                            code="null_foreign_key",
                        )
                    )
            elif item[source_field] not in valid_keys:
                errors.append(
                    ErrorDetail(
                        path=f"[{idx}].{source_field}",
                        message=f"Foreign key '{item[source_field]}' not found in target",
                        severity="error",
                        code="invalid_foreign_key",
                        context={
                            "foreign_key": item[source_field],
                            "target_field": target_field,
                        },
                    )
                )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="foreign_key_validation",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={
                "source_count": len(data),
                "target_count": len(target_data),
                "valid_keys": len(valid_keys),
            },
        )

    def validate_cardinality(
        self,
        data: list[dict[str, Any]],
        source_field: str,
        target_data: list[dict[str, Any]],
        target_field: str,
        cardinality: str,
    ) -> ValidationResult:
        """
        Validate relationship cardinality.

        Args:
            data: Source data
            source_field: Field name in source data
            target_data: Target data
            target_field: Field name in target data
            cardinality: Cardinality type ("1-to-1", "1-to-many", "many-to-many")

        Returns:
            ValidationResult with cardinality violations
        """
        start_time = time.time()
        errors = []

        # Count references
        source_refs = defaultdict(list)
        target_refs = defaultdict(list)

        for idx, item in enumerate(data):
            if source_field in item and item[source_field] is not None:
                source_refs[item[source_field]].append(idx)

        for idx, item in enumerate(target_data):
            if target_field in item and item[target_field] is not None:
                target_refs[item[target_field]].append(idx)

        if cardinality == "1-to-1":
            # Each source value should appear once, each target value should appear once
            for key, indices in source_refs.items():
                if len(indices) > 1:
                    errors.append(
                        ErrorDetail(
                            path=source_field,
                            message=f"Value '{key}' appears {len(indices)} times (expected 1)",
                            severity="error",
                            code="cardinality_violation_1to1",
                            context={"value": key, "indices": indices},
                        )
                    )

            for key, indices in target_refs.items():
                if len(indices) > 1:
                    errors.append(
                        ErrorDetail(
                            path=target_field,
                            message=f"Value '{key}' appears {len(indices)} times (expected 1)",
                            severity="error",
                            code="cardinality_violation_1to1",
                            context={"value": key, "indices": indices},
                        )
                    )

        elif cardinality == "1-to-many":
            # Each source value should appear once
            for key, indices in source_refs.items():
                if len(indices) > 1:
                    errors.append(
                        ErrorDetail(
                            path=source_field,
                            message=f"Value '{key}' appears {len(indices)} times (expected 1)",
                            severity="error",
                            code="cardinality_violation_1tomany",
                            context={"value": key, "indices": indices},
                        )
                    )

        elif cardinality == "many-to-many":
            # Any count is valid
            pass

        else:
            errors.append(
                ErrorDetail(
                    path="",
                    message=f"Unknown cardinality type: {cardinality}",
                    severity="error",
                    code="unknown_cardinality",
                )
            )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="cardinality_validation",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={
                "cardinality": cardinality,
                "source_unique": len(source_refs),
                "target_unique": len(target_refs),
            },
        )

    def detect_cycles(
        self,
        data: list[dict[str, Any]],
        id_field: str,
        parent_field: str,
    ) -> ValidationResult:
        """
        Detect circular dependencies in hierarchical data.

        Args:
            data: Data to check
            id_field: Field containing node ID
            parent_field: Field containing parent ID

        Returns:
            ValidationResult with cycle detection results
        """
        start_time = time.time()
        errors = []

        # Build adjacency list
        graph: dict[Any, list[Any]] = defaultdict(list)
        nodes = set()

        for item in data:
            if id_field in item and parent_field in item:
                node_id = item[id_field]
                parent_id = item[parent_field]
                nodes.add(node_id)

                if parent_id is not None:
                    graph[node_id].append(parent_id)

        # DFS to detect cycles
        def has_cycle(node: Any, visited: set[Any], rec_stack: set[Any]) -> list[Any] | None:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    cycle = has_cycle(neighbor, visited, rec_stack)
                    if cycle is not None:
                        return [node] + cycle
                elif neighbor in rec_stack:
                    return [node, neighbor]

            rec_stack.remove(node)
            return None

        visited: set[Any] = set()
        for node in nodes:
            if node not in visited:
                cycle = has_cycle(node, visited, set())
                if cycle is not None:
                    errors.append(
                        ErrorDetail(
                            path="",
                            message=f"Circular dependency detected: {' -> '.join(map(str, cycle))}",
                            severity="error",
                            code="circular_dependency",
                            context={"cycle": cycle},
                        )
                    )
                    break  # Report first cycle only

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="cycle_detection",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={
                "nodes_checked": len(nodes),
                "cycles_found": len(errors),
            },
        )

    def validate_external_reference(
        self,
        data: list[dict[str, Any]],
        field: str,
        lookup_func: Callable[[Any], bool],
    ) -> ValidationResult:
        """
        Validate references using external lookup function.

        Args:
            data: Data to validate
            field: Field containing reference to validate
            lookup_func: Function that returns True if reference is valid

        Returns:
            ValidationResult with invalid reference errors
        """
        start_time = time.time()
        errors = []

        for idx, item in enumerate(data):
            if field not in item:
                errors.append(
                    ErrorDetail(
                        path=f"[{idx}].{field}",
                        message=f"Reference field '{field}' is missing",
                        severity="error",
                        code="missing_reference",
                    )
                )
            elif item[field] is not None:
                try:
                    # Check cache first
                    cache_key = f"{field}:{item[field]}"
                    if cache_key not in self._reference_cache:
                        self._reference_cache[cache_key] = {lookup_func(item[field])}

                    is_valid = list(self._reference_cache[cache_key])[0]

                    if not is_valid:
                        errors.append(
                            ErrorDetail(
                                path=f"[{idx}].{field}",
                                message=f"Invalid external reference: {item[field]}",
                                severity="error",
                                code="invalid_external_reference",
                                context={"reference": item[field]},
                            )
                        )

                except Exception as e:
                    errors.append(
                        ErrorDetail(
                            path=f"[{idx}].{field}",
                            message=f"External lookup failed: {str(e)}",
                            severity="error",
                            code="external_lookup_error",
                            context={"exception": str(e), "reference": item[field]},
                        )
                    )

        execution_time = time.time() - start_time

        return ValidationResult(
            validator_name="external_reference_validation",
            status="passed" if not errors else "failed",
            errors=errors,
            timing=execution_time,
            metadata={
                "items_checked": len(data),
                "cache_size": len(self._reference_cache),
            },
        )

    def clear_cache(self) -> None:
        """Clear the reference lookup cache."""
        self._reference_cache.clear()
