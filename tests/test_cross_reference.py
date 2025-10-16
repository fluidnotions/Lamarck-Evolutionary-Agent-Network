"""Tests for cross-reference validators."""

import pytest

from src.validators.cross_reference import CrossReferenceValidator


class TestCrossReferenceValidator:
    """Test cross-reference validator."""

    def test_validate_foreign_key_pass(self) -> None:
        """Test foreign key validation with valid references."""
        source_data = [
            {"id": 1, "user_id": 10},
            {"id": 2, "user_id": 20},
        ]

        target_data = [
            {"user_id": 10, "name": "Alice"},
            {"user_id": 20, "name": "Bob"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_foreign_key(
            source_data, "user_id", target_data, "user_id"
        )

        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_validate_foreign_key_invalid_reference(self) -> None:
        """Test foreign key validation with invalid reference."""
        source_data = [
            {"id": 1, "user_id": 10},
            {"id": 2, "user_id": 99},  # Invalid reference
        ]

        target_data = [
            {"user_id": 10, "name": "Alice"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_foreign_key(
            source_data, "user_id", target_data, "user_id"
        )

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "99" in result.errors[0].message

    def test_validate_foreign_key_null_not_allowed(self) -> None:
        """Test foreign key validation with null when not allowed."""
        source_data = [
            {"id": 1, "user_id": None},
        ]

        target_data = [
            {"user_id": 10, "name": "Alice"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_foreign_key(
            source_data, "user_id", target_data, "user_id", allow_null=False
        )

        assert result.status == "failed"
        assert len(result.errors) == 1

    def test_validate_foreign_key_null_allowed(self) -> None:
        """Test foreign key validation with null when allowed."""
        source_data = [
            {"id": 1, "user_id": None},
        ]

        target_data = [
            {"user_id": 10, "name": "Alice"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_foreign_key(
            source_data, "user_id", target_data, "user_id", allow_null=True
        )

        assert result.status == "passed"

    def test_validate_cardinality_1to1_pass(self) -> None:
        """Test 1-to-1 cardinality validation pass."""
        source_data = [
            {"id": 1, "key": "A"},
            {"id": 2, "key": "B"},
        ]

        target_data = [
            {"id": 10, "key": "A"},
            {"id": 20, "key": "B"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_cardinality(
            source_data, "key", target_data, "key", cardinality="1-to-1"
        )

        assert result.status == "passed"

    def test_validate_cardinality_1to1_fail(self) -> None:
        """Test 1-to-1 cardinality validation fail."""
        source_data = [
            {"id": 1, "key": "A"},
            {"id": 2, "key": "A"},  # Duplicate key
        ]

        target_data = [
            {"id": 10, "key": "A"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_cardinality(
            source_data, "key", target_data, "key", cardinality="1-to-1"
        )

        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_validate_cardinality_1tomany_pass(self) -> None:
        """Test 1-to-many cardinality validation pass."""
        source_data = [
            {"id": 1, "category": "books"},
            {"id": 2, "category": "electronics"},
        ]

        target_data = [
            {"id": 10, "category": "books"},
            {"id": 11, "category": "books"},  # Multiple items in same category (OK)
            {"id": 20, "category": "electronics"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_cardinality(
            source_data, "category", target_data, "category", cardinality="1-to-many"
        )

        assert result.status == "passed"

    def test_validate_cardinality_manytomany(self) -> None:
        """Test many-to-many cardinality validation."""
        source_data = [
            {"id": 1, "tag": "python"},
            {"id": 2, "tag": "python"},
        ]

        target_data = [
            {"id": 10, "tag": "python"},
            {"id": 11, "tag": "python"},
        ]

        validator = CrossReferenceValidator()
        result = validator.validate_cardinality(
            source_data, "tag", target_data, "tag", cardinality="many-to-many"
        )

        # Many-to-many allows any cardinality
        assert result.status == "passed"

    def test_detect_cycles_no_cycle(self) -> None:
        """Test cycle detection with no cycles."""
        data = [
            {"id": 1, "parent_id": None},
            {"id": 2, "parent_id": 1},
            {"id": 3, "parent_id": 1},
            {"id": 4, "parent_id": 2},
        ]

        validator = CrossReferenceValidator()
        result = validator.detect_cycles(data, "id", "parent_id")

        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_detect_cycles_with_cycle(self) -> None:
        """Test cycle detection with a cycle."""
        data = [
            {"id": 1, "parent_id": 3},
            {"id": 2, "parent_id": 1},
            {"id": 3, "parent_id": 2},  # Creates cycle
        ]

        validator = CrossReferenceValidator()
        result = validator.detect_cycles(data, "id", "parent_id")

        assert result.status == "failed"
        assert len(result.errors) > 0
        assert "circular" in result.errors[0].message.lower()

    def test_validate_external_reference_pass(self) -> None:
        """Test external reference validation pass."""
        data = [
            {"id": 1, "external_id": "ext-1"},
            {"id": 2, "external_id": "ext-2"},
        ]

        # Mock lookup function that always returns True
        def lookup_func(ref: str) -> bool:
            return ref.startswith("ext-")

        validator = CrossReferenceValidator()
        result = validator.validate_external_reference(data, "external_id", lookup_func)

        assert result.status == "passed"

    def test_validate_external_reference_fail(self) -> None:
        """Test external reference validation fail."""
        data = [
            {"id": 1, "external_id": "ext-1"},
            {"id": 2, "external_id": "invalid"},
        ]

        # Mock lookup function
        def lookup_func(ref: str) -> bool:
            return ref.startswith("ext-")

        validator = CrossReferenceValidator()
        result = validator.validate_external_reference(data, "external_id", lookup_func)

        assert result.status == "failed"
        assert len(result.errors) == 1

    def test_validate_external_reference_with_exception(self) -> None:
        """Test external reference validation with exception."""
        data = [
            {"id": 1, "external_id": "test"},
        ]

        # Mock lookup function that raises exception
        def lookup_func(ref: str) -> bool:
            raise ValueError("Lookup failed")

        validator = CrossReferenceValidator()
        result = validator.validate_external_reference(data, "external_id", lookup_func)

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "failed" in result.errors[0].message.lower()

    def test_clear_cache(self) -> None:
        """Test clearing reference cache."""
        data = [{"id": 1, "external_id": "test"}]

        def lookup_func(ref: str) -> bool:
            return True

        validator = CrossReferenceValidator()
        validator.validate_external_reference(data, "external_id", lookup_func)

        # Cache should have entries
        assert len(validator._reference_cache) > 0

        validator.clear_cache()

        # Cache should be empty
        assert len(validator._reference_cache) == 0
