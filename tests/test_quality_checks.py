"""Tests for quality check validators."""

import pytest

from src.validators.quality_checks import QualityChecker


class TestQualityChecker:
    """Test quality checker validators."""

    def test_completeness_check_pass(self) -> None:
        """Test completeness check with all required fields."""
        data = {"name": "John", "age": 30, "email": "john@example.com"}
        required_fields = ["name", "age", "email"]

        result = QualityChecker.check_completeness(data, required_fields)

        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_completeness_check_missing_field(self) -> None:
        """Test completeness check with missing field."""
        data = {"name": "John", "age": 30}
        required_fields = ["name", "age", "email"]

        result = QualityChecker.check_completeness(data, required_fields)

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "email" in result.errors[0].path

    def test_completeness_check_null_value(self) -> None:
        """Test completeness check with null value."""
        data = {"name": "John", "age": None}
        required_fields = ["name", "age"]

        result = QualityChecker.check_completeness(data, required_fields)

        assert result.status == "failed"
        assert len(result.errors) == 1
        assert "null" in result.errors[0].message.lower()

    def test_completeness_check_empty_string(self) -> None:
        """Test completeness check with empty string."""
        data = {"name": ""}
        required_fields = ["name"]

        # Should fail when empty strings not allowed
        result = QualityChecker.check_completeness(data, required_fields, allow_empty_strings=False)
        assert result.status == "failed"

        # Should pass when empty strings allowed
        result = QualityChecker.check_completeness(data, required_fields, allow_empty_strings=True)
        assert result.status == "passed"

    def test_consistency_check_pass(self) -> None:
        """Test consistency check with valid data."""
        data = {"start_date": "2024-01-01", "end_date": "2024-12-31"}
        rules = [
            ("end_date", lambda d: d["end_date"] > d["start_date"], "End date must be after start date"),
        ]

        result = QualityChecker.check_consistency(data, rules)

        assert result.status == "passed"
        assert len(result.errors) == 0

    def test_consistency_check_fail(self) -> None:
        """Test consistency check with invalid data."""
        data = {"start_date": "2024-12-31", "end_date": "2024-01-01"}
        rules = [
            ("end_date", lambda d: d["end_date"] > d["start_date"], "End date must be after start date"),
        ]

        result = QualityChecker.check_consistency(data, rules)

        assert result.status == "failed"
        assert len(result.errors) == 1

    def test_format_check_email(self) -> None:
        """Test format check for email."""
        # Valid email
        data = {"email": "test@example.com"}
        result = QualityChecker.check_format(data, "email", "email")
        assert result.status == "passed"

        # Invalid email
        data = {"email": "not-an-email"}
        result = QualityChecker.check_format(data, "email", "email")
        assert result.status == "failed"

    def test_format_check_phone(self) -> None:
        """Test format check for phone."""
        # Valid phone
        data = {"phone": "+1234567890"}
        result = QualityChecker.check_format(data, "phone", "phone")
        assert result.status == "passed"

        # Invalid phone
        data = {"phone": "abc"}
        result = QualityChecker.check_format(data, "phone", "phone")
        assert result.status == "failed"

    def test_format_check_url(self) -> None:
        """Test format check for URL."""
        # Valid URL
        data = {"website": "https://example.com"}
        result = QualityChecker.check_format(data, "website", "url")
        assert result.status == "passed"

        # Invalid URL
        data = {"website": "not-a-url"}
        result = QualityChecker.check_format(data, "website", "url")
        assert result.status == "failed"

    def test_format_check_uuid(self) -> None:
        """Test format check for UUID."""
        # Valid UUID
        data = {"id": "550e8400-e29b-41d4-a716-446655440000"}
        result = QualityChecker.check_format(data, "id", "uuid")
        assert result.status == "passed"

        # Invalid UUID
        data = {"id": "not-a-uuid"}
        result = QualityChecker.check_format(data, "id", "uuid")
        assert result.status == "failed"

    def test_format_check_date(self) -> None:
        """Test format check for date."""
        # Valid date
        data = {"date": "2024-01-01"}
        result = QualityChecker.check_format(data, "date", "date")
        assert result.status == "passed"

        # Invalid date
        data = {"date": "01/01/2024"}
        result = QualityChecker.check_format(data, "date", "date")
        assert result.status == "failed"

    def test_range_check_pass(self) -> None:
        """Test range check with valid value."""
        data = {"age": 25}
        result = QualityChecker.check_range(data, "age", min_value=0, max_value=120)

        assert result.status == "passed"

    def test_range_check_below_minimum(self) -> None:
        """Test range check with value below minimum."""
        data = {"age": -5}
        result = QualityChecker.check_range(data, "age", min_value=0)

        assert result.status == "failed"
        assert "below minimum" in result.errors[0].message.lower()

    def test_range_check_above_maximum(self) -> None:
        """Test range check with value above maximum."""
        data = {"age": 150}
        result = QualityChecker.check_range(data, "age", max_value=120)

        assert result.status == "failed"
        assert "exceeds maximum" in result.errors[0].message.lower()

    def test_range_check_not_numeric(self) -> None:
        """Test range check with non-numeric value."""
        data = {"age": "not a number"}
        result = QualityChecker.check_range(data, "age", min_value=0)

        assert result.status == "failed"
        assert "not numeric" in result.errors[0].message.lower()

    def test_enum_check_pass(self) -> None:
        """Test enum check with valid value."""
        data = {"status": "active"}
        result = QualityChecker.check_enum(data, "status", ["active", "inactive", "pending"])

        assert result.status == "passed"

    def test_enum_check_fail(self) -> None:
        """Test enum check with invalid value."""
        data = {"status": "unknown"}
        result = QualityChecker.check_enum(data, "status", ["active", "inactive", "pending"])

        assert result.status == "failed"
        assert len(result.errors) == 1

    def test_detect_outliers_zscore(self) -> None:
        """Test outlier detection using z-score."""
        values = [10, 12, 13, 12, 11, 100]  # 100 is an outlier

        result = QualityChecker.detect_outliers(values, threshold=2.0, method="zscore")

        assert result.status == "passed"  # Outliers are warnings, not errors
        assert len(result.warnings) > 0

    def test_detect_outliers_iqr(self) -> None:
        """Test outlier detection using IQR."""
        values = [10, 12, 13, 12, 11, 100]  # 100 is an outlier

        result = QualityChecker.detect_outliers(values, method="iqr")

        assert result.status == "passed"
        assert len(result.warnings) > 0

    def test_detect_outliers_no_outliers(self) -> None:
        """Test outlier detection with no outliers."""
        values = [10, 11, 12, 13, 14]

        result = QualityChecker.detect_outliers(values, threshold=3.0)

        assert result.status == "passed"
        assert len(result.warnings) == 0

    def test_detect_outliers_empty_list(self) -> None:
        """Test outlier detection with empty list."""
        values: list[float] = []

        result = QualityChecker.detect_outliers(values)

        assert result.status == "passed"
        assert len(result.warnings) == 0

    def test_detect_duplicates(self) -> None:
        """Test duplicate detection."""
        values = [1, 2, 3, 2, 4, 3, 5]  # 2 and 3 are duplicates

        result = QualityChecker.detect_duplicates(values)

        assert result.status == "passed"  # Duplicates are warnings
        assert len(result.warnings) == 2  # Two duplicate values

    def test_detect_duplicates_no_duplicates(self) -> None:
        """Test duplicate detection with no duplicates."""
        values = [1, 2, 3, 4, 5]

        result = QualityChecker.detect_duplicates(values)

        assert result.status == "passed"
        assert len(result.warnings) == 0
