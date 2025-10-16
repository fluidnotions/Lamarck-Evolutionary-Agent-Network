"""Unit tests for confidence calculator."""

import pytest
from src.aggregator.confidence import ConfidenceCalculator
from src.models import ValidationResult, ErrorDetail


class TestConfidenceCalculator:
    """Test cases for ConfidenceCalculator."""

    def test_calculate_empty_results(self):
        """Test confidence calculation with no results."""
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate([])
        assert confidence == 0.0

    def test_calculate_all_passed(self):
        """Test confidence with all validators passing."""
        calculator = ConfidenceCalculator()

        results = [
            ValidationResult(
                validator_name=f"validator_{i}",
                status="passed",
                errors=[],
            )
            for i in range(3)
        ]

        confidence = calculator.calculate(results)
        assert confidence == 1.0

    def test_calculate_all_failed(self):
        """Test confidence with all validators failing."""
        calculator = ConfidenceCalculator()

        results = [
            ValidationResult(
                validator_name=f"validator_{i}",
                status="failed",
                errors=[ErrorDetail(
                    severity="critical",
                    message="Error",
                    validator=f"validator_{i}",
                )],
            )
            for i in range(3)
        ]

        confidence = calculator.calculate(results)
        assert confidence < 0.5  # Should be low

    def test_calculate_mixed_results(self, mixed_results):
        """Test confidence with mixed results."""
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate(mixed_results)

        # Should be between 0 and 1
        assert 0.0 <= confidence <= 1.0
        # Should not be perfect since there are failures
        assert confidence < 1.0

    def test_severity_affects_confidence(self):
        """Test that error severity affects confidence score."""
        calculator = ConfidenceCalculator()

        # Result with critical error
        result_critical = ValidationResult(
            validator_name="critical_validator",
            status="failed",
            errors=[ErrorDetail(
                severity="critical",
                message="Critical error",
                validator="critical_validator",
            )],
        )

        # Result with warning
        result_warning = ValidationResult(
            validator_name="warning_validator",
            status="passed",
            warnings=[ErrorDetail(
                severity="warning",
                message="Warning",
                validator="warning_validator",
            )],
        )

        confidence_critical = calculator.calculate([result_critical])
        confidence_warning = calculator.calculate([result_warning])

        # Warning should have higher confidence than critical error
        assert confidence_warning > confidence_critical

    def test_custom_weights(self):
        """Test custom factor weights."""
        # Custom weights that heavily favor pass rate
        custom_weights = {
            "pass_rate": 0.7,
            "severity": 0.1,
            "coverage": 0.1,
            "reliability": 0.1,
        }

        calculator = ConfidenceCalculator(factor_weights=custom_weights)

        results = [
            ValidationResult(validator_name="v1", status="passed"),
            ValidationResult(validator_name="v2", status="passed"),
            ValidationResult(
                validator_name="v3",
                status="failed",
                errors=[ErrorDetail(
                    severity="critical",
                    message="Error",
                    validator="v3",
                )],
            ),
        ]

        confidence = calculator.calculate(results)
        # With 2/3 pass rate and high pass_rate weight, should be decent
        assert confidence > 0.5

    def test_invalid_weights_raises_error(self):
        """Test that invalid weights raise an error."""
        with pytest.raises(ValueError):
            ConfidenceCalculator(factor_weights={
                "pass_rate": 0.5,
                "severity": 0.3,
                "coverage": 0.1,
                "reliability": 0.05,  # Sums to 0.95, not 1.0
            })

    def test_calculate_with_breakdown(self, mixed_results):
        """Test confidence calculation with detailed breakdown."""
        calculator = ConfidenceCalculator()
        result = calculator.calculate_with_breakdown(mixed_results)

        assert "confidence" in result
        assert "breakdown" in result
        assert "pass_rate" in result["breakdown"]
        assert "severity" in result["breakdown"]
        assert "coverage" in result["breakdown"]
        assert "reliability" in result["breakdown"]
        assert "total_validators" in result
        assert "passed_validators" in result
        assert "failed_validators" in result

    def test_validator_reliability(self):
        """Test that validator reliability affects confidence."""
        calculator = ConfidenceCalculator()

        results = [
            ValidationResult(validator_name="reliable", status="passed"),
            ValidationResult(validator_name="unreliable", status="passed"),
        ]

        reliability = {
            "reliable": 1.0,
            "unreliable": 0.5,
        }

        confidence = calculator.calculate(results, reliability)
        # Should be slightly lower due to unreliable validator
        assert 0.0 < confidence < 1.0

    def test_confidence_level_labels(self):
        """Test confidence level string labels."""
        calculator = ConfidenceCalculator()

        assert calculator.get_confidence_level(0.95) == "very high"
        assert calculator.get_confidence_level(0.80) == "high"
        assert calculator.get_confidence_level(0.60) == "medium"
        assert calculator.get_confidence_level(0.30) == "low"
        assert calculator.get_confidence_level(0.10) == "very low"

    def test_recommendations(self):
        """Test confidence-based recommendations."""
        calculator = ConfidenceCalculator()

        rec_high = calculator.get_recommendation(0.95)
        rec_low = calculator.get_recommendation(0.10)

        assert "excellent" in rec_high.lower() or "safe" in rec_high.lower()
        assert "poor" in rec_low.lower() or "remediation" in rec_low.lower()

    def test_coverage_affects_confidence(self):
        """Test that coverage affects confidence score."""
        calculator = ConfidenceCalculator()

        result_full = ValidationResult(
            validator_name="full_coverage",
            status="passed",
            coverage=1.0,
        )

        result_partial = ValidationResult(
            validator_name="partial_coverage",
            status="passed",
            coverage=0.5,
        )

        confidence_full = calculator.calculate([result_full])
        confidence_partial = calculator.calculate([result_partial])

        # Full coverage should have higher confidence
        assert confidence_full > confidence_partial

    def test_pass_rate_calculation(self):
        """Test pass rate factor calculation."""
        calculator = ConfidenceCalculator()

        # All passed
        results_all_pass = [
            ValidationResult(validator_name=f"v{i}", status="passed")
            for i in range(5)
        ]

        # Half passed
        results_half_pass = [
            ValidationResult(validator_name=f"v{i}", status="passed")
            for i in range(2)
        ] + [
            ValidationResult(
                validator_name=f"v{i}",
                status="failed",
                errors=[ErrorDetail(severity="error", message="err", validator=f"v{i}")]
            )
            for i in range(2, 4)
        ]

        conf_all = calculator.calculate(results_all_pass)
        conf_half = calculator.calculate(results_half_pass)

        assert conf_all > conf_half
