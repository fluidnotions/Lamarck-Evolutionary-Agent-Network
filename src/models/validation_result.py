"""Validation result model for validator outputs."""

from dataclasses import dataclass, field
from typing import Any

from .error_detail import ErrorDetail


@dataclass
class ValidationResult:
    """
    Result of a validation check.

    Attributes:
        validator_name: Name of the validator that produced this result
        status: Validation status ("passed", "failed", "skipped")
        errors: List of errors found during validation
        warnings: List of warnings found during validation
        info: List of informational messages
        timing: Execution time in seconds
        metadata: Additional metadata about the validation
    """

    validator_name: str
    status: str
    errors: list[ErrorDetail] = field(default_factory=list)
    warnings: list[ErrorDetail] = field(default_factory=list)
    info: list[ErrorDetail] = field(default_factory=list)
    timing: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result after initialization."""
        if self.status not in ("passed", "failed", "skipped"):
            raise ValueError(
                f"Invalid status: {self.status}. Must be 'passed', 'failed', or 'skipped'"
            )

    @property
    def has_errors(self) -> bool:
        """Check if result has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if result has warnings."""
        return len(self.warnings) > 0

    @property
    def error_count(self) -> int:
        """Get total error count."""
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Get total warning count."""
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "validator_name": self.validator_name,
            "status": self.status,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "info": [i.to_dict() for i in self.info],
            "timing": self.timing,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create validation result from dictionary."""
        return cls(
            validator_name=data["validator_name"],
            status=data["status"],
            errors=[ErrorDetail.from_dict(e) for e in data.get("errors", [])],
            warnings=[ErrorDetail.from_dict(w) for w in data.get("warnings", [])],
            info=[ErrorDetail.from_dict(i) for i in data.get("info", [])],
            timing=data.get("timing", 0.0),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation of validation result."""
        parts = [f"Validator: {self.validator_name}", f"Status: {self.status.upper()}"]

        if self.errors:
            parts.append(f"Errors: {self.error_count}")
        if self.warnings:
            parts.append(f"Warnings: {self.warning_count}")
        if self.timing > 0:
            parts.append(f"Timing: {self.timing:.3f}s")

        return " | ".join(parts)
