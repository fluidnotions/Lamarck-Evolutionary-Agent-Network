"""Error detail model for validation errors."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorDetail:
    """
    Detailed information about a validation error.

    Attributes:
        path: JSON path to the error location (e.g., "data.users[0].email")
        message: Human-readable error message
        severity: Error severity level ("error", "warning", "info")
        code: Machine-readable error code for categorization
        context: Additional context information about the error
    """

    path: str
    message: str
    severity: str = "error"
    code: str = "validation_error"
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate error detail after initialization."""
        if self.severity not in ("error", "warning", "info"):
            raise ValueError(
                f"Invalid severity: {self.severity}. Must be 'error', 'warning', or 'info'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert error detail to dictionary."""
        return {
            "path": self.path,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ErrorDetail":
        """Create error detail from dictionary."""
        return cls(
            path=data["path"],
            message=data["message"],
            severity=data.get("severity", "error"),
            code=data.get("code", "validation_error"),
            context=data.get("context", {}),
        )

    def __str__(self) -> str:
        """String representation of error detail."""
        if self.path:
            return f"[{self.severity.upper()}] {self.path}: {self.message}"
        return f"[{self.severity.upper()}] {self.message}"
