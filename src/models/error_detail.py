"""Error detail model for validation errors."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Detailed error information for validation failures."""

    path: str = Field(description="Path to the field that failed validation")
    message: str = Field(description="Human-readable error message")
    severity: str = Field(
        default="error",
        description="Error severity: critical, error, warning, info"
    )
    code: str = Field(
        default="validation_error",
        description="Error code for categorization"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the error occurred"
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional context about the error"
    )
    recoverable: bool = Field(
        default=True,
        description="Whether this error can be recovered from"
    )

    def __str__(self) -> str:
        """String representation of the error."""
        return f"[{self.severity.upper()}] {self.path}: {self.message}"
