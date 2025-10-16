"""Base agent class for HVAS-Mini validators.

This module provides the abstract BaseAgent class that all validators inherit from.
It standardizes the agent interface, error handling, and integration with the
validation state workflow.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from src.graph.state import ErrorDetail, ValidationResult, ValidationState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all validation agents.

    This class provides a standard interface for validators and handles:
    - State validation
    - Error handling and recovery
    - Logging integration
    - Performance metrics collection
    - Common validation patterns

    Subclasses must implement the _execute method to provide
    validator-specific logic.

    Attributes:
        name: Unique identifier for this agent
        description: Human-readable description of what this agent validates
        capabilities: List of capabilities this agent provides
        metadata: Additional agent-specific configuration
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Initialize the base agent.

        Args:
            name: Unique identifier for this agent
            description: Human-readable description
            capabilities: List of capabilities (e.g., ['syntax_check', 'type_validation'])
            metadata: Additional configuration data
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.metadata = metadata or {}
        logger.info(f"Initialized agent: {self.name}", extra={"agent": self.name})

    def execute(self, state: ValidationState) -> ValidationState:
        """Execute the validation agent on the given state.

        This is the main entry point for agent execution. It handles:
        - State validation
        - Error handling and recovery
        - Performance tracking
        - State updates

        Args:
            state: Current validation state

        Returns:
            Updated validation state with results or errors added
        """
        start_time = time.time()

        logger.info(
            f"Starting agent execution: {self.name}",
            extra={"agent": self.name, "state_status": state.get("overall_status")},
        )

        try:
            # Validate state before processing
            self._validate_state(state)

            # Execute the agent-specific logic
            result = self._execute(state)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Update result with execution metadata
            result.execution_time = execution_time
            result.timestamp = datetime.now()

            logger.info(
                f"Agent completed successfully: {self.name}",
                extra={
                    "agent": self.name,
                    "status": result.status,
                    "confidence": result.confidence,
                    "execution_time": execution_time,
                },
            )

            # Add result to state
            from src.graph.state import add_validation_result

            return add_validation_result(state, result)

        except Exception as e:
            execution_time = time.time() - start_time

            logger.error(
                f"Agent execution failed: {self.name}",
                extra={
                    "agent": self.name,
                    "error": str(e),
                    "execution_time": execution_time,
                },
                exc_info=True,
            )

            # Create error detail
            error = ErrorDetail(
                error_type=type(e).__name__,
                message=str(e),
                validator_name=self.name,
                timestamp=datetime.now(),
                context={
                    "execution_time": execution_time,
                    "state_status": state.get("overall_status"),
                },
                recoverable=self._is_recoverable_error(e),
            )

            # Add error to state
            from src.graph.state import add_error

            return add_error(state, error)

    @abstractmethod
    def _execute(self, state: ValidationState) -> ValidationResult:
        """Execute the agent-specific validation logic.

        This method must be implemented by subclasses to provide
        their specific validation functionality.

        Args:
            state: Current validation state

        Returns:
            ValidationResult with findings and recommendations

        Raises:
            Any exception that occurs during validation
        """
        pass

    def _validate_state(self, state: ValidationState) -> None:
        """Validate that the state contains required fields.

        Args:
            state: Validation state to check

        Raises:
            ValueError: If required state fields are missing
        """
        required_fields = [
            "input_data",
            "validation_request",
            "validation_results",
            "errors",
        ]

        for field in required_fields:
            if field not in state:
                raise ValueError(f"Missing required state field: {field}")

        logger.debug(f"State validation passed for agent: {self.name}")

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if an error is recoverable and can be retried.

        Args:
            error: Exception that occurred

        Returns:
            True if the error is recoverable, False otherwise
        """
        # Common recoverable errors (network, rate limits, temporary failures)
        recoverable_types = (
            "TimeoutError",
            "ConnectionError",
            "RateLimitError",
            "ServiceUnavailableError",
        )

        # Non-recoverable errors (validation, syntax, configuration)
        non_recoverable_types = (
            "ValueError",
            "TypeError",
            "KeyError",
            "AttributeError",
        )

        error_type = type(error).__name__

        if error_type in non_recoverable_types:
            return False
        if error_type in recoverable_types:
            return True

        # Default to recoverable for unknown errors (safer for retry)
        return True

    def get_info(self) -> dict[str, Any]:
        """Get information about this agent.

        Returns:
            Dictionary containing agent metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }

    def supports_capability(self, capability: str) -> bool:
        """Check if this agent supports a specific capability.

        Args:
            capability: Capability to check for

        Returns:
            True if the capability is supported, False otherwise
        """
        return capability in self.capabilities

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name={self.name})"


class ValidationAgent(BaseAgent):
    """Specialized base class for validation agents.

    This class extends BaseAgent with validation-specific functionality,
    such as confidence calculation and finding categorization.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        min_confidence: float = 0.5,
    ):
        """Initialize validation agent.

        Args:
            name: Unique identifier for this agent
            description: Human-readable description
            capabilities: List of capabilities
            metadata: Additional configuration
            min_confidence: Minimum confidence threshold for passing validation
        """
        super().__init__(name, description, capabilities, metadata)
        self.min_confidence = min_confidence

    def calculate_confidence(
        self, findings: list[str], severity_scores: Optional[list[float]] = None
    ) -> float:
        """Calculate confidence score based on findings.

        Args:
            findings: List of validation findings
            severity_scores: Optional severity scores for each finding (0.0-1.0)

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not findings:
            return 1.0  # No findings = high confidence

        if severity_scores:
            if len(severity_scores) != len(findings):
                raise ValueError("severity_scores must match findings length")
            avg_severity = sum(severity_scores) / len(severity_scores)
            return max(0.0, 1.0 - avg_severity)

        # Default: reduce confidence based on number of findings
        # More findings = lower confidence
        confidence = max(0.0, 1.0 - (len(findings) * 0.1))
        return confidence

    def categorize_findings(
        self, findings: list[str]
    ) -> dict[str, list[str]]:
        """Categorize findings by severity or type.

        This is a basic implementation. Subclasses can override for
        more sophisticated categorization.

        Args:
            findings: List of validation findings

        Returns:
            Dictionary mapping categories to findings
        """
        categorized = {
            "critical": [],
            "warning": [],
            "info": [],
        }

        for finding in findings:
            finding_lower = finding.lower()
            if any(
                keyword in finding_lower
                for keyword in ["error", "critical", "must", "required"]
            ):
                categorized["critical"].append(finding)
            elif any(
                keyword in finding_lower for keyword in ["warning", "should", "recommend"]
            ):
                categorized["warning"].append(finding)
            else:
                categorized["info"].append(finding)

        return categorized
