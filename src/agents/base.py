"""Base agent class for all validation agents."""
import time
import logging
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.graph.state import ValidationState
from src.models.validation_result import ValidationResult, ErrorDetail
from src.utils.config import get_config
from src.utils.retry import retry_with_backoff, RetryConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all validation agents."""

    def __init__(
        self,
        name: str,
        llm: Optional[BaseChatModel] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize base agent.

        Args:
            name: Agent name
            llm: Optional LLM instance (creates default if None)
            config: Optional agent-specific configuration
        """
        self.name = name
        self.config = config or {}
        self.app_config = get_config()

        if llm is None:
            llm = self._create_default_llm()
        self.llm = llm

        logger.info(f"Initialized agent: {name}")

    def _create_default_llm(self) -> BaseChatModel:
        """Create default LLM based on configuration.

        Returns:
            LLM instance
        """
        llm_config = self.app_config.llm

        if llm_config.provider == "anthropic":
            return ChatAnthropic(
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
            )
        else:
            return ChatOpenAI(
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                timeout=llm_config.timeout,
            )

    @abstractmethod
    def process(self, state: ValidationState) -> Dict[str, Any]:
        """Process the validation state and return updates.

        Args:
            state: Current validation state

        Returns:
            Dictionary of state updates
        """
        pass

    def __call__(self, state: ValidationState) -> Dict[str, Any]:
        """Make agent callable for LangGraph integration.

        Args:
            state: Current validation state

        Returns:
            Dictionary of state updates
        """
        start_time = time.time()
        logger.info(f"Agent {self.name} starting execution")

        try:
            # Add retry logic
            retry_config = RetryConfig(
                max_retries=self.app_config.validation.max_retries
            )

            @retry_with_backoff(config=retry_config)
            def process_with_retry() -> Dict[str, Any]:
                return self.process(state)

            updates = process_with_retry()

            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Agent {self.name} completed in {execution_time:.2f}ms"
            )

            # Add execution time to updates if result is present
            if "validation_results" in updates and updates["validation_results"]:
                for result in updates["validation_results"]:
                    if isinstance(result, ValidationResult):
                        result.execution_time_ms = execution_time

            return updates

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(
                f"Agent {self.name} failed after {execution_time:.2f}ms: {e}",
                exc_info=True,
            )

            # Return error result
            error_result = ValidationResult(
                validator_name=self.name,
                status="error",
                confidence=0.0,
                errors=[
                    ErrorDetail(
                        path="agent",
                        message=str(e),
                        code="AGENT_ERROR",
                        severity="error",
                        context={"agent": self.name, "exception_type": type(e).__name__},
                    )
                ],
                execution_time_ms=execution_time,
            )

            return {
                "validation_results": [error_result],
                "errors": error_result.errors,
                "completed_validators": [self.name],
            }

    def create_result(
        self,
        status: str,
        confidence: float,
        errors: Optional[list] = None,
        warnings: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Helper to create a validation result.

        Args:
            status: Validation status
            confidence: Confidence score 0-1
            errors: Optional list of errors
            warnings: Optional list of warnings
            metadata: Optional metadata

        Returns:
            ValidationResult instance
        """
        return ValidationResult(
            validator_name=self.name,
            status=status,
            confidence=confidence,
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata or {},
        )
