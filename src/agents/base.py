"""Base agent class for all HVAS agents."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import logging

from src.graph.state import ValidationState


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the HVAS system.

    All agents (supervisor, validators, aggregator) extend this class
    and implement the execute method.
    """

    def __init__(self, name: str, description: str, capabilities: Optional[list[str]] = None):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for this agent
            description: Human-readable description of what this agent does
            capabilities: List of capabilities this agent provides
        """
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def execute(self, state: ValidationState) -> ValidationState:
        """
        Execute the agent's logic on the current state.

        This is the main entry point called by LangGraph when this node
        is executed in the workflow.

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        pass

    def __call__(self, state: ValidationState) -> ValidationState:
        """
        Make the agent callable for LangGraph integration.

        This allows the agent to be used directly as a node function.
        """
        self.logger.info(f"Executing agent: {self.name}")
        try:
            return self.execute(state)
        except Exception as e:
            self.logger.error(f"Error in agent {self.name}: {str(e)}", exc_info=True)
            # Add error to state
            from src.graph.state import create_error_detail
            state["errors"].append(
                create_error_detail(
                    error_type="agent_execution_error",
                    message=f"Error in {self.name}: {str(e)}",
                    validator=self.name,
                    context={"exception_type": type(e).__name__}
                )
            )
            return state

    def get_metadata(self) -> dict[str, Any]:
        """Get agent metadata for registry."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
