"""Base agent class for HVAS-Mini."""

from abc import ABC, abstractmethod
from typing import Any
from src.graph.state import ValidationState


class BaseAgent(ABC):
    """Base class for all agents in HVAS-Mini."""

    def __init__(self, name: str, description: str):
        """
        Initialize base agent.

        Args:
            name: Agent name
            description: Agent description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, state: ValidationState) -> ValidationState:
        """
        Execute the agent's task.

        Args:
            state: Current validation state

        Returns:
            Updated validation state
        """
        pass

    def __call__(self, state: ValidationState) -> ValidationState:
        """Make agent callable for LangGraph integration."""
        return self.execute(state)
