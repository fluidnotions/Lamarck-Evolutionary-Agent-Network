"""Supervisor agent for orchestrating validation."""
from typing import Any, Dict, List
import logging

from src.agents.base import BaseAgent
from src.graph.state import ValidationState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Supervisor agent that orchestrates the validation workflow."""

    def __init__(self, **kwargs: Any):
        """Initialize supervisor agent."""
        super().__init__(name="supervisor", **kwargs)

    def process(self, state: ValidationState) -> Dict[str, Any]:
        """Analyze validation request and determine validators to run.

        Args:
            state: Current validation state

        Returns:
            State updates with validators to run
        """
        logger.info("Supervisor analyzing validation request")

        validation_request = state.get("validation_request", {})
        requested_validators = validation_request.get("validators", [])

        # Determine which validators to run
        validators_to_run = self._determine_validators(state, requested_validators)

        logger.info(f"Supervisor determined validators to run: {validators_to_run}")

        return {
            "pending_validators": validators_to_run,
            "current_step": "validation",
            "overall_status": "in_progress",
        }

    def _determine_validators(
        self, state: ValidationState, requested: List[str]
    ) -> List[str]:
        """Determine which validators should run.

        Args:
            state: Current validation state
            requested: Requested validator names

        Returns:
            List of validator names to run
        """
        # Map of validator aliases to canonical names
        validator_map = {
            "schema": "schema_validator",
            "schema_validator": "schema_validator",
            "business": "business_rules",
            "business_rules": "business_rules",
            "rules": "business_rules",
            "quality": "data_quality",
            "data_quality": "data_quality",
            "dq": "data_quality",
        }

        validators = []
        for name in requested:
            canonical_name = validator_map.get(name.lower(), name)
            if canonical_name not in validators:
                validators.append(canonical_name)

        # If no validators specified, run all
        if not validators:
            validators = ["schema_validator", "business_rules", "data_quality"]

        return validators
