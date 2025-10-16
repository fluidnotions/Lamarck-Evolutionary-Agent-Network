"""Routing logic for the validation workflow."""
from typing import Literal
import logging

from src.graph.state import ValidationState, should_continue_validation

logger = logging.getLogger(__name__)


def route_from_supervisor(
    state: ValidationState,
) -> Literal["schema_validator", "business_rules", "data_quality", "aggregator"]:
    """Route from supervisor to next validator or aggregator.

    Args:
        state: Current validation state

    Returns:
        Name of next node to execute
    """
    pending = state.get("pending_validators", [])
    completed = state.get("completed_validators", [])

    logger.debug(f"Routing: pending={pending}, completed={completed}")

    # If no more validators, go to aggregator
    if not pending:
        logger.info("No pending validators, routing to aggregator")
        return "aggregator"

    # Get next validator to run
    next_validator = pending[0]

    logger.info(f"Routing to next validator: {next_validator}")

    # Map to node names
    if next_validator in ["schema_validator", "schema"]:
        return "schema_validator"
    elif next_validator in ["business_rules", "business", "rules"]:
        return "business_rules"
    elif next_validator in ["data_quality", "quality", "dq"]:
        return "data_quality"
    else:
        # Default to aggregator if unknown validator
        logger.warning(f"Unknown validator {next_validator}, routing to aggregator")
        return "aggregator"


def route_from_validator(
    state: ValidationState,
) -> Literal["schema_validator", "business_rules", "data_quality", "aggregator", "END"]:
    """Route from validator to next validator or aggregator.

    Args:
        state: Current validation state

    Returns:
        Name of next node to execute or END
    """
    pending = state.get("pending_validators", [])

    logger.debug(f"Routing from validator: pending={pending}")

    # If no more validators, go to aggregator
    if not pending:
        logger.info("No more validators, routing to aggregator")
        return "aggregator"

    # Get next validator
    next_validator = pending[0]

    logger.info(f"Routing to next validator: {next_validator}")

    # Map to node names
    if next_validator in ["schema_validator", "schema"]:
        return "schema_validator"
    elif next_validator in ["business_rules", "business", "rules"]:
        return "business_rules"
    elif next_validator in ["data_quality", "quality", "dq"]:
        return "data_quality"
    else:
        # If unknown, go to aggregator
        return "aggregator"


def should_end(state: ValidationState) -> Literal["END", "supervisor"]:
    """Determine if workflow should end.

    Args:
        state: Current validation state

    Returns:
        "END" if done, "supervisor" to continue
    """
    current_step = state.get("current_step", "")
    overall_status = state.get("overall_status", "")

    # End if we've completed aggregation
    if current_step == "completed":
        logger.info("Workflow complete")
        return "END"

    # End if status is failed and fail_fast is enabled
    if overall_status == "failed":
        logger.info("Workflow failed")
        return "END"

    # Continue otherwise
    return "supervisor"
