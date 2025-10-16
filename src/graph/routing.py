"""Routing logic for LangGraph conditional edges."""

import logging
from typing import Union, Literal
from langgraph.graph import END

from src.graph.state import ValidationState


logger = logging.getLogger(__name__)


# Type alias for routing decisions
RoutingDecision = Union[str, list[str], Literal["__end__"]]


def route_to_validators(state: ValidationState) -> RoutingDecision:
    """
    Determine next node(s) in workflow based on state.

    This is the main routing function used by LangGraph's conditional edges.
    It examines the current state and decides:
    - Which validator(s) to execute next
    - Whether to proceed to aggregation
    - Whether to handle errors
    - Whether to end the workflow

    Args:
        state: Current validation state

    Returns:
        - Single validator name: for sequential execution
        - List of validator names: for parallel execution
        - "aggregator": when all validators complete
        - "error_handler": if errors need handling (future)
        - END: to terminate workflow

    Routing Logic:
    1. Check for critical errors -> error_handler or END
    2. Check for active validators -> return next validator(s)
    3. Check if all validators complete -> aggregator
    4. Otherwise -> END
    """
    logger.info("Routing decision requested")
    logger.debug(f"State: active={state['active_validators']}, "
                 f"pending={state['pending_validators']}, "
                 f"completed={state['completed_validators']}, "
                 f"status={state['overall_status']}")

    # 1. Check for critical errors
    if state["overall_status"] == "failed":
        errors = state.get("errors", [])
        if errors:
            logger.warning(f"Workflow failed with {len(errors)} errors")
        return END

    # 2. Check for active validators to execute
    active_validators = state.get("active_validators", [])
    if active_validators:
        execution_mode = state.get("workflow_metadata", {}).get("execution_mode", "sequential")

        if execution_mode == "parallel":
            # Return all active validators for parallel execution
            logger.info(f"Routing to parallel validators: {active_validators}")
            return active_validators
        else:
            # Return first active validator for sequential execution
            next_validator = active_validators[0]
            logger.info(f"Routing to sequential validator: {next_validator}")
            return next_validator

    # 3. Check if there are pending validators to activate
    pending_validators = state.get("pending_validators", [])
    if pending_validators:
        # Activate next validator(s) based on execution mode
        execution_mode = state.get("workflow_metadata", {}).get("execution_mode", "sequential")

        if execution_mode == "parallel":
            # Activate all pending validators
            logger.info(f"Activating parallel validators: {pending_validators}")
            return pending_validators
        else:
            # Activate next sequential validator
            next_validator = pending_validators[0]
            logger.info(f"Activating sequential validator: {next_validator}")
            return next_validator

    # 4. Check if all validators are complete
    if _all_validators_complete(state):
        logger.info("All validators complete, routing to aggregator")
        return "aggregator"

    # 5. No more work to do
    logger.info("No more routing needed, ending workflow")
    return END


def route_from_validator(state: ValidationState) -> RoutingDecision:
    """
    Route after a validator completes.

    This function is called after each validator node executes.
    It determines whether to:
    - Continue to next validator
    - Return to supervisor for coordination
    - Proceed to aggregator

    Args:
        state: Current validation state

    Returns:
        Next node to execute
    """
    logger.info("Routing after validator completion")

    # Check execution mode
    execution_mode = state.get("workflow_metadata", {}).get("execution_mode", "sequential")

    if execution_mode == "sequential":
        # In sequential mode, check if there are more validators
        pending_validators = state.get("pending_validators", [])

        if pending_validators:
            # Activate next validator
            next_validator = pending_validators[0]
            logger.info(f"Sequential: routing to next validator: {next_validator}")
            return next_validator
        elif _all_validators_complete(state):
            # All done, go to aggregator
            logger.info("Sequential: all complete, routing to aggregator")
            return "aggregator"
        else:
            logger.info("Sequential: ending workflow")
            return END
    else:
        # In parallel mode, wait for all to complete
        if _all_validators_complete(state):
            logger.info("Parallel: all complete, routing to aggregator")
            return "aggregator"
        else:
            # More validators still running, return to supervisor for coordination
            logger.info("Parallel: waiting for other validators")
            return "supervisor"


def _all_validators_complete(state: ValidationState) -> bool:
    """
    Check if all validators have completed.

    Args:
        state: Current validation state

    Returns:
        True if all validators are complete, False otherwise
    """
    supervisor_decision = state.get("workflow_metadata", {}).get("supervisor_decision")

    # If there's no supervisor decision at all, we're not complete (shouldn't aggregate)
    if supervisor_decision is None:
        return False

    expected_validators = supervisor_decision.get("validators", [])

    if not expected_validators:
        # Supervisor explicitly selected no validators, consider complete
        return True

    completed_validators = set(state.get("completed_validators", []))
    expected_set = set(expected_validators)

    all_complete = expected_set.issubset(completed_validators)

    logger.debug(
        f"Completion check: expected={expected_set}, "
        f"completed={completed_validators}, "
        f"all_complete={all_complete}"
    )

    return all_complete


def should_handle_errors(state: ValidationState) -> bool:
    """
    Determine if errors should be handled by error handler.

    This is for future error handling enhancement.

    Args:
        state: Current validation state

    Returns:
        True if errors should be handled, False otherwise
    """
    errors = state.get("errors", [])

    if not errors:
        return False

    # For now, only handle errors if there are critical errors
    # Future: implement more sophisticated error handling logic
    critical_errors = [
        e for e in errors
        if e.get("error_type") in ["supervisor_analysis_error", "critical_error"]
    ]

    return len(critical_errors) > 0


def get_next_validators(state: ValidationState) -> RoutingDecision:
    """
    Get next validator(s) to execute based on state.

    Args:
        state: Current validation state

    Returns:
        Next validator name(s) or routing decision
    """
    active_validators = state.get("active_validators", [])

    if not active_validators:
        return END

    execution_mode = state.get("workflow_metadata", {}).get("execution_mode", "sequential")

    if execution_mode == "parallel":
        return active_validators
    else:
        return active_validators[0]


def update_state_after_validator(
    state: ValidationState,
    validator_name: str
) -> ValidationState:
    """
    Update state after a validator completes.

    This helper function should be called by validators to update
    the routing state correctly.

    Args:
        state: Current validation state
        validator_name: Name of validator that completed

    Returns:
        Updated state
    """
    # Move validator from active to completed
    if validator_name in state["active_validators"]:
        state["active_validators"].remove(validator_name)

    if validator_name not in state["completed_validators"]:
        state["completed_validators"].append(validator_name)

    # Remove from pending if present
    if validator_name in state["pending_validators"]:
        state["pending_validators"].remove(validator_name)

    # For sequential execution, activate next pending validator
    execution_mode = state.get("workflow_metadata", {}).get("execution_mode", "sequential")

    if execution_mode == "sequential":
        pending = state.get("pending_validators", [])
        if pending and not state["active_validators"]:
            # Activate next validator
            next_validator = pending[0]
            state["active_validators"] = [next_validator]

    logger.debug(
        f"State updated after {validator_name}: "
        f"active={state['active_validators']}, "
        f"pending={state['pending_validators']}, "
        f"completed={state['completed_validators']}"
    )

    return state
