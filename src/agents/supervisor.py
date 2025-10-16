"""Supervisor agent for analyzing requests and routing to validators."""

import json
import logging
from typing import Any, Optional

from src.agents.base import BaseAgent
from src.agents.registry import ValidatorRegistry
from src.graph.state import ValidationState, create_error_detail
from src.agents.prompts.supervisor_prompts import (
    SUPERVISOR_SYSTEM_PROMPT,
    create_task_analysis_prompt,
)


logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Top-level agent that analyzes validation requests and routes to validators.

    The supervisor:
    1. Receives validation requests
    2. Analyzes input data characteristics
    3. Queries the validator registry
    4. Uses LLM to select appropriate validators
    5. Updates state with routing decisions
    """

    def __init__(self, llm: Any, validator_registry: ValidatorRegistry):
        """
        Initialize the supervisor agent.

        Args:
            llm: Language model for task analysis (LangChain LLM)
            validator_registry: Registry of available validators
        """
        super().__init__(
            name="supervisor",
            description="Analyzes validation requests and routes to appropriate validators",
            capabilities=["task_analysis", "validator_selection", "workflow_orchestration"]
        )
        self.llm = llm
        self.validator_registry = validator_registry

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Analyze validation request and determine required validators.

        Args:
            state: Current validation state

        Returns:
            Updated state with:
            - active_validators: list of validators to execute
            - pending_validators: ordered list of validators to run
            - overall_status: "in_progress"
            - workflow_metadata: execution mode and reasoning
        """
        self.logger.info("Supervisor analyzing validation request")

        try:
            # Extract request details
            validation_request = state.get("validation_request", {})
            input_data = state.get("input_data", {})

            # Check if this is initial analysis or progress monitoring
            if state["overall_status"] == "pending":
                # Initial task analysis
                decision = self._analyze_task(validation_request, input_data)
            else:
                # Progress monitoring (for future enhancement)
                decision = self._monitor_progress(state)

            # Update state with routing decisions
            state = self._update_state_with_decision(state, decision)

            self.logger.info(
                f"Supervisor selected validators: {decision.get('validators', [])} "
                f"in {decision.get('execution_mode', 'unknown')} mode"
            )

        except Exception as e:
            self.logger.error(f"Error in supervisor analysis: {str(e)}", exc_info=True)
            state["errors"].append(
                create_error_detail(
                    error_type="supervisor_analysis_error",
                    message=f"Failed to analyze task: {str(e)}",
                    validator="supervisor",
                    context={"exception_type": type(e).__name__}
                )
            )
            state["overall_status"] = "failed"

        return state

    def _analyze_task(
        self,
        validation_request: dict[str, Any],
        input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Analyze validation task using LLM.

        Args:
            validation_request: The validation request details
            input_data: Input data to validate

        Returns:
            Dictionary with validator selection decision
        """
        # Get available validators
        validators_metadata = self.validator_registry.get_validator_metadata()

        if not validators_metadata:
            self.logger.warning("No validators available in registry")
            return {
                "validators": [],
                "execution_mode": "parallel",
                "reasoning": "No validators available",
                "priority_order": []
            }

        # Create prompt for LLM
        prompt = create_task_analysis_prompt(
            validation_request=validation_request,
            input_data_sample=self._create_data_sample(input_data),
            validators_metadata=validators_metadata
        )

        # Query LLM for validator selection
        try:
            response = self._query_llm(prompt)
            decision = self._parse_llm_response(response)

            # Validate the decision
            decision = self._validate_decision(decision, validators_metadata)

            return decision

        except Exception as e:
            self.logger.error(f"Error querying LLM: {str(e)}")
            # Fallback: use all available validators in sequential mode
            return self._create_fallback_decision(validators_metadata)

    def _query_llm(self, prompt: str) -> str:
        """
        Query the LLM with the given prompt.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response as string
        """
        # Build messages for chat model
        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Invoke LLM
        try:
            # Support both LangChain chat models and completion models
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(messages)
                if hasattr(response, "content"):
                    return response.content
                return str(response)
            else:
                # Fallback for older LLM interfaces
                return self.llm(prompt)
        except Exception as e:
            self.logger.error(f"LLM invocation error: {str(e)}")
            raise

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """
        Parse LLM response into structured decision.

        Args:
            response: Raw LLM response

        Returns:
            Parsed decision dictionary
        """
        # Extract JSON from response
        # LLM might include markdown code blocks
        response = response.strip()

        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        try:
            decision = json.loads(response)
            return decision
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            self.logger.debug(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")

    def _validate_decision(
        self,
        decision: dict[str, Any],
        validators_metadata: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate and sanitize the LLM's decision.

        Args:
            decision: Decision from LLM
            validators_metadata: Available validators

        Returns:
            Validated decision
        """
        available_validators = {v["name"] for v in validators_metadata}
        selected_validators = decision.get("validators", [])

        # Filter out invalid validators
        valid_validators = [v for v in selected_validators if v in available_validators]

        if len(valid_validators) < len(selected_validators):
            invalid = set(selected_validators) - available_validators
            self.logger.warning(f"Removing invalid validators: {invalid}")

        decision["validators"] = valid_validators

        # Ensure execution_mode is valid
        if decision.get("execution_mode") not in ["parallel", "sequential"]:
            self.logger.warning(f"Invalid execution mode, defaulting to sequential")
            decision["execution_mode"] = "sequential"

        # Ensure priority_order matches validators and filter out invalid ones
        priority_order = decision.get("priority_order", [])
        valid_priority_order = [v for v in priority_order if v in available_validators]

        if not valid_priority_order:
            decision["priority_order"] = valid_validators
        else:
            decision["priority_order"] = valid_priority_order

        return decision

    def _create_fallback_decision(
        self,
        validators_metadata: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Create fallback decision when LLM fails.

        Args:
            validators_metadata: Available validators

        Returns:
            Fallback decision using all validators
        """
        validator_names = [v["name"] for v in validators_metadata]

        return {
            "validators": validator_names,
            "execution_mode": "sequential",
            "reasoning": "Fallback decision: using all available validators in sequential mode",
            "priority_order": validator_names
        }

    def _create_data_sample(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Create a sample of input data for prompt.

        Args:
            input_data: Full input data

        Returns:
            Sample or summary of input data
        """
        # If data is small, return as-is
        data_str = json.dumps(input_data)
        if len(data_str) <= 1000:
            return input_data

        # Otherwise, create a summary
        sample: dict[str, Any] = {}

        if isinstance(input_data, dict):
            # Include first few keys and their types
            for i, (key, value) in enumerate(input_data.items()):
                if i >= 5:
                    sample["..."] = f"({len(input_data) - 5} more keys)"
                    break
                sample[key] = self._summarize_value(value)

        return sample

    def _summarize_value(self, value: Any) -> Any:
        """Summarize a value for the sample."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, list):
            if len(value) <= 3:
                return value
            return [value[0], "...", f"({len(value)} items total)"]
        elif isinstance(value, dict):
            return {k: self._summarize_value(v) for k, v in list(value.items())[:3]}
        else:
            return f"<{type(value).__name__}>"

    def _monitor_progress(self, state: ValidationState) -> dict[str, Any]:
        """
        Monitor progress and determine next steps.

        This is for future enhancement. Currently just returns
        decision to continue with pending validators.

        Args:
            state: Current validation state

        Returns:
            Decision for next steps
        """
        # For now, just continue with pending validators
        return {
            "validators": state.get("pending_validators", []),
            "execution_mode": state.get("workflow_metadata", {}).get("execution_mode", "sequential"),
            "reasoning": "Continuing with pending validators",
            "priority_order": state.get("pending_validators", [])
        }

    def _update_state_with_decision(
        self,
        state: ValidationState,
        decision: dict[str, Any]
    ) -> ValidationState:
        """
        Update state with supervisor's decision.

        Args:
            state: Current state
            decision: Supervisor's decision

        Returns:
            Updated state
        """
        validators = decision.get("validators", [])
        execution_mode = decision.get("execution_mode", "sequential")
        priority_order = decision.get("priority_order", validators)

        # Set status to in_progress
        state["overall_status"] = "in_progress"

        # Set pending validators based on priority order
        state["pending_validators"] = priority_order.copy()

        # For sequential execution, set first validator as active
        # For parallel execution, all validators are active
        if execution_mode == "sequential":
            if priority_order:
                state["active_validators"] = [priority_order[0]]
        else:
            state["active_validators"] = validators.copy()

        # Store decision metadata
        state["workflow_metadata"]["execution_mode"] = execution_mode
        state["workflow_metadata"]["reasoning"] = decision.get("reasoning", "")
        state["workflow_metadata"]["supervisor_decision"] = decision

        return state
