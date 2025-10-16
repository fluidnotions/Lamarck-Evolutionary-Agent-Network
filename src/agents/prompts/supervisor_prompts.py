"""Prompt templates for the supervisor agent."""

from typing import Any


SUPERVISOR_SYSTEM_PROMPT = """You are a supervisor agent in a hierarchical validation system (HVAS-Mini).

Your role is to analyze validation requests and determine which specialized validators are needed to complete the validation task.

You have access to a registry of validators, each with specific capabilities. Your job is to:
1. Understand what needs to be validated
2. Analyze the characteristics of the input data
3. Select the appropriate validators from the available registry
4. Determine if validators should run in parallel or sequentially
5. Provide clear reasoning for your decisions

Be thorough but efficient - select only the validators that are necessary for the task.
"""


TASK_ANALYSIS_PROMPT = """Analyze the following validation request and select the appropriate validators.

Available Validators:
{validator_capabilities}

Validation Request:
{validation_request}

Input Data Sample:
{input_data_sample}

Analyze the request and provide your decision in the following JSON format:
{{
    "validators": ["validator1", "validator2"],
    "execution_mode": "parallel" or "sequential",
    "reasoning": "Clear explanation of why you selected these validators and this execution mode",
    "priority_order": ["validator1", "validator2"]
}}

Guidelines:
- Select only validators that are necessary for the validation task
- Use "parallel" execution when validators are independent
- Use "sequential" execution when there are dependencies (e.g., schema must pass before business rules)
- Provide clear reasoning for your decisions
- Ensure all selected validators are from the available list
- Priority order matters for sequential execution

Your response (JSON only):"""


FEW_SHOT_EXAMPLES = """
Example 1:
Request: Validate user registration data
Available Validators: schema_validator, business_rules, data_quality
Response:
{
    "validators": ["schema_validator", "business_rules", "data_quality"],
    "execution_mode": "sequential",
    "reasoning": "Schema validation must pass first to ensure data structure is correct. Then business rules check domain constraints. Finally, data quality validates completeness and format. Sequential execution ensures dependencies are respected.",
    "priority_order": ["schema_validator", "business_rules", "data_quality"]
}

Example 2:
Request: Validate product catalog data
Available Validators: schema_validator, data_quality, cross_reference
Response:
{
    "validators": ["schema_validator", "data_quality", "cross_reference"],
    "execution_mode": "sequential",
    "reasoning": "Schema validation ensures structure. Data quality and cross-reference can check different aspects but cross-reference needs valid data first, so sequential execution is safer.",
    "priority_order": ["schema_validator", "data_quality", "cross_reference"]
}

Example 3:
Request: Quick schema validation only
Available Validators: schema_validator, business_rules, data_quality
Response:
{
    "validators": ["schema_validator"],
    "execution_mode": "parallel",
    "reasoning": "Request explicitly asks for schema validation only. No need for other validators.",
    "priority_order": ["schema_validator"]
}
"""


def format_validator_capabilities(validators_metadata: list[dict[str, Any]]) -> str:
    """
    Format validator metadata for the prompt.

    Args:
        validators_metadata: List of validator metadata dictionaries

    Returns:
        Formatted string describing available validators
    """
    if not validators_metadata:
        return "No validators available"

    lines = []
    for validator in validators_metadata:
        name = validator.get("name", "unknown")
        description = validator.get("description", "No description")
        capabilities = validator.get("capabilities", [])

        lines.append(f"- {name}:")
        lines.append(f"  Description: {description}")
        if capabilities:
            lines.append(f"  Capabilities: {', '.join(capabilities)}")

    return "\n".join(lines)


def create_task_analysis_prompt(
    validation_request: dict[str, Any],
    input_data_sample: dict[str, Any],
    validators_metadata: list[dict[str, Any]]
) -> str:
    """
    Create the complete task analysis prompt.

    Args:
        validation_request: The validation request details
        input_data_sample: Sample of input data to validate
        validators_metadata: Available validators and their capabilities

    Returns:
        Formatted prompt string
    """
    import json

    validator_capabilities_str = format_validator_capabilities(validators_metadata)

    # Truncate input data sample if too large
    input_sample_str = json.dumps(input_data_sample, indent=2)
    if len(input_sample_str) > 1000:
        input_sample_str = input_sample_str[:1000] + "\n... (truncated)"

    validation_request_str = json.dumps(validation_request, indent=2)

    return TASK_ANALYSIS_PROMPT.format(
        validator_capabilities=validator_capabilities_str,
        validation_request=validation_request_str,
        input_data_sample=input_sample_str
    )


PROGRESS_MONITORING_PROMPT = """You are monitoring the progress of a validation workflow.

Current State:
- Active Validators: {active_validators}
- Completed Validators: {completed_validators}
- Pending Validators: {pending_validators}
- Errors: {error_count}

Recent Results:
{recent_results}

Determine the next action:
1. Continue with pending validators
2. Retry failed validators
3. Proceed to aggregation
4. Handle errors

Provide your decision as JSON:
{{
    "action": "continue" | "retry" | "aggregate" | "handle_errors",
    "reason": "explanation",
    "validators_to_execute": ["validator1"]
}}
"""


def create_progress_monitoring_prompt(
    active_validators: list[str],
    completed_validators: list[str],
    pending_validators: list[str],
    error_count: int,
    recent_results: list[dict[str, Any]]
) -> str:
    """
    Create progress monitoring prompt.

    Args:
        active_validators: Currently active validators
        completed_validators: Completed validators
        pending_validators: Pending validators
        error_count: Number of errors
        recent_results: Recent validation results

    Returns:
        Formatted prompt string
    """
    import json

    recent_results_str = json.dumps(recent_results, indent=2)
    if len(recent_results_str) > 500:
        recent_results_str = recent_results_str[:500] + "\n... (truncated)"

    return PROGRESS_MONITORING_PROMPT.format(
        active_validators=", ".join(active_validators) if active_validators else "None",
        completed_validators=", ".join(completed_validators) if completed_validators else "None",
        pending_validators=", ".join(pending_validators) if pending_validators else "None",
        error_count=error_count,
        recent_results=recent_results_str
    )
