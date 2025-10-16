# Task 2: Supervisor Agent
## Implementation Plan

## Objective
Implement the top-level supervisor agent that analyzes validation requests, determines required validators, and orchestrates the workflow through LangGraph's conditional routing.

## Dependencies
- Task 1: Core Infrastructure (requires BaseAgent, ValidationState)

## Components to Implement

### 2.1 Supervisor Agent Core
**File**: `src/agents/supervisor.py`
**Actions**:
- Extend `BaseAgent` class
- Implement task analysis logic using LLM
- Create validation request parser
- Build validator selection algorithm
- Implement state initialization logic
- Add workflow orchestration hooks

### 2.2 Routing Decision System
**File**: `src/graph/routing.py`
**Actions**:
- Implement `route_to_validators()` function for LangGraph conditional edges
- Create routing decision logic based on supervisor output
- Support parallel routing (multiple validators)
- Support sequential routing (when dependencies exist)
- Handle routing to aggregator when all validators complete
- Add routing decision logging

### 2.3 Task Analysis Prompt Engineering
**File**: `src/agents/prompts/supervisor_prompts.py`
**Actions**:
- Design prompt for task decomposition
- Create few-shot examples for validator selection
- Define output structure for routing decisions
- Add prompt for progress monitoring

**Innovation**: Use structured output with JSON schema to ensure reliable routing decisions

### 2.4 Validator Registry
**File**: `src/agents/registry.py`
**Actions**:
- Create validator registry to track available validators
- Store validator capabilities and metadata
- Implement validator lookup by capability
- Support dynamic validator registration
- Add validator health checking

## Testing Strategy

### Unit Tests
**File**: `tests/test_supervisor.py`
- Test task analysis with various input types
- Test validator selection logic
- Test state initialization
- Mock LLM responses for deterministic testing

**File**: `tests/test_routing.py`
- Test routing decisions for single validator
- Test routing decisions for multiple parallel validators
- Test routing to aggregator
- Test error handling in routing

### Integration Tests
**File**: `tests/test_supervisor_integration.py`
- Test supervisor within LangGraph workflow
- Test end-to-end routing decisions
- Test state transitions through supervisor

## Technical Specifications

### Supervisor Agent Interface
```python
class SupervisorAgent(BaseAgent):
    """Top-level agent that analyzes requests and routes to validators."""

    def __init__(self, llm, validator_registry: ValidatorRegistry):
        super().__init__(name="supervisor", description="Request analysis and routing")
        self.llm = llm
        self.validator_registry = validator_registry

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Analyze validation request and determine required validators.

        Updates state with:
        - active_validators: list of validators to execute
        - overall_status: "in_progress"
        """
        # 1. Parse validation request
        # 2. Analyze input data characteristics
        # 3. Query validator registry for capabilities
        # 4. Use LLM to select appropriate validators
        # 5. Update state with routing decisions
        pass
```

### Routing Function
```python
def route_to_validators(state: ValidationState) -> str | list[str]:
    """
    Determine next node(s) in workflow based on state.

    Returns:
        - Validator name(s) if validators pending
        - "aggregator" if all validators complete
        - "error_handler" if errors need handling
    """
    if state["errors"] and should_handle_errors(state):
        return "error_handler"

    if state["active_validators"]:
        # Return next validator(s) to execute
        return get_next_validators(state)

    if all_validators_complete(state):
        return "aggregator"

    return END
```

### Supervisor Prompt Template
```python
SUPERVISOR_PROMPT = """You are a supervisor agent in a hierarchical validation system.

Your task is to analyze the validation request and determine which validators are needed.

Available Validators:
{validator_capabilities}

Validation Request:
{validation_request}

Input Data Sample:
{input_data_sample}

Analyze the request and select the appropriate validators. Consider:
1. What aspects need validation (schema, business rules, quality, etc.)
2. What validators are available
3. Dependencies between validators
4. Optimal execution order

Output your decision as JSON:
{{
    "validators": ["validator1", "validator2"],
    "execution_mode": "parallel" | "sequential",
    "reasoning": "explanation of your decision"
}}
"""
```

## Innovation Highlights

1. **LLM-based Dynamic Routing**: Supervisor uses LLM to intelligently select validators based on request characteristics
2. **Capability-based Selection**: Matches validation needs to validator capabilities automatically
3. **Adaptive Workflow**: Can handle new validator types without code changes
4. **Execution Mode Selection**: Determines optimal parallel vs sequential execution

## Acceptance Criteria

- ✅ Supervisor correctly analyzes validation requests
- ✅ Validator selection logic works for common scenarios
- ✅ Routing function properly integrates with LangGraph
- ✅ Supports both parallel and sequential validator execution
- ✅ State updates correctly reflect supervisor decisions
- ✅ All unit tests passing (>85% coverage)
- ✅ Integration tests show proper workflow orchestration

## Implementation Order

1. Create validator registry structure
2. Implement supervisor agent core logic
3. Design and test supervisor prompts
4. Implement routing decision function
5. Add integration with LangGraph conditional edges
6. Write comprehensive tests
7. Add monitoring and logging

## Estimated Complexity
**High** - Complex decision-making logic with LLM integration and workflow orchestration

## Notes
- Supervisor is the "brain" of the system - needs robust error handling
- Routing decisions must be deterministic for testing
- Consider caching common routing patterns for performance
- Prompt engineering critical for reliable validator selection
