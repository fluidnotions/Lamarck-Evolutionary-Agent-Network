# HVAS-Mini: Hierarchical Validation Agent System

A hierarchical multi-agent validation framework built on LangGraph that orchestrates complex validation tasks through specialized agents.

## Overview

HVAS-Mini implements a tiered agent architecture where:
- A **Supervisor Agent** analyzes validation requests and routes to appropriate validators
- **Domain Validators** perform specialized validation tasks
- An **Aggregator Agent** synthesizes results into comprehensive reports
- **LangGraph** manages state and orchestrates the workflow

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Supervisor Agent                        │
│            (LLM-based Task Analysis)                     │
└─────────────┬───────────────────────────────┬───────────┘
              │                               │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  Schema Validator │         │ Business Rules    │
    │                   │         │   Validator       │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                               │
              └───────────┬───────────────────┘
                          │
                ┌─────────▼─────────┐
                │  Aggregator Agent │
                │  (Result Synthesis)│
                └───────────────────┘
```

## Task 2: Supervisor Agent Implementation

This implementation includes:

### Core Components

1. **Supervisor Agent** (`src/agents/supervisor.py`)
   - Analyzes validation requests using LLM
   - Selects appropriate validators from registry
   - Determines execution mode (parallel/sequential)
   - Updates workflow state with routing decisions

2. **Validator Registry** (`src/agents/registry.py`)
   - Maintains registry of available validators
   - Capability-based validator lookup
   - Dynamic validator registration

3. **Routing Logic** (`src/graph/routing.py`)
   - Conditional edge routing for LangGraph
   - Supports both parallel and sequential execution
   - Handles workflow transitions

4. **Prompt Engineering** (`src/agents/prompts/supervisor_prompts.py`)
   - LLM prompts for task analysis
   - Structured JSON output format
   - Few-shot examples for validator selection

### Base Infrastructure (Task 1)

5. **State Management** (`src/graph/state.py`)
   - ValidationState TypedDict for workflow state
   - ValidationResult and ErrorDetail types
   - State creation utilities

6. **Base Agent** (`src/agents/base.py`)
   - Abstract base class for all agents
   - Callable interface for LangGraph integration
   - Error handling and logging

## Installation

```bash
# Install dependencies using UV
uv sync

# Or install with dev dependencies
uv sync --extra dev
```

## Usage

### Basic Workflow

```python
from langgraph.graph import StateGraph
from src.agents.supervisor import SupervisorAgent
from src.agents.registry import ValidatorRegistry
from src.graph.state import create_initial_state
from src.graph.routing import route_to_validators

# Setup registry and register validators
registry = ValidatorRegistry()
registry.register(schema_validator)
registry.register(business_validator)

# Create supervisor with LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
supervisor = SupervisorAgent(llm, registry)

# Create LangGraph workflow
workflow = StateGraph(ValidationState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("schema_validator", schema_validator)
workflow.add_node("business_rules", business_validator)
workflow.add_node("aggregator", aggregator)

# Add conditional routing
workflow.add_conditional_edges("supervisor", route_to_validators, {
    "schema_validator": "schema_validator",
    "business_rules": "business_rules",
    "aggregator": "aggregator"
})

workflow.set_entry_point("supervisor")
app = workflow.compile()

# Execute validation
initial_state = create_initial_state(
    input_data={"user": "test", "email": "test@example.com"},
    validation_request={"type": "user_validation"}
)

result = app.invoke(initial_state)
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_supervisor.py

# Run integration tests only
uv run pytest tests/integration/
```

## Key Features

### 1. LLM-based Dynamic Routing
The supervisor uses an LLM to intelligently analyze validation requests and select appropriate validators:

```python
# Supervisor analyzes request and selects validators
decision = supervisor.execute(state)
# Returns: {"validators": ["schema", "business"], "execution_mode": "sequential"}
```

### 2. Flexible Execution Modes
- **Sequential**: Validators run one after another (with dependencies)
- **Parallel**: Independent validators run concurrently

### 3. Capability-based Selection
Validators register their capabilities, and the supervisor matches them to task requirements:

```python
registry.register(validator)
# Validator capabilities: ["schema", "json", "structure"]

# Supervisor automatically selects validators with required capabilities
```

### 4. Comprehensive State Management
All workflow state is maintained in a single `ValidationState` object:
- Active, pending, and completed validators
- Validation results and errors
- Execution metadata and decisions

## Project Structure

```
hvas-mini/
├── src/
│   ├── agents/
│   │   ├── base.py              # Base agent class
│   │   ├── supervisor.py        # Supervisor agent
│   │   ├── registry.py          # Validator registry
│   │   └── prompts/
│   │       └── supervisor_prompts.py  # LLM prompts
│   └── graph/
│       ├── state.py             # State definitions
│       └── routing.py           # Routing logic
├── tests/
│   ├── unit/
│   │   ├── test_supervisor.py   # Supervisor tests
│   │   ├── test_registry.py     # Registry tests
│   │   └── test_routing.py      # Routing tests
│   └── integration/
│       └── test_supervisor_integration.py  # LangGraph integration tests
├── pyproject.toml
└── README.md
```

## Implementation Highlights

### Supervisor Decision Making

The supervisor uses structured prompts with JSON schema output to ensure reliable routing decisions:

```python
# Prompt includes:
# - Available validators and capabilities
# - Validation request details
# - Input data sample

# LLM returns structured decision:
{
    "validators": ["schema_validator", "business_rules"],
    "execution_mode": "sequential",
    "reasoning": "Schema must pass before business rules can be validated",
    "priority_order": ["schema_validator", "business_rules"]
}
```

### Routing Logic

The routing function integrates with LangGraph's conditional edges:

```python
def route_to_validators(state: ValidationState) -> str | list[str]:
    # Check for errors -> END
    # Check for active validators -> return next validator(s)
    # Check if complete -> "aggregator"
    # Otherwise -> END
```

### Error Handling

All agents include comprehensive error handling with detailed error context:

```python
try:
    result = agent.execute(state)
except Exception as e:
    # Add structured error to state
    state["errors"].append(create_error_detail(
        error_type="agent_execution_error",
        message=str(e),
        validator=agent.name,
        context={"exception_type": type(e).__name__}
    ))
```

## Test Coverage

- **Unit Tests**: 85%+ coverage for all components
- **Integration Tests**: End-to-end workflow testing with LangGraph
- **Mocked LLM**: Deterministic testing with mock LLM responses

## Future Enhancements

- [ ] Progress monitoring and adaptive routing
- [ ] Advanced error recovery strategies
- [ ] Caching for common routing patterns
- [ ] Human-in-the-loop validation approval
- [ ] Performance metrics and analytics

## Contributing

This is part of the HVAS-Mini project. See the main project documentation for contribution guidelines.

## License

MIT License
