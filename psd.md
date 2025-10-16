# HVAS-Mini: Hierarchical Validation Agent System
## Product Specification & Architecture Document

## 1. Overview

HVAS-Mini is a hierarchical multi-agent system built on LangGraph that validates complex data through a tiered architecture of specialized agents. The system orchestrates validation tasks from high-level supervisory agents down to specialized validators, with built-in error handling, retry logic, and comprehensive reporting.

## 2. Product Vision

Create a scalable, hierarchical validation framework where:
- Complex validation tasks are decomposed hierarchically
- Specialized agents handle specific validation domains
- Supervisors coordinate and aggregate results
- The system is observable, debuggable, and extensible
- LangGraph's state management handles coordination

## 3. Core Features

### 3.1 Hierarchical Agent Structure
- **Supervisor Agent**: Top-level orchestrator that routes tasks
- **Domain Validators**: Mid-tier agents for specific validation domains
- **Atomic Validators**: Leaf nodes performing single validation checks
- **Aggregator Agent**: Collects and synthesizes validation results

### 3.2 Validation Domains
- **Schema Validation**: JSON/YAML schema compliance
- **Business Rules**: Domain-specific business logic
- **Data Quality**: Completeness, accuracy, consistency checks
- **Cross-Reference**: Inter-entity relationship validation

### 3.3 LangGraph Integration
- State-based workflow orchestration
- Conditional routing between agents
- Parallel execution of independent validators
- State persistence and recovery
- Built-in visualization support

### 3.4 Error Handling & Resilience
- Configurable retry logic with exponential backoff
- Graceful degradation on partial failures
- Detailed error reporting with context
- Validation result aggregation with confidence scores

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Supervisor Agent                      │
│              (Task Decomposition & Routing)              │
└─────────────┬───────────────────────────────┬───────────┘
              │                               │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  Schema Validator │         │ Business Rules    │
    │     Domain        │         │   Validator       │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                               │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │ JSON Schema Check │         │ Rule Engine Check │
    │  Atomic Validator │         │  Atomic Validator │
    └───────────────────┘         └───────────────────┘
                    │                       │
                    └───────────┬───────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Aggregator Agent │
                    │  (Result Synthesis)│
                    └───────────────────┘
```

### 4.2 Agent Responsibilities

#### Supervisor Agent
- Receives validation requests
- Analyzes input to determine required validators
- Routes tasks to domain validators
- Monitors overall progress
- Triggers aggregation

#### Domain Validators
- Schema Validator: Validates data structure and types
- Business Rules Validator: Applies domain-specific rules
- Data Quality Validator: Checks completeness and quality
- Cross-Reference Validator: Validates relationships

#### Atomic Validators
- Perform single, focused validation checks
- Return structured results with pass/fail and details
- Execute quickly with minimal dependencies
- Can run in parallel when independent

#### Aggregator Agent
- Collects results from all validators
- Calculates overall validation status
- Generates comprehensive report
- Assigns confidence scores

### 4.3 State Schema

```python
class ValidationState(TypedDict):
    input_data: dict
    validation_request: dict
    active_validators: list[str]
    completed_validators: list[str]
    validation_results: list[ValidationResult]
    errors: list[ErrorDetail]
    overall_status: str  # "pending", "in_progress", "completed", "failed"
    confidence_score: float
    final_report: Optional[dict]
```

### 4.4 LangGraph Workflow

```python
# Graph structure
graph = StateGraph(ValidationState)

# Add nodes
graph.add_node("supervisor", supervisor_agent)
graph.add_node("schema_validator", schema_validator_agent)
graph.add_node("business_rules", business_rules_agent)
graph.add_node("data_quality", data_quality_agent)
graph.add_node("aggregator", aggregator_agent)

# Conditional routing
graph.add_conditional_edges(
    "supervisor",
    route_to_validators,
    {
        "schema": "schema_validator",
        "business": "business_rules",
        "quality": "data_quality",
        "aggregate": "aggregator"
    }
)

# Set entry point
graph.set_entry_point("supervisor")
```

## 5. Technical Implementation

### 5.1 Technology Stack
- **Framework**: LangGraph (0.2+)
- **LLM**: OpenAI GPT-4 or Anthropic Claude
- **Language**: Python 3.11+
- **Package Manager**: UV
- **Testing**: pytest
- **Visualization**: LangGraph built-in visualization

### 5.2 Project Structure

```
hvas-mini/
├── src/
│   ├── agents/
│   │   ├── supervisor.py
│   │   ├── schema_validator.py
│   │   ├── business_rules.py
│   │   ├── data_quality.py
│   │   ├── aggregator.py
│   │   └── base.py
│   ├── graph/
│   │   ├── state.py
│   │   ├── workflow.py
│   │   └── routing.py
│   ├── validators/
│   │   ├── json_schema.py
│   │   ├── rule_engine.py
│   │   └── quality_checks.py
│   ├── models/
│   │   ├── validation_result.py
│   │   └── error_detail.py
│   └── utils/
│       ├── retry.py
│       └── config.py
├── tests/
│   ├── test_supervisor.py
│   ├── test_validators.py
│   ├── test_workflow.py
│   └── test_integration.py
├── docs/
│   └── task_plans/
├── examples/
│   └── sample_validation.py
├── pyproject.toml
├── README.md
└── .gitignore
```

### 5.3 Key Innovation Points

1. **Hierarchical State Management**: Multi-level state updates with parent-child relationships
2. **Dynamic Routing**: LLM-based routing decisions in supervisor
3. **Parallel Execution**: Independent validators run concurrently
4. **Confidence Scoring**: Aggregated confidence from multiple validators
5. **Retry with Context**: State-aware retry logic that learns from failures

## 6. Implementation Tasks

### Task 1: Core Infrastructure & State Management
- Define state schemas
- Implement base agent class
- Create state management utilities
- Setup project structure

### Task 2: Supervisor Agent
- Implement task analysis logic
- Create routing decision system
- Add monitoring capabilities
- Integrate with LangGraph conditional edges

### Task 3: Domain Validators
- Implement schema validator agent
- Implement business rules validator agent
- Implement data quality validator agent
- Create validator registry

### Task 4: Atomic Validators & Execution
- Build JSON schema checker
- Build rule engine
- Build quality check functions
- Implement parallel execution

### Task 5: Aggregator & Reporting
- Implement result aggregation logic
- Create confidence scoring algorithm
- Build comprehensive report generator
- Add visualization support

### Task 6: Error Handling & Resilience
- Implement retry logic with exponential backoff
- Add graceful degradation
- Create error context capture
- Build recovery mechanisms

### Task 7: Testing & Documentation
- Write unit tests for each agent
- Create integration tests
- Add workflow visualization
- Write comprehensive README

## 7. Success Criteria

- ✅ Supervisor correctly routes to appropriate validators
- ✅ Domain validators execute in parallel when possible
- ✅ Aggregator produces accurate confidence scores
- ✅ Error handling gracefully manages failures
- ✅ LangGraph visualization clearly shows workflow
- ✅ System handles 100+ validation rules efficiently
- ✅ Tests cover all critical paths

## 8. Future Enhancements

- Human-in-the-loop validation approval
- Machine learning for confidence scoring
- Custom validator plugins
- Distributed execution across workers
- Real-time validation streaming
- Advanced analytics and insights
