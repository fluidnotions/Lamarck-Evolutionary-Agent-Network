# Task 3: Domain Validators
## Implementation Plan

## Objective
Implement mid-tier domain validator agents that handle specific validation domains (Schema, Business Rules, Data Quality, Cross-Reference). Each validator coordinates atomic validation checks within its domain.

## Dependencies
- Task 1: Core Infrastructure (requires BaseAgent, ValidationState)
- Task 2: Supervisor Agent (for integration testing)

## Components to Implement

### 3.1 Schema Validator Agent
**File**: `src/agents/schema_validator.py`
**Actions**:
- Extend `BaseAgent` class
- Implement schema validation orchestration
- Detect schema type (JSON Schema, Pydantic, custom)
- Coordinate with atomic JSON schema validators
- Aggregate schema validation results
- Generate detailed error messages with paths

### 3.2 Business Rules Validator Agent
**File**: `src/agents/business_rules.py`
**Actions**:
- Extend `BaseAgent` class
- Load and parse business rule definitions
- Implement rule evaluation orchestration
- Support different rule types (constraint, derivation, inference)
- Coordinate with atomic rule validators
- Provide rule violation explanations using LLM

**Innovation**: Use LLM to generate human-readable explanations for rule violations

### 3.3 Data Quality Validator Agent
**File**: `src/agents/data_quality.py`
**Actions**:
- Extend `BaseAgent` class
- Implement quality check orchestration
- Check completeness (missing values, null handling)
- Check consistency (cross-field validation)
- Check accuracy (format validation, range checks)
- Calculate quality scores per dimension
- Generate quality improvement suggestions

### 3.4 Cross-Reference Validator Agent
**File**: `src/agents/cross_reference.py`
**Actions**:
- Extend `BaseAgent` class
- Implement relationship validation
- Support referential integrity checks
- Handle foreign key validation
- Validate cardinality constraints
- Check cyclic dependencies
- Generate relationship diagrams for violations

### 3.5 Validator Base Class Enhancements
**File**: `src/agents/base.py` (enhancement)
**Actions**:
- Add domain validator specific methods
- Implement result aggregation utilities
- Add atomic validator coordination logic
- Support parallel execution of atomic validators within domain

## Testing Strategy

### Unit Tests
**File**: `tests/test_schema_validator.py`
- Test with valid schemas (JSON Schema, Pydantic)
- Test with invalid data
- Test error message generation
- Test schema detection logic

**File**: `tests/test_business_rules.py`
- Test rule loading and parsing
- Test rule evaluation
- Test LLM explanation generation
- Test with complex rule sets

**File**: `tests/test_data_quality.py`
- Test completeness checks
- Test consistency checks
- Test accuracy checks
- Test quality score calculation

**File**: `tests/test_cross_reference.py`
- Test referential integrity
- Test foreign key validation
- Test cardinality constraints
- Test cyclic dependency detection

### Integration Tests
**File**: `tests/test_domain_validators_integration.py`
- Test validators in LangGraph workflow
- Test parallel execution of multiple validators
- Test state updates from validators
- Test error propagation

## Technical Specifications

### Schema Validator
```python
class SchemaValidatorAgent(BaseAgent):
    """Validates data against schemas (JSON Schema, Pydantic, etc.)."""

    def __init__(self, llm):
        super().__init__(
            name="schema_validator",
            description="Validates data structure and types against schemas"
        )
        self.llm = llm

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Validate input data against detected or provided schema.

        Steps:
        1. Detect or load schema definition
        2. Validate data structure
        3. Validate data types
        4. Validate constraints (required fields, patterns, etc.)
        5. Generate detailed error paths
        6. Update state with results
        """
        input_data = state["input_data"]
        schema = self._detect_schema(state)

        # Coordinate atomic validators
        results = []
        results.append(self._validate_structure(input_data, schema))
        results.append(self._validate_types(input_data, schema))
        results.append(self._validate_constraints(input_data, schema))

        # Aggregate results
        validation_result = self._aggregate_results(results)

        # Update state
        new_state = state.copy()
        new_state["validation_results"].append(validation_result)
        new_state["completed_validators"].append(self.name)

        return new_state
```

### Business Rules Validator
```python
class BusinessRulesAgent(BaseAgent):
    """Validates business rules and constraints."""

    def __init__(self, llm, rule_engine):
        super().__init__(
            name="business_rules",
            description="Validates domain-specific business rules"
        )
        self.llm = llm
        self.rule_engine = rule_engine

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Evaluate business rules against input data.

        Steps:
        1. Load applicable rules from rule engine
        2. Evaluate each rule
        3. For violations, generate LLM explanations
        4. Calculate overall compliance score
        5. Update state with results
        """
        pass
```

### Data Quality Validator
```python
class DataQualityAgent(BaseAgent):
    """Validates data quality across multiple dimensions."""

    def __init__(self, llm):
        super().__init__(
            name="data_quality",
            description="Validates data completeness, consistency, and accuracy"
        )
        self.llm = llm

    def execute(self, state: ValidationState) -> ValidationState:
        """
        Assess data quality across dimensions.

        Quality Dimensions:
        - Completeness: Missing values, required fields
        - Consistency: Cross-field validation, logical consistency
        - Accuracy: Format, range, domain validity
        - Timeliness: Freshness checks (if timestamps available)

        Returns quality scores per dimension and overall.
        """
        pass
```

## Innovation Highlights

1. **LLM-Enhanced Error Messages**: Business rules validator uses LLM to generate human-readable violation explanations
2. **Multi-Dimensional Quality Scoring**: Data quality validator provides granular scores across quality dimensions
3. **Schema Auto-Detection**: Schema validator can detect schema type automatically
4. **Relationship Visualization**: Cross-reference validator generates visual diagrams for complex violations
5. **Atomic Validator Coordination**: Each domain validator efficiently coordinates multiple atomic checks

## Acceptance Criteria

- ✅ All four domain validators implemented and functional
- ✅ Schema validator handles JSON Schema and Pydantic models
- ✅ Business rules validator generates clear violation explanations
- ✅ Data quality validator provides multi-dimensional scoring
- ✅ Cross-reference validator detects referential integrity issues
- ✅ All validators properly update ValidationState
- ✅ All unit tests passing (>85% coverage per validator)
- ✅ Integration tests show proper workflow integration
- ✅ Validators can execute in parallel without conflicts

## Implementation Order

1. Enhance BaseAgent with domain validator utilities
2. Implement SchemaValidatorAgent (most fundamental)
3. Implement DataQualityAgent (builds on schema)
4. Implement BusinessRulesAgent (most complex)
5. Implement CrossReferenceAgent (most specialized)
6. Write comprehensive tests for each validator
7. Integration tests with supervisor and routing

## Estimated Complexity
**High** - Multiple complex validators with different logic patterns and LLM integration

## Notes
- Each validator should be independently testable
- Validators must handle partial data gracefully
- Consider performance implications of parallel atomic validators
- LLM prompts for business rules explanations need careful engineering
- Schema detection heuristics should be robust
