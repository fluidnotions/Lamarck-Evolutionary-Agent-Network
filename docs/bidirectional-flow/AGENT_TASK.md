# Agent Task: M7 - Bidirectional Flow

## Branch: `feature/bidirectional-flow`

## Priority: HIGH

## Execution: SEQUENTIAL (Depends on M6)

## Objective

Implement bidirectional information flow through the 3-layer hierarchy:
- **Downward flow**: Context distribution from parent to children
- **Upward flow**: Result aggregation from children to parent
- Confidence-based weighting for aggregations

## Dependencies

- ✅ M6 (Hierarchical Structure) - merged to master

## Implementation

### Files Created

#### 1. HierarchicalExecutor (`src/hvas_mini/hierarchy/executor.py`)

**Purpose**: Manages bidirectional flow through hierarchy

**Key Methods**:
- `execute_downward(state, layer)` - Distributes context and executes layer
- `execute_upward(state, layer)` - Aggregates child results to parent
- `execute_full_cycle(state)` - Complete down + up pass
- `_estimate_confidence(content)` - Heuristic confidence scoring
- `_aggregate_outputs(outputs)` - Confidence-weighted aggregation

**Downward Flow**:
```python
# Gets context from parent layer
if parent_role:
    context = state["layer_outputs"][layer - 1][parent_role]["content"]
else:
    context = state["topic"]  # Layer 1 gets topic

# Executes agent with parent context
output = await agent.generate_content(state, memories, weighted_context=context)

# Stores with confidence
state["layer_outputs"][layer][agent_role] = AgentOutput(
    content=output,
    confidence=self._estimate_confidence(output),
    metadata={"parent": parent_role}
)
```

**Upward Flow**:
```python
# Gathers child outputs
child_outputs = [
    state["layer_outputs"][layer + 1][child]
    for child in children
]

# Aggregates with confidence weighting
aggregated = self._aggregate_outputs(child_outputs)

# Updates parent metadata
state["layer_outputs"][layer][agent_role]["metadata"]["children"] = aggregated
```

#### 2. Tests (`test_bidirectional_flow.py`)

**Test Coverage**:
- Executor creation
- Downward flow (context distribution)
- Upward flow (result aggregation)
- Confidence estimation
- Weighted aggregation
- Full cycle execution

**Expected**: 12/12 tests passing

### Confidence Estimation

Simple heuristic based on:
- **Length score**: `min(len(content) / 500, 1.0)` (70% weight)
- **Structure score**: `min(paragraphs / 3, 1.0)` (30% weight)
- **Range**: [0.1, 1.0]

Can be enhanced with LLM-based scoring in future iterations.

### Result Aggregation

Confidence-weighted combination:
```python
weights = [confidence / total_confidence for each output]
weighted_confidence = sum(conf * weight for conf, weight in zip(confidences, weights))
```

Higher confidence outputs contribute more to aggregate.

## Integration Points

### With M6 (Hierarchical Structure)
- Uses `AgentHierarchy` for parent-child relationships
- Uses `HierarchicalState` and `AgentOutput` types
- Works with all 7 agents (coordinator + content + specialists)

### With M8 (Closed-Loop Refinement)
- `execute_full_cycle()` will be called repeatedly in refinement loop
- Confidence scores trigger revision decisions
- Pass history tracks improvements

### With M9 (Semantic Distance)
- Executor will use semantic distance for context filtering
- Currently distributes full context; M9 adds filtering

## Deliverables Checklist

- [x] `src/hvas_mini/hierarchy/executor.py` created
- [x] `HierarchicalExecutor` class with downward/upward methods
- [x] Confidence estimation implemented
- [x] Weighted aggregation implemented
- [x] Full cycle execution method
- [x] Package exports updated (`__init__.py`)
- [x] Comprehensive test suite created
- [ ] All tests passing
- [ ] Committed to branch
- [ ] Merged to main

## Testing

```bash
cd worktrees/bidirectional-flow
uv run pytest test_bidirectional_flow.py -v
```

## Next Steps

After M7 completion:
1. Fix any failing tests
2. Commit implementation
3. Merge to main
4. Begin M8 (Closed-Loop Refinement)

M8 will extend the executor with multi-pass execution and coordinator critique.

## Timeline

**Estimated**: 3-4 days (M7 spec)
**Actual**: Implementation complete, testing in progress

## Notes

- Bidirectional flow is foundation for closed-loop refinement
- Confidence scoring is heuristic; can be enhanced with LLM evaluation
- Aggregation preserves individual scores for debugging/analysis
- Full cycle executes entire hierarchy once (single pass)
