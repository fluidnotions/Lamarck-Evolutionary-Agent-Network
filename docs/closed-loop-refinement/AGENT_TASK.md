# Agent Task: M8 - Closed-Loop Refinement

## Branch: `feature/closed-loop-refinement`

## Priority: HIGH

## Execution: SEQUENTIAL (Depends on M6, M7)

## Objective

Implement multi-pass execution with coordinator critique and revision:
- Execute multiple passes through the hierarchy
- Coordinator critiques outputs after each pass
- Quality threshold checking for early exit
- Revision requests with detailed feedback
- Pass history tracking for improvement analysis

## Dependencies

- ✅ M6 (Hierarchical Structure) - merged
- ✅ M7 (Bidirectional Flow) - merged

## Implementation

### Extended HierarchicalExecutor

**File**: `src/hvas_mini/hierarchy/executor.py`

#### 1. Multi-Pass Execution (`execute_with_refinement`)

**Purpose**: Execute multiple refinement passes until quality threshold met or max passes reached

**Key Logic**:
```python
for pass_num in range(1, max_passes + 1):
    # Execute full cycle (downward + upward)
    # Record pass scores
    # Critique and decide if revision needed
    # Early exit if quality threshold met
    # Request revision for next pass if needed
```

**Features**:
- Maximum of 3 passes (configurable via `max_passes`)
- Quality threshold checking (default: 0.8 confidence)
- Early exit when quality met
- Pass history tracking
- Revision preparation between passes

#### 2. Quality Assessment (`critique_and_decide`)

**Purpose**: Determine if outputs meet quality threshold

**Criteria**:
- Average confidence across Layer 2 agents (intro, body, conclusion)
- Coordinator critique for each agent
- Quality threshold comparison (env: `QUALITY_THRESHOLD`)

**Returns**: `True` if revision needed, `False` if quality sufficient

**Exit Conditions**:
- Quality threshold met → No revision
- Max passes reached → No revision (final attempt)
- Otherwise → Revision needed

#### 3. Revision Request (`request_revision`)

**Purpose**: Generate specific revision feedback for agents

**Process**:
1. Review coordinator critiques
2. Skip agents with "Good quality"
3. Generate detailed revision prompts for low-quality outputs
4. Include original output and specific issues
5. Set `revision_requested` flag

**Revision Prompt Format**:
```
Revision needed for {role}:

Issue: {critique}

Original output:
{current_output}

Please revise to address the issues identified. Focus on improving:
- Length and completeness
- Structure and organization
- Quality and coherence
```

## Configuration

**Environment Variables**:
```env
QUALITY_THRESHOLD=0.8        # Minimum average confidence for quality gate
MAX_REFINEMENT_PASSES=3      # Maximum refinement iterations
```

## State Management

### Pass History Structure
```python
{
    "pass": 1,
    "scores": {
        "intro": 0.75,
        "body": 0.80,
        "conclusion": 0.70
    }
}
```

### Quality Tracking
- `quality_threshold_met`: Boolean flag for early exit
- `revision_requested`: Boolean flag for revision state
- `current_pass`: Current pass number (1-indexed)
- `coordinator_critique`: Dict of feedback per agent

## Test Coverage

**File**: `test_closed_loop_refinement.py`

**Test Classes**:
1. `TestMultiPassExecution` - Basic multi-pass flow
2. `TestQualityThreshold` - Threshold checking logic
3. `TestCoordinatorCritique` - Critique generation
4. `TestRevisionRequest` - Revision feedback
5. `TestEarlyExit` - Quality-based early exit
6. `TestEndToEnd` - Full refinement cycle

**Expected**: 10/10 tests passing

## Integration Points

### With M6 (Hierarchical Structure)
- Uses `AgentHierarchy` for layer organization
- Uses `HierarchicalState` with pass tracking fields
- Uses `CoordinatorAgent.critique_outputs()` method

### With M7 (Bidirectional Flow)
- Extends `HierarchicalExecutor` class
- Uses `execute_downward()` and `execute_upward()` in each pass
- Uses confidence scores from `AgentOutput`

### With M9 (Semantic Distance)
- Refinement passes will use semantic filtering (M9)
- Context becomes more targeted with each revision

### With M10 (Visualization v2)
- Pass history visualized in Rich UI
- Score improvements shown per pass
- Critique feedback displayed in terminal

## Deliverables Checklist

- [x] `execute_with_refinement()` method implemented
- [x] `critique_and_decide()` method implemented
- [x] `request_revision()` method implemented
- [x] Pass history tracking
- [x] Quality threshold checking
- [x] Early exit logic
- [x] Comprehensive test suite (10 tests)
- [x] Documentation complete
- [ ] Tests passing
- [ ] Committed to branch
- [ ] Merged to main

## Testing

```bash
cd worktrees/closed-loop-refinement
uv run pytest test_closed_loop_refinement.py -v
```

## Example Usage

```python
# Create executor with agents
agents, hierarchy = create_hierarchical_agents()
executor = HierarchicalExecutor(agents)

# Create state with max passes
state = create_hierarchical_state("AI Safety")
state["max_passes"] = 3

# Execute with refinement
state = await executor.execute_with_refinement(state)

# Check results
print(f"Passes executed: {len(state['pass_history'])}")
print(f"Quality threshold met: {state['quality_threshold_met']}")
print(f"Final scores: {state['pass_history'][-1]['scores']}")
```

## Expected Behavior

### Scenario 1: High Quality on First Pass
- Pass 1: avg_confidence = 0.85 (> 0.8 threshold)
- Result: Early exit, `quality_threshold_met = True`
- Passes: 1

### Scenario 2: Improvement Across Passes
- Pass 1: avg_confidence = 0.60 → Revision requested
- Pass 2: avg_confidence = 0.75 → Revision requested
- Pass 3: avg_confidence = 0.82 → Quality met
- Result: Exit on pass 3, `quality_threshold_met = True`
- Passes: 3

### Scenario 3: Max Passes Reached
- Pass 1: avg_confidence = 0.55 → Revision requested
- Pass 2: avg_confidence = 0.62 → Revision requested
- Pass 3: avg_confidence = 0.70 (< 0.8, but max reached)
- Result: Exit after max, `quality_threshold_met = False`
- Passes: 3

## Next Steps

After M8 completion:
1. Commit implementation
2. Merge to main
3. Begin M9 (Semantic Distance) - Can run in parallel with M10
4. M10 will visualize the multi-pass improvements

## Timeline

**Estimated**: 2-3 days (M8 spec)
**Actual**: Implementation complete, testing in progress

## Notes

- Closed-loop refinement is core to demonstrating hierarchical learning
- Each pass provides opportunity for improvement based on feedback
- Quality threshold prevents unnecessary iterations
- Pass history enables visualization of learning trajectory
- Coordinator critique is currently heuristic-based (can enhance with LLM)
