# Agent Task: M9 - Semantic Distance Weighting

## Branch: `feature/semantic-distance`

## Priority: MEDIUM

## Execution: PARALLEL (Can run with M10, depends only on M6)

## Objective

Implement vector-based semantic distance for intelligent context filtering:
- Cosine similarity between agent semantic vectors
- Distance-based context filtering (closer agents get more context)
- Integration with coordinator for smarter distribution
- Weight computation for context relevance

## Dependencies

- ✅ M6 (Hierarchical Structure) - for semantic vectors in AgentNode

## Implementation

### 1. Semantic Distance Calculator

**File**: `src/hvas_mini/hierarchy/semantic.py`

**Key Functions**:

#### `compute_semantic_distance(hierarchy, agent_a, agent_b)`
- Calculates cosine distance between agent semantic vectors
- Returns distance in [0, 1] where:
  - 0 = identical focus (share all context)
  - 1 = opposite focus (share minimal context)
- Uses numpy for efficient vector operations

**Algorithm**:
```python
similarity = dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
distance = (1.0 - similarity) / 2.0
```

#### `filter_context_by_distance(context, distance, min_ratio=0.3)`
- Filters context string based on semantic distance
- Strength inversely proportional to distance
- Preserves beginning of context (most important)
- Minimum 30% context always retained

**Filtering Logic**:
```python
strength = 1.0 - (distance * (1.0 - min_ratio))
keep_count = max(1, int(num_sentences * strength))
```

#### `compute_context_weights(hierarchy, parent_role, child_roles)`
- Computes relevance weights for all children
- Weight = 1 - distance (inverse relationship)
- Returns dict of {child_role: weight}

#### `get_contextual_relevance(hierarchy, source, target)`
- Single relevance score for source → target
- Higher score = more relevant context
- Inverse of semantic distance

### 2. Coordinator Integration

**Modified**: `src/hvas_mini/hierarchy/coordinator.py`

**Method**: `distribute_context(state, use_semantic_filtering=True)`

**Changes**:
- Added `use_semantic_filtering` parameter (default: True)
- Calculates semantic distance for each child
- Filters context based on distance
- Backward compatible (can disable filtering)

**Behavior**:
```python
if use_semantic_filtering:
    distance = compute_semantic_distance(self.hierarchy, self.role, child_role)
    filtered_context = filter_context_by_distance(intent, distance)
else:
    filtered_context = intent  # Full context (M6/M7/M8 behavior)
```

### 3. Package Exports

**Updated**: `src/hvas_mini/hierarchy/__init__.py`

**New Exports**:
- `compute_semantic_distance`
- `filter_context_by_distance`
- `compute_context_weights`
- `get_contextual_relevance`

## Semantic Vectors

**Defined in M6** (`AgentNode.semantic_vector`):

```python
"coordinator":  [0.0, 1.0, 0.0]  # Integration-focused
"intro":        [0.8, 0.5, 0.2]  # Hook, engagement
"body":         [0.5, 0.8, 0.9]  # Content-heavy
"conclusion":   [0.7, 0.6, 0.3]  # Synthesis
"researcher":   [0.3, 0.9, 1.0]  # Factual, deep
"fact_checker": [0.2, 0.8, 0.9]  # Accuracy-focused
"stylist":      [0.9, 0.4, 0.2]  # Style, tone
```

**Vector Dimensions** (interpretive):
1. Style/Creativity
2. Integration/Synthesis
3. Facts/Content

## Test Coverage

**File**: `test_semantic_distance.py`

**Test Classes**:
1. `TestSemanticDistance` - Distance calculation
2. `TestContextFiltering` - Context filtering logic
3. `TestContextWeights` - Weight computation
4. `TestContextualRelevance` - Relevance scoring
5. `TestSimilarityMatrix` - Pairwise matrix
6. `TestCoordinatorIntegration` - Integration tests
7. `TestSemanticProperties` - Vector relationships

**Expected**: 20+ tests passing

## Integration Points

### With M6 (Hierarchical Structure)
- Uses `AgentNode.semantic_vector` from M6
- Uses `AgentHierarchy` for agent lookups
- Semantic vectors defined in `structure.py`

### With M7 (Bidirectional Flow)
- Executor can use filtered contexts
- More targeted context distribution
- Reduces noise in child inputs

### With M8 (Closed-Loop Refinement)
- Refinement passes use semantic filtering
- More focused revision instructions
- Better quality improvements per pass

### With M10 (Visualization)
- Can visualize semantic distances
- Show context filtering in action
- Display relevance scores

## Example Usage

```python
from hvas_mini.hierarchy import (
    AgentHierarchy,
    compute_semantic_distance,
    filter_context_by_distance,
)

# Create hierarchy
hierarchy = AgentHierarchy()

# Compute distance
distance = compute_semantic_distance(hierarchy, "coordinator", "researcher")
print(f"Distance: {distance:.2f}")  # e.g., 0.45

# Filter context
full_context = "This is a long context. With many sentences. About the topic."
filtered = filter_context_by_distance(full_context, distance, min_ratio=0.3)
print(f"Filtered: {filtered}")  # Keeps relevant portion

# Use in coordinator
coordinator = agents["coordinator"]
state["coordinator_intent"] = "Detailed intent..."
contexts = coordinator.distribute_context(state, use_semantic_filtering=True)
# Each child gets appropriately filtered context
```

## Deliverables Checklist

- [x] `src/hvas_mini/hierarchy/semantic.py` created
- [x] `compute_semantic_distance()` implemented
- [x] `filter_context_by_distance()` implemented
- [x] `compute_context_weights()` implemented
- [x] `get_contextual_relevance()` implemented
- [x] `compute_similarity_matrix()` utility
- [x] Coordinator integration complete
- [x] Package exports updated
- [x] Comprehensive test suite (20+ tests)
- [x] Documentation complete
- [ ] Tests passing
- [ ] Committed to branch
- [ ] Merged to main

## Testing

```bash
cd worktrees/semantic-distance
uv run pytest test_semantic_distance.py -v
```

## Configuration

**No new environment variables required.**

Uses semantic vectors defined in M6's `AgentHierarchy`.

## Benefits

1. **Targeted Context**: Agents receive context relevant to their focus
2. **Reduced Noise**: Less irrelevant information in agent inputs
3. **Efficient Processing**: Shorter contexts = faster generation
4. **Better Specialization**: Agents can focus on their domain
5. **Scalability**: Easy to add new agents with appropriate vectors

## Next Steps

After M9 completion:
1. Commit implementation
2. Merge to main
3. M10 (Visualization v2) can visualize semantic relationships
4. Iteration 3 complete!

## Timeline

**Estimated**: 2-3 days (M9 spec)
**Actual**: Implementation complete

## Notes

- Semantic vectors are hand-crafted based on agent roles
- Could be learned/optimized in future iterations
- Cosine distance is efficient and interpretable
- Context filtering preserves sentence boundaries
- Minimum 30% context ensures basic information flow
- Backward compatible (can disable filtering)
