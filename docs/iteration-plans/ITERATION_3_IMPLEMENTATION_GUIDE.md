# Iteration 3: Implementation Guide

**Quick Reference** for implementing M6-M10 hierarchical architecture.

See `ITERATION_3_PLAN.md` for detailed specifications and `HIERARCHY_GAP_ANALYSIS.md` for conceptual foundation.

---

## Branches & Status

| Milestone | Branch | Worktree | Status | Tests |
|-----------|--------|----------|--------|-------|
| M6 | feature/hierarchical-structure | ✅ Ready | 📋 AGENT_TASK.md created | 8 |
| M7 | feature/bidirectional-flow | ✅ Ready | 📋 See below | 12 |
| M8 | feature/closed-loop-refinement | ✅ Ready | 📋 See below | 10 |
| M9 | feature/semantic-distance | ✅ Ready | 📋 See below | 8 |
| M10 | feature/visualization-v2 | ✅ Ready | 📋 See below | 5 |

---

## M6: Hierarchical Structure (DETAILED)

**Location**: `worktrees/hierarchical-structure/`
**Docs**: `docs/hierarchical-structure/AGENT_TASK.md` (COMPLETE)

**Files to Create**:
1. `src/hvas_mini/hierarchy/structure.py` - AgentHierarchy class
2. `src/hvas_mini/hierarchy/coordinator.py` - CoordinatorAgent
3. `src/hvas_mini/hierarchy/specialists.py` - 3 specialists
4. `src/hvas_mini/hierarchy/factory.py` - create_hierarchical_agents()
5. `src/hvas_mini/state.py` - Add HierarchicalState
6. `test_hierarchical_structure.py` - 8 tests

**Start Here**: Read `docs/hierarchical-structure/AGENT_TASK.md` for complete implementation details.

---

## M7: Bidirectional Flow

**Location**: `worktrees/bidirectional-flow/`
**Dependencies**: M6 complete
**Duration**: 3-4 days

### Objective
Implement downward context distribution and upward result aggregation.

### Key Components

#### 1. HierarchicalExecutor
**File**: `src/hvas_mini/hierarchy/executor.py`

```python
class HierarchicalExecutor:
    """Executes workflow with bidirectional flow."""

    async def execute_downward(self, state, layer: int):
        """Distribute context and execute layer."""
        for agent_role in state["hierarchy"].get_layer_agents(layer):
            parent_role = state["hierarchy"].get_parent(agent_role)

            if parent_role:
                # Get context from parent's output
                context = state["layer_outputs"][layer - 1][parent_role]["content"]
            else:
                context = state["topic"]

            # Execute with context
            agent = self.agents[agent_role]
            output = await agent(state, context)

            # Store with confidence
            state["layer_outputs"][layer][agent_role] = AgentOutput(
                content=output,
                confidence=self._estimate_confidence(output),
                metadata={"parent": parent_role}
            )

    async def execute_upward(self, state, layer: int):
        """Aggregate child results to parent."""
        for agent_role in state["hierarchy"].get_layer_agents(layer):
            children = state["hierarchy"].get_children(agent_role)

            if not children:
                continue

            # Gather child outputs
            child_outputs = [
                state["layer_outputs"][layer + 1][child]
                for child in children
            ]

            # Aggregate
            aggregated = self._aggregate_outputs(child_outputs)

            # Update parent's metadata
            state["layer_outputs"][layer][agent_role]["metadata"]["children"] = aggregated
```

#### 2. Confidence Estimation
**File**: Same as above

```python
def _estimate_confidence(self, content: str) -> float:
    """Estimate output confidence.

    Simple heuristic based on length and structure.
    Can be enhanced with LLM-based quality scoring.
    """
    # Length score (500 chars = 1.0)
    length_score = min(len(content) / 500, 1.0)

    # Structure score (has paragraphs?)
    paragraphs = content.count("\n\n")
    structure_score = min(paragraphs / 3, 1.0)

    # Combined
    confidence = (length_score * 0.7) + (structure_score * 0.3)

    return max(0.1, min(1.0, confidence))
```

#### 3. Result Aggregation
```python
def _aggregate_outputs(self, outputs: List[AgentOutput]) -> Dict:
    """Combine multiple child outputs with confidence weighting."""

    total_confidence = sum(o["confidence"] for o in outputs)

    if total_confidence == 0:
        weights = [1.0 / len(outputs)] * len(outputs)
    else:
        weights = [o["confidence"] / total_confidence for o in outputs]

    return {
        "combined_content": "\n\n".join([o["content"] for o in outputs]),
        "weighted_confidence": sum(
            o["confidence"] * w for o, w in zip(outputs, weights)
        ),
        "source_count": len(outputs)
    }
```

### Tests
**File**: `test_bidirectional_flow.py`

```python
def test_downward_execution():
    """Context should flow from parent to child."""

def test_upward_aggregation():
    """Children results should aggregate to parent."""

def test_confidence_calculation():
    """Confidence should be in [0, 1]."""

def test_weighted_aggregation():
    """Higher confidence outputs should have more weight."""
```

**Expected**: 12/12 tests passing

---

## M8: Closed-Loop Refinement

**Location**: `worktrees/closed-loop-refinement/`
**Dependencies**: M6, M7 complete
**Duration**: 2-3 days

### Objective
Multi-pass execution with coordinator critique and revision requests.

### Key Components

#### 1. Multi-Pass Execution Loop
**File**: `src/hvas_mini/hierarchy/executor.py` (extend from M7)

```python
async def execute_with_refinement(self, state) -> HierarchicalState:
    """Execute with closed-loop refinement."""

    for pass_num in range(1, state["max_passes"] + 1):
        state["current_pass"] = pass_num

        # Execute all layers (down then up)
        for layer in [1, 2, 3]:
            await self.execute_downward(state, layer)

        for layer in [3, 2, 1]:
            await self.execute_upward(state, layer)

        # Record pass
        state["pass_history"].append({
            "pass": pass_num,
            "scores": {
                role: state["layer_outputs"][2][role]["confidence"]
                for role in ["intro", "body", "conclusion"]
            }
        })

        # Critique and decide
        needs_revision = await self.critique_and_decide(state)

        if not needs_revision:
            state["quality_threshold_met"] = True
            break

        # Prepare revision
        if pass_num < state["max_passes"]:
            await self.request_revision(state)

    return state
```

#### 2. Critique Logic
```python
async def critique_and_decide(self, state) -> bool:
    """Critique outputs and decide if revision needed."""

    # Get coordinator critique
    critiques = self.agents["coordinator"].critique_outputs(state)
    state["coordinator_critique"] = critiques

    # Calculate average quality
    avg_confidence = sum(
        state["layer_outputs"][2][role]["confidence"]
        for role in ["intro", "body", "conclusion"]
    ) / 3

    threshold = float(os.getenv("QUALITY_THRESHOLD", "0.8"))

    # Check if quality met
    if avg_confidence >= threshold:
        return False  # No revision needed

    # Check if passes remaining
    if state["current_pass"] >= state["max_passes"]:
        return False  # No more tries

    return True  # Revision needed
```

#### 3. Revision Request
```python
async def request_revision(self, state):
    """Generate specific revision feedback."""

    for role in ["intro", "body", "conclusion"]:
        critique = state["coordinator_critique"][role]

        if "Good quality" in critique:
            continue  # No revision needed

        # Generate revision instruction
        revision_prompt = f"""Revision needed for {role}:

Issue: {critique}

Original output:
{state["layer_outputs"][2][role]["content"]}

Please revise to address the issues identified."""

        state["coordinator_critique"][role] = revision_prompt

    state["revision_requested"] = True
```

### Configuration
Add to `.env.example`:
```env
# Closed-Loop Refinement (M8)
QUALITY_THRESHOLD=0.8
MAX_REFINEMENT_PASSES=3
```

### Tests
**File**: `test_closed_loop_refinement.py`

Expected: 10/10 tests passing

---

## M9: Semantic Distance Weighting

**Location**: `worktrees/semantic-distance/`
**Dependencies**: M6 complete
**Duration**: 2-3 days

### Objective
Use vector-based semantic distance to weight context distribution.

### Key Components

#### 1. Semantic Distance Calculation
**File**: `src/hvas_mini/hierarchy/semantic.py`

```python
import numpy as np
from hvas_mini.hierarchy.structure import AgentHierarchy

def compute_semantic_distance(
    hierarchy: AgentHierarchy, agent_a: str, agent_b: str
) -> float:
    """Compute cosine distance between agent vectors.

    Args:
        hierarchy: AgentHierarchy instance
        agent_a: First agent role
        agent_b: Second agent role

    Returns:
        Distance in [0, 1] where 0=identical, 1=opposite
    """
    vec_a = np.array(hierarchy.nodes[agent_a].semantic_vector)
    vec_b = np.array(hierarchy.nodes[agent_b].semantic_vector)

    # Cosine similarity
    similarity = np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    )

    # Convert to distance [0, 1]
    distance = (1.0 - similarity) / 2.0

    return max(0.0, min(1.0, distance))
```

#### 2. Distance-Based Context Filtering
```python
def filter_context_by_distance(
    context: str,
    distance: float,
    min_ratio: float = 0.3
) -> str:
    """Filter context based on semantic distance.

    Args:
        context: Full context string
        distance: Semantic distance [0, 1]
        min_ratio: Minimum context to retain

    Returns:
        Filtered context
    """
    # Strength inversely related to distance
    # distance=0 → strength=1.0 (full context)
    # distance=1 → strength=min_ratio (minimal context)
    strength = 1.0 - (distance * (1.0 - min_ratio))

    # Filter by keeping top strength% of sentences
    sentences = context.split(". ")
    keep_count = max(1, int(len(sentences) * strength))

    return ". ".join(sentences[:keep_count]) + "."
```

#### 3. Integration with Coordinator
**Modify**: `src/hvas_mini/hierarchy/coordinator.py`

```python
def distribute_context(self, state) -> Dict[str, str]:
    """Create filtered context for each child based on distance."""
    contexts = {}
    intent = state["coordinator_intent"]

    for child_role in self.hierarchy.get_children(self.role):
        # Calculate distance
        distance = compute_semantic_distance(
            self.hierarchy, self.role, child_role
        )

        # Filter context by distance
        filtered = filter_context_by_distance(intent, distance)

        contexts[child_role] = filtered

    return contexts
```

### Tests
**File**: `test_semantic_distance.py`

Expected: 8/8 tests passing

---

## M10: Visualization v2

**Location**: `worktrees/visualization-v2/`
**Dependencies**: M6-M9 complete
**Duration**: 1-2 days

### Objective
Visualize hierarchical structure, flows, and refinement cycles.

### Key Components

#### 1. Hierarchy Tree Visualization
**File**: `src/hvas_mini/visualization/hierarchy_viz.py`

```python
from rich.tree import Tree
from rich.console import Console

def render_hierarchy_tree(hierarchy: AgentHierarchy, state: HierarchicalState):
    """Render hierarchy as tree with current state."""

    tree = Tree("[bold]Agent Hierarchy[/bold]")

    # Layer 1
    coord_node = tree.add(
        f"[cyan]Coordinator[/cyan] (confidence: {state['layer_outputs'][1].get('coordinator', {}).get('confidence', 0):.2f})"
    )

    # Layer 2
    for role in ["intro", "body", "conclusion"]:
        output = state["layer_outputs"][2].get(role, {})
        confidence = output.get("confidence", 0)

        content_node = coord_node.add(
            f"[green]{role.title()}[/green] (conf: {confidence:.2f})"
        )

        # Layer 3
        for child in hierarchy.get_children(role):
            child_output = state["layer_outputs"][3].get(child, {})
            child_conf = child_output.get("confidence", 0)

            content_node.add(
                f"[yellow]{child}[/yellow] (conf: {child_conf:.2f})"
            )

    return tree
```

#### 2. Pass Comparison View
```python
def render_pass_comparison(state: HierarchicalState):
    """Show improvement across passes."""

    from rich.table import Table

    table = Table(title="Refinement Progress")

    table.add_column("Pass", style="cyan")
    table.add_column("Intro", style="green")
    table.add_column("Body", style="green")
    table.add_column("Conclusion", style="green")
    table.add_column("Average", style="bold")

    for pass_data in state["pass_history"]:
        pass_num = pass_data["pass"]
        scores = pass_data["scores"]

        avg = sum(scores.values()) / len(scores)

        table.add_row(
            f"Pass {pass_num}",
            f"{scores.get('intro', 0):.2f}",
            f"{scores.get('body', 0):.2f}",
            f"{scores.get('conclusion', 0):.2f}",
            f"{avg:.2f}"
        )

    return table
```

#### 3. Flow Animation
```python
def animate_flow(state: HierarchicalState):
    """Animate downward/upward flow."""

    from rich.live import Live
    from time import sleep

    # Show downward flow (blue arrows)
    # Show upward flow (green arrows)
    # Show aggregation points
    # Implementation details in file
```

### Tests
**File**: `test_visualization.py`

Expected: 5/5 tests passing

---

## Implementation Order

**Strict Dependencies**:
```
M6 (foundation) ──┬──→ M7 (bidirectional) ──→ M8 (closed-loop)
                  │
                  └──→ M9 (semantic) ──────→ M10 (visualization)
```

**Recommended Sequence**:
1. **M6** - Complete hierarchical structure (CRITICAL)
2. **M7** + **M9** - Can be done in parallel after M6
3. **M8** - Requires M7 complete
4. **M10** - Requires M6-M9 complete

---

## Testing Strategy

### Per Milestone
Run tests in worktree:
```bash
cd worktrees/<milestone-name>
uv run pytest test_*.py -v
```

### Integration Testing
After each milestone merge:
```bash
# On master branch
uv run pytest test_agent_weighting.py test_memory_decay.py \
  test_meta_agent.py test_hierarchical_structure.py \
  test_bidirectional_flow.py -v
```

### Full Suite
```bash
uv run pytest -v
```

**Expected Total**: 49 (M1-M4) + 43 (M6-M10) = 92 tests

---

## Commit Strategy

Each milestone:
1. Implement in worktree
2. Create comprehensive tests
3. Commit to feature branch
4. Move AGENT_TASK.md to docs/<milestone>/
5. Push or prepare PR

Example:
```bash
cd worktrees/hierarchical-structure
git add -A
git commit -m "Implement M6: Hierarchical structure foundation

- AgentHierarchy with 3 layers
- CoordinatorAgent (Layer 1)
- 3 specialist agents (Layer 3)
- HierarchicalState extension
- 8/8 tests passing"
```

---

## Configuration Summary

New `.env` variables for Iteration 3:

```env
# M8: Closed-Loop Refinement
QUALITY_THRESHOLD=0.8
MAX_REFINEMENT_PASSES=3

# M9: Semantic Distance
MIN_CONTEXT_RATIO=0.3  # Minimum context to retain at max distance
```

---

## Success Metrics

| Milestone | Metric | Target |
|-----------|--------|--------|
| M6 | Layers defined | 3 layers |
| M7 | Context filtering | Child < 70% parent |
| M8 | Quality improvement | +15% per pass |
| M9 | Distance correlation | r > 0.7 |
| M10 | Visualization | All panels working |

---

## Quick Start

1. **Read detailed plan**: `ITERATION_3_PLAN.md`
2. **Start with M6**: `worktrees/hierarchical-structure/`
3. **Read M6 task**: `docs/hierarchical-structure/AGENT_TASK.md`
4. **Implement incrementally**: One file at a time
5. **Test continuously**: Run tests after each component
6. **Commit frequently**: Small atomic commits

---

## Resources

- **Conceptual**: `HIERARCHY_GAP_ANALYSIS.md`
- **Detailed Plan**: `ITERATION_3_PLAN.md`
- **M6 Full Spec**: `worktrees/hierarchical-structure/docs/hierarchical-structure/AGENT_TASK.md`
- **Current Status**: `IMPLEMENTATION_STATUS.md`

---

## Timeline

**Total**: 11-16 days

**Breakdown**:
- M6: 3-4 days (foundation)
- M7: 3-4 days (bidirectional flow)
- M8: 2-3 days (closed-loop)
- M9: 2-3 days (semantic distance)
- M10: 1-2 days (visualization)

**Parallel Option** (if multiple developers):
- M7 and M9 can run in parallel after M6
- Reduces timeline to 9-12 days

---

## Help & Troubleshooting

**Issue**: Tests failing after M6
- **Solution**: Ensure all 7 agents created by factory
- **Check**: `create_hierarchical_agents()` returns correct count

**Issue**: Context not filtering
- **Solution**: Verify parent-child relationships in hierarchy
- **Check**: `get_parent()` and `get_children()` correctness

**Issue**: Confidence always 0
- **Solution**: Check `_estimate_confidence()` logic
- **Check**: Content length > 0

**Issue**: Revision loop not triggering
- **Solution**: Lower QUALITY_THRESHOLD temporarily
- **Check**: Critique logic identifying issues

---

Ready to implement! Start with M6 and work through the milestones systematically.
