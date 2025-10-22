# Visualization Update Requirements

## Problem

The current `visualization.py` (line 74) only shows the V2 linear architecture with 3 agents:

```python
for role in ["intro", "body", "conclusion"]:
```

This doesn't represent the current **hierarchical ensemble architecture** with:
- **Layer 1**: Coordinator (research, orchestration, critique)
- **Layer 2**: Content Agents in AgentPools (intro/body/conclusion populations)
- **Layer 3**: Specialist Agents (researcher, fact-checker, stylist)

## Current Visualization Structure

### `create_status_table()` (lines 58-95)
- Only loops through intro/body/conclusion
- Missing coordinator status
- Missing specialist agent status
- Missing pool information (population size, active agent)
- Missing revision loop status

### `create_memory_panel()` (lines 97-119)
- Only shows memories for intro/body/conclusion
- Missing coordinator memory retrieval
- Missing specialist memory usage

### `create_evolution_panel()` (lines 121-166)
- Only shows temperature evolution for content agents
- Missing pool evolution events
- Missing parent lineage information
- Missing selection strategy info

## Required Changes

### 1. Update Status Table to Show Hierarchy

**New structure needed:**

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🤖 Agent Execution Status (Hierarchical)                     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Layer │ Agent         │ Status      │ Pool │ Memories │ Score │
├───────┼───────────────┼─────────────┼──────┼──────────┼───────┤
│ L1    │ Coordinator   │ ✓ Research  │ -    │ 5        │ -     │
│ L2    │ Intro Pool    │ ⟳ Gen       │ 3/5  │ 4        │ 8.5   │
│ L2    │ Body Pool     │ ⟳ Gen       │ 3/5  │ 6        │ 7.2   │
│ L2    │ Conclusion    │ ⏸ Wait      │ 2/5  │ 0        │ -     │
│ L3    │ Researcher    │ ○ Ready     │ -    │ 0        │ -     │
│ L3    │ FactChecker   │ ○ Ready     │ -    │ 0        │ -     │
│ L3    │ Stylist       │ ○ Ready     │ -    │ 0        │ -     │
└───────┴───────────────┴─────────────┴──────┴──────────┴───────┘
```

### 2. Add Pool Evolution Panel

**New panel showing:**
- Current generation number
- Pool statistics (size, avg fitness, diversity)
- Evolution triggers (every N generations)
- Active agent selection (which agent from pool is being used)
- Recent reproduction events

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🧬 Agent Pool Evolution                                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Generation: 5/20 (Evolution at: 5, 10, 15, 20)               │
│                                                               │
│ INTRO POOL (tournament selection)                            │
│   Active: agent_2 (fitness: 8.5)                            │
│   Population: 3 agents | Avg Fitness: 7.8 | Diversity: 0.42 │
│                                                               │
│ BODY POOL (tournament selection)                             │
│   Active: agent_1 (fitness: 8.2)                            │
│   Population: 3 agents | Avg Fitness: 7.5 | Diversity: 0.38 │
│                                                               │
│ 🧬 Evolution Event: Intro pool evolved (parents: agent_2 +   │
│    agent_3 → offspring: agent_4)                             │
└───────────────────────────────────────────────────────────────┘
```

### 3. Add Coordinator Activity Panel

**New panel showing:**
- Research status (if Tavily enabled)
- Context synthesis status
- Critique results
- Revision requests

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🎯 Coordinator Activity                                       ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ [1. RESEARCH] ✓ Complete (5 sources via Tavily)             │
│ [2. DISTRIBUTE] ✓ Context synthesized for content agents    │
│ [3-5. GENERATE] ⟳ Content agents working...                 │
│ [6. AGGREGATE] ⏸ Waiting for content completion             │
│ [7. CRITIQUE] ○ Not started                                 │
│                                                               │
│ Revision Loop: 0/2 iterations used                           │
└───────────────────────────────────────────────────────────────┘
```

### 4. Update Memory Panel to Show All Layers

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 🧠 Retrieved Memories (Hierarchical)                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ COORDINATOR (L1):                                             │
│   Reasoning: 3 patterns | Domain: 2 facts                    │
│                                                               │
│ INTRO (L2):                                                   │
│   Reasoning: 4 patterns (2 inherited) | Domain: 3 facts      │
│   1. Start with hook about recent breakthroughs...           │
│                                                               │
│ BODY (L2):                                                    │
│   Reasoning: 6 patterns (3 inherited) | Domain: 5 facts      │
│   1. Use comparative analysis structure...                    │
└───────────────────────────────────────────────────────────────┘
```

## State Data Available

From `state.py`, we have:

### HierarchicalState Fields
```python
- current_layer: int  # Which layer is executing (1-3)
- coordinator_intent: str  # Parsed intent
- coordinator_critique: Dict[str, str]  # Feedback per agent
- revision_requested: bool  # Whether refinement needed
- quality_threshold_met: bool  # Quality acceptable
- layer_outputs: Dict[int, Dict[str, AgentOutput]]  # Outputs by layer
```

### BlogState Fields (still used)
```python
- intro_reasoning: str  # <think> content
- body_reasoning: str
- conclusion_reasoning: str
- reasoning_patterns_used: Dict[str, int]  # {role: count}
- domain_knowledge_used: Dict[str, int]  # {role: count}
- generation_number: int  # Which generation
```

## Pipeline Data Available

From `pipeline.py`, the Pipeline class has:

```python
self.agent_pools: Dict[str, AgentPool]
# {'intro': AgentPool, 'body': AgentPool, 'conclusion': AgentPool}

self.coordinator: CoordinatorAgent
self.specialists: Dict[str, BaseAgent]
# {'researcher': ResearcherAgent, 'fact_checker': FactCheckerAgent, ...}

self.enable_research: bool
self.enable_specialists: bool
self.enable_revision: bool
self.max_revisions: int
```

### AgentPool API
```python
pool.size() -> int  # Current population size
pool.avg_fitness() -> float  # Average fitness
pool.measure_diversity() -> float  # Diversity metric
pool.generation -> int  # Current generation
pool.select_active_agent() -> BaseAgent  # Currently active agent
```

## Implementation Strategy

### Option 1: Extend Existing Visualizer
Add methods to `StreamVisualizer`:
- `create_hierarchy_status_table(state, pipeline)`
- `create_pool_evolution_panel(state, pipeline)`
- `create_coordinator_panel(state)`

Update `display_stream()` to use new panels.

### Option 2: Create New HierarchicalVisualizer
Inherit from `StreamVisualizer` and override methods:
```python
class HierarchicalVisualizer(StreamVisualizer):
    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline  # Access to pools, coordinator

    def create_status_table(self, state: BlogState) -> Table:
        # New hierarchical version
        ...
```

**Recommended:** Option 2 - cleaner separation, backward compatible

## Files to Update

1. **`src/lean/visualization.py`**
   - Add `HierarchicalVisualizer` class
   - Implement new panel methods
   - Add pool/coordinator tracking

2. **`src/lean/pipeline.py`**
   - Update `self.visualizer` initialization to use `HierarchicalVisualizer`
   - Pass `self` (pipeline) to visualizer so it can access pools

3. **`src/lean/state.py`**
   - Ensure all hierarchical fields are populated correctly
   - Add helper methods if needed (e.g., `get_active_agents()`)

## Testing

Create test showing all states:
1. Coordinator research phase
2. Content generation with pools
3. Specialist invocation
4. Evolution event
5. Revision loop

Run with: `python main.py --config test` (has `visualization: enabled: true`)

## Related Issues

- Coordinator status not visible (research progress, critique)
- Pool evolution events not visible (selection, reproduction)
- Specialist agent activity not shown
- Revision loop progress not tracked
- Memory inheritance not visualized (which patterns came from parents)

## Benefits of Update

1. **Transparency**: Users see full hierarchical workflow
2. **Debugging**: Easier to spot issues in coordinator/specialist flow
3. **Learning**: Visualize pattern inheritance and pool evolution
4. **Motivation**: See the LEAN features actually working

---

**Created**: 2025-10-22
**Status**: Documented for feature branch implementation
**Priority**: High (visualization is key to understanding LEAN)
