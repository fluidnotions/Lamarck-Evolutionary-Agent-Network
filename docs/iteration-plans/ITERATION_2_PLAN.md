# HVAS Mini - Iteration 2: True Hierarchical Architecture

## Overview

This iteration transforms HVAS Mini from a sequential pipeline into a true hierarchical, self-organizing agent system with:
- Concurrent async agent execution
- Agent-to-agent trust weighting
- Timestamped memory decay
- Dynamic graph topology via meta-agent
- Enhanced visualization of emergent organization

---

## Current State Analysis

### What's Implemented (Iteration 1)

- **Sequential execution**: intro → body → conclusion → evaluate → evolve
- **Individual RAG memory**: Each agent has ChromaDB collection
- **Parameter evolution**: Temperature adjustment based on scores
- **Fixed graph**: Static 3-agent workflow in LangGraph
- **Score-based storage**: Only memories ≥7.0 stored
- **Rich visualization**: Terminal UI with status tables

### Limitations Identified

1. ❌ **No true concurrency**: Agents run sequentially, not in parallel
2. ❌ **No inter-agent relationships**: No trust weights between agents
3. ❌ **No memory decay**: Old memories persist indefinitely with equal weight
4. ❌ **Static topology**: Cannot spawn/merge/remove agents at runtime
5. ❌ **Limited feedback**: Only downward evaluation, no upward propagation

---

## Refactor Goals

### 1. True Concurrency (M1)
Replace sequential execution with async parallel processing where agents in the same layer run concurrently.

**Target Architecture**:
```
        TopAgent (intent)
             ↓
    ┌────────┼────────┐
    ↓        ↓        ↓  (concurrent)
IntroAgent BodyAgent ConcAgent
    ↓        ↓        ↓
    └────────┼────────┘
             ↓
        Evaluator (aggregate)
```

### 2. Agent Weighting System (M2)
Each agent maintains trust weights toward other agents it receives input from.

**Data Structure**:
```python
agent.weights = {
    "intro": 0.8,     # Trust in intro agent's output
    "body": 0.6,      # Trust in body agent's output
    "meta": 1.0       # Trust in meta-agent's directives
}
```

### 3. Memory Decay (M3)
ChromaDB entries include timestamps and relevance decays over time.

**Decay Formula**:
```
relevance = similarity * exp(-λ * Δt) * (score/10)
```

### 4. Meta-Agent (M4)
Runtime graph modification based on system metrics.

**Capabilities**:
- Spawn new agents if diversity drops
- Merge agents if coherence is too high (redundancy)
- Remove underperforming agents
- Adjust edge weights in graph

### 5. Enhanced Visualization (M5)
Show emergent organization through:
- Agent embedding space (2D projection)
- Weight matrix heatmap
- Graph topology evolution
- Memory age distribution

---

## Dependency Graph

```
M1: async-orchestration (BLOCKING - must be first)
    ├── M2: agent-weighting (parallel after M1)
    ├── M3: memory-decay (parallel after M1)
    └── M4: meta-agent (depends on M1, can run parallel with M2/M3)
        └── M5: visualization (depends on M1-M4, integrates all)
```

**Execution Strategy**:
1. M1 first (blocking) - enables concurrency
2. M2, M3, M4 in parallel (all need M1)
3. M5 last (needs data from M1-M4)

---

## Feature Branches

### Branch 1: `feature/async-orchestration` (M1)
**Priority**: CRITICAL - BLOCKING
**Execution**: Sequential (must complete first)

**Changes**:
- Convert `HVASMiniPipeline._build_graph()` to support parallel nodes
- Make all agent `__call__` methods truly async
- Add synchronization barriers between layers
- Implement concurrent execution with `asyncio.gather()`
- Add timing instrumentation for profiling

**Files Modified**:
- `src/hvas_mini/pipeline.py`
- `src/hvas_mini/agents.py`
- `src/hvas_mini/state.py` (add timing fields)

**New Files**:
- `src/hvas_mini/orchestration/` (package)
  - `async_coordinator.py`
  - `sync_barrier.py`

---

### Branch 2: `feature/agent-weighting` (M2)
**Priority**: HIGH
**Execution**: Parallel (after M1)

**Changes**:
- Add `weights: Dict[str, float]` to BaseAgent
- Implement weight update logic based on peer performance
- Store weights in agent state and memory metadata
- Add weighted context aggregation when reading from multiple sources

**Files Modified**:
- `src/hvas_mini/agents.py` (add weights attribute)
- `src/hvas_mini/state.py` (add weight tracking to BlogState)

**New Files**:
- `src/hvas_mini/weighting/` (package)
  - `trust_manager.py`
  - `weight_updates.py`

---

### Branch 3: `feature/memory-decay` (M3)
**Priority**: HIGH
**Execution**: Parallel (after M1)

**Changes**:
- Add timestamp metadata to all ChromaDB entries
- Implement decay-weighted retrieval in MemoryManager
- Add pruning mechanism (keep top-N most relevant)
- Make decay rate (`λ`) configurable via .env

**Files Modified**:
- `src/hvas_mini/memory.py`
- `src/hvas_mini/state.py` (AgentMemory includes timestamp weight)

**New Files**:
- `src/hvas_mini/memory/decay.py`

**Config Added**:
```bash
# .env
MEMORY_DECAY_LAMBDA=0.01  # Higher = faster decay
MEMORY_MAX_AGE_DAYS=30
MEMORY_PRUNE_TO_TOP_N=100
```

---

### Branch 4: `feature/meta-agent` (M4)
**Priority**: MEDIUM
**Execution**: Parallel (after M1, works with M2/M3)

**Changes**:
- Create MetaAgent class that monitors system metrics
- Implement graph mutation operations (spawn, merge, remove)
- Add thresholds for triggering topology changes
- Record graph evolution history

**Files Modified**:
- `src/hvas_mini/pipeline.py` (integrate meta-agent node)
- `src/hvas_mini/state.py` (add topology tracking)

**New Files**:
- `src/hvas_mini/meta/` (package)
  - `meta_agent.py`
  - `graph_mutations.py`
  - `metrics_monitor.py`

**Config Added**:
```bash
# .env
META_AGENT_ENABLED=true
META_DIVERSITY_THRESHOLD=0.3
META_COHERENCE_THRESHOLD=0.8
META_PERFORMANCE_THRESHOLD=6.0
```

---

### Branch 5: `feature/visualization-v2` (M5)
**Priority**: MEDIUM
**Execution**: Sequential (after M1-M4)

**Changes**:
- Add embedding space visualization (2D t-SNE/UMAP)
- Add weight matrix heatmap
- Add graph topology viewer with networkx
- Add memory age histogram
- Integrate with existing StreamVisualizer

**Files Modified**:
- `src/hvas_mini/visualization.py`

**New Files**:
- `src/hvas_mini/visualization/` (package)
  - `embedding_plot.py`
  - `weight_heatmap.py`
  - `topology_view.py`
  - `memory_age_plot.py`

**Dependencies Added**:
```toml
# pyproject.toml
"matplotlib>=3.7.0",
"networkx>=3.0",
"umap-learn>=0.5.0",  # Optional, falls back to PCA
```

---

## State Schema Changes

### BlogState Extensions

```python
class BlogState(TypedDict):
    # Existing fields...
    topic: str
    intro: str
    body: str
    conclusion: str
    scores: Dict[str, float]
    retrieved_memories: Dict[str, List[str]]
    parameter_updates: Dict[str, Dict[str, float]]
    generation_id: str
    timestamp: str
    stream_logs: List[str]

    # NEW: Concurrency tracking
    agent_timings: Dict[str, Dict[str, float]]  # {agent: {start, end, duration}}

    # NEW: Weight tracking
    agent_weights: Dict[str, Dict[str, float]]  # {agent: {peer: trust_weight}}

    # NEW: Topology tracking
    active_agents: List[str]  # Current agents in graph
    topology_version: int  # Increments on graph mutation
    topology_history: List[Dict]  # [{timestamp, action, agents}]
```

---

## Implementation Order

### Phase 1: Foundation (Week 1)
1. ✅ Create work division plan (this document)
2. Create feature branches and worktrees
3. Implement M1 (async-orchestration)
4. Test concurrent execution
5. Merge M1 to main

### Phase 2: Core Features (Week 2)
6. Implement M2 (agent-weighting) in parallel
7. Implement M3 (memory-decay) in parallel
8. Implement M4 (meta-agent) in parallel
9. Test each branch independently
10. Merge M2, M3, M4 to main

### Phase 3: Integration (Week 3)
11. Implement M5 (visualization-v2)
12. Integration testing
13. Performance benchmarking
14. Documentation updates
15. Final merge and release

---

## Testing Strategy

### Unit Tests (per branch)
- `test_async_orchestration.py` - Verify concurrent execution
- `test_agent_weighting.py` - Verify trust weight updates
- `test_memory_decay.py` - Verify decay calculations
- `test_meta_agent.py` - Verify graph mutations
- `test_visualization_v2.py` - Verify plot generation

### Integration Tests
- `test_concurrent_learning.py` - Full system with concurrency
- `test_emergent_organization.py` - Meta-agent graph evolution
- `test_performance_comparison.py` - Sequential vs concurrent benchmarks

### Acceptance Criteria

| Feature | Metric | Target |
|---------|--------|--------|
| Concurrency | Agent overlap time | >50% |
| Weighting | Weight convergence | <0.1 change over 10 gens |
| Memory Decay | Old entry relevance | <20% after 30 days |
| Meta-Agent | Graph mutations | ≥1 per 20 generations |
| Visualization | Plot generation | <2s per update |

---

## Rollback Plan

Each milestone is independently mergeable. If any milestone fails:
1. Revert specific branch
2. System remains functional on previous milestone
3. No breaking changes to core API

---

## Success Metrics

### Before (Iteration 1)
- Sequential execution: 100% sequential
- No inter-agent relationships
- Memory grows indefinitely
- Fixed 3-agent topology
- No topology visualization

### After (Iteration 2)
- Concurrent execution: 50-80% overlap
- Agent trust weights: 3x3 matrix evolving
- Memory: Decay-weighted, top-100 pruned
- Dynamic topology: Meta-agent can modify
- Full visualization: embeddings, weights, topology, memory age

---

## Next Steps

1. Create feature branches following CLAUDE.md workflow
2. Create AGENT_TASK.md in each worktree
3. Implement milestones M1-M5 in order
4. Test, merge, and validate
5. Update documentation and demos
