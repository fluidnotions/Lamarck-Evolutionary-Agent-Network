# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HVAS Mini** is a research prototype exploring whether AI agents can learn from experience through:
- Individual RAG memory (ChromaDB) for storing successful outputs
- Parameter evolution (temperature) based on performance scores
- Hierarchical coordination through a 3-layer structure
- Multi-pass refinement with quality gates
- Semantic distance-based context filtering

**Core Research Question**: Can agents with individual memory and self-evolution demonstrate measurable learning improvements across similar tasks?

## Current Architecture

### 3-Layer Hierarchy

```
Layer 1: Coordinator
         ├─ Parses high-level intent
         ├─ Distributes filtered context to children
         ├─ Critiques output quality
         └─ Requests revisions when quality below threshold
              ↓ context flows down
Layer 2: Content Agents (Intro, Body, Conclusion)
         ├─ Generate their section with context from coordinator
         ├─ Query specialist children for expertise
         └─ Aggregate specialist outputs
              ↓ context flows down
Layer 3: Specialists (Researcher, Fact-Checker, Stylist)
         ├─ Provide deep domain expertise
         ├─ No children (leaf nodes)
         └─ Results bubble up to parents
              ↑ results aggregate up
```

### Key Features

**Bidirectional Flow** (M7):
- Context flows down from parent to children
- Results aggregate up from children to parent
- Confidence-weighted aggregation

**Closed-Loop Refinement** (M8):
- Multi-pass execution (up to 3 passes by default)
- Coordinator critique after each pass
- Early exit when quality threshold met (avg confidence ≥ 0.8)
- Revision requests with specific feedback

**Semantic Distance Weighting** (M9):
- Hand-crafted semantic vectors for each agent
- Cosine distance calculation between agents
- Context filtering based on semantic relevance
- Closer agents share more context

### The Learning Loop

Each agent, each generation:
1. **Retrieve**: Query own ChromaDB collection for past successful outputs (semantic similarity to topic)
2. **Generate**: Create content informed by retrieved memories + weighted context from parent
3. **Evaluate**: Get scored (0-10) by ContentEvaluator heuristics
4. **Store**: If score ≥ `MEMORY_SCORE_THRESHOLD` (default 7.0), persist to ChromaDB
5. **Evolve**: Adjust temperature based on rolling average of last 5 scores

## Module Responsibilities

### Core Modules

- **`state.py`**: `BlogState` TypedDict (LangGraph state), `AgentMemory` Pydantic model, `HierarchicalState` (extends BlogState with hierarchy fields)
- **`memory.py`**: `MemoryManager` - ChromaDB wrapper (store, retrieve, stats), `DecayCalculator` (M3), `MemoryPruner` (M3)
- **`agents.py`**: `BaseAgent` abstract class, `IntroAgent`, `BodyAgent`, `ConclusionAgent`, `create_agents()` factory
- **`evaluation.py`**: `ContentEvaluator` - multi-factor heuristic scoring
- **`evolution.py`**: Temperature adjustment utilities
- **`visualization.py`**: `StreamVisualizer` - Rich terminal UI
- **`pipeline.py`**: `HVASMiniPipeline` - LangGraph StateGraph orchestration

### Hierarchy Modules (`src/hvas_mini/hierarchy/`)

- **`structure.py`**: `AgentHierarchy` class - defines 3-layer parent-child relationships, semantic vectors
- **`coordinator.py`**: `CoordinatorAgent` (Layer 1) - intent parsing, context distribution with semantic filtering, critique generation
- **`specialists.py`**: `ResearchAgent`, `FactCheckerAgent`, `StyleAgent` (Layer 3) - leaf nodes with specialized prompts
- **`executor.py`**: `HierarchicalExecutor` - manages bidirectional flow, multi-pass refinement, confidence estimation
- **`semantic.py`**: Semantic distance functions (cosine similarity, context filtering, weight computation)
- **`factory.py`**: `create_hierarchical_agents()` - instantiates all 7 agents with hierarchy

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run main demo (5 topics, shows learning)
uv run python main.py

# Run tests
uv run pytest                              # All tests
uv run pytest tests/test_hierarchical_structure.py -v  # Specific file
uv run pytest --cov=src/hvas_mini          # With coverage

# Type checking
uv run mypy src/hvas_mini
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add `ANTHROPIC_API_KEY`
3. Key configuration:
   - `MEMORY_SCORE_THRESHOLD=7.0` - Only store outputs scoring ≥7.0
   - `BASE_TEMPERATURE=0.7` - Starting temperature for all agents
   - `EVOLUTION_LEARNING_RATE=0.1` - Parameter evolution aggressiveness
   - `QUALITY_THRESHOLD=0.8` - Multi-pass early exit threshold
   - `MAX_PASSES=3` - Maximum refinement iterations

## Key Design Patterns

### 1. LangGraph Node Pattern

All agents implement `async def __call__(self, state: BlogState) -> BlogState` to be compatible as LangGraph nodes:

```python
workflow = StateGraph(BlogState)
workflow.add_node("intro", self.agents["intro"])  # IntroAgent instance
workflow.add_node("body", self.agents["body"])
workflow.add_edge("intro", "body")  # Sequential execution
```

### 2. Pending Memory Pattern

Agents don't store memories immediately:
1. Agent generates content, stores in `self.pending_memory`
2. Evaluator scores the content
3. Evolution node calls `agent.store_memory(score)` which decides whether to persist

**Why?** Evaluation must happen before storage decision.

### 3. Hierarchical Execution

`HierarchicalExecutor` manages:
- **Downward pass**: `execute_downward(state, layer)` - distribute context from parent to children
- **Upward pass**: `execute_upward(state, layer)` - aggregate child results back to parent
- **Full cycle**: `execute_full_cycle(state)` - layers 1→2→3 down, then 3→2→1 up
- **With refinement**: `execute_with_refinement(state)` - multi-pass with critique

### 4. Semantic Context Filtering

Coordinator's `distribute_context()` uses semantic distance:
```python
distance = compute_semantic_distance(hierarchy, "coordinator", child_role)
filtered_context = filter_context_by_distance(intent, distance, min_ratio=0.3)
```

Closer semantic vectors = more context shared.

## Common Development Patterns

### Adding a New Agent Type

1. Create class inheriting from `BaseAgent`
2. Implement `content_key` property (key in BlogState)
3. Implement `generate_content(state, memories, weighted_context)` method
4. Add to hierarchy in `structure.py` (if hierarchical) or `create_agents()` (if flat)
5. Update `BlogState` in `state.py` to include your field
6. Add to evaluator's scoring methods
7. Wire into pipeline workflow

### Customizing Evaluation

Edit `evaluation.py` scoring methods:
```python
def _score_intro(self, intro: str, topic: str) -> float:
    score = 5.0  # Base score
    # Add custom criteria
    if my_criterion(intro):
        score += 1.5
    return min(10.0, score)
```

### Modifying Evolution Strategy

Edit `agents.py` `BaseAgent.evolve_parameters()`:
```python
def evolve_parameters(self, score: float, state: BlogState):
    # Current: temperature only
    # Add: max_tokens, top_p, etc.
    self.parameters["max_tokens"] = adjust_based_on_output_length(state)
```

### Extending Hierarchy

Edit `src/hvas_mini/hierarchy/structure.py`:
```python
self.nodes = {
    "coordinator": AgentNode(
        role="coordinator",
        layer=1,
        children=["intro", "body", "conclusion", "your_new_agent"],
        semantic_vector=[0.0, 1.0, 0.0]
    ),
    # Add your agent
    "your_new_agent": AgentNode(
        role="your_new_agent",
        layer=2,
        children=["some_specialist"],
        semantic_vector=[0.5, 0.5, 0.5]  # Tune based on role
    ),
}
```

## Testing Strategy

Tests organized by component:
- `test_state.py` - State validation, creation
- `test_memory.py` - ChromaDB operations, thresholds
- `test_memory_decay.py` - Time-based decay (M3)
- `test_agent_weighting.py` - Trust-based weighting (M2)
- `test_meta_agent.py` - Graph optimization (M4)
- `test_async_orchestration.py` - Concurrent execution (M1)
- `test_hierarchical_structure.py` - Hierarchy structure (M6)
- `test_bidirectional_flow.py` - Context distribution, aggregation (M7)
- `test_closed_loop_refinement.py` - Multi-pass refinement (M8)
- `test_semantic_distance.py` - Semantic vector operations (M9)

## Observing Learning

After running `main.py`:
1. **Memory accumulation**: `ls -lh data/memories/` - ChromaDB data
2. **Score trends**: Compare Generation 1 vs Generation 5 scores
3. **Parameter convergence**: Watch temperature values in output
4. **Retrieval patterns**: Generations 2 and 4 should show memory retrievals

Expected: ~0.5-1.0 point score improvement on similar topics after memories stored.

## Important Notes

- **Hierarchical Execution**: Current implementation uses 3-layer hierarchy with bidirectional flow
- **Score Threshold**: Default 7.0 means only high-quality outputs persist
- **Temperature Bounds**: Evolution keeps temperature between 0.5 and 1.0
- **Multi-Pass**: Coordinator runs up to 3 refinement passes, exits early if quality ≥ 0.8
- **Semantic Vectors**: Hand-crafted, not learned (research question: does this help?)

## Research Context

This prototype tests hypotheses:
1. **Individual Memory**: Do agents with RAG perform better?
2. **Parameter Evolution**: Do self-adjusting parameters converge optimally?
3. **Transfer Learning**: Do memories improve performance on similar tasks?
4. **Hierarchical Coordination**: Does multi-layer structure with bidirectional flow produce better outputs?
5. **Semantic Filtering**: Does distance-based context filtering improve relevance?

The demo (5 topics, 2 pairs of similar) is designed to test transfer learning (#3).

## Documentation

- **`README.md`**: Research motivation, experiment design, thinking process
- **`docs/`**: Implementation patterns and customization guides

---

**This is research code. Expect experiments, not production polish.**
