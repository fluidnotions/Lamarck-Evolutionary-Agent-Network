# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HVAS Mini** is a research prototype demonstrating Hierarchical Vector Agent System concepts. It features multiple AI agents that:
- Maintain individual RAG memory (ChromaDB) for storing successful outputs
- Evolve parameters (temperature) based on performance scores
- Coordinate through LangGraph's StateGraph workflow
- Learn across generations by retrieving and reusing past successful patterns

**Core Research Question**: Can agents with individual memory and parameter evolution demonstrate measurable learning improvements across similar tasks?

## Build & Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run the main demo (generates 5 blog posts, shows learning)
uv run python main.py

# Run tests
uv run pytest                          # All tests
uv run pytest tests/test_agents.py -v  # Specific test file
uv run pytest -m integration           # Integration tests only
uv run pytest --cov=src/hvas_mini      # With coverage

# Type checking
uv run mypy src/hvas_mini

# Import verification (after uv sync)
uv run python test_imports.py
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add your `ANTHROPIC_API_KEY`
3. Key configuration knobs:
   - `MEMORY_SCORE_THRESHOLD=7.0` - Only memories scoring ≥7.0 are stored
   - `BASE_TEMPERATURE=0.7` - Starting temperature for all agents
   - `EVOLUTION_LEARNING_RATE=0.1` - How aggressively parameters evolve
   - `ENABLE_VISUALIZATION=true` - Show Rich terminal UI during execution

## Architecture: The Big Picture

### State Flow Through LangGraph

The system uses **LangGraph's StateGraph** with a sequential workflow:

```
BlogState (TypedDict) →
  IntroAgent → writes state["intro"]
    ↓
  BodyAgent → reads state["intro"], writes state["body"]
    ↓
  ConclusionAgent → reads state["intro"] + state["body"], writes state["conclusion"]
    ↓
  ContentEvaluator → reads all content, writes state["scores"]
    ↓
  _evolution_node → stores memories if score ≥ threshold, updates agent.parameters
```

**Critical**: `BlogState` is the **single source of truth** passed between all nodes. Agents don't directly communicate—they coordinate through shared state.

### The Learning Loop

Each agent goes through this cycle **per generation**:

1. **Retrieve**: Query ChromaDB for semantically similar past outputs (using topic as query)
2. **Generate**: Create content with current `temperature`, informed by retrieved memories
3. **Evaluate**: ContentEvaluator scores output (0-10 scale, multi-factor heuristics)
4. **Store**: If score ≥ `MEMORY_SCORE_THRESHOLD`, store in ChromaDB with embeddings
5. **Evolve**: Adjust temperature based on rolling average of last 5 scores:
   - avg < 6.0 → decrease temperature (reduce randomness)
   - avg > 8.0 → increase temperature (increase creativity)
   - else → nudge toward target of 7.0

### Memory Architecture (RAG)

Each agent has a **separate ChromaDB collection**:
- `intro_memories` - IntroAgent's successful introductions
- `body_memories` - BodyAgent's successful body sections
- `conclusion_memories` - ConclusionAgent's successful conclusions

**Why separate?** Prevents cross-contamination and allows per-agent specialization. An intro pattern shouldn't influence body generation.

**Storage**: ChromaDB persists to `./data/memories/` with sentence-transformers embeddings (`all-MiniLM-L6-v2`).

### Module Responsibilities

- **`state.py`**: Defines `BlogState` TypedDict (LangGraph state) and `AgentMemory` Pydantic model
- **`memory.py`**: `MemoryManager` class - wraps ChromaDB operations (store, retrieve, stats)
- **`agents.py`**:
  - `BaseAgent` abstract class with RAG retrieval + evolution logic
  - `IntroAgent`, `BodyAgent`, `ConclusionAgent` - specialized implementations
  - `create_agents()` factory - instantiates all agents with their MemoryManagers
- **`evaluation.py`**: `ContentEvaluator` - multi-factor heuristic scoring (length, hooks, structure, etc.)
- **`evolution.py`**: Utility functions for temperature adjustment calculations
- **`visualization.py`**: `StreamVisualizer` - Rich terminal UI for live execution updates
- **`pipeline.py`**: `HVASMiniPipeline` - orchestrates everything via LangGraph's `StateGraph`

## Key Design Patterns

### 1. LangGraph Node Pattern
All agents implement `async def __call__(self, state: BlogState) -> BlogState`. This makes them compatible as LangGraph nodes. The pipeline builds the graph:

```python
workflow = StateGraph(BlogState)
workflow.add_node("intro", self.agents["intro"])  # IntroAgent instance
workflow.add_node("body", self.agents["body"])
# ... etc
workflow.add_edge("intro", "body")  # Sequential execution
```

### 2. Pending Memory Pattern
Agents don't store memories immediately after generation. Instead:
1. Agent generates content, stores in `self.pending_memory`
2. Evaluator scores the content
3. Evolution node calls `agent.store_memory(score)` which decides whether to persist

**Why?** Evaluation must happen before storage decision (score-based threshold).

### 3. Streaming Visualization
Pipeline uses `astream()` with `stream_mode="values"` to get state snapshots after each node execution. The visualizer consumes this stream and updates Rich UI panels in real-time.

## Common Development Patterns

### Adding a New Agent Type

1. **Inherit from BaseAgent**:
```python
class SummaryAgent(BaseAgent):
    @property
    def content_key(self) -> str:
        return "summary"  # Key in BlogState

    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        # Your generation logic with memory examples
```

2. **Update BlogState** in `state.py`:
```python
class BlogState(TypedDict):
    # ... existing fields ...
    summary: str  # Add your field
```

3. **Add to pipeline** in `pipeline.py`:
```python
workflow.add_node("summary", self.agents["summary"])
workflow.add_edge("conclusion", "summary")  # Insert in workflow
```

4. **Add to evaluator** in `evaluation.py`:
```python
scores["summary"] = self._score_summary(state["summary"], state["topic"])
```

### Customizing Evaluation Criteria

Edit `evaluation.py` scoring methods:
```python
def _score_intro(self, intro: str, topic: str) -> float:
    score = 5.0  # Base score

    # Add your custom criteria
    if "your_criterion" in intro.lower():
        score += 1.5

    return min(10.0, score)
```

See `docs/custom-evaluation.md` for detailed patterns (LLM-based evaluation, multi-objective scoring, etc.).

### Modifying Evolution Strategy

Edit `agents.py` `BaseAgent.evolve_parameters()`:
```python
def evolve_parameters(self, score: float, state: BlogState):
    # Current: temperature only
    # To add: max_tokens, top_p, etc.

    # Example: evolve max_tokens based on body length
    if self.role == "body":
        target_length = 300
        actual_length = len(state["body"].split())
        if actual_length < target_length * 0.8:
            self.parameters["max_tokens"] += 50
```

## Testing Strategy

Tests are organized by component:
- `test_state.py` - Pydantic validation, state creation
- `test_memory.py` - ChromaDB operations, threshold filtering
- `test_agents.py` - BaseAgent evolution, memory storage
- `test_evaluation.py` - Scoring function correctness
- `test_visualization.py` - Rich UI component creation
- `test_pipeline.py` - Full pipeline execution
- `test_integration.py` - End-to-end learning demonstration

**Note**: Tests require actual dependencies (ChromaDB, transformers). Use temp directories for ChromaDB persistence in tests.

## Observing Learning

After running `main.py`, check:

1. **Memory accumulation**: `ls -lh data/memories/` - ChromaDB persisted data
2. **Score trends**: Compare Generation 1 vs Generation 5 scores in output
3. **Parameter convergence**: Watch temperature values in visualization
4. **Retrieval patterns**: Generations 2 and 4 (similar topics) should show memory retrievals

Expected: ~0.5-1.0 point score improvement on similar topics after memories are stored.

## Important Notes

- **Sequential Execution**: Despite "hierarchical" naming, current implementation runs agents sequentially (intro → body → conclusion). See `docs/langgraph-patterns.md` for parallel execution patterns.

- **Score Threshold**: Default 7.0 means only high-quality outputs persist. Lower threshold = more memories but potentially lower quality examples.

- **Temperature Bounds**: Evolution keeps temperature between `MIN_TEMPERATURE` (0.5) and `MAX_TEMPERATURE` (1.0) to prevent extreme values.

- **Memory Retrieval Count**: Tracks how often each memory is reused. High retrieval count = valuable pattern.

- **Worktrees Directory**: Contains git worktrees for feature branches (gitignored). Each has `docs/<branch-name>/AGENT_TASK.md` with implementation details.

## Documentation

- `README.md` - Theory, quick start, configuration
- `spec.md` - Complete technical specification
- `WORK_DIVISION.md` - Implementation plan (8 feature branches)
- `docs/extending-agents.md` - Agent customization patterns
- `docs/custom-evaluation.md` - Evaluation system customization
- `docs/langgraph-patterns.md` - Advanced LangGraph workflows (parallel execution, conditional routing, etc.)

## Research Context

This prototype tests 4 hypotheses:
1. **Individual Memory**: Agents with RAG perform better than without
2. **Parameter Evolution**: Self-adjusting parameters converge to optimal values
3. **Transfer Learning**: Memories improve performance on similar (not identical) tasks
4. **Hierarchical Coordination**: Sequential agents with context passing produce coherent outputs

The demo (5 topics, 2 pairs of similar topics) is designed to demonstrate transfer learning (#3).
