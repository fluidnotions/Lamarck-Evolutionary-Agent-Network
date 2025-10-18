# HVAS Mini - Technical Documentation

Complete implementation details, setup instructions, and architecture reference for the HVAS Mini prototype.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Demo](#running-the-demo)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Observing Learning](#observing-learning)
- [Customization](#customization)
- [Development Workflow](#development-workflow)

---

## Quick Start

```bash
# Clone repository
cd hvas-mini

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run demo (5 topic generations)
uv run python main.py
```

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Anthropic API key** ([get one here](https://console.anthropic.com/))
- **uv package manager** ([installation guide](https://github.com/astral-sh/uv))

### Dependencies

All dependencies are managed via `uv` and defined in `pyproject.toml`:

- **LangGraph** - Workflow orchestration and streaming
- **LangChain** - LLM integration (Anthropic Claude)
- **ChromaDB** - Vector database for RAG memory
- **sentence-transformers** - Text embeddings
- **Rich** - Terminal visualization
- **Pydantic** - Type-safe state management

Install with:
```bash
uv sync
```

---

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# LLM Configuration
ANTHROPIC_API_KEY=your_key_here
MODEL_NAME=claude-3-haiku-20240307  # or claude-3-5-sonnet-20241022
BASE_TEMPERATURE=0.7

# Memory Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
MEMORY_SCORE_THRESHOLD=7.0  # Only store outputs scoring ≥ 7.0
MAX_MEMORIES_RETRIEVE=3

# Evolution Configuration
ENABLE_PARAMETER_EVOLUTION=true
EVOLUTION_LEARNING_RATE=0.1
MIN_TEMPERATURE=0.5
MAX_TEMPERATURE=1.0

# Visualization
ENABLE_VISUALIZATION=true
SHOW_MEMORY_RETRIEVAL=true
SHOW_PARAMETER_CHANGES=true
```

### Configuration Details

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MEMORY_SCORE_THRESHOLD` | 7.0 | Minimum score for memory storage (credibility filter) |
| `BASE_TEMPERATURE` | 0.7 | Starting temperature for all agents |
| `EVOLUTION_LEARNING_RATE` | 0.1 | How aggressively parameters adjust |
| `MAX_MEMORIES_RETRIEVE` | 3 | Number of past examples to retrieve per generation |
| `MIN_TEMPERATURE` | 0.5 | Lower bound for temperature evolution |
| `MAX_TEMPERATURE` | 1.0 | Upper bound for temperature evolution |

---

## Running the Demo

### Basic Execution

```bash
uv run python main.py
```

Expected output:
- Live visualization of agent execution (Rich UI)
- Memory retrieval logs showing semantic matches
- Parameter evolution tracking (temperature changes)
- Performance scores for each section (intro, body, conclusion)
- Final statistics showing learning trends

### Demo Experiment Design

The default demo runs 5 topic generations:

```python
topics = [
    "introduction to machine learning",     # Gen 1: Baseline (no memories)
    "machine learning applications",        # Gen 2: Similar (tests transfer)
    "python programming basics",            # Gen 3: New domain
    "python for data science",              # Gen 4: Similar (tests transfer)
    "artificial intelligence ethics"        # Gen 5: New domain
]
```

**Why this sequence?**
- Generations 2 and 4 test **transfer learning** (semantic proximity to Gen 1 and 3)
- Validates that agents retrieve geometrically similar memories
- Demonstrates score improvement on similar topics

---

## Architecture

### System Components

```
┌─────────────────────────────────────────┐
│      LangGraph Orchestrator             │
│      (StateGraph + Streaming)           │
└────────────┬────────────────────────────┘
             │
    ┌────────┼────────┬─────────┐
    │        │        │         │
┌───▼──┐ ┌──▼───┐ ┌──▼───┐ ┌──▼────┐
│Intro │ │Body  │ │Concl │ │Eval & │
│Agent │ │Agent │ │Agent │ │Evolve │
└───┬──┘ └──┬───┘ └──┬───┘ └───────┘
    │       │        │
    ▼       ▼        ▼
[ChromaDB] [ChromaDB] [ChromaDB]
```

### State Flow Through LangGraph

The system uses **LangGraph's StateGraph** with sequential workflow:

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

**Critical:** `BlogState` is the **single source of truth** passed between all nodes. Agents coordinate through shared state, not direct communication.

### The Learning Loop (Per Generation)

Each agent executes this cycle:

1. **Retrieve**: Query ChromaDB for semantically similar past outputs
   - Uses topic as query embedding
   - Retrieves top-k results by cosine similarity
   - Only searches agent's own memory collection

2. **Generate**: Create content with current parameters
   - Uses retrieved memories as examples in prompt
   - Applies current temperature setting
   - Reads relevant state fields (topic, previous sections)

3. **Evaluate**: ContentEvaluator scores output (0-10 scale)
   - Multi-factor heuristics (length, hooks, structure, etc.)
   - Independent scores per section
   - Writes to `state["scores"]`

4. **Store**: If score ≥ `MEMORY_SCORE_THRESHOLD`
   - Embed output with sentence-transformers
   - Store in ChromaDB with metadata (score, topic, timestamp)
   - Creates semantic memory for future retrieval

5. **Evolve**: Adjust parameters based on rolling average
   - Tracks last 5 scores per agent
   - Adjusts temperature:
     - `avg < 6.0` → decrease (reduce randomness)
     - `avg > 8.0` → increase (increase creativity)
     - else → nudge toward target of 7.0

### Memory Architecture (RAG)

Each agent maintains a **separate ChromaDB collection**:

```
intro_memories       → IntroAgent's successful introductions
body_memories        → BodyAgent's successful body sections
conclusion_memories  → ConclusionAgent's successful conclusions
```

**Why separate collections?**
- Prevents cross-contamination between agent roles
- Allows per-agent specialization in memory topology
- Intro patterns don't influence conclusion generation
- Each agent builds its own "map of what works"

**Storage location:** `./data/memories/` (gitignored)

**Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (384 dimensions)

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| **`state.py`** | Defines `BlogState` TypedDict (LangGraph state) and `AgentMemory` Pydantic model |
| **`memory.py`** | `MemoryManager` class - wraps ChromaDB operations (store, retrieve, stats) |
| **`agents.py`** | `BaseAgent` abstract class + `IntroAgent`, `BodyAgent`, `ConclusionAgent` implementations |
| **`evaluation.py`** | `ContentEvaluator` - multi-factor heuristic scoring |
| **`evolution.py`** | Utility functions for temperature adjustment calculations |
| **`visualization.py`** | `StreamVisualizer` - Rich terminal UI for live updates |
| **`pipeline.py`** | `HVASMiniPipeline` - orchestrates everything via LangGraph |

---

## Project Structure

```
hvas-mini/
├── main.py                      # Main entry point and demo
├── pyproject.toml               # uv configuration
├── .env.example                 # Environment template
├── README.md                    # Research narrative (for employers)
├── CLAUDE.md                    # Guidance for Claude Code
├── docs/
│   ├── technical.md            # This file
│   ├── extending-agents.md     # Agent customization patterns
│   ├── custom-evaluation.md    # Evaluation system customization
│   └── langgraph-patterns.md   # Advanced LangGraph workflows
│
├── src/lean/              # Main package
│   ├── __init__.py
│   ├── state.py                # State definitions
│   ├── memory.py               # MemoryManager (ChromaDB + RAG)
│   ├── agents.py               # BaseAgent + specialized agents
│   ├── evolution.py            # Parameter evolution logic
│   ├── evaluation.py           # ContentEvaluator
│   ├── visualization.py        # StreamVisualizer (Rich UI)
│   └── pipeline.py             # HVASMiniPipeline (LangGraph)
│
├── data/                        # Runtime data (gitignored)
│   └── memories/               # ChromaDB persistence
│
├── logs/                        # Execution logs (gitignored)
│   └── runs/
│
└── tests/                       # Test files
    ├── test_state.py
    ├── test_memory.py
    ├── test_agents.py
    ├── test_evaluation.py
    ├── test_visualization.py
    ├── test_pipeline.py
    └── test_integration.py
```

---

## Testing

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test File

```bash
uv run pytest tests/test_agents.py -v
```

### Run with Coverage

```bash
uv run pytest --cov=src/lean
```

### Run Integration Tests Only

```bash
uv run pytest -m integration
```

### Type Checking

```bash
uv run mypy src/lean
```

### Test Organization

Tests are organized by component:

| Test File | Coverage |
|-----------|----------|
| `test_state.py` | Pydantic validation, state creation |
| `test_memory.py` | ChromaDB operations, threshold filtering |
| `test_agents.py` | BaseAgent evolution, memory storage |
| `test_evaluation.py` | Scoring function correctness |
| `test_visualization.py` | Rich UI component creation |
| `test_pipeline.py` | Full pipeline execution |
| `test_integration.py` | End-to-end learning demonstration |

**Note:** Tests require actual dependencies (ChromaDB, transformers). Uses temp directories for ChromaDB persistence.

---

## Observing Learning

### Metrics to Watch

1. **Memory Growth**
   - Track how many memories each agent accumulates
   - Check `data/memories/` directory size

2. **Score Trends**
   - Compare Generation 1 vs Generation 5 scores
   - Look for improvement on similar topics (Gen 2 vs Gen 1, Gen 4 vs Gen 3)

3. **Parameter Convergence**
   - Watch temperature values stabilize over time
   - Different agents should converge to different values

4. **Retrieval Patterns**
   - Generations 2 and 4 should show memory retrievals
   - Check which memories get reused (retrieval_count in metadata)

### Expected Results

After 5 generations:

- **Memory Accumulation**: Each agent stores 2-4 high-quality memories
- **Score Improvement**: Later similar topics score ~0.5-1.0 points higher
- **Parameter Stability**: Temperatures converge to optimal ranges (0.6-0.8)
- **Retrieval Effectiveness**: Agents retrieve 1-3 relevant memories per generation

### Analysis Commands

```bash
# Check memory persistence
ls -lh data/memories/

# View agent parameters in output
# Look for "Temperature updated" messages

# Compare scores
# Generation 1 intro score vs Generation 2 intro score
```

### Metrics Tracked

**Content Quality Scores** (0-10 scale):
- **Introduction**: hooks, relevance, length
- **Body**: detail, structure, examples
- **Conclusion**: summary, call-to-action, coherence

**Memory Statistics**:
- Total memories stored per agent
- Average quality of stored memories
- Retrieval frequency per memory

**Parameter Evolution**:
- Temperature changes over time
- Score trends per agent
- Convergence behavior

---

## Customization

### Adding a New Agent Type

**1. Create Agent Class** (in `agents.py`):

```python
class SummaryAgent(BaseAgent):
    @property
    def content_key(self) -> str:
        return "summary"  # Key in BlogState

    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        # Your generation logic with memory examples
        prompt = f"Generate summary for: {state['topic']}\n\n"

        if memories:
            prompt += "Examples of good summaries:\n"
            for mem in memories:
                prompt += f"- {mem['content']}\n"

        # Call LLM, return result
        return result
```

**2. Update State Definition** (in `state.py`):

```python
class BlogState(TypedDict):
    # ... existing fields ...
    summary: str  # Add your field
```

**3. Add to Pipeline** (in `pipeline.py`):

```python
# In create_workflow method
workflow.add_node("summary", self.agents["summary"])
workflow.add_edge("conclusion", "summary")  # Insert in workflow
```

**4. Add to Evaluator** (in `evaluation.py`):

```python
def evaluate(self, state: BlogState) -> BlogState:
    # ... existing scoring ...
    scores["summary"] = self._score_summary(state["summary"], state["topic"])
    return state

def _score_summary(self, summary: str, topic: str) -> float:
    score = 5.0
    # Your scoring logic
    return min(10.0, score)
```

See `docs/extending-agents.md` for detailed patterns.

### Customizing Evaluation Criteria

Edit scoring methods in `evaluation.py`:

```python
def _score_intro(self, intro: str, topic: str) -> float:
    score = 5.0  # Base score

    # Length check
    word_count = len(intro.split())
    if 50 <= word_count <= 150:
        score += 1.0

    # Add your custom criteria
    if "compelling hook" in intro.lower():
        score += 1.5

    if has_question(intro):
        score += 0.5

    return min(10.0, score)
```

See `docs/custom-evaluation.md` for LLM-based evaluation and multi-objective scoring.

### Modifying Evolution Strategy

Edit `BaseAgent.evolve_parameters()` in `agents.py`:

```python
def evolve_parameters(self, score: float, state: BlogState):
    """Evolve agent parameters based on performance."""

    self.performance_history.append(score)
    if len(self.performance_history) < 5:
        return  # Need at least 5 scores

    recent_avg = sum(self.performance_history[-5:]) / 5

    # Current: temperature only
    # Add more evolvable parameters:

    # Example: evolve max_tokens based on body length
    if self.role == "body":
        target_length = 300
        actual_length = len(state["body"].split())
        if actual_length < target_length * 0.8:
            self.parameters["max_tokens"] += 50
        elif actual_length > target_length * 1.2:
            self.parameters["max_tokens"] -= 50

    # Example: evolve top_p based on score variance
    score_variance = np.var(self.performance_history[-5:])
    if score_variance > 2.0:  # High variance
        self.parameters["top_p"] -= 0.05  # More focused
    elif score_variance < 0.5:  # Low variance
        self.parameters["top_p"] += 0.05  # More diverse
```

### Changing Workflow Execution

For parallel execution or conditional routing, see `docs/langgraph-patterns.md`.

Example parallel execution:

```python
# In pipeline.py
workflow.add_node("intro", self.agents["intro"])
workflow.add_node("body", self.agents["body"])
workflow.add_node("conclusion", self.agents["conclusion"])

# Parallel execution (all run concurrently)
workflow.set_entry_point("intro")
workflow.add_edge("intro", "body")
workflow.add_edge("intro", "conclusion")  # Both read intro, run in parallel

workflow.add_node("merge", self._merge_parallel_results)
workflow.add_edge("body", "merge")
workflow.add_edge("conclusion", "merge")
workflow.set_finish_point("merge")
```

---

## Development Workflow

### Using Git Worktrees for Feature Development

See `CLAUDE.md` for the complete iteration workflow.

**Quick reference:**

```bash
# Create worktree for new feature
git worktree add worktrees/feature-name -b feature/feature-name

# Work in worktree
cd worktrees/feature-name
# ... implement feature ...

# Commit
git add -A
git commit -m "Implement feature-name"

# Move AGENT_TASK.md to avoid merge conflicts
mkdir -p docs/feature-name
git mv AGENT_TASK.md docs/feature-name/
git commit -m "Move AGENT_TASK.md to docs"

# Merge to main
cd ../../
git merge feature/feature-name
```

### Build & Run Commands

```bash
# Install dependencies
uv sync

# Run main demo
uv run python main.py

# Run specific test file
uv run pytest tests/test_agents.py -v

# Run with coverage
uv run pytest --cov=src/lean

# Type checking
uv run mypy src/lean

# Import verification
uv run python test_imports.py
```

---

## Important Notes

### Sequential vs Parallel Execution

Current implementation runs agents **sequentially**: intro → body → conclusion.

This is intentional for the blog generation use case (body needs intro context).

For parallel execution patterns, see `docs/langgraph-patterns.md`.

### Score Threshold Tuning

- **Default 7.0**: Only high-quality outputs persist
- **Lower (6.0)**: More memories, but lower quality examples
- **Higher (8.0)**: Fewer memories, only exceptional outputs

Adjust based on your use case and model capabilities.

### Temperature Bounds

Evolution keeps temperature between `MIN_TEMPERATURE` (0.5) and `MAX_TEMPERATURE` (1.0).

- **0.5**: Deterministic, focused, conservative
- **0.7**: Balanced (default starting point)
- **1.0**: Creative, diverse, exploratory

Bounds prevent extreme values that could degrade performance.

### Memory Retrieval Count

Default: retrieve top 3 memories per generation.

- **Higher (5-7)**: More context, but potentially noisy
- **Lower (1-2)**: Focused context, but may miss relevant patterns

Adjust `MAX_MEMORIES_RETRIEVE` based on your domain.

---

## Troubleshooting

### ChromaDB Persistence Issues

If memory isn't persisting between runs:

```bash
# Check data directory exists
ls -la data/memories/

# Reset memories (fresh start)
rm -rf data/memories/
uv run python main.py
```

### Model API Errors

```bash
# Verify API key
echo $ANTHROPIC_API_KEY

# Check .env file
cat .env | grep ANTHROPIC_API_KEY

# Test connection
uv run python -c "from langchain_anthropic import ChatAnthropic; ChatAnthropic(model='claude-3-haiku-20240307').invoke('test')"
```

### Import Errors

```bash
# Reinstall dependencies
uv sync --force

# Verify imports
uv run python test_imports.py
```

### Visualization Not Showing

If Rich UI doesn't display:

```bash
# Check environment variable
echo $ENABLE_VISUALIZATION

# Force enable in .env
ENABLE_VISUALIZATION=true

# Run again
uv run python main.py
```

---

## Additional Resources

- **Extending Agents**: `docs/extending-agents.md`
- **Custom Evaluation**: `docs/custom-evaluation.md`
- **LangGraph Patterns**: `docs/langgraph-patterns.md`
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Claude API Docs**: https://docs.anthropic.com/

---

**Last Updated:** 2025-01-17
