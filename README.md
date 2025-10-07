# HVAS Mini - Hierarchical Vector Agent System Prototype

A research prototype demonstrating **Hierarchical Vector Agent System (HVAS)** concepts using LangGraph, featuring concurrent AI agents with individual RAG memory, parameter evolution, and real-time learning visualization.

---

## 🧠 Theory & Research Goals

### What is HVAS?

**Hierarchical Vector Agent System (HVAS)** is a research concept exploring how multiple AI agents can:

1. **Maintain Individual Memory** - Each agent has its own vector database (RAG) for storing and retrieving successful patterns
2. **Learn Through Evolution** - Agents adjust their parameters based on performance feedback
3. **Work Concurrently** - Multiple agents execute in parallel or sequence with shared state
4. **Demonstrate Emergent Behavior** - System-level learning emerges from individual agent improvements

### Core Research Questions

This prototype investigates:

- **Memory Effectiveness**: Do agents with access to their own successful past outputs generate better content?
- **Parameter Evolution**: Can agents learn optimal parameters (temperature, etc.) through score-based feedback?
- **Transfer Learning**: Do memories from similar tasks improve performance on new tasks?
- **Hierarchical Coordination**: How do specialized agents coordinate through shared state?

### Key Concepts Demonstrated

#### 1. **Individual Agent Memory (RAG)**

Each agent maintains its own ChromaDB collection storing high-quality past outputs:

```
IntroAgent Memory:
- Successful introductions (score ≥ 7.0)
- Embedded with sentence-transformers
- Retrieved by semantic similarity

BodyAgent Memory:
- Successful body sections
- Retrieved based on topic similarity

ConclusionAgent Memory:
- Successful conclusions
- Retrieved to guide future outputs
```

**Hypothesis**: Agents that retrieve relevant high-quality examples produce better outputs.

#### 2. **Parameter Evolution**

Agents evolve their parameters based on performance:

```python
if avg_score < 6.0:
    # Poor performance → reduce randomness
    temperature -= learning_rate
elif avg_score > 8.0:
    # Good performance → increase creativity
    temperature += learning_rate
```

**Hypothesis**: Self-adjusting parameters lead to optimal performance over time.

#### 3. **Hierarchical Orchestration**

LangGraph coordinates agents with shared state:

```
State: {topic, intro, body, conclusion, scores, memories, parameters}
  ↓
IntroAgent → reads topic, writes intro
  ↓
BodyAgent → reads topic + intro, writes body
  ↓
ConclusionAgent → reads all, writes conclusion
  ↓
Evaluator → scores all sections
  ↓
Evolution → stores memories, updates parameters
```

**Hypothesis**: Sequential execution with context passing enables coherent multi-agent outputs.

#### 4. **Learning Across Generations**

System improves over multiple executions:

```
Generation 1: No memories → baseline performance
Generation 2: Retrieves memories → improved output
Generation 3: Better parameters → further improvement
Generation N: Converges to optimal strategy
```

**Hypothesis**: Cumulative learning leads to measurable performance improvements.

---

## 🏗️ Architecture

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

### Technology Stack

- **LangGraph**: Workflow orchestration and streaming
- **LangChain**: LLM integration (Anthropic Claude)
- **ChromaDB**: Vector database for RAG memory
- **sentence-transformers**: Text embeddings
- **Rich**: Terminal visualization
- **Pydantic**: Type-safe state management
- **uv**: Fast Python package management

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key
- uv package manager

### Installation

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
```

### Running the Demo

```bash
# Run the demo with 5 topic generations
uv run python main.py
```

Expected output:
- Live visualization of agent execution
- Memory retrieval logs
- Parameter evolution tracking
- Performance scores for each section
- Final statistics showing learning

---

## 📊 Experiment Design

### Default Demo Experiment

The `main.py` demo runs an experiment with 5 topics:

```python
topics = [
    "introduction to machine learning",     # Gen 1: Baseline
    "machine learning applications",        # Gen 2: Similar (memory reuse)
    "python programming basics",            # Gen 3: New topic
    "python for data science",              # Gen 4: Similar (memory reuse)
    "artificial intelligence ethics"        # Gen 5: New topic
]
```

### Metrics Tracked

1. **Content Quality Scores** (0-10 scale)
   - Introduction: hooks, relevance, length
   - Body: detail, structure, examples
   - Conclusion: summary, CTA, coherence

2. **Memory Statistics**
   - Total memories stored per agent
   - Average quality of memories
   - Retrieval frequency

3. **Parameter Evolution**
   - Temperature changes over time
   - Score trends per agent
   - Convergence behavior

### Expected Results

After 5 generations, you should observe:

- **Memory Accumulation**: Each agent stores 2-4 high-quality memories
- **Score Improvement**: Later similar topics score ~0.5-1.0 points higher
- **Parameter Stability**: Temperatures converge to optimal ranges
- **Retrieval Effectiveness**: Agents retrieve 1-3 relevant memories per generation

---

## 🎯 Use Cases & Extensions

### Current Implementation: Blog Generation

- Three specialized agents (intro, body, conclusion)
- Content evaluation based on structure and quality
- Learning to write better blog posts over time

### Potential Extensions

1. **Code Generation**
   - Agents: architect, implementer, tester
   - Memory: successful code patterns
   - Evolution: code quality metrics

2. **Research Synthesis**
   - Agents: summarizer, analyzer, synthesizer
   - Memory: effective summaries
   - Evolution: comprehension metrics

3. **Creative Writing**
   - Agents: plot, characters, dialogue
   - Memory: engaging story elements
   - Evolution: reader engagement scores

4. **Data Analysis**
   - Agents: cleaner, analyzer, visualizer
   - Memory: effective analysis approaches
   - Evolution: insight quality metrics

---

## 🔧 Configuration

### Environment Variables

```bash
# LLM Configuration
MODEL_NAME=claude-3-haiku-20240307  # or claude-3-5-sonnet-20241022
BASE_TEMPERATURE=0.7

# Memory Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
MEMORY_SCORE_THRESHOLD=7.0  # Only store high-quality outputs
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

### Customization Points

1. **Evaluation Metrics** (`src/hvas_mini/evaluation.py`)
   - Modify scoring functions
   - Adjust score thresholds
   - Add new quality factors

2. **Evolution Strategy** (`src/hvas_mini/evolution.py`)
   - Change temperature adjustment logic
   - Add new evolvable parameters
   - Implement different learning rates

3. **Agent Prompts** (`src/hvas_mini/agents.py`)
   - Customize prompts for each agent
   - Adjust memory example formatting
   - Add context from other sources

4. **Workflow** (`src/hvas_mini/pipeline.py`)
   - Modify agent execution order
   - Add parallel execution branches
   - Insert additional processing nodes

---

## 📁 Project Structure

```
hvas-mini/
├── main.py                      # Main entry point and demo
├── pyproject.toml               # uv configuration
├── .env.example                 # Environment template
├── README.md                    # This file
├── WORK_DIVISION.md            # Implementation plan
├── spec.md                      # Technical specification
│
├── src/hvas_mini/              # Main package
│   ├── __init__.py
│   ├── state.py                # State definitions (BlogState, AgentMemory)
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
├── docs/                        # Customization guides
│   ├── extending-agents.md
│   ├── custom-evaluation.md
│   └── langgraph-patterns.md
│
├── worktrees/                   # Feature branch worktrees (gitignored)
│   ├── project-foundation/
│   ├── state-management/
│   ├── memory-system/
│   ├── base-agent/
│   ├── specialized-agents/
│   ├── evaluation-system/
│   ├── visualization/
│   └── langgraph-orchestration/
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

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_agents.py -v

# Run with coverage
uv run pytest --cov=src/hvas_mini

# Run integration tests only
uv run pytest -m integration

# Type checking
uv run mypy src/hvas_mini
```

---

## 📈 Observing Learning

### Metrics to Watch

1. **Memory Growth**: Track how many memories each agent accumulates
2. **Score Trends**: Plot scores over generations for each agent
3. **Parameter Convergence**: Watch temperatures stabilize
4. **Retrieval Patterns**: See which memories get reused most

### Example Analysis

After running the demo, check:

```bash
# Memory statistics in ChromaDB
ls -lh data/memories/

# Agent parameters in output
# Look for temperature changes

# Score progression
# Compare Generation 1 vs Generation 5 scores
```

---

## 🤝 Contributing

This is a research prototype. Areas for exploration:

1. **Alternative Evolution Strategies**
   - Genetic algorithms
   - Reinforcement learning
   - Multi-objective optimization

2. **Memory Enhancements**
   - Memory consolidation
   - Forgetting mechanisms
   - Cross-agent memory sharing

3. **Evaluation Improvements**
   - LLM-based evaluation
   - Human feedback integration
   - Multi-criteria scoring

4. **Visualization**
   - Web dashboard
   - Interactive exploration
   - Learning curves

---

## 📚 Documentation

- **[spec.md](spec.md)**: Complete technical specification
- **[WORK_DIVISION.md](WORK_DIVISION.md)**: Implementation breakdown
- **[docs/](docs/)**: Customization guides (see below)

---

## 🔒 LangGraph Patterns Used

This prototype demonstrates several LangGraph patterns:

1. **StateGraph with TypedDict**: Type-safe state management
2. **Async Nodes**: Parallel execution capability
3. **Streaming with `astream()`**: Real-time updates
4. **Checkpointing with MemorySaver**: Execution history
5. **Custom Node Functions**: `_evolution_node` for post-processing

See `docs/langgraph-patterns.md` for detailed explanations.

---

## ⚠️ Limitations

- **Single-threaded**: Agents execute sequentially (not truly parallel)
- **Naive Evaluation**: Simple heuristic scoring (not LLM-based)
- **No Forgetting**: Memories accumulate indefinitely
- **Fixed Workflow**: Graph structure is static
- **Limited Context**: Each agent only sees its own memories

These are intentional simplifications for the prototype. See docs for how to address them.

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- **LangGraph** by LangChain for workflow orchestration
- **Anthropic** for Claude API
- **ChromaDB** for vector storage
- **Rich** by Will McGugan for beautiful terminal UI

---

## 📧 Contact

For questions about the research or implementation, please open an issue in the repository.

---

## 🎓 Citation

If you use this prototype in your research, please cite:

```bibtex
@software{hvas_mini_2024,
  title={HVAS Mini: Hierarchical Vector Agent System Prototype},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hvas-mini}
}
```

---

**Happy Researching! 🚀🧠**
