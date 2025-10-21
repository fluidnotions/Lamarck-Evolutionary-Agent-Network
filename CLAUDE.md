# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LEAN (Lamarck Evolutionary Agent Network)** is a research prototype testing **Lamarckian evolution** for AI agents through:
- **ReasoningMemory** for storing cognitive patterns (how to think)
- **SharedRAG** for domain knowledge (what to know)
- **Evolutionary agent pools** with selection, compaction, and reproduction
- **Pattern inheritance** from successful parent agents
- **YAML-based experiment configuration**

**Core Research Question**: Can agents improve by inheriting their parents' reasoning patterns rather than through prompt engineering?

## V2 Architecture (Current)

The V2 system uses a **3-layer separation** of concerns:

### Layer 1: Fixed Interface (Prompts)
- Frozen system prompts define agent roles
- Never mutate during evolution
- Provides stable behavioral API
- Located in `config/prompts/agents.yml`

### Layer 2: Shared Knowledge (SharedRAG)
- Domain facts and reference materials
- Available to all agents equally
- Standard semantic retrieval
- Separate from reasoning patterns

### Layer 3: Evolving Reasoning (ReasoningMemory)
- Planning sequences and problem-solving steps
- Cognitive strategies that worked
- **Inherited from parents** during evolution
- **Personal patterns** from own experience
- Retrieved by structural similarity

## Key Components

### V2 Agents (`src/lean/base_agent_v2.py`)

```python
class BaseAgentV2:
    """Agent with reasoning pattern externalization."""

    def __init__(self, role: str, reasoning_memory, shared_rag, ...):
        self.reasoning_memory = ReasoningMemory()  # Cognitive patterns
        self.shared_rag = SharedRAG()              # Domain knowledge
        self.system_prompt = ...                    # Frozen prompt
```

**Agent Types**:
- `IntroAgentV2` - Introduction sections
- `BodyAgentV2` - Main content sections
- `ConclusionAgentV2` - Conclusion sections

### M2 Evolution System

**Agent Pools** (`src/lean/agent_pool.py`):
- Maintains population of agents per role
- Tracks fitness scores
- Triggers evolution at specified frequency
- Manages agent lifecycle

**Selection Strategies** (`src/lean/selection.py`):
- `tournament_selection` - Best of random subset
- `rank_based_selection` - Probability proportional to rank
- `fitness_proportionate_selection` - Roulette wheel

**Compaction Strategies** (`src/lean/compaction.py`):
- `score_based_compaction` - Keep highest-scoring patterns
- `diversity_based_compaction` - Probabilistic for variety
- `hybrid_compaction` - Combine both approaches

**Reproduction Strategies** (`src/lean/reproduction.py`):
- `sexual_reproduction` - Two parents, merge patterns
- `asexual_reproduction` - Clone with mutation

### Memory Systems

**ReasoningMemory** (`src/lean/reasoning_memory.py`):
- Stores `<think>...</think>` reasoning traces
- Separate inherited vs personal patterns
- Compaction during reproduction
- CPU-based embeddings (avoid GPU OOM)

**SharedRAG** (`src/lean/shared_rag.py`):
- Stores domain knowledge from all agents
- Metadata: source (role, generation, score)
- Query by semantic similarity
- Statistics tracking

**ContextManager** (`src/lean/context_manager.py`):
- Distributes context with 40/30/20/10 weighting:
  - 40%: Current topic
  - 30%: Retrieved reasoning patterns
  - 20%: Domain knowledge from SharedRAG
  - 10%: Reserved for future use

### Pipeline V2 (`src/lean/pipeline_v2.py`)

8-step learning cycle per generation:

1. **Select** active agents from pools
2. **Retrieve** reasoning patterns + domain knowledge
3. **Generate** content with structured prompts
4. **Extract** reasoning from `<think>` tags
5. **Evaluate** content quality (0-10 scores)
6. **Store** successful patterns + knowledge
7. **Update** agent fitness scores
8. **Evolve** (if generation % frequency == 0):
   - Select parents by fitness
   - Compact reasoning patterns
   - Reproduce offspring
   - Replace population

## Module Responsibilities

### Core V2 Modules

- **`base_agent_v2.py`**: V2 agent architecture with reasoning externalization
- **`pipeline_v2.py`**: V2 pipeline orchestrator with M2 evolution
- **`reasoning_memory.py`**: Cognitive pattern storage (Layer 3)
- **`shared_rag.py`**: Shared knowledge base (Layer 2)
- **`context_manager.py`**: Context distribution with weighting
- **`config_loader.py`**: YAML configuration loader

### M2 Evolution Modules

- **`agent_pool.py`**: Agent pool management
- **`selection.py`**: Selection strategies
- **`compaction.py`**: Memory compaction strategies
- **`reproduction.py`**: Reproduction strategies

### Shared Core Modules

- **`state.py`**: `BlogState` TypedDict, state creation
- **`evaluation.py`**: `ContentEvaluator` for scoring (0-10)
- **`visualization.py`**: `StreamVisualizer` for Rich terminal UI
- **`web_search.py`**: Tavily research integration (optional)
- **`human_in_the_loop.py`**: HITL feedback interface (optional)

### Configuration System

- **`config/experiments/`**: Experiment configurations (topics, evolution params)
- **`config/prompts/`**: Agent system prompts and evaluation criteria
- **`config/docs/`**: Markdown documentation for topic blocks and roles

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run V2 pipeline with default config
uv run python main_v2.py

# Run with custom experiment config
uv run python main_v2.py --config healthcare_study

# Run tests
uv run pytest                                     # All tests
uv run pytest tests/test_pipeline_v2.py -v       # V2 pipeline
uv run pytest tests/test_agent_pool.py -v        # M2 evolution
uv run pytest --cov=src/lean                     # With coverage

# Type checking
uv run mypy src/lean
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add `ANTHROPIC_API_KEY`
3. Optional: Add `TAVILY_API_KEY` for research integration
4. Key configuration (in YAML or .env):
   - `MEMORY_SCORE_THRESHOLD=7.0` - Store patterns scoring ≥7.0
   - `BASE_TEMPERATURE=0.7` - Starting temperature
   - `EVOLUTION_LEARNING_RATE=0.1` - Evolution aggressiveness

## YAML Configuration

### Experiment Config (`config/experiments/default.yml`)

```yaml
experiment:
  name: "Default AI Topics Experiment"
  population_size: 3
  evolution_frequency: 5
  total_generations: 20

topic_blocks:
  - name: "AI Fundamentals"
    generation_range: [1, 4]
    topics:
      - title: "The Future of Artificial Intelligence"
        keywords: ["AI", "future", "innovation"]

research:
  enabled: true
  provider: "tavily"
  max_results: 5
```

### Agent Prompts (`config/prompts/agents.yml`)

```yaml
coordinator:
  system_prompt: |
    You are a Coordinator Agent...
  documentation: "config/docs/coordinator-role.md"

intro:
  system_prompt: |
    You are an Introduction Agent...
  reasoning_focus: "engagement, clarity, preview"
```

## Key Design Patterns

### 1. Reasoning Pattern Externalization

Agents output structured responses:

```
<think>
I should start with a hook about recent AI breakthroughs.
Then establish why this matters to readers.
Finally preview the three main points I'll cover.
</think>

<final>
[Actual introduction text...]
</final>
```

The `<think>` content is:
- Extracted and stored in ReasoningMemory
- Scored based on final output quality
- Inherited by offspring if score ≥ threshold
- Retrieved for similar future tasks

### 2. Agent Pool Pattern

```python
class AgentPool:
    def __init__(self, role: str, population_size: int):
        self.population: List[BaseAgentV2] = []
        self.fitness_scores: Dict[str, float] = {}

    def select_active_agent(self) -> BaseAgentV2:
        """Select best agent for current generation."""
        return max(self.population, key=lambda a: self.fitness_scores[a.id])

    def evolve(self) -> List[BaseAgentV2]:
        """Trigger evolution: select, compact, reproduce."""
        parents = self.select_parents()
        offspring = self.reproduce(parents)
        self.population = offspring
        return offspring
```

### 3. Compaction Pattern

Before inheritance, reasoning patterns are compacted:

```python
def compact_reasoning(
    patterns: List[Dict],
    strategy: str,
    target_size: int
) -> List[Dict]:
    """Reduce patterns to target size."""
    if strategy == "score_based":
        # Keep highest-scoring patterns
        return sorted(patterns, key=lambda p: p['score'])[-target_size:]
    elif strategy == "diversity_based":
        # Probabilistic selection for variety
        return random.sample(patterns, target_size)
```

### 4. Sexual Reproduction Pattern

```python
def sexual_reproduction(parent1, parent2):
    """Merge reasoning patterns from two parents."""
    # Compact each parent's patterns
    p1_compact = compact(parent1.reasoning_memory.patterns)
    p2_compact = compact(parent2.reasoning_memory.patterns)

    # Offspring inherits from both
    offspring_patterns = p1_compact + p2_compact

    # Create new agent
    offspring = BaseAgentV2(
        role=parent1.role,
        inherited_reasoning=offspring_patterns
    )
    return offspring
```

## Common Development Patterns

### Adding a New Experiment

1. Create YAML config: `config/experiments/my_experiment.yml`
2. Define topic blocks with generation ranges
3. Run: `python main_v2.py --config my_experiment`

### Customizing Agent Prompts

1. Edit `config/prompts/agents.yml`
2. Modify `system_prompt` for any agent
3. No code changes needed

### Adding a New Selection Strategy

1. Add function to `src/lean/selection.py`:
   ```python
   def my_selection(agents, fitness_scores, num_parents):
       # Your selection logic
       return selected_parents
   ```

2. Use in config or pipeline:
   ```python
   parents = my_selection(pool.population, pool.fitness_scores, 2)
   ```

### Enabling Tavily Research

1. Get API key from https://tavily.com
2. Add to `.env`: `TAVILY_API_KEY=your_key`
3. Enable in config:
   ```yaml
   research:
     enabled: true
     max_results: 5
     search_depth: "advanced"
   ```

## Testing Strategy

Tests organized by component:
- `test_agent_factory_v2.py` - V2 agent creation
- `test_pipeline_v2.py` - V2 pipeline
- `test_agent_pool.py` - Agent pool management
- `test_selection.py` - Selection strategies
- `test_compaction.py` - Compaction strategies
- `test_reproduction.py` - Reproduction strategies
- `test_reasoning_integration.py` - Reasoning pattern flow
- `test_evolution_integration.py` - Full evolution cycle
- `test_state.py` - State validation
- `test_web_search.py` - Tavily integration

## Observing Learning

After running `main_v2.py`:

1. **Reasoning patterns**: `ls data/reasoning/` - Stored cognitive patterns
2. **Domain knowledge**: `ls data/shared_rag/` - Shared knowledge base
3. **Score trends**: Compare Generation 1 vs Generation 20 scores
4. **Evolution events**: Watch for "🧬 EVOLUTION" markers in output
5. **Pattern inheritance**: Offspring should inherit parent patterns

Expected: Gradual improvement in scores as patterns accumulate and best agents reproduce.

## Important Notes

- **V2 Architecture**: Current implementation uses flat evolutionary pools, not hierarchical structure
- **3-Layer Separation**: Prompts (frozen) | Knowledge (shared) | Reasoning (evolved)
- **Pattern Inheritance**: Offspring inherit compacted reasoning from parents
- **YAML Configuration**: Experiments and prompts defined declaratively
- **CPU Embeddings**: Shared embedder on CPU to avoid GPU OOM
- **No Hierarchy**: Removed M6-M9 hierarchical code (coordinator, specialists, semantic distance)
- **No Trust Weighting**: Removed M2 trust-based weighting system
- **Focus on Evolution**: Current focus is M2 agent pools with pattern inheritance

## Research Context

This prototype tests hypotheses:
1. **Pattern Inheritance**: Do agents improve by inheriting reasoning?
2. **Lamarckian Evolution**: Does acquired knowledge transfer to offspring?
3. **Separation of Concerns**: Is frozen prompts + evolved reasoning better than prompt engineering?
4. **Transfer Learning**: Do reasoning patterns help on similar tasks?
5. **Population Dynamics**: Which selection/compaction strategies work best?

The demo (20 topics, 4 blocks) is designed to test transfer learning across thematic groups.

## Documentation

- **`README.md`**: Research motivation, architecture overview
- **`docs/yaml-configuration-guide.md`**: Complete YAML config reference
- **`config/README.md`**: Quick start for configuration
- **`HITL_README.md`**: Human-in-the-loop feature guide
- **`docs/*.md`**: Analysis documents and guides

---

**This is research code focused on V2 reasoning pattern evolution. V1 hierarchical code has been removed.**
