# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Single Version Policy

**This codebase maintains only the CURRENT version**. There are no V2, V3, or other versioned files. When the architecture evolves:

- Old files are REMOVED (they remain in git history)
- New implementations REPLACE old ones with base names (no version suffixes)
- Documentation reflects ONLY the current state
- Tests are for the CURRENT implementation only

**Examples:**
- ❌ `pipeline_v2.py`, `pipeline_v3.py`
- ✅ `pipeline.py` (current implementation)
- ❌ `BaseAgentV2`, `BaseAgentV3`
- ✅ `BaseAgent` (current implementation)

## Documentation Location Policy

All markdown documentation must be stored in the `analysis/` directory, NOT in `docs/` or any other location:

- **✅ CORRECT**: `analysis/architecture-guide.md`
- **✅ CORRECT**: `analysis/implementation-notes.md`
- **❌ WRONG**: `docs/some-guide.md`
- **❌ WRONG**: `./some-guide.md`

**Exception**: Role documentation referenced by YAML configs belongs in `config/docs/` (e.g., `config/docs/intro-role.md`).

## MCP Tools Configuration

This project uses **code-graph-mcp** for AST-based code structure analysis.

### Code Graph MCP - Automatic Usage Rules

**When to Use code-graph-mcp:**
Claude MUST use code-graph-mcp tools for these query patterns:

1. **Call Graph Queries**
   - "what calls [function]" → Use `find_callers` tool
   - "what does [function] call" → Use `find_callees` tool
   - "show call graph for [feature]" → Use `analyze_codebase` + `find_callers` + `find_callees`
   - "trace execution from [function]" → Use `find_callees` recursively

2. **Code Structure Queries**
   - "find [symbol/function/class]" → Use `find_definition` tool
   - "where is [symbol] used" → Use `find_references` tool
   - "show me all [type] in codebase" → Use `search_symbol` tool

3. **Complexity Analysis**
   - "complex functions" → Use `complexity_analysis` tool (threshold: 15)
   - "functions with complexity > [N]" → Use `complexity_analysis` with specified threshold
   - "code that needs refactoring" → Use `complexity_analysis` (threshold: 20)
   - "hotspots" → Use `complexity_analysis` + `find_callers` for impact

4. **Dependency Analysis**
   - "dependencies" → Use `dependency_analysis` tool
   - "circular dependencies" → Use `dependency_analysis` and filter for cycles
   - "what imports [module]" → Use `find_references` at module level
   - "module dependencies" → Use `dependency_analysis`

5. **Project Overview**
   - "project structure" → Use `analyze_codebase` + `project_statistics`
   - "codebase statistics" → Use `project_statistics` tool
   - "code metrics" → Use `project_statistics` + `complexity_analysis`

**Tool Execution Workflow:**
1. First use in session: Always call `analyze_codebase` to build/refresh the code graph
2. For function-specific queries: Call specific tool (`find_callers`, `find_definition`, etc.)
3. For complex analysis: Chain multiple tools (e.g., `find_callers` → `complexity_analysis`)
4. Include file paths and line numbers in all responses

**Response Format for Code Graph Queries:**
Always structure responses with:
- **Location**: `file.py:line_number`
- **Context**: Surrounding code context if relevant
- **Relationships**: What calls it / what it calls
- **Metrics**: Complexity score if applicable
- **Impact**: Number of callers/callees for change impact assessment

## Project Overview

**LEAN (Lamarck Evolutionary Agent Network)** is a research prototype testing **Lamarckian evolution** for AI agents through:
- **ReasoningMemory** for storing cognitive patterns (how to think)
- **SharedRAG** for domain knowledge (what to know)
- **Evolutionary agent pools** with selection, compaction, and reproduction
- **Pattern inheritance** from successful parent agents
- **YAML-based experiment configuration**
- **Hierarchical 3-layer architecture** with coordinator, content agents, and specialists

**Core Research Question**: Can agents improve by inheriting their parents' reasoning patterns rather than through prompt engineering?

## Current Architecture

The system uses a **3-layer hierarchical architecture**:

### Layer 1: Coordinator
- **CoordinatorAgent** orchestrates the entire workflow
- Researches topics using Tavily API
- Synthesizes research into context for content agents
- Aggregates outputs and critiques quality
- Requests revisions if quality is below threshold

### Layer 2: Content Agents
- **IntroAgent** - Writes introductions
- **BodyAgent** - Writes main content
- **ConclusionAgent** - Writes conclusions
- Organized in evolutionary **AgentPools** (population-based evolution)
- Each agent has ReasoningMemory (cognitive patterns) and SharedRAG access

### Layer 3: Specialist Agents
- **ResearcherAgent** - Deep research and evidence validation
- **FactCheckerAgent** - Claim verification and accuracy checking
- **StylistAgent** - Clarity and readability enhancement
- Invoked by content agents when needed

### Workflow

```
START
  ↓
[1. RESEARCH]        ← Coordinator researches topic via Tavily
  ↓
[2. DISTRIBUTE]      ← Coordinator synthesizes & distributes context
  ↓
[3. INTRO]           ← IntroAgent generates with coordinator context
  ↓
[4. BODY]            ← BodyAgent generates (with optional specialist support)
  ↓
[5. CONCLUSION]      ← ConclusionAgent generates with full context
  ↓
[6. AGGREGATE]       ← Coordinator aggregates outputs
  ↓
[7. CRITIQUE]        ← Coordinator critiques quality
  ↓
[REVISION LOOP?]     ← If quality < threshold & revisions < max
  │                     YES: feedback → back to [3. INTRO]
  ↓ NO
[8. EVALUATE]        ← ContentEvaluator scores sections
  ↓
[9. EVOLVE]          ← Store patterns, trigger pool evolution
  ↓
END
```

## Key Components

### Core Agents

- **`base_agent.py`**: Base agent architecture with reasoning externalization
  - `BaseAgent` abstract class
  - `IntroAgent`, `BodyAgent`, `ConclusionAgent` implementations
  - `create_agents()` factory function
  - All agents use `<think>` tags for reasoning externalization
  - All agents have ReasoningMemory and SharedRAG access

- **`coordinator.py`**: Coordinator agent with research and critique
  - `CoordinatorAgent` extends `BaseAgent`
  - Tavily research integration
  - Context synthesis and distribution
  - Quality critique and revision requests

- **`specialists.py`**: Specialist support agents
  - `ResearcherAgent` - Evidence validation
  - `FactCheckerAgent` - Accuracy verification
  - `StylistAgent` - Style enhancement
  - All extend `BaseAgent`

### Pipeline & Orchestration

- **`pipeline.py`**: Hierarchical pipeline orchestrator
  - `Pipeline` class manages workflow
  - LangGraph-based execution flow
  - Configurable features (research, specialists, revision)
  - Evolution integration

- **`agent_pool.py`**: Population-based evolution
  - `AgentPool` class manages agent populations
  - Fitness tracking and selection
  - Evolution triggers (every N generations)

### Memory Systems

- **`reasoning_memory.py`**: Cognitive pattern storage (Layer 3)
  - Stores `<think>...</think>` reasoning traces
  - Separate inherited vs personal patterns
  - Compaction during reproduction
  - CPU-based embeddings (avoid GPU OOM)

- **`shared_rag.py`**: Shared knowledge base (Layer 2)
  - Stores domain knowledge from all agents
  - Metadata: source (role, generation, score)
  - Query by semantic similarity
  - Statistics tracking

- **`context_manager.py`**: Context distribution with weighting
  - 40%: Current topic/hierarchy context
  - 30%: High credibility reasoning patterns
  - 20%: Diverse reasoning patterns
  - 10%: Peer reasoning patterns

### Evolution Components

- **`selection.py`**: Selection strategies
  - Tournament selection
  - Rank-based selection
  - Fitness-proportionate selection

- **`compaction.py`**: Memory compaction strategies
  - Score-based compaction (keep best patterns)
  - Diversity-based compaction (maintain variety)
  - Hybrid compaction

- **`reproduction.py`**: Reproduction strategies
  - Sexual reproduction (two parents)
  - Asexual reproduction (clone with mutation)

### Supporting Modules

- **`state.py`**: `BlogState` TypedDict, state creation
- **`evaluation.py`**: `ContentEvaluator` for scoring (0-10)
- **`visualization.py`**: `StreamVisualizer` for Rich terminal UI
- **`config_loader.py`**: YAML configuration loader

### Configuration System

- **`config/experiments/`**: Experiment configurations (topics, evolution params)
- **`config/prompts/`**: Agent system prompts and evaluation criteria
- **`config/docs/`**: Markdown documentation for topic blocks and roles

## Build & Run Commands

```bash
# Install dependencies
uv sync

# Run pipeline with default config
uv run python main.py

# Run with custom experiment config
uv run python main.py --config healthcare_study

# Run tests
uv run pytest                           # All tests
uv run pytest tests/test_pipeline.py -v  # Pipeline tests
uv run pytest tests/test_agent_pool.py -v  # Evolution tests
uv run pytest --cov=src/lean            # With coverage

# Type checking
uv run mypy src/lean
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add `ANTHROPIC_API_KEY`
3. Optional: Add `TAVILY_API_KEY` for research integration
4. Key configuration:
   - `ENABLE_SPECIALISTS=true` - Enable Layer 3 specialist agents
   - `ENABLE_REVISION=true` - Enable revision loop
   - `MAX_REVISIONS=2` - Maximum revision iterations
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
  documentation: "config/docs/intro-role.md"
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
        self.population: List[BaseAgent] = []
        self.fitness_scores: Dict[str, float] = {}

    def select_active_agent(self) -> BaseAgent:
        """Select best agent for current generation."""
        return fitness_proportionate_selection(self.population)

    def evolve(self) -> List[BaseAgent]:
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
    offspring = BaseAgent(
        role=parent1.role,
        inherited_reasoning=offspring_patterns
    )
    return offspring
```

## Common Development Patterns

### Adding a New Experiment

1. Create YAML config: `config/experiments/my_experiment.yml`
2. Define topic blocks with generation ranges
3. Run: `python main.py --config my_experiment`

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
3. Research is automatically enabled if key present

## Testing Strategy

Tests organized by component:
- `test_pipeline.py` - Pipeline orchestration
- `test_agent_pool.py` - Agent pool management
- `test_coordinator.py` - Coordinator agent
- `test_specialists.py` - Specialist agents
- `test_selection.py` - Selection strategies
- `test_compaction.py` - Compaction strategies
- `test_reproduction.py` - Reproduction strategies
- `test_reasoning_integration.py` - Reasoning pattern flow
- `test_evolution_integration.py` - Full evolution cycle
- `test_state.py` - State validation

## Observing Learning

After running `main.py`:

1. **Reasoning patterns**: `ls data/reasoning/` - Stored cognitive patterns
2. **Domain knowledge**: `ls data/shared_rag/` - Shared knowledge base
3. **Score trends**: Compare Generation 1 vs Generation 20 scores
4. **Evolution events**: Watch for "🧬 EVOLUTION" markers in output
5. **Pattern inheritance**: Offspring should inherit parent patterns

Expected: Gradual improvement in scores as patterns accumulate and best agents reproduce.

## Important Notes

- **Single Version**: No V2/V3 files - only current implementation
- **3-Layer Architecture**: Coordinator → Content Agents → Specialists
- **Pattern Inheritance**: Offspring inherit compacted reasoning from parents
- **YAML Configuration**: Experiments and prompts defined declaratively
- **CPU Embeddings**: Shared embedder on CPU to avoid GPU OOM
- **Research Integration**: Tavily API for real-time information gathering
- **Revision Loop**: Quality assurance through coordinator critique

## Research Context

This prototype tests hypotheses:
1. **Pattern Inheritance**: Do agents improve by inheriting reasoning?
2. **Lamarckian Evolution**: Does acquired knowledge transfer to offspring?
3. **Separation of Concerns**: Is frozen prompts + evolved reasoning better than prompt engineering?
4. **Transfer Learning**: Do reasoning patterns help on similar tasks?
5. **Population Dynamics**: Which selection/compaction strategies work best?
6. **Hierarchical Coordination**: Does coordinator-driven workflow improve quality?

The demo (20 topics, 4 blocks) is designed to test transfer learning across thematic groups.

## Analysis Documents

Comprehensive guides are in `analysis/`:
- `analysis/architecture-implementation-gap.md` - Original gap analysis
- `analysis/v3-hierarchical-implementation-guide.md` - Implementation guide
- `analysis/v3-implementation-summary.md` - Feature summary
- `analysis/EXPERIMENTS.md` - Experiment tracking

---

**This is research code focused on reasoning pattern evolution with hierarchical coordination.**
