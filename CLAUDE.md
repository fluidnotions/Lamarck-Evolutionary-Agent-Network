# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Tools Configuration

This project uses two MCP servers for enhanced code analysis and knowledge management:
- **code-graph-mcp**: AST-based code structure analysis
- **project-memory**: Persistent knowledge graph storage

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

**Examples:**

User: "What calls the evolve function?"
Claude Action:
1. Call `analyze_codebase` (if not done in session)
2. Call `find_callers("evolve")`
3. Respond: "The `evolve()` function is called by:
   - `population_manager.py:145` in `run_generation()`
   - `evolution_loop.py:89` in `main_evolution_cycle()`
   - `test_evolution.py:67` in `test_evolve_population()`"

User: "Show me functions with complexity over 15"
Claude Action:
1. Call `complexity_analysis(threshold=15)`
2. Sort by complexity descending
3. Respond with formatted list including locations and scores

### Project Memory MCP - Knowledge Management Rules

**When to Use project-memory:**
Claude MUST use project-memory tools for these patterns:

1. **Storing Knowledge** (User explicitly asks to remember)
   - "remember [fact]" → Use `create_entities` tool
   - "store this" → Use `create_entities` + `add_observations`
   - "save to memory" → Use `create_entities`
   - After solving complex bugs → Prompt user: "Should I remember this solution?"
   - After architectural decisions → Prompt user: "Should I store this decision?"

2. **Retrieving Knowledge** (User asks about past context)
   - "recall [topic]" → Use `search_nodes` tool
   - "what do you know about [topic]" → Use `search_nodes` tool
   - "do you remember [topic]" → Use `search_nodes` tool
   - "why did we [decision]" → Use `search_nodes` tool
   - Before major refactoring → Automatically call `search_nodes` to check for relevant decisions

3. **Exploring Knowledge** (User wants to browse)
   - "show knowledge graph" → Use `read_graph` tool
   - "what entities exist" → Use `open_nodes` tool
   - "what's related to [topic]" → Use `read_graph` filtered by entity

4. **Updating Knowledge** (User provides new context)
   - "add observation about [entity]" → Use `add_observations` tool
   - "update [entity]" → Use `add_observations` tool
   - "link [A] and [B]" → Use `create_relations` tool

**What to Store in Memory:**

✅ DO Store:
- Architectural decisions and their rationale
- Performance optimization strategies and measurements
- Bug fixes with root cause analysis
- Complex algorithm explanations
- Integration patterns and gotchas
- Domain-specific knowledge (evolutionary algorithms, agent behavior, etc.)
- Testing strategies for specific features
- Design tradeoffs and why alternatives were rejected
- Non-obvious code relationships
- Historical context for "why it's done this way"

❌ DON'T Store:
- Trivial facts easily found in code
- Temporary implementation details
- Simple variable names or values
- Current session debugging notes (use `/note` command instead)
- Code snippets (store in files, not memory)
- Information that will quickly become outdated

**Entity Types to Use:**
- `architecture` - Design decisions, system structure
- `solution` - Bug fixes, problem solutions
- `pattern` - Reusable design patterns, best practices
- `domain_knowledge` - Project-specific concepts (evolution, agents, fitness, etc.)
- `optimization` - Performance improvements
- `integration` - How components work together
- `constraint` - Known limitations or requirements

**Memory Storage Format:**

When storing with `create_entities`:
```json
{
  "name": "Clear, concise title (under 50 chars)",
  "entityType": "architecture|solution|pattern|domain_knowledge|optimization|integration|constraint",
  "observations": [
    "Primary observation: [the main fact/decision]",
    "Rationale: [why this decision was made]",
    "Context: [when/where this applies]",
    "Impact: [what this affects]",
    "Stored: [date]"
  ]
}
```

When creating relationships with `create_relations`:
```json
{
  "from": "entity_id_1",
  "to": "entity_id_2", 
  "relationType": "depends_on|implements|solves|relates_to|conflicts_with"
}
```

**Examples:**

User: "Remember: we use async/await for all agent communication"
Claude Action:
1. Call `create_entities`:
```json
   {
     "name": "Async Agent Communication Pattern",
     "entityType": "architecture",
     "observations": [
       "All agent-to-agent communication uses async/await",
       "Rationale: Prevents blocking during long-running fitness evaluations",
       "Impact: Allows concurrent evolution of multiple populations",
       "Critical for scaling beyond 100 agents",
       "See: agent_network/communication.py"
     ]
   }
```
2. Call `create_relations` to link to `agent_network` module
3. Respond: "✅ Stored architectural decision: Async Agent Communication Pattern"

User: "What do you know about fitness evaluation?"
Claude Action:
1. Call `search_nodes("fitness evaluation")`
2. Retrieve top 3-5 relevant memories
3. Present formatted results with:
   - Entity names and types
   - Stored observations
   - Related entities/code locations
   - Relevance scores

User: "Why did we implement copy-on-write for fitness?"
Claude Action:
1. Call `search_nodes("copy-on-write fitness")`
2. If found, present the stored rationale
3. If not found: "No memory found. Should I search the code with code-graph-mcp?"

### Combined Workflow - Using Both Tools Together

For complex development tasks, use both MCP tools strategically:

**Before Refactoring:**
1. `code-graph-mcp`: Analyze current structure
```
   "What calls the fitness_evaluation function?" 
   "Show complexity analysis for fitness module"
```
2. `project-memory`: Recall design decisions
```
   "What do you know about fitness evaluation design?"
```
3. Make informed decision with full context

**After Solving a Complex Bug:**
1. Document the solution in memory:
```
   /remember "Race condition in fitness calculation: Multiple agents were 
   accessing shared cache concurrently. Fixed with copy-on-write semantics 
   in fitness.py:145-167. Root cause: Missing synchronization in CacheManager."
```
2. Use code-graph to verify the fix doesn't break callers:
```
   "What functions call CacheManager methods?"
```

**Starting Work on New Feature:**
1. Check for existing knowledge:
```
   "What do you know about [feature area]?"
```
2. Analyze related code:
```
   "Show me the structure of the [module] module"
   "Find functions related to [feature]"
```
3. Plan implementation with full context

### Proactive Tool Usage

Claude should proactively suggest using these tools:

**Suggest code-graph when:**
- User mentions refactoring without analyzing impact
- User asks about code structure in vague terms
- Discussing changes to functions with many callers
- Need to understand cascade effects of changes

**Suggest memory when:**
- User makes an important architectural decision (offer to store it)
- User solves a non-trivial bug (offer to document solution)
- User asks "why" questions that might have historical context
- Before major refactoring (check for documented constraints)

**Example proactive responses:**

User: "I'm going to refactor the evolve function"
Claude: "Let me first analyze its impact with code-graph:
1. What calls evolve() [using find_callers]
2. Complexity analysis [using complexity_analysis]
3. Check for documented design decisions [using memory search_nodes]

Then we can plan the refactoring safely."

User: "I fixed the race condition by adding locks"
Claude: "Great! That was a tricky bug. Would you like me to store this solution 
in project memory so we remember:
- What the bug was
- Root cause
- How it was fixed
- Where to look if similar issues appear?"

### Error Handling

**If code-graph tools fail:**
1. Check if `analyze_codebase` has been run
2. If not, run it automatically: "Building code graph first..."
3. If it fails, fall back to manual code reading and explain: "Code graph unavailable, analyzing files directly"

**If memory tools fail:**
1. Check database connection (MEMORY_DB_PATH)
2. Provide helpful error message
3. Offer alternative: "Can't access memory. Should I use /note to document this instead?"

**If tools return no results:**
1. Suggest alternative search terms
2. Offer to search with broader criteria
3. For code-graph: "Function not found. Did you mean: [suggestions]?"
4. For memory: "No memories found. Should I search the code instead?"

### Performance Considerations

- Code graph analysis may take 10-30 seconds for large projects
- Memory searches are fast (<1 second typically)
- Cache code graph results within a session
- Rebuild code graph after major changes or at session start
- Don't rebuild unnecessarily - check if recent analysis exists

### Integration with Existing Commands

These MCP tools enhance existing commands:

- `/work` → Use code-graph to understand dependencies before implementing
- `/issue` → Use memory to recall related architectural decisions
- `/explore` → Use both tools to gather context
- `/note` → For session notes; use memory for permanent knowledge
- `/kanban` → Use code-graph to analyze impact of completed work

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
