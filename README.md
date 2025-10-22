# Lamarck Evolutionary Agent Network

**Evolutionary Agent Learning Research**

## The Core Question

**Can AI agents improve by inheriting their parents' reasoning patterns?**

This is a research prototype testing **Lamarckian evolution** for AI agent cognition:

- **Prompts stay frozen** (consistent behavioral interface)
- **Domain knowledge lives in shared RAG** (facts available to all agents)
- **Reasoning patterns are inherited** (successful cognitive strategies pass to offspring)
- **Selection determines what propagates** (natural selection on thinking patterns, not content)

Think of it as: **Acquired reasoning strategies become heritable DNA.**

The goal is to test whether cognitive pattern accumulation across generations produces agents that demonstrably improve at problem-solving—without the fragility of prompt engineering.

---

## Why This Matters

### The Problem: We're Evolving the Wrong Layer

Current AI agent systems break with every model update. They rely on carefully crafted text prompts that need constant rewriting. We're trying to encode *how to think* in natural language instructions, when we should be letting reasoning patterns emerge from experience.

**The insight:** Models don't think in text—they think in embeddings. Vector databases excel at finding similar reasoning structures. "Find planning sequences like mine" is exactly what embedding search was built for.

What if agents evolved their cognitive strategies through inheritance rather than prompt engineering?

### The Three-Layer Architecture

This system cleanly separates three concerns that current approaches conflate:

**1. Fixed Interface Layer (Prompts)**
```yaml
# config/prompts/agents.yml
intro:
  system_prompt: |
    You are an Introduction Agent in the LEAN evolutionary system.
    Your role is to craft compelling introductions that hook readers.
```
- Never changes, never mutates
- Provides stable API for the system
- Pure role definition
- Loaded from YAML configuration

**2. Shared Knowledge Layer (RAG)**
- Domain facts and content
- Reference materials
- Historical outputs
- Available to all agents equally
- Standard semantic retrieval

**3. Evolving Reasoning Layer (What Gets Inherited)**
- Planning sequences ("First establish context, then add statistics, finally pose question")
- Problem-solving steps
- Reasoning traces
- Cognitive strategies that worked
- Retrieved by structural similarity

**Result:** Systems that improve their reasoning through use, not through prompt engineering.

---
## Status — Experimental Debug Cycle
  The core architecture is complete and the system is now in its empirical tuning phase. Early tests are focused on verifying cognitive inheritance, optimizing agent evolution dynamics, and resolving the usual first-generation quirks of a new architecture.
  
  Expect rapid iteration and occasional chaos — this is research in motion.
  
---
## The Hierarchical Architecture

### Three-Layer Ensemble System

**Layer 1: Coordinator**
- **CoordinatorAgent** orchestrates the entire workflow
- Researches topics using Tavily API (optional)
- Synthesizes research into context for content agents
- Aggregates outputs and critiques quality
- Requests revisions if quality is below threshold

**Layer 2: Content Agents**
- **IntroAgent** - Writes introductions
- **BodyAgent** - Writes main content
- **ConclusionAgent** - Writes conclusions
- Organized in evolutionary **AgentPools** (population-based evolution)
- Each agent has ReasoningMemory (cognitive patterns) and SharedRAG access

**Layer 3: Specialist Agents**
- **ResearcherAgent** - Deep research and evidence validation
- **FactCheckerAgent** - Claim verification and accuracy checking
- **StylistAgent** - Clarity and readability enhancement
- Invoked by content agents when needed

### How Reasoning Inheritance Works

Each agent maintains two types of cognitive patterns:

**1. Inherited Reasoning** (from parents)
- Compacted/distilled planning strategies from both parents
- High-performing cognitive patterns that led to success
- Passed down through reproduction

**2. Personal Reasoning** (from experience)
- Own discovered planning sequences
- Reasoning traces with scores
- Added to inheritance pool when reproducing

**Memory structure using `<think>/<final>` tags:**
```python
# Agent generates structured output
response = """
<think>
I should start with a hook about recent AI breakthroughs.
Then establish why this matters to readers.
Finally preview the three main points I'll cover.
</think>

<final>
Did you know that artificial intelligence is already powering...
</final>
"""

# <think> content is extracted and stored
reasoning_pattern = {
    "thinking": "I should start with a hook...",
    "score": 8.5,
    "generation": 5,
    "role": "intro"
}

# Offspring inherit compacted patterns from parents
```

### What Gets Stored: Cognitive Strategies, Not Content

**Critical distinction (storage has two paths):**
- **Individual memory** (per-agent): ALL reasoning patterns stored (no threshold) → periodically compacted → best patterns inherited by offspring
- **Shared RAG** (global): ONLY high-quality content (score ≥8.0) stored → available to all agents as domain knowledge

This separation enables Lamarckian evolution: agents inherit their parents' **best** cognitive strategies (after forgetting unsuccessful ones) while sharing domain facts.

### Evolutionary Operators

**Selection** (choose parents):
- Tournament selection (currently used)
- Rank-based selection
- Fitness-proportionate selection
- Based on how well their reasoning patterns perform

**Reasoning Pattern Compaction** (create offspring):
```python
def sexual_reproduction(parent1, parent2):
    # Merge both parents' reasoning patterns
    p1_patterns = compact(parent1.reasoning_memory.patterns)
    p2_patterns = compact(parent2.reasoning_memory.patterns)

    # Child gets compacted cognitive strategies
    child = Agent(
        role=parent1.role,
        system_prompt=parent1.system_prompt,  # Fixed (from YAML)
        inherited_reasoning=p1_patterns + p2_patterns,  # Evolved
        shared_rag=global_knowledge_base  # Shared
    )
    return child
```

**Population Dynamics:**
- Configurable agents per role (default: 3)
- Evolution frequency configurable (default: every 5 generations)
- AgentPools manage populations per role

### The Learning Cycle (Hierarchical)

Each generation follows this workflow:

1. **[COORDINATOR] Research** - Coordinator researches topic via Tavily (if enabled)
2. **[COORDINATOR] Distribute** - Coordinator synthesizes context for content agents
3. **[ENSEMBLE] Generate** - Content agents generate sections:
   - **Retrieve reasoning patterns**: Query inherited + personal patterns
   - **Retrieve knowledge**: Get facts from shared RAG
   - **Invoke specialists**: Call researcher/fact-checker/stylist if needed
   - **Generate with `<think>/<final>` tags**: Externalize reasoning
4. **[COORDINATOR] Aggregate** - Coordinator combines all outputs
5. **[COORDINATOR] Critique** - Coordinator scores quality
6. **[REVISION LOOP]** - If quality < threshold, request revisions (configurable)
7. **[EVALUATION]** - ContentEvaluator scores each section (0-10)
8. **[STORAGE]** - Two paths:
   - Reasoning patterns (`<think>` content) → individual memory
   - Output content → shared RAG (if score ≥ 8.0)
9. **[EVOLUTION]** - Every N generations:
   - **Compaction**: Forget unsuccessful patterns
   - **Selection**: Best reasoners chosen as parents (tournament/rank/fitness-weighted)
   - **Reproduction**: Offspring inherit compacted reasoning
   - **Pool replacement**: New generation replaces old

**Key insight:** The ensemble coordinator orchestrates multiple content agents, each with their own reasoning patterns, creating a collaborative evolutionary system.

---

## Memory Compaction Strategies

Three strategies for cognitive inheritance:

#### Strategy A: Score-Weighted Selection
```python
def compact_reasoning(combined, max_size=100):
    return sorted(combined, key=lambda m: m.score)[-max_size:]
```
- **Hypothesis**: Quality over quantity—only pass on proven thinking patterns

#### Strategy B: Diversity Preservation
```python
def compact_reasoning(combined, max_size=100):
    clusters = cluster_by_pattern_structure(combined)
    return [cluster.best_pattern for cluster in clusters][:max_size]
```
- **Hypothesis**: Coverage beats optimization—need different cognitive strategies

#### Strategy C: Hybrid (Currently Used)
```python
def compact_reasoning(combined, max_size=100):
    # Combine score and usage metrics
    return sorted(combined,
                 key=lambda m: m.score * m.retrieval_count
                )[-max_size:]
```
- **Hypothesis**: Field-tested thinking beats theoretical quality

---

## The Geometric Insight

Traditional prompt engineering uses discrete, textual instructions:

```python
prompt = "Always consider counterarguments in paragraph 3"  # Brittle text
```

Reasoning pattern inheritance uses continuous, geometric relationships:

```python
reasoning_vector = similar_patterns + δ(innovation)  # Semantic evolution
```

**This is the natural way to evolve cognitive strategies in embedding space.**

In this system:
- **Distance encodes similarity**: Close reasoning patterns (in embedding space) solve similar problems
- **Retrieval becomes natural**: Vector search excels at finding similar cognitive structures
- **Evolution emerges from geometry**: Successful patterns cluster and propagate

---

## Why This Design

### Clean Separation of Concerns

**Critical decision:** Keep three layers completely separate:

1. **Prompts (Interface)**: What agents should do—never changes (YAML)
2. **RAG (Knowledge)**: What agents know—shared by all
3. **Reasoning (Cognition)**: How agents think—evolves through inheritance

This prevents contamination between layers and makes the system more robust.

### Why Lamarckian > Darwinian for AI Reasoning

**Darwinian evolution** (biological): Random mutations hope to improve
**Lamarckian evolution** (AI systems): Successful reasoning gets passed on directly

For AI agents, Lamarckian is superior because:
- **Direct cognitive transfer**: No need to rediscover reasoning patterns
- **Faster convergence**: Start each generation with proven thinking strategies
- **Preserves insights**: Successful cognitive approaches don't get lost
- **Natural selection still applies**: Bad reasoning leads to poor performance → no reproduction

**Example:**
```python
# Generation 1 agent discovers through experience:
personal_reasoning = {
    "thinking": "establish_context, add_statistics, pose_question",
    "score": 7.8
}

# Generation 2 inherits this as prior reasoning
# Generation 2 refines it:
personal_reasoning = {
    "thinking": "establish_context, add_statistics, contrast_viewpoint, pose_question",
    "score": 8.6
}

# Generation 3 inherits the refined pattern
```

Cognitive strategies accumulate. Each generation reasons better.

---

## What Success Looks Like

### Measurable Outcomes

- **Reasoning improvement**: Average scores increase through better cognitive strategies
- **Emergent specialization**: Roles develop distinct reasoning patterns
- **Sustained diversity**: Multiple problem-solving approaches coexist
- **Pattern effectiveness**: Retrieved reasoning correlates with performance
- **Strategy winner**: One compaction approach demonstrably outperforms

### Observable Phenomena

- **Reasoning lineages**: Successful cognitive patterns propagate across generations
- **Pattern refinement**: Later generations have more sophisticated reasoning
- **Cognitive accumulation**: Inherited + personal patterns per agent
- **Cross-pollination**: Diverse reasoning prevents cognitive monoculture
- **Role specialization**: Intro agents think differently than Body agents

---

## Technical Implementation

### Technology Stack

- **LangGraph**: Workflow orchestration
- **Multi-Provider LLM Support**:
  - **Anthropic Claude** (default): claude-3-5-sonnet, claude-3-haiku
  - **OpenAI GPT**: gpt-4-turbo-preview, gpt-4, gpt-3.5-turbo
- **ChromaDB**: Vector database for reasoning patterns (per-agent collections)
- **sentence-transformers**: Embeddings for reasoning similarity (all-MiniLM-L6-v2)
- **Shared RAG**: Separate vector DB for domain knowledge
- **Tavily**: Web search for external knowledge (optional)
- **Rich**: Terminal visualization for monitoring evolution

### Key Architecture Points

```
┌─────────────────────────────────────────────────────┐
│              Fixed Prompt Layer (YAML)              │
│           config/prompts/agents.yml                 │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│          Shared Knowledge Layer (RAG)               │
│      Domain facts, content, references              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│       Evolving Reasoning Layer (Inherited)          │
│                                                     │
│   ┌────────────┐  ┌────────────┐  ┌─────────────┐ │
│   │Coordinator │  │   Intro    │  │    Body     │ │
│   │   Pool     │  │   Pool     │  │    Pool     │ │
│   │ (N agents) │  │ (N agents) │  │  (N agents) │ │
│   └────────────┘  └────────────┘  └─────────────┘ │
│                                                     │
│   ┌──────────────────────────────────────────────┐ │
│   │         Conclusion Pool (N agents)           │ │
│   └──────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Project Structure

```
src/lean/
├── base_agent.py          # BaseAgent with multi-provider LLM support
├── coordinator.py         # CoordinatorAgent (Layer 1)
├── specialists.py         # Specialist agents (Layer 3)
├── agent_pool.py          # AgentPool for population management
├── reasoning_memory.py    # ReasoningMemory for cognitive patterns
├── shared_rag.py          # SharedRAG for domain knowledge
├── context_manager.py     # ContextManager for reasoning distribution
├── evaluation.py          # ContentEvaluator for scoring outputs
├── pipeline.py            # LangGraph workflow orchestration
├── state.py               # BlogState TypedDict
├── visualization.py       # StreamVisualizer for Rich terminal UI
├── config_loader.py       # YAML configuration loader
├── selection.py           # Selection strategies (tournament, rank, fitness)
├── compaction.py          # Compaction strategies (score, diversity, hybrid)
└── reproduction.py        # Reproduction strategies (sexual, asexual)

config/
├── experiments/           # Experiment configurations
│   ├── default.yml        # Production config (20 generations)
│   ├── fast_test.yml      # Fast test config (3 generations)
│   └── test.yml           # Test with visualization
├── prompts/
│   └── agents.yml         # Agent system prompts and evaluation criteria
└── docs/                  # Documentation for topic blocks and roles
    ├── ai-fundamentals.md
    ├── healthcare-ai.md
    └── ...
```

---

## Running the Experiment

### Prerequisites

- Python 3.11+
- **Either** Anthropic API key **OR** OpenAI API key
- uv package manager

### Installation

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env and add your API key
```

### Configuration

#### Using Anthropic Claude (Default)

```bash
# .env
LLM_PROVIDER=anthropic
MODEL_NAME=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your_key_here
```

#### Using OpenAI GPT

```bash
# .env
LLM_PROVIDER=openai
MODEL_NAME=gpt-4-turbo-preview
OPENAI_API_KEY=your_key_here
```

### YAML Configuration System

All experiment settings are in YAML files:

**Experiment Configuration** (`config/experiments/default.yml`):
```yaml
experiment:
  name: "Default AI Topics Experiment"
  population_size: 3              # Agents per role
  evolution_frequency: 5          # Evolve every N generations
  total_generations: 20
  reasoning_dir: "./data/reasoning"
  shared_rag_dir: "./data/shared_rag"
  domain: "General"

# Topic blocks for transfer learning
topic_blocks:
  - name: "AI Fundamentals"
    generation_range: [1, 4]
    documentation: "config/docs/ai-fundamentals.md"
    topics:
      - title: "The Future of Artificial Intelligence"
        keywords: ["AI", "future", "innovation"]
        difficulty: "intermediate"

# Model configuration
model:
  base_temperature: 0.7
  embedding_model: "all-MiniLM-L6-v2"

# Memory configuration
memory:
  max_knowledge_retrieve: 3
  max_reasoning_retrieve: 5
  inherited_reasoning_size: 100
  score_threshold: 7.0

# Research (Tavily)
research:
  enabled: true
  max_results: 5
  search_depth: "advanced"

# Evolution
evolution:
  selection_strategy: "tournament"    # or "rank" or "fitness"
  compaction_strategy: "hybrid"       # or "score" or "diversity"
  reproduction_strategy: "sexual"     # or "asexual"

# Human-in-the-loop
hitl:
  enabled: true
  auto_approve: false

# Visualization
visualization:
  enabled: true
```

**Agent Prompts** (`config/prompts/agents.yml`):
```yaml
coordinator:
  system_prompt: |
    You are a Coordinator Agent in the LEAN evolutionary system.
    Your role is to research, orchestrate, and critique.
  documentation: "config/docs/coordinator-role.md"

intro:
  system_prompt: |
    You are an Introduction Agent.
    Craft compelling introductions that hook readers.
  reasoning_focus: "engagement, clarity, preview"
```

### Run Experiments

```bash
# Run default experiment (20 generations)
uv run python main.py

# Run fast test (3 generations)
uv run python main.py --config fast_test

# Run with visualization enabled
uv run python main.py --config test
```

### Key Environment Variables

```bash
# LLM Provider Selection
LLM_PROVIDER=anthropic              # or "openai"
MODEL_NAME=claude-3-5-sonnet-20241022

# API Keys
ANTHROPIC_API_KEY=your_key_here     # If using Claude
OPENAI_API_KEY=your_key_here        # If using GPT
TAVILY_API_KEY=your_key_here        # Optional: for research

# Pipeline Features
ENABLE_SPECIALISTS=true             # Enable Layer 3 specialists
ENABLE_REVISION=true                # Enable revision loop
MAX_REVISIONS=2                     # Maximum revisions per generation
```

---

## What This Research Explores

### For AI Engineering

- **Stable multi-agent systems**: Fixed prompts + evolving reasoning
- **Self-improving cognition**: Reasoning patterns get better with use
- **Clean architecture**: Separation of interface, knowledge, and cognition
- **Dynamic prompt-equivalent behavior**: Without fragile prompt engineering
- **Hierarchical coordination**: Ensemble systems with coordinator oversight

### For Evolutionary Computation

- **Cognitive inheritance**: Can reasoning strategies be inherited?
- **Pattern compaction**: How to distill cognitive strategies?
- **Reasoning selection**: Which patterns produce best outcomes?
- **Population dynamics**: How do agent pools evolve?
- **Selection strategies**: Tournament vs rank vs fitness-weighted

### For Cognitive Architecture

- **Generational learning**: Each generation thinks better
- **Semantic reasoning**: Cognitive patterns in embedding space
- **Emergent specialization**: Roles develop unique thinking styles
- **Cultural transmission**: Reasoning strategies pass through generations
- **Hierarchical cognition**: Coordinator + content + specialist layers

---

## Research Objectives

This experiment investigates:
- Does reasoning pattern inheritance produce demonstrable improvement?
- Which cognitive compaction strategy works best?
- How do reasoning patterns evolve differently from content?
- Can agents develop role-specific cognitive strategies?
- Does hierarchical coordination improve over flat agent systems?
- Does this approach outperform traditional prompt engineering?

**Core hypothesis:** Prompts are bad at encoding how to think (that's why CoT helps). So keep prompts as simple interfaces (YAML), put content in RAG, and let reasoning patterns evolve through inheritance.

**Approach:** Empirical testing with statistical analysis. We're evolving cognition, not content.
