# LEAN - Consolidated Architecture & Implementation Plan

**Lamarck Evolutionary Agent Network**

> Last updated: 2025-10-20 (MAJOR ARCHITECTURE REVISION)
>
> This document consolidates all brainstorming docs into a single reference for understanding the architecture and roadmap.
>
> **CRITICAL CHANGE**: Architecture now uses reasoning patterns (cognitive strategies), NOT content-based memories.
>
> **Note**: The project was previously named "HVAS Mini". References to that name in older worktree branches and historical documents refer to this same project.

---

## Executive Summary

**Research Question**: Can AI agents improve by inheriting their parents' reasoning patterns?

**Core Mechanism**: Lamarckian evolution - stable prompts + inherited reasoning strategies + natural selection

**Critical Insight**: We're evolving **how agents think** (reasoning patterns), not what they know (content goes to shared RAG)

**Status**: Proof of concept complete, M1 partially complete (3/5 branches merged) - **NEEDS REVISION for reasoning-pattern architecture**

---

## The Architecture (What We're Building)

### Three-Layer Separation of Concerns

**This is the foundational design principle:**

```
┌─────────────────────────────────────────┐
│  Layer 1: Fixed Prompts (Interface)     │
│  "You are an intro writer"              │
│  - Never changes, never mutates          │
│  - Stable behavioral interface           │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Layer 2: Shared RAG (Knowledge)        │
│  Domain facts, content, references      │
│  - Available to all agents equally       │
│  - Standard semantic retrieval           │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Layer 3: Evolving Reasoning (What      │
│            Gets Inherited)               │
│  - Planning sequences                    │
│  - Problem-solving strategies            │
│  - Reasoning traces                      │
│  - Retrieved by structural similarity    │
│  - Per-agent ChromaDB collections        │
└─────────────────────────────────────────┘
```

**Critical Distinction:**
- **Layer 1 (Prompts)**: WHAT agents should do
- **Layer 2 (Shared RAG)**: WHAT agents know
- **Layer 3 (Reasoning)**: HOW agents think (this is what evolves!)

### Core Principles

1. **Frozen Prompts**: Interface layer stays completely unchanged
   - "You are an intro writer" never changes
   - Avoids prompt engineering fragility
   - Consistent behavior interface across all generations

2. **Reasoning Pattern Inheritance**: Offspring inherit cognitive strategies
   - 50-100 inherited reasoning patterns from parents
   - 50-150 personal reasoning patterns from own experience
   - Total: ~100-250 reasoning patterns per agent
   - **NOT content**: Domain knowledge goes to shared RAG

3. **Natural Selection**: Best reasoners reproduce, poor performers removed
   - ε-greedy, tournament, or fitness-weighted selection
   - Evolution every 10 generations
   - Add agents when reasoning diversity drops
   - Remove agents when fitness < 6.0 (after 20+ tasks)

4. **Reasoning Context Distribution**: Agents share cognitive traces
   - 40% reasoning traces from hierarchy (coordinator/parent)
   - 30% reasoning traces from high-credibility cross-role agents
   - 20% reasoning traces from random low-performer (forced diversity)
   - 10% reasoning traces from same-role peer

### Population Structure

```
3 Roles × 5 Agents = 15 Total Population

IntroAgents (5)         BodyAgents (5)         ConclusionAgents (5)
├─ intro_agent_1        ├─ body_agent_1        ├─ conclusion_agent_1
├─ intro_agent_2        ├─ body_agent_2        ├─ conclusion_agent_2
├─ intro_agent_3        ├─ body_agent_3        ├─ conclusion_agent_3
├─ intro_agent_4        ├─ body_agent_4        ├─ conclusion_agent_4
└─ intro_agent_5        └─ body_agent_5        └─ conclusion_agent_5

Each agent has isolated ChromaDB collection for REASONING PATTERNS:
- intro_agent_1_reasoning  (planning steps, cognitive strategies)
- intro_agent_2_reasoning  (not content, but how to think)
- ... (15 total reasoning collections)

Separate shared RAG for domain knowledge:
- shared_rag_collection (facts, content, references for all agents)
```

### The Learning Cycle

```
For each generation, each agent:

1. START WITH INHERITANCE
   ↓ Already has 50-100 compacted REASONING PATTERNS from parents
   ↓ (planning strategies that worked for parents)

2. PLAN APPROACH
   ↓ Query own reasoning patterns for similar tasks
   ↓ "How did I/my parents solve problems like this?"

3. RETRIEVE KNOWLEDGE
   ↓ Query shared RAG for domain facts/content
   ↓ (separate from reasoning patterns)

4. RECEIVE CONTEXT (40/30/20/10)
   ↓ Get reasoning traces from other agents
   ↓ (how they approached their tasks, not what they produced)

5. GENERATE (with reasoning externalization)
   ↓ Fixed prompt + evolved reasoning patterns + domain knowledge + reasoning context
   ↓ LLM responds with <think> tags (reasoning) + <final> tags (output)

6. EVALUATE (LLM scoring)
   ↓ Quality of output (engagement, clarity, depth, informativeness)

7. STORE REASONING PATTERN
   ↓ Extract <think> section, store with metadata + score
   ↓ (NOT the content - that goes to shared RAG if needed)

8. EVOLVE (every 10 generations)
   ├─ SELECTION: Choose best reasoners as parents
   ├─ COMPACTION: Merge + distill parents' cognitive strategies
   ├─ REPRODUCTION: Child inherits compacted reasoning patterns
   └─ POPULATION MGMT: Add/remove agents based on reasoning effectiveness
```

---

## Implementation: Capturing Reasoning Patterns

**Key Challenge**: LLM APIs don't expose internal reasoning—only final outputs.

**Solution**: Induce the model to externalize reasoning via structured prompts.

### Step 1: Capture the Reasoning Trace

**Prompt structure:**
```python
SYSTEM: You are an intro writer. When generating content, include your
reasoning under <think> tags and your final output under <final> tags.

USER: Write an introduction for "The Evolution of Understanding"

CONTEXT (retrieved reasoning patterns):
- "Start with historical example to ground reader"
- "Use statistics in second sentence for credibility"
- "End with provocative question to drive engagement"

CONTEXT (shared RAG knowledge):
- WWII computing history
- Turing's work on machine intelligence
- Modern interpretability research
```

**LLM Response:**
```xml
<think>
The task is writing an introduction about AI understanding evolution.
Based on past patterns, I should:
1. Start with a historical anchor (Turing seems perfect given the topic)
2. Add a statistic about modern AI interpretability research
3. Build tension by contrasting past and present
4. End with a question about future implications

I'll use Turing's imitation game as the hook, cite recent interpretability
progress statistics, then pose a question about whether machines can truly
understand themselves.
</think>

<final>
In 1950, Alan Turing proposed a test: if a machine could fool humans into
thinking it was conscious, did the distinction matter? Seventy-five years
later, as neural networks process billions of parameters, a new question
emerges—not whether machines can think, but whether they can explain how
they think. Can AI learn to interpret itself?
</final>
```

### Step 2: Extract and Store the Reasoning

**Parse and structure:**
```python
reasoning_unit = {
    # Extracted from <think> tags
    "situation": "writing intro for AI evolution topic",
    "tactic": "historical anchor → statistic → tension → question",
    "reasoning": "The task is writing an introduction... (full <think> content)",

    # Metadata
    "score": 0.87,  # From evaluator
    "provenance": {
        "agent_id": "intro_agent_2",
        "generation": 5,
        "timestamp": "2025-10-20T..."
    },

    # For retrieval
    "embedding": [...],  # Vector embedding of the reasoning
    "retrieval_count": 0,  # How many times this pattern was retrieved

    # NOT stored
    # "output": "In 1950, Alan Turing..."  <- Goes to shared RAG if needed
}
```

### Step 3: Retrieval by Reasoning Similarity

**Next generation, same agent:**
```python
# New task: "Write intro for quantum computing and consciousness"
new_task_embedding = embed("intro for quantum computing consciousness")

# Search by reasoning pattern similarity (NOT content)
similar_reasoning = vector_db.search(
    query_embedding=new_task_embedding,
    filter={"score": {">": 7.0}},  # Only high-performing reasoning
    search_field="reasoning",  # Search the <think> content, not output
    top_k=5
)

# Returns:
# - "historical anchor → statistic → tension → question" (score: 8.7)
# - "provocative opening → credibility → intrigue" (score: 8.2)
# - ...
```

### Step 4: Scoring and Compaction

**After evaluation:**
```python
# Update reasoning unit with score
reasoning_unit["score"] = evaluator.score(output)  # 8.5
reasoning_unit["retrieval_count"] += 1  # Track usage

# Periodically (every 10 generations):
# 1. Prune low performers (score < 6.0 after 20+ retrievals)
# 2. Merge similar reasoning patterns (cluster by embedding distance)
# 3. Abstract new patterns from successful combinations
# 4. Pass compacted reasoning to offspring
```

---

## How This Differs from Traditional RAG

**Traditional RAG:**
- Stores: External facts, documents, knowledge
- Retrieves: "What information is relevant?"
- Example: Wikipedia articles, product docs

**Reasoning Pattern Retrieval:**
- Stores: Cognitive strategies, problem-solving approaches
- Retrieves: "How did I solve similar problems?"
- Example: "Use historical anchor → add statistic → pose question"

**Shared RAG (Layer 2):**
- Stores: Domain knowledge available to ALL agents
- Retrieves: "What facts do I need?"
- Example: Turing's work, interpretability statistics

**Key Insight**: We're retrieving fragments of **reasoning** (HOW to think), not fragments of **knowledge** (WHAT to know). Models think in embeddings—vector search excels at finding similar cognitive structures.

---

## Example Storage Schema

```python
# Per-agent reasoning collection (Layer 3: Evolving)
intro_agent_2_reasoning = [
    {
        "id": "reasoning_unit_001",
        "situation": "technical intro requiring credibility",
        "tactic": "historical_anchor → statistic → question",
        "reasoning": "<full think content>",
        "score": 8.7,
        "retrieval_count": 12,
        "generation": 5,
        "inherited_from": ["parent1_reasoning_087", "parent2_reasoning_134"]
    },
    # ... 100-250 reasoning units per agent
]

# Shared knowledge collection (Layer 2: Fixed)
shared_rag = [
    {
        "id": "fact_001",
        "content": "Alan Turing proposed the imitation game in 1950...",
        "topic": "AI history",
        "source": "Tavily search"
    },
    # ... domain facts available to all agents
]

# Fixed prompts (Layer 1: Frozen)
prompts = {
    "intro": "You are an intro writer. Include reasoning under <think> tags."
}
```

### Reasoning Pattern Compaction Strategies (Testing 3 in Parallel)

**Strategy A: Score-Weighted (Quality-focused)**
- Keep highest-scoring reasoning patterns only
- Selection: ε-greedy (90/10)
- Evolution: Slow (every 20 gens)
- Hypothesis: Elite cognitive strategies work best

**Strategy B: Diversity-Based (Coverage-focused)**
- Cluster reasoning patterns, keep representatives
- Selection: Tournament (top 3)
- Evolution: Fast (every 5 gens)
- Hypothesis: Diverse cognitive approaches beat optimization

**Strategy C: Usage-Based (Practical-focused)**
- Keep most-retrieved reasoning patterns (retrieval_count × score)
- Selection: Fitness-proportional
- Evolution: Adaptive
- Hypothesis: Field-tested cognitive strategies win

---

## Why This Design

### Problem: Prompts Are Bad at Encoding How to Think

Traditional approach:
```python
# Prompt engineering (fragile, verbose, degrades with mutation)
prompt = """You are an intro writer.
Use hooks. Be engaging. Start with questions.
Make it compelling. Use statistics when relevant.
Consider the reader's attention span...
[hundreds of lines of cognitive instructions]"""
```

Our approach (Three-Layer Separation):
```python
# Layer 1: Stable prompt (simple interface)
prompt = "You are an intro writer."

# Layer 2: Shared RAG (domain knowledge available to all)
shared_knowledge = ["WWII facts", "economic statistics", "historical context"]

# Layer 3: Reasoning patterns (cognitive strategies, evolves through inheritance)
reasoning_patterns = [
    {
        "planning_steps": ["establish_context", "add_statistics", "pose_question"],
        "execution_trace": "Used historical example → 3 stats → provocative question",
        "score": 8.5
    },
    # More cognitive strategies...
]
```

**Key Insight**: Models don't think in text—they think in embeddings. Vector databases excel at finding similar reasoning structures. "Find planning sequences like mine" is exactly what embedding search was built for. So keep prompts stable, put content in shared RAG, and evolve reasoning patterns through inheritance.

### Why Lamarckian > Darwinian for AI Reasoning

**Darwinian** (biological): Random mutations hope to find improvements
**Lamarckian** (AI systems): Acquired reasoning strategies inherited directly

For AI agents, Lamarckian is superior because:
- Direct cognitive transfer (no need to rediscover reasoning patterns)
- Faster convergence (start with proven thinking strategies)
- Preserves successful cognitive approaches
- Natural selection still applies (bad reasoning → poor performance → no reproduction)

Example progression:
```python
# Generation 1 discovers through experience:
personal_reasoning = {
    "pattern": ["establish_context", "add_statistics", "pose_question"],
    "score": 7.8
}

# Generation 2 inherits this as prior reasoning:
inherited_reasoning = {
    "pattern": ["establish_context", "add_statistics", "pose_question"],
    "score": 7.8,
    "generation": 1
}

# Generation 2 refines it:
personal_reasoning = {
    "pattern": ["establish_context", "add_statistics", "contrast_viewpoint", "pose_question"],
    "score": 8.6
}

# Generation 3 inherits the refined pattern
```

Cognitive strategies accumulate. Each generation reasons better.

---

## Implementation Status

### ⚠️ ARCHITECTURE CHANGE IMPACT

**The shift from content-based memories to reasoning patterns affects most branches.**

Key changes needed:
1. Separate ChromaDB collections: agent reasoning patterns + shared RAG
2. Store planning steps/execution traces, NOT content
3. Context distribution shares reasoning traces, not outputs
4. Retrieval queries reasoning patterns by structural similarity

### Completed (Merged to Master)

✅ **M1.1: agent-pool-infrastructure** - **STILL VALID**
- `AgentPool` class managing 5 agents per role
- Selection strategies: ε-greedy, fitness-weighted, tournament, best
- Helper methods: `get_top_n()`, `get_random_lower_half()`
- **No changes needed**: Population management is independent of what's stored

⚠️ **M1.2: individual-memory-collections** - **NEEDS REVISION**
- Currently stores content/experiences
- **Must change to**: Store reasoning patterns (planning steps + execution traces)
- **Must add**: Separate shared RAG for domain knowledge
- Weighted retrieval concept is still valid
- Per-agent ChromaDB collections structure is correct
- **Action**: Refactor to `ReasoningMemory` class

✅ **M1.6: tavily-web-search** - **STILL VALID**
- Tavily API integration for external knowledge
- **Purpose unchanged**: Provides domain facts for shared RAG
- Dynamic topic generation works as-is
- **No changes needed**: External knowledge gathering is orthogonal to reasoning patterns

### Current Work (In Progress)

**Active worktrees** (branches in development):

✅ **M1.3: fitness-tracking** - **STILL VALID** (worktree exists)
- Overall fitness tracking based on output quality
- Domain-specific performance (ML, Python, Web, General)
- Specialization detection
- Pool-level helpers
- **No changes needed**: Fitness is measured by output quality, independent of what's stored
- **Worktree**: `worktrees/fitness-tracking`
- **AGENT_TASK.md**: `docs/feature-plans/fitness-tracking/AGENT_TASK.md`

⚠️ **M1.4: context-distribution** - **NEEDS REVISION** (worktree exists)
- `ContextManager` with 40/30/20/10 weighted distribution
- **Must change**: Distribute reasoning traces, NOT content/outputs
- **From README**: "Receive context: Get reasoning traces from other agents"
- Forced diversity requirement still valid
- **Action**: Update to share cognitive strategies, not generated content
- **Worktree**: `worktrees/context-distribution`
- **AGENT_TASK.md**: `docs/feature-plans/context-distribution/AGENT_TASK.md`

⚠️ **M1.5: pipeline-integration** - **NEEDS MAJOR REVISION** (worktree exists)
- `EvolutionaryWorkflow` needs to implement new 8-step cycle
- **Must change**:
  1. Add "Plan Approach" step (query reasoning patterns)
  2. Separate "Retrieve Knowledge" (shared RAG) from reasoning patterns
  3. Store reasoning patterns + traces, NOT content
- LangGraph pipeline structure stays same: intro → body → conclusion → assemble
- `LLMEvaluator` for scoring still valid (scores output quality)
- **Action**: Major refactor to align with three-layer architecture
- **Worktree**: `worktrees/pipeline-integration`
- **AGENT_TASK.md**: `docs/feature-plans/pipeline-integration/AGENT_TASK.md`

**Other active worktrees**:

❓ **core-concept-refactor** - **REVIEW NEEDED** (exploratory)
- Worktree: `worktrees/core-concept-refactor`
- Design documents for credibility and evolutionary multi-agent system
- **May be superseded by** this README refinement
- **Action**: Review to see if still relevant

✅ **visualization-v2** - **STILL VALID** (enhancement)
- Worktree: `worktrees/visualization-v2`
- Improved visualization system
- **No changes needed**: Visualization is output-focused, independent of storage architecture

### Next Steps (Milestones 2-5)

**Milestone 2: Reasoning Pattern Inheritance (Evolution)**
- M2.1: Reasoning pattern compaction strategies
- M2.2: Reproduction with cognitive strategy inheritance
- M2.3: Population management (add/remove agents based on reasoning effectiveness)
- M2.4: Evolution pipeline integration

**Milestone 3: Strategies**
- M3.1: Strategy abstraction layer
- M3.2: Three baseline strategies (A/B/C for reasoning pattern compaction)
- M3.3: Parallel execution framework

**Milestone 4: Experimentation**
- M4.1: 100-generation experiment runner
- M4.2: Statistical analysis (t-tests, ANOVA, regression)
- M4.3: Cognitive lineage tracking and visualization

**Milestone 5: Enhancement**
- M5.1: Streamlit dashboard
- M5.2: Shared RAG integration (domain knowledge layer)
- M5.3: Meta-evolution (evolve reasoning pattern compaction strategies)

---

## Success Metrics (100 Generations)

### Primary Outcomes

1. **Reasoning Improvement**: Average scores increase >0.5 points through better cognitive strategies
2. **Emergent Specialization**: Roles develop distinct reasoning patterns
3. **Sustained Diversity**: Multiple problem-solving approaches coexist (std dev >0.5)
4. **Pattern Effectiveness**: Retrieved reasoning patterns correlate with performance
5. **Strategy Winner**: One compaction approach demonstrably outperforms others (p < 0.05)

### Observable Phenomena

- **Reasoning lineages**: Successful cognitive patterns propagate across generations
- **Pattern refinement**: Later generations have more sophisticated reasoning
- **Cognitive accumulation**: 50-100 inherited + 50-150 personal patterns per agent
- **Cross-pollination**: Diverse reasoning from 20% low-performer context injection
- **Role specialization**: Intro agents reason differently than Body agents

### Failure Modes (What to Watch For)

❌ **No reasoning improvement**: Cognitive patterns don't evolve
❌ **Pattern convergence**: All agents think the same way
❌ **Noise accumulation**: Inherited patterns add confusion instead of clarity
❌ **Diversity collapse**: Selection pressure eliminates alternative reasoning
❌ **No specialization**: Roles don't develop distinct cognitive strategies

**Note**: Any failure mode provides valuable negative results. Adjust and retry.

---

## Configuration

### Key Parameters

```bash
# Reasoning Patterns (Layer 3: Evolving)
INHERITED_REASONING_SIZE=100    # Max reasoning patterns inherited from parents
PERSONAL_REASONING_SIZE=150     # Max personal reasoning patterns stored

# Shared Knowledge (Layer 2: Fixed)
USE_SHARED_RAG=true             # Separate domain knowledge layer
REASONING_SEARCH_ONLY=true      # Don't mix content with reasoning patterns

# Prompts (Layer 1: Fixed)
PROMPTS_IMMUTABLE=true          # Never modify prompts during evolution

# Population
MIN_POPULATION=3                # Per role
MAX_POPULATION=8                # Per role
INITIAL_POPULATION=5            # Per role

# Evolution
EVOLUTION_FREQUENCY=10          # Generations between cycles
REMOVAL_FITNESS_THRESHOLD=6.0   # Remove if below (after 20+ tasks)

# Reasoning Context Distribution
CONTEXT_WEIGHTS=40,30,20,10     # Hierarchy/High/Low/Peer (reasoning traces, not content)

# Strategy-specific (varies by strategy)
SELECTION_METHOD=epsilon_greedy|tournament|fitness_weighted
COMPACTION_METHOD=score_weighted|diversity_based|usage_based
```

---

## Test Dataset (5 Topics)

1. **Tracing Thought: How Neural Activation Maps Reveal Machine Cognition**
   - Baseline: Mechanistic interpretability

2. **Evolving Insight: Can AI Learn to Interpret Itself?**
   - Builds on #1: Self-interpreting agents

3. **Quantum Selection: What Evolution Might Look Like in Quantum AI**
   - Domain shift: Quantum computing + evolution

4. **Quantum Minds and Digital Species: Evolution Beyond Classical Computation**
   - Extends #3: Quantum + evolutionary biology

5. **The Evolution of Understanding: From Biological Brains to Self-Explaining Machines**
   - Synthesis: Integrates all previous concepts

**Design**: Tests retention (1→2), domain transfer (2→3), synthesis (all→5)

---

## Technology Stack

- **LangGraph**: Workflow orchestration
- **Anthropic Claude**: LLM for generation and evaluation
- **ChromaDB**: Vector database
  - 15 isolated collections for reasoning patterns (per-agent cognitive strategies)
  - 1 shared collection for domain knowledge (shared RAG)
- **sentence-transformers**: Embeddings for reasoning pattern similarity
- **Tavily**: Web search for external knowledge (feeds shared RAG)
- **Streamlit**: Dashboard (M5)

---

## Quick Reference Commands

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY

# Run current prototype
uv run python main.py

# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_memory.py -v

# Coverage report
uv run pytest tests/ --cov=src/lean --cov-report=html
open htmlcov/index.html

# Type checking
uv run mypy src/lean
```

---

## Immediate Next Actions

### Priority 0: Architecture Alignment (CRITICAL)

0. **Refactor M1.2** (individual-memory-collections) - **MUST DO FIRST**
   ```bash
   # This branch is merged but needs refactoring
   ```
   - Rename `MemoryManager` → `ReasoningMemory`
   - Store reasoning patterns (planning steps + execution traces), NOT content
   - Create separate `SharedRAG` class for domain knowledge
   - **Implementation approach**:
     - Add `<think>` and `<final>` tags to system prompts
     - Parse LLM responses to extract `<think>` section
     - Store `<think>` content as reasoning pattern with metadata
     - Optionally store `<final>` content in shared RAG
   - Update storage schema:
     - `situation`, `tactic`, `reasoning` (extracted from `<think>`)
     - `score`, `retrieval_count`, `generation`, `provenance`, `embedding`
     - `inherited_from` (lineage tracking)
   - **This is foundational**: Other branches depend on this

### Priority 1: Update M1 Branches for Reasoning Patterns

1. **Complete M1.3** (fitness-tracking) - ✅ No changes needed
   ```bash
   cd worktrees/fitness-tracking
   cat ../../docs/feature-plans/fitness-tracking/AGENT_TASK.md
   ```
   - Implement domain-specific fitness tracking
   - Add specialization detection
   - Write comprehensive tests
   - Merge to master when complete

2. **Revise M1.4** (context-distribution) - ⚠️ Update AGENT_TASK.md first
   ```bash
   cd worktrees/context-distribution
   cat ../../docs/feature-plans/context-distribution/AGENT_TASK.md
   ```
   - **BEFORE implementing**: Update AGENT_TASK.md to share reasoning traces
   - Implement `ContextManager` with 40/30/20/10 distribution
   - Distribute reasoning traces (planning steps + execution traces), NOT outputs
   - Add broadcast tracking
   - Write comprehensive tests
   - Merge to master when complete

3. **Revise M1.5** (pipeline-integration) - ⚠️ Major refactor needed
   ```bash
   cd worktrees/pipeline-integration
   cat ../../docs/feature-plans/pipeline-integration/AGENT_TASK.md
   ```
   - **BEFORE implementing**: Update AGENT_TASK.md to reflect 8-step cycle
   - Implement new workflow:
     1. Start with inheritance
     2. Plan approach (query reasoning patterns)
     3. Retrieve knowledge (shared RAG)
     4. Receive context (reasoning traces)
     5. Generate
     6. Evaluate
     7. Store reasoning pattern
     8. Evolve
   - Integrate ReasoningMemory + SharedRAG
   - Implement LLMEvaluator
   - Create main runner
   - Write integration tests
   - Merge to master when complete

### Priority 2: Verify M1 Complete

4. **Run integration tests**
   - Can run 20 generations successfully
   - All tests passing (>90% coverage)
   - Population statistics working
   - Reasoning context distribution verified (40/30/20/10)
   - Reasoning pattern retrieval working (inherited + personal)
   - Shared RAG working (domain knowledge separation)

### Priority 3: Begin M2

5. **Proceed to M2** (reasoning pattern inheritance)
   - Create M2.1 and M2.2 worktrees
   - Start with M2.1 and M2.2 in parallel
   - Then M2.3, M2.4 sequentially

---

## Future Enhancement: Neural Circuit Evolution

**Status**: Documented in README, NOT currently implemented

**Concept**: Evolve neural pathways using open-source models with LoRA adapters

**Why Later**:
- Requires significant GPU resources
- Beyond scope of initial memory inheritance research
- Only pursue if memory evolution shows promise

**Toggle**: `USE_NEURAL_EVOLUTION = False` (default)

---

## Document Index

- **README.md**: Research overview, architecture, motivation
- **CLAUDE.md**: Developer guidance for working with codebase
- **docs/technical.md**: Setup, configuration, architecture details
- **docs/CONSOLIDATED_PLAN.md**: This file (single reference)
- **docs/brainstorming/**:
  - `NEXT_ITTERATION.md`: Design decisions questionnaire
  - `ANOTHER_ASPECT.MD`: Memory inheritance insight (critical)
  - `IMPLEMENTATION_ORCHESTRATION_REVISED.md`: Detailed milestone breakdown
  - `IMPLEMENTATION_PLAN_FINAL.md`: Comprehensive plan
- **docs/feature-plans/<branch>/AGENT_TASK.md**: Per-branch implementation guides

---

## Key Takeaways

1. **Architecture**: Three-layer separation
   - Layer 1 (Prompts): WHAT to do - frozen
   - Layer 2 (Shared RAG): WHAT to know - shared by all
   - Layer 3 (Reasoning): HOW to think - evolves through inheritance

2. **Critical Insight**: We're evolving cognitive strategies (reasoning patterns), not content
   - Content goes to shared RAG
   - Reasoning patterns (planning steps + execution traces) get inherited
   - Models think in embeddings, not text

3. **Status**: M1 needs architecture alignment
   - M1.1 (agent-pool): ✅ Valid
   - M1.2 (memory): ⚠️ Needs refactor to ReasoningMemory + SharedRAG
   - M1.3 (fitness): ✅ Valid
   - M1.4 (context): ⚠️ Needs revision (share reasoning traces, not content)
   - M1.5 (pipeline): ⚠️ Major revision (8-step cycle with reasoning patterns)
   - M1.6 (search): ✅ Valid

4. **Research Question**: Does reasoning pattern inheritance create demonstrable improvement over generations?

5. **Testing 3 Strategies**: Score-weighted, diversity-based, usage-based compaction (of cognitive strategies)

6. **Success**: Reasoning improves >0.5 points, cognitive specialization emerges, diversity sustained

7. **This is Research**: May work, may not—both outcomes provide valuable empirical data
