# HVAS Mini - Consolidated Architecture & Implementation Plan

> Last updated: 2025-10-18
>
> This document consolidates all brainstorming docs into a single reference for understanding the architecture and roadmap.

---

## Executive Summary

**Research Question**: Can AI agents improve by inheriting their parents' learned knowledge?

**Core Mechanism**: Lamarckian evolution - stable prompts + inherited memories + natural selection

**Status**: Proof of concept complete, M1 partially complete (3/5 branches merged)

---

## The Architecture (What We're Building)

### Core Principles

1. **Stable Prompts**: Genome (prompt templates) stays mostly unchanged
   - "You are an intro writer" never changes
   - Avoids prompt engineering fragility
   - Consistent behavior interface

2. **Memory Inheritance**: Offspring inherit compacted parent memories
   - 50-100 inherited memories from parents
   - 50-150 personal memories from own experience
   - Total: ~100-250 memories per agent

3. **Natural Selection**: Best performers reproduce, poor performers removed
   - ε-greedy, tournament, or fitness-weighted selection
   - Evolution every 10 generations
   - Add agents when diversity drops
   - Remove agents when fitness < 6.0 (after 20+ tasks)

4. **Context is Additive, Not Selective**: High performers broadcast more, but everyone receives equally
   - 40% from hierarchy (coordinator/parent)
   - 30% from high-credibility cross-role agents
   - 20% from random low-performer (forced diversity)
   - 10% from same-role peer

### Population Structure

```
3 Roles × 5 Agents = 15 Total Population

IntroAgents (5)         BodyAgents (5)         ConclusionAgents (5)
├─ intro_agent_1        ├─ body_agent_1        ├─ conclusion_agent_1
├─ intro_agent_2        ├─ body_agent_2        ├─ conclusion_agent_2
├─ intro_agent_3        ├─ body_agent_3        ├─ conclusion_agent_3
├─ intro_agent_4        ├─ body_agent_4        ├─ conclusion_agent_4
└─ intro_agent_5        └─ body_agent_5        └─ conclusion_agent_5

Each agent has isolated ChromaDB collection:
- intro_agent_1_memories
- intro_agent_2_memories
- ... (15 total collections)
```

### The Learning Cycle

```
For each generation, each agent:

1. START WITH INHERITANCE
   ↓ Already has 50-100 compacted memories from parents

2. RECEIVE CONTEXT (40/30/20/10)
   ↓ Weighted context from hierarchy + high-performers + diversity + peers

3. RETRIEVE MEMORIES
   ↓ Query own collection: inherited + personal
   ↓ Weighted by: similarity × (score/10)

4. GENERATE CONTENT
   ↓ Stable prompt + retrieved memories + received context

5. EVALUATE (LLM scoring)
   ↓ Engagement, clarity, depth, informativeness

6. STORE (ALL experiences)
   ↓ Add to personal memory bank with score

7. EVOLVE (every 10 generations)
   ├─ SELECTION: Choose best agents as parents
   ├─ COMPACTION: Merge + distill parents' memories
   ├─ REPRODUCTION: Child inherits compacted knowledge
   └─ POPULATION MGMT: Add/remove agents based on performance
```

### Memory Compaction Strategies (Testing 3 in Parallel)

**Strategy A: Score-Weighted (Quality-focused)**
- Keep highest-scoring memories only
- Selection: ε-greedy (90/10)
- Evolution: Slow (every 20 gens)
- Hypothesis: Elite inheritance works best

**Strategy B: Diversity-Based (Coverage-focused)**
- Cluster memories, keep representatives
- Selection: Tournament (top 3)
- Evolution: Fast (every 5 gens)
- Hypothesis: Coverage beats optimization

**Strategy C: Usage-Based (Practical-focused)**
- Keep most-retrieved memories (retrieval_count × score)
- Selection: Fitness-proportional
- Evolution: Adaptive
- Hypothesis: Field-tested knowledge wins

---

## Why This Design

### Problem: Prompts Are Bad at Encoding Nuance

Traditional approach:
```python
# Prompt engineering (fragile, verbose, degrades with mutation)
prompt = """You are an intro writer.
Use hooks. Be engaging. Start with questions.
Make it compelling. Use statistics when relevant.
Consider the reader's attention span...
[hundreds of lines of instructions]"""
```

Our approach:
```python
# Stable prompt (simple, consistent)
prompt = "You are an intro writer."

# Knowledge in memories (specific, tested, valuable)
memories = [
    "Questions increased engagement by 31% on technical topics",
    "Statistics work best in second sentence, not first",
    "Personal anecdotes outperformed questions for emotional topics"
]
```

**Key Insight**: That's why we need RAG in the first place—prompts can't hold nuanced knowledge. So keep prompts stable and evolve the knowledge base (memories) instead.

### Why Lamarckian > Darwinian for AI

**Darwinian** (biological): Random mutations hope to find improvements
**Lamarckian** (AI): Acquired knowledge inherited directly

For AI agents, Lamarckian is superior:
- Direct knowledge transfer (no re-learning)
- Faster convergence (start with inherited wisdom)
- Preserves hard-won insights
- Natural selection still applies (bad knowledge → poor performance → no reproduction)

Example progression:
```
Gen 1: "Questions increased engagement by 31%"
       ↓ (learns through experience)
Gen 2: Inherits Gen 1's insight +
       "Questions + statistics = 43% engagement"
       ↓ (builds on inherited knowledge)
Gen 3: Inherits both insights +
       "Questions + statistics + personal anecdote = 58% engagement"
```

Knowledge accumulates. Each generation starts ahead.

---

## Implementation Status

### Completed (Merged to Master)

✅ **M1.1: agent-pool-infrastructure**
- `AgentPool` class managing 5 agents per role
- Selection strategies: ε-greedy, fitness-weighted, tournament, best
- Helper methods: `get_top_n()`, `get_random_lower_half()`

✅ **M1.2: individual-memory-collections**
- Store ALL experiences (no score threshold)
- Weighted retrieval: `similarity × (score/10)`
- Inherited memory support
- Per-agent ChromaDB collections

✅ **M1.6: tavily-web-search**
- Tavily API integration for external knowledge
- Dynamic topic generation
- Search result incorporation

### Current Work (In Progress)

**Active worktrees** (branches in development):

⏳ **M1.3: fitness-tracking** (worktree exists)
- Overall fitness tracking
- Domain-specific performance (ML, Python, Web, General)
- Specialization detection
- Pool-level helpers
- **Worktree**: `worktrees/fitness-tracking`
- **AGENT_TASK.md**: `docs/feature-plans/fitness-tracking/AGENT_TASK.md`

⏳ **M1.4: context-distribution** (worktree exists)
- `ContextManager` with 40/30/20/10 weighted distribution
- Cross-role context flow
- Forced diversity requirement
- Depends on M1.1
- **Worktree**: `worktrees/context-distribution`
- **AGENT_TASK.md**: `docs/feature-plans/context-distribution/AGENT_TASK.md`

⏳ **M1.5: pipeline-integration** (worktree exists)
- `EvolutionaryWorkflow` integrating all M1 components
- LangGraph pipeline: intro → body → conclusion → assemble
- `LLMEvaluator` for scoring
- Main runner for multi-generation experiments
- Depends on M1.1-M1.4
- **Worktree**: `worktrees/pipeline-integration`
- **AGENT_TASK.md**: `docs/feature-plans/pipeline-integration/AGENT_TASK.md`

**Other active worktrees**:

🔧 **core-concept-refactor** (exploratory)
- Worktree: `worktrees/core-concept-refactor`
- Design documents for credibility and evolutionary multi-agent system

🔧 **visualization-v2** (enhancement)
- Worktree: `worktrees/visualization-v2`
- Improved visualization system

### Next Steps (Milestones 2-5)

**Milestone 2: Memory Inheritance (Evolution)**
- M2.1: Memory compaction strategies
- M2.2: Reproduction with memory inheritance
- M2.3: Population management (add/remove agents)
- M2.4: Evolution pipeline integration

**Milestone 3: Strategies**
- M3.1: Strategy abstraction layer
- M3.2: Three baseline strategies (A/B/C)
- M3.3: Parallel execution framework

**Milestone 4: Experimentation**
- M4.1: 100-generation experiment runner
- M4.2: Statistical analysis (t-tests, ANOVA, regression)
- M4.3: Lineage tracking and visualization

**Milestone 5: Enhancement**
- M5.1: Streamlit dashboard
- M5.2: Search integration (already merged, may enhance)
- M5.3: Meta-evolution (evolve compaction strategies)

---

## Success Metrics (100 Generations)

### Primary Outcomes

1. **Fitness Improvement**: Average scores increase >0.5 points
2. **Emergent Specialization**: Domain-specific variance >1.0
3. **Sustained Diversity**: Population std dev >0.5
4. **Memory Effectiveness**: Retrieval count correlates with performance
5. **Strategy Winner**: One configuration significantly outperforms others (p < 0.05)

### Observable Phenomena

- Knowledge lineages propagate across generations
- Memory accumulation (inherited + personal = 100-250 per agent)
- Compaction quality improves over time
- Cross-pollination from 20% diversity injection
- Role-specific specialization (intro agents ≠ body agents)

### Failure Modes (What to Watch For)

❌ **All strategies converge**: Compaction approach doesn't matter
❌ **No generational improvement**: Later gens perform same as Gen 1
❌ **Memory bloat**: Inherited memories add noise instead of signal
❌ **Diversity collapse**: All agents become identical
❌ **No specialization**: Agents don't develop distinct knowledge

**Note**: Any failure mode provides valuable negative results. Adjust and retry.

---

## Configuration

### Key Parameters

```bash
# Memory
MEMORY_SCORE_THRESHOLD=7.0      # DEPRECATED in M1: store ALL experiences
INHERITED_MEMORY_SIZE=100       # Max memories from parents

# Population
MIN_POPULATION=3                # Per role
MAX_POPULATION=8                # Per role
INITIAL_POPULATION=5            # Per role

# Evolution
EVOLUTION_FREQUENCY=10          # Generations between cycles
REMOVAL_FITNESS_THRESHOLD=6.0   # Remove if below (after 20+ tasks)

# Context Distribution
CONTEXT_WEIGHTS=40,30,20,10     # Hierarchy/High/Low/Peer

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
- **ChromaDB**: Vector database (15 isolated collections)
- **sentence-transformers**: Embeddings for semantic similarity
- **Tavily**: Web search integration
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
uv run pytest tests/ --cov=src/hvas_mini --cov-report=html
open htmlcov/index.html

# Type checking
uv run mypy src/hvas_mini
```

---

## Immediate Next Actions

### Priority 1: Complete M1 (3 branches in progress)

1. **Complete M1.3** (fitness-tracking) - worktree active
   ```bash
   cd worktrees/fitness-tracking
   cat ../../docs/feature-plans/fitness-tracking/AGENT_TASK.md
   ```
   - Implement domain-specific fitness tracking
   - Add specialization detection
   - Write comprehensive tests
   - Merge to master when complete

2. **Complete M1.4** (context-distribution) - worktree active
   ```bash
   cd worktrees/context-distribution
   cat ../../docs/feature-plans/context-distribution/AGENT_TASK.md
   ```
   - Implement `ContextManager` with 40/30/20/10 distribution
   - Add broadcast tracking
   - Write comprehensive tests
   - Merge to master when complete

3. **Complete M1.5** (pipeline-integration) - worktree active
   ```bash
   cd worktrees/pipeline-integration
   cat ../../docs/feature-plans/pipeline-integration/AGENT_TASK.md
   ```
   - Integrate all M1 components into workflow
   - Implement LLMEvaluator
   - Create main runner
   - Write integration tests
   - Merge to master when complete

### Priority 2: Verify M1 Complete

4. **Run integration tests**
   - Can run 20 generations successfully
   - All tests passing (>90% coverage)
   - Population statistics working
   - Context distribution verified (40/30/20/10)
   - Memory retrieval working (inherited + personal)

### Priority 3: Begin M2

5. **Proceed to M2** (memory inheritance)
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

1. **Architecture**: Lamarckian evolution = stable prompts + inherited memories + natural selection

2. **Status**: M1 is 3/5 complete (M1.1, M1.2, M1.6 merged; need M1.3, M1.4, M1.5)

3. **Next**: Complete M1, then proceed to M2 (memory inheritance/evolution)

4. **Research Question**: Does memory inheritance create demonstrable improvement over generations?

5. **Testing 3 Strategies**: Score-weighted, diversity-based, usage-based compaction

6. **Success**: Fitness improves >0.5 points, specialization emerges, diversity sustained

7. **This is Research**: May work, may not—both outcomes provide valuable empirical data
