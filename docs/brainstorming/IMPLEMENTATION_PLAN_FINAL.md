# HVAS Mini - Final Implementation Plan

## Overview

This document provides the complete, finalized implementation plan incorporating:
- Memory inheritance architecture (from `ANOTHER_ASPECT.MD`)
- Design decisions (from `NEXT_ITERATION.md`)
- Revised orchestration (from `IMPLEMENTATION_ORCHESTRATION_REVISED.md`)
- Future neural evolution feature (documented in README, not implemented initially)

## Core Architecture (Current Implementation)

### What We're Building

An evolutionary multi-agent system testing **Lamarckian evolution** through memory inheritance:

**Core Mechanisms**:
1. **Stable Prompts**: Genome (prompt templates) stays mostly unchanged
2. **Memory Inheritance**: Offspring inherit compacted parent memories (50-100)
3. **Personal Experience**: Each agent accumulates own memories (50-150)
4. **Natural Selection**: Best performers reproduce, poor performers removed
5. **Knowledge Accumulation**: Each generation starts with inherited wisdom

**Population**:
- 3 roles (intro, body, conclusion)
- 5 agents per role = 15 total agents
- Each agent has isolated ChromaDB collection
- Evolution every 10 generations

**Key Insight**: Prompts bad at encoding nuance → Keep stable, evolve knowledge (memories) instead

## Implementation Timeline

**NO FIXED TIMELINE** - Concurrent development, complete when done

**Approach**:
- Parallel branches where possible
- Comprehensive unit tests for every module (>90% coverage)
- Integration after each milestone
- All work happens in feature branches via worktrees

## Milestone 1: Core System (Foundation)

### Goal
Multi-agent populations with selection, context distribution, fitness tracking, and memory inheritance STRUCTURE (not yet reproduction)

### Branches

#### M1.1: `agent-pool-infrastructure`
**Status**: ✅ Worktree created, AGENT_TASK.md complete
**Can start**: Immediately
**Parallel with**: M1.2, M1.3

**Implements**:
- `AgentPool` class managing 5 agents per role
- Selection strategies: ε-greedy, fitness-weighted, tournament, best
- Helper methods: `get_top_n()`, `get_random_lower_half()`
- Factory function: `create_agent_pool()`

**Tests**: `tests/test_agent_pool.py`

---

#### M1.2: `individual-memory-collections`
**Status**: ✅ Worktree created, AGENT_TASK.md complete
**Can start**: Immediately
**Parallel with**: M1.1, M1.3

**Implements**:
- **Remove score threshold** (store ALL experiences)
- **Weighted retrieval**: `relevance = similarity × (score/10)`
- **Inherited memory support**: Load parent memories on init
- Per-agent collections: `intro_agent_1_memories`, etc.
- Separate personal vs inherited memories

**Critical Change**:
```python
class MemoryManager:
    def __init__(self, collection_name, persist_directory, inherited_memories=None):
        # Load inherited memories if provided
        if inherited_memories:
            self._load_inherited_memories(inherited_memories)
```

**Tests**: `tests/test_memory_manager.py`

---

#### M1.3: `fitness-tracking`
**Status**: ✅ Worktree created, AGENT_TASK.md complete
**Can start**: Immediately
**Parallel with**: M1.1, M1.2

**Implements**:
- Overall fitness tracking (`fitness_history`, `avg_fitness()`)
- **Domain-specific tracking**: ML, Python, Web, General categories
- Specialization detection (`is_specialist()`)
- Pool-level helpers: `select_specialist()`, `is_stagnating()`

**Tests**: `tests/test_fitness_tracking.py`

---

#### M1.4: `context-distribution`
**Status**: ✅ Worktree created, AGENT_TASK.md complete
**Can start**: After M1.1 complete (needs `get_top_n()`)
**Depends on**: M1.1

**Implements**:
- `ContextManager` with **40/30/20/10 weighted distribution**:
  - 40% Hierarchy/parent context
  - 30% High-credibility cross-role agents
  - 20% Random low-performer (forced diversity)
  - 10% Same-role peer
- Broadcast tracking and diversity measurement
- **Principle**: "Credibility grants REACH, not ACCESS"

**Tests**: `tests/test_context_manager.py`

---

#### M1.5: `pipeline-integration`
**Status**: ✅ AGENT_TASK.md complete
**Can start**: After M1.1-M1.4 complete
**Depends on**: All M1 branches

**Implements**:
- `EvolutionaryWorkflow` integrating all M1 components
- LangGraph pipeline: intro → body → conclusion → assemble
- `LLMEvaluator` for scoring outputs
- Main runner for multi-generation experiments
- Results export and population statistics
- **Remove**: Parameter evolution (deprecated)

**Tests**: `tests/test_integration.py`, `tests/test_workflow.py`, `tests/test_evaluator.py`

---

### M1 Completion Criteria

- [ ] All 5 branches implemented
- [ ] All unit tests passing (>90% coverage)
- [ ] Can run 20 generations successfully
- [ ] Fitness tracked correctly (overall + domain-specific)
- [ ] Context distribution working (40/30/20/10)
- [ ] Memory storage working (ALL experiences stored)
- [ ] Population statistics accessible
- [ ] NO evolution yet (that's M2)

---

## Milestone 2: Memory Inheritance (Evolution)

### Goal
Enable Lamarckian evolution through memory compaction and inheritance

### Branches

#### M2.1: `memory-compaction`
**Can start**: After M1 complete
**Parallel with**: M2.2

**Implements**:
- `MemoryCompactor` class with 3+ strategies:
  - **Score-weighted**: Keep highest-scoring memories
  - **Diversity-based**: Cluster and keep representatives
  - **Frequency-based**: Keep most-used (retrieval_count × score × recency)
  - **Balanced**: 50% score, 30% diversity, 20% frequency
- Compaction from `target_size=100` to manageable inheritance

**Tests**: `tests/test_memory_compaction.py`

---

#### M2.2: `memory-inheritance-reproduction`
**Can start**: After M1 complete
**Parallel with**: M2.1

**Implements**:
- `ReproductionManager` class
- Sexual reproduction with memory inheritance (NOT prompt mutation)
- Parent selection for reproduction
- Offspring initialization with inherited memories
- Lineage tracking
- **Minimal prompt variation** (5% chance, simple synonyms only)

**Critical Implementation**:
```python
def reproduce(parent1, parent2):
    # 1. Compact parent memories
    inherited = compactor.compact_memories(
        parent1.all_memories(),
        parent2.all_memories(),
        target_size=100,
        strategy="balanced"
    )

    # 2. Minimal prompt variation (mostly stable)
    if random.random() < 0.05:
        child_genome = minimal_variation(parent1.genome)
    else:
        child_genome = parent1.genome  # Same as parent

    # 3. Create child with inherited memories
    child = Agent(
        genome=child_genome,
        inherited_memories=inherited  # KEY
    )
    return child
```

**Tests**: `tests/test_reproduction.py`

---

#### M2.3: `population-management`
**Can start**: After M2.1, M2.2 complete
**Depends on**: M2.1, M2.2

**Implements**:
- `EvolutionManager` class
- Add agent triggers (diversity < threshold, stagnation detected)
- Remove agent triggers (fitness < 6.0 after 20+ tasks)
- Evolution cycle (every 10 generations)
- Population bounds (min 3, max 8 per role)
- **Uses memory inheritance** for reproduction
- **No genome mutation** logic

**Tests**: `tests/test_population_management.py`, `tests/test_evolution_manager.py`

---

#### M2.4: `evolution-pipeline-integration`
**Can start**: After M2.3 complete
**Depends on**: All M2 branches

**Implements**:
- Integrate reproduction into workflow
- Evolution triggers in pipeline
- Lineage tracking across generations
- Generation counter and statistics

**Tests**: `tests/test_evolution_integration.py`

---

### M2 Completion Criteria

- [ ] Memory compaction working (3+ strategies)
- [ ] Reproduction creates offspring with inherited memories
- [ ] Offspring start with 50-100 inherited memories
- [ ] Population adds/removes agents correctly
- [ ] Evolution runs automatically every 10 gens
- [ ] **Prompts remain stable** (minimal 5% variation)
- [ ] All M2 tests passing (>90% coverage)
- [ ] Can run 30+ generations with memory inheritance
- [ ] Lineage tracking functional

---

## Milestone 3: Strategies

### Goal
Abstract evolutionary configurations into testable strategies

### Branches

#### M3.1: `strategy-abstraction`
**Can start**: After M2 complete

**Implements**:
- `EvolutionaryStrategy` class
- Strategy components:
  - Selection method
  - Memory compaction approach
  - Context distribution weights
  - Evolution frequency
  - Memory retrieval weighting

**Tests**: `tests/test_strategy.py`

---

#### M3.2: `baseline-strategies`
**Can start**: Parallel with M3.1

**Implements three baseline strategies**:

**Strategy A: Conservative (Quality-focused)**
- Selection: ε-greedy (90/10)
- Compaction: Score-weighted
- Evolution: Slow (every 20 gens)
- Retrieval: Heavy quality weighting

**Strategy B: Aggressive (Coverage-focused)**
- Selection: Tournament (top 3)
- Compaction: Diversity-based
- Evolution: Fast (every 5 gens)
- Retrieval: Pure similarity

**Strategy C: Balanced (Adaptive)**
- Selection: Fitness-proportional
- Compaction: Balanced (50/30/20)
- Evolution: Adaptive
- Retrieval: Balanced weighting

**Tests**: `tests/test_baseline_strategies.py`

---

#### M3.3: `parallel-execution`
**Can start**: After M3.1, M3.2 complete

**Implements**:
- Run 3 strategies in isolated populations
- Same task sequences for fair comparison
- No cross-contamination between populations

**Tests**: `tests/test_parallel_execution.py`

---

### M3 Completion Criteria

- [ ] Strategy abstraction working
- [ ] 3 baseline strategies implemented
- [ ] Can run strategies in parallel
- [ ] Isolated populations (no cross-contamination)
- [ ] All M3 tests passing
- [ ] Can run 50 generations per strategy

---

## Milestone 4: Experimentation

### Goal
100-generation experiments with statistical analysis

### Branches

#### M4.1: `experiment-runner`
**Can start**: After M3 complete

**Implements**:
- 100-generation experiment orchestration
- Task dataset (12 topics × 4 domains)
- Results collection and export
- Progress tracking and checkpointing

**Tests**: `tests/test_experiment_runner.py`

---

#### M4.2: `statistical-analysis`
**Can start**: Parallel with M4.1

**Implements**:
- Fitness trajectory analysis
- t-tests between strategies
- ANOVA for multi-strategy comparison
- Regression analysis (fitness vs generation)
- Diversity metrics over time
- Specialization emergence patterns

**Tests**: `tests/test_statistical_analysis.py`

---

#### M4.3: `lineage-tracking`
**Can start**: Parallel with M4.1

**Implements**:
- Family tree visualization
- Memory inheritance flow analysis
- Knowledge propagation tracking
- Identify successful lineages

**Tests**: `tests/test_lineage_tracking.py`

---

### M4 Completion Criteria

- [ ] Can run 100 generations successfully
- [ ] All 3 strategies complete 100-gen runs
- [ ] Statistical analysis implemented
- [ ] Lineage tracking working
- [ ] Results exportable for analysis
- [ ] All M4 tests passing

---

## Milestone 5: Enhancement

### Goal
Dashboard, visualization, and advanced features

### Branches

#### M5.1: `streamlit-dashboard`
**Can start**: After M4 complete (or parallel)

**Implements**:
- Real-time fitness plots
- Population diversity visualization
- Context flow diagrams
- Domain specialization heatmaps
- Lineage tree explorer

**Tests**: Manual testing (UI)

---

#### M5.2: `search-integration`
**Can start**: Parallel with M5.1

**Implements**:
- Brave Search API integration
- Dynamic topic generation
- External knowledge retrieval
- Search result incorporation

**Tests**: `tests/test_search_integration.py`

---

#### M5.3: `meta-evolution`
**Can start**: Parallel with M5.1, M5.2

**Implements**:
- Evolve compaction strategies themselves
- Strategy mutation and crossover
- Strategy fitness tracking
- Meta-level selection

**Tests**: `tests/test_meta_evolution.py`

---

### M5 Completion Criteria

- [ ] Dashboard functional
- [ ] Search integration working
- [ ] Meta-evolution implemented
- [ ] All M5 tests passing
- [ ] Full system demonstration ready

---

## Future Enhancement: Neural Circuit Evolution

**Status**: Documented in README, NOT currently implemented

### Concept

Evolve the **neural pathways** themselves using open-source models with LoRA adapters:

**Features**:
- Circuit discovery (find active neural pathways)
- LoRA adapter evolution (genetic algorithms on model weights)
- Sexual reproduction of LoRA weights
- Specialized neural pathways per agent

**Models**:
- Lightweight: `microsoft/phi-2` (2.7B)
- Advanced: `mistralai/Mistral-7B-Instruct-v0.2`

**Why Later**:
- Requires significant GPU resources
- Beyond scope of initial memory inheritance research
- Only pursue if memory evolution experiments show promise

**Toggle**:
```python
USE_NEURAL_EVOLUTION = False  # Default: memory-only
```

---

## Testing Requirements

### Coverage Targets

- **Per Module**: >90% code coverage
- **Overall**: >85% code coverage
- **Critical Modules**: 100% coverage
  - `memory.py`
  - `reproduction.py`
  - `evolution_manager.py`

### Test Structure

```
tests/
├── test_agent_pool.py              # M1.1
├── test_memory_manager.py          # M1.2
├── test_fitness_tracking.py        # M1.3
├── test_context_manager.py         # M1.4
├── test_integration.py             # M1.5
├── test_workflow.py                # M1.5
├── test_evaluator.py               # M1.5
├── test_memory_compaction.py       # M2.1
├── test_reproduction.py            # M2.2
├── test_population_management.py   # M2.3
├── test_evolution_manager.py       # M2.3
├── test_evolution_integration.py   # M2.4
├── test_strategy.py                # M3.1
├── test_baseline_strategies.py     # M3.2
├── test_parallel_execution.py      # M3.3
├── test_experiment_runner.py       # M4.1
├── test_statistical_analysis.py    # M4.2
├── test_lineage_tracking.py        # M4.3
├── test_search_integration.py      # M5.2
└── test_meta_evolution.py          # M5.3
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific module
uv run pytest tests/test_memory_manager.py -v

# With coverage
uv run pytest tests/ --cov=src/hvas_mini --cov-report=html

# Coverage report location
open htmlcov/index.html
```

### Test Requirements Per Branch

Each AGENT_TASK.md includes:
- Specific test cases to implement
- Expected behaviors to verify
- Edge cases to handle
- Integration test scenarios

**All branches must have tests BEFORE merging to master**

---

## Worktree Workflow

### Creating Worktrees

```bash
# M1 worktrees (already created)
git worktree add worktrees/agent-pool-infrastructure -b agent-pool-infrastructure
git worktree add worktrees/individual-memory-collections -b individual-memory-collections
git worktree add worktrees/fitness-tracking -b fitness-tracking

# M1 remaining (to create)
git worktree add worktrees/context-distribution -b context-distribution
git worktree add worktrees/pipeline-integration -b pipeline-integration

# M2 worktrees (create after M1 complete)
git worktree add worktrees/memory-compaction -b memory-compaction
git worktree add worktrees/memory-inheritance-reproduction -b memory-inheritance-reproduction
git worktree add worktrees/population-management -b population-management
git worktree add worktrees/evolution-pipeline-integration -b evolution-pipeline-integration

# ... M3, M4, M5 worktrees as needed
```

### Working in Worktrees

```bash
# Navigate to worktree
cd worktrees/agent-pool-infrastructure

# Check AGENT_TASK.md
cat ../../docs/feature-plans/agent-pool-infrastructure/AGENT_TASK.md

# Implement feature
# ... write code ...

# Run tests
uv run pytest tests/test_agent_pool.py -v

# Commit when tests pass
git add .
git commit -m "Implement AgentPool with selection strategies"

# Push branch
git push -u origin agent-pool-infrastructure

# Return to main workspace
cd ../..

# Merge when ready (after review)
git checkout master
git merge agent-pool-infrastructure
```

---

## Key Architectural Principles

### 1. Memory Inheritance > Prompt Mutation

**Old (Wrong)**: Mutate prompts, agents start with empty memories
**New (Correct)**: Stable prompts, agents inherit compacted memories

**Why**: Prompts bad at encoding nuance (that's why we need RAG)

### 2. Store ALL Experiences

**Old (Wrong)**: Only store if score >= 7.0
**New (Correct)**: Store everything, use weighted retrieval

**Why**: Low scores teach what NOT to do, need full dataset for compaction

### 3. Credibility Grants REACH, Not ACCESS

**Principle**: High-performers broadcast more widely, but everyone receives same quality context

**Why**: Prevents information monopolies, maintains diversity

### 4. Forced Diversity Requirement

**Implementation**: 20% of context from random low-performer

**Why**: Prevents echo chambers, enables cross-pollination

### 5. Lamarckian > Darwinian for AI

**Lamarckian**: Acquired knowledge (memories) inherited directly
**Darwinian**: Random mutations hope to find improvements

**For AI**: Lamarckian superior because knowledge can be directly transferred

---

## Success Metrics

### Fitness Improvement
- Average scores increase >0.5 points over 100 generations
- Best agents achieve consistent 8+ scores

### Emergent Specialization
- Domain-specific variance >1.0
- Agents develop distinct expertise areas

### Sustained Diversity
- Population std dev >0.5
- Multiple strategies persist

### Memory Effectiveness
- Retrieval count correlates with performance
- Inherited memories contribute to fitness

### Strategy Winner
- One configuration demonstrably outperforms others
- Statistical significance (p < 0.05)

---

## Failure Modes (What to Watch For)

### No Generational Improvement
- Later generations perform same as Generation 1
- **Indicates**: Memory inheritance doesn't help

### Memory Bloat
- Inherited memories add noise instead of signal
- **Indicates**: Compaction strategy failing

### Diversity Collapse
- All agents converge to single strategy
- **Indicates**: Selection pressure too strong

### No Specialization
- Agents don't develop distinct knowledge bases
- **Indicates**: Context distribution too homogeneous

### Strategy Equivalence
- All 3 strategies produce same results
- **Indicates**: Compaction approach doesn't matter

**If any occur**: Valuable negative results, adjust and retry

---

## Current Status

**Completed**:
- ✅ Proof of concept (single agent RAG memory)
- ✅ All M1 AGENT_TASK.md files created
- ✅ 3 M1 worktrees created (M1.1, M1.2, M1.3)
- ✅ README updated with future neural evolution feature
- ✅ Testing requirements documented

**Next Steps**:
1. Create remaining M1 worktrees (M1.4, M1.5)
2. Implement M1.1, M1.2, M1.3 in parallel
3. Implement M1.4 (after M1.1)
4. Implement M1.5 (integration)
5. Run 20-generation test
6. Proceed to M2

---

## Documentation Index

- **README.md**: Research overview, architecture, future enhancements
- **docs/technical.md**: Setup, configuration, architecture details
- **docs/NEXT_ITERATION.md**: Design decisions (from questionnaire)
- **docs/ANOTHER_ASPECT.MD**: Memory inheritance insight (critical)
- **docs/IMPLEMENTATION_ORCHESTRATION_REVISED.md**: Detailed milestone breakdown
- **docs/IMPLEMENTATION_PLAN_FINAL.md**: This file (complete plan)
- **docs/feature-plans/<branch>/AGENT_TASK.md**: Per-branch implementation guides

---

## Final Notes

### This is Research

- Goal: Test if memory inheritance works for AI agents
- Hypothesis: Knowledge accumulation across generations beats starting fresh
- Approach: Empirical testing with statistical analysis
- Outcome: May work, may not—both are valuable

### Emphasis on Testing

- **>90% coverage required**
- Tests written alongside implementation
- No merge without passing tests
- Tests documented in AGENT_TASK.md files

### Concurrent Development

- No fixed timeline (complete when done)
- Parallel work where possible
- Feature branches via worktrees
- Integration points between milestones

### What's Current vs Future

**Current (M1-M5)**:
- Memory inheritance
- Claude API for generation
- Fixed model architecture
- Knowledge evolution only

**Future (Phase 2)**:
- Neural circuit discovery
- LoRA adapter evolution
- Open-source models
- Weight evolution

**Start with current, explore future if promising**

---

**Ready to implement M1**
