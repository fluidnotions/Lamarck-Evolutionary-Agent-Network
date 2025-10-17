# HVAS Mini Implementation Orchestration Plan

## Overview

This document orchestrates the implementation of the evolutionary multi-agent system based on decisions in `NEXT_ITERATION.md`.

**Implementation Philosophy**:
- Break into milestones
- Create feature branches for parallel work
- Use worktrees for concurrent development
- AGENT_TASK.md files guide each branch
- Sequential dependencies respected

---

## Milestone Structure

```
M1: Core System (foundational)
├─ M2: Evolution (depends on M1)
│   └─ M3: Strategies (depends on M1, M2)
│       └─ M4: Experimentation (depends on M3)
└─ M5: Enhancement (parallel with M4)
```

---

## Milestone 1: Core System

**Goal**: Foundation for multi-agent populations with basic selection and context distribution

**Duration**: 3-4 days

### Feature Branches (Parallel Work Possible)

#### Branch 1.1: `agent-pool-infrastructure`
**Can start**: Immediately
**Depends on**: None

**Tasks**:
- Create `AgentPool` class
- Implement selection strategies (ε-greedy, fitness-weighted)
- Agent lifecycle management
- Helper methods (get_top_n, get_random_lower_half)

**Files**:
- `src/hvas_mini/agent_pool.py` (NEW)
- `tests/test_agent_pool.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/agent-pool-infrastructure/AGENT_TASK.md`

---

#### Branch 1.2: `individual-memory-collections`
**Can start**: Immediately (parallel with 1.1)
**Depends on**: None

**Tasks**:
- Modify `MemoryManager` to remove score threshold
- Support per-agent collections (`intro_agent_1_memories`)
- Implement weighted retrieval (`similarity × (score/10)`)
- Store all experiences with metadata

**Files**:
- `src/hvas_mini/memory.py` (MODIFY)
- `tests/test_memory.py` (MODIFY)

**AGENT_TASK.md location**: `docs/feature-plans/individual-memory-collections/AGENT_TASK.md`

---

#### Branch 1.3: `fitness-tracking`
**Can start**: Immediately (parallel with 1.1, 1.2)
**Depends on**: None

**Tasks**:
- Extend `BaseAgent` with fitness tracking
- Domain-specific performance tracking
- Rolling averages, specialization metrics
- Agent statistics API

**Files**:
- `src/hvas_mini/agents.py` (MODIFY)
- `tests/test_fitness_tracking.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/fitness-tracking/AGENT_TASK.md`

---

#### Branch 1.4: `context-distribution`
**Can start**: After 1.1 completes (needs AgentPool)
**Depends on**: Branch 1.1

**Tasks**:
- Create `ContextManager` class
- Implement 40/30/20/10 weighted distribution
- Cross-role context flow
- Forced diversity requirement

**Files**:
- `src/hvas_mini/context_manager.py` (NEW)
- `tests/test_context_distribution.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/context-distribution/AGENT_TASK.md`

---

#### Branch 1.5: `pipeline-integration`
**Can start**: After 1.1, 1.2, 1.3, 1.4 complete
**Depends on**: All M1 branches

**Tasks**:
- Update `HVASMiniPipeline` to use agent pools
- Integrate context distribution
- Update LangGraph workflow
- Remove old mechanisms (trust manager, parameter evolution)

**Files**:
- `src/hvas_mini/pipeline.py` (MODIFY)
- `src/hvas_mini/state.py` (MODIFY - simplify)
- `tests/test_pipeline_multi_agent.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/pipeline-integration/AGENT_TASK.md`

---

### Milestone 1 Completion Criteria

- [ ] Agent pools instantiate with 5 agents per role
- [ ] Each agent has isolated ChromaDB collection
- [ ] Fitness tracking operational (domain-specific)
- [ ] Context distribution working (40/30/20/10 verified)
- [ ] ε-greedy selection functional
- [ ] All M1 tests passing
- [ ] Can run single generation with population

---

## Milestone 2: Evolution

**Goal**: Enable population dynamics (add/remove agents, mutation, reproduction)

**Duration**: 2-3 days

**Starts**: After M1 complete

### Feature Branches (Parallel Work Possible)

#### Branch 2.1: `genome-mutation`
**Can start**: Immediately after M1
**Depends on**: M1 complete

**Tasks**:
- Implement mutation functions (add, modify, remove, reorder)
- 10% mutation rate
- Mutation types for prompt templates
- Mutation history tracking

**Files**:
- `src/hvas_mini/evolution/genome_mutation.py` (NEW)
- `tests/test_genome_mutation.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/genome-mutation/AGENT_TASK.md`

---

#### Branch 2.2: `sexual-reproduction`
**Can start**: Parallel with 2.1
**Depends on**: M1 complete

**Tasks**:
- Implement crossover logic (2 parents → 1 child)
- Parent selection for reproduction
- Offspring initialization
- Lineage tracking

**Files**:
- `src/hvas_mini/evolution/reproduction.py` (NEW)
- `tests/test_reproduction.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/sexual-reproduction/AGENT_TASK.md`

---

#### Branch 2.3: `population-management`
**Can start**: After 2.1, 2.2 complete
**Depends on**: Branches 2.1, 2.2

**Tasks**:
- Create `EvolutionManager` class
- Add agent triggers (population < min, diversity < threshold, stagnation)
- Remove agent triggers (fitness < 6.0, task_count ≥ 20)
- Evolution cycle (every 10 generations)

**Files**:
- `src/hvas_mini/evolution/evolution_manager.py` (NEW)
- `tests/test_population_management.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/population-management/AGENT_TASK.md`

---

#### Branch 2.4: `evolution-pipeline-integration`
**Can start**: After 2.3 completes
**Depends on**: All M2 branches

**Tasks**:
- Integrate EvolutionManager into pipeline
- Trigger evolution every 10 generations
- Log evolution events
- Update state tracking

**Files**:
- `src/hvas_mini/pipeline.py` (MODIFY)
- `tests/test_pipeline_evolution.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/evolution-pipeline-integration/AGENT_TASK.md`

---

### Milestone 2 Completion Criteria

- [ ] Genome mutation working (verified with examples)
- [ ] Sexual reproduction creates valid offspring
- [ ] Population adds agents when needed
- [ ] Population removes poor performers
- [ ] Evolution runs automatically every 10 gens
- [ ] All M2 tests passing
- [ ] Can run 30 generations with population evolution

---

## Milestone 3: Strategies

**Goal**: Implement multiple evolutionary strategies for A/B testing

**Duration**: 2-3 days

**Starts**: After M2 complete

### Feature Branches (Parallel Work Possible)

#### Branch 3.1: `strategy-abstraction`
**Can start**: Immediately after M2
**Depends on**: M2 complete

**Tasks**:
- Create `EvolutionaryStrategy` abstract class
- Strategy component interfaces (selection, context, evolution, memory)
- Strategy configuration system
- Strategy factory

**Files**:
- `src/hvas_mini/strategies/base.py` (NEW)
- `src/hvas_mini/strategies/__init__.py` (NEW)
- `tests/test_strategy_abstraction.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/strategy-abstraction/AGENT_TASK.md`

---

#### Branch 3.2: `baseline-strategies`
**Can start**: Parallel with 3.1
**Depends on**: M2 complete

**Tasks**:
- Implement Strategy A: Conservative Evolution
- Implement Strategy B: Aggressive Evolution
- Implement Strategy C: Balanced Adaptive
- Strategy-specific configurations

**Files**:
- `src/hvas_mini/strategies/conservative.py` (NEW)
- `src/hvas_mini/strategies/aggressive.py` (NEW)
- `src/hvas_mini/strategies/balanced.py` (NEW)
- `tests/test_baseline_strategies.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/baseline-strategies/AGENT_TASK.md`

---

#### Branch 3.3: `metric-collection`
**Can start**: Parallel with 3.1, 3.2
**Depends on**: M2 complete

**Tasks**:
- Create `MetricsCollector` class
- Track fitness trajectory, diversity, efficiency
- Specialization metrics
- Innovation rate calculation

**Files**:
- `src/hvas_mini/metrics/collector.py` (NEW)
- `src/hvas_mini/metrics/analysis.py` (NEW)
- `tests/test_metrics.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/metric-collection/AGENT_TASK.md`

---

#### Branch 3.4: `parallel-execution-framework`
**Can start**: After 3.1, 3.2, 3.3 complete
**Depends on**: All M3 branches

**Tasks**:
- Create experiment runner
- Parallel strategy execution
- Isolated populations per strategy
- Results aggregation

**Files**:
- `src/hvas_mini/experiments/runner.py` (NEW)
- `tests/test_experiment_runner.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/parallel-execution-framework/AGENT_TASK.md`

---

### Milestone 3 Completion Criteria

- [ ] Strategy abstraction layer working
- [ ] Three baseline strategies implemented
- [ ] Metrics collection operational
- [ ] Can run 3 strategies in parallel
- [ ] All M3 tests passing
- [ ] Metrics export to JSON/CSV

---

## Milestone 4: Experimentation

**Goal**: Run 100-generation experiments, compare strategies, select optimal

**Duration**: 1 week (mostly runtime)

**Starts**: After M3 complete

### Feature Branches (Parallel Work Possible)

#### Branch 4.1: `experiment-configuration`
**Can start**: Immediately after M3
**Depends on**: M3 complete

**Tasks**:
- Define 20 test topics (5 ML, 5 Python, 5 Web, 5 General)
- Experiment configuration files
- Topic domain classifier
- Reproducible seeds

**Files**:
- `experiments/topics.json` (NEW)
- `experiments/config.yaml` (NEW)
- `src/hvas_mini/utils/topic_classifier.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/experiment-configuration/AGENT_TASK.md`

---

#### Branch 4.2: `llm-evaluation`
**Can start**: Parallel with 4.1
**Depends on**: M3 complete

**Tasks**:
- Create `LLMEvaluator` class
- Claude-based scoring (engagement, clarity, depth, etc.)
- Criteria definitions
- Fallback to heuristic on error

**Files**:
- `src/hvas_mini/evaluation/llm_evaluator.py` (NEW)
- `tests/test_llm_evaluation.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/llm-evaluation/AGENT_TASK.md`

---

#### Branch 4.3: `statistical-analysis`
**Can start**: Parallel with 4.1, 4.2
**Depends on**: M3 complete

**Tasks**:
- Statistical comparison functions
- Hypothesis testing (t-tests, ANOVA)
- Visualization generation (matplotlib/plotly)
- Report generation

**Files**:
- `src/hvas_mini/analysis/statistics.py` (NEW)
- `src/hvas_mini/analysis/visualizations.py` (NEW)
- `tests/test_statistical_analysis.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/statistical-analysis/AGENT_TASK.md`

---

#### Branch 4.4: `experiment-execution`
**Can start**: After 4.1, 4.2 complete
**Depends on**: Branches 4.1, 4.2

**Tasks**:
- Run 100-gen experiments for each strategy
- Monitor progress
- Handle errors/resume
- Save checkpoints

**Files**:
- `experiments/run_baseline_experiments.py` (NEW)
- `experiments/results/` (directory)

**AGENT_TASK.md location**: `docs/feature-plans/experiment-execution/AGENT_TASK.md`

---

### Milestone 4 Completion Criteria

- [ ] 100 generations completed for Strategy A
- [ ] 100 generations completed for Strategy B
- [ ] 100 generations completed for Strategy C
- [ ] Statistical analysis complete
- [ ] Winning strategy identified
- [ ] Results report generated

---

## Milestone 5: Enhancement

**Goal**: Dashboard, search integration, production readiness

**Duration**: 1 week

**Starts**: Can run parallel with M4

### Feature Branches (Parallel Work Possible)

#### Branch 5.1: `streamlit-dashboard`
**Can start**: After M3 (can run parallel with M4)
**Depends on**: M3 complete

**Tasks**:
- Population fitness table
- Fitness/diversity charts
- Agent specialization heatmap
- Memory accumulation metrics
- Real-time generation monitoring

**Files**:
- `streamlit_app.py` (NEW)
- `src/hvas_mini/dashboard/` (NEW directory)

**AGENT_TASK.md location**: `docs/feature-plans/streamlit-dashboard/AGENT_TASK.md`

---

#### Branch 5.2: `internet-search-integration`
**Can start**: Parallel with 5.1
**Depends on**: M3 complete

**Tasks**:
- Search allocation strategies
- Tavily/Perplexity API integration
- ROI tracking
- Search result incorporation

**Files**:
- `src/hvas_mini/search/allocator.py` (NEW)
- `src/hvas_mini/search/provider.py` (NEW)
- `tests/test_search_integration.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/internet-search-integration/AGENT_TASK.md`

---

#### Branch 5.3: `meta-evolution`
**Can start**: Parallel with 5.1, 5.2
**Depends on**: M3 complete

**Tasks**:
- Strategy mutation
- Strategy crossover
- Meta-fitness tracking
- Strategy evolution cycle

**Files**:
- `src/hvas_mini/meta/strategy_evolution.py` (NEW)
- `tests/test_meta_evolution.py` (NEW)

**AGENT_TASK.md location**: `docs/feature-plans/meta-evolution/AGENT_TASK.md`

---

### Milestone 5 Completion Criteria

- [ ] Streamlit dashboard operational
- [ ] Search integration working
- [ ] Meta-evolution functional
- [ ] Production-ready deployment config
- [ ] Documentation complete

---

## Implementation Timeline

### Phase 1: Milestone 1 (Core System)
```
 1-2: Branches 1.1, 1.2, 1.3 (parallel)
 3:   Branch 1.4 (after 1.1)
 4:   Branch 1.5 (after all)
```

### Phase 2: Milestone 2 (Evolution)
```
 1-2: Branches 2.1, 2.2 (parallel)
 3:   Branch 2.3 (after 2.1, 2.2)
 4:   Branch 2.4 (after 2.3)
```

### Phase 3: Milestone 3 (Strategies)
```
 1-2: Branches 3.1, 3.2, 3.3 (parallel)
 3:   Branch 3.4 (after all)
```

### Phase 4: Milestone 4 + 5 (Parallel)
```
 1-2: M4 branches (4.1, 4.2, 4.3 parallel)
 3-5: M4.4 (experiments running)
 1-5: M5 branches (5.1, 5.2, 5.3 parallel with M4)
```

**Total Duration**: 4 weeks

---

## Worktree and Branch Creation

### Milestone 1 Worktrees

```bash
# From main repo
cd /home/justin/Documents/dev/workspaces/hvas-mini

# Create worktrees for M1
git worktree add worktrees/agent-pool-infrastructure -b agent-pool-infrastructure
git worktree add worktrees/individual-memory-collections -b individual-memory-collections
git worktree add worktrees/fitness-tracking -b fitness-tracking
git worktree add worktrees/context-distribution -b context-distribution
git worktree add worktrees/pipeline-integration -b pipeline-integration
```

### Milestone 2 Worktrees

```bash
git worktree add worktrees/genome-mutation -b genome-mutation
git worktree add worktrees/sexual-reproduction -b sexual-reproduction
git worktree add worktrees/population-management -b population-management
git worktree add worktrees/evolution-pipeline-integration -b evolution-pipeline-integration
```

### Milestone 3 Worktrees

```bash
git worktree add worktrees/strategy-abstraction -b strategy-abstraction
git worktree add worktrees/baseline-strategies -b baseline-strategies
git worktree add worktrees/metric-collection -b metric-collection
git worktree add worktrees/parallel-execution-framework -b parallel-execution-framework
```

### Milestone 4 Worktrees

```bash
git worktree add worktrees/experiment-configuration -b experiment-configuration
git worktree add worktrees/llm-evaluation -b llm-evaluation
git worktree add worktrees/statistical-analysis -b statistical-analysis
git worktree add worktrees/experiment-execution -b experiment-execution
```

### Milestone 5 Worktrees

```bash
git worktree add worktrees/streamlit-dashboard -b streamlit-dashboard
git worktree add worktrees/internet-search-integration -b internet-search-integration
git worktree add worktrees/meta-evolution -b meta-evolution
```

---

## AGENT_TASK.md Template

Each branch will have an `AGENT_TASK.md` in `docs/feature-plans/{branch-name}/` with this structure:

```markdown
# {Feature Name}

## Objective
[Clear statement of what this feature accomplishes]

## Context
[Why this feature is needed, how it fits into the system]

## Dependencies
- Requires: [List of branches that must complete first]
- Enables: [List of branches that depend on this]

## Implementation Tasks

### Task 1: [Name]
**File**: `path/to/file.py`
**Action**: CREATE/MODIFY/DELETE

**Requirements**:
- [ ] Requirement 1
- [ ] Requirement 2

**Implementation notes**:
[Detailed guidance]

### Task 2: [Name]
...

## Testing Strategy
[What tests to write, what to verify]

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Estimated Time
[Duration estimate]
```

---

## Next Steps

1. **Immediate**: Create M1 worktrees and AGENT_TASK.md files
2. **Phase 1**: Implement M1 branches (3 parallel, 2 sequential)
3. **Phase 2**: Implement M2 branches
4. **Phase 3**: Implement M3 branches
5. **Phase 4**: Run M4 experiments + build M5 enhancements

**Ready to start?** Begin with creating Milestone 1 worktrees and AGENT_TASK.md files.
