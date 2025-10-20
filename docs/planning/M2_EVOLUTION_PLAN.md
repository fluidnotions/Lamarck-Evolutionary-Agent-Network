# M2: Evolution & Reproduction Implementation Plan
**Date**: 2025-10-20
**Branch**: develop
**Status**: Planning Phase

---

## Overview

M2 implements **Step 8: EVOLVE** of the 8-step learning cycle:
- Selection: Choose best reasoners as parents
- Compaction: Forget unsuccessful patterns
- Reproduction: Create offspring with inherited patterns
- Population management: Maintain agent pools

**Goal**: Enable Lamarckian evolution where agents inherit their parents' BEST cognitive strategies.

---

## LangGraph Integration

**M2 uses a hybrid architecture:**

### LangGraph Components (Workflow Orchestration)
- **`_evolve_node()`** in PipelineV2 - LangGraph node that orchestrates Step 8 (EVOLVE)
- **Multi-generation loops** - Conditional edges to run N generations
- **`_should_continue_evolving()`** - Conditional edge for generation limits
- **Pool-based agent selection** - Generation nodes select agents from pools

### Python Utility Components (Pure Algorithms)
- **CompactionStrategy** - Abstract base class with 4 implementations
- **SelectionStrategy** - Abstract base class with 4 implementations
- **ReproductionStrategy** - Abstract base class with 2 implementations
- **AgentPool** - Data structure (called BY LangGraph nodes)

**Key Design Principle**: LangGraph handles workflow orchestration, Python classes handle algorithms.

### Subgraph Architecture (Reusable Evolution Layer)

M2 uses **subgraphs** to separate task logic from evolution logic:

- **Inner Subgraph** (Task Workflow): Pure task logic (blog, research, etc.) - knows nothing about evolution
- **Outer Wrapper** (Evolution): Reusable evolution logic - works with ANY task workflow
- **Composition**: Compile task workflow and use it as a node in evolution wrapper

This makes evolution a **reusable layer** you can attach to any existing LangGraph workflow!

See detailed architecture:
- **Subgraph design**: `docs/brainstorming/2025-10-20-M2-SUBGRAPH-ARCHITECTURE.md`
- **LangGraph integration**: `docs/brainstorming/2025-10-20-M2-LANGGRAPH-ARCHITECTURE.md`

---

## Architecture Components

### 1. Compaction Strategies (Feature: `m2-compaction`)

**Purpose**: Forget unsuccessful reasoning patterns

**Implementation**: `src/lean/compaction.py`

```python
class CompactionStrategy(ABC):
    """Base class for reasoning pattern compaction."""

    @abstractmethod
    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Compact patterns to max_size."""
        pass

class ScoreBasedCompaction(CompactionStrategy):
    """Keep highest-scoring patterns."""
    pass

class FrequencyBasedCompaction(CompactionStrategy):
    """Keep most-retrieved patterns."""
    pass

class DiversityPreservingCompaction(CompactionStrategy):
    """Keep diverse strategies (cluster-based)."""
    pass

class HybridCompaction(CompactionStrategy):
    """Combine score, frequency, and diversity."""
    pass
```

**Key Features**:
- Multiple strategies (score, frequency, diversity, hybrid)
- Configurable thresholds
- Metadata tracking (what was forgotten)
- Statistics (compaction rate, diversity metrics)

**Tests**: `tests/test_compaction.py`

---

### 2. Selection Algorithms (Feature: `m2-selection`)

**Purpose**: Choose best agents as parents for next generation

**Implementation**: `src/lean/selection.py`

```python
class SelectionStrategy(ABC):
    """Base class for parent selection."""

    @abstractmethod
    def select_parents(
        self,
        pool: AgentPool,
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List[BaseAgentV2]:
        """Select parents from pool."""
        pass

class TournamentSelection(SelectionStrategy):
    """Tournament selection (k agents compete)."""
    pass

class FitnessProportionateSelection(SelectionStrategy):
    """Roulette wheel selection by fitness."""
    pass

class RankBasedSelection(SelectionStrategy):
    """Select top N by fitness rank."""
    pass

class DiversityAwareSelection(SelectionStrategy):
    """Balance fitness and diversity."""
    pass
```

**Key Features**:
- Multiple selection strategies
- Elitism support (preserve best agents)
- Diversity preservation
- Tournament size configuration

**Tests**: `tests/test_selection.py`

---

### 3. Reproduction (Feature: `m2-reproduction`)

**Purpose**: Create offspring agents with inherited reasoning patterns

**Implementation**: `src/lean/reproduction.py`

```python
class ReproductionStrategy(ABC):
    """Base class for agent reproduction."""

    @abstractmethod
    def reproduce(
        self,
        parent1: BaseAgentV2,
        parent2: Optional[BaseAgentV2],
        compaction_strategy: CompactionStrategy,
        generation: int
    ) -> BaseAgentV2:
        """Create offspring from parent(s)."""
        pass

class AsexualReproduction(ReproductionStrategy):
    """Single parent → offspring (cloning with mutation)."""
    pass

class SexualReproduction(ReproductionStrategy):
    """Two parents → offspring (crossover + mutation)."""
    pass

def create_offspring(
    parent1: BaseAgentV2,
    parent2: Optional[BaseAgentV2],
    role: str,
    generation: int,
    compaction_strategy: CompactionStrategy,
    shared_rag: SharedRAG
) -> BaseAgentV2:
    """Factory function for creating offspring."""

    # 1. Get parent reasoning patterns
    parent1_patterns = parent1.reasoning_memory.get_all_reasoning()
    parent2_patterns = parent2.reasoning_memory.get_all_reasoning() if parent2 else []

    # 2. Compact (forget unsuccessful patterns)
    combined = parent1_patterns + parent2_patterns
    inherited = compaction_strategy.compact(
        patterns=combined,
        max_size=int(os.getenv('INHERITED_REASONING_SIZE', '100'))
    )

    # 3. Create child agent with inherited patterns
    child_id = f"{role}_gen{generation}_child{uuid.uuid4().hex[:6]}"
    child_memory = ReasoningMemory(
        collection_name=generate_reasoning_collection_name(role, child_id),
        inherited_reasoning=inherited
    )

    # 4. Create agent
    child = create_agent_by_role(
        role=role,
        agent_id=child_id,
        reasoning_memory=child_memory,
        shared_rag=shared_rag
    )

    return child
```

**Key Features**:
- Asexual reproduction (single parent)
- Sexual reproduction (two parents with crossover)
- Mutation rate (add noise to inherited patterns)
- Lineage tracking (parent → child relationships)

**Tests**: `tests/test_reproduction.py`

---

### 4. Agent Pools (Feature: `m2-agent-pools`)

**Purpose**: Manage populations of agents with fitness tracking

**Implementation**: `src/lean/agent_pool.py`

```python
class AgentPool:
    """Population of agents for a specific role."""

    def __init__(
        self,
        role: str,
        initial_agents: List[BaseAgentV2],
        max_size: int = 10,
        selection_strategy: SelectionStrategy = None,
        compaction_strategy: CompactionStrategy = None
    ):
        self.role = role
        self.agents = initial_agents
        self.max_size = max_size
        self.selection_strategy = selection_strategy
        self.compaction_strategy = compaction_strategy
        self.generation = 0
        self.history = []  # Track pool evolution

    def select_agent(self, strategy: str = "fitness_proportionate") -> BaseAgentV2:
        """Select agent for task execution."""
        pass

    def evolve_generation(
        self,
        reproduction_strategy: ReproductionStrategy,
        shared_rag: SharedRAG
    ):
        """Create next generation of agents."""

        # 1. Select parents
        parents = self.selection_strategy.select_parents(
            pool=self,
            num_parents=self.max_size // 2
        )

        # 2. Create offspring
        offspring = []
        for i in range(self.max_size):
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)] if len(parents) > 1 else None

            child = reproduction_strategy.reproduce(
                parent1=parent1,
                parent2=parent2,
                compaction_strategy=self.compaction_strategy,
                generation=self.generation + 1
            )
            offspring.append(child)

        # 3. Replace population (or merge with elitism)
        self.agents = offspring
        self.generation += 1

        # 4. Track history
        self.history.append({
            'generation': self.generation,
            'avg_fitness': self.avg_fitness(),
            'diversity': self.measure_diversity()
        })

    def get_top_n(self, n: int) -> List[BaseAgentV2]:
        """Get top N agents by fitness."""
        return sorted(self.agents, key=lambda a: a.avg_fitness(), reverse=True)[:n]

    def get_random_lower_half(self) -> BaseAgentV2:
        """Get random agent from lower half (for diversity)."""
        sorted_agents = sorted(self.agents, key=lambda a: a.avg_fitness())
        lower_half = sorted_agents[:len(sorted_agents)//2]
        return random.choice(lower_half)

    def avg_fitness(self) -> float:
        """Average fitness across pool."""
        return sum(a.avg_fitness() for a in self.agents) / len(self.agents)

    def measure_diversity(self) -> float:
        """Measure reasoning pattern diversity."""
        # Compare reasoning embeddings across agents
        pass

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        return {
            'role': self.role,
            'generation': self.generation,
            'size': len(self.agents),
            'avg_fitness': self.avg_fitness(),
            'fitness_range': (min(a.avg_fitness() for a in self.agents),
                            max(a.avg_fitness() for a in self.agents)),
            'diversity': self.measure_diversity()
        }
```

**Key Features**:
- Population management (fixed size)
- Selection for execution (which agent does task)
- Generational evolution (create next generation)
- Fitness tracking and statistics
- Diversity measurement
- History tracking (generation progression)

**Tests**: `tests/test_agent_pool.py`

---

## Integration with Pipeline V2

**Current Pipeline V2** (M1):
```python
# Steps 1-7 implemented
pipeline = PipelineV2()
state = await pipeline.generate(topic, generation_number=1)
# Reasoning patterns stored, but no evolution
```

**Enhanced Pipeline V2** (M2):
```python
# Add agent pools
pipeline = PipelineV2(
    intro_pool_size=5,
    body_pool_size=5,
    conclusion_pool_size=5,
    selection_strategy='tournament',
    compaction_strategy='hybrid'
)

# Run multiple generations with evolution
for gen in range(1, 21):
    # Execute tasks with current population
    state = await pipeline.generate(topic, generation_number=gen)

    # Every N generations, evolve pools
    if gen % 5 == 0:
        pipeline.evolve_pools(reproduction_strategy='sexual')
```

**New method in PipelineV2**:
```python
def evolve_pools(self, reproduction_strategy: str = 'sexual'):
    """Evolve all agent pools to next generation."""

    for role in ['intro', 'body', 'conclusion']:
        pool = self.agent_pools[role]

        # Trigger evolution
        pool.evolve_generation(
            reproduction_strategy=self.reproduction_strategies[reproduction_strategy],
            shared_rag=self.shared_rag
        )

        # Update pipeline's agent references
        self.agents[role] = pool.select_agent(strategy='best')
```

---

## Development Workflow

### Worktree Setup

```bash
# Current: main repo in develop branch
cd /home/justin/Documents/dev/workspaces/hvas-mini

# Create worktrees for parallel development
git worktree add ../lean-compaction -b m2-compaction develop
git worktree add ../lean-selection -b m2-selection develop
git worktree add ../lean-reproduction -b m2-reproduction develop
git worktree add ../lean-agent-pools -b m2-agent-pools develop

# Each team member (or parallel work) gets a worktree
```

### Development Order

**Phase 1: Python Utilities (Parallel - No LangGraph)**
- [ ] Worktree 1: Compaction strategies (`m2-compaction`)
  - Pure Python strategy classes
  - Unit tests in isolation
- [ ] Worktree 2: Selection algorithms (`m2-selection`)
  - Pure Python strategy classes
  - Unit tests with mock pools
- [ ] Worktree 3: Reproduction logic (`m2-reproduction`)
  - Pure Python strategy classes
  - Unit tests with mock agents

**Phase 2: Data Structures (Depends on Phase 1)**
- [ ] Worktree 4: Agent pools (`m2-agent-pools`)
  - AgentPool class using strategy pattern
  - Integration tests with real strategies
  - Merge worktrees 1-3 first before implementing

**Phase 3: Workflow Refactoring (Subgraph Separation)**
- [ ] Extract task workflow from PipelineV2
  - Create `src/lean/workflows/blog_workflow.py`
  - Move intro/body/conclusion nodes
  - Remove evolution logic (pure task execution)
  - Update to use `state['selected_agents']`

**Phase 4: Evolution Wrapper (Reusable Layer)**
- [ ] Create `src/lean/evolution/wrapper.py`
  - EvolutionWrapper class
  - select_agents_node
  - execute_task_node (subgraph!)
  - evaluate_node
  - store_node
  - evolve_node
  - Accepts ANY task workflow as subgraph

**Phase 5: Integration (Update PipelineV2)**
- [ ] Update `src/lean/pipeline_v2.py`
  - Use EvolutionWrapper internally
  - Maintain backward compatible API
  - Pass blog_workflow as subgraph
- [ ] Update `src/lean/state.py`
  - Add `selected_agents` field to BlogState

**Phase 6: Testing**
- [ ] Unit tests for EvolutionWrapper
- [ ] Integration tests with blog workflow
- [ ] Integration tests with mock workflows
- [ ] Backward compatibility tests

**Phase 7: Validation**
- [ ] Run 20-generation experiment
- [ ] Compare evolved vs. non-evolved agents
- [ ] Measure fitness improvement over generations
- [ ] Analyze compaction effectiveness
- [ ] Document best practices

---

## Feature Specifications

### Feature: `m2-compaction`

**Files to create**:
- `src/lean/compaction.py` (base + 4 strategies)
- `tests/test_compaction.py` (unit tests)
- `examples/compaction_demo.py` (demonstration)

**Tasks**:
1. Implement `CompactionStrategy` base class
2. Implement 4 strategies (score, frequency, diversity, hybrid)
3. Add configuration options
4. Test with sample reasoning patterns
5. Benchmark compaction effectiveness

**Acceptance criteria**:
- All tests passing
- Can reduce 100 patterns to 20 with each strategy
- Strategies produce different results (diversity preserved)
- Performance: <100ms for 1000 patterns

---

### Feature: `m2-selection`

**Files to create**:
- `src/lean/selection.py` (base + 4 strategies)
- `tests/test_selection.py` (unit tests)
- `examples/selection_demo.py` (demonstration)

**Tasks**:
1. Implement `SelectionStrategy` base class
2. Implement 4 strategies (tournament, proportionate, rank, diversity-aware)
3. Add elitism support
4. Test with mock agent pools
5. Validate diversity preservation

**Acceptance criteria**:
- All tests passing
- Higher fitness agents selected more often (but not always)
- Diversity maintained across selections
- Configurable parameters (tournament size, etc.)

---

### Feature: `m2-reproduction`

**Files to create**:
- `src/lean/reproduction.py` (strategies + factory)
- `tests/test_reproduction.py` (unit tests)
- `examples/reproduction_demo.py` (demonstration)

**Tasks**:
1. Implement `ReproductionStrategy` base class
2. Implement asexual reproduction
3. Implement sexual reproduction (crossover)
4. Add mutation support
5. Track lineage metadata

**Acceptance criteria**:
- All tests passing
- Offspring have inherited reasoning patterns
- Lineage traceable (parent → child)
- Compaction integrated (inherited patterns pruned)
- Works with existing BaseAgentV2

---

### Feature: `m2-agent-pools`

**Files to create**:
- `src/lean/agent_pool.py` (AgentPool class)
- `tests/test_agent_pool.py` (unit tests)
- `examples/agent_pool_demo.py` (demonstration)

**Tasks**:
1. Implement `AgentPool` class
2. Add population management
3. Integrate selection strategies
4. Implement generational evolution
5. Add diversity measurement

**Acceptance criteria**:
- All tests passing
- Can manage pool of 5-10 agents
- Evolution produces next generation
- Fitness trends upward over generations
- Statistics and history tracking

---

## Testing Strategy

### Unit Tests (Per Feature)
- Test each strategy in isolation
- Mock dependencies (agents, patterns)
- Fast execution (<1s per test)

### Integration Tests (Cross-Feature)
- Test compaction → reproduction
- Test selection → reproduction
- Test agent pool evolution
- Slower execution (~5s per test)

### System Tests (Full Pipeline)
- Run 20-generation experiment
- Measure fitness improvement
- Validate reasoning inheritance
- Long execution (~5-10 minutes)

---

## Configuration

### Environment Variables (New)

```bash
# Compaction
COMPACTION_STRATEGY=hybrid                # score, frequency, diversity, hybrid
COMPACTION_THRESHOLD=0.5                  # Prune bottom 50%
INHERITED_REASONING_SIZE=100              # Max inherited patterns

# Selection
SELECTION_STRATEGY=tournament             # tournament, proportionate, rank, diversity
TOURNAMENT_SIZE=3                         # For tournament selection
ELITISM_COUNT=1                           # Best agents preserved

# Reproduction
REPRODUCTION_STRATEGY=sexual              # asexual, sexual
MUTATION_RATE=0.1                         # 10% chance of pattern mutation
CROSSOVER_RATE=0.5                        # 50% patterns from each parent

# Agent Pools
POOL_SIZE=5                               # Agents per role
EVOLUTION_FREQUENCY=5                     # Evolve every N generations
```

---

## Success Metrics

### After 20 Generations:

1. **Fitness improvement**: +2 points average score
2. **Compaction effectiveness**: 100 patterns → 30 (70% reduction)
3. **Diversity maintained**: ≥3 distinct reasoning clusters
4. **Inheritance validated**: Children outperform random initialization
5. **Performance**: <30s per generation with 5-agent pools

---

## Timeline Estimate

**Parallel development (4 worktrees)**:
- Week 1: Core implementations (compaction, selection, reproduction, pools)
- Week 2: Unit tests + integration
- Week 3: Pipeline integration + system tests
- Week 4: Validation + documentation

**Total: 4 weeks** (with parallel development)
**Sequential: 8 weeks** (one person, one feature at a time)

---

## Next Steps

1. ✅ Create develop branch
2. ✅ Create M2 plan document
3. ⏳ Set up 4 worktrees
4. ⏳ Assign features (or work in parallel)
5. ⏳ Implement core components
6. ⏳ Integration and testing
7. ⏳ Validation and documentation

---

## References

- M1 implementation: `src/lean/base_agent_v2.py`, `pipeline_v2.py`
- MIGRATION_GUIDE.md: Step 7 (compaction strategies)
- Session summaries: `docs/brainstorming/2025-10-20-*`

---

**Status**: Ready to begin implementation. Develop branch created. Waiting for worktree setup.
