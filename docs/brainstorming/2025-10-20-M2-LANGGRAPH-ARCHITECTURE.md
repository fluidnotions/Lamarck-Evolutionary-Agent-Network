# M2 LangGraph Architecture
**Date**: 2025-10-20
**Question**: Should M2 features use LangGraph where possible?
**Answer**: YES - Hybrid approach with LangGraph orchestration + Python utilities

---

## Architecture Decision

### LangGraph Components (Workflow Orchestration)

**1. Evolution Node in PipelineV2**
```python
# In pipeline_v2.py
async def _evolve_node(self, state: BlogState) -> BlogState:
    """Step 8: EVOLVE - Create next generation (LangGraph node)."""

    # Orchestrates the evolution cycle using Python utilities
    for role in ['intro', 'body', 'conclusion']:
        pool = self.pools[role]

        # Call Python utilities in sequence
        pool.evolve_generation(
            reproduction_strategy=self.reproduction_strategy,
            shared_rag=self.shared_rag
        )

    state['generation_number'] += 1
    return state
```

**Why LangGraph?**
- Integrates with existing PipelineV2 workflow
- Enables checkpointing/persistence
- Allows conditional evolution (e.g., "evolve every 5 generations")
- Part of the blog generation workflow

**2. Multi-Generation Workflow**
```python
# In pipeline_v2.py
def create_evolution_workflow(self) -> StateGraph:
    """Create workflow that runs multiple generations."""

    workflow = StateGraph(BlogState)

    # Generation workflow
    workflow.add_node("intro", self._intro_node)
    workflow.add_node("body", self._body_node)
    workflow.add_node("conclusion", self._conclusion_node)
    workflow.add_node("evaluate", self._evaluate_node)
    workflow.add_node("evolve", self._evolve_node)  # NEW: Evolution node

    # Flow: intro → body → conclusion → evaluate → evolve → (repeat or end)
    workflow.add_edge("intro", "body")
    workflow.add_edge("body", "conclusion")
    workflow.add_edge("conclusion", "evaluate")
    workflow.add_edge("evaluate", "evolve")

    # Conditional: evolve → intro (next gen) OR END (if max_generations)
    workflow.add_conditional_edges(
        "evolve",
        self._should_continue_evolving,
        {
            "continue": "intro",  # Next generation
            "end": END
        }
    )

    workflow.set_entry_point("intro")
    return workflow.compile(checkpointer=self.checkpointer)
```

**Why LangGraph?**
- Conditional loops (run N generations)
- Checkpoint evolution state
- Visualize evolution progress
- Pause/resume experiments

**3. Pool-Based Agent Selection**
```python
# In pipeline_v2.py
async def _intro_node(self, state: BlogState) -> BlogState:
    """Intro agent node - now selects from pool."""

    # Select agent from pool (using Python utility)
    agent = self.intro_pool.select_agent(strategy="fitness_proportionate")

    # Rest of node logic stays the same
    reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(...)
    # ... generate content ...

    return state
```

**Why LangGraph?**
- Agent selection is part of generation workflow
- State management for which agent was selected
- Tracking lineage in state

---

### Python Utility Components (Pure Algorithms)

**1. CompactionStrategy (Pure Python ABC)**
```python
# In src/lean/compaction.py
from abc import ABC, abstractmethod
from typing import List, Dict

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
    """Keep top N by score."""

    def compact(self, patterns, max_size, metadata=None):
        return sorted(patterns, key=lambda p: p['score'], reverse=True)[:max_size]

class FrequencyBasedCompaction(CompactionStrategy):
    """Keep most-used patterns."""

    def compact(self, patterns, max_size, metadata=None):
        return sorted(
            patterns,
            key=lambda p: p['retrieval_count'] * p['score'],
            reverse=True
        )[:max_size]
```

**Why NOT LangGraph?**
- Pure algorithmic logic
- No state dependencies
- Needs to be swappable (strategy pattern)
- Called by multiple contexts (reproduction, agent pools)
- Easier to test in isolation

**2. SelectionStrategy (Pure Python ABC)**
```python
# In src/lean/selection.py
class SelectionStrategy(ABC):
    """Base class for parent selection."""

    @abstractmethod
    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents from pool."""
        pass

class TournamentSelection(SelectionStrategy):
    """k agents compete, best wins."""

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def select_parents(self, pool, num_parents, metadata=None):
        parents = []
        for _ in range(num_parents):
            # Run tournament
            competitors = random.sample(pool.agents, self.tournament_size)
            winner = max(competitors, key=lambda a: a.avg_fitness())
            parents.append(winner)
        return parents
```

**Why NOT LangGraph?**
- Stateless selection algorithm
- Called by AgentPool.evolve_generation()
- Needs to be configurable/swappable
- No workflow dependencies

**3. ReproductionStrategy (Pure Python ABC)**
```python
# In src/lean/reproduction.py
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

class SexualReproduction(ReproductionStrategy):
    """Two parents → offspring with crossover."""

    def reproduce(self, parent1, parent2, compaction_strategy, generation):
        # Get patterns from both parents
        p1_patterns = parent1.reasoning_memory.get_all_reasoning()
        p2_patterns = parent2.reasoning_memory.get_all_reasoning()

        # Compact combined patterns
        combined = p1_patterns + p2_patterns
        inherited = compaction_strategy.compact(combined, max_size=100)

        # Create child agent with inherited patterns
        return create_offspring(parent1, parent2, inherited, generation)
```

**Why NOT LangGraph?**
- Pure data transformation
- Called by AgentPool.evolve_generation()
- No state dependencies
- Easier to test with mocks

---

### Hybrid Component (AgentPool)

**AgentPool: Data structure with methods called by LangGraph**
```python
# In src/lean/agent_pool.py
class AgentPool:
    """Population of agents for a specific role.

    This is a data structure that manages agent populations.
    It gets called BY LangGraph nodes, but is not a node itself.
    """

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
        self.selection_strategy = selection_strategy  # Python utility
        self.compaction_strategy = compaction_strategy  # Python utility
        self.generation = 0
        self.history = []

    def select_agent(self, strategy: str = "fitness_proportionate") -> BaseAgentV2:
        """Select agent for task execution.

        Called by LangGraph nodes (_intro_node, _body_node, etc.)
        """
        if strategy == "fitness_proportionate":
            # Roulette wheel selection
            total_fitness = sum(a.avg_fitness() for a in self.agents)
            pick = random.uniform(0, total_fitness)
            current = 0
            for agent in self.agents:
                current += agent.avg_fitness()
                if current >= pick:
                    return agent
        return self.agents[0]

    def evolve_generation(
        self,
        reproduction_strategy: ReproductionStrategy,  # Python utility
        shared_rag: SharedRAG
    ):
        """Create next generation of agents.

        Called by LangGraph _evolve_node().
        Uses Python utility strategies for algorithms.
        """
        # 1. Select parents (using Python utility)
        parents = self.selection_strategy.select_parents(
            pool=self,
            num_parents=self.max_size // 2
        )

        # 2. Create offspring (using Python utility)
        offspring = []
        for i in range(self.max_size):
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)] if len(parents) > 1 else None

            child = reproduction_strategy.reproduce(
                parent1=parent1,
                parent2=parent2,
                compaction_strategy=self.compaction_strategy,  # Python utility
                generation=self.generation + 1
            )
            offspring.append(child)

        # 3. Replace population
        self.agents = offspring
        self.generation += 1

        # 4. Track history
        self.history.append({
            'generation': self.generation,
            'avg_fitness': self.avg_fitness(),
            'diversity': self.measure_diversity()
        })
```

**Why Hybrid?**
- **Data structure**: Manages agent populations
- **Called by LangGraph**: `_evolve_node()` calls `pool.evolve_generation()`
- **Uses Python utilities**: Delegates to strategy objects for algorithms
- **Stateful**: Tracks generations, history

---

## Updated PipelineV2 Architecture

```python
# In pipeline_v2.py
class PipelineV2:
    """Pipeline V2 with agent pools and evolution."""

    def __init__(
        self,
        reasoning_dir: str = "./data/reasoning",
        shared_rag_dir: str = "./data/shared_rag",
        pool_size: int = 5,
        evolution_frequency: int = 5  # Evolve every N generations
    ):
        # Create shared RAG
        self.shared_rag = SharedRAG(shared_rag_dir)

        # Create Python utility strategies
        self.compaction_strategy = HybridCompaction()  # Python utility
        self.selection_strategy = TournamentSelection(tournament_size=3)  # Python utility
        self.reproduction_strategy = SexualReproduction()  # Python utility

        # Create agent pools (Hybrid data structures)
        agents = create_agents_v2(shared_rag=self.shared_rag)
        self.pools = {
            'intro': AgentPool(
                role='intro',
                initial_agents=[agents['intro']],
                max_size=pool_size,
                selection_strategy=self.selection_strategy,
                compaction_strategy=self.compaction_strategy
            ),
            'body': AgentPool(
                role='body',
                initial_agents=[agents['body']],
                max_size=pool_size,
                selection_strategy=self.selection_strategy,
                compaction_strategy=self.compaction_strategy
            ),
            'conclusion': AgentPool(
                role='conclusion',
                initial_agents=[agents['conclusion']],
                max_size=pool_size,
                selection_strategy=self.selection_strategy,
                compaction_strategy=self.compaction_strategy
            )
        }

        self.context_manager = ContextManager(agents)
        self.evaluator = ContentEvaluator()
        self.checkpointer = MemorySaver()
        self.evolution_frequency = evolution_frequency

    # LangGraph nodes
    async def _intro_node(self, state: BlogState) -> BlogState:
        """Generate intro (LangGraph node)."""
        # Select agent from pool
        agent = self.pools['intro'].select_agent(strategy="fitness_proportionate")

        # Steps 2-5 of learning cycle
        reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(...)
        domain_knowledge = agent.shared_rag.retrieve(...)
        reasoning_context = self.context_manager.assemble_context(...)
        result = agent.generate_with_reasoning(...)

        agent.prepare_reasoning_storage(...)
        state['intro'] = result['final_output']
        state['intro_reasoning'] = result['reasoning']
        return state

    async def _body_node(self, state: BlogState) -> BlogState:
        """Generate body (LangGraph node)."""
        agent = self.pools['body'].select_agent(strategy="fitness_proportionate")
        # ... same pattern ...
        return state

    async def _conclusion_node(self, state: BlogState) -> BlogState:
        """Generate conclusion (LangGraph node)."""
        agent = self.pools['conclusion'].select_agent(strategy="fitness_proportionate")
        # ... same pattern ...
        return state

    async def _evaluate_node(self, state: BlogState) -> BlogState:
        """Evaluate and store (LangGraph node)."""
        # Score content
        scores = self.evaluator.evaluate(state)

        # Store reasoning + outputs for each agent
        for role in ['intro', 'body', 'conclusion']:
            pool = self.pools[role]
            agent = pool.agents[0]  # Get the agent that just executed
            agent.store_reasoning_and_output(score=scores[role])

        state['scores'] = scores
        return state

    async def _evolve_node(self, state: BlogState) -> BlogState:
        """Step 8: EVOLVE pools (LangGraph node)."""
        gen = state.get('generation_number', 0)

        # Only evolve every N generations
        if gen > 0 and gen % self.evolution_frequency == 0:
            for role in ['intro', 'body', 'conclusion']:
                pool = self.pools[role]

                # Evolve pool (calls Python utilities)
                pool.evolve_generation(
                    reproduction_strategy=self.reproduction_strategy,
                    shared_rag=self.shared_rag
                )

            state['evolved'] = True

        state['generation_number'] = gen + 1
        return state

    def _should_continue_evolving(self, state: BlogState) -> str:
        """Conditional edge: continue or end? (LangGraph decision)."""
        max_generations = int(os.getenv('MAX_GENERATIONS', '20'))
        if state.get('generation_number', 0) < max_generations:
            return "continue"
        return "end"

    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow with evolution."""
        workflow = StateGraph(BlogState)

        # Add nodes
        workflow.add_node("intro", self._intro_node)
        workflow.add_node("body", self._body_node)
        workflow.add_node("conclusion", self._conclusion_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("evolve", self._evolve_node)

        # Add edges
        workflow.add_edge("intro", "body")
        workflow.add_edge("body", "conclusion")
        workflow.add_edge("conclusion", "evaluate")
        workflow.add_edge("evaluate", "evolve")

        # Conditional loop
        workflow.add_conditional_edges(
            "evolve",
            self._should_continue_evolving,
            {
                "continue": "intro",
                "end": END
            }
        )

        workflow.set_entry_point("intro")
        return workflow.compile(checkpointer=self.checkpointer)
```

---

## Summary: What Uses LangGraph?

### ✅ Use LangGraph For:
1. **Workflow orchestration** - `_intro_node`, `_body_node`, `_conclusion_node`, `_evaluate_node`, `_evolve_node`
2. **Multi-generation loops** - Conditional edges to run N generations
3. **State management** - BlogState with generation tracking
4. **Checkpointing** - Save/resume evolution experiments
5. **Agent selection** - Choosing which agent from pool to execute (inside nodes)

### ❌ Do NOT Use LangGraph For:
1. **Compaction algorithms** - Pure Python strategy classes
2. **Selection algorithms** - Pure Python strategy classes
3. **Reproduction logic** - Pure Python strategy classes
4. **Fitness calculations** - Methods on BaseAgentV2 and AgentPool
5. **Diversity measurement** - Utility functions

### 🔄 Hybrid (Called BY LangGraph):
1. **AgentPool** - Data structure with methods called by `_evolve_node`
2. **ContextManager** - Already exists, called by generation nodes
3. **ContentEvaluator** - Already exists, called by `_evaluate_node`

---

## Benefits of This Architecture

**1. Separation of Concerns**
- LangGraph handles workflow/orchestration
- Python classes handle algorithms
- Clear boundaries between "what to do" and "how to do it"

**2. Testability**
- Test compaction strategies in isolation (pure functions)
- Test LangGraph workflow with mock strategies
- Integration tests with real components

**3. Flexibility**
- Swap compaction strategies without changing workflow
- Add new selection algorithms by implementing interface
- Configure different strategies per experiment

**4. LangGraph Benefits**
- Checkpoint evolution state
- Visualize workflow graph
- Pause/resume experiments
- Conditional evolution (e.g., "only evolve if diversity < threshold")

**5. Code Organization**
- Clear module boundaries
- Strategy pattern for algorithms
- LangGraph for state machines

---

## Implementation Order

**Phase 1: Python Utilities (Can develop in parallel)**
1. `src/lean/compaction.py` - CompactionStrategy + 4 implementations
2. `src/lean/selection.py` - SelectionStrategy + 4 implementations
3. `src/lean/reproduction.py` - ReproductionStrategy + 2 implementations

**Phase 2: Hybrid Component (Depends on Phase 1)**
4. `src/lean/agent_pool.py` - AgentPool class using strategies

**Phase 3: LangGraph Integration (Depends on Phase 2)**
5. Update `src/lean/pipeline_v2.py`:
   - Add `_evolve_node()`
   - Add `_should_continue_evolving()`
   - Update `__init__()` to create pools
   - Update generation nodes to use `pool.select_agent()`
   - Add conditional edge for multi-generation loop

**Phase 4: Testing**
6. Unit tests for each strategy (Python)
7. Integration tests for AgentPool (Hybrid)
8. Workflow tests for evolution cycle (LangGraph)
9. End-to-end 20-generation experiment

---

## Configuration

```bash
# In .env
POOL_SIZE=5                      # Number of agents per pool
EVOLUTION_FREQUENCY=5            # Evolve every N generations
MAX_GENERATIONS=20               # Total generations to run
COMPACTION_STRATEGY=hybrid       # score|frequency|diversity|hybrid
SELECTION_STRATEGY=tournament    # tournament|proportionate|rank|diversity
REPRODUCTION_STRATEGY=sexual     # asexual|sexual
TOURNAMENT_SIZE=3                # For tournament selection
INHERITED_REASONING_SIZE=100     # Max patterns inherited
```

---

## Conclusion

**YES, M2 uses LangGraph where appropriate:**
- Evolution workflow is a LangGraph node (`_evolve_node`)
- Multi-generation loops use LangGraph conditional edges
- Agent selection happens inside LangGraph nodes
- State management through BlogState

**But algorithmic components are pure Python:**
- Compaction, selection, reproduction strategies
- These are called BY LangGraph nodes, not nodes themselves
- Follows strategy pattern for flexibility

This hybrid approach gives us the best of both:
- LangGraph for workflow orchestration and state management
- Python classes for testable, swappable algorithms
