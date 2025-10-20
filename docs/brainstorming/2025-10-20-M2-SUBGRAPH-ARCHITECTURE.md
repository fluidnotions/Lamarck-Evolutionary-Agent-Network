# M2 Subgraph Architecture: Evolution as a Reusable Wrapper
**Date**: 2025-10-20
**Question**: Can we make M2 evolution a reusable layer on top of any LangGraph workflow?
**Answer**: YES - Use LangGraph subgraphs to separate task logic from evolution logic

---

## LangGraph Subgraph Capabilities

### 1. Compiled Graphs as Nodes
```python
# Create a workflow
inner_graph = StateGraph(MyState)
inner_graph.add_node("step1", node1)
inner_graph.add_node("step2", node2)
inner_graph.add_edge("step1", "step2")

# Compile it
compiled_inner = inner_graph.compile()

# Use it as a node in another graph!
outer_graph = StateGraph(MyState)
outer_graph.add_node("my_subgraph", compiled_inner)  # Subgraph as node
outer_graph.add_node("other_step", other_node)
outer_graph.add_edge("my_subgraph", "other_step")
```

### 2. Invoke vs Direct Call
```python
# Option 1: Direct call (pass through state)
outer_graph.add_node("subgraph", compiled_inner)

# Option 2: Wrapper function (transform state if needed)
def subgraph_wrapper(state):
    result = compiled_inner.invoke(state)
    # Transform result if needed
    return result

outer_graph.add_node("subgraph", subgraph_wrapper)
```

---

## M2 Evolution as Reusable Wrapper

### Architecture Overview

```
┌─────────────────────────────────────────┐
│ Evolution Wrapper (Outer Graph)         │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ select_agents_node              │   │ ← Select from pools
│  └──────────┬──────────────────────┘   │
│             ↓                           │
│  ┌─────────────────────────────────┐   │
│  │ TASK SUBGRAPH (Inner Graph)     │   │ ← ANY workflow
│  │                                 │   │
│  │  intro → body → conclusion      │   │
│  │  (or research → analyze → etc)  │   │
│  └──────────┬──────────────────────┘   │
│             ↓                           │
│  ┌─────────────────────────────────┐   │
│  │ evaluate_node                   │   │ ← Score outputs
│  └──────────┬──────────────────────┘   │
│             ↓                           │
│  ┌─────────────────────────────────┐   │
│  │ store_node                      │   │ ← Store reasoning
│  └──────────┬──────────────────────┘   │
│             ↓                           │
│  ┌─────────────────────────────────┐   │
│  │ evolve_node                     │   │ ← Create offspring
│  └──────────┬──────────────────────┘   │
│             ↓                           │
│       Continue or END?                  │
│                                         │
└─────────────────────────────────────────┘
```

### Key Insight

**Separation of Concerns:**
- **Task Subgraph**: What the agents DO (blog, research, analysis, etc.)
- **Evolution Wrapper**: How the agents LEARN and EVOLVE

The evolution wrapper doesn't care what task is being performed - it just:
1. Selects agents from pools
2. Runs the task (subgraph)
3. Evaluates results
4. Stores memories
5. Evolves pools

---

## Implementation: Reusable Evolution Wrapper

### Step 1: Define Task Workflow (Subgraph)

```python
# In src/lean/workflows/blog_workflow.py
from langgraph.graph import StateGraph, END
from lean.state import BlogState

def create_blog_workflow() -> StateGraph:
    """Create blog writing workflow (TASK SUBGRAPH).

    This workflow knows NOTHING about evolution.
    It just writes blogs using whatever agents it's given.
    """

    workflow = StateGraph(BlogState)

    # Task-specific nodes
    async def intro_node(state: BlogState) -> BlogState:
        # Agent is already selected by evolution wrapper
        agent = state['selected_agents']['intro']

        # Just execute the task
        reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(...)
        domain_knowledge = agent.shared_rag.retrieve(...)
        result = agent.generate_with_reasoning(...)

        state['intro'] = result['final_output']
        state['intro_reasoning'] = result['reasoning']
        return state

    async def body_node(state: BlogState) -> BlogState:
        agent = state['selected_agents']['body']
        # ... same pattern ...
        return state

    async def conclusion_node(state: BlogState) -> BlogState:
        agent = state['selected_agents']['conclusion']
        # ... same pattern ...
        return state

    # Build workflow
    workflow.add_node("intro", intro_node)
    workflow.add_node("body", body_node)
    workflow.add_node("conclusion", conclusion_node)

    workflow.add_edge("intro", "body")
    workflow.add_edge("body", "conclusion")
    workflow.add_edge("conclusion", END)

    workflow.set_entry_point("intro")

    return workflow  # Return uncompiled (we'll compile in wrapper)
```

**Key: Task workflow is PURE** - no evolution logic, just task execution.

---

### Step 2: Create Evolution Wrapper (Outer Graph)

```python
# In src/lean/evolution/wrapper.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Callable, Dict, List
from lean.state import BlogState
from lean.agent_pool import AgentPool
from lean.evaluation import ContentEvaluator

class EvolutionWrapper:
    """Reusable evolution wrapper for ANY LangGraph task workflow.

    Wraps any task workflow with:
    - Agent pool management
    - Agent selection
    - Evaluation
    - Memory storage
    - Generational evolution
    """

    def __init__(
        self,
        task_workflow: StateGraph,  # The inner task graph
        pools: Dict[str, AgentPool],  # Agent pools
        evaluator: ContentEvaluator,
        evolution_frequency: int = 5,
        max_generations: int = 20
    ):
        """Initialize evolution wrapper.

        Args:
            task_workflow: Uncompiled StateGraph for the task
            pools: Agent pools (role → AgentPool)
            evaluator: Content evaluator
            evolution_frequency: Evolve every N generations
            max_generations: Total generations to run
        """
        self.task_workflow_compiled = task_workflow.compile()  # Compile task subgraph
        self.pools = pools
        self.evaluator = evaluator
        self.evolution_frequency = evolution_frequency
        self.max_generations = max_generations
        self.checkpointer = MemorySaver()

    def _select_agents_node(self, state: BlogState) -> BlogState:
        """Select agents from pools (EVOLUTION LOGIC)."""
        selected = {}
        for role, pool in self.pools.items():
            agent = pool.select_agent(strategy="fitness_proportionate")
            selected[role] = agent

        state['selected_agents'] = selected
        return state

    def _execute_task_node(self, state: BlogState) -> BlogState:
        """Execute the task subgraph (TASK LOGIC - delegated)."""
        # This calls the ENTIRE task workflow as a subgraph!
        result_state = self.task_workflow_compiled.invoke(state)
        return result_state

    def _evaluate_node(self, state: BlogState) -> BlogState:
        """Evaluate outputs (EVOLUTION LOGIC)."""
        scores = self.evaluator.evaluate(state)
        state['scores'] = scores
        return state

    def _store_node(self, state: BlogState) -> BlogState:
        """Store reasoning and outputs (EVOLUTION LOGIC)."""
        for role, agent in state['selected_agents'].items():
            score = state['scores'].get(role, 0.0)
            agent.store_reasoning_and_output(score=score)
        return state

    def _evolve_node(self, state: BlogState) -> BlogState:
        """Evolve agent pools (EVOLUTION LOGIC)."""
        gen = state.get('generation_number', 0)

        # Only evolve every N generations
        if gen > 0 and gen % self.evolution_frequency == 0:
            for pool in self.pools.values():
                pool.evolve_generation(
                    reproduction_strategy=self.reproduction_strategy,
                    shared_rag=self.shared_rag
                )
            state['evolved'] = True

        state['generation_number'] = gen + 1
        return state

    def _should_continue(self, state: BlogState) -> str:
        """Conditional edge: continue or end?"""
        if state.get('generation_number', 0) < self.max_generations:
            return "continue"
        return "end"

    def create_workflow(self) -> StateGraph:
        """Create the evolution wrapper workflow."""
        workflow = StateGraph(BlogState)

        # Evolution nodes
        workflow.add_node("select_agents", self._select_agents_node)
        workflow.add_node("execute_task", self._execute_task_node)  # SUBGRAPH!
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("store", self._store_node)
        workflow.add_node("evolve", self._evolve_node)

        # Linear flow
        workflow.add_edge("select_agents", "execute_task")
        workflow.add_edge("execute_task", "evaluate")
        workflow.add_edge("evaluate", "store")
        workflow.add_edge("store", "evolve")

        # Loop back or end
        workflow.add_conditional_edges(
            "evolve",
            self._should_continue,
            {
                "continue": "select_agents",  # Next generation
                "end": END
            }
        )

        workflow.set_entry_point("select_agents")

        return workflow.compile(checkpointer=self.checkpointer)
```

---

## Usage: Attach Evolution to ANY Workflow

### Example 1: Blog Writing with Evolution

```python
from lean.workflows.blog_workflow import create_blog_workflow
from lean.evolution.wrapper import EvolutionWrapper
from lean.agent_pool import AgentPool
from lean.base_agent_v2 import create_agents_v2

# 1. Create task workflow (PURE - no evolution logic)
blog_workflow = create_blog_workflow()

# 2. Create agent pools
agents = create_agents_v2(...)
pools = {
    'intro': AgentPool(role='intro', initial_agents=[agents['intro']], ...),
    'body': AgentPool(role='body', initial_agents=[agents['body']], ...),
    'conclusion': AgentPool(role='conclusion', initial_agents=[agents['conclusion']], ...)
}

# 3. Wrap with evolution
wrapper = EvolutionWrapper(
    task_workflow=blog_workflow,  # Plug in task
    pools=pools,
    evaluator=ContentEvaluator(),
    evolution_frequency=5,
    max_generations=20
)

# 4. Get evolved workflow
evolved_workflow = wrapper.create_workflow()

# 5. Run!
state = create_initial_state(topic="AI Safety")
final_state = evolved_workflow.invoke(state)
```

### Example 2: Research Assistant with Evolution

```python
# Different task, SAME evolution wrapper!

from lean.workflows.research_workflow import create_research_workflow

# 1. Create different task workflow
research_workflow = create_research_workflow()  # Different nodes, different logic

# 2. Create agent pools (different roles!)
agents = create_research_agents_v2(...)
pools = {
    'search': AgentPool(role='search', ...),
    'analyze': AgentPool(role='analyze', ...),
    'summarize': AgentPool(role='summarize', ...)
}

# 3. Use SAME EvolutionWrapper class!
wrapper = EvolutionWrapper(
    task_workflow=research_workflow,  # Different task
    pools=pools,  # Different pools
    evaluator=ResearchEvaluator(),  # Different evaluator
    evolution_frequency=5,
    max_generations=20
)

# 4. Evolution logic is identical!
evolved_workflow = wrapper.create_workflow()
```

**Key: Evolution wrapper is REUSABLE across different tasks!**

---

## Benefits of Subgraph Architecture

### 1. Separation of Concerns
```
Task Workflow:
- Knows: How to write blogs / do research / etc.
- Doesn't know: Pools, evolution, selection

Evolution Wrapper:
- Knows: How to evolve agents
- Doesn't know: What task agents are doing
```

### 2. Reusability
```python
# Same wrapper, different tasks
blog_evolved = EvolutionWrapper(blog_workflow, ...)
research_evolved = EvolutionWrapper(research_workflow, ...)
coding_evolved = EvolutionWrapper(coding_workflow, ...)
```

### 3. Testing
```python
# Test task workflow WITHOUT evolution
blog_workflow_compiled = blog_workflow.compile()
state = create_initial_state(...)
state['selected_agents'] = fixed_agents  # Mock selection
result = blog_workflow_compiled.invoke(state)

# Test evolution wrapper WITH mock task
mock_task = create_mock_workflow()
wrapper = EvolutionWrapper(mock_task, ...)
# Test evolution logic independently
```

### 4. Gradual Migration
```python
# Phase 1: Use task workflow without evolution
blog_workflow_compiled = blog_workflow.compile()
result = blog_workflow_compiled.invoke(state)

# Phase 2: Add evolution when ready
wrapper = EvolutionWrapper(blog_workflow, ...)
evolved_workflow = wrapper.create_workflow()
result = evolved_workflow.invoke(state)
```

### 5. Mix and Match
```python
# Run 5 generations without evolution, then enable it
wrapper = EvolutionWrapper(
    blog_workflow,
    pools=pools,
    evolution_frequency=5,  # Don't evolve first 5
    max_generations=20
)
```

---

## Updated State Schema

```python
# In state.py
class BlogState(TypedDict):
    # Task fields (workflow-specific)
    topic: str
    intro: str
    body: str
    conclusion: str
    intro_reasoning: str
    body_reasoning: str
    conclusion_reasoning: str

    # Evolution fields (wrapper-specific)
    selected_agents: Dict[str, BaseAgentV2]  # NEW: Which agents were selected
    scores: Dict[str, float]  # Evaluation scores
    generation_number: int  # Current generation
    evolved: bool  # Whether evolution happened this cycle

    # Tracking
    reasoning_patterns_used: Dict[str, int]
    domain_knowledge_used: Dict[str, int]
```

**Key: `selected_agents` bridges the wrapper and task workflow.**

---

## M2 Implementation Plan (Updated)

### Phase 1: Python Utilities (Unchanged)
1. Compaction strategies (pure Python)
2. Selection strategies (pure Python)
3. Reproduction strategies (pure Python)

### Phase 2: Agent Pools (Unchanged)
4. AgentPool class using strategies

### Phase 3: Task Workflow (NEW - Separate from Evolution)
5. **Extract current PipelineV2 into pure task workflow**
   - Create `src/lean/workflows/blog_workflow.py`
   - Move intro/body/conclusion nodes
   - Remove evolution logic
   - Agents come from `state['selected_agents']`

### Phase 4: Evolution Wrapper (NEW - Reusable)
6. **Create `src/lean/evolution/wrapper.py`**
   - EvolutionWrapper class
   - select_agents_node
   - evaluate_node
   - store_node
   - evolve_node
   - Accepts ANY task workflow as subgraph

### Phase 5: Integration
7. **Update `src/lean/pipeline_v2.py`**
   - Use EvolutionWrapper
   - Pass blog_workflow as subgraph
   - Backward compatible API

### Phase 6: Testing
8. Unit tests for wrapper
9. Integration tests with blog workflow
10. Integration tests with mock workflows
11. 20-generation validation

---

## Example: Adding Evolution to Existing Research Assistant

Let's say you have this existing workflow:

```python
# Your existing research_assistant.py
from langgraph.graph import StateGraph, END

class ResearchState(TypedDict):
    query: str
    search_results: str
    analysis: str
    summary: str
    selected_agents: Dict[str, Any]  # ADD THIS

def create_research_workflow():
    workflow = StateGraph(ResearchState)

    def search_node(state):
        agent = state['selected_agents']['search']  # Get from wrapper
        # ... search logic ...
        return state

    def analyze_node(state):
        agent = state['selected_agents']['analyze']
        # ... analyze logic ...
        return state

    workflow.add_node("search", search_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_edge("search", "analyze")
    workflow.set_entry_point("search")

    return workflow

# OLD: Run without evolution
research_workflow = create_research_workflow().compile()
result = research_workflow.invoke({"query": "..."})

# NEW: Add evolution (ONE LINE!)
from lean.evolution.wrapper import EvolutionWrapper

wrapper = EvolutionWrapper(
    task_workflow=create_research_workflow(),  # Your existing workflow
    pools=create_research_pools(),
    evaluator=ResearchEvaluator(),
    max_generations=10
)
evolved_research = wrapper.create_workflow()

# Now has evolution!
result = evolved_research.invoke({"query": "..."})
```

**You added evolution to your existing workflow without changing the workflow logic!**

---

## Directory Structure

```
src/lean/
├── workflows/              # NEW: Task-specific workflows
│   ├── blog_workflow.py    # Blog writing (intro/body/conclusion)
│   ├── research_workflow.py  # Research assistant (future)
│   └── ...
│
├── evolution/              # NEW: Reusable evolution layer
│   ├── wrapper.py          # EvolutionWrapper class
│   └── ...
│
├── compaction.py           # Python utilities (no LangGraph)
├── selection.py            # Python utilities (no LangGraph)
├── reproduction.py         # Python utilities (no LangGraph)
├── agent_pool.py           # Data structure (used by wrapper)
│
├── pipeline_v2.py          # Updated: Uses EvolutionWrapper
├── base_agent_v2.py
├── state.py                # Updated: Add selected_agents field
└── ...
```

---

## Summary

**Question**: Can we make M2 evolution a reusable layer?

**Answer**: YES, using LangGraph subgraphs!

**Architecture**:
1. **Task Workflow** = Inner subgraph (pure task logic)
2. **Evolution Wrapper** = Outer graph (reusable evolution logic)
3. Wrapper treats task as a black box node

**Benefits**:
- Clean separation of concerns
- Reusable across different tasks
- Testable independently
- Gradual migration path
- Mix and match evolution settings

**Next Step**: Implement EvolutionWrapper and refactor existing PipelineV2 to use it.
