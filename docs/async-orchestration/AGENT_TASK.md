# Agent Task: Async Orchestration (M1)

## Branch: `feature/async-orchestration`

## Priority: CRITICAL - BLOCKING (must complete before M2, M3, M4)

## Execution: SEQUENTIAL (blocks all other iteration 2 work)

## Objective

Transform the sequential LangGraph workflow into a truly concurrent system where agents in the same hierarchical layer execute in parallel, with synchronization barriers between layers.

**Current**: intro → body → conclusion (100% sequential)
**Target**: intro → [body ∥ conclusion] (50%+ concurrent execution)

## Dependencies

- ✅ Current main branch (Iteration 1 complete)
- No other iteration 2 dependencies (this is the foundation)

## Background

Current implementation in `pipeline.py`:
```python
workflow.set_entry_point("intro")
workflow.add_edge("intro", "body")
workflow.add_edge("body", "conclusion")
workflow.add_edge("conclusion", "evaluate")
```

This creates a **chain**, not a hierarchy. Body waits for intro, conclusion waits for body.

## Tasks

### 1. Add Timing Instrumentation to BlogState

**File**: `src/hvas_mini/state.py`

Add timing fields to track concurrent execution:

```python
class BlogState(TypedDict):
    # ... existing fields ...

    # NEW: Concurrency tracking
    agent_timings: Dict[str, Dict[str, float]]  # {agent: {start, end, duration}}
    layer_barriers: List[Dict[str, float]]  # [{layer_id, agents, wait_time}]
```

Update `create_initial_state()`:
```python
def create_initial_state(topic: str) -> BlogState:
    return BlogState(
        # ... existing fields ...
        agent_timings={},
        layer_barriers=[],
    )
```

### 2. Create Async Coordination Infrastructure

**New Directory**: `src/hvas_mini/orchestration/`

**File**: `src/hvas_mini/orchestration/__init__.py`
```python
"""
Async orchestration utilities for concurrent agent execution.
"""

__all__ = ["AsyncCoordinator", "LayerBarrier"]
```

**File**: `src/hvas_mini/orchestration/async_coordinator.py`
```python
"""
Coordinates concurrent agent execution with layer synchronization.
"""

import asyncio
from typing import List, Dict, Callable, Awaitable
from datetime import datetime
import time

class AsyncCoordinator:
    """Manages concurrent execution of agents within layers."""

    def __init__(self):
        self.active_agents = {}
        self.layer_results = {}

    async def execute_layer(
        self,
        layer_name: str,
        agents: List[Callable],
        state: Dict,
        timeout: float = 30.0
    ) -> Dict:
        """Execute multiple agents concurrently in a layer.

        Args:
            layer_name: Identifier for this execution layer
            agents: List of async callables (agent instances)
            state: Shared state dict
            timeout: Max time to wait for layer completion

        Returns:
            Updated state after all agents complete
        """
        layer_start = time.time()

        # Record start times for each agent
        for agent in agents:
            agent_name = agent.role if hasattr(agent, 'role') else str(agent)
            state["agent_timings"][agent_name] = {
                "start": layer_start,
                "end": None,
                "duration": None
            }

        # Execute all agents concurrently
        try:
            tasks = [agent(state) for agent in agents]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Merge results (all agents update same state)
            # Last write wins for conflicting keys
            final_state = state
            for result in results:
                if isinstance(result, Exception):
                    print(f"[AsyncCoordinator] Agent error: {result}")
                    continue
                if isinstance(result, dict):
                    final_state = result  # Agents return updated state

            # Record end times
            layer_end = time.time()
            for agent in agents:
                agent_name = agent.role if hasattr(agent, 'role') else str(agent)
                if agent_name in final_state["agent_timings"]:
                    final_state["agent_timings"][agent_name]["end"] = layer_end
                    final_state["agent_timings"][agent_name]["duration"] = (
                        layer_end - final_state["agent_timings"][agent_name]["start"]
                    )

            # Record barrier
            final_state["layer_barriers"].append({
                "layer": layer_name,
                "agents": [a.role if hasattr(a, 'role') else str(a) for a in agents],
                "wait_time": layer_end - layer_start
            })

            return final_state

        except asyncio.TimeoutError:
            print(f"[AsyncCoordinator] Layer {layer_name} timed out after {timeout}s")
            return state


class LayerBarrier:
    """Synchronization point between hierarchical layers."""

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.waiting_agents = set()
        self.completed_agents = set()
        self.lock = asyncio.Lock()

    async def wait(self, agent_name: str, expected_count: int):
        """Wait for all agents in layer to complete.

        Args:
            agent_name: Name of agent waiting
            expected_count: Total agents expected in this layer
        """
        async with self.lock:
            self.completed_agents.add(agent_name)

        # Wait until all agents complete
        while len(self.completed_agents) < expected_count:
            await asyncio.sleep(0.01)

    def reset(self):
        """Reset barrier for next use."""
        self.completed_agents.clear()
```

### 3. Modify BaseAgent for True Async Execution

**File**: `src/hvas_mini/agents.py`

Current agents are async but execute sequentially. Modify to support concurrent execution:

```python
class BaseAgent(ABC):
    # ... existing code ...

    async def __call__(self, state: BlogState) -> BlogState:
        """Execute agent - called by LangGraph.

        MODIFIED: Now truly async, can run concurrently with peers.
        """
        import time

        agent_start = time.time()

        # 1. Retrieve relevant memories (async-safe)
        memories = self.memory.retrieve(state["topic"])
        state["retrieved_memories"][self.role] = [m["content"] for m in memories]

        # 2. Log retrieval
        if os.getenv("SHOW_MEMORY_RETRIEVAL", "true").lower() == "true":
            state["stream_logs"].append(
                f"[{self.role}] Retrieved {len(memories)} memories (concurrent)"
            )

        # 3. Generate content (async)
        self.llm.temperature = self.parameters["temperature"]
        content = await self.generate_content(state, memories)

        # 4. Store in state
        state[self.content_key] = content

        # 5. Record timing
        agent_end = time.time()
        state["agent_timings"][self.role] = {
            "start": agent_start,
            "end": agent_end,
            "duration": agent_end - agent_start
        }

        # 6. Prepare for memory storage
        self.pending_memory = {
            "content": content,
            "topic": state["topic"],
            "timestamp": datetime.now().isoformat(),
        }

        return state
```

### 4. Refactor Pipeline for Concurrent Execution

**File**: `src/hvas_mini/pipeline.py`

Replace sequential graph with layered concurrent execution:

```python
from hvas_mini.orchestration.async_coordinator import AsyncCoordinator

class HVASMiniPipeline:
    def __init__(self, persist_directory: str = "./data/memories"):
        # ... existing initialization ...
        self.coordinator = AsyncCoordinator()

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow with concurrent execution.

        NEW Graph structure:
        START → intro → [body ∥ conclusion] → evaluate → evolve → END

        Agents in brackets execute concurrently.
        """
        workflow = StateGraph(BlogState)

        # Layer 1: Intro (sequential - needs topic context)
        workflow.add_node("intro", self.agents["intro"])

        # Layer 2: Body & Conclusion (CONCURRENT)
        # Both can read intro, but don't depend on each other
        workflow.add_node("body_and_conclusion", self._concurrent_layer_2)

        # Layer 3: Evaluation (sequential - needs all content)
        workflow.add_node("evaluate", self.evaluator)

        # Layer 4: Evolution (sequential)
        workflow.add_node("evolve", self._evolution_node)

        # Define execution flow
        workflow.set_entry_point("intro")
        workflow.add_edge("intro", "body_and_conclusion")
        workflow.add_edge("body_and_conclusion", "evaluate")
        workflow.add_edge("evaluate", "evolve")
        workflow.add_edge("evolve", END)

        # Compile
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _concurrent_layer_2(self, state: BlogState) -> BlogState:
        """Execute body and conclusion agents concurrently.

        This is a LangGraph node that internally runs multiple agents in parallel.
        """
        # Execute both agents concurrently using coordinator
        agents = [self.agents["body"], self.agents["conclusion"]]

        updated_state = await self.coordinator.execute_layer(
            layer_name="layer_2_content",
            agents=agents,
            state=state,
            timeout=60.0
        )

        # Log concurrent execution
        updated_state["stream_logs"].append(
            "[Pipeline] Body and Conclusion executed concurrently"
        )

        return updated_state
```

### 5. Add Concurrency Metrics to Visualization

**File**: `src/hvas_mini/visualization.py`

Add panel showing concurrent execution:

```python
class StreamVisualizer:
    # ... existing methods ...

    def create_concurrency_panel(self, state: BlogState) -> Panel:
        """Show agent execution timing and overlap.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with concurrency metrics
        """
        timings = state.get("agent_timings", {})

        if not timings:
            return Panel(
                "[dim]No timing data yet[/dim]",
                title="⏱️  Concurrent Execution",
                border_style="magenta"
            )

        timing_text = ""
        total_time = 0
        overlapping_time = 0

        for agent, timing in timings.items():
            if timing.get("duration"):
                duration = timing["duration"]
                total_time += duration
                timing_text += f"[cyan]{agent}:[/cyan] {duration:.2f}s\n"

        # Calculate concurrency percentage
        if state.get("layer_barriers"):
            last_barrier = state["layer_barriers"][-1]
            barrier_time = last_barrier.get("wait_time", 0)
            if barrier_time > 0 and total_time > 0:
                concurrency_pct = (1 - (barrier_time / total_time)) * 100
                timing_text += f"\n[bold green]Concurrency: {concurrency_pct:.1f}%[/bold green]"

        return Panel(
            timing_text,
            title="⏱️  Concurrent Execution",
            border_style="magenta"
        )

    async def display_stream(self, state_stream: AsyncIterator[BlogState]):
        """Display real-time execution updates.

        MODIFIED: Add concurrency panel
        """
        if not self.show_visualization:
            async for _ in state_stream:
                pass
            return

        layout = Layout()
        layout.split_column(
            Layout(name="status", size=10),
            Layout(name="concurrency", size=6),  # NEW
            Layout(name="memories", size=10),
            Layout(name="evolution", size=8),
            Layout(name="logs", size=7),
        )

        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["concurrency"].update(self.create_concurrency_panel(state))  # NEW
                layout["memories"].update(self.create_memory_panel(state))
                layout["evolution"].update(self.create_evolution_panel(state))
                layout["logs"].update(self.create_logs_panel(state))
```

### 6. Create Tests

**File**: `test_async_orchestration.py`

```python
"""Tests for async orchestration."""

import pytest
import asyncio
from hvas_mini.orchestration.async_coordinator import AsyncCoordinator
from hvas_mini.state import create_initial_state


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, role: str, delay: float = 0.1):
        self.role = role
        self.delay = delay

    async def __call__(self, state):
        await asyncio.sleep(self.delay)
        state[self.role] = f"Content from {self.role}"
        return state


@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test that agents execute concurrently."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    agent1 = MockAgent("agent1", delay=0.2)
    agent2 = MockAgent("agent2", delay=0.2)

    start_time = asyncio.get_event_loop().time()

    result = await coordinator.execute_layer(
        "test_layer",
        [agent1, agent2],
        state,
        timeout=5.0
    )

    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time

    # If truly concurrent, should take ~0.2s not ~0.4s
    assert duration < 0.3, f"Expected <0.3s (concurrent), got {duration:.2f}s (sequential?)"

    # Both agents should have completed
    assert "agent1" in result
    assert "agent2" in result


@pytest.mark.asyncio
async def test_timing_instrumentation():
    """Test that timing data is recorded."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    agent = MockAgent("test_agent", delay=0.1)
    result = await coordinator.execute_layer("layer", [agent], state)

    assert "agent_timings" in result
    assert "test_agent" in result["agent_timings"]
    assert "duration" in result["agent_timings"]["test_agent"]
    assert result["agent_timings"]["test_agent"]["duration"] >= 0.1
```

## Deliverables Checklist

- [ ] `src/hvas_mini/state.py` updated with timing fields
- [ ] `src/hvas_mini/orchestration/__init__.py` created
- [ ] `src/hvas_mini/orchestration/async_coordinator.py` created
- [ ] `src/hvas_mini/agents.py` modified for true async
- [ ] `src/hvas_mini/pipeline.py` refactored with concurrent layer
- [ ] `src/hvas_mini/visualization.py` updated with concurrency panel
- [ ] `test_async_orchestration.py` created with passing tests
- [ ] Tests confirm >30% concurrency improvement

## Acceptance Criteria

1. ✅ Body and conclusion agents execute concurrently (timing overlap >30%)
2. ✅ `agent_timings` correctly records start/end/duration for all agents
3. ✅ `layer_barriers` tracks synchronization points
4. ✅ Visualization shows concurrency percentage
5. ✅ All existing tests still pass (no regressions)
6. ✅ New async tests pass with timing validation
7. ✅ System remains stable under concurrent load

## Testing

```bash
cd worktrees/async-orchestration

# Run new async tests
uv run pytest test_async_orchestration.py -v

# Run all tests (ensure no regressions)
uv run pytest

# Run demo to see concurrency in action
export ANTHROPIC_API_KEY=your_key
uv run python main.py
```

Expected output: Concurrency panel shows >30% concurrent execution time.

## Performance Benchmarks

Before (Sequential):
- 3 agents × ~2s each = ~6s total

After (Concurrent):
- Intro: ~2s
- Body ∥ Conclusion: ~2s (concurrent)
- Total: ~4s (33% faster)

## Integration Notes

This milestone enables:
- M2: Agent weighting (needs concurrent execution to show weight impact)
- M3: Memory decay (concurrent retrieval stress-tests decay logic)
- M4: Meta-agent (can spawn agents that run concurrently)

## Next Steps

After merging M1 to main:
1. Start M2 (agent-weighting) in parallel with M3 (memory-decay) and M4 (meta-agent)
2. All three can proceed concurrently using the async infrastructure
