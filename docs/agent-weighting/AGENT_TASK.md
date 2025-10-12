# Agent Task: Agent Weighting System (M2)

## Branch: `feature/agent-weighting`

## Priority: HIGH (parallel after M1)

## Execution: PARALLEL (with M3, M4 after M1 completes)

## Objective

Implement agent-to-agent trust weighting system where each agent maintains relationship weights toward peers, enabling self-organizing hierarchical feedback loops.

**Current**: All agents treated equally, no relationship modeling
**Target**: Each agent has personalized trust weights toward other agents, weights evolve based on performance

## Dependencies

- ✅ M1 (async-orchestration) - MUST be merged first
- Can run parallel with M3 (memory-decay) and M4 (meta-agent)

## Background

Current agents have no memory of peer performance. All agent outputs are weighted equally when used as context. This prevents:
- Learning which agents provide high-quality context
- Adapting to individual agent strengths/weaknesses
- Emergent specialization through selective attention

## Tasks

### 1. Extend BlogState with Weight Tracking

**File**: `src/hvas_mini/state.py`

Add weight tracking fields:

```python
class BlogState(TypedDict):
    # ... existing fields ...

    # NEW: Agent relationship weights
    agent_weights: Dict[str, Dict[str, float]]  # {agent: {peer: trust_weight}}
    weight_history: List[Dict[str, Any]]  # [{generation, agent, peer, weight, delta}]
```

Update `create_initial_state()`:
```python
def create_initial_state(topic: str) -> BlogState:
    return BlogState(
        # ... existing fields ...
        agent_weights={
            "intro": {},
            "body": {},
            "conclusion": {},
        },
        weight_history=[],
    )
```

### 2. Create Weighting Infrastructure

**New Directory**: `src/hvas_mini/weighting/`

**File**: `src/hvas_mini/weighting/__init__.py`
```python
"""
Agent-to-agent trust weighting system.
"""

__all__ = ["TrustManager", "WeightUpdater"]
```

**File**: `src/hvas_mini/weighting/trust_manager.py`
```python
"""
Manages trust relationships between agents.
"""

from typing import Dict, List
import numpy as np


class TrustManager:
    """Manages trust weights between agents."""

    def __init__(self, initial_weight: float = 0.5, learning_rate: float = 0.1):
        """Initialize trust manager.

        Args:
            initial_weight: Starting trust weight for new relationships
            learning_rate: How quickly weights adapt (0-1)
        """
        self.initial_weight = initial_weight
        self.learning_rate = learning_rate
        self.weights: Dict[str, Dict[str, float]] = {}

    def initialize_agent(self, agent_name: str, peers: List[str]):
        """Initialize weights for a new agent.

        Args:
            agent_name: Name of agent
            peers: List of peer agent names
        """
        if agent_name not in self.weights:
            self.weights[agent_name] = {}

        for peer in peers:
            if peer not in self.weights[agent_name]:
                self.weights[agent_name][peer] = self.initial_weight

    def get_weight(self, agent: str, peer: str) -> float:
        """Get trust weight from agent toward peer.

        Args:
            agent: Observing agent
            peer: Observed agent

        Returns:
            Trust weight (0-1)
        """
        if agent not in self.weights:
            return self.initial_weight
        return self.weights[agent].get(peer, self.initial_weight)

    def update_weight(
        self, agent: str, peer: str, performance_signal: float
    ) -> float:
        """Update trust weight based on peer performance.

        Uses gradient descent: w_new = w_old + α * (signal - w_old)

        Args:
            agent: Agent updating its trust
            peer: Peer being evaluated
            performance_signal: Performance metric (0-1, where 1 = perfect)

        Returns:
            New weight value
        """
        current_weight = self.get_weight(agent, peer)

        # Gradient descent toward signal
        delta = self.learning_rate * (performance_signal - current_weight)
        new_weight = np.clip(current_weight + delta, 0.0, 1.0)

        # Store
        if agent not in self.weights:
            self.weights[agent] = {}
        self.weights[agent][peer] = new_weight

        return new_weight

    def get_weighted_context(
        self, agent: str, peer_outputs: Dict[str, str]
    ) -> str:
        """Create weighted context from peer outputs.

        Args:
            agent: Agent requesting context
            peer_outputs: {peer_name: output_text}

        Returns:
            Weighted context string
        """
        weighted_parts = []

        for peer, output in peer_outputs.items():
            weight = self.get_weight(agent, peer)

            # Weight determines prominence in context
            if weight >= 0.7:
                prefix = "[HIGH TRUST]"
            elif weight >= 0.4:
                prefix = "[MEDIUM TRUST]"
            else:
                prefix = "[LOW TRUST]"

            weighted_parts.append(f"{prefix} {peer}: {output}")

        return "\n\n".join(weighted_parts)

    def get_all_weights(self) -> Dict[str, Dict[str, float]]:
        """Get all trust weights.

        Returns:
            Full weight matrix
        """
        return self.weights.copy()
```

**File**: `src/hvas_mini/weighting/weight_updates.py`
```python
"""
Weight update logic based on evaluation scores.
"""

from typing import Dict


def calculate_performance_signal(
    agent_score: float, peer_score: float, max_score: float = 10.0
) -> float:
    """Calculate performance signal for weight update.

    Signal represents how useful peer's context was for agent's performance.

    Args:
        agent_score: Score received by agent using peer's context
        peer_score: Score received by peer itself
        max_score: Maximum possible score

    Returns:
        Performance signal (0-1)
    """
    # Normalize scores
    agent_norm = agent_score / max_score
    peer_norm = peer_score / max_score

    # High agent score + high peer score = strong signal
    # (peer's quality helped agent succeed)
    signal = (agent_norm + peer_norm) / 2

    return max(0.0, min(1.0, signal))


def update_all_weights(
    trust_manager, state: Dict, scores: Dict[str, float]
) -> Dict[str, List[Dict]]:
    """Update all agent weights based on current scores.

    Args:
        trust_manager: TrustManager instance
        state: Current BlogState
        scores: Current evaluation scores

    Returns:
        Weight update history entries
    """
    updates = []
    agents = list(scores.keys())

    for agent in agents:
        for peer in agents:
            if agent == peer:
                continue

            # Calculate signal
            signal = calculate_performance_signal(
                agent_score=scores[agent],
                peer_score=scores[peer],
            )

            # Update weight
            old_weight = trust_manager.get_weight(agent, peer)
            new_weight = trust_manager.update_weight(agent, peer, signal)

            # Record
            updates.append({
                "agent": agent,
                "peer": peer,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "delta": new_weight - old_weight,
                "signal": signal,
            })

    return updates
```

### 3. Integrate Weights into BaseAgent

**File**: `src/hvas_mini/agents.py`

Modify BaseAgent to use weighted context:

```python
from hvas_mini.weighting.trust_manager import TrustManager

class BaseAgent(ABC):
    """Base agent with trust weighting."""

    def __init__(
        self,
        role: str,
        content_key: str,
        llm,
        memory: MemoryManager,
        trust_manager: TrustManager,
    ):
        self.role = role
        self.content_key = content_key
        self.llm = llm
        self.memory = memory
        self.trust_manager = trust_manager  # NEW
        # ... rest of init ...

    async def __call__(self, state: BlogState) -> BlogState:
        """Execute agent with weighted peer context."""

        # 1. Retrieve own memories
        memories = self.memory.retrieve(state["topic"])
        state["retrieved_memories"][self.role] = [m["content"] for m in memories]

        # 2. NEW: Get weighted context from peer agents
        peer_outputs = {}
        if self.role == "body":
            peer_outputs["intro"] = state.get("intro", "")
        elif self.role == "conclusion":
            peer_outputs["intro"] = state.get("intro", "")
            peer_outputs["body"] = state.get("body", "")

        weighted_context = ""
        if peer_outputs:
            weighted_context = self.trust_manager.get_weighted_context(
                self.role, peer_outputs
            )

        # 3. Generate content with weighted context
        content = await self.generate_content(
            state, memories, weighted_context
        )

        # 4. Store in state
        state[self.content_key] = content

        # 5. Prepare pending memory
        self.pending_memory = {
            "content": content,
            "topic": state["topic"],
            "timestamp": datetime.now().isoformat(),
        }

        return state

    @abstractmethod
    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict],
        weighted_context: str = ""  # NEW
    ) -> str:
        """Generate content with weighted peer context."""
        pass
```

Update specialized agents to accept weighted_context:

```python
class BodyAgent(BaseAgent):
    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict],
        weighted_context: str = ""
    ) -> str:
        """Generate body with weighted intro context."""

        memory_examples = "\n".join([m["content"] for m in memories[:3]])

        prompt = f"""Write the body section for: {state['topic']}

PEER CONTEXT (trust-weighted):
{weighted_context}

PAST EXAMPLES:
{memory_examples}

Write comprehensive body content."""

        # ... rest of generation ...
```

### 4. Integrate into Pipeline Evolution Node

**File**: `src/hvas_mini/pipeline.py`

```python
from hvas_mini.weighting.trust_manager import TrustManager
from hvas_mini.weighting.weight_updates import update_all_weights

class HVASMiniPipeline:
    def __init__(self, persist_directory: str = "./data/memories"):
        # ... existing init ...

        # NEW: Trust manager
        self.trust_manager = TrustManager(
            initial_weight=0.5,
            learning_rate=0.1
        )

        # Initialize agents with trust manager
        self.agents = create_agents(persist_directory, self.trust_manager)

    async def _evolution_node(self, state: BlogState) -> BlogState:
        """Evolution: update weights, store memories, evolve parameters."""

        # 1. NEW: Update trust weights based on scores
        weight_updates = update_all_weights(
            self.trust_manager, state, state["scores"]
        )

        # Store in state
        state["agent_weights"] = self.trust_manager.get_all_weights()
        state["weight_history"].extend(weight_updates)

        # 2. Store memories and evolve parameters (existing)
        for role, agent in self.agents.items():
            score = state["scores"].get(role, 0)
            agent.store_memory(score)
            agent.evolve_parameters(score, state)

        # 3. Log
        state["stream_logs"].append(
            f"[Evolution] Weights updated: {len(weight_updates)} relationships"
        )

        return state
```

### 5. Add Weight Visualization

**File**: `src/hvas_mini/visualization.py`

```python
from rich.table import Table

class StreamVisualizer:
    # ... existing methods ...

    def create_weights_panel(self, state: BlogState) -> Panel:
        """Show agent trust weight matrix.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with weight matrix
        """
        weights = state.get("agent_weights", {})

        if not weights:
            return Panel(
                "[dim]No weights yet[/dim]",
                title="🔗 Trust Weights",
                border_style="blue"
            )

        # Create weight matrix table
        table = Table(show_header=True, header_style="bold")
        table.add_column("From \\ To", style="cyan")

        agents = sorted(weights.keys())
        for agent in agents:
            table.add_column(agent, justify="center")

        for agent in agents:
            row = [agent]
            for peer in agents:
                if agent == peer:
                    row.append("-")
                else:
                    weight = weights.get(agent, {}).get(peer, 0.5)

                    # Color code weights
                    if weight >= 0.7:
                        color = "green"
                    elif weight >= 0.4:
                        color = "yellow"
                    else:
                        color = "red"

                    row.append(f"[{color}]{weight:.2f}[/{color}]")

            table.add_row(*row)

        return Panel(
            table,
            title="🔗 Trust Weights",
            border_style="blue"
        )

    async def display_stream(self, state_stream: AsyncIterator[BlogState]):
        """Display with weights panel."""
        # ... existing code ...

        layout.split_column(
            Layout(name="status", size=10),
            Layout(name="concurrency", size=6),
            Layout(name="weights", size=8),  # NEW
            Layout(name="memories", size=10),
            Layout(name="evolution", size=8),
            Layout(name="logs", size=7),
        )

        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["concurrency"].update(self.create_concurrency_panel(state))
                layout["weights"].update(self.create_weights_panel(state))  # NEW
                layout["memories"].update(self.create_memory_panel(state))
                layout["evolution"].update(self.create_evolution_panel(state))
                layout["logs"].update(self.create_logs_panel(state))
```

### 6. Create Tests

**File**: `test_agent_weighting.py`

```python
"""Tests for agent weighting system."""

import pytest
from hvas_mini.weighting.trust_manager import TrustManager
from hvas_mini.weighting.weight_updates import (
    calculate_performance_signal,
    update_all_weights,
)


def test_trust_manager_initialization():
    """Test TrustManager creates correct initial weights."""
    tm = TrustManager(initial_weight=0.5)
    tm.initialize_agent("agent1", ["agent2", "agent3"])

    assert tm.get_weight("agent1", "agent2") == 0.5
    assert tm.get_weight("agent1", "agent3") == 0.5


def test_weight_update():
    """Test weight updates move toward performance signal."""
    tm = TrustManager(initial_weight=0.5, learning_rate=0.1)
    tm.initialize_agent("agent1", ["agent2"])

    # High performance signal should increase weight
    new_weight = tm.update_weight("agent1", "agent2", performance_signal=0.9)
    assert new_weight > 0.5
    assert new_weight < 0.9  # Partial step

    # Low signal should decrease weight
    new_weight = tm.update_weight("agent1", "agent2", performance_signal=0.1)
    assert new_weight < 0.6


def test_performance_signal():
    """Test performance signal calculation."""
    # Both high = strong signal
    signal = calculate_performance_signal(agent_score=8.0, peer_score=9.0)
    assert signal >= 0.8

    # Both low = weak signal
    signal = calculate_performance_signal(agent_score=3.0, peer_score=4.0)
    assert signal <= 0.4


def test_weighted_context():
    """Test weighted context generation."""
    tm = TrustManager()
    tm.weights = {
        "body": {
            "intro": 0.8,  # High trust
        }
    }

    context = tm.get_weighted_context(
        "body",
        {"intro": "Machine learning is a subset of AI."}
    )

    assert "[HIGH TRUST]" in context
    assert "intro" in context


def test_weight_convergence():
    """Test weights converge with consistent signals."""
    tm = TrustManager(learning_rate=0.1)
    tm.initialize_agent("agent1", ["agent2"])

    # Apply consistent high signal
    for _ in range(20):
        tm.update_weight("agent1", "agent2", performance_signal=0.9)

    final_weight = tm.get_weight("agent1", "agent2")
    assert final_weight > 0.85  # Should converge close to signal
```

## Deliverables Checklist

- [ ] `src/hvas_mini/state.py` updated with weight tracking fields
- [ ] `src/hvas_mini/weighting/__init__.py` created
- [ ] `src/hvas_mini/weighting/trust_manager.py` created
- [ ] `src/hvas_mini/weighting/weight_updates.py` created
- [ ] `src/hvas_mini/agents.py` modified to use weighted context
- [ ] Specialized agents updated to accept weighted_context parameter
- [ ] `src/hvas_mini/pipeline.py` integrated with TrustManager
- [ ] `src/hvas_mini/visualization.py` updated with weights panel
- [ ] `test_agent_weighting.py` created with passing tests
- [ ] Weight matrix converges over 10+ generations

## Acceptance Criteria

1. ✅ Each agent maintains trust weights toward all peers
2. ✅ Weights update based on evaluation scores
3. ✅ Weighted context includes trust indicators
4. ✅ Weight matrix visualized in terminal UI
5. ✅ Weights converge within 10 generations (Δw < 0.1)
6. ✅ All existing tests still pass (no regressions)
7. ✅ New weighting tests pass

## Testing

```bash
cd worktrees/agent-weighting

# Run new weighting tests
uv run pytest test_agent_weighting.py -v

# Run all tests
uv run pytest

# Run demo to see weights evolve
export ANTHROPIC_API_KEY=your_key
uv run python main.py
```

Expected output: Weight matrix panel shows weights converging toward stable values after 5-10 generations.

## Integration Notes

This milestone enables:
- Emergent specialization through selective attention
- Performance-based relationship modeling
- Foundation for meta-agent to analyze network structure (M4)
- Visualization of trust network evolution (M5)

## Next Steps

After merging M2 to main:
- M3 (memory-decay) can reference weights in decay formula
- M4 (meta-agent) can use weight variance to detect redundant agents
- M5 (visualization-v2) can render weight network graphs
