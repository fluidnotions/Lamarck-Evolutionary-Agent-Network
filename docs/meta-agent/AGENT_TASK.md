# Agent Task: Meta-Agent System (M4)

## Branch: `feature/meta-agent`

## Priority: MEDIUM (parallel after M1)

## Execution: PARALLEL (with M2, M3 after M1 completes)

## Objective

Implement a meta-agent that monitors system metrics and dynamically modifies graph topology at runtime by spawning, merging, or removing agents based on performance thresholds.

**Current**: Fixed 3-agent topology (intro, body, conclusion)
**Target**: Self-organizing graph that adapts structure based on diversity, coherence, and performance metrics

## Dependencies

- ✅ M1 (async-orchestration) - MUST be merged first
- Can run parallel with M2 (agent-weighting) and M3 (memory-decay)
- Benefits from M2 if available (uses weight variance in decisions)

## Background

Current system has static graph topology defined in `pipeline.py`. Problems:
- Cannot adapt to changing task requirements
- No mechanism to detect redundant agents
- Cannot spawn specialists for underperforming areas
- Topology is hard-coded, not learned

Solution: Meta-agent monitors system-level metrics (diversity, coherence, performance) and applies graph mutations (spawn, merge, remove) when thresholds are crossed.

## Tasks

### 1. Extend BlogState with Topology Tracking

**File**: `src/hvas_mini/state.py`

Add topology tracking fields:

```python
class BlogState(TypedDict):
    # ... existing fields ...

    # NEW: Topology tracking
    active_agents: List[str]  # Current agents in graph
    topology_version: int  # Increments on graph mutation
    topology_history: List[Dict[str, Any]]  # [{timestamp, action, agents, reason}]
    meta_agent_metrics: Dict[str, float]  # System-level metrics
```

Update `create_initial_state()`:
```python
def create_initial_state(topic: str) -> BlogState:
    return BlogState(
        # ... existing fields ...
        active_agents=["intro", "body", "conclusion"],
        topology_version=1,
        topology_history=[],
        meta_agent_metrics={},
    )
```

### 2. Create Meta-Agent Infrastructure

**New Directory**: `src/hvas_mini/meta/`

**File**: `src/hvas_mini/meta/__init__.py`
```python
"""
Meta-agent system for dynamic graph topology modification.
"""

__all__ = ["MetaAgent", "GraphMutator", "MetricsMonitor"]
```

**File**: `src/hvas_mini/meta/metrics_monitor.py`
```python
"""
System-level metrics monitoring for meta-agent decisions.
"""

from typing import Dict, List
import numpy as np
from datetime import datetime


class MetricsMonitor:
    """Monitors system-level performance metrics."""

    def __init__(self):
        """Initialize metrics monitor."""
        self.history: List[Dict] = []

    def calculate_diversity(
        self, agent_outputs: Dict[str, str], embeddings_fn=None
    ) -> float:
        """Calculate output diversity across agents.

        Uses pairwise cosine similarity between agent outputs.
        Low similarity = high diversity.

        Args:
            agent_outputs: {agent_name: output_text}
            embeddings_fn: Optional function to compute embeddings

        Returns:
            Diversity score (0-1, where 1 = maximum diversity)
        """
        if len(agent_outputs) < 2:
            return 0.0

        outputs = list(agent_outputs.values())

        # Simple heuristic: character-level Jaccard similarity
        # (In production, use embeddings_fn for semantic similarity)
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                set1 = set(outputs[i].lower().split())
                set2 = set(outputs[j].lower().split())

                if not set1 or not set2:
                    similarity = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0.0

                similarities.append(similarity)

        if not similarities:
            return 0.0

        # Diversity is inverse of average similarity
        avg_similarity = sum(similarities) / len(similarities)
        diversity = 1.0 - avg_similarity

        return max(0.0, min(1.0, diversity))

    def calculate_coherence(self, agent_outputs: Dict[str, str]) -> float:
        """Calculate coherence across agent outputs.

        Measures how well outputs relate to each other.
        High coherence = agents are aligned.

        Args:
            agent_outputs: {agent_name: output_text}

        Returns:
            Coherence score (0-1, where 1 = perfect coherence)
        """
        if len(agent_outputs) < 2:
            return 1.0

        outputs = list(agent_outputs.values())

        # Simple heuristic: shared vocabulary
        # (In production, use semantic similarity)
        all_words = set()
        word_sets = []

        for output in outputs:
            words = set(output.lower().split())
            word_sets.append(words)
            all_words.update(words)

        if not all_words:
            return 0.0

        # Coherence = average proportion of shared words
        shared_ratios = []
        for word_set in word_sets:
            shared = len(word_set & all_words) / len(all_words)
            shared_ratios.append(shared)

        coherence = sum(shared_ratios) / len(shared_ratios)

        return max(0.0, min(1.0, coherence))

    def calculate_performance(self, scores: Dict[str, float]) -> float:
        """Calculate average performance across agents.

        Args:
            scores: {agent_name: score}

        Returns:
            Average performance (0-10 scale)
        """
        if not scores:
            return 0.0

        return sum(scores.values()) / len(scores)

    def calculate_weight_variance(
        self, agent_weights: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate variance in trust weights.

        High variance = agents have differentiated opinions.
        Low variance = agents trust everyone equally (potential redundancy).

        Args:
            agent_weights: {agent: {peer: weight}}

        Returns:
            Weight variance (0-1)
        """
        if not agent_weights:
            return 0.0

        all_weights = []
        for agent_dict in agent_weights.values():
            all_weights.extend(agent_dict.values())

        if not all_weights:
            return 0.0

        variance = np.var(all_weights)
        return min(1.0, variance)  # Cap at 1.0

    def compute_system_metrics(self, state: Dict) -> Dict[str, float]:
        """Compute all system-level metrics.

        Args:
            state: Current BlogState

        Returns:
            Dict of metric values
        """
        agent_outputs = {
            "intro": state.get("intro", ""),
            "body": state.get("body", ""),
            "conclusion": state.get("conclusion", ""),
        }

        # Filter out empty outputs
        agent_outputs = {k: v for k, v in agent_outputs.items() if v}

        metrics = {
            "diversity": self.calculate_diversity(agent_outputs),
            "coherence": self.calculate_coherence(agent_outputs),
            "performance": self.calculate_performance(state.get("scores", {})),
        }

        # Add weight variance if available
        if state.get("agent_weights"):
            metrics["weight_variance"] = self.calculate_weight_variance(
                state["agent_weights"]
            )

        # Record history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.copy(),
        })

        return metrics
```

**File**: `src/hvas_mini/meta/graph_mutations.py`
```python
"""
Graph mutation operations for meta-agent.
"""

from typing import Dict, List, Any
from datetime import datetime
import os


class GraphMutator:
    """Applies topology mutations to the agent graph."""

    def __init__(self):
        """Initialize graph mutator."""
        self.mutation_history: List[Dict] = []

    def spawn_agent(
        self,
        state: Dict,
        role: str,
        reason: str,
    ) -> Dict:
        """Spawn a new agent into the graph.

        Args:
            state: Current BlogState
            role: Role for new agent
            reason: Why this spawn was triggered

        Returns:
            Updated state
        """
        if role in state["active_agents"]:
            return state  # Already exists

        # Add to active agents
        state["active_agents"].append(role)

        # Increment topology version
        state["topology_version"] += 1

        # Record mutation
        mutation = {
            "timestamp": datetime.now().isoformat(),
            "action": "spawn",
            "agent": role,
            "reason": reason,
            "topology_version": state["topology_version"],
        }
        state["topology_history"].append(mutation)
        self.mutation_history.append(mutation)

        # Log
        state["stream_logs"].append(
            f"[MetaAgent] Spawned new agent '{role}': {reason}"
        )

        return state

    def remove_agent(
        self,
        state: Dict,
        role: str,
        reason: str,
    ) -> Dict:
        """Remove an agent from the graph.

        Args:
            state: Current BlogState
            role: Role to remove
            reason: Why this removal was triggered

        Returns:
            Updated state
        """
        if role not in state["active_agents"]:
            return state  # Doesn't exist

        # Don't remove if it's the last agent
        if len(state["active_agents"]) <= 1:
            return state

        # Remove from active agents
        state["active_agents"].remove(role)

        # Increment topology version
        state["topology_version"] += 1

        # Record mutation
        mutation = {
            "timestamp": datetime.now().isoformat(),
            "action": "remove",
            "agent": role,
            "reason": reason,
            "topology_version": state["topology_version"],
        }
        state["topology_history"].append(mutation)
        self.mutation_history.append(mutation)

        # Log
        state["stream_logs"].append(
            f"[MetaAgent] Removed agent '{role}': {reason}"
        )

        return state

    def merge_agents(
        self,
        state: Dict,
        agent1: str,
        agent2: str,
        merged_role: str,
        reason: str,
    ) -> Dict:
        """Merge two agents into one.

        Args:
            state: Current BlogState
            agent1: First agent to merge
            agent2: Second agent to merge
            merged_role: Role for merged agent
            reason: Why this merge was triggered

        Returns:
            Updated state
        """
        if agent1 not in state["active_agents"] or agent2 not in state["active_agents"]:
            return state  # One doesn't exist

        # Remove both agents
        state["active_agents"].remove(agent1)
        state["active_agents"].remove(agent2)

        # Add merged agent
        state["active_agents"].append(merged_role)

        # Increment topology version
        state["topology_version"] += 1

        # Record mutation
        mutation = {
            "timestamp": datetime.now().isoformat(),
            "action": "merge",
            "agents": [agent1, agent2],
            "merged_role": merged_role,
            "reason": reason,
            "topology_version": state["topology_version"],
        }
        state["topology_history"].append(mutation)
        self.mutation_history.append(mutation)

        # Log
        state["stream_logs"].append(
            f"[MetaAgent] Merged '{agent1}' + '{agent2}' → '{merged_role}': {reason}"
        )

        return state
```

**File**: `src/hvas_mini/meta/meta_agent.py`
```python
"""
Meta-agent for dynamic topology management.
"""

import os
from typing import Dict
from hvas_mini.meta.metrics_monitor import MetricsMonitor
from hvas_mini.meta.graph_mutations import GraphMutator


class MetaAgent:
    """Meta-agent that monitors and modifies graph topology."""

    def __init__(
        self,
        diversity_threshold: float = 0.3,
        coherence_threshold: float = 0.8,
        performance_threshold: float = 6.0,
        enabled: bool = True,
    ):
        """Initialize meta-agent.

        Args:
            diversity_threshold: Spawn new agent if diversity < this
            coherence_threshold: Merge agents if coherence > this
            performance_threshold: Remove agent if performance < this
            enabled: Whether meta-agent is active
        """
        self.diversity_threshold = diversity_threshold
        self.coherence_threshold = coherence_threshold
        self.performance_threshold = performance_threshold
        self.enabled = enabled

        self.monitor = MetricsMonitor()
        self.mutator = GraphMutator()

        self.generation_count = 0
        self.min_generations_before_mutation = int(
            os.getenv("META_MIN_GENERATIONS", "5")
        )

    async def __call__(self, state: Dict) -> Dict:
        """Analyze system and apply topology mutations if needed.

        Args:
            state: Current BlogState

        Returns:
            Updated state (potentially with topology changes)
        """
        if not self.enabled:
            return state

        self.generation_count += 1

        # Don't mutate too early (need data to accumulate)
        if self.generation_count < self.min_generations_before_mutation:
            return state

        # Compute metrics
        metrics = self.monitor.compute_system_metrics(state)
        state["meta_agent_metrics"] = metrics

        # Log metrics
        state["stream_logs"].append(
            f"[MetaAgent] Diversity: {metrics['diversity']:.2f}, "
            f"Coherence: {metrics['coherence']:.2f}, "
            f"Performance: {metrics['performance']:.2f}"
        )

        # Apply mutation rules
        state = self._apply_mutation_rules(state, metrics)

        return state

    def _apply_mutation_rules(self, state: Dict, metrics: Dict[str, float]) -> Dict:
        """Apply topology mutation rules based on metrics.

        Args:
            state: Current BlogState
            metrics: Computed system metrics

        Returns:
            Updated state
        """
        # Rule 1: Low diversity → spawn explorer
        if metrics["diversity"] < self.diversity_threshold:
            state = self.mutator.spawn_agent(
                state,
                role="explorer",
                reason=f"Diversity {metrics['diversity']:.2f} < {self.diversity_threshold}",
            )

        # Rule 2: High coherence + low weight variance → merge redundant agents
        if metrics["coherence"] > self.coherence_threshold:
            weight_variance = metrics.get("weight_variance", 0.5)

            if weight_variance < 0.2:  # Agents trust everyone equally
                # Example: merge body and conclusion if too similar
                if "body" in state["active_agents"] and "conclusion" in state["active_agents"]:
                    state = self.mutator.merge_agents(
                        state,
                        agent1="body",
                        agent2="conclusion",
                        merged_role="body_conclusion",
                        reason=f"Coherence {metrics['coherence']:.2f} > {self.coherence_threshold}, "
                               f"Weight variance {weight_variance:.2f} < 0.2",
                    )

        # Rule 3: Low performance → remove underperforming agents
        if metrics["performance"] < self.performance_threshold:
            # Identify lowest scoring agent
            scores = state.get("scores", {})
            if scores:
                worst_agent = min(scores, key=scores.get)
                worst_score = scores[worst_agent]

                if worst_score < self.performance_threshold - 1.0:  # Significantly bad
                    state = self.mutator.remove_agent(
                        state,
                        role=worst_agent,
                        reason=f"Performance {worst_score:.2f} < {self.performance_threshold - 1.0}",
                    )

        return state

    def get_mutation_history(self) -> list:
        """Get history of all topology mutations.

        Returns:
            List of mutation records
        """
        return self.mutator.mutation_history
```

### 3. Configure Meta-Agent Settings

**File**: `.env.example`

Add meta-agent configuration:

```bash
# Meta-Agent Settings
META_AGENT_ENABLED=true                # Enable/disable meta-agent
META_DIVERSITY_THRESHOLD=0.3           # Spawn if diversity < this
META_COHERENCE_THRESHOLD=0.8           # Merge if coherence > this
META_PERFORMANCE_THRESHOLD=6.0         # Remove if performance < this
META_MIN_GENERATIONS=5                 # Wait this many generations before mutating
```

### 4. Integrate Meta-Agent into Pipeline

**File**: `src/hvas_mini/pipeline.py`

```python
import os
from hvas_mini.meta.meta_agent import MetaAgent


class HVASMiniPipeline:
    def __init__(self, persist_directory: str = "./data/memories"):
        # ... existing init ...

        # NEW: Meta-agent
        meta_enabled = os.getenv("META_AGENT_ENABLED", "true").lower() == "true"
        self.meta_agent = MetaAgent(
            diversity_threshold=float(os.getenv("META_DIVERSITY_THRESHOLD", "0.3")),
            coherence_threshold=float(os.getenv("META_COHERENCE_THRESHOLD", "0.8")),
            performance_threshold=float(os.getenv("META_PERFORMANCE_THRESHOLD", "6.0")),
            enabled=meta_enabled,
        )

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow with meta-agent."""

        workflow = StateGraph(BlogState)

        # Layer 1: Intro
        workflow.add_node("intro", self.agents["intro"])

        # Layer 2: Body & Conclusion (concurrent)
        workflow.add_node("body_and_conclusion", self._concurrent_layer_2)

        # Layer 3: Evaluation
        workflow.add_node("evaluate", self.evaluator)

        # Layer 4: Meta-agent (NEW)
        workflow.add_node("meta", self.meta_agent)

        # Layer 5: Evolution
        workflow.add_node("evolve", self._evolution_node)

        # Define execution flow
        workflow.set_entry_point("intro")
        workflow.add_edge("intro", "body_and_conclusion")
        workflow.add_edge("body_and_conclusion", "evaluate")
        workflow.add_edge("evaluate", "meta")  # NEW
        workflow.add_edge("meta", "evolve")
        workflow.add_edge("evolve", END)

        # Compile
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
```

### 5. Add Topology Visualization

**File**: `src/hvas_mini/visualization.py`

```python
class StreamVisualizer:
    # ... existing methods ...

    def create_topology_panel(self, state: BlogState) -> Panel:
        """Show current graph topology.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with topology info
        """
        active = state.get("active_agents", [])
        version = state.get("topology_version", 1)
        history = state.get("topology_history", [])

        topology_text = f"[bold]Version {version}[/bold]\n\n"
        topology_text += f"[cyan]Active Agents:[/cyan] {', '.join(active)}\n\n"

        # Show recent mutations
        if history:
            topology_text += "[yellow]Recent Mutations:[/yellow]\n"
            for mutation in history[-3:]:  # Last 3
                action = mutation.get("action", "unknown")
                reason = mutation.get("reason", "")

                if action == "spawn":
                    agent = mutation.get("agent", "")
                    topology_text += f"  ✨ Spawned {agent}\n"
                elif action == "remove":
                    agent = mutation.get("agent", "")
                    topology_text += f"  ❌ Removed {agent}\n"
                elif action == "merge":
                    agents = mutation.get("agents", [])
                    merged = mutation.get("merged_role", "")
                    topology_text += f"  🔗 Merged {agents} → {merged}\n"

                topology_text += f"     [dim]{reason}[/dim]\n"
        else:
            topology_text += "[dim]No mutations yet[/dim]"

        return Panel(
            topology_text,
            title="🔧 Graph Topology",
            border_style="magenta"
        )

    async def display_stream(self, state_stream: AsyncIterator[BlogState]):
        """Display with topology panel."""
        # ... existing code ...

        layout.split_column(
            Layout(name="status", size=10),
            Layout(name="topology", size=10),  # NEW
            Layout(name="concurrency", size=6),
            Layout(name="weights", size=8),
            Layout(name="memories", size=10),
            Layout(name="memory_age", size=6),
            Layout(name="evolution", size=8),
            Layout(name="logs", size=7),
        )

        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["topology"].update(self.create_topology_panel(state))  # NEW
                layout["concurrency"].update(self.create_concurrency_panel(state))
                layout["weights"].update(self.create_weights_panel(state))
                layout["memories"].update(self.create_memory_panel(state))
                layout["memory_age"].update(self.create_memory_age_panel(state))
                layout["evolution"].update(self.create_evolution_panel(state))
                layout["logs"].update(self.create_logs_panel(state))
```

### 6. Create Tests

**File**: `test_meta_agent.py`

```python
"""Tests for meta-agent system."""

import pytest
from hvas_mini.meta.meta_agent import MetaAgent
from hvas_mini.meta.metrics_monitor import MetricsMonitor
from hvas_mini.meta.graph_mutations import GraphMutator
from hvas_mini.state import create_initial_state


def test_diversity_calculation():
    """Test diversity metric calculation."""
    monitor = MetricsMonitor()

    # Identical outputs = low diversity
    outputs = {
        "agent1": "machine learning is great",
        "agent2": "machine learning is great",
    }
    diversity = monitor.calculate_diversity(outputs)
    assert diversity < 0.2

    # Different outputs = high diversity
    outputs = {
        "agent1": "machine learning is great",
        "agent2": "python programming rocks",
    }
    diversity = monitor.calculate_diversity(outputs)
    assert diversity > 0.5


def test_coherence_calculation():
    """Test coherence metric calculation."""
    monitor = MetricsMonitor()

    # Overlapping vocabulary = high coherence
    outputs = {
        "agent1": "machine learning with python",
        "agent2": "python for machine learning",
    }
    coherence = monitor.calculate_coherence(outputs)
    assert coherence > 0.5

    # Disjoint vocabulary = low coherence
    outputs = {
        "agent1": "cats dogs animals",
        "agent2": "numbers math equations",
    }
    coherence = monitor.calculate_coherence(outputs)
    assert coherence < 0.5


def test_spawn_agent():
    """Test spawning new agent."""
    mutator = GraphMutator()
    state = create_initial_state("test")

    initial_count = len(state["active_agents"])

    state = mutator.spawn_agent(
        state,
        role="explorer",
        reason="Low diversity"
    )

    assert len(state["active_agents"]) == initial_count + 1
    assert "explorer" in state["active_agents"]
    assert state["topology_version"] == 2
    assert len(state["topology_history"]) == 1


def test_remove_agent():
    """Test removing agent."""
    mutator = GraphMutator()
    state = create_initial_state("test")

    initial_count = len(state["active_agents"])

    state = mutator.remove_agent(
        state,
        role="conclusion",
        reason="Low performance"
    )

    assert len(state["active_agents"]) == initial_count - 1
    assert "conclusion" not in state["active_agents"]
    assert state["topology_version"] == 2


def test_merge_agents():
    """Test merging agents."""
    mutator = GraphMutator()
    state = create_initial_state("test")

    state = mutator.merge_agents(
        state,
        agent1="body",
        agent2="conclusion",
        merged_role="body_conclusion",
        reason="High coherence"
    )

    assert "body" not in state["active_agents"]
    assert "conclusion" not in state["active_agents"]
    assert "body_conclusion" in state["active_agents"]
    assert state["topology_version"] == 2


@pytest.mark.asyncio
async def test_meta_agent_spawns_on_low_diversity():
    """Test meta-agent spawns agent when diversity is low."""
    meta = MetaAgent(diversity_threshold=0.5, enabled=True)
    meta.generation_count = 10  # Skip warm-up period

    state = create_initial_state("test")
    state["intro"] = "machine learning is great"
    state["body"] = "machine learning is great"
    state["conclusion"] = "machine learning is great"
    state["scores"] = {"intro": 8.0, "body": 8.0, "conclusion": 8.0}

    initial_count = len(state["active_agents"])

    state = await meta(state)

    # Should spawn due to low diversity
    assert len(state["active_agents"]) > initial_count


@pytest.mark.asyncio
async def test_meta_agent_removes_on_low_performance():
    """Test meta-agent removes agent on low performance."""
    meta = MetaAgent(performance_threshold=6.0, enabled=True)
    meta.generation_count = 10

    state = create_initial_state("test")
    state["intro"] = "content"
    state["body"] = "content"
    state["conclusion"] = "content"
    state["scores"] = {"intro": 8.0, "body": 3.0, "conclusion": 8.0}  # Body underperforms

    initial_count = len(state["active_agents"])

    state = await meta(state)

    # May remove body agent
    assert len(state["active_agents"]) <= initial_count
```

## Deliverables Checklist

- [ ] `src/hvas_mini/state.py` updated with topology tracking fields
- [ ] `.env.example` updated with meta-agent configuration
- [ ] `src/hvas_mini/meta/__init__.py` created
- [ ] `src/hvas_mini/meta/metrics_monitor.py` created
- [ ] `src/hvas_mini/meta/graph_mutations.py` created
- [ ] `src/hvas_mini/meta/meta_agent.py` created
- [ ] `src/hvas_mini/pipeline.py` integrated with MetaAgent node
- [ ] `src/hvas_mini/visualization.py` updated with topology panel
- [ ] `test_meta_agent.py` created with passing tests
- [ ] Meta-agent triggers at least 1 mutation in 20 generations

## Acceptance Criteria

1. ✅ Meta-agent monitors diversity, coherence, performance metrics
2. ✅ Low diversity triggers agent spawn
3. ✅ High coherence + low weight variance triggers merge
4. ✅ Low performance triggers agent removal
5. ✅ Topology history tracks all mutations with timestamps
6. ✅ Visualization shows current topology and recent mutations
7. ✅ All existing tests still pass
8. ✅ New meta-agent tests pass

## Testing

```bash
cd worktrees/meta-agent

# Run new meta-agent tests
uv run pytest test_meta_agent.py -v

# Run all tests
uv run pytest

# Run demo with meta-agent enabled
export ANTHROPIC_API_KEY=your_key
export META_AGENT_ENABLED=true
export META_DIVERSITY_THRESHOLD=0.4
uv run python main.py
```

Expected output: After 10-20 generations, topology panel shows at least one mutation (spawn/merge/remove).

## Integration Notes

This milestone enables:
- Self-organizing graph topology
- Adaptive response to task requirements
- Redundancy detection and removal
- Foundation for M5 to visualize topology evolution
- Works best with M2 (uses weight variance in decisions)

## Next Steps

After merging M4 to main:
- M5 (visualization-v2) can create topology evolution animations
- Future: Meta-agent can spawn task-specific agents dynamically
- Future: Meta-agent can learn mutation policies from outcomes
