# Agent Task: Memory Decay System (M3)

## Branch: `feature/memory-decay`

## Priority: HIGH (parallel after M1)

## Execution: PARALLEL (with M2, M4 after M1 completes)

## Objective

Implement timestamped memory decay so that older memories have reduced relevance during retrieval, preventing indefinite memory growth and ensuring recent, high-quality patterns dominate.

**Current**: All memories weighted equally by similarity, no time dimension
**Target**: Memories decay exponentially with age, pruning keeps top-N most relevant

## Dependencies

- ✅ M1 (async-orchestration) - MUST be merged first
- Can run parallel with M2 (agent-weighting) and M4 (meta-agent)

## Background

Current MemoryManager retrieves memories based solely on semantic similarity. Problems:
- Old, outdated patterns persist indefinitely
- No mechanism to prioritize recent learnings
- Memory collections grow without bound
- Retrieval quality degrades as noise accumulates

Solution: Add timestamp metadata and decay relevance over time using exponential decay formula.

## Tasks

### 1. Extend AgentMemory with Timestamp

**File**: `src/hvas_mini/state.py`

Modify AgentMemory Pydantic model:

```python
from datetime import datetime
from pydantic import BaseModel, Field

class AgentMemory(BaseModel):
    """Agent memory entry with decay metadata."""

    content: str = Field(..., description="Generated content")
    topic: str = Field(..., description="Topic that triggered generation")
    score: float = Field(..., ge=0, le=10, description="Evaluation score")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO 8601 timestamp"
    )
    retrieval_count: int = Field(default=0, description="Times retrieved")

    # NEW: Decay metadata
    last_retrieved: str | None = Field(
        default=None,
        description="Last time this memory was retrieved"
    )
    effective_score: float = Field(
        default=0.0,
        description="Score after applying decay (recalculated at retrieval)"
    )
```

### 2. Add Configuration for Decay Parameters

**File**: `.env.example`

Add decay configuration:

```bash
# Memory Decay Settings
MEMORY_DECAY_LAMBDA=0.01          # Decay rate (higher = faster decay)
MEMORY_MAX_AGE_DAYS=30            # Auto-delete memories older than this
MEMORY_PRUNE_TO_TOP_N=100         # Keep only top N memories per agent
MEMORY_MIN_EFFECTIVE_SCORE=3.0    # Delete memories below this after decay
```

### 3. Create Memory Decay Module

**New Directory**: `src/hvas_mini/memory/`

**File**: `src/hvas_mini/memory/__init__.py`
```python
"""
Memory management with timestamped decay.
"""

__all__ = ["DecayCalculator", "MemoryPruner"]
```

**File**: `src/hvas_mini/memory/decay.py`
```python
"""
Time-based memory decay calculations.
"""

import os
from datetime import datetime
from typing import Dict, List
import numpy as np


class DecayCalculator:
    """Calculates time-based decay for memory relevance."""

    def __init__(self, decay_lambda: float = 0.01):
        """Initialize decay calculator.

        Args:
            decay_lambda: Decay rate (higher = faster decay)
                         0.01 = ~63% relevance after 100 days
                         0.1  = ~63% relevance after 10 days
        """
        self.decay_lambda = decay_lambda

    def calculate_decay_factor(
        self, timestamp: str, current_time: datetime | None = None
    ) -> float:
        """Calculate decay factor for a timestamp.

        Formula: e^(-λ * Δt)
        where Δt is days elapsed since timestamp

        Args:
            timestamp: ISO 8601 timestamp string
            current_time: Reference time (defaults to now)

        Returns:
            Decay factor (0-1, where 1 = no decay)
        """
        if current_time is None:
            current_time = datetime.now()

        # Parse timestamp
        memory_time = datetime.fromisoformat(timestamp)

        # Calculate days elapsed
        delta_seconds = (current_time - memory_time).total_seconds()
        delta_days = delta_seconds / 86400.0

        # Exponential decay
        decay_factor = np.exp(-self.decay_lambda * delta_days)

        return max(0.0, min(1.0, decay_factor))

    def calculate_effective_score(
        self,
        similarity: float,
        original_score: float,
        timestamp: str,
        max_score: float = 10.0,
    ) -> float:
        """Calculate effective relevance with decay.

        Formula: relevance = similarity * e^(-λ * Δt) * (score / max_score)

        Args:
            similarity: Semantic similarity (0-1)
            original_score: Original evaluation score
            timestamp: When memory was created
            max_score: Maximum possible score

        Returns:
            Effective relevance score
        """
        decay_factor = self.calculate_decay_factor(timestamp)
        score_weight = original_score / max_score

        effective_score = similarity * decay_factor * score_weight

        return effective_score


class MemoryPruner:
    """Manages memory pruning to prevent unbounded growth."""

    def __init__(
        self,
        max_age_days: int = 30,
        prune_to_top_n: int = 100,
        min_effective_score: float = 3.0,
    ):
        """Initialize memory pruner.

        Args:
            max_age_days: Delete memories older than this
            prune_to_top_n: Keep only top N memories
            min_effective_score: Delete memories below this threshold
        """
        self.max_age_days = max_age_days
        self.prune_to_top_n = prune_to_top_n
        self.min_effective_score = min_effective_score

    def should_delete(
        self,
        timestamp: str,
        effective_score: float,
        current_time: datetime | None = None,
    ) -> bool:
        """Determine if memory should be deleted.

        Args:
            timestamp: Memory creation time
            effective_score: Score after decay
            current_time: Reference time (defaults to now)

        Returns:
            True if memory should be deleted
        """
        if current_time is None:
            current_time = datetime.now()

        # Check age threshold
        memory_time = datetime.fromisoformat(timestamp)
        age_days = (current_time - memory_time).total_seconds() / 86400.0

        if age_days > self.max_age_days:
            return True

        # Check score threshold
        if effective_score < self.min_effective_score:
            return True

        return False

    def prune_memories(
        self, memories: List[Dict], decay_calculator: DecayCalculator
    ) -> List[Dict]:
        """Prune memories based on age and relevance.

        Args:
            memories: List of memory dicts with metadata
            decay_calculator: DecayCalculator instance

        Returns:
            Filtered list of memories
        """
        current_time = datetime.now()
        pruned = []

        for memory in memories:
            # Calculate effective score
            effective_score = decay_calculator.calculate_effective_score(
                similarity=memory.get("similarity", 1.0),
                original_score=memory.get("score", 5.0),
                timestamp=memory["timestamp"],
            )

            # Check deletion criteria
            if self.should_delete(
                memory["timestamp"], effective_score, current_time
            ):
                continue

            # Add effective score to metadata
            memory["effective_score"] = effective_score
            pruned.append(memory)

        # Sort by effective score and keep top N
        pruned.sort(key=lambda m: m["effective_score"], reverse=True)
        return pruned[: self.prune_to_top_n]
```

### 4. Integrate Decay into MemoryManager

**File**: `src/hvas_mini/memory.py`

Modify MemoryManager to use decay:

```python
import os
from datetime import datetime
from hvas_mini.memory.decay import DecayCalculator, MemoryPruner


class MemoryManager:
    """Memory manager with time-based decay."""

    def __init__(self, collection_name: str, persist_directory: str):
        # ... existing init ...

        # NEW: Decay components
        decay_lambda = float(os.getenv("MEMORY_DECAY_LAMBDA", "0.01"))
        max_age_days = int(os.getenv("MEMORY_MAX_AGE_DAYS", "30"))
        prune_to_top_n = int(os.getenv("MEMORY_PRUNE_TO_TOP_N", "100"))
        min_effective_score = float(os.getenv("MEMORY_MIN_EFFECTIVE_SCORE", "3.0"))

        self.decay_calculator = DecayCalculator(decay_lambda=decay_lambda)
        self.pruner = MemoryPruner(
            max_age_days=max_age_days,
            prune_to_top_n=prune_to_top_n,
            min_effective_score=min_effective_score,
        )

    def store(self, memory: AgentMemory) -> None:
        """Store memory with timestamp metadata."""

        # Ensure timestamp is set
        if not memory.timestamp:
            memory.timestamp = datetime.now().isoformat()

        # Existing storage logic
        threshold = float(os.getenv("MEMORY_SCORE_THRESHOLD", "7.0"))
        if memory.score < threshold:
            return

        # Store in ChromaDB with timestamp metadata
        self.collection.add(
            documents=[memory.content],
            metadatas=[{
                "topic": memory.topic,
                "score": memory.score,
                "timestamp": memory.timestamp,  # NEW
                "retrieval_count": memory.retrieval_count,
            }],
            ids=[f"{self.collection_name}_{datetime.now().timestamp()}"],
        )

        self.total_stored += 1

    def retrieve(
        self, query: str, top_k: int = 5, apply_decay: bool = True
    ) -> List[Dict]:
        """Retrieve memories with decay-weighted relevance.

        Args:
            query: Search query (topic)
            top_k: Number of results before decay filtering
            apply_decay: Whether to apply time decay

        Returns:
            List of memories sorted by effective relevance
        """
        if self.collection.count() == 0:
            return []

        # Retrieve more than top_k since some may be pruned
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k * 3, self.collection.count()),
        )

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            similarity = 1.0 - results["distances"][0][i]  # Convert distance to similarity

            memory = {
                "content": doc,
                "topic": metadata.get("topic", ""),
                "score": metadata.get("score", 5.0),
                "timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                "retrieval_count": metadata.get("retrieval_count", 0),
                "similarity": similarity,
            }

            memories.append(memory)

        # Apply decay and pruning
        if apply_decay:
            memories = self._apply_decay_to_results(memories)

        # Update retrieval counts
        self.total_retrieved += len(memories)

        # Return top K after decay
        return memories[:top_k]

    def _apply_decay_to_results(self, memories: List[Dict]) -> List[Dict]:
        """Apply decay and prune memories.

        Args:
            memories: Raw retrieval results

        Returns:
            Decay-weighted and pruned memories
        """
        current_time = datetime.now()

        # Calculate effective scores
        for memory in memories:
            effective_score = self.decay_calculator.calculate_effective_score(
                similarity=memory.get("similarity", 1.0),
                original_score=memory.get("score", 5.0),
                timestamp=memory["timestamp"],
            )
            memory["effective_score"] = effective_score

        # Prune
        memories = self.pruner.prune_memories(memories, self.decay_calculator)

        # Sort by effective score
        memories.sort(key=lambda m: m.get("effective_score", 0), reverse=True)

        return memories

    def prune_collection(self) -> int:
        """Manually prune old/low-quality memories from collection.

        Returns:
            Number of memories deleted
        """
        # Retrieve all memories
        all_results = self.collection.get()

        if not all_results["ids"]:
            return 0

        ids_to_delete = []
        current_time = datetime.now()

        for i, metadata in enumerate(all_results["metadatas"]):
            timestamp = metadata.get("timestamp", datetime.now().isoformat())
            score = metadata.get("score", 5.0)

            # Calculate effective score with zero similarity (worst case)
            effective_score = self.decay_calculator.calculate_effective_score(
                similarity=0.0,
                original_score=score,
                timestamp=timestamp,
            )

            # Mark for deletion if too old or low score
            if self.pruner.should_delete(timestamp, effective_score, current_time):
                ids_to_delete.append(all_results["ids"][i])

        # Delete from collection
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        return len(ids_to_delete)

    def get_stats(self) -> Dict:
        """Get memory statistics including age distribution."""
        base_stats = {
            "total_memories": self.collection.count(),
            "total_stored": self.total_stored,
            "total_retrieved": self.total_retrieved,
        }

        # Calculate age distribution
        all_results = self.collection.get()
        if all_results["metadatas"]:
            timestamps = [m.get("timestamp") for m in all_results["metadatas"] if m.get("timestamp")]

            if timestamps:
                current_time = datetime.now()
                ages_days = [
                    (current_time - datetime.fromisoformat(ts)).total_seconds() / 86400.0
                    for ts in timestamps
                ]

                base_stats.update({
                    "avg_age_days": sum(ages_days) / len(ages_days),
                    "max_age_days": max(ages_days),
                    "min_age_days": min(ages_days),
                })

        return base_stats
```

### 5. Add Decay Visualization

**File**: `src/hvas_mini/visualization.py`

Add memory age panel:

```python
class StreamVisualizer:
    # ... existing methods ...

    def create_memory_age_panel(self, state: BlogState) -> Panel:
        """Show memory age distribution.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with age metrics
        """
        memories = state.get("retrieved_memories", {})

        if not any(memories.values()):
            return Panel(
                "[dim]No memories retrieved yet[/dim]",
                title="⏰ Memory Age",
                border_style="yellow"
            )

        age_text = ""

        # For each agent, show freshness of retrieved memories
        # (This requires adding age metadata to retrieved_memories in state)
        # For now, show count and note that detailed stats are in logs

        for role, mem_list in memories.items():
            if mem_list:
                age_text += f"[cyan]{role}:[/cyan] {len(mem_list)} memories retrieved\n"

        age_text += "\n[dim]Age distribution available in memory stats[/dim]"

        return Panel(
            age_text,
            title="⏰ Memory Freshness",
            border_style="yellow"
        )

    async def display_stream(self, state_stream: AsyncIterator[BlogState]):
        """Display with memory age panel."""
        # ... existing code ...

        layout.split_column(
            Layout(name="status", size=10),
            Layout(name="concurrency", size=6),
            Layout(name="weights", size=8),
            Layout(name="memories", size=10),
            Layout(name="memory_age", size=6),  # NEW
            Layout(name="evolution", size=8),
            Layout(name="logs", size=7),
        )

        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["concurrency"].update(self.create_concurrency_panel(state))
                layout["weights"].update(self.create_weights_panel(state))
                layout["memories"].update(self.create_memory_panel(state))
                layout["memory_age"].update(self.create_memory_age_panel(state))  # NEW
                layout["evolution"].update(self.create_evolution_panel(state))
                layout["logs"].update(self.create_logs_panel(state))
```

### 6. Add Periodic Pruning to Pipeline

**File**: `src/hvas_mini/pipeline.py`

Add pruning to evolution node:

```python
class HVASMiniPipeline:
    def __init__(self, persist_directory: str = "./data/memories"):
        # ... existing init ...
        self.generation_count = 0
        self.prune_interval = int(os.getenv("MEMORY_PRUNE_INTERVAL", "10"))

    async def _evolution_node(self, state: BlogState) -> BlogState:
        """Evolution with periodic memory pruning."""

        # Existing weight updates
        # ... weight update code ...

        # Existing memory storage and parameter evolution
        for role, agent in self.agents.items():
            score = state["scores"].get(role, 0)
            agent.store_memory(score)
            agent.evolve_parameters(score, state)

        # NEW: Periodic pruning
        self.generation_count += 1
        if self.generation_count % self.prune_interval == 0:
            total_deleted = 0
            for role, agent in self.agents.items():
                deleted = agent.memory.prune_collection()
                total_deleted += deleted

            if total_deleted > 0:
                state["stream_logs"].append(
                    f"[Memory] Pruned {total_deleted} old/low-quality memories"
                )

        return state
```

### 7. Create Tests

**File**: `test_memory_decay.py`

```python
"""Tests for memory decay system."""

import pytest
from datetime import datetime, timedelta
from hvas_mini.memory.decay import DecayCalculator, MemoryPruner


def test_decay_factor_calculation():
    """Test exponential decay calculation."""
    calculator = DecayCalculator(decay_lambda=0.01)

    # New memory (0 days old) should have decay factor ~1.0
    now = datetime.now()
    factor = calculator.calculate_decay_factor(now.isoformat(), current_time=now)
    assert factor == pytest.approx(1.0, abs=0.01)

    # Old memory (100 days) should have decay factor ~0.37 (e^-1)
    old_time = now - timedelta(days=100)
    factor = calculator.calculate_decay_factor(old_time.isoformat(), current_time=now)
    assert factor == pytest.approx(0.368, abs=0.01)


def test_effective_score_calculation():
    """Test effective score with decay."""
    calculator = DecayCalculator(decay_lambda=0.1)

    now = datetime.now()
    old_time = now - timedelta(days=10)

    # Perfect similarity + high score + old = decayed relevance
    effective = calculator.calculate_effective_score(
        similarity=1.0,
        original_score=9.0,
        timestamp=old_time.isoformat(),
    )

    # Should be significantly less than max (1.0 * 0.9 * e^-1)
    assert effective < 0.4


def test_memory_pruning():
    """Test pruning removes old/low-quality memories."""
    calculator = DecayCalculator(decay_lambda=0.01)
    pruner = MemoryPruner(
        max_age_days=30,
        prune_to_top_n=5,
        min_effective_score=3.0,
    )

    now = datetime.now()

    memories = [
        # Good recent memory
        {
            "content": "Good content",
            "timestamp": now.isoformat(),
            "score": 9.0,
            "similarity": 0.9,
        },
        # Old memory (should be deleted)
        {
            "content": "Old content",
            "timestamp": (now - timedelta(days=50)).isoformat(),
            "score": 8.0,
            "similarity": 0.8,
        },
        # Low effective score (should be deleted)
        {
            "content": "Low quality",
            "timestamp": (now - timedelta(days=5)).isoformat(),
            "score": 3.0,
            "similarity": 0.3,
        },
    ]

    pruned = pruner.prune_memories(memories, calculator)

    # Should keep only good recent memory
    assert len(pruned) == 1
    assert pruned[0]["content"] == "Good content"


def test_prune_to_top_n():
    """Test pruning keeps only top N memories."""
    calculator = DecayCalculator(decay_lambda=0.01)
    pruner = MemoryPruner(prune_to_top_n=3, max_age_days=1000)

    now = datetime.now()

    memories = [
        {"content": f"Memory {i}", "timestamp": now.isoformat(), "score": float(i), "similarity": 0.9}
        for i in range(10)
    ]

    pruned = pruner.prune_memories(memories, calculator)

    # Should keep only top 3
    assert len(pruned) == 3

    # Should be highest scoring
    scores = [m["score"] for m in pruned]
    assert scores == [9.0, 8.0, 7.0]


def test_should_delete():
    """Test deletion criteria."""
    pruner = MemoryPruner(max_age_days=30, min_effective_score=3.0)

    now = datetime.now()

    # Too old
    assert pruner.should_delete(
        (now - timedelta(days=50)).isoformat(),
        effective_score=5.0,
        current_time=now,
    ) is True

    # Too low score
    assert pruner.should_delete(
        now.isoformat(),
        effective_score=2.0,
        current_time=now,
    ) is True

    # Good memory
    assert pruner.should_delete(
        now.isoformat(),
        effective_score=7.0,
        current_time=now,
    ) is False
```

## Deliverables Checklist

- [ ] `src/hvas_mini/state.py` updated with decay metadata in AgentMemory
- [ ] `.env.example` updated with decay configuration
- [ ] `src/hvas_mini/memory/__init__.py` created
- [ ] `src/hvas_mini/memory/decay.py` created with DecayCalculator and MemoryPruner
- [ ] `src/hvas_mini/memory.py` modified to use decay in retrieval
- [ ] `src/hvas_mini/pipeline.py` integrated with periodic pruning
- [ ] `src/hvas_mini/visualization.py` updated with memory age panel
- [ ] `test_memory_decay.py` created with passing tests
- [ ] Old memories (>30 days) are automatically pruned

## Acceptance Criteria

1. ✅ Memories include ISO 8601 timestamps
2. ✅ Retrieval applies exponential decay: relevance = similarity * e^(-λ * Δt) * (score/10)
3. ✅ Memories older than MAX_AGE_DAYS are deleted
4. ✅ Only top PRUNE_TO_TOP_N memories kept per agent
5. ✅ Effective scores recalculated at each retrieval
6. ✅ Visualization shows memory age distribution
7. ✅ All existing tests still pass
8. ✅ New decay tests pass

## Testing

```bash
cd worktrees/memory-decay

# Run new decay tests
uv run pytest test_memory_decay.py -v

# Run all tests
uv run pytest

# Run demo with decay enabled
export ANTHROPIC_API_KEY=your_key
export MEMORY_DECAY_LAMBDA=0.05  # Faster decay for testing
uv run python main.py
```

Expected output: After 10 generations, memory stats show pruning activity and age distribution.

## Integration Notes

This milestone enables:
- Prevents memory pollution from outdated patterns
- Recent high-quality memories dominate retrieval
- Bounded memory growth (top-N per agent)
- Foundation for M4 meta-agent to analyze memory health
- M5 visualization can plot memory age histograms

## Next Steps

After merging M3 to main:
- M2 (agent-weighting) can use effective_score in weight updates
- M4 (meta-agent) can trigger pruning based on memory health metrics
- M5 (visualization-v2) can create age distribution histograms
