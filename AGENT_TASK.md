# Agent Task: Base Agent Framework

## Branch: `feature/base-agent`

## Priority: HIGH - Required for all agents

## Execution: SEQUENTIAL (blocks feature/specialized-agents)

## Objective
Implement the abstract `BaseAgent` class with RAG memory integration, parameter evolution, and LangGraph compatibility.

## Dependencies
- ✅ feature/project-foundation
- ✅ feature/state-management (must be merged)
- ✅ feature/memory-system (must be merged)

## Tasks

### 1. Create `src/hvas_mini/agents.py`

Implement according to spec (section 3.2):

```python
"""
Agent implementations for HVAS Mini.

Base agent provides RAG memory, parameter evolution, and LangGraph integration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from langchain_anthropic import ChatAnthropic
from datetime import datetime
import os
from dotenv import load_dotenv
import json

from hvas_mini.state import BlogState
from hvas_mini.memory import MemoryManager

load_dotenv()


class BaseAgent(ABC):
    """Base agent with RAG memory and parameter evolution.

    Each agent:
    - Has its own ChromaDB memory collection
    - Retrieves relevant memories before generation
    - Evolves parameters based on performance scores
    - Stores successful outputs for future use
    """

    def __init__(self, role: str, memory_manager: MemoryManager):
        """Initialize base agent.

        Args:
            role: Agent role name (intro, body, conclusion)
            memory_manager: Memory manager for this agent's memories
        """
        self.role = role
        self.memory = memory_manager

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=os.getenv("MODEL_NAME", "claude-3-haiku-20240307"),
            temperature=float(os.getenv("BASE_TEMPERATURE", "0.7"))
        )

        # Evolutionary parameters
        self.parameters = {
            "temperature": float(os.getenv("BASE_TEMPERATURE", "0.7")),
            "score_history": [],
            "generation_count": 0
        }

        # Configuration
        self.enable_evolution = (
            os.getenv("ENABLE_PARAMETER_EVOLUTION", "true").lower() == "true"
        )

        # Pending memory storage
        self.pending_memory: Optional[Dict] = None

    async def __call__(self, state: BlogState) -> BlogState:
        """Execute agent - called by LangGraph.

        Args:
            state: Current workflow state

        Returns:
            Updated state with this agent's content
        """
        # 1. Retrieve relevant memories
        memories = self.memory.retrieve(state["topic"])
        state["retrieved_memories"][self.role] = [
            m["content"] for m in memories
        ]

        # 2. Log retrieval for visualization
        if os.getenv("SHOW_MEMORY_RETRIEVAL", "true").lower() == "true":
            state["stream_logs"].append(
                f"[{self.role}] Retrieved {len(memories)} memories"
            )

        # 3. Generate content with current parameters
        self.llm.temperature = self.parameters["temperature"]
        content = await self.generate_content(state, memories)

        # 4. Store in state
        state[self.content_key] = content

        # 5. Prepare for memory storage (will be stored after evaluation)
        self.pending_memory = {
            "content": content,
            "topic": state["topic"],
            "timestamp": datetime.now().isoformat()
        }

        return state

    def store_memory(self, score: float):
        """Store successful content in memory.

        Args:
            score: Quality score from evaluator
        """
        if self.pending_memory is None:
            return

        memory_id = self.memory.store(
            content=self.pending_memory["content"],
            topic=self.pending_memory["topic"],
            score=score,
            metadata={
                "timestamp": self.pending_memory["timestamp"],
                "parameters": json.dumps(self.parameters)
            }
        )

        if memory_id:
            # Memory was stored (met threshold)
            pass

    def evolve_parameters(self, score: float, state: BlogState):
        """Adjust parameters based on performance.

        Args:
            score: Latest quality score
            state: Current state (for logging parameter changes)
        """
        if not self.enable_evolution:
            return

        # Track score history
        self.parameters["score_history"].append(score)
        self.parameters["generation_count"] += 1

        # Calculate rolling average (last 5)
        recent_scores = self.parameters["score_history"][-5:]
        avg_score = sum(recent_scores) / len(recent_scores)

        # Adjust temperature based on performance
        learning_rate = float(os.getenv("EVOLUTION_LEARNING_RATE", "0.1"))

        if avg_score < 6.0:
            # Poor performance: reduce randomness
            delta = -learning_rate
        elif avg_score > 8.0:
            # Good performance: increase creativity
            delta = learning_rate
        else:
            # Stable performance: minor adjustments toward target
            delta = (7.0 - avg_score) * learning_rate * 0.5

        # Apply change with bounds
        old_temp = self.parameters["temperature"]
        new_temp = old_temp + delta
        new_temp = max(
            float(os.getenv("MIN_TEMPERATURE", "0.5")),
            min(float(os.getenv("MAX_TEMPERATURE", "1.0")), new_temp)
        )

        self.parameters["temperature"] = new_temp

        # Log parameter change
        if os.getenv("SHOW_PARAMETER_CHANGES", "true").lower() == "true":
            state["parameter_updates"][self.role] = {
                "old_temperature": old_temp,
                "new_temperature": new_temp,
                "score": score,
                "avg_score": avg_score
            }

    @property
    @abstractmethod
    def content_key(self) -> str:
        """State key for this agent's content.

        Returns:
            Key name (e.g., 'intro', 'body', 'conclusion')
        """
        pass

    @abstractmethod
    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict]
    ) -> str:
        """Generate content based on state and memories.

        Args:
            state: Current workflow state
            memories: Retrieved relevant memories

        Returns:
            Generated content string
        """
        pass
```

### 2. Create `src/hvas_mini/evolution.py`

Utility functions for parameter evolution:

```python
"""
Parameter evolution utilities for HVAS Mini.
"""

from typing import Dict, List


def calculate_temperature_adjustment(
    score_history: List[float],
    current_temp: float,
    learning_rate: float = 0.1,
    min_temp: float = 0.5,
    max_temp: float = 1.0
) -> float:
    """Calculate new temperature based on score history.

    Args:
        score_history: Recent scores
        current_temp: Current temperature value
        learning_rate: Learning rate for adjustments
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature

    Returns:
        New temperature value
    """
    if not score_history:
        return current_temp

    # Use last 5 scores
    recent = score_history[-5:]
    avg = sum(recent) / len(recent)

    # Calculate delta
    if avg < 6.0:
        delta = -learning_rate
    elif avg > 8.0:
        delta = learning_rate
    else:
        delta = (7.0 - avg) * learning_rate * 0.5

    # Apply bounds
    new_temp = current_temp + delta
    return max(min_temp, min(max_temp, new_temp))


def get_evolution_stats(parameters: Dict) -> Dict:
    """Get evolution statistics for an agent.

    Args:
        parameters: Agent parameters dictionary

    Returns:
        Statistics dictionary
    """
    history = parameters.get("score_history", [])

    if not history:
        return {
            "generations": 0,
            "avg_score": 0.0,
            "current_temp": parameters.get("temperature", 0.7)
        }

    return {
        "generations": parameters.get("generation_count", 0),
        "avg_score": sum(history) / len(history),
        "recent_avg": sum(history[-5:]) / len(history[-5:]),
        "best_score": max(history),
        "worst_score": min(history),
        "current_temp": parameters.get("temperature", 0.7)
    }
```

### 3. Create Tests

Create `test_agents.py`:

```python
"""Tests for base agent."""

from hvas_mini.agents import BaseAgent
from hvas_mini.state import create_initial_state
from hvas_mini.memory import MemoryManager
from hvas_mini.evolution import calculate_temperature_adjustment
import tempfile
import shutil
import pytest


class TestAgent(BaseAgent):
    """Concrete agent for testing."""

    @property
    def content_key(self) -> str:
        return "test"

    async def generate_content(self, state, memories):
        return "Test content"


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def test_agent(temp_dir):
    memory = MemoryManager("test", persist_directory=temp_dir)
    return TestAgent("test", memory)


@pytest.mark.asyncio
async def test_agent_execution(test_agent):
    """Test agent can execute."""
    state = create_initial_state("test topic")
    result = await test_agent(state)

    assert result["test"] == "Test content"
    assert "test" in result["retrieved_memories"]


def test_parameter_evolution(test_agent):
    """Test parameter evolution."""
    initial_temp = test_agent.parameters["temperature"]
    state = create_initial_state("test")

    # Simulate good performance
    test_agent.evolve_parameters(9.0, state)

    # Temperature should increase
    assert test_agent.parameters["temperature"] >= initial_temp


def test_temperature_calculation():
    """Test temperature adjustment calculation."""
    # Good scores should increase temp
    new_temp = calculate_temperature_adjustment(
        score_history=[8.5, 9.0, 8.8],
        current_temp=0.7
    )
    assert new_temp > 0.7

    # Poor scores should decrease temp
    new_temp = calculate_temperature_adjustment(
        score_history=[5.0, 5.5, 5.2],
        current_temp=0.7
    )
    assert new_temp < 0.7
```

## Deliverables Checklist

- [ ] `src/hvas_mini/agents.py` with complete `BaseAgent`
- [ ] `src/hvas_mini/evolution.py` with utility functions
- [ ] Memory retrieval integration
- [ ] Parameter evolution logic
- [ ] LangGraph `__call__` compatibility
- [ ] `test_agents.py` with passing tests
- [ ] Complete docstrings

## Acceptance Criteria

1. ✅ BaseAgent can be instantiated with memory manager
2. ✅ `__call__` method works with BlogState
3. ✅ Memory retrieval functions correctly
4. ✅ Parameter evolution adjusts temperature
5. ✅ Successful content gets stored in memory
6. ✅ All tests pass: `uv run pytest test_agents.py`
7. ✅ Configuration loaded from .env

## Testing

```bash
cd worktrees/base-agent
uv run pytest test_agents.py -v
```

## Next Steps

After completion, merge to main and proceed with:
- feature/specialized-agents (will inherit from BaseAgent)
