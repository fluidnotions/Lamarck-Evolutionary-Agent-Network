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

load_dotenv()

# Import dependencies from other branches
try:
    from hvas_mini.state import BlogState
    from hvas_mini.memory import MemoryManager
except ImportError:
    # For standalone development - use placeholder types
    from typing import TypedDict

    class BlogState(TypedDict):
        """Temporary type definition."""

        topic: str
        intro: str
        body: str
        conclusion: str
        scores: Dict[str, float]
        retrieved_memories: Dict[str, List[str]]
        parameter_updates: Dict[str, Dict[str, float]]
        generation_id: str
        timestamp: str
        stream_logs: List[str]

    class MemoryManager:
        """Placeholder for memory manager."""

        def __init__(self, *args, **kwargs):
            pass

        def retrieve(self, query: str) -> List[Dict]:
            return []

        def store(self, *args, **kwargs) -> str:
            return ""

        def get_stats(self) -> Dict:
            return {}


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
            temperature=float(os.getenv("BASE_TEMPERATURE", "0.7")),
        )

        # Evolutionary parameters
        self.parameters = {
            "temperature": float(os.getenv("BASE_TEMPERATURE", "0.7")),
            "score_history": [],
            "generation_count": 0,
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
        state["retrieved_memories"][self.role] = [m["content"] for m in memories]

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
            "timestamp": datetime.now().isoformat(),
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
                "parameters": json.dumps(self.parameters),
            },
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
            min(float(os.getenv("MAX_TEMPERATURE", "1.0")), new_temp),
        )

        self.parameters["temperature"] = new_temp

        # Log parameter change
        if os.getenv("SHOW_PARAMETER_CHANGES", "true").lower() == "true":
            state["parameter_updates"][self.role] = {
                "old_temperature": old_temp,
                "new_temperature": new_temp,
                "score": score,
                "avg_score": avg_score,
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
    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        """Generate content based on state and memories.

        Args:
            state: Current workflow state
            memories: Retrieved relevant memories

        Returns:
            Generated content string
        """
        pass
