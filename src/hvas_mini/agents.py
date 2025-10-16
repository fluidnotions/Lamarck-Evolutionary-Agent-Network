"""
Agent implementations for HVAS Mini.

Specialized agents that inherit from BaseAgent and provide domain-specific content generation.
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
    """Base agent with RAG memory, parameter evolution, and trust weighting."""

    def __init__(self, role: str, memory_manager: MemoryManager, trust_manager=None):
        self.role = role
        self.memory = memory_manager
        self.trust_manager = trust_manager  # NEW: Trust weighting
        self.llm = ChatAnthropic(
            model=os.getenv("MODEL_NAME", "claude-3-haiku-20240307"),
            temperature=float(os.getenv("BASE_TEMPERATURE", "0.7")),
        )
        self.parameters = {
            "temperature": float(os.getenv("BASE_TEMPERATURE", "0.7")),
            "score_history": [],
            "generation_count": 0,
        }
        self.enable_evolution = (
            os.getenv("ENABLE_PARAMETER_EVOLUTION", "true").lower() == "true"
        )
        self.pending_memory: Optional[Dict] = None

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

        # 3. NEW: Get weighted context from peer agents
        weighted_context = ""
        if self.trust_manager:
            peer_outputs = {}
            if self.role == "body" and state.get("intro"):
                peer_outputs["intro"] = state["intro"]
            elif self.role == "conclusion":
                if state.get("intro"):
                    peer_outputs["intro"] = state["intro"]
                if state.get("body"):
                    peer_outputs["body"] = state["body"]

            if peer_outputs:
                weighted_context = self.trust_manager.get_weighted_context(
                    self.role, peer_outputs
                )

        # 4. Generate content (async) with weighted context
        self.llm.temperature = self.parameters["temperature"]
        content = await self.generate_content(state, memories, weighted_context)

        # 5. Store in state
        state[self.content_key] = content

        # 6. Record timing
        agent_end = time.time()
        if "agent_timings" not in state:
            state["agent_timings"] = {}
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

    def store_memory(self, score: float):
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

    def evolve_parameters(self, score: float, state: BlogState):
        if not self.enable_evolution:
            return

        self.parameters["score_history"].append(score)
        self.parameters["generation_count"] += 1

        recent_scores = self.parameters["score_history"][-5:]
        avg_score = sum(recent_scores) / len(recent_scores)

        learning_rate = float(os.getenv("EVOLUTION_LEARNING_RATE", "0.1"))

        if avg_score < 6.0:
            delta = -learning_rate
        elif avg_score > 8.0:
            delta = learning_rate
        else:
            delta = (7.0 - avg_score) * learning_rate * 0.5

        old_temp = self.parameters["temperature"]
        new_temp = old_temp + delta
        new_temp = max(
            float(os.getenv("MIN_TEMPERATURE", "0.5")),
            min(float(os.getenv("MAX_TEMPERATURE", "1.0")), new_temp),
        )

        self.parameters["temperature"] = new_temp

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
        pass

    @abstractmethod
    async def generate_content(self, state: BlogState, memories: List[Dict], weighted_context: str = "") -> str:
        pass


class IntroAgent(BaseAgent):
    """Agent specialized in writing introductions."""

    @property
    def content_key(self) -> str:
        return "intro"

    async def generate_content(self, state: BlogState, memories: List[Dict], weighted_context: str = "") -> str:
        # Format memory examples
        memory_examples = ""
        if memories:
            memory_examples = "\n\n".join(
                [
                    f"Example (score: {m['score']:.1f}):\n{m['content']}"
                    for m in memories[:2]
                ]
            )
        else:
            memory_examples = "No previous examples available."

        # Construct prompt
        prompt = f"""Write an engaging introduction for a blog post about: {state['topic']}

Previous successful introductions on similar topics:
{memory_examples}

Requirements:
- 2-3 sentences
- Hook the reader immediately
- Mention the topic naturally
- Set expectations for what follows

Introduction:"""

        # Generate
        response = await self.llm.ainvoke(prompt)
        return response.content


class BodyAgent(BaseAgent):
    """Agent specialized in writing main body content."""

    @property
    def content_key(self) -> str:
        return "body"

    async def generate_content(self, state: BlogState, memories: List[Dict], weighted_context: str = "") -> str:
        # Use weighted context if available, otherwise fallback to direct access
        if weighted_context:
            context = f"PEER CONTEXT (trust-weighted):\n{weighted_context}"
        else:
            context = f"Introduction: {state.get('intro', 'Not yet written')}"

        # Format memory examples (truncated for prompt)
        memory_examples = ""
        if memories:
            memory_examples = "\n\n".join(
                [
                    f"Example body (score: {m['score']:.1f}):\n{m['content'][:200]}..."
                    for m in memories[:2]
                ]
            )

        # Construct prompt
        prompt = f"""Write the main body for a blog post about: {state['topic']}

{context}

Previous successful body sections:
{memory_examples}

Requirements:
- 3-4 paragraphs
- Informative and detailed
- Include specific examples or data
- Natural flow from the introduction

Body:"""

        # Generate
        response = await self.llm.ainvoke(prompt)
        return response.content


class ConclusionAgent(BaseAgent):
    """Agent specialized in writing conclusions."""

    @property
    def content_key(self) -> str:
        return "conclusion"

    async def generate_content(self, state: BlogState, memories: List[Dict], weighted_context: str = "") -> str:
        # Use weighted context if available, otherwise fallback to direct access
        if weighted_context:
            context = f"PEER CONTEXT (trust-weighted):\n{weighted_context}"
        else:
            intro_preview = state.get("intro", "Not yet written")
            body_preview = state.get("body", "Not yet written")[:200]
            context = f"""
Introduction: {intro_preview}

Body preview: {body_preview}...
"""

        # Format memory examples
        memory_examples = ""
        if memories:
            memory_examples = "\n".join(
                [f"Example: {m['content']}" for m in memories[:1]]
            )

        # Construct prompt
        prompt = f"""Write a conclusion for this blog post about: {state['topic']}

{context}

Previous successful conclusions:
{memory_examples}

Requirements:
- 2-3 sentences
- Summarize key points
- End with memorable statement
- Call to action or thought to ponder

Conclusion:"""

        # Generate
        response = await self.llm.ainvoke(prompt)
        return response.content


def create_agents(persist_directory: str = "./data/memories", trust_manager=None) -> Dict[str, BaseAgent]:
    """Create all specialized agents.

    Args:
        persist_directory: Where to persist memories
        trust_manager: Optional TrustManager for agent weighting

    Returns:
        Dictionary of agent instances
    """
    agents = {}

    for role in ["intro", "body", "conclusion"]:
        # Create memory manager for this agent
        memory = MemoryManager(
            collection_name=f"{role}_memories", persist_directory=persist_directory
        )

        # Create appropriate agent with trust_manager
        if role == "intro":
            agents[role] = IntroAgent(role, memory, trust_manager)
        elif role == "body":
            agents[role] = BodyAgent(role, memory, trust_manager)
        elif role == "conclusion":
            agents[role] = ConclusionAgent(role, memory, trust_manager)

    return agents
