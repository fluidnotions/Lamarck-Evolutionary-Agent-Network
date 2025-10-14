"""
Factory for creating hierarchical agent instances.
"""

from typing import Dict, Tuple
from hvas_mini.agents import BaseAgent, IntroAgent, BodyAgent, ConclusionAgent
from hvas_mini.hierarchy.coordinator import CoordinatorAgent
from hvas_mini.hierarchy.specialists import ResearchAgent, FactCheckerAgent, StyleAgent
from hvas_mini.hierarchy.structure import AgentHierarchy
from hvas_mini.memory import MemoryManager


def create_hierarchical_agents(
    persist_directory: str = "./data/memories",
    trust_manager=None
) -> Tuple[Dict[str, BaseAgent], AgentHierarchy]:
    """Create all agents in hierarchical structure.

    Args:
        persist_directory: Where to persist memories
        trust_manager: Optional TrustManager

    Returns:
        (agents_dict, hierarchy) tuple
    """
    hierarchy = AgentHierarchy()
    agents = {}

    # Layer 1: Coordinator
    coordinator_memory = MemoryManager(
        collection_name="coordinator_memories",
        persist_directory=persist_directory
    )
    agents["coordinator"] = CoordinatorAgent(
        hierarchy=hierarchy,
        memory_manager=coordinator_memory,
        trust_manager=trust_manager
    )

    # Layer 2: Content Agents (existing, but now aware of children)
    for role in ["intro", "body", "conclusion"]:
        memory = MemoryManager(
            collection_name=f"{role}_memories",
            persist_directory=persist_directory
        )

        if role == "intro":
            agents[role] = IntroAgent(role, memory, trust_manager)
        elif role == "body":
            agents[role] = BodyAgent(role, memory, trust_manager)
        elif role == "conclusion":
            agents[role] = ConclusionAgent(role, memory, trust_manager)

    # Layer 3: Specialists
    specialist_roles = {
        "researcher": ResearchAgent,
        "fact_checker": FactCheckerAgent,
        "stylist": StyleAgent
    }

    for role, AgentClass in specialist_roles.items():
        memory = MemoryManager(
            collection_name=f"{role}_memories",
            persist_directory=persist_directory
        )
        agents[role] = AgentClass(
            memory_manager=memory,
            trust_manager=trust_manager
        )

    return agents, hierarchy
