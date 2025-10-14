"""
Hierarchical agent system components.

Provides the structure and agents for a 3-layer hierarchy:
- Layer 1: Coordinator (orchestration)
- Layer 2: Content Agents (intro, body, conclusion)
- Layer 3: Specialists (researcher, fact_checker, stylist)
"""

from hvas_mini.hierarchy.structure import AgentHierarchy, AgentNode
from hvas_mini.hierarchy.coordinator import CoordinatorAgent
from hvas_mini.hierarchy.specialists import ResearchAgent, FactCheckerAgent, StyleAgent
from hvas_mini.hierarchy.factory import create_hierarchical_agents

__all__ = [
    "AgentHierarchy",
    "AgentNode",
    "CoordinatorAgent",
    "ResearchAgent",
    "FactCheckerAgent",
    "StyleAgent",
    "create_hierarchical_agents",
]
