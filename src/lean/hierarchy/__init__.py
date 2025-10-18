"""
Hierarchical agent system components.

Provides the structure and agents for a 3-layer hierarchy:
- Layer 1: Coordinator (orchestration)
- Layer 2: Content Agents (intro, body, conclusion)
- Layer 3: Specialists (researcher, fact_checker, stylist)
"""

from lean.hierarchy.structure import AgentHierarchy, AgentNode
from lean.hierarchy.coordinator import CoordinatorAgent
from lean.hierarchy.specialists import ResearchAgent, FactCheckerAgent, StyleAgent
from lean.hierarchy.factory import create_hierarchical_agents
from lean.hierarchy.executor import HierarchicalExecutor
from lean.hierarchy.semantic import (
    compute_semantic_distance,
    filter_context_by_distance,
    compute_context_weights,
    get_contextual_relevance,
)

__all__ = [
    "AgentHierarchy",
    "AgentNode",
    "CoordinatorAgent",
    "ResearchAgent",
    "FactCheckerAgent",
    "StyleAgent",
    "create_hierarchical_agents",
    "HierarchicalExecutor",
    "compute_semantic_distance",
    "filter_context_by_distance",
    "compute_context_weights",
    "get_contextual_relevance",
]
