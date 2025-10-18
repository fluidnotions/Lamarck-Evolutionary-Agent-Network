"""
Agent hierarchy structure definition.

Defines parent-child relationships and layer organization.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class AgentNode:
    """Single node in agent hierarchy."""
    role: str
    layer: int  # 1=coordinator, 2=content, 3=specialist
    children: List[str]  # Direct child roles
    semantic_vector: List[float]  # For M9 semantic distance


class AgentHierarchy:
    """Defines hierarchical relationships between agents."""

    def __init__(self):
        """Initialize 3-layer hierarchy."""
        self.nodes = {
            # Layer 1: Orchestration
            "coordinator": AgentNode(
                role="coordinator",
                layer=1,
                children=["intro", "body", "conclusion"],
                semantic_vector=[0.0, 1.0, 0.0]  # Integration-focused
            ),

            # Layer 2: Content Agents
            "intro": AgentNode(
                role="intro",
                layer=2,
                children=["researcher", "stylist"],
                semantic_vector=[0.8, 0.5, 0.2]  # Engaging, hook
            ),
            "body": AgentNode(
                role="body",
                layer=2,
                children=["researcher", "fact_checker"],
                semantic_vector=[0.5, 0.8, 0.9]  # Content-heavy
            ),
            "conclusion": AgentNode(
                role="conclusion",
                layer=2,
                children=["stylist"],
                semantic_vector=[0.7, 0.6, 0.3]  # Synthesis
            ),

            # Layer 3: Specialists
            "researcher": AgentNode(
                role="researcher",
                layer=3,
                children=[],  # Leaf node
                semantic_vector=[0.3, 0.9, 1.0]  # Factual, deep
            ),
            "fact_checker": AgentNode(
                role="fact_checker",
                layer=3,
                children=[],
                semantic_vector=[0.2, 0.8, 0.9]  # Accuracy-focused
            ),
            "stylist": AgentNode(
                role="stylist",
                layer=3,
                children=[],
                semantic_vector=[0.9, 0.4, 0.2]  # Style, tone
            ),
        }

    def get_children(self, agent_role: str) -> List[str]:
        """Get direct children of an agent.

        Args:
            agent_role: Parent agent role

        Returns:
            List of child agent roles
        """
        if agent_role not in self.nodes:
            return []
        return self.nodes[agent_role].children.copy()

    def get_parent(self, agent_role: str) -> str | None:
        """Get parent of an agent.

        Args:
            agent_role: Child agent role

        Returns:
            Parent role or None if top-level
        """
        for role, node in self.nodes.items():
            if agent_role in node.children:
                return role
        return None

    def get_layer(self, agent_role: str) -> int:
        """Get layer number for agent.

        Args:
            agent_role: Agent role

        Returns:
            Layer number (1-3)
        """
        if agent_role not in self.nodes:
            raise ValueError(f"Unknown agent: {agent_role}")
        return self.nodes[agent_role].layer

    def get_layer_agents(self, layer: int) -> List[str]:
        """Get all agents in a layer.

        Args:
            layer: Layer number (1-3)

        Returns:
            List of agent roles in that layer
        """
        return [
            role for role, node in self.nodes.items()
            if node.layer == layer
        ]

    def get_siblings(self, agent_role: str) -> List[str]:
        """Get sibling agents (same parent).

        Args:
            agent_role: Agent role

        Returns:
            List of sibling roles
        """
        parent = self.get_parent(agent_role)
        if not parent:
            return []

        siblings = self.get_children(parent)
        return [s for s in siblings if s != agent_role]

    def is_ancestor(self, potential_ancestor: str, agent_role: str) -> bool:
        """Check if one agent is an ancestor of another.

        Args:
            potential_ancestor: Potential ancestor role
            agent_role: Agent to check

        Returns:
            True if ancestor relationship exists
        """
        current = agent_role
        while current:
            parent = self.get_parent(current)
            if parent == potential_ancestor:
                return True
            current = parent
        return False

    def get_all_descendants(self, agent_role: str) -> List[str]:
        """Get all descendants (children, grandchildren, etc.).

        Args:
            agent_role: Root agent

        Returns:
            List of all descendant roles
        """
        descendants = []
        children = self.get_children(agent_role)

        for child in children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child))

        return descendants
