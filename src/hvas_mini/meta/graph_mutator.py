"""
Graph topology mutation for meta-agent system.

Provides utilities for modifying LangGraph structure based on performance insights.
"""

from typing import Dict, List, Optional, Callable
from enum import Enum


class MutationType(Enum):
    """Types of graph mutations."""

    ADD_PARALLEL_LAYER = "add_parallel_layer"
    REMOVE_NODE = "remove_node"
    ADD_CONDITIONAL = "add_conditional"
    REORDER_NODES = "reorder_nodes"
    MODIFY_EDGES = "modify_edges"


class GraphMutation:
    """Represents a proposed graph topology change."""

    def __init__(
        self,
        mutation_type: MutationType,
        description: str,
        affected_nodes: List[str],
        rationale: str,
        estimated_impact: str,
    ):
        """Initialize mutation proposal.

        Args:
            mutation_type: Type of mutation
            description: Human-readable description
            affected_nodes: Nodes affected by this mutation
            rationale: Why this mutation is proposed
            estimated_impact: Expected impact on performance
        """
        self.mutation_type = mutation_type
        self.description = description
        self.affected_nodes = affected_nodes
        self.rationale = rationale
        self.estimated_impact = estimated_impact
        self.applied = False

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "mutation_type": self.mutation_type.value,
            "description": self.description,
            "affected_nodes": self.affected_nodes,
            "rationale": self.rationale,
            "estimated_impact": self.estimated_impact,
            "applied": self.applied,
        }


class GraphMutator:
    """Proposes and tracks graph topology mutations."""

    def __init__(self):
        """Initialize graph mutator."""
        self.mutation_history: List[GraphMutation] = []
        self.pending_mutations: List[GraphMutation] = []

    def propose_parallelization(
        self, nodes: List[str], rationale: str
    ) -> GraphMutation:
        """Propose parallel execution of independent nodes.

        Args:
            nodes: Nodes to execute in parallel
            rationale: Why these nodes should be parallel

        Returns:
            GraphMutation proposal
        """
        mutation = GraphMutation(
            mutation_type=MutationType.ADD_PARALLEL_LAYER,
            description=f"Execute {', '.join(nodes)} in parallel",
            affected_nodes=nodes,
            rationale=rationale,
            estimated_impact="Reduced total execution time through concurrency",
        )

        self.pending_mutations.append(mutation)
        return mutation

    def propose_node_removal(self, node: str, rationale: str) -> GraphMutation:
        """Propose removal of an underperforming node.

        Args:
            node: Node to remove
            rationale: Why this node should be removed

        Returns:
            GraphMutation proposal
        """
        mutation = GraphMutation(
            mutation_type=MutationType.REMOVE_NODE,
            description=f"Remove node '{node}' from graph",
            affected_nodes=[node],
            rationale=rationale,
            estimated_impact="Improved overall quality by removing low-performing component",
        )

        self.pending_mutations.append(mutation)
        return mutation

    def propose_conditional_routing(
        self, source: str, targets: List[str], condition: str
    ) -> GraphMutation:
        """Propose conditional routing between nodes.

        Args:
            source: Source node
            targets: Possible target nodes
            condition: Condition for routing

        Returns:
            GraphMutation proposal
        """
        mutation = GraphMutation(
            mutation_type=MutationType.ADD_CONDITIONAL,
            description=f"Add conditional routing from '{source}' to {targets}",
            affected_nodes=[source] + targets,
            rationale=condition,
            estimated_impact="Dynamic workflow adaptation based on runtime conditions",
        )

        self.pending_mutations.append(mutation)
        return mutation

    def propose_reordering(
        self, nodes: List[str], new_order: List[str], rationale: str
    ) -> GraphMutation:
        """Propose reordering of node execution.

        Args:
            nodes: Current node order
            new_order: Proposed new order
            rationale: Why reordering helps

        Returns:
            GraphMutation proposal
        """
        mutation = GraphMutation(
            mutation_type=MutationType.REORDER_NODES,
            description=f"Reorder nodes from {nodes} to {new_order}",
            affected_nodes=nodes + new_order,
            rationale=rationale,
            estimated_impact="Improved information flow and context availability",
        )

        self.pending_mutations.append(mutation)
        return mutation

    def mark_applied(self, mutation: GraphMutation):
        """Mark a mutation as applied.

        Args:
            mutation: Mutation that was applied
        """
        mutation.applied = True
        self.mutation_history.append(mutation)

        if mutation in self.pending_mutations:
            self.pending_mutations.remove(mutation)

    def get_pending_mutations(self) -> List[GraphMutation]:
        """Get list of pending mutations.

        Returns:
            List of unapplied mutations
        """
        return self.pending_mutations.copy()

    def get_mutation_history(self) -> List[Dict]:
        """Get history of applied mutations.

        Returns:
            List of mutation dictionaries
        """
        return [m.to_dict() for m in self.mutation_history]

    def clear_pending(self):
        """Clear all pending mutations."""
        self.pending_mutations.clear()

    def get_statistics(self) -> Dict:
        """Get statistics about mutations.

        Returns:
            Statistics dictionary
        """
        total_mutations = len(self.mutation_history)
        pending_count = len(self.pending_mutations)

        mutation_types = {}
        for mutation in self.mutation_history:
            mt = mutation.mutation_type.value
            mutation_types[mt] = mutation_types.get(mt, 0) + 1

        return {
            "total_applied": total_mutations,
            "pending": pending_count,
            "mutation_types": mutation_types,
        }
