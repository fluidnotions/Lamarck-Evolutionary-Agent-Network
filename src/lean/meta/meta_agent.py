"""
Meta-agent for dynamic graph optimization.

Analyzes system performance and proposes topology improvements.
"""

from typing import Dict, List, Optional
from lean.meta.metrics_monitor import MetricsMonitor
from lean.meta.graph_mutator import GraphMutator, GraphMutation


class MetaAgent:
    """High-level agent that optimizes the graph topology based on performance."""

    def __init__(
        self,
        metrics_monitor: Optional[MetricsMonitor] = None,
        graph_mutator: Optional[GraphMutator] = None,
        mutation_threshold: int = 5,
    ):
        """Initialize meta-agent.

        Args:
            metrics_monitor: MetricsMonitor instance
            graph_mutator: GraphMutator instance
            mutation_threshold: Minimum generations before proposing mutations
        """
        self.metrics = metrics_monitor or MetricsMonitor()
        self.mutator = graph_mutator or GraphMutator()
        self.mutation_threshold = mutation_threshold

    def analyze_and_propose(self, state: Dict) -> List[GraphMutation]:
        """Analyze current state and propose graph mutations.

        Args:
            state: Current BlogState with scores and timings

        Returns:
            List of proposed mutations
        """
        # Record this generation
        scores = state.get("scores", {})
        timings = state.get("agent_timings", {})

        self.metrics.record_generation(scores, timings)

        # Only propose mutations after sufficient data
        if self.metrics.generation_count < self.mutation_threshold:
            return []

        # Identify optimization opportunities
        opportunities = self.metrics.identify_optimization_opportunities()

        # Propose mutations based on opportunities
        proposals = []

        for opp in opportunities:
            if opp["type"] == "agent_performance":
                agent = opp["agent"]
                issues = opp["issues"]

                # If agent has consistently low performance, consider removal
                if "low_performance" in issues:
                    analysis = self.metrics.analyze_agent_performance(agent)
                    if analysis["avg_score"] < 5.0:  # Very low
                        mutation = self.mutator.propose_node_removal(
                            node=agent,
                            rationale=f"Agent '{agent}' consistently scores below 5.0 (avg: {analysis['avg_score']:.2f})",
                        )
                        proposals.append(mutation)

            elif opp["type"] == "parallelization":
                agents = opp.get("agents", [])
                if len(agents) >= 2:
                    mutation = self.mutator.propose_parallelization(
                        nodes=agents,
                        rationale="Agents have independent execution and could benefit from parallel execution",
                    )
                    proposals.append(mutation)

        return proposals

    def get_recommendations(self) -> Dict:
        """Get human-readable recommendations for system improvement.

        Returns:
            Recommendations dictionary
        """
        summary = self.metrics.get_summary_statistics()
        opportunities = self.metrics.identify_optimization_opportunities()
        pending_mutations = self.mutator.get_pending_mutations()

        recommendations = {
            "summary": summary,
            "opportunities": opportunities,
            "pending_mutations": [m.to_dict() for m in pending_mutations],
            "mutation_history": self.mutator.get_mutation_history(),
        }

        return recommendations

    def reset(self):
        """Reset meta-agent state."""
        self.metrics.reset()
        self.mutator.clear_pending()


def create_meta_agent(
    history_window: int = 10,
    low_score_threshold: float = 6.0,
    mutation_threshold: int = 5,
) -> MetaAgent:
    """Factory function to create a configured MetaAgent.

    Args:
        history_window: Generations to track
        low_score_threshold: Threshold for low performance
        mutation_threshold: Min generations before mutations

    Returns:
        Configured MetaAgent instance
    """
    metrics = MetricsMonitor(
        history_window=history_window,
        low_score_threshold=low_score_threshold,
    )

    mutator = GraphMutator()

    return MetaAgent(
        metrics_monitor=metrics,
        graph_mutator=mutator,
        mutation_threshold=mutation_threshold,
    )
