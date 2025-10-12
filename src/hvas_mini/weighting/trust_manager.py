"""
Manages trust relationships between agents.
"""

from typing import Dict, List
import numpy as np


class TrustManager:
    """Manages trust weights between agents."""

    def __init__(self, initial_weight: float = 0.5, learning_rate: float = 0.1):
        """Initialize trust manager.

        Args:
            initial_weight: Starting trust weight for new relationships
            learning_rate: How quickly weights adapt (0-1)
        """
        self.initial_weight = initial_weight
        self.learning_rate = learning_rate
        self.weights: Dict[str, Dict[str, float]] = {}

    def initialize_agent(self, agent_name: str, peers: List[str]):
        """Initialize weights for a new agent.

        Args:
            agent_name: Name of agent
            peers: List of peer agent names
        """
        if agent_name not in self.weights:
            self.weights[agent_name] = {}

        for peer in peers:
            if peer not in self.weights[agent_name]:
                self.weights[agent_name][peer] = self.initial_weight

    def get_weight(self, agent: str, peer: str) -> float:
        """Get trust weight from agent toward peer.

        Args:
            agent: Observing agent
            peer: Observed agent

        Returns:
            Trust weight (0-1)
        """
        if agent not in self.weights:
            return self.initial_weight
        return self.weights[agent].get(peer, self.initial_weight)

    def update_weight(
        self, agent: str, peer: str, performance_signal: float
    ) -> float:
        """Update trust weight based on peer performance.

        Uses gradient descent: w_new = w_old + α * (signal - w_old)

        Args:
            agent: Agent updating its trust
            peer: Peer being evaluated
            performance_signal: Performance metric (0-1, where 1 = perfect)

        Returns:
            New weight value
        """
        current_weight = self.get_weight(agent, peer)

        # Gradient descent toward signal
        delta = self.learning_rate * (performance_signal - current_weight)
        new_weight = np.clip(current_weight + delta, 0.0, 1.0)

        # Store
        if agent not in self.weights:
            self.weights[agent] = {}
        self.weights[agent][peer] = new_weight

        return new_weight

    def get_weighted_context(
        self, agent: str, peer_outputs: Dict[str, str]
    ) -> str:
        """Create weighted context from peer outputs.

        Args:
            agent: Agent requesting context
            peer_outputs: {peer_name: output_text}

        Returns:
            Weighted context string
        """
        weighted_parts = []

        for peer, output in peer_outputs.items():
            weight = self.get_weight(agent, peer)

            # Weight determines prominence in context
            if weight >= 0.7:
                prefix = "[HIGH TRUST]"
            elif weight >= 0.4:
                prefix = "[MEDIUM TRUST]"
            else:
                prefix = "[LOW TRUST]"

            weighted_parts.append(f"{prefix} {peer}: {output}")

        return "\n\n".join(weighted_parts)

    def get_all_weights(self) -> Dict[str, Dict[str, float]]:
        """Get all trust weights.

        Returns:
            Full weight matrix
        """
        return self.weights.copy()
