"""
Agent pool management for evolutionary multi-agent system.

Manages populations of competing agents within each role, providing selection
strategies and population statistics.
"""

from typing import List, Dict, Optional
import random
import uuid
from hvas_mini.agents import BaseAgent, IntroAgent, BodyAgent, ConclusionAgent
from hvas_mini.memory import MemoryManager


class AgentPool:
    """Manages a population of agents for a specific role."""

    def __init__(
        self,
        role: str,
        population_size: int = 5,
        min_size: int = 3,
        max_size: int = 8
    ):
        """Initialize agent pool.

        Args:
            role: Agent role (intro, body, conclusion)
            population_size: Initial number of agents
            min_size: Minimum population size
            max_size: Maximum population size
        """
        self.role = role
        self.min_size = min_size
        self.max_size = max_size
        self.agents: List[BaseAgent] = []
        self.generation = 0

    def add_agent(self, agent: BaseAgent):
        """Add agent to pool.

        Args:
            agent: Agent instance to add

        Raises:
            ValueError: If pool is at maximum size
        """
        if len(self.agents) >= self.max_size:
            raise ValueError(
                f"Pool at maximum size ({self.max_size}), cannot add agent"
            )
        self.agents.append(agent)

    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from pool.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if agent was removed, False if not found

        Raises:
            ValueError: If pool would go below minimum size
        """
        if len(self.agents) <= self.min_size:
            raise ValueError(
                f"Pool at minimum size ({self.min_size}), cannot remove agent"
            )

        for i, agent in enumerate(self.agents):
            if agent.agent_id == agent_id:
                self.agents.pop(i)
                return True

        return False

    def size(self) -> int:
        """Get current population size.

        Returns:
            Number of agents in pool
        """
        return len(self.agents)

    def select_agent(
        self,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.2,
        **kwargs
    ) -> BaseAgent:
        """Select an agent using specified strategy.

        Args:
            strategy: Selection strategy (epsilon_greedy, best, fitness_weighted, tournament)
            epsilon: Exploration parameter for epsilon_greedy
            **kwargs: Additional strategy-specific parameters

        Returns:
            Selected agent

        Raises:
            ValueError: If strategy is unknown or pool is empty
        """
        if not self.agents:
            raise ValueError("Cannot select from empty pool")

        if strategy == "best":
            return self._select_best()
        elif strategy == "epsilon_greedy":
            return self._select_epsilon_greedy(epsilon)
        elif strategy == "fitness_weighted":
            return self._select_fitness_weighted()
        elif strategy == "tournament":
            tournament_size = kwargs.get("tournament_size", 3)
            return self._select_tournament(tournament_size)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

    def _select_best(self) -> BaseAgent:
        """Select agent with highest average fitness.

        Returns:
            Best performing agent
        """
        return max(self.agents, key=lambda a: a.avg_fitness())

    def _select_epsilon_greedy(self, epsilon: float) -> BaseAgent:
        """Epsilon-greedy selection (explore vs exploit).

        Args:
            epsilon: Probability of random exploration

        Returns:
            Selected agent
        """
        if random.random() < epsilon:
            # Explore: random selection
            return random.choice(self.agents)
        else:
            # Exploit: best agent
            return self._select_best()

    def _select_fitness_weighted(self) -> BaseAgent:
        """Select agent with probability proportional to fitness.

        Returns:
            Selected agent
        """
        # Get fitness scores
        fitnesses = [a.avg_fitness() for a in self.agents]

        # Handle case where all fitnesses are 0
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choice(self.agents)

        # Calculate weights (normalize to probabilities)
        weights = [f / total_fitness for f in fitnesses]

        # Select using weights
        return random.choices(self.agents, weights=weights, k=1)[0]

    def _select_tournament(self, tournament_size: int = 3) -> BaseAgent:
        """Tournament selection (best from random subset).

        Args:
            tournament_size: Number of agents in tournament

        Returns:
            Winner of tournament
        """
        # Ensure tournament size doesn't exceed population
        tournament_size = min(tournament_size, len(self.agents))

        # Random subset
        tournament = random.sample(self.agents, tournament_size)

        # Best from tournament
        return max(tournament, key=lambda a: a.avg_fitness())

    def get_top_n(self, n: int = 2) -> List[BaseAgent]:
        """Get top N agents by fitness.

        Args:
            n: Number of top agents to return

        Returns:
            List of top N agents (sorted best to worst)
        """
        sorted_agents = sorted(
            self.agents,
            key=lambda a: a.avg_fitness(),
            reverse=True
        )
        return sorted_agents[:n]

    def get_random_lower_half(self) -> BaseAgent:
        """Get random agent from lower half of fitness distribution.

        Used for forced diversity in context distribution.

        Returns:
            Random agent from lower-performing half
        """
        sorted_agents = sorted(self.agents, key=lambda a: a.avg_fitness())
        lower_half = sorted_agents[:len(sorted_agents) // 2]

        if not lower_half:
            # If population too small, return random agent
            return random.choice(self.agents)

        return random.choice(lower_half)

    def get_all_stats(self) -> Dict:
        """Get comprehensive pool statistics.

        Returns:
            Dict with population statistics
        """
        if not self.agents:
            return {
                "role": self.role,
                "population_size": 0,
                "generation": self.generation,
                "agents": []
            }

        # Calculate population-level stats
        fitnesses = [a.avg_fitness() for a in self.agents if a.task_count > 0]

        if not fitnesses:
            avg_fitness = 0.0
            best_fitness = 0.0
            worst_fitness = 0.0
        else:
            avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness = max(fitnesses)
            worst_fitness = min(fitnesses)

        # Per-agent stats
        agent_stats = [
            {
                "agent_id": a.agent_id,
                "avg_fitness": a.avg_fitness(),
                "task_count": a.task_count,
                "is_specialist": a.is_specialist() if hasattr(a, 'is_specialist') else False
            }
            for a in self.agents
        ]

        return {
            "role": self.role,
            "population_size": len(self.agents),
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "best_fitness": best_fitness,
            "worst_fitness": worst_fitness,
            "agents": agent_stats
        }


def create_agent_pool(
    role: str,
    population_size: int = 5,
    persist_directory: str = "./data/memories",
    min_size: int = 3,
    max_size: int = 8
) -> AgentPool:
    """Factory function to create and populate an agent pool.

    Args:
        role: Agent role (intro, body, conclusion)
        population_size: Number of agents to create
        persist_directory: Memory storage directory
        min_size: Minimum population size
        max_size: Maximum population size

    Returns:
        Populated AgentPool instance

    Raises:
        ValueError: If role is unknown or population_size invalid
    """
    if role not in ["intro", "body", "conclusion"]:
        raise ValueError(f"Unknown role: {role}. Must be intro, body, or conclusion")

    if not (min_size <= population_size <= max_size):
        raise ValueError(
            f"population_size ({population_size}) must be between "
            f"min_size ({min_size}) and max_size ({max_size})"
        )

    # Create pool
    pool = AgentPool(role, population_size, min_size, max_size)

    # Map role to agent class
    agent_classes = {
        "intro": IntroAgent,
        "body": BodyAgent,
        "conclusion": ConclusionAgent
    }

    agent_class = agent_classes[role]

    # Create agents
    for i in range(population_size):
        # Unique agent ID
        agent_id = f"{role}_agent_{i+1}"

        # Create individual memory collection
        collection_name = f"{role}_agent_{i+1}_memories"
        memory_manager = MemoryManager(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

        # Create agent instance
        agent = agent_class(
            role=role,
            memory_manager=memory_manager,
            trust_manager=None  # Trust manager deprecated
        )

        # Set agent ID (needs to be added to BaseAgent)
        agent.agent_id = agent_id

        # Initialize fitness tracking attributes (will be formalized in M1.3)
        agent.fitness_history = []
        agent.domain_fitness = {}
        agent.task_count = 0

        # Add placeholder methods (will be implemented in M1.3)
        def _avg_fitness(self):
            if not self.fitness_history:
                return 0.0
            return sum(self.fitness_history) / len(self.fitness_history)

        # Bind method to instance
        import types
        agent.avg_fitness = types.MethodType(_avg_fitness, agent)

        # Add to pool
        pool.add_agent(agent)

    return pool
