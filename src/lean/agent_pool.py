"""
Agent Pool - Population Management for Evolution

Integrates Phase 1 utilities (compaction, selection, reproduction) into
a complete population management system with generational evolution.

This is a HYBRID component:
- Data structure (not a LangGraph node)
- Methods called BY LangGraph nodes (specifically _evolve_node)
- Uses Python utility strategies for algorithms
"""

from typing import List, Dict, Optional
import random
import numpy as np

from lean.base_agent_v2 import BaseAgentV2
from lean.selection import SelectionStrategy, TournamentSelection
from lean.compaction import CompactionStrategy, HybridCompaction
from lean.reproduction import ReproductionStrategy, SexualReproduction
from lean.shared_rag import SharedRAG


class AgentPool:
    """Population of agents for a specific role.

    Manages a pool of agents that evolve over generations using:
    - Selection: Choose best agents as parents
    - Reproduction: Create offspring with inherited patterns
    - Compaction: Forget unsuccessful patterns before inheritance

    This is a data structure (not a LangGraph node).
    Its methods are called BY LangGraph nodes.
    """

    def __init__(
        self,
        role: str,
        initial_agents: List[BaseAgentV2],
        max_size: int = 5,
        selection_strategy: Optional[SelectionStrategy] = None,
        compaction_strategy: Optional[CompactionStrategy] = None
    ):
        """Initialize agent pool.

        Args:
            role: Role of agents in pool (intro, body, conclusion)
            initial_agents: Starting population (can be 1 agent that reproduces)
            max_size: Maximum pool size (population size)
            selection_strategy: How to select parents (default: TournamentSelection)
            compaction_strategy: How to compact reasoning (default: HybridCompaction)
        """
        self.role = role
        self.agents = initial_agents
        self.max_size = max_size

        # Default strategies if not provided
        self.selection_strategy = selection_strategy or TournamentSelection(tournament_size=3)
        self.compaction_strategy = compaction_strategy or HybridCompaction()

        self.generation = 0
        self.history = []

        # Track initial stats
        self._record_generation_stats()

    def select_agent(self, strategy: str = "fitness_proportionate") -> BaseAgentV2:
        """Select agent for task execution.

        Called by LangGraph generation nodes (_intro_node, _body_node, etc.)
        to choose which agent from the pool should execute the task.

        Args:
            strategy: Selection strategy (fitness_proportionate, random, best)

        Returns:
            Selected agent from pool
        """
        if not self.agents:
            raise ValueError(f"No agents in pool for role {self.role}")

        if len(self.agents) == 1:
            return self.agents[0]

        if strategy == "fitness_proportionate":
            # Roulette wheel selection
            fitnesses = [max(a.avg_fitness(), 0.1) for a in self.agents]
            total_fitness = sum(fitnesses)

            if total_fitness == 0:
                return random.choice(self.agents)

            probabilities = [f / total_fitness for f in fitnesses]
            return random.choices(self.agents, weights=probabilities, k=1)[0]

        elif strategy == "random":
            return random.choice(self.agents)

        elif strategy == "best":
            return max(self.agents, key=lambda a: a.avg_fitness())

        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")

    def evolve_generation(
        self,
        reproduction_strategy: ReproductionStrategy,
        shared_rag: SharedRAG
    ):
        """Create next generation of agents.

        Called by LangGraph _evolve_node() to trigger evolution.

        This is the core evolution cycle:
        1. Select parents (using SelectionStrategy)
        2. Create offspring (using ReproductionStrategy + CompactionStrategy)
        3. Replace population
        4. Track history

        Args:
            reproduction_strategy: How to create offspring
            shared_rag: Shared knowledge base
        """
        if not self.agents:
            return

        # 1. Select parents using SelectionStrategy
        num_parents = max(2, self.max_size // 2)
        parents = self.selection_strategy.select_parents(
            pool=self,
            num_parents=num_parents
        )

        if not parents:
            return  # No evolution possible

        # 2. Create offspring using ReproductionStrategy
        offspring = []
        for i in range(self.max_size):
            # Pair parents (with wraparound)
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)] if len(parents) > 1 else None

            # Reproduce (includes compaction!)
            child = reproduction_strategy.reproduce(
                parent1=parent1,
                parent2=parent2,
                compaction_strategy=self.compaction_strategy,  # FORGETTING!
                generation=self.generation + 1,
                shared_rag=shared_rag
            )

            offspring.append(child)

        # 3. Replace population
        self.agents = offspring
        self.generation += 1

        # 4. Track history
        self._record_generation_stats()

    def get_top_n(self, n: int) -> List[BaseAgentV2]:
        """Get top N agents by fitness."""
        return sorted(
            self.agents,
            key=lambda a: a.avg_fitness(),
            reverse=True
        )[:n]

    def get_random_lower_half(self) -> BaseAgentV2:
        """Get random agent from lower half (for diversity)."""
        sorted_agents = sorted(self.agents, key=lambda a: a.avg_fitness())
        lower_half = sorted_agents[:len(sorted_agents) // 2]

        if not lower_half:
            return self.agents[0] if self.agents else None

        return random.choice(lower_half)

    def avg_fitness(self) -> float:
        """Average fitness across pool."""
        if not self.agents:
            return 0.0

        return sum(a.avg_fitness() for a in self.agents) / len(self.agents)

    def measure_diversity(self) -> float:
        """Measure reasoning pattern diversity using embedding distances."""
        if len(self.agents) < 2:
            return 0.0

        try:
            # Get average embedding for each agent
            agent_embeddings = []
            for agent in self.agents:
                patterns = agent.reasoning_memory.get_all_reasoning()

                if not patterns:
                    continue

                # Get embeddings from patterns
                embeddings = [p.get('embedding') for p in patterns if p.get('embedding')]

                if not embeddings:
                    continue

                # Average embedding for this agent
                avg_embedding = np.mean(embeddings, axis=0)
                agent_embeddings.append(avg_embedding)

            if len(agent_embeddings) < 2:
                return 0.0

            # Calculate pairwise distances
            from sklearn.metrics.pairwise import cosine_distances

            embeddings_array = np.array(agent_embeddings)
            distances = cosine_distances(embeddings_array)

            # Average distance (excluding diagonal)
            n = len(agent_embeddings)
            total_distance = 0
            count = 0

            for i in range(n):
                for j in range(i + 1, n):
                    total_distance += distances[i][j]
                    count += 1

            avg_distance = total_distance / count if count > 0 else 0
            return float(avg_distance)

        except Exception:
            # If anything fails, return 0
            return 0.0

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        if not self.agents:
            return {
                'role': self.role,
                'generation': self.generation,
                'size': 0,
                'avg_fitness': 0.0,
                'fitness_range': (0.0, 0.0),
                'diversity': 0.0
            }

        fitnesses = [a.avg_fitness() for a in self.agents]

        return {
            'role': self.role,
            'generation': self.generation,
            'size': len(self.agents),
            'avg_fitness': np.mean(fitnesses),
            'fitness_range': (min(fitnesses), max(fitnesses)),
            'diversity': self.measure_diversity(),
            'agents': [
                {
                    'id': agent.agent_id,
                    'fitness': agent.avg_fitness(),
                    'patterns': len(agent.reasoning_memory.get_all_reasoning())
                }
                for agent in self.agents
            ]
        }

    def get_history(self) -> List[Dict]:
        """Get evolution history."""
        return self.history.copy()

    def _record_generation_stats(self):
        """Record statistics for current generation."""
        stats = {
            'generation': self.generation,
            'avg_fitness': self.avg_fitness(),
            'diversity': self.measure_diversity(),
            'size': len(self.agents)
        }

        self.history.append(stats)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AgentPool(role='{self.role}', "
            f"generation={self.generation}, "
            f"size={len(self.agents)}, "
            f"avg_fitness={self.avg_fitness():.2f})"
        )


# Convenience function for creating pools
def create_agent_pools(
    agents: Dict[str, BaseAgentV2],
    pool_size: int = 5,
    selection_strategy: Optional[SelectionStrategy] = None,
    compaction_strategy: Optional[CompactionStrategy] = None
) -> Dict[str, AgentPool]:
    """Create agent pools from initial agents."""
    pools = {}

    for role, agent in agents.items():
        pool = AgentPool(
            role=role,
            initial_agents=[agent],  # Start with 1 agent, will reproduce
            max_size=pool_size,
            selection_strategy=selection_strategy,
            compaction_strategy=compaction_strategy
        )
        pools[role] = pool

    return pools
