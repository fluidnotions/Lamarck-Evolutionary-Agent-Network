"""
Selection Strategies for Evolutionary Agent Pools

Implements Step 8 (EVOLVE) - Parent selection component:
- Choose best agents as parents for next generation
- Multiple strategies: tournament, proportionate, rank, diversity
- Balance fitness maximization with diversity preservation

Pure Python utilities (NOT LangGraph nodes).
Called by AgentPool.evolve_generation().
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, TYPE_CHECKING
import random
import numpy as np

if TYPE_CHECKING:
    from lean.agent_pool import AgentPool
    from lean.base_agent_v2 import BaseAgentV2


class SelectionStrategy(ABC):
    """Base class for parent selection.

    Selection determines which agents reproduce to create the next generation.
    Key tension: Exploit best performers vs explore diversity.

    This is a pure Python utility class (NOT a LangGraph node).
    Called by AgentPool.evolve_generation() to select parents.
    """

    def __init__(self):
        """Initialize selection strategy."""
        self.stats = {
            'total_selections': 0,
            'unique_parents_selected': set(),
            'diversity_scores': []
        }

    @abstractmethod
    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents from pool.

        Args:
            pool: AgentPool to select from (has .agents list)
            num_parents: Number of parents to select
            metadata: Optional context (domain, generation, etc.)

        Returns:
            List of selected parent agents

        Note:
            Parents can be selected multiple times (with replacement).
            This allows fitter agents to contribute more to next generation.
        """
        pass

    def get_stats(self) -> Dict:
        """Get selection statistics."""
        return {
            'strategy': self.__class__.__name__,
            'total_selections': self.stats['total_selections'],
            'unique_parents': len(self.stats['unique_parents_selected']),
            'avg_diversity': (
                np.mean(self.stats['diversity_scores'])
                if self.stats['diversity_scores'] else 0.0
            )
        }

    def _update_stats(self, selected_parents: List):
        """Update internal statistics."""
        self.stats['total_selections'] += 1
        for parent in selected_parents:
            parent_id = id(parent)
            self.stats['unique_parents_selected'].add(parent_id)

        # Calculate diversity (how many unique parents)
        unique_count = len(set(id(p) for p in selected_parents))
        diversity = unique_count / len(selected_parents) if selected_parents else 0
        self.stats['diversity_scores'].append(diversity)


class TournamentSelection(SelectionStrategy):
    """Tournament selection (k agents compete, best wins).

    Strategy:
    1. Randomly sample k agents from pool
    2. Select the fittest from this tournament
    3. Repeat until num_parents selected

    Best for:
    - Balancing selection pressure with diversity
    - Avoiding premature convergence
    - Giving weaker agents a chance

    Tournament size controls selection pressure:
    - k=2: Gentle pressure, high diversity
    - k=3-5: Moderate pressure (recommended)
    - k>5: Strong pressure, risk of monoculture
    """

    def __init__(self, tournament_size: int = 3):
        """Initialize tournament selection.

        Args:
            tournament_size: Number of agents per tournament (2-5 recommended)
        """
        super().__init__()
        self.tournament_size = tournament_size

    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents via tournament selection."""

        if not pool.agents:
            return []

        parents = []

        for _ in range(num_parents):
            # Run tournament
            tournament_size = min(self.tournament_size, len(pool.agents))
            competitors = random.sample(pool.agents, tournament_size)

            # Winner = highest fitness
            winner = max(competitors, key=lambda agent: agent.avg_fitness())
            parents.append(winner)

        self._update_stats(parents)
        return parents


class FitnessProportionateSelection(SelectionStrategy):
    """Fitness-proportionate selection (roulette wheel).

    Strategy:
    1. Calculate total fitness of pool
    2. Each agent gets probability proportional to fitness
    3. Spin roulette wheel num_parents times

    Best for:
    - When fitness differences are meaningful
    - Exploitation of best performers

    Drawback:
    - Can lead to premature convergence
    - Weak performers rarely selected
    - Requires positive fitness values
    """

    def __init__(self, min_fitness: float = 0.1):
        """Initialize fitness-proportionate selection.

        Args:
            min_fitness: Minimum fitness floor to avoid zero probability
        """
        super().__init__()
        self.min_fitness = min_fitness

    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents via roulette wheel."""

        if not pool.agents:
            return []

        # Get fitness values with floor
        fitnesses = [max(agent.avg_fitness(), self.min_fitness) for agent in pool.agents]
        total_fitness = sum(fitnesses)

        if total_fitness == 0:
            # Fallback to uniform random
            return random.choices(pool.agents, k=num_parents)

        # Calculate probabilities
        probabilities = [f / total_fitness for f in fitnesses]

        # Roulette wheel selection
        parents = random.choices(pool.agents, weights=probabilities, k=num_parents)

        self._update_stats(parents)
        return parents


class RankBasedSelection(SelectionStrategy):
    """Rank-based selection with elitism.

    Strategy:
    1. Sort agents by fitness
    2. Assign selection probability by rank (not raw fitness)
    3. Always keep top N (elitism)
    4. Select remaining from ranked distribution

    Best for:
    - When raw fitness values vary wildly
    - Ensuring best agents survive (elitism)
    - Moderate selection pressure

    Elitism: Top agents always selected (guaranteed survival).
    """

    def __init__(self, elitism_count: int = 1, pressure: float = 1.5):
        """Initialize rank-based selection.

        Args:
            elitism_count: Number of top agents to always select
            pressure: Selection pressure (1.0-2.0)
                     1.0 = uniform, 2.0 = strong bias toward top
        """
        super().__init__()
        self.elitism_count = elitism_count
        self.pressure = pressure

    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents via rank-based selection with elitism."""

        if not pool.agents:
            return []

        # Sort by fitness (descending)
        sorted_agents = sorted(
            pool.agents,
            key=lambda agent: agent.avg_fitness(),
            reverse=True
        )

        parents = []

        # Elitism: Always select top N
        elite_count = min(self.elitism_count, num_parents, len(sorted_agents))
        parents.extend(sorted_agents[:elite_count])

        # Rank-based selection for remaining slots
        remaining = num_parents - elite_count

        if remaining > 0 and len(sorted_agents) > 0:
            # Assign probabilities by rank
            n = len(sorted_agents)
            ranks = np.arange(1, n + 1)

            # Linear ranking: P(i) = (2 - pressure) / n + 2 * i * (pressure - 1) / (n * (n - 1))
            probabilities = (2 - self.pressure) / n + 2 * (n - ranks) * (self.pressure - 1) / (n * (n - 1))
            probabilities = probabilities / probabilities.sum()  # Normalize

            # Select remaining parents
            selected_indices = np.random.choice(
                len(sorted_agents),
                size=remaining,
                p=probabilities,
                replace=True
            )

            parents.extend([sorted_agents[i] for i in selected_indices])

        self._update_stats(parents)
        return parents


class DiversityAwareSelection(SelectionStrategy):
    """Diversity-aware selection (balance fitness and novelty).

    Strategy:
    1. Calculate diversity metric (embedding distance)
    2. Score agents by: α * fitness + (1-α) * diversity
    3. Select based on combined score

    Best for:
    - Preventing monoculture
    - Maintaining strategic diversity
    - Long-term adaptability

    Requires embeddings for diversity calculation.
    Fallback to fitness-proportionate if no embeddings.
    """

    def __init__(self, diversity_weight: float = 0.3):
        """Initialize diversity-aware selection.

        Args:
            diversity_weight: Weight for diversity (0.0-1.0)
                             0.0 = pure fitness, 1.0 = pure diversity
        """
        super().__init__()
        self.diversity_weight = diversity_weight
        self.fitness_weight = 1.0 - diversity_weight

    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents with diversity awareness."""

        if not pool.agents:
            return []

        # Try to calculate diversity scores
        diversity_scores = self._calculate_diversity_scores(pool)

        if diversity_scores is None:
            # Fallback to fitness-proportionate
            return FitnessProportionateSelection().select_parents(pool, num_parents, metadata)

        # Combine fitness and diversity
        combined_scores = []
        for i, agent in enumerate(pool.agents):
            fitness_norm = agent.avg_fitness() / 10.0  # Normalize to 0-1
            diversity_norm = diversity_scores[i]

            combined = (
                self.fitness_weight * fitness_norm +
                self.diversity_weight * diversity_norm
            )
            combined_scores.append(combined)

        # Select based on combined scores
        total_score = sum(combined_scores)

        if total_score == 0:
            return random.choices(pool.agents, k=num_parents)

        probabilities = [s / total_score for s in combined_scores]
        parents = random.choices(pool.agents, weights=probabilities, k=num_parents)

        self._update_stats(parents)
        return parents

    def _calculate_diversity_scores(self, pool: 'AgentPool') -> Optional[List[float]]:
        """Calculate diversity scores for agents in pool.

        Returns average distance to other agents (higher = more diverse).
        """
        try:
            # Get reasoning patterns from each agent
            agent_patterns = []
            for agent in pool.agents:
                patterns = agent.reasoning_memory.get_all_reasoning()

                if not patterns:
                    return None  # No patterns to measure diversity

                # Get embeddings
                embeddings = [p.get('embedding') for p in patterns if p.get('embedding')]

                if not embeddings:
                    return None  # No embeddings available

                # Average embedding for this agent
                avg_embedding = np.mean(embeddings, axis=0)
                agent_patterns.append(avg_embedding)

            if len(agent_patterns) < 2:
                return None

            # Calculate pairwise distances
            from sklearn.metrics.pairwise import cosine_distances

            agent_embeddings = np.array(agent_patterns)
            distances = cosine_distances(agent_embeddings)

            # Diversity score = average distance to other agents
            diversity_scores = []
            for i in range(len(pool.agents)):
                # Average distance to all other agents
                other_distances = [distances[i][j] for j in range(len(pool.agents)) if i != j]
                avg_distance = np.mean(other_distances) if other_distances else 0
                diversity_scores.append(avg_distance)

            # Normalize to 0-1
            max_dist = max(diversity_scores) if diversity_scores else 1
            if max_dist > 0:
                diversity_scores = [d / max_dist for d in diversity_scores]

            return diversity_scores

        except Exception:
            # If anything fails, return None to trigger fallback
            return None


# Convenience factory function
def create_selection_strategy(
    strategy_name: str = "tournament",
    **kwargs
) -> SelectionStrategy:
    """Create selection strategy by name.

    Args:
        strategy_name: One of: tournament, proportionate, rank, diversity
        **kwargs: Strategy-specific parameters

    Returns:
        SelectionStrategy instance

    Example:
        strategy = create_selection_strategy('tournament', tournament_size=3)
    """

    strategies = {
        'tournament': TournamentSelection,
        'proportionate': FitnessProportionateSelection,
        'rank': RankBasedSelection,
        'diversity': DiversityAwareSelection
    }

    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: {list(strategies.keys())}"
        )

    return strategy_class(**kwargs)
