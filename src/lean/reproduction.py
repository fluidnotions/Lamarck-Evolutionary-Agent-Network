"""
Reproduction Strategies for Agent Evolution

Implements Step 8 (EVOLVE) - offspring creation component:
- Create new agents that inherit reasoning patterns from parent(s)
- Asexual reproduction (one parent) or sexual (two parents with crossover)
- Compaction before inheritance (forget unsuccessful patterns)
- Mutation for exploration

Pure Python utilities (NOT LangGraph nodes).
Called by AgentPool.evolve_generation().
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, TYPE_CHECKING
import uuid
import os
import random

if TYPE_CHECKING:
    from lean.base_agent import BaseAgent
    from lean.compaction import CompactionStrategy
    from lean.shared_rag import SharedRAG
    from lean.reasoning_memory import ReasoningMemory


class ReproductionStrategy(ABC):
    """Base class for agent reproduction.

    Reproduction creates offspring agents that inherit compacted reasoning
    patterns from parent(s). This is Lamarckian evolution: agents pass on
    their learned cognitive strategies, not genes.

    This is a pure Python utility class (NOT a LangGraph node).
    Called by AgentPool.evolve_generation() to create offspring.
    """

    def __init__(self, mutation_rate: float = 0.0):
        """Initialize reproduction strategy.

        Args:
            mutation_rate: Probability of mutation (0.0-1.0)
        """
        self.mutation_rate = mutation_rate
        self.stats = {
            'total_reproductions': 0,
            'total_mutations': 0
        }

    @abstractmethod
    def reproduce(
        self,
        parent1: 'BaseAgent',
        parent2: Optional['BaseAgent'],
        compaction_strategy: 'CompactionStrategy',
        generation: int,
        shared_rag: Optional['SharedRAG'] = None
    ) -> 'BaseAgent':
        """Create offspring from parent(s).

        Args:
            parent1: First parent agent
            parent2: Second parent (None for asexual)
            compaction_strategy: How to compact inherited patterns
            generation: Generation number for offspring
            shared_rag: Shared knowledge base (optional)

        Returns:
            New agent with inherited reasoning patterns

        Note:
            The offspring inherits COMPACTED patterns only.
            Parent patterns (100+) → Compact to 20-30 → Inherit
        """
        pass

    def _get_stats(self) -> Dict:
        """Get reproduction statistics."""
        return {
            'strategy': self.__class__.__name__,
            'total_reproductions': self.stats['total_reproductions'],
            'total_mutations': self.stats['total_mutations'],
            'mutation_rate': self.mutation_rate
        }

    def _update_stats(self, mutated: bool = False):
        """Update internal statistics."""
        self.stats['total_reproductions'] += 1
        if mutated:
            self.stats['total_mutations'] += 1


class AsexualReproduction(ReproductionStrategy):
    """Asexual reproduction (one parent → one offspring).

    Strategy:
    1. Get parent's reasoning patterns
    2. Compact patterns to best 20-30
    3. Create offspring with inherited patterns
    4. Optional mutation (add noise)

    Best for:
    - Exploiting proven strategies
    - Stable environments
    - When diversity not critical

    Drawback:
    - Limited exploration
    - Can get stuck in local optima
    """

    def reproduce(
        self,
        parent1: 'BaseAgent',
        parent2: Optional['BaseAgent'],
        compaction_strategy: 'CompactionStrategy',
        generation: int,
        shared_rag: Optional['SharedRAG'] = None
    ) -> 'BaseAgent':
        """Create offspring from single parent."""

        # Get parent's reasoning patterns
        parent_patterns = parent1.reasoning_memory.get_all_reasoning()

        # Compact to best patterns
        max_inherited = int(os.getenv('INHERITED_REASONING_SIZE', '100'))
        inherited_patterns = compaction_strategy.compact(
            parent_patterns,
            max_size=max_inherited
        )

        # Apply mutation if specified
        mutated = False
        if random.random() < self.mutation_rate:
            inherited_patterns = self._mutate_patterns(inherited_patterns)
            mutated = True

        # Create offspring (inherit system_prompt from parent)
        offspring = self._create_offspring(
            role=parent1.role,
            generation=generation,
            inherited_patterns=inherited_patterns,
            shared_rag=shared_rag or parent1.shared_rag,
            system_prompt=parent1.system_prompt
        )

        self._update_stats(mutated)
        return offspring

    def _mutate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Apply mutation to patterns (add randomness).

        Simple mutation: slightly adjust scores.
        """
        mutated = []
        for pattern in patterns:
            p = pattern.copy()
            # Add small random noise to score
            if 'score' in p:
                noise = (random.random() - 0.5) * 1.0  # ±0.5
                p['score'] = max(0, min(10, p['score'] + noise))
            mutated.append(p)
        return mutated

    def _create_offspring(
        self,
        role: str,
        generation: int,
        inherited_patterns: List[Dict],
        shared_rag: 'SharedRAG',
        system_prompt: Optional[str] = None
    ) -> 'BaseAgent':
        """Create offspring agent with inherited patterns."""
        from lean.base_agent import IntroAgent, BodyAgent, ConclusionAgent
        from lean.reasoning_memory import ReasoningMemory

        # Create agent with inherited patterns
        child_id = f"{role}_gen{generation}_child{uuid.uuid4().hex[:6]}"

        # Create reasoning memory with inherited patterns
        memory = ReasoningMemory(
            collection_name=f"{role}_{child_id}_reasoning",
            inherited_reasoning=inherited_patterns
        )

        # Map role to agent class
        agent_classes = {
            'intro': IntroAgent,
            'body': BodyAgent,
            'conclusion': ConclusionAgent
        }

        agent_class = agent_classes.get(role)
        if not agent_class:
            raise ValueError(f"Unknown role: {role}")

        # Create agent directly (inherit system_prompt from parent)
        agent = agent_class(
            role=role,
            agent_id=f"{role}_{child_id}",
            reasoning_memory=memory,
            shared_rag=shared_rag,
            system_prompt=system_prompt
        )

        return agent


class SexualReproduction(ReproductionStrategy):
    """Sexual reproduction (two parents → one offspring with crossover).

    Strategy:
    1. Get patterns from both parents
    2. Combine patterns
    3. Compact combined set to best 20-30
    4. Create offspring with inherited patterns
    5. Optional mutation

    Best for:
    - Exploring strategy combinations
    - Dynamic environments
    - Maximum diversity

    Crossover strategies:
    - Simple: Combine all patterns, then compact
    - Advanced: Interleave patterns by score/diversity
    """

    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.5):
        """Initialize sexual reproduction.

        Args:
            mutation_rate: Probability of mutation (0.0-1.0)
            crossover_rate: How to balance parents (0.5 = equal contribution)
        """
        super().__init__(mutation_rate)
        self.crossover_rate = crossover_rate

    def reproduce(
        self,
        parent1: 'BaseAgent',
        parent2: Optional['BaseAgent'],
        compaction_strategy: 'CompactionStrategy',
        generation: int,
        shared_rag: Optional['SharedRAG'] = None
    ) -> 'BaseAgent':
        """Create offspring from two parents with crossover."""

        if parent2 is None:
            # Fallback to asexual if only one parent
            asexual = AsexualReproduction(mutation_rate=self.mutation_rate)
            return asexual.reproduce(parent1, None, compaction_strategy, generation, shared_rag)

        # Get patterns from both parents
        p1_patterns = parent1.reasoning_memory.get_all_reasoning()
        p2_patterns = parent2.reasoning_memory.get_all_reasoning()

        # Crossover: combine patterns
        combined_patterns = self._crossover(p1_patterns, p2_patterns)

        # Compact combined set to best patterns
        max_inherited = int(os.getenv('INHERITED_REASONING_SIZE', '100'))
        inherited_patterns = compaction_strategy.compact(
            combined_patterns,
            max_size=max_inherited
        )

        # Apply mutation if specified
        mutated = False
        if random.random() < self.mutation_rate:
            inherited_patterns = self._mutate_patterns(inherited_patterns)
            mutated = True

        # Create offspring (inherit system_prompt from parent1)
        offspring = self._create_offspring(
            role=parent1.role,
            generation=generation,
            inherited_patterns=inherited_patterns,
            shared_rag=shared_rag or parent1.shared_rag,
            system_prompt=parent1.system_prompt
        )

        self._update_stats(mutated)
        return offspring

    def _crossover(self, p1_patterns: List[Dict], p2_patterns: List[Dict]) -> List[Dict]:
        """Combine patterns from two parents.

        Simple crossover: concatenate and let compaction handle selection.
        """
        # Simple: Combine all patterns
        combined = p1_patterns + p2_patterns

        # Optional: Could implement more sophisticated crossover here
        # (e.g., interleave by rank, weighted sampling, etc.)

        return combined

    def _mutate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Apply mutation to patterns."""
        mutated = []
        for pattern in patterns:
            p = pattern.copy()
            # Add small random noise to score
            if 'score' in p:
                noise = (random.random() - 0.5) * 1.5  # ±0.75 (stronger than asexual)
                p['score'] = max(0, min(10, p['score'] + noise))
            mutated.append(p)
        return mutated

    def _create_offspring(
        self,
        role: str,
        generation: int,
        inherited_patterns: List[Dict],
        shared_rag: 'SharedRAG',
        system_prompt: Optional[str] = None
    ) -> 'BaseAgent':
        """Create offspring agent with inherited patterns."""
        # Same as asexual for now
        from lean.base_agent import IntroAgent, BodyAgent, ConclusionAgent
        from lean.reasoning_memory import ReasoningMemory

        child_id = f"{role}_gen{generation}_child{uuid.uuid4().hex[:6]}"

        memory = ReasoningMemory(
            collection_name=f"{role}_{child_id}_reasoning",
            inherited_reasoning=inherited_patterns
        )

        # Map role to agent class
        agent_classes = {
            'intro': IntroAgent,
            'body': BodyAgent,
            'conclusion': ConclusionAgent
        }

        agent_class = agent_classes.get(role)
        if not agent_class:
            raise ValueError(f"Unknown role: {role}")

        # Create agent directly (inherit system_prompt from parent)
        agent = agent_class(
            role=role,
            agent_id=f"{role}_{child_id}",
            reasoning_memory=memory,
            shared_rag=shared_rag,
            system_prompt=system_prompt
        )

        return agent


# Convenience factory function
def create_reproduction_strategy(
    strategy_name: str = "sexual",
    **kwargs
) -> ReproductionStrategy:
    """Create reproduction strategy by name.

    Args:
        strategy_name: One of: asexual, sexual
        **kwargs: Strategy-specific parameters

    Returns:
        ReproductionStrategy instance

    Example:
        strategy = create_reproduction_strategy('sexual', mutation_rate=0.1)
    """

    strategies = {
        'asexual': AsexualReproduction,
        'sexual': SexualReproduction
    }

    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: {list(strategies.keys())}"
        )

    return strategy_class(**kwargs)
