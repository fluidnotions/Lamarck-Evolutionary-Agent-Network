"""
Compaction Strategies for Reasoning Pattern Evolution

Implements Step 8 (EVOLVE) - Forgetting component:
- Reduce large reasoning pattern collections to best patterns
- Multiple strategies: score, frequency, diversity, hybrid
- Used before inheritance to pass only successful patterns to offspring
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances


class CompactionStrategy(ABC):
    """Base class for reasoning pattern compaction.

    Compaction is the "forgetting" mechanism in Lamarckian evolution:
    - Agents accumulate 100+ reasoning patterns over their lifetime
    - Before reproduction, compact to top 20-30 patterns
    - Offspring inherit ONLY the compacted (successful) patterns

    This is a pure Python utility class (NOT a LangGraph node).
    Called by ReproductionStrategy during offspring creation.
    """

    def __init__(self):
        """Initialize compaction strategy."""
        self.stats = {
            'total_compactions': 0,
            'total_patterns_before': 0,
            'total_patterns_after': 0,
            'avg_compaction_rate': 0.0
        }

    @abstractmethod
    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Compact patterns to max_size.

        Args:
            patterns: List of reasoning pattern dicts with keys:
                - reasoning: str (the cognitive pattern text)
                - score: float (quality score 0-10)
                - retrieval_count: int (how often retrieved)
                - embedding: Optional[List[float]] (semantic vector)
                - situation: str (context where pattern was used)
                - timestamp: float (when stored)
            max_size: Maximum number of patterns to keep
            metadata: Optional context (domain, generation, etc.)

        Returns:
            Compacted list of patterns (length <= max_size)

        Note:
            Patterns dict structure matches ReasoningMemory.get_all_reasoning()
            Each pattern has metadata from ChromaDB storage
        """
        pass

    def get_stats(self) -> Dict:
        """Get compaction statistics.

        Returns:
            Dict with compaction metrics
        """
        return {
            'strategy': self.__class__.__name__,
            'total_compactions': self.stats['total_compactions'],
            'total_patterns_before': self.stats['total_patterns_before'],
            'total_patterns_after': self.stats['total_patterns_after'],
            'avg_compaction_rate': self.stats['avg_compaction_rate'],
            'avg_patterns_kept': (
                self.stats['total_patterns_after'] / self.stats['total_compactions']
                if self.stats['total_compactions'] > 0 else 0
            )
        }

    def _update_stats(self, before_count: int, after_count: int):
        """Update internal statistics."""
        self.stats['total_compactions'] += 1
        self.stats['total_patterns_before'] += before_count
        self.stats['total_patterns_after'] += after_count

        if self.stats['total_patterns_before'] > 0:
            self.stats['avg_compaction_rate'] = (
                self.stats['total_patterns_after'] /
                self.stats['total_patterns_before']
            )


class ScoreBasedCompaction(CompactionStrategy):
    """Keep highest-scoring patterns.

    Strategy: Sort by score, keep top N.

    Best for:
    - Quality-focused evolution
    - When scores accurately reflect pattern effectiveness

    Drawback:
    - May lose diversity (all high-scoring patterns might be similar)
    """

    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Keep top N patterns by score."""

        if len(patterns) <= max_size:
            return patterns

        # Sort by score descending
        sorted_patterns = sorted(
            patterns,
            key=lambda p: p.get('score', 0.0),
            reverse=True
        )

        compacted = sorted_patterns[:max_size]
        self._update_stats(len(patterns), len(compacted))

        return compacted


class FrequencyBasedCompaction(CompactionStrategy):
    """Keep most-retrieved patterns.

    Strategy: Patterns that were retrieved often were likely useful.
    Weight by retrieval_count × score to balance frequency and quality.

    Best for:
    - Keeping battle-tested patterns
    - Patterns that proved useful across multiple tasks

    Drawback:
    - May favor generic patterns over specialized ones
    """

    def __init__(self, score_weight: float = 0.5):
        """Initialize frequency-based compaction.

        Args:
            score_weight: How much to weight score vs frequency (0.0-1.0)
                         0.0 = pure frequency, 1.0 = pure score
        """
        super().__init__()
        self.score_weight = score_weight

    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Keep top N patterns by usage frequency."""

        if len(patterns) <= max_size:
            return patterns

        # Calculate combined score: frequency × quality
        def combined_score(pattern: Dict) -> float:
            retrieval_count = pattern.get('retrieval_count', 0)
            score = pattern.get('score', 0.0)

            # Weighted combination
            freq_component = (1 - self.score_weight) * retrieval_count
            score_component = self.score_weight * score

            return freq_component + score_component

        # Sort by combined score
        sorted_patterns = sorted(
            patterns,
            key=combined_score,
            reverse=True
        )

        compacted = sorted_patterns[:max_size]
        self._update_stats(len(patterns), len(compacted))

        return compacted


class DiversityPreservingCompaction(CompactionStrategy):
    """Keep diverse strategies through clustering.

    Strategy:
    1. Cluster patterns by embedding similarity
    2. From each cluster, keep the best pattern (highest score)
    3. Continue until max_size reached

    Best for:
    - Maintaining strategic diversity
    - Avoiding monoculture of similar patterns

    Drawback:
    - Requires embeddings
    - More computationally expensive
    """

    def __init__(self, min_clusters: int = 5):
        """Initialize diversity-preserving compaction.

        Args:
            min_clusters: Minimum number of pattern clusters to maintain
        """
        super().__init__()
        self.min_clusters = min_clusters

    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Keep diverse patterns using clustering."""

        if len(patterns) <= max_size:
            return patterns

        # Check if embeddings available
        embeddings = [p.get('embedding') for p in patterns]
        if not all(embeddings):
            # Fallback to score-based if no embeddings
            return ScoreBasedCompaction().compact(patterns, max_size, metadata)

        # Convert to numpy array
        X = np.array(embeddings)

        # Determine number of clusters
        n_clusters = min(max_size, max(self.min_clusters, len(patterns) // 10))

        # Cluster patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # From each cluster, select best pattern
        compacted = []
        for cluster_id in range(n_clusters):
            # Get patterns in this cluster
            cluster_patterns = [
                p for i, p in enumerate(patterns)
                if cluster_labels[i] == cluster_id
            ]

            if cluster_patterns:
                # Keep highest-scoring pattern from cluster
                best = max(cluster_patterns, key=lambda p: p.get('score', 0.0))
                compacted.append(best)

        # If we have room, add more high-scoring patterns
        if len(compacted) < max_size:
            remaining = [p for p in patterns if p not in compacted]
            remaining_sorted = sorted(
                remaining,
                key=lambda p: p.get('score', 0.0),
                reverse=True
            )
            compacted.extend(remaining_sorted[:max_size - len(compacted)])

        self._update_stats(len(patterns), len(compacted))

        return compacted


class HybridCompaction(CompactionStrategy):
    """Combine score, frequency, and diversity.

    Strategy:
    1. Use diversity clustering to identify pattern families
    2. Within each family, weight by score × frequency
    3. Balance family representation to maintain diversity

    Best for:
    - Most scenarios (balanced approach)
    - When you want quality, utility, AND diversity

    This is the recommended default strategy.
    """

    def __init__(
        self,
        score_weight: float = 0.4,
        frequency_weight: float = 0.3,
        diversity_weight: float = 0.3,
        min_clusters: int = 5
    ):
        """Initialize hybrid compaction.

        Args:
            score_weight: Weight for pattern score (0.0-1.0)
            frequency_weight: Weight for retrieval frequency (0.0-1.0)
            diversity_weight: Weight for diversity preservation (0.0-1.0)
            min_clusters: Minimum clusters for diversity

        Note: Weights should sum to 1.0
        """
        super().__init__()

        # Normalize weights
        total = score_weight + frequency_weight + diversity_weight
        self.score_weight = score_weight / total
        self.frequency_weight = frequency_weight / total
        self.diversity_weight = diversity_weight / total
        self.min_clusters = min_clusters

    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Keep patterns using hybrid approach."""

        if len(patterns) <= max_size:
            return patterns

        # Check if embeddings available for diversity component
        embeddings = [p.get('embedding') for p in patterns]
        has_embeddings = all(embeddings)

        if has_embeddings:
            # Full hybrid with diversity
            compacted = self._hybrid_with_diversity(patterns, max_size)
        else:
            # Fallback: score + frequency only
            compacted = self._hybrid_without_diversity(patterns, max_size)

        self._update_stats(len(patterns), len(compacted))

        return compacted

    def _hybrid_without_diversity(
        self,
        patterns: List[Dict],
        max_size: int
    ) -> List[Dict]:
        """Hybrid compaction without diversity component."""

        # Normalize weights (no diversity)
        score_w = self.score_weight / (self.score_weight + self.frequency_weight)
        freq_w = self.frequency_weight / (self.score_weight + self.frequency_weight)

        def combined_score(pattern: Dict) -> float:
            score = pattern.get('score', 0.0)
            freq = pattern.get('retrieval_count', 0)

            # Normalize score (0-10) and frequency (assume max ~20)
            score_norm = score / 10.0
            freq_norm = min(freq / 20.0, 1.0)

            return score_w * score_norm + freq_w * freq_norm

        sorted_patterns = sorted(
            patterns,
            key=combined_score,
            reverse=True
        )

        return sorted_patterns[:max_size]

    def _hybrid_with_diversity(
        self,
        patterns: List[Dict],
        max_size: int
    ) -> List[Dict]:
        """Full hybrid compaction with diversity preservation."""

        # 1. Cluster for diversity
        X = np.array([p['embedding'] for p in patterns])
        n_clusters = min(max_size, max(self.min_clusters, len(patterns) // 10))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)

        # 2. Calculate per-cluster quota based on cluster quality
        cluster_scores = {}
        for cluster_id in range(n_clusters):
            cluster_patterns = [
                p for i, p in enumerate(patterns)
                if cluster_labels[i] == cluster_id
            ]

            if cluster_patterns:
                # Cluster quality = avg score of patterns
                avg_score = np.mean([p.get('score', 0.0) for p in cluster_patterns])
                cluster_scores[cluster_id] = avg_score

        # 3. Allocate slots proportionally to cluster quality
        total_quality = sum(cluster_scores.values())
        cluster_quotas = {}
        for cluster_id, quality in cluster_scores.items():
            quota = max(1, int(max_size * (quality / total_quality)))
            cluster_quotas[cluster_id] = quota

        # 4. Select top patterns from each cluster
        compacted = []
        for cluster_id in range(n_clusters):
            cluster_patterns = [
                p for i, p in enumerate(patterns)
                if cluster_labels[i] == cluster_id
            ]

            if not cluster_patterns:
                continue

            # Sort by combined score + frequency
            def pattern_quality(p: Dict) -> float:
                score = p.get('score', 0.0) / 10.0
                freq = min(p.get('retrieval_count', 0) / 20.0, 1.0)

                # Adjust for diversity weight (less aggressive selection)
                score_component = self.score_weight * score
                freq_component = self.frequency_weight * freq

                return score_component + freq_component

            sorted_cluster = sorted(
                cluster_patterns,
                key=pattern_quality,
                reverse=True
            )

            # Take quota from this cluster
            quota = cluster_quotas.get(cluster_id, 1)
            compacted.extend(sorted_cluster[:quota])

        # 5. Fill remaining slots with highest-scoring unchoosen patterns
        if len(compacted) < max_size:
            remaining = [p for p in patterns if p not in compacted]
            remaining_sorted = sorted(
                remaining,
                key=lambda p: p.get('score', 0.0),
                reverse=True
            )
            compacted.extend(remaining_sorted[:max_size - len(compacted)])

        # 6. Ensure we don't exceed max_size
        return compacted[:max_size]


# Convenience factory function
def create_compaction_strategy(
    strategy_name: str = "hybrid",
    **kwargs
) -> CompactionStrategy:
    """Create compaction strategy by name.

    Args:
        strategy_name: One of: score, frequency, diversity, hybrid
        **kwargs: Strategy-specific parameters

    Returns:
        CompactionStrategy instance

    Example:
        strategy = create_compaction_strategy('hybrid', score_weight=0.5)
    """

    strategies = {
        'score': ScoreBasedCompaction,
        'frequency': FrequencyBasedCompaction,
        'diversity': DiversityPreservingCompaction,
        'hybrid': HybridCompaction
    }

    strategy_class = strategies.get(strategy_name.lower())
    if not strategy_class:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Choose from: {list(strategies.keys())}"
        )

    return strategy_class(**kwargs)
