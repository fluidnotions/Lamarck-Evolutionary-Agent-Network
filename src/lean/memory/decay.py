"""
Time-based memory decay calculations.
"""

from datetime import datetime
from typing import Dict, List
import numpy as np


class DecayCalculator:
    """Calculates time-based decay for memory relevance."""

    def __init__(self, decay_lambda: float = 0.01):
        """Initialize decay calculator.

        Args:
            decay_lambda: Decay rate (higher = faster decay)
                         0.01 = ~63% relevance after 100 days
                         0.1  = ~63% relevance after 10 days
        """
        self.decay_lambda = decay_lambda

    def calculate_decay_factor(
        self, timestamp: str, current_time: datetime | None = None
    ) -> float:
        """Calculate decay factor for a timestamp.

        Formula: e^(-λ * Δt)
        where Δt is days elapsed since timestamp

        Args:
            timestamp: ISO 8601 timestamp string
            current_time: Reference time (defaults to now)

        Returns:
            Decay factor (0-1, where 1 = no decay)
        """
        if current_time is None:
            current_time = datetime.now()

        # Parse timestamp
        memory_time = datetime.fromisoformat(timestamp)

        # Calculate days elapsed
        delta_seconds = (current_time - memory_time).total_seconds()
        delta_days = delta_seconds / 86400.0

        # Exponential decay
        decay_factor = float(np.exp(-self.decay_lambda * delta_days))

        return max(0.0, min(1.0, decay_factor))

    def calculate_effective_score(
        self,
        similarity: float,
        original_score: float,
        timestamp: str,
        max_score: float = 10.0,
    ) -> float:
        """Calculate effective relevance with decay.

        Formula: relevance = similarity * e^(-λ * Δt) * (score / max_score)

        Args:
            similarity: Semantic similarity (0-1)
            original_score: Original evaluation score
            timestamp: When memory was created
            max_score: Maximum possible score

        Returns:
            Effective relevance score
        """
        decay_factor = self.calculate_decay_factor(timestamp)
        score_weight = original_score / max_score

        effective_score = float(similarity * decay_factor * score_weight)

        return effective_score


class MemoryPruner:
    """Manages memory pruning to prevent unbounded growth."""

    def __init__(
        self,
        max_age_days: int = 30,
        prune_to_top_n: int = 100,
        min_effective_score: float = 3.0,
    ):
        """Initialize memory pruner.

        Args:
            max_age_days: Delete memories older than this
            prune_to_top_n: Keep only top N memories
            min_effective_score: Delete memories below this threshold
        """
        self.max_age_days = max_age_days
        self.prune_to_top_n = prune_to_top_n
        self.min_effective_score = min_effective_score

    def should_delete(
        self,
        timestamp: str,
        effective_score: float,
        current_time: datetime | None = None,
    ) -> bool:
        """Determine if memory should be deleted.

        Args:
            timestamp: Memory creation time
            effective_score: Score after decay
            current_time: Reference time (defaults to now)

        Returns:
            True if memory should be deleted
        """
        if current_time is None:
            current_time = datetime.now()

        # Check age threshold
        memory_time = datetime.fromisoformat(timestamp)
        age_days = (current_time - memory_time).total_seconds() / 86400.0

        if age_days > self.max_age_days:
            return True

        # Check score threshold
        if effective_score < self.min_effective_score:
            return True

        return False

    def prune_memories(
        self, memories: List[Dict], decay_calculator: "DecayCalculator"
    ) -> List[Dict]:
        """Prune memories based on age and relevance.

        Args:
            memories: List of memory dicts with metadata
            decay_calculator: DecayCalculator instance

        Returns:
            Filtered list of memories
        """
        current_time = datetime.now()
        pruned = []

        for memory in memories:
            # Calculate effective score
            effective_score = decay_calculator.calculate_effective_score(
                similarity=memory.get("similarity", 1.0),
                original_score=memory.get("score", 5.0),
                timestamp=memory["timestamp"],
            )

            # Check deletion criteria
            if self.should_delete(
                memory["timestamp"], effective_score, current_time
            ):
                continue

            # Add effective score to metadata
            memory["effective_score"] = effective_score
            pruned.append(memory)

        # Sort by effective score and keep top N
        pruned.sort(key=lambda m: m["effective_score"], reverse=True)
        return pruned[: self.prune_to_top_n]
