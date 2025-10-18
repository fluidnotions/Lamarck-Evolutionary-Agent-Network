"""
Test suite for memory decay functionality (M3).

Tests the time-based memory relevance decay:
- DecayCalculator exponential decay formula
- Effective score calculation
- MemoryPruner deletion criteria
- Integration with MemoryManager
"""

import pytest
from datetime import datetime, timedelta
from lean.memory.decay import DecayCalculator, MemoryPruner


class TestDecayCalculator:
    """Test DecayCalculator exponential decay."""

    def test_no_decay_for_recent_memory(self):
        """Recent memories should have decay factor close to 1.0."""
        calc = DecayCalculator(decay_lambda=0.01)

        # Memory from 1 hour ago
        timestamp = (datetime.now() - timedelta(hours=1)).isoformat()
        decay_factor = calc.calculate_decay_factor(timestamp)

        # Should be very close to 1.0 (minimal decay)
        assert decay_factor > 0.99

    def test_decay_for_old_memory(self):
        """Old memories should have lower decay factor."""
        calc = DecayCalculator(decay_lambda=0.01)

        # Memory from 100 days ago
        timestamp = (datetime.now() - timedelta(days=100)).isoformat()
        decay_factor = calc.calculate_decay_factor(timestamp)

        # e^(-0.01 * 100) ≈ 0.368
        assert 0.35 < decay_factor < 0.40

    def test_higher_lambda_faster_decay(self):
        """Higher lambda should produce faster decay."""
        calc_slow = DecayCalculator(decay_lambda=0.01)
        calc_fast = DecayCalculator(decay_lambda=0.1)

        # Memory from 10 days ago
        timestamp = (datetime.now() - timedelta(days=10)).isoformat()

        decay_slow = calc_slow.calculate_decay_factor(timestamp)
        decay_fast = calc_fast.calculate_decay_factor(timestamp)

        # Fast decay should be significantly lower
        assert decay_fast < decay_slow
        assert decay_slow > 0.9  # Slow decay: e^(-0.01 * 10) ≈ 0.905
        assert decay_fast < 0.4  # Fast decay: e^(-0.1 * 10) ≈ 0.368

    def test_effective_score_calculation(self):
        """Effective score should combine similarity, decay, and original score."""
        calc = DecayCalculator(decay_lambda=0.01)

        # Recent high-quality memory
        timestamp_recent = datetime.now().isoformat()
        effective_recent = calc.calculate_effective_score(
            similarity=0.9, original_score=8.5, timestamp=timestamp_recent, max_score=10.0
        )

        # Old high-quality memory
        timestamp_old = (datetime.now() - timedelta(days=100)).isoformat()
        effective_old = calc.calculate_effective_score(
            similarity=0.9, original_score=8.5, timestamp=timestamp_old, max_score=10.0
        )

        # Recent should have higher effective score
        assert effective_recent > effective_old
        # Recent should be close to: 0.9 * 1.0 * 0.85 ≈ 0.765
        assert effective_recent > 0.75

    def test_decay_factor_bounds(self):
        """Decay factor should always be between 0 and 1."""
        calc = DecayCalculator(decay_lambda=0.01)

        # Test various timestamps
        timestamps = [
            datetime.now().isoformat(),  # Now
            (datetime.now() - timedelta(days=1)).isoformat(),  # 1 day ago
            (datetime.now() - timedelta(days=365)).isoformat(),  # 1 year ago
        ]

        for ts in timestamps:
            decay = calc.calculate_decay_factor(ts)
            assert 0.0 <= decay <= 1.0


class TestMemoryPruner:
    """Test MemoryPruner deletion logic."""

    def test_delete_old_memories(self):
        """Memories older than max_age_days should be deleted."""
        pruner = MemoryPruner(max_age_days=30, prune_to_top_n=100, min_effective_score=3.0)

        # Memory from 40 days ago
        old_timestamp = (datetime.now() - timedelta(days=40)).isoformat()

        should_delete = pruner.should_delete(
            timestamp=old_timestamp, effective_score=5.0
        )

        assert should_delete is True

    def test_keep_recent_memories(self):
        """Recent memories should not be deleted."""
        pruner = MemoryPruner(max_age_days=30, prune_to_top_n=100, min_effective_score=3.0)

        # Memory from 10 days ago
        recent_timestamp = (datetime.now() - timedelta(days=10)).isoformat()

        should_delete = pruner.should_delete(
            timestamp=recent_timestamp, effective_score=5.0
        )

        assert should_delete is False

    def test_delete_low_score_memories(self):
        """Memories below min_effective_score should be deleted."""
        pruner = MemoryPruner(max_age_days=30, prune_to_top_n=100, min_effective_score=3.0)

        # Recent but low-scoring memory
        recent_timestamp = (datetime.now() - timedelta(days=1)).isoformat()

        should_delete = pruner.should_delete(
            timestamp=recent_timestamp, effective_score=2.5
        )

        assert should_delete is True

    def test_prune_memories_keeps_top_n(self):
        """prune_memories should keep only top N memories."""
        calc = DecayCalculator(decay_lambda=0.01)
        pruner = MemoryPruner(max_age_days=365, prune_to_top_n=3, min_effective_score=0.0)

        # Create 5 memories with varying scores
        memories = [
            {
                "content": f"memory_{i}",
                "score": float(i + 5),
                "timestamp": datetime.now().isoformat(),
                "similarity": 0.9,
            }
            for i in range(5)
        ]

        pruned = pruner.prune_memories(memories, calc)

        # Should keep only top 3
        assert len(pruned) == 3

        # Should keep highest-scoring memories
        scores = [m["score"] for m in pruned]
        assert scores == [9.0, 8.0, 7.0]

    def test_prune_memories_adds_effective_score(self):
        """prune_memories should add effective_score to each memory."""
        calc = DecayCalculator(decay_lambda=0.01)
        pruner = MemoryPruner(max_age_days=365, prune_to_top_n=10, min_effective_score=0.0)

        memories = [
            {
                "content": "test memory",
                "score": 8.0,
                "timestamp": datetime.now().isoformat(),
                "similarity": 0.9,
            }
        ]

        pruned = pruner.prune_memories(memories, calc)

        assert len(pruned) == 1
        assert "effective_score" in pruned[0]
        assert pruned[0]["effective_score"] > 0


class TestDecayIntegration:
    """Test integration of decay with memory retrieval."""

    def test_recent_high_score_beats_old_perfect_score(self):
        """Recent high-quality memory should rank above old perfect-score memory."""
        calc = DecayCalculator(decay_lambda=0.1)  # Fast decay

        # Recent memory: score 8.5
        recent_effective = calc.calculate_effective_score(
            similarity=0.9,
            original_score=8.5,
            timestamp=datetime.now().isoformat(),
            max_score=10.0,
        )

        # Old memory: score 10.0 (perfect)
        old_effective = calc.calculate_effective_score(
            similarity=0.9,
            original_score=10.0,
            timestamp=(datetime.now() - timedelta(days=30)).isoformat(),
            max_score=10.0,
        )

        # With fast decay (0.1), recent 8.5 should beat old 10.0
        # Recent: 0.9 * 1.0 * 0.85 = 0.765
        # Old: 0.9 * e^(-0.1*30) * 1.0 = 0.9 * 0.0498 ≈ 0.045
        assert recent_effective > old_effective

    def test_similarity_still_matters(self):
        """Higher similarity should still produce higher effective score."""
        calc = DecayCalculator(decay_lambda=0.01)

        timestamp = datetime.now().isoformat()

        high_sim = calc.calculate_effective_score(
            similarity=0.9, original_score=8.0, timestamp=timestamp
        )

        low_sim = calc.calculate_effective_score(
            similarity=0.5, original_score=8.0, timestamp=timestamp
        )

        assert high_sim > low_sim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
