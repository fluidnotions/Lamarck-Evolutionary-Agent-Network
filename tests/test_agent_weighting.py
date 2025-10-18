"""
Test suite for agent trust weighting functionality (M2).

Tests the TrustManager integration:
- Weight initialization
- Weight updates based on performance signals
- Weighted context generation
- Weight persistence in state
"""

import pytest
from typing import Dict
from lean.state import BlogState, create_initial_state
from lean.weighting.trust_manager import TrustManager
from lean.weighting.weight_updates import (
    calculate_performance_signal,
    update_all_weights,
)


class TestPerformanceSignal:
    """Test performance signal calculation."""

    def test_both_high_scores(self):
        """High agent and peer scores should produce high signal."""
        signal = calculate_performance_signal(
            agent_score=9.0, peer_score=8.5, max_score=10.0
        )
        # (9/10 + 8.5/10) / 2 = (0.9 + 0.85) / 2 = 0.875
        assert signal == pytest.approx(0.875, rel=0.01)

    def test_both_low_scores(self):
        """Low agent and peer scores should produce low signal."""
        signal = calculate_performance_signal(
            agent_score=3.0, peer_score=4.0, max_score=10.0
        )
        # (3/10 + 4/10) / 2 = (0.3 + 0.4) / 2 = 0.35
        assert signal == pytest.approx(0.35, rel=0.01)

    def test_mixed_scores(self):
        """Mixed scores should produce medium signal."""
        signal = calculate_performance_signal(
            agent_score=8.0, peer_score=5.0, max_score=10.0
        )
        # (8/10 + 5/10) / 2 = (0.8 + 0.5) / 2 = 0.65
        assert signal == pytest.approx(0.65, rel=0.01)

    def test_signal_bounds(self):
        """Signal should be clamped to [0, 1]."""
        signal = calculate_performance_signal(
            agent_score=10.0, peer_score=10.0, max_score=10.0
        )
        assert 0.0 <= signal <= 1.0


class TestTrustManager:
    """Test TrustManager weight management."""

    def test_initialization(self):
        """TrustManager should start with default weights."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        # First access should return initial weight
        assert tm.get_weight("body", "intro") == 0.5
        assert tm.get_weight("conclusion", "intro") == 0.5
        assert tm.get_weight("conclusion", "body") == 0.5

    def test_weight_update_positive_signal(self):
        """Positive signal should increase weight."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        # High signal (0.8) should pull weight up
        new_weight = tm.update_weight("body", "intro", performance_signal=0.8)

        # w_new = w_old + α * (signal - w_old)
        # w_new = 0.5 + 0.1 * (0.8 - 0.5) = 0.5 + 0.03 = 0.53
        assert new_weight == pytest.approx(0.53, rel=0.01)

    def test_weight_update_negative_signal(self):
        """Negative signal should decrease weight."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        # Low signal (0.2) should pull weight down
        new_weight = tm.update_weight("body", "intro", performance_signal=0.2)

        # w_new = 0.5 + 0.1 * (0.2 - 0.5) = 0.5 - 0.03 = 0.47
        assert new_weight == pytest.approx(0.47, rel=0.01)

    def test_weight_bounds(self):
        """Weights should stay within [0, 1]."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.5)

        # Try to push weight above 1.0
        tm.update_weight("body", "intro", performance_signal=1.0)
        tm.update_weight("body", "intro", performance_signal=1.0)
        tm.update_weight("body", "intro", performance_signal=1.0)
        weight = tm.get_weight("body", "intro")
        assert weight <= 1.0

        # Try to push weight below 0.0
        tm.update_weight("conclusion", "intro", performance_signal=0.0)
        tm.update_weight("conclusion", "intro", performance_signal=0.0)
        tm.update_weight("conclusion", "intro", performance_signal=0.0)
        weight = tm.get_weight("conclusion", "intro")
        assert weight >= 0.0

    def test_get_all_weights(self):
        """get_all_weights should return complete weight matrix."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        # Update some weights to actually populate the dictionary
        # (get_weight is read-only and doesn't populate)
        tm.update_weight("body", "intro", performance_signal=0.6)
        tm.update_weight("conclusion", "body", performance_signal=0.7)

        all_weights = tm.get_all_weights()

        assert "body" in all_weights
        assert "intro" in all_weights["body"]
        assert "conclusion" in all_weights
        assert "body" in all_weights["conclusion"]

    def test_weighted_context_generation(self):
        """Weighted context should include trust level prefixes."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        # Set different trust levels
        tm.update_weight("conclusion", "intro", performance_signal=0.9)  # High trust
        tm.update_weight("conclusion", "body", performance_signal=0.4)  # Medium trust

        # Push weights to clear boundaries
        for _ in range(5):
            tm.update_weight("conclusion", "intro", performance_signal=0.9)

        peer_outputs = {
            "intro": "This is the introduction.",
            "body": "This is the body content.",
        }

        context = tm.get_weighted_context("conclusion", peer_outputs)

        # Should contain trust level indicators
        assert "[HIGH TRUST]" in context or "[MEDIUM TRUST]" in context
        assert "This is the introduction" in context
        assert "This is the body content" in context


class TestWeightUpdates:
    """Test batch weight update logic."""

    def test_update_all_weights(self):
        """update_all_weights should process all agent-peer pairs."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        state = create_initial_state("test topic")
        scores = {"intro": 8.0, "body": 7.5, "conclusion": 8.5}

        updates = update_all_weights(tm, state, scores)

        # Should have 6 updates (3 agents * 2 peers each)
        # intro sees: body, conclusion
        # body sees: intro, conclusion
        # conclusion sees: intro, body
        assert len(updates) == 6

        # Each update should have required fields
        for update in updates:
            assert "agent" in update
            assert "peer" in update
            assert "old_weight" in update
            assert "new_weight" in update
            assert "delta" in update
            assert "signal" in update

    def test_weight_history_accumulation(self):
        """Weight history should accumulate across generations."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        state = create_initial_state("test topic")
        scores = {"intro": 8.0, "body": 7.5, "conclusion": 8.5}

        # First generation
        updates_gen1 = update_all_weights(tm, state, scores)
        state["weight_history"].extend(updates_gen1)

        # Second generation (different scores)
        scores = {"intro": 7.0, "body": 8.0, "conclusion": 7.5}
        updates_gen2 = update_all_weights(tm, state, scores)
        state["weight_history"].extend(updates_gen2)

        # Should have 12 total updates (6 per generation)
        assert len(state["weight_history"]) == 12


class TestWeightIntegration:
    """Test integration of weights with state."""

    def test_initial_state_has_weight_fields(self):
        """Initial state should have agent_weights and weight_history."""
        state = create_initial_state("test topic")

        assert "agent_weights" in state
        assert "weight_history" in state

        # Should be empty initially
        assert state["agent_weights"] == {"intro": {}, "body": {}, "conclusion": {}}
        assert state["weight_history"] == []

    def test_weights_stored_in_state(self):
        """Weights should be stored in state after updates."""
        tm = TrustManager(initial_weight=0.5, learning_rate=0.1)

        state = create_initial_state("test topic")
        scores = {"intro": 8.0, "body": 7.5, "conclusion": 8.5}

        # Update weights
        updates = update_all_weights(tm, state, scores)
        state["agent_weights"] = tm.get_all_weights()

        # State should reflect updated weights
        assert state["agent_weights"]["body"]["intro"] != 0.5  # Changed from initial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
