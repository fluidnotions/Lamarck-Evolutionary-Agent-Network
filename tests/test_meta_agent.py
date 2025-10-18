"""
Test suite for meta-agent system (M4).

Tests MetricsMonitor, GraphMutator, and MetaAgent functionality.
"""

import pytest
from lean.meta.metrics_monitor import MetricsMonitor
from lean.meta.graph_mutator import GraphMutator, MutationType
from lean.meta.meta_agent import MetaAgent, create_meta_agent


class TestMetricsMonitor:
    """Test MetricsMonitor metric collection and analysis."""

    def test_initialization(self):
        """MetricsMonitor should initialize with empty history."""
        monitor = MetricsMonitor(history_window=10)

        assert monitor.generation_count == 0
        assert len(monitor.score_history) == 0
        assert len(monitor.timing_history) == 0

    def test_record_generation(self):
        """record_generation should store scores and timings."""
        monitor = MetricsMonitor(history_window=10)

        scores = {"intro": 8.0, "body": 7.5, "conclusion": 8.5}
        timings = {
            "intro": {"start": 0.0, "end": 1.0, "duration": 1.0},
            "body": {"start": 1.0, "end": 3.0, "duration": 2.0},
            "conclusion": {"start": 3.0, "end": 4.5, "duration": 1.5},
        }

        monitor.record_generation(scores, timings)

        assert monitor.generation_count == 1
        assert len(monitor.score_history) == 3
        assert monitor.score_history["intro"][0] == 8.0

    def test_history_window_limit(self):
        """History should be limited to window size."""
        monitor = MetricsMonitor(history_window=3)

        scores = {"intro": 8.0}
        timings = {"intro": {"start": 0.0, "end": 1.0, "duration": 1.0}}

        # Record 5 generations
        for i in range(5):
            monitor.record_generation(scores, timings)

        # Should only keep last 3
        assert len(monitor.score_history["intro"]) == 3

    def test_analyze_agent_performance_low_score(self):
        """Should identify agents with consistently low scores."""
        monitor = MetricsMonitor(history_window=10, low_score_threshold=6.0)

        # Record several generations with low scores
        for _ in range(5):
            scores = {"intro": 5.0}
            timings = {"intro": {"start": 0.0, "end": 1.0, "duration": 1.0}}
            monitor.record_generation(scores, timings)

        analysis = monitor.analyze_agent_performance("intro")

        assert analysis["status"] == "needs_attention"
        assert "low_performance" in analysis["issues"]
        assert analysis["avg_score"] == 5.0

    def test_analyze_agent_performance_high_variance(self):
        """Should identify agents with unstable performance."""
        monitor = MetricsMonitor(history_window=10, high_variance_threshold=1.5)

        # Record generations with high variance
        scores_list = [3.0, 9.0, 4.0, 8.5, 5.0]
        for score in scores_list:
            scores = {"intro": score}
            timings = {"intro": {"start": 0.0, "end": 1.0, "duration": 1.0}}
            monitor.record_generation(scores, timings)

        analysis = monitor.analyze_agent_performance("intro")

        assert "high_variance" in analysis["issues"]
        assert analysis["score_variance"] > 1.5

    def test_analyze_agent_performance_insufficient_data(self):
        """Should report insufficient data for new agents."""
        monitor = MetricsMonitor(history_window=10)

        # Record only 1 generation
        scores = {"intro": 8.0}
        timings = {"intro": {"start": 0.0, "end": 1.0, "duration": 1.0}}
        monitor.record_generation(scores, timings)

        analysis = monitor.analyze_agent_performance("intro")

        assert analysis["status"] == "insufficient_data"

    def test_identify_optimization_opportunities(self):
        """Should identify agents needing attention."""
        monitor = MetricsMonitor(history_window=10, low_score_threshold=6.0)

        # Record several generations with one low-performing agent
        for _ in range(5):
            scores = {"intro": 8.0, "body": 4.5, "conclusion": 7.5}
            timings = {
                "intro": {"start": 0.0, "end": 1.0, "duration": 1.0},
                "body": {"start": 1.0, "end": 2.0, "duration": 1.0},
                "conclusion": {"start": 2.0, "end": 3.0, "duration": 1.0},
            }
            monitor.record_generation(scores, timings)

        opportunities = monitor.identify_optimization_opportunities()

        # Should identify body as needing attention
        agent_issues = [o for o in opportunities if o["type"] == "agent_performance"]
        assert len(agent_issues) >= 1
        assert any(o["agent"] == "body" for o in agent_issues)

    def test_get_summary_statistics(self):
        """Should compute summary statistics across all agents."""
        monitor = MetricsMonitor(history_window=10)

        for _ in range(3):
            scores = {"intro": 8.0, "body": 7.0, "conclusion": 8.5}
            timings = {
                "intro": {"start": 0.0, "end": 1.0, "duration": 1.0},
                "body": {"start": 1.0, "end": 2.0, "duration": 1.0},
                "conclusion": {"start": 2.0, "end": 3.0, "duration": 1.0},
            }
            monitor.record_generation(scores, timings)

        stats = monitor.get_summary_statistics()

        assert stats["generation_count"] == 3
        assert stats["agents_tracked"] == 3
        assert stats["overall_avg_score"] > 7.0


class TestGraphMutator:
    """Test GraphMutator mutation proposals."""

    def test_initialization(self):
        """GraphMutator should initialize with empty history."""
        mutator = GraphMutator()

        assert len(mutator.mutation_history) == 0
        assert len(mutator.pending_mutations) == 0

    def test_propose_parallelization(self):
        """Should create parallelization mutation proposal."""
        mutator = GraphMutator()

        mutation = mutator.propose_parallelization(
            nodes=["body", "conclusion"], rationale="Independent agents"
        )

        assert mutation.mutation_type == MutationType.ADD_PARALLEL_LAYER
        assert "body" in mutation.affected_nodes
        assert "conclusion" in mutation.affected_nodes
        assert mutation.applied is False
        assert len(mutator.pending_mutations) == 1

    def test_propose_node_removal(self):
        """Should create node removal mutation proposal."""
        mutator = GraphMutator()

        mutation = mutator.propose_node_removal(
            node="intro", rationale="Consistently low performance"
        )

        assert mutation.mutation_type == MutationType.REMOVE_NODE
        assert "intro" in mutation.affected_nodes
        assert len(mutator.pending_mutations) == 1

    def test_mark_applied(self):
        """Should move mutation from pending to history."""
        mutator = GraphMutator()

        mutation = mutator.propose_parallelization(
            nodes=["body", "conclusion"], rationale="Test"
        )

        assert len(mutator.pending_mutations) == 1
        assert len(mutator.mutation_history) == 0

        mutator.mark_applied(mutation)

        assert len(mutator.pending_mutations) == 0
        assert len(mutator.mutation_history) == 1
        assert mutation.applied is True

    def test_get_statistics(self):
        """Should compute mutation statistics."""
        mutator = GraphMutator()

        # Propose and apply several mutations
        m1 = mutator.propose_parallelization(
            nodes=["a", "b"], rationale="Test"
        )
        m2 = mutator.propose_node_removal(node="c", rationale="Test")

        mutator.mark_applied(m1)

        stats = mutator.get_statistics()

        assert stats["total_applied"] == 1
        assert stats["pending"] == 1


class TestMetaAgent:
    """Test MetaAgent decision-making."""

    def test_initialization(self):
        """MetaAgent should initialize with components."""
        meta = MetaAgent()

        assert meta.metrics is not None
        assert meta.mutator is not None
        assert meta.mutation_threshold == 5

    def test_analyze_and_propose_insufficient_data(self):
        """Should not propose mutations with insufficient data."""
        meta = MetaAgent(mutation_threshold=5)

        # Provide state for only 3 generations
        for _ in range(3):
            state = {
                "scores": {"intro": 8.0, "body": 7.0, "conclusion": 8.5},
                "agent_timings": {
                    "intro": {"start": 0.0, "end": 1.0, "duration": 1.0},
                    "body": {"start": 1.0, "end": 2.0, "duration": 1.0},
                    "conclusion": {"start": 2.0, "end": 3.0, "duration": 1.0},
                },
            }
            proposals = meta.analyze_and_propose(state)

        # Should not propose mutations yet (< 5 generations)
        assert len(proposals) == 0

    def test_analyze_and_propose_parallelization(self):
        """Should propose parallelization after sufficient data."""
        meta = MetaAgent(mutation_threshold=3)

        # Provide state for sufficient generations
        for _ in range(5):
            state = {
                "scores": {"intro": 8.0, "body": 7.5, "conclusion": 8.5},
                "agent_timings": {
                    "intro": {"start": 0.0, "end": 1.0, "duration": 1.0},
                    "body": {"start": 1.0, "end": 2.0, "duration": 1.0},
                    "conclusion": {"start": 2.0, "end": 3.0, "duration": 1.0},
                },
            }
            proposals = meta.analyze_and_propose(state)

        # Should have proposed something (likely parallelization)
        assert len(proposals) > 0

    def test_get_recommendations(self):
        """Should provide human-readable recommendations."""
        meta = MetaAgent()

        for _ in range(3):
            state = {
                "scores": {"intro": 8.0, "body": 7.0, "conclusion": 8.5},
                "agent_timings": {
                    "intro": {"start": 0.0, "end": 1.0, "duration": 1.0},
                    "body": {"start": 1.0, "end": 2.0, "duration": 1.0},
                    "conclusion": {"start": 2.0, "end": 3.0, "duration": 1.0},
                },
            }
            meta.analyze_and_propose(state)

        recommendations = meta.get_recommendations()

        assert "summary" in recommendations
        assert "opportunities" in recommendations
        assert "pending_mutations" in recommendations

    def test_create_meta_agent_factory(self):
        """Factory should create configured MetaAgent."""
        meta = create_meta_agent(
            history_window=5, low_score_threshold=7.0, mutation_threshold=3
        )

        assert isinstance(meta, MetaAgent)
        assert meta.metrics.history_window == 5
        assert meta.metrics.low_score_threshold == 7.0
        assert meta.mutation_threshold == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
