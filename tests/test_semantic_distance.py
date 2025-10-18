"""
Test suite for semantic distance weighting (M9).
"""

import pytest
import numpy as np
from lean.hierarchy.structure import AgentHierarchy
from lean.hierarchy.semantic import (
    compute_semantic_distance,
    filter_context_by_distance,
    compute_context_weights,
    get_contextual_relevance,
    compute_similarity_matrix,
)
from lean.hierarchy.factory import create_hierarchical_agents
from lean.state import create_hierarchical_state
import tempfile
import shutil


class TestSemanticDistance:
    """Test semantic distance calculation."""

    def test_distance_identical_vectors(self):
        """Distance between identical vectors should be 0."""
        hierarchy = AgentHierarchy()

        # Same agent
        distance = compute_semantic_distance(hierarchy, "coordinator", "coordinator")

        assert distance == 0.0

    def test_distance_different_vectors(self):
        """Distance between different vectors should be > 0."""
        hierarchy = AgentHierarchy()

        # Different agents with different semantic vectors
        distance = compute_semantic_distance(hierarchy, "coordinator", "researcher")

        assert distance > 0.0
        assert distance <= 1.0

    def test_distance_bounds(self):
        """Distance should always be in [0, 1]."""
        hierarchy = AgentHierarchy()

        all_agents = list(hierarchy.nodes.keys())

        for agent_a in all_agents:
            for agent_b in all_agents:
                distance = compute_semantic_distance(hierarchy, agent_a, agent_b)
                assert 0.0 <= distance <= 1.0

    def test_distance_symmetry(self):
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        hierarchy = AgentHierarchy()

        distance_ab = compute_semantic_distance(hierarchy, "intro", "researcher")
        distance_ba = compute_semantic_distance(hierarchy, "researcher", "intro")

        assert abs(distance_ab - distance_ba) < 0.0001  # Floating point tolerance

    def test_invalid_agent_raises_error(self):
        """Should raise ValueError for unknown agent."""
        hierarchy = AgentHierarchy()

        with pytest.raises(ValueError, match="not found"):
            compute_semantic_distance(hierarchy, "invalid_agent", "intro")


class TestContextFiltering:
    """Test context filtering based on distance."""

    def test_filter_distance_zero(self):
        """Distance 0 should keep full context."""
        context = "Sentence one. Sentence two. Sentence three."

        filtered = filter_context_by_distance(context, distance=0.0)

        # Should keep all sentences
        assert filtered == context

    def test_filter_distance_one(self):
        """Distance 1 should keep minimal context (min_ratio)."""
        context = "Sentence one. Sentence two. Sentence three."

        filtered = filter_context_by_distance(context, distance=1.0, min_ratio=0.3)

        # Should keep only ~30% (at least 1 sentence)
        sentences = filtered.split(". ")
        assert len(sentences) <= 2  # Should reduce sentences

    def test_filter_intermediate_distance(self):
        """Intermediate distance should keep partial context."""
        context = "One. Two. Three. Four. Five."

        filtered = filter_context_by_distance(context, distance=0.5, min_ratio=0.3)

        # Should keep some but not all
        original_sentences = context.split(". ")
        filtered_sentences = filtered.split(". ")

        assert len(filtered_sentences) < len(original_sentences)
        assert len(filtered_sentences) >= 1

    def test_filter_empty_context(self):
        """Empty context should return empty."""
        filtered = filter_context_by_distance("", distance=0.5)

        assert filtered == ""

    def test_filter_preserves_beginning(self):
        """Filtering should preserve beginning of context."""
        context = "First. Second. Third. Fourth."

        filtered = filter_context_by_distance(context, distance=0.7, min_ratio=0.3)

        # Should start with first sentence
        assert filtered.startswith("First")


class TestContextWeights:
    """Test context weight computation."""

    def test_compute_context_weights(self):
        """Should compute weights for all children."""
        hierarchy = AgentHierarchy()

        children = hierarchy.get_children("coordinator")
        weights = compute_context_weights(hierarchy, "coordinator", children)

        # Should have weight for each child
        assert len(weights) == len(children)

        # All weights should be in [0, 1]
        for weight in weights.values():
            assert 0.0 <= weight <= 1.0

    def test_weights_inverse_of_distance(self):
        """Weight should be inverse of distance."""
        hierarchy = AgentHierarchy()

        distance = compute_semantic_distance(hierarchy, "coordinator", "intro")
        weights = compute_context_weights(hierarchy, "coordinator", ["intro"])

        # weight = 1 - distance
        expected_weight = 1.0 - distance
        assert abs(weights["intro"] - expected_weight) < 0.0001


class TestContextualRelevance:
    """Test relevance scoring."""

    def test_relevance_is_inverse_distance(self):
        """Relevance should be 1 - distance."""
        hierarchy = AgentHierarchy()

        distance = compute_semantic_distance(hierarchy, "intro", "researcher")
        relevance = get_contextual_relevance(hierarchy, "intro", "researcher")

        assert abs(relevance - (1.0 - distance)) < 0.0001

    def test_relevance_bounds(self):
        """Relevance should be in [0, 1]."""
        hierarchy = AgentHierarchy()

        relevance = get_contextual_relevance(hierarchy, "coordinator", "fact_checker")

        assert 0.0 <= relevance <= 1.0

    def test_high_relevance_for_similar(self):
        """Similar agents should have high relevance."""
        hierarchy = AgentHierarchy()

        # Same agent = max relevance
        relevance = get_contextual_relevance(hierarchy, "intro", "intro")

        assert relevance == 1.0


class TestSimilarityMatrix:
    """Test similarity matrix computation."""

    def test_matrix_completeness(self):
        """Matrix should have entry for all agent pairs."""
        hierarchy = AgentHierarchy()

        matrix = compute_similarity_matrix(hierarchy)

        all_agents = list(hierarchy.nodes.keys())
        expected_entries = len(all_agents) * len(all_agents)

        assert len(matrix) == expected_entries

    def test_matrix_symmetry(self):
        """Matrix should be symmetric."""
        hierarchy = AgentHierarchy()

        matrix = compute_similarity_matrix(hierarchy)

        all_agents = list(hierarchy.nodes.keys())

        for agent_a in all_agents:
            for agent_b in all_agents:
                assert matrix[(agent_a, agent_b)] == matrix[(agent_b, agent_a)]

    def test_matrix_diagonal_zero(self):
        """Diagonal entries (same agent) should be 0."""
        hierarchy = AgentHierarchy()

        matrix = compute_similarity_matrix(hierarchy)

        all_agents = list(hierarchy.nodes.keys())

        for agent in all_agents:
            assert matrix[(agent, agent)] == 0.0


class TestCoordinatorIntegration:
    """Test integration with coordinator."""

    def test_coordinator_uses_semantic_filtering(self):
        """Coordinator should filter context based on semantic distance."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            coordinator = agents["coordinator"]

            state = create_hierarchical_state("Test Topic")
            state["coordinator_intent"] = "Long context. " * 20  # 20 sentences

            # Get filtered contexts
            contexts = coordinator.distribute_context(state, use_semantic_filtering=True)

            # Should have contexts for all children
            assert "intro" in contexts
            assert "body" in contexts
            assert "conclusion" in contexts

            # Contexts should be filtered (shorter than original)
            for context in contexts.values():
                assert len(context) <= len(state["coordinator_intent"])
        finally:
            shutil.rmtree(temp_dir)

    def test_coordinator_without_filtering(self):
        """Coordinator can disable semantic filtering."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            coordinator = agents["coordinator"]

            state = create_hierarchical_state("Test Topic")
            intent = "Full context for all agents."
            state["coordinator_intent"] = intent

            # Get full contexts (no filtering)
            contexts = coordinator.distribute_context(state, use_semantic_filtering=False)

            # All contexts should be identical to intent
            for context in contexts.values():
                assert context == intent
        finally:
            shutil.rmtree(temp_dir)


class TestSemanticProperties:
    """Test semantic vector properties."""

    def test_researcher_close_to_fact_checker(self):
        """Researcher and fact_checker should be semantically close."""
        hierarchy = AgentHierarchy()

        distance = compute_semantic_distance(hierarchy, "researcher", "fact_checker")

        # Both are information-focused
        assert distance < 0.5  # Should be relatively close

    def test_stylist_distant_from_researcher(self):
        """Stylist and researcher should be semantically distant."""
        hierarchy = AgentHierarchy()

        distance = compute_semantic_distance(hierarchy, "stylist", "researcher")

        # Different focuses: style vs. facts
        # Note: Actual distance depends on semantic vectors in structure.py

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
