"""
Test suite for hierarchical structure (M6).
"""

import pytest
from hvas_mini.hierarchy.structure import AgentHierarchy, AgentNode
from hvas_mini.hierarchy.coordinator import CoordinatorAgent
from hvas_mini.hierarchy.specialists import ResearchAgent, FactCheckerAgent, StyleAgent
from hvas_mini.hierarchy.factory import create_hierarchical_agents
from hvas_mini.state import create_hierarchical_state, AgentOutput
from hvas_mini.memory import MemoryManager
import tempfile
import shutil


class TestAgentHierarchy:
    """Test hierarchy structure and relationships."""

    def test_three_layers_defined(self):
        """Hierarchy should have 3 distinct layers."""
        hierarchy = AgentHierarchy()

        assert hierarchy.get_layer("coordinator") == 1
        assert hierarchy.get_layer("intro") == 2
        assert hierarchy.get_layer("body") == 2
        assert hierarchy.get_layer("conclusion") == 2
        assert hierarchy.get_layer("researcher") == 3
        assert hierarchy.get_layer("fact_checker") == 3
        assert hierarchy.get_layer("stylist") == 3

    def test_parent_child_relationships(self):
        """Parent-child relationships should be correct."""
        hierarchy = AgentHierarchy()

        # Coordinator children
        assert set(hierarchy.get_children("coordinator")) == {"intro", "body", "conclusion"}

        # Content agent children
        assert set(hierarchy.get_children("intro")) == {"researcher", "stylist"}
        assert set(hierarchy.get_children("body")) == {"researcher", "fact_checker"}
        assert set(hierarchy.get_children("conclusion")) == {"stylist"}

        # Specialists have no children
        assert hierarchy.get_children("researcher") == []

    def test_get_parent(self):
        """Should correctly identify parent agents."""
        hierarchy = AgentHierarchy()

        assert hierarchy.get_parent("intro") == "coordinator"
        assert hierarchy.get_parent("body") == "coordinator"
        assert hierarchy.get_parent("conclusion") == "coordinator"

        # Researcher has multiple parents (intro and body)
        # get_parent returns the first one found
        researcher_parent = hierarchy.get_parent("researcher")
        assert researcher_parent in ["intro", "body"]

        # Stylist appears as child of both intro and conclusion
        stylist_parent = hierarchy.get_parent("stylist")
        assert stylist_parent in ["intro", "conclusion"]

        assert hierarchy.get_parent("coordinator") is None

    def test_get_layer_agents(self):
        """Should retrieve all agents in a layer."""
        hierarchy = AgentHierarchy()

        layer1 = hierarchy.get_layer_agents(1)
        assert layer1 == ["coordinator"]

        layer2 = hierarchy.get_layer_agents(2)
        assert set(layer2) == {"intro", "body", "conclusion"}

        layer3 = hierarchy.get_layer_agents(3)
        assert set(layer3) == {"researcher", "fact_checker", "stylist"}

    def test_get_siblings(self):
        """Should get sibling agents."""
        hierarchy = AgentHierarchy()

        intro_siblings = hierarchy.get_siblings("intro")
        assert set(intro_siblings) == {"body", "conclusion"}

    def test_is_ancestor(self):
        """Should detect ancestor relationships."""
        hierarchy = AgentHierarchy()

        assert hierarchy.is_ancestor("coordinator", "intro") is True
        assert hierarchy.is_ancestor("coordinator", "researcher") is True
        assert hierarchy.is_ancestor("intro", "researcher") is True
        assert hierarchy.is_ancestor("body", "intro") is False


class TestCoordinatorAgent:
    """Test coordinator agent functionality."""

    def test_coordinator_creation(self):
        """CoordinatorAgent should initialize."""
        hierarchy = AgentHierarchy()
        temp_dir = tempfile.mkdtemp()

        try:
            memory = MemoryManager("coordinator_test", persist_directory=temp_dir)
            coordinator = CoordinatorAgent(hierarchy, memory)

            assert coordinator.role == "coordinator"
            assert coordinator.hierarchy is not None
        finally:
            shutil.rmtree(temp_dir)

    def test_distribute_context(self):
        """Should create context for all content agents."""
        hierarchy = AgentHierarchy()
        temp_dir = tempfile.mkdtemp()

        try:
            memory = MemoryManager("coordinator_test", persist_directory=temp_dir)
            coordinator = CoordinatorAgent(hierarchy, memory)

            state = create_hierarchical_state("Machine Learning")
            state["coordinator_intent"] = "Explain ML basics clearly"

            contexts = coordinator.distribute_context(state)

            assert "intro" in contexts
            assert "body" in contexts
            assert "conclusion" in contexts
            assert contexts["intro"] == "Explain ML basics clearly"
        finally:
            shutil.rmtree(temp_dir)


class TestSpecialistAgents:
    """Test specialist agent creation."""

    def test_researcher_creation(self):
        """ResearchAgent should initialize."""
        temp_dir = tempfile.mkdtemp()

        try:
            memory = MemoryManager("researcher_test", persist_directory=temp_dir)
            agent = ResearchAgent(memory)

            assert agent.role == "researcher"
            assert agent.content_key == "research"
        finally:
            shutil.rmtree(temp_dir)

    def test_fact_checker_creation(self):
        """FactCheckerAgent should initialize."""
        temp_dir = tempfile.mkdtemp()

        try:
            memory = MemoryManager("fact_checker_test", persist_directory=temp_dir)
            agent = FactCheckerAgent(memory)

            assert agent.role == "fact_checker"
            assert agent.content_key == "fact_check"
        finally:
            shutil.rmtree(temp_dir)

    def test_stylist_creation(self):
        """StyleAgent should initialize."""
        temp_dir = tempfile.mkdtemp()

        try:
            memory = MemoryManager("stylist_test", persist_directory=temp_dir)
            agent = StyleAgent(memory)

            assert agent.role == "stylist"
            assert agent.content_key == "style"
        finally:
            shutil.rmtree(temp_dir)


class TestAgentFactory:
    """Test agent factory."""

    def test_create_all_agents(self):
        """Factory should create all 7 agents."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)

            assert len(agents) == 7
            assert "coordinator" in agents
            assert "intro" in agents
            assert "body" in agents
            assert "conclusion" in agents
            assert "researcher" in agents
            assert "fact_checker" in agents
            assert "stylist" in agents
        finally:
            shutil.rmtree(temp_dir)

    def test_hierarchy_returned(self):
        """Factory should return hierarchy instance."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)

            assert isinstance(hierarchy, AgentHierarchy)
            assert hierarchy.get_layer("coordinator") == 1
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
