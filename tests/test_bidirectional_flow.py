"""
Test suite for bidirectional flow (M7).
"""

import pytest
from lean.hierarchy.executor import HierarchicalExecutor
from lean.hierarchy.factory import create_hierarchical_agents
from lean.state import create_hierarchical_state, AgentOutput
import tempfile
import shutil


class TestHierarchicalExecutor:
    """Test executor initialization and basic functionality."""

    def test_executor_creation(self):
        """HierarchicalExecutor should initialize with agents."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            assert executor.agents is not None
            assert len(executor.agents) == 7
        finally:
            shutil.rmtree(temp_dir)


class TestDownwardFlow:
    """Test context distribution from parent to children."""

    @pytest.mark.asyncio
    async def test_downward_execution_layer1(self):
        """Layer 1 (coordinator) should execute with topic as context."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Quantum Computing")

            # Execute coordinator (Layer 1)
            await executor.execute_downward(state, 1)

            # Coordinator should have output
            assert "coordinator" in state["layer_outputs"][1]
            assert len(state["layer_outputs"][1]["coordinator"]["content"]) > 0
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_downward_execution_layer2(self):
        """Layer 2 should receive context from coordinator."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Machine Learning")

            # Execute Layer 1 first
            await executor.execute_downward(state, 1)

            # Store coordinator intent for Layer 2 context
            state["coordinator_intent"] = state["layer_outputs"][1]["coordinator"]["content"]

            # Execute Layer 2
            await executor.execute_downward(state, 2)

            # All Layer 2 agents should have outputs
            assert "intro" in state["layer_outputs"][2]
            assert "body" in state["layer_outputs"][2]
            assert "conclusion" in state["layer_outputs"][2]
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_context_propagation(self):
        """Child agents should receive parent context."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Artificial Intelligence")

            # Execute layers 1 and 2
            await executor.execute_downward(state, 1)
            state["coordinator_intent"] = state["layer_outputs"][1]["coordinator"]["content"]
            await executor.execute_downward(state, 2)

            # Check that Layer 2 outputs have parent metadata
            for role in ["intro", "body", "conclusion"]:
                assert "parent" in state["layer_outputs"][2][role]["metadata"]
                assert state["layer_outputs"][2][role]["metadata"]["parent"] == "coordinator"
        finally:
            shutil.rmtree(temp_dir)


class TestUpwardFlow:
    """Test result aggregation from children to parent."""

    @pytest.mark.asyncio
    async def test_upward_aggregation_layer2(self):
        """Layer 2 should aggregate Layer 3 results."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Blockchain")

            # Execute full downward pass
            await executor.execute_downward(state, 1)
            state["coordinator_intent"] = state["layer_outputs"][1]["coordinator"]["content"]
            await executor.execute_downward(state, 2)
            await executor.execute_downward(state, 3)

            # Execute upward from Layer 3 to 2
            await executor.execute_upward(state, 2)

            # Check that Layer 2 agents have children metadata
            # Intro has children: researcher, stylist
            assert "children" in state["layer_outputs"][2]["intro"]["metadata"]
            assert state["layer_outputs"][2]["intro"]["metadata"]["children"]["source_count"] > 0
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_aggregation_with_confidence(self):
        """Aggregation should weight by confidence."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            # Create mock outputs with different confidences
            output1 = AgentOutput(
                content="High quality output" * 20,  # Longer = higher confidence
                confidence=0.8,
                metadata={}
            )
            output2 = AgentOutput(
                content="Short",
                confidence=0.3,
                metadata={}
            )

            aggregated = executor._aggregate_outputs([output1, output2])

            # Check aggregation structure
            assert "weighted_confidence" in aggregated
            assert "source_count" in aggregated
            assert aggregated["source_count"] == 2

            # Weighted confidence should be closer to higher-confidence output
            assert aggregated["weighted_confidence"] > 0.5
        finally:
            shutil.rmtree(temp_dir)


class TestConfidenceEstimation:
    """Test confidence scoring heuristics."""

    def test_confidence_empty_content(self):
        """Empty content should have minimum confidence."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, _ = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            confidence = executor._estimate_confidence("")
            assert confidence == 0.1
        finally:
            shutil.rmtree(temp_dir)

    def test_confidence_length_based(self):
        """Longer content should have higher confidence."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, _ = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            short = executor._estimate_confidence("Short text")
            long = executor._estimate_confidence("A" * 500)

            assert long > short
            assert 0.0 <= short <= 1.0
            assert 0.0 <= long <= 1.0
        finally:
            shutil.rmtree(temp_dir)

    def test_confidence_structure_based(self):
        """Well-structured content should boost confidence."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, _ = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            unstructured = "A" * 300
            structured = "Paragraph 1\n\nParagraph 2\n\nParagraph 3" + ("A" * 200)

            conf_unstructured = executor._estimate_confidence(unstructured)
            conf_structured = executor._estimate_confidence(structured)

            # Structured should have higher confidence
            assert conf_structured >= conf_unstructured
        finally:
            shutil.rmtree(temp_dir)

    def test_confidence_bounds(self):
        """Confidence should always be in [0, 1]."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, _ = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            # Test various content lengths
            test_cases = [
                "",
                "Short",
                "A" * 100,
                "A" * 500,
                "A" * 1000,
                "Para1\n\nPara2\n\nPara3\n\nPara4" + ("A" * 600)
            ]

            for content in test_cases:
                confidence = executor._estimate_confidence(content)
                assert 0.0 <= confidence <= 1.0
                assert confidence >= 0.1  # Minimum bound
        finally:
            shutil.rmtree(temp_dir)


class TestFullCycle:
    """Test complete downward + upward execution cycle."""

    @pytest.mark.asyncio
    async def test_full_cycle_execution(self):
        """Full cycle should execute all layers down then up."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Cloud Computing")

            # Execute full cycle
            state = await executor.execute_full_cycle(state)

            # All layers should have outputs
            assert len(state["layer_outputs"][1]) > 0  # Coordinator
            assert len(state["layer_outputs"][2]) > 0  # Content agents
            assert len(state["layer_outputs"][3]) > 0  # Specialists

            # Layer 2 should have aggregated children
            assert "children" in state["layer_outputs"][2]["intro"]["metadata"]
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_weighted_aggregation_ordering(self):
        """Higher confidence children should influence aggregation more."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, _ = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            # Create outputs with very different confidences
            high_conf = AgentOutput(
                content="This is high quality content" * 30,
                confidence=0.9,
                metadata={}
            )

            low_conf = AgentOutput(
                content="Low",
                confidence=0.2,
                metadata={}
            )

            aggregated = executor._aggregate_outputs([high_conf, low_conf])

            # Weighted confidence should be heavily influenced by high_conf
            # Expected: (0.9 * 0.9/(0.9+0.2)) + (0.2 * 0.2/(0.9+0.2))
            # ≈ 0.74 + 0.04 ≈ 0.78
            assert aggregated["weighted_confidence"] > 0.7
            assert aggregated["weighted_confidence"] < 0.9
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
