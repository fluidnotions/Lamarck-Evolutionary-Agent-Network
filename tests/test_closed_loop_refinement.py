"""
Test suite for closed-loop refinement (M8).
"""

import pytest
from lean.hierarchy.executor import HierarchicalExecutor
from lean.hierarchy.factory import create_hierarchical_agents
from lean.state import create_hierarchical_state, AgentOutput
import tempfile
import shutil
import os


class TestMultiPassExecution:
    """Test multi-pass execution with refinement."""

    @pytest.mark.asyncio
    async def test_execute_with_refinement_basic(self):
        """Should execute multiple passes if quality not met."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Artificial Intelligence")
            state["max_passes"] = 2

            # Execute with refinement
            state = await executor.execute_with_refinement(state)

            # Should have pass history
            assert len(state["pass_history"]) > 0
            assert state["pass_history"][0]["pass"] == 1
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_pass_history_tracking(self):
        """Pass history should track scores for each pass."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Machine Learning")
            state["max_passes"] = 2

            state = await executor.execute_with_refinement(state)

            # Each pass should have scores
            for pass_record in state["pass_history"]:
                assert "pass" in pass_record
                assert "scores" in pass_record
                assert isinstance(pass_record["scores"], dict)
        finally:
            shutil.rmtree(temp_dir)


class TestQualityThreshold:
    """Test quality threshold checking."""

    @pytest.mark.asyncio
    async def test_critique_and_decide_quality_met(self):
        """Should return False if quality threshold met."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Test Topic")
            state["current_pass"] = 1
            state["max_passes"] = 3

            # Set high confidence outputs
            state["layer_outputs"][2]["intro"] = AgentOutput(
                content="High quality intro" * 50,
                confidence=0.9,
                metadata={}
            )
            state["layer_outputs"][2]["body"] = AgentOutput(
                content="High quality body" * 50,
                confidence=0.9,
                metadata={}
            )
            state["layer_outputs"][2]["conclusion"] = AgentOutput(
                content="High quality conclusion" * 50,
                confidence=0.9,
                metadata={}
            )

            needs_revision = await executor.critique_and_decide(state)

            # Quality met, no revision needed
            assert needs_revision is False
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_critique_and_decide_quality_not_met(self):
        """Should return True if quality below threshold."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Test Topic")
            state["current_pass"] = 1
            state["max_passes"] = 3

            # Set low confidence outputs
            state["layer_outputs"][2]["intro"] = AgentOutput(
                content="Short",
                confidence=0.3,
                metadata={}
            )
            state["layer_outputs"][2]["body"] = AgentOutput(
                content="Brief",
                confidence=0.4,
                metadata={}
            )
            state["layer_outputs"][2]["conclusion"] = AgentOutput(
                content="Done",
                confidence=0.3,
                metadata={}
            )

            # Set quality threshold
            os.environ["QUALITY_THRESHOLD"] = "0.8"

            needs_revision = await executor.critique_and_decide(state)

            # Quality not met, revision needed
            assert needs_revision is True
        finally:
            shutil.rmtree(temp_dir)
            if "QUALITY_THRESHOLD" in os.environ:
                del os.environ["QUALITY_THRESHOLD"]

    @pytest.mark.asyncio
    async def test_max_passes_limit(self):
        """Should stop after max passes even if quality not met."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Test Topic")
            state["current_pass"] = 3  # At max
            state["max_passes"] = 3

            # Set low confidence
            state["layer_outputs"][2]["intro"] = AgentOutput(
                content="Low",
                confidence=0.3,
                metadata={}
            )

            needs_revision = await executor.critique_and_decide(state)

            # No more passes, so no revision
            assert needs_revision is False
        finally:
            shutil.rmtree(temp_dir)


class TestCoordinatorCritique:
    """Test coordinator critique generation."""

    @pytest.mark.asyncio
    async def test_critique_generation(self):
        """Coordinator should generate critiques for each agent."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Test Topic")
            state["current_pass"] = 1

            # Set outputs
            state["layer_outputs"][2]["intro"] = AgentOutput(
                content="Short intro",
                confidence=0.5,
                metadata={}
            )
            state["layer_outputs"][2]["body"] = AgentOutput(
                content="Body content here",
                confidence=0.6,
                metadata={}
            )
            state["layer_outputs"][2]["conclusion"] = AgentOutput(
                content="Conclusion",
                confidence=0.4,
                metadata={}
            )

            await executor.critique_and_decide(state)

            # Should have critiques
            assert "coordinator_critique" in state
            assert isinstance(state["coordinator_critique"], dict)
        finally:
            shutil.rmtree(temp_dir)


class TestRevisionRequest:
    """Test revision request generation."""

    @pytest.mark.asyncio
    async def test_revision_requested_flag(self):
        """Revision should set revision_requested flag."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Test Topic")

            # Set critiques
            state["coordinator_critique"] = {
                "intro": "Issues: too short (< 100 chars)",
                "body": "Good quality",
                "conclusion": "Issues: low confidence (0.40)"
            }

            # Set outputs
            state["layer_outputs"][2]["intro"] = AgentOutput(
                content="Short",
                confidence=0.3,
                metadata={}
            )
            state["layer_outputs"][2]["conclusion"] = AgentOutput(
                content="Brief",
                confidence=0.4,
                metadata={}
            )

            await executor.request_revision(state)

            # Flag should be set
            assert state["revision_requested"] is True
        finally:
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_revision_prompts_generated(self):
        """Revision should generate detailed prompts."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Test Topic")

            # Set critiques
            state["coordinator_critique"] = {
                "intro": "Issues: too short"
            }

            # Set output
            state["layer_outputs"][2]["intro"] = AgentOutput(
                content="Short intro",
                confidence=0.3,
                metadata={}
            )

            await executor.request_revision(state)

            # Revision prompt should be generated
            critique = state["coordinator_critique"]["intro"]
            assert "Revision needed" in critique
            assert "Short intro" in critique  # Original output included
        finally:
            shutil.rmtree(temp_dir)


class TestEarlyExit:
    """Test early exit when quality threshold met."""

    @pytest.mark.asyncio
    async def test_early_exit_on_quality(self):
        """Should exit early if quality threshold met on first pass."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Simple Topic")
            state["max_passes"] = 3

            # Set low threshold for early exit
            os.environ["QUALITY_THRESHOLD"] = "0.5"

            state = await executor.execute_with_refinement(state)

            # Should exit early if quality met
            # (actual result depends on generated content quality)
            assert "quality_threshold_met" in state
            assert isinstance(state["quality_threshold_met"], bool)
        finally:
            shutil.rmtree(temp_dir)
            if "QUALITY_THRESHOLD" in os.environ:
                del os.environ["QUALITY_THRESHOLD"]


class TestEndToEnd:
    """Test full refinement cycle."""

    @pytest.mark.asyncio
    async def test_full_refinement_cycle(self):
        """Should complete full refinement process."""
        temp_dir = tempfile.mkdtemp()

        try:
            agents, hierarchy = create_hierarchical_agents(temp_dir)
            executor = HierarchicalExecutor(agents)

            state = create_hierarchical_state("Cloud Computing")
            state["max_passes"] = 2

            # Execute with refinement
            state = await executor.execute_with_refinement(state)

            # Should have executed
            assert len(state["pass_history"]) > 0

            # Should have layer outputs
            assert len(state["layer_outputs"][1]) > 0  # Coordinator
            assert len(state["layer_outputs"][2]) > 0  # Content agents

            # Should have critiques
            assert "coordinator_critique" in state
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
