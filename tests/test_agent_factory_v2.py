"""
Test the create_agents_v2() factory function.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.base_agent_v2 import create_agents_v2, IntroAgentV2, BodyAgentV2, ConclusionAgentV2


def test_create_agents_v2_basic():
    """Test basic agent creation with factory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reasoning_dir = os.path.join(temp_dir, "reasoning")
        rag_dir = os.path.join(temp_dir, "shared_rag")

        agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=rag_dir
        )

        # Verify all agents created
        assert 'intro' in agents
        assert 'body' in agents
        assert 'conclusion' in agents

        # Verify agent types
        assert isinstance(agents['intro'], IntroAgentV2)
        assert isinstance(agents['body'], BodyAgentV2)
        assert isinstance(agents['conclusion'], ConclusionAgentV2)

        # Verify agent IDs
        assert agents['intro'].agent_id == 'intro_agent_1'
        assert agents['body'].agent_id == 'body_agent_1'
        assert agents['conclusion'].agent_id == 'conclusion_agent_1'

        # Verify roles
        assert agents['intro'].role == 'intro'
        assert agents['body'].role == 'body'
        assert agents['conclusion'].role == 'conclusion'


def test_create_agents_v2_custom_ids():
    """Test agent creation with custom IDs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reasoning_dir = os.path.join(temp_dir, "reasoning")
        rag_dir = os.path.join(temp_dir, "shared_rag")

        custom_ids = {
            'intro': 'custom_intro_1',
            'body': 'custom_body_2',
            'conclusion': 'custom_conclusion_3'
        }

        agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=rag_dir,
            agent_ids=custom_ids
        )

        # Verify custom IDs
        assert agents['intro'].agent_id == 'intro_custom_intro_1'
        assert agents['body'].agent_id == 'body_custom_body_2'
        assert agents['conclusion'].agent_id == 'conclusion_custom_conclusion_3'


def test_create_agents_v2_shared_rag():
    """Test that all agents share the same SharedRAG instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reasoning_dir = os.path.join(temp_dir, "reasoning")
        rag_dir = os.path.join(temp_dir, "shared_rag")

        agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=rag_dir
        )

        # All agents should share the same SharedRAG instance
        intro_rag = agents['intro'].shared_rag
        body_rag = agents['body'].shared_rag
        conclusion_rag = agents['conclusion'].shared_rag

        assert intro_rag is body_rag
        assert body_rag is conclusion_rag

        # Add something to shared RAG from one agent
        intro_rag.store(
            content="Test knowledge",
            metadata={'topic': 'test'},
            source='manual'
        )

        # Should be visible to all agents
        assert intro_rag.count() == 1
        assert body_rag.count() == 1
        assert conclusion_rag.count() == 1


def test_create_agents_v2_separate_reasoning_memory():
    """Test that each agent has its own ReasoningMemory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reasoning_dir = os.path.join(temp_dir, "reasoning")
        rag_dir = os.path.join(temp_dir, "shared_rag")

        agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=rag_dir
        )

        # Each agent should have different ReasoningMemory instance
        intro_memory = agents['intro'].reasoning_memory
        body_memory = agents['body'].reasoning_memory
        conclusion_memory = agents['conclusion'].reasoning_memory

        assert intro_memory is not body_memory
        assert body_memory is not conclusion_memory

        # Each should have different collection names
        assert intro_memory.collection_name != body_memory.collection_name
        assert body_memory.collection_name != conclusion_memory.collection_name

        # Add reasoning to one agent
        intro_memory.store_reasoning_pattern(
            reasoning="Test reasoning",
            score=8.5,
            situation="test",
            tactic="test"
        )

        # Should not be visible to other agents
        assert intro_memory.count() == 1
        assert body_memory.count() == 0
        assert conclusion_memory.count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
