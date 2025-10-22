"""
Tests for the LEAN hierarchical pipeline.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.pipeline import Pipeline
from lean.state import create_initial_state


@pytest.fixture
def minimal_pipeline():
    """Create a minimal pipeline for testing."""
    return Pipeline(
        reasoning_dir="./data/test_reasoning",
        shared_rag_dir="./data/test_shared_rag",
        population_size=1,  # Minimal for speed
        evolution_frequency=10,
        enable_research=False,  # Disable for testing
        enable_specialists=False,  # Disable for testing
        enable_revision=False  # Disable for testing
    )


@pytest.mark.asyncio
async def test_pipeline_initialization(minimal_pipeline):
    """Test that pipeline initializes correctly."""
    assert minimal_pipeline is not None
    assert minimal_pipeline.coordinator is not None
    assert 'intro' in minimal_pipeline.agent_pools
    assert 'body' in minimal_pipeline.agent_pools
    assert 'conclusion' in minimal_pipeline.agent_pools
    assert minimal_pipeline.enable_research is False
    assert minimal_pipeline.enable_specialists is False
    assert minimal_pipeline.enable_revision is False


@pytest.mark.asyncio
async def test_pipeline_generate_basic(minimal_pipeline):
    """Test basic content generation."""
    result = await minimal_pipeline.generate(
        topic="Test Topic",
        generation_number=1
    )

    # Check that all sections were generated
    assert 'intro' in result
    assert 'body' in result
    assert 'conclusion' in result

    # Check that content is not empty
    assert len(result['intro']) > 0
    assert len(result['body']) > 0
    assert len(result['conclusion']) > 0

    # Check that scores were assigned
    assert 'scores' in result
    assert 'intro' in result['scores']
    assert 'body' in result['scores']
    assert 'conclusion' in result['scores']


@pytest.mark.asyncio
async def test_pipeline_with_research():
    """Test pipeline with research enabled (if TAVILY_API_KEY set)."""
    import os
    if not os.getenv('TAVILY_API_KEY'):
        pytest.skip("TAVILY_API_KEY not set")

    pipeline = Pipeline(
        reasoning_dir="./data/test_reasoning",
        shared_rag_dir="./data/test_shared_rag",
        population_size=1,
        evolution_frequency=10,
        enable_research=True,
        enable_specialists=False,
        enable_revision=False
    )

    result = await pipeline.generate(
        topic="AI in Healthcare",
        generation_number=1
    )

    # Check for research results
    assert 'research_results' in result
    assert result['research_results'] is not None


@pytest.mark.asyncio
async def test_pipeline_agent_pools(minimal_pipeline):
    """Test that agent pools work correctly."""
    # Check initial state
    assert minimal_pipeline.agent_pools['intro'].size() == 1
    assert minimal_pipeline.agent_pools['body'].size() == 1
    assert minimal_pipeline.agent_pools['conclusion'].size() == 1

    # Generate to update fitness
    await minimal_pipeline.generate(
        topic="Test Topic",
        generation_number=1
    )

    # Check that agents have fitness scores
    intro_agent = minimal_pipeline.agent_pools['intro'].agents[0]
    assert len(intro_agent.fitness_history) > 0


def test_pipeline_state_structure():
    """Test that initial state has correct structure."""
    state = create_initial_state("Test Topic")

    # Check required keys
    assert 'topic' in state
    assert 'intro' in state
    assert 'body' in state
    assert 'conclusion' in state
    assert 'scores' in state
    assert 'stream_logs' in state
    assert 'agent_timings' in state

    # Check initial values
    assert state['topic'] == "Test Topic"
    assert state['intro'] == ""
    assert state['body'] == ""
    assert state['conclusion'] == ""
    assert state['scores'] == {}
    assert isinstance(state['stream_logs'], list)
    assert isinstance(state['agent_timings'], dict)


@pytest.mark.asyncio
async def test_pipeline_reasoning_storage(minimal_pipeline):
    """Test that reasoning patterns are stored."""
    result = await minimal_pipeline.generate(
        topic="Test Topic for Reasoning",
        generation_number=1
    )

    # Check that reasoning was captured
    assert 'intro_reasoning' in result
    assert 'body_reasoning' in result
    assert 'conclusion_reasoning' in result

    # Check that at least one has content
    has_reasoning = (
        len(result.get('intro_reasoning', '')) > 0 or
        len(result.get('body_reasoning', '')) > 0 or
        len(result.get('conclusion_reasoning', '')) > 0
    )
    assert has_reasoning, "At least one section should have reasoning"
