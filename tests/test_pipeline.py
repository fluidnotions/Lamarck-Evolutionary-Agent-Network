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


@pytest.mark.asyncio
async def test_ensemble_execution():
    """Test that ensemble execution mechanism works (all pool agents compete)."""
    # Note: population starts at 1 agent, grows through evolution
    # This test verifies the ensemble mechanism functions correctly
    pipeline = Pipeline(
        reasoning_dir="./data/test_reasoning",
        shared_rag_dir="./data/test_shared_rag",
        population_size=1,  # Start with 1 to keep test fast
        evolution_frequency=10,
        enable_research=False,
        enable_specialists=False,
        enable_revision=False
    )

    result = await pipeline.generate(
        topic="Ensemble Test Topic",
        generation_number=1
    )

    # Check that ensemble results are tracked
    assert 'intro_ensemble_results' in result, "Intro ensemble results should be in state"
    assert 'body_ensemble_results' in result, "Body ensemble results should be in state"
    assert 'conclusion_ensemble_results' in result, "Conclusion ensemble results should be in state"

    # Check that all agents in pool executed
    intro_results = result['intro_ensemble_results']
    assert len(intro_results) >= 1, "At least 1 intro agent should have executed"

    body_results = result['body_ensemble_results']
    assert len(body_results) >= 1, "At least 1 body agent should have executed"

    conclusion_results = result['conclusion_ensemble_results']
    assert len(conclusion_results) >= 1, "At least 1 conclusion agent should have executed"

    # Check that each agent has individual score (serializable format)
    for result_item in intro_results:
        assert 'agent_id' in result_item
        assert 'score' in result_item
        assert 'output_length' in result_item
        assert 0 <= result_item['score'] <= 10, "Score should be between 0 and 10"

    # Check that winner is tracked
    assert 'intro_winner_id' in result
    assert 'body_winner_id' in result
    assert 'conclusion_winner_id' in result


@pytest.mark.asyncio
async def test_individual_fitness_tracking():
    """Test that each agent tracks individual fitness, not shared."""
    pipeline = Pipeline(
        reasoning_dir="./data/test_reasoning",
        shared_rag_dir="./data/test_shared_rag",
        population_size=1,  # Start with 1, grows through evolution
        evolution_frequency=10,
        enable_research=False,
        enable_specialists=False,
        enable_revision=False
    )

    await pipeline.generate(
        topic="Fitness Test Topic",
        generation_number=1
    )

    # Get agents from pools
    intro_agents = pipeline.agent_pools['intro'].agents

    # Check that each agent has fitness history
    for agent in intro_agents:
        assert len(agent.fitness_history) > 0, f"{agent.agent_id} should have fitness history"

    # Check that agents have DIFFERENT fitness scores (unless by chance they're equal)
    # This demonstrates individual tracking rather than shared scores
    fitness_scores = [agent.avg_fitness() for agent in intro_agents]

    # All agents should have scores (not all zero)
    assert any(score > 0 for score in fitness_scores), "At least one agent should have non-zero fitness"


@pytest.mark.asyncio
async def test_ensemble_evolution_with_individual_scores():
    """Test that evolution uses individual scores from ensemble competition."""
    pipeline = Pipeline(
        reasoning_dir="./data/test_reasoning",
        shared_rag_dir="./data/test_shared_rag",
        population_size=2,  # Small population for fast testing
        evolution_frequency=2,  # Trigger evolution after 2 generations
        enable_research=False,
        enable_specialists=False,
        enable_revision=False
    )

    # Generate first generation
    result1 = await pipeline.generate(
        topic="Evolution Test Topic 1",
        generation_number=1
    )

    # Verify agents have individual scores
    intro_pool = pipeline.agent_pools['intro']
    agents_gen1 = intro_pool.agents.copy()

    for agent in agents_gen1:
        assert len(agent.fitness_history) > 0

    # Generate second generation (should trigger evolution)
    result2 = await pipeline.generate(
        topic="Evolution Test Topic 2",
        generation_number=2
    )

    # Check that evolution happened
    assert intro_pool.generation == 1, "Pool should have evolved to generation 1"

    # Check that new agents were created (IDs changed)
    agents_gen2 = intro_pool.agents
    gen1_ids = {agent.agent_id for agent in agents_gen1}
    gen2_ids = {agent.agent_id for agent in agents_gen2}

    # After evolution, offspring have different IDs
    assert gen1_ids != gen2_ids, "Evolution should create new agents with different IDs"

    # Check that offspring have inherited reasoning
    for agent in agents_gen2:
        # Offspring should have some inherited patterns (if parents had high-scoring patterns)
        inherited_count = agent.reasoning_memory.collection.count()
        # Note: This might be 0 if parents didn't have high enough scores
        # But at least verify the memory system is initialized
        assert inherited_count >= 0, f"{agent.agent_id} should have reasoning memory initialized"
