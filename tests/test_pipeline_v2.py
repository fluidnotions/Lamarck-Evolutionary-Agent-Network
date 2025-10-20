"""
Test PipelineV2 with reasoning pattern architecture.
"""

import pytest
import os
import tempfile
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.pipeline_v2 import PipelineV2


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        reasoning_dir = os.path.join(temp_dir, "reasoning")
        rag_dir = os.path.join(temp_dir, "shared_rag")
        os.makedirs(reasoning_dir, exist_ok=True)
        os.makedirs(rag_dir, exist_ok=True)
        yield {
            'reasoning': reasoning_dir,
            'rag': rag_dir
        }


def test_pipeline_v2_initialization(temp_dirs):
    """Test that PipelineV2 initializes correctly."""
    pipeline = PipelineV2(
        reasoning_dir=temp_dirs['reasoning'],
        shared_rag_dir=temp_dirs['rag'],
        domain="Test"
    )

    # Verify agents created
    assert 'intro' in pipeline.agents
    assert 'body' in pipeline.agents
    assert 'conclusion' in pipeline.agents

    # Verify context manager
    assert pipeline.context_manager is not None

    # Verify agent pools
    assert 'intro' in pipeline.agent_pools
    assert 'body' in pipeline.agent_pools
    assert 'conclusion' in pipeline.agent_pools

    # Verify graph compiled
    assert pipeline.app is not None


def test_pipeline_v2_agent_pools(temp_dirs):
    """Test that agent pools work with ContextManager."""
    pipeline = PipelineV2(
        reasoning_dir=temp_dirs['reasoning'],
        shared_rag_dir=temp_dirs['rag']
    )

    # Test pool interface
    intro_pool = pipeline.agent_pools['intro']
    assert intro_pool.size() == 1
    assert intro_pool.get_top_n(n=1)[0] == pipeline.agents['intro']


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY for LLM calls"
)
@pytest.mark.asyncio
async def test_pipeline_v2_single_generation(temp_dirs):
    """Test a single generation through the V2 pipeline."""
    pipeline = PipelineV2(
        reasoning_dir=temp_dirs['reasoning'],
        shared_rag_dir=temp_dirs['rag'],
        domain="Test"
    )

    # Seed shared RAG
    pipeline.agents['intro'].shared_rag.store(
        content="Python is a programming language known for simplicity.",
        metadata={'topic': 'python', 'domain': 'Test'},
        source='manual'
    )

    # Run single generation
    final_state = await pipeline.generate(
        topic="Getting Started with Python",
        generation_number=1
    )

    # Verify state fields populated
    assert final_state['intro'] != ""
    assert final_state['body'] != ""
    assert final_state['conclusion'] != ""

    # Verify reasoning fields
    assert final_state['intro_reasoning'] != ""
    assert final_state['body_reasoning'] != ""
    assert final_state['conclusion_reasoning'] != ""

    # Verify scores
    assert 'intro' in final_state['scores']
    assert 'body' in final_state['scores']
    assert 'conclusion' in final_state['scores']

    # Verify reasoning/knowledge tracking
    assert 'intro' in final_state['reasoning_patterns_used']
    assert 'intro' in final_state['domain_knowledge_used']

    # Verify timing
    assert 'intro' in final_state['agent_timings']
    assert 'body' in final_state['agent_timings']
    assert 'conclusion' in final_state['agent_timings']

    # Verify logs
    assert len(final_state['stream_logs']) > 0


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY for LLM calls"
)
@pytest.mark.asyncio
async def test_pipeline_v2_multi_generation(temp_dirs):
    """Test multiple generations to verify reasoning pattern accumulation."""
    pipeline = PipelineV2(
        reasoning_dir=temp_dirs['reasoning'],
        shared_rag_dir=temp_dirs['rag'],
        domain="Test"
    )

    # Run 3 generations on same topic
    topic = "Understanding Machine Learning"
    states = []

    for gen in range(1, 4):
        state = await pipeline.generate(
            topic=topic,
            generation_number=gen
        )
        states.append(state)

    # Verify reasoning patterns accumulated
    agent_stats = pipeline.get_agent_stats()

    # Each agent should have 3 reasoning patterns (one per generation)
    assert agent_stats['intro']['reasoning_patterns'] == 3
    assert agent_stats['body']['reasoning_patterns'] == 3
    assert agent_stats['conclusion']['reasoning_patterns'] == 3

    # Verify shared RAG grew
    rag_stats = pipeline.get_shared_rag_stats()
    assert rag_stats['total_knowledge'] > 0  # Should have some high-quality outputs

    # Verify later generations used reasoning patterns
    # Generation 2 and 3 should have retrieved patterns
    assert states[1]['reasoning_patterns_used']['intro'] > 0
    assert states[2]['reasoning_patterns_used']['intro'] > 0


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY for LLM calls"
)
@pytest.mark.asyncio
async def test_pipeline_v2_context_distribution(temp_dirs):
    """Test that context distribution works correctly."""
    pipeline = PipelineV2(
        reasoning_dir=temp_dirs['reasoning'],
        shared_rag_dir=temp_dirs['rag']
    )

    # Run one generation to populate reasoning
    await pipeline.generate(
        topic="Test Topic",
        generation_number=1
    )

    # Check context flow
    context_stats = pipeline.get_context_flow_stats()
    assert 'diversity_score' in context_stats


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY for LLM calls"
)
@pytest.mark.asyncio
async def test_pipeline_v2_quality_threshold(temp_dirs):
    """Test that only high-quality outputs go to shared RAG."""
    pipeline = PipelineV2(
        reasoning_dir=temp_dirs['reasoning'],
        shared_rag_dir=temp_dirs['rag']
    )

    # Get initial shared RAG count
    initial_count = pipeline.get_shared_rag_stats()['total_knowledge']

    # Run generation
    final_state = await pipeline.generate(
        topic="Test Topic",
        generation_number=1
    )

    # Get final shared RAG count
    final_count = pipeline.get_shared_rag_stats()['total_knowledge']

    # Only outputs with score >= 8.0 should be added
    high_quality_count = sum(
        1 for score in final_state['scores'].values() if score >= 8.0
    )

    # Shared RAG growth should match high-quality count
    assert final_count - initial_count == high_quality_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
