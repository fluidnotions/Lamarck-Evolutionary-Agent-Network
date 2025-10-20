"""
Integration test for M2 evolutionary learning cycle.

Tests the complete pipeline with evolution:
- Multi-generation execution
- Pool evolution trigger
- Population replacement
- Fitness tracking
"""

import pytest
import tempfile
import shutil
import os


@pytest.mark.asyncio
async def test_evolution_cycle():
    """Test that evolution triggers and populations evolve."""
    # Create temp directories
    reasoning_dir = tempfile.mkdtemp()
    shared_rag_dir = tempfile.mkdtemp()

    try:
        from lean.pipeline_v2 import PipelineV2

        # Initialize pipeline with small population and fast evolution
        pipeline = PipelineV2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=shared_rag_dir,
            population_size=3,  # Small for testing
            evolution_frequency=5  # Evolve every 5 generations
        )

        # Verify initial setup
        assert len(pipeline.agent_pools) == 3
        assert pipeline.evolution_frequency == 5

        # Check initial pool states
        for role, pool in pipeline.agent_pools.items():
            assert pool.generation == 0
            assert len(pool.agents) == 1  # Start with 1 agent

        # Run 3 generations (no evolution yet)
        topics = [
            "test topic 1",
            "test topic 2",
            "test topic 3"
        ]

        for i, topic in enumerate(topics, 1):
            result = await pipeline.generate(topic, generation_number=i)

            # Check output was generated
            assert result['intro']
            assert result['body']
            assert result['conclusion']
            assert 'scores' in result

        # Verify no evolution yet (gen 3 < 5)
        for role, pool in pipeline.agent_pools.items():
            assert pool.generation == 0, f"{role} evolved too early"

        # Run 2 more generations to trigger evolution (gen 5)
        for i in range(4, 6):
            result = await pipeline.generate(f"test topic {i}", generation_number=i)

        # Verify evolution occurred at generation 5
        for role, pool in pipeline.agent_pools.items():
            assert pool.generation == 1, f"{role} didn't evolve at gen 5"
            assert len(pool.agents) == 3, f"{role} pool didn't grow to population_size"

        # Run one more to verify pools work post-evolution
        result = await pipeline.generate("test topic 6", generation_number=6)
        assert result['intro']

        print("✅ Evolution cycle test passed!")

    finally:
        # Cleanup
        shutil.rmtree(reasoning_dir, ignore_errors=True)
        shutil.rmtree(shared_rag_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_fitness_tracking():
    """Test that agent fitness is tracked across generations."""
    reasoning_dir = tempfile.mkdtemp()
    shared_rag_dir = tempfile.mkdtemp()

    try:
        from lean.pipeline_v2 import PipelineV2

        pipeline = PipelineV2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=shared_rag_dir,
            population_size=2,
            evolution_frequency=10
        )

        # Run 2 generations
        await pipeline.generate("topic 1", generation_number=1)
        await pipeline.generate("topic 2", generation_number=2)

        # Check fitness was recorded
        for role, pool in pipeline.agent_pools.items():
            for agent in pool.agents:
                assert agent.get_stats()['task_count'] >= 2, f"{role} agent didn't track tasks"

        print("✅ Fitness tracking test passed!")

    finally:
        shutil.rmtree(reasoning_dir, ignore_errors=True)
        shutil.rmtree(shared_rag_dir, ignore_errors=True)


if __name__ == "__main__":
    import asyncio

    print("Running M2 evolution integration tests...")
    asyncio.run(test_evolution_cycle())
    asyncio.run(test_fitness_tracking())
    print("\n✅ All tests passed!")
