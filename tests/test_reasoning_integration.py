"""
Integration test for reasoning pattern architecture.

Tests the 8-step cycle with ReasoningMemory, SharedRAG, and BaseAgentV2.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path

# Import new classes
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.shared_rag import SharedRAG
from lean.base_agent_v2 import IntroAgentV2


class TestReasoningPatternIntegration:
    """Integration tests for reasoning pattern architecture."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        reasoning_dir = tempfile.mkdtemp(prefix="test_reasoning_")
        rag_dir = tempfile.mkdtemp(prefix="test_rag_")

        yield reasoning_dir, rag_dir

        # Cleanup
        shutil.rmtree(reasoning_dir, ignore_errors=True)
        shutil.rmtree(rag_dir, ignore_errors=True)

    def test_reasoning_memory_storage(self, temp_dirs):
        """Test reasoning pattern storage and retrieval."""
        reasoning_dir, _ = temp_dirs

        # Create reasoning memory
        collection_name = generate_reasoning_collection_name("intro", "test_1")
        memory = ReasoningMemory(
            collection_name=collection_name,
            persist_directory=reasoning_dir
        )

        # Store a reasoning pattern
        thinking = """My approach for this task:
1. Start with a compelling historical example
2. Add 2-3 supporting statistics
3. End with a provocative question

This pattern worked well for technical topics in the past."""

        pattern_id = memory.store_reasoning_pattern(
            reasoning=thinking,
            score=8.5,
            situation="writing intro for ML topic",
            tactic="historical example → statistics → question",
            metadata={'topic': 'neural networks', 'domain': 'ML', 'generation': 1}
        )

        assert pattern_id is not None
        assert memory.count() == 1

        # Retrieve similar reasoning
        results = memory.retrieve_similar_reasoning(
            query="Write introduction about deep learning",
            k=1
        )

        assert len(results) == 1
        assert results[0]['reasoning'] == thinking
        assert results[0]['score'] == 8.5
        assert results[0]['tactic'] == "historical example → statistics → question"

    def test_shared_rag_storage(self, temp_dirs):
        """Test shared RAG storage and retrieval."""
        _, rag_dir = temp_dirs

        # Create shared RAG
        rag = SharedRAG(persist_directory=rag_dir)

        # Store high-quality content
        content = "Neural networks are computational models inspired by biological brains..."
        knowledge_id = rag.store(
            content=content,
            metadata={'topic': 'neural networks', 'domain': 'ML', 'score': 8.5},
            source='generated'
        )

        assert knowledge_id is not None
        assert rag.count() == 1

        # Retrieve knowledge
        results = rag.retrieve(
            query="What are neural networks?",
            k=1
        )

        assert len(results) == 1
        assert content in results[0]['content']
        assert results[0]['domain'] == 'ML'

    def test_shared_rag_quality_threshold(self, temp_dirs):
        """Test that only high-quality content is stored in shared RAG."""
        _, rag_dir = temp_dirs

        rag = SharedRAG(persist_directory=rag_dir)

        # Try to store low-quality content
        low_id = rag.store_if_high_quality(
            content="Some mediocre content",
            score=6.0,  # Below threshold (8.0)
            metadata={'topic': 'test'}
        )

        assert low_id is None
        assert rag.count() == 0

        # Store high-quality content
        high_id = rag.store_if_high_quality(
            content="Excellent content",
            score=8.5,  # Above threshold
            metadata={'topic': 'test'}
        )

        assert high_id is not None
        assert rag.count() == 1

    def test_reasoning_inheritance(self, temp_dirs):
        """Test reasoning pattern inheritance from parents."""
        reasoning_dir, _ = temp_dirs

        # Create parent reasoning patterns
        parent_patterns = [
            {
                'reasoning': "Start with example, add stats, end with question",
                'score': 8.0,
                'situation': 'intro for ML topic',
                'tactic': 'example → stats → question',
                'metadata': {'generation': 0}
            },
            {
                'reasoning': "Use analogy, provide context, build intrigue",
                'score': 7.5,
                'situation': 'intro for Python topic',
                'tactic': 'analogy → context → intrigue',
                'metadata': {'generation': 0}
            }
        ]

        # Create child with inherited patterns
        collection_name = generate_reasoning_collection_name("intro", "child_1")
        child_memory = ReasoningMemory(
            collection_name=collection_name,
            persist_directory=reasoning_dir,
            inherited_reasoning=parent_patterns
        )

        # Check inheritance
        assert child_memory.count() == 2

        stats = child_memory.get_stats()
        assert stats['inherited_patterns'] == 2
        assert stats['personal_patterns'] == 0

        # Add personal pattern
        child_memory.store_reasoning_pattern(
            reasoning="My own approach: contrast past and present",
            score=8.2,
            situation="intro for Web topic",
            tactic="contrast → implications",
            metadata={'generation': 1}
        )

        # Check stats
        stats = child_memory.get_stats()
        assert stats['total_patterns'] == 3
        assert stats['inherited_patterns'] == 2
        assert stats['personal_patterns'] == 1

    def test_agent_generate_with_reasoning(self, temp_dirs):
        """Test agent generation with <think>/<final> parsing."""
        reasoning_dir, rag_dir = temp_dirs

        # Create agent
        collection_name = generate_reasoning_collection_name("intro", "test_agent")
        reasoning_memory = ReasoningMemory(
            collection_name=collection_name,
            persist_directory=reasoning_dir
        )
        shared_rag = SharedRAG(persist_directory=rag_dir)

        agent = IntroAgentV2(
            role="intro",
            agent_id="test_agent_1",
            reasoning_memory=reasoning_memory,
            shared_rag=shared_rag
        )

        # Note: This test requires ANTHROPIC_API_KEY
        # Skip if not available
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        # Generate with reasoning
        result = agent.generate_with_reasoning(
            topic="Understanding Neural Networks",
            reasoning_patterns=[],  # No patterns yet
            domain_knowledge=[],  # No knowledge yet
            reasoning_context=""
        )

        # Check response structure
        assert 'thinking' in result
        assert 'output' in result
        assert len(result['thinking']) > 0
        assert len(result['output']) > 0

        # Prepare and store reasoning
        agent.prepare_reasoning_storage(
            thinking=result['thinking'],
            output=result['output'],
            topic="Understanding Neural Networks",
            domain="ML",
            generation=1,
            context_sources=[]
        )

        agent.store_reasoning_and_output(score=8.5)

        # Verify storage
        assert reasoning_memory.count() == 1
        assert shared_rag.count() == 1  # High-quality output stored

    def test_eight_step_cycle(self, temp_dirs):
        """Test the complete 8-step reasoning cycle."""
        reasoning_dir, rag_dir = temp_dirs

        # Setup
        collection_name = generate_reasoning_collection_name("intro", "cycle_test")

        # Create parent patterns (STEP 1: Inheritance)
        parent_patterns = [
            {
                'reasoning': "Use historical context then modern application",
                'score': 8.0,
                'situation': 'intro for ML topic',
                'tactic': 'historical → modern',
                'metadata': {'generation': 0}
            }
        ]

        reasoning_memory = ReasoningMemory(
            collection_name=collection_name,
            persist_directory=reasoning_dir,
            inherited_reasoning=parent_patterns
        )

        shared_rag = SharedRAG(persist_directory=rag_dir)

        # Add some domain knowledge
        shared_rag.store(
            content="Neural networks consist of layers of interconnected nodes...",
            metadata={'topic': 'neural networks', 'domain': 'ML'},
            source='manual'
        )

        agent = IntroAgentV2(
            role="intro",
            agent_id="cycle_agent_1",
            reasoning_memory=reasoning_memory,
            shared_rag=shared_rag
        )

        # STEP 2: Plan approach (retrieve reasoning patterns)
        reasoning_patterns = reasoning_memory.retrieve_similar_reasoning(
            query="Write introduction about neural networks",
            k=5
        )
        assert len(reasoning_patterns) == 1  # Inherited pattern

        # STEP 3: Retrieve knowledge (shared RAG)
        domain_knowledge = shared_rag.retrieve(
            query="neural networks architecture",
            k=3
        )
        assert len(domain_knowledge) == 1

        # STEP 4: Receive context (simulated)
        reasoning_context = "Other agents used analogy-based approaches"

        # STEP 5: Generate (requires API key)
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set for full cycle test")

        result = agent.generate_with_reasoning(
            topic="Understanding Neural Networks",
            reasoning_patterns=reasoning_patterns,
            domain_knowledge=domain_knowledge,
            reasoning_context=reasoning_context
        )

        # STEP 6: Evaluate (simulated)
        score = 8.7

        # STEP 7: Store reasoning pattern
        agent.prepare_reasoning_storage(
            thinking=result['thinking'],
            output=result['output'],
            topic="Understanding Neural Networks",
            domain="ML",
            generation=1,
            context_sources=['hierarchy', 'high_credibility']
        )

        agent.record_fitness(score=score, domain="ML")
        agent.store_reasoning_and_output(score=score)

        # Verify results
        stats = reasoning_memory.get_stats()
        assert stats['total_patterns'] == 2  # 1 inherited + 1 personal
        assert stats['inherited_patterns'] == 1
        assert stats['personal_patterns'] == 1

        assert shared_rag.count() == 2  # 1 manual + 1 generated (score >= 8.0)

        assert agent.avg_fitness() == 8.7
        assert agent.task_count == 1

        # STEP 8: Evolve (would happen every 10 generations in M2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
