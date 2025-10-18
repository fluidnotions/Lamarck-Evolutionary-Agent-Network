"""
Tests for MemoryManager with inheritance support.
"""

import pytest
import tempfile
import shutil
from hvas_mini.memory import MemoryManager, generate_collection_name


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


def test_generate_collection_name():
    """Test collection name generation."""
    name = generate_collection_name("intro", "agent_1")
    assert name == "intro_agent_agent_1_memories"

    # Test sanitization
    name = generate_collection_name("intro!", "agent-1")
    assert name == "intro_agent_agent1_memories"


def test_store_all_experiences(temp_dir):
    """Verify ALL experiences stored regardless of score."""
    memory = MemoryManager(
        collection_name="test_store_all",
        persist_directory=temp_dir
    )

    # Store low, medium, high scores
    scores = [2.0, 5.0, 8.0, 10.0]
    for i, score in enumerate(scores):
        memory.store_experience(
            output=f"Output with score {score}",
            score=score,
            metadata={"test_id": i}
        )

    # Retrieve all
    all_memories = memory.get_all_memories()
    assert len(all_memories) == 4

    # Verify all scores present
    stored_scores = [m['score'] for m in all_memories]
    assert set(stored_scores) == set(scores)


def test_weighted_retrieval(temp_dir):
    """Verify high-score memories surface more in retrieval."""
    memory = MemoryManager(
        collection_name="test_weighted",
        persist_directory=temp_dir
    )

    # Store similar content with different scores
    memory.store_experience("Machine learning tutorial", score=9.0)
    memory.store_experience("Machine learning tutorial", score=3.0)
    memory.store_experience("Machine learning tutorial", score=7.0)

    # Retrieve with score weighting
    results = memory.retrieve_similar("machine learning", k=3, score_weight=1.0)

    # Highest score should rank first
    assert results[0]['score'] == 9.0
    assert results[0]['weighted_relevance'] > results[1]['weighted_relevance']


def test_inherited_memories(temp_dir):
    """Verify inherited memories load correctly."""
    inherited = [
        {'content': "Parent memory 1", 'score': 8.0, 'metadata': {'topic': 'ML'}},
        {'content': "Parent memory 2", 'score': 7.5, 'metadata': {'topic': 'Python'}}
    ]

    memory = MemoryManager(
        collection_name="child_agent",
        persist_directory=temp_dir,
        inherited_memories=inherited
    )

    all_mems = memory.get_all_memories()
    assert len(all_mems) == 2

    # Check inherited flag
    assert all(m['metadata']['inherited'] for m in all_mems)

    # Personal memories should be empty
    personal = memory.get_personal_memories()
    assert len(personal) == 0

    # Add personal memory
    memory.store_experience("Child's own experience", score=6.0)
    personal = memory.get_personal_memories()
    assert len(personal) == 1
    assert not personal[0]['metadata']['inherited']


def test_retrieval_counting(temp_dir):
    """Verify retrieval counts tracked."""
    memory = MemoryManager(
        collection_name="test_counting",
        persist_directory=temp_dir
    )

    # Store memories
    memory.store_experience("First memory", score=8.0)
    memory.store_experience("Second memory", score=7.0)

    # Retrieve multiple times
    for _ in range(3):
        results = memory.retrieve_similar("memory", k=2)

    # Check retrieval counts
    all_mems = memory.get_all_memories()
    assert all(m['retrieval_count'] == 3 for m in all_mems)


def test_legacy_compatibility(temp_dir):
    """Test legacy store/retrieve methods still work."""
    memory = MemoryManager(
        collection_name="test_legacy",
        persist_directory=temp_dir
    )

    # Use legacy store method
    memory_id = memory.store(
        content="Legacy content",
        topic="test",
        score=8.0
    )
    assert memory_id != ""

    # Use legacy retrieve method
    results = memory.retrieve("test")
    assert len(results) > 0
    assert results[0]['content'] == "Legacy content"


def test_get_stats(temp_dir):
    """Test stats include inherited/personal counts."""
    inherited = [
        {'content': "Inherited", 'score': 8.0, 'metadata': {}}
    ]

    memory = MemoryManager(
        collection_name="test_stats",
        persist_directory=temp_dir,
        inherited_memories=inherited
    )

    # Add personal memory
    memory.store_experience("Personal", score=7.0)

    stats = memory.get_stats()
    assert stats['total_memories'] == 2
    assert stats['inherited_memories'] == 1
    assert stats['personal_memories'] == 1
