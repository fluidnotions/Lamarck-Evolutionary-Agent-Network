"""Tests for memory system."""

from lean.memory import MemoryManager
import tempfile
import shutil
import pytest


@pytest.fixture
def temp_dir():
    """Create temporary directory for ChromaDB."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def memory_manager(temp_dir):
    """Create memory manager for testing."""
    return MemoryManager(collection_name="test_collection", persist_directory=temp_dir)


def test_memory_storage(memory_manager):
    """Test storing memories."""
    memory_id = memory_manager.store(
        content="Machine learning is fascinating", topic="ML basics", score=8.5
    )

    assert memory_id != ""
    assert memory_manager.count() == 1


def test_memory_threshold(memory_manager):
    """Test score threshold filtering."""
    # Below threshold (default 7.0)
    memory_id = memory_manager.store(
        content="Low quality content", topic="test", score=5.0
    )

    assert memory_id == ""
    assert memory_manager.count() == 0


def test_memory_retrieval(memory_manager):
    """Test retrieving similar memories."""
    # Store some memories
    memory_manager.store(
        content="Machine learning uses algorithms to learn from data",
        topic="ML",
        score=9.0,
    )
    memory_manager.store(
        content="Deep learning is a subset of machine learning", topic="ML", score=8.5
    )

    # Retrieve
    results = memory_manager.retrieve("what is machine learning")

    assert len(results) > 0
    assert "machine learning" in results[0]["content"].lower()
    assert results[0]["score"] >= 7.0


def test_memory_stats(memory_manager):
    """Test memory statistics."""
    memory_manager.store("Content 1", "topic", 8.0)
    memory_manager.store("Content 2", "topic", 9.0)

    stats = memory_manager.get_stats()

    assert stats["total_memories"] == 2
    assert stats["avg_score"] == 8.5
