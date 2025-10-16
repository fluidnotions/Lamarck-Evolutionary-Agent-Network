"""Tests for state management."""

from hvas_mini.state import (
    AgentMemory,
    BlogState,
    create_initial_state,
    validate_state,
)
import pytest


def test_agent_memory_creation():
    """Test AgentMemory model creation."""
    memory = AgentMemory(content="Test content", topic="test topic", score=8.5)

    assert memory.content == "Test content"
    assert memory.topic == "test topic"
    assert memory.score == 8.5
    assert memory.retrieval_count == 0
    assert memory.timestamp is not None


def test_agent_memory_validation():
    """Test AgentMemory score validation."""
    with pytest.raises(Exception):
        AgentMemory(
            content="Test",
            topic="test",
            score=11.0,  # Invalid: > 10
        )


def test_create_initial_state():
    """Test initial state creation."""
    state = create_initial_state("machine learning")

    assert state["topic"] == "machine learning"
    assert state["intro"] == ""
    assert state["body"] == ""
    assert state["conclusion"] == ""
    assert state["generation_id"] != ""
    assert len(state["stream_logs"]) == 0


def test_validate_state():
    """Test state validation."""
    state = create_initial_state("test")
    assert validate_state(state) is True

    # Test missing key
    incomplete_state = {"topic": "test"}  # type: ignore
    with pytest.raises(ValueError):
        validate_state(incomplete_state)  # type: ignore
