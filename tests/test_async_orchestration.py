"""Tests for async orchestration."""

import pytest
import asyncio
from lean.orchestration.async_coordinator import AsyncCoordinator
from lean.state import create_initial_state


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, role: str, delay: float = 0.1):
        self.role = role
        self.delay = delay

    async def __call__(self, state):
        await asyncio.sleep(self.delay)
        state[self.role] = f"Content from {self.role}"
        return state


@pytest.mark.asyncio
async def test_concurrent_execution():
    """Test that agents execute concurrently."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    agent1 = MockAgent("agent1", delay=0.2)
    agent2 = MockAgent("agent2", delay=0.2)

    start_time = asyncio.get_event_loop().time()

    result = await coordinator.execute_layer(
        "test_layer",
        [agent1, agent2],
        state,
        timeout=5.0
    )

    end_time = asyncio.get_event_loop().time()
    duration = end_time - start_time

    # If truly concurrent, should take ~0.2s not ~0.4s
    assert duration < 0.3, f"Expected <0.3s (concurrent), got {duration:.2f}s (sequential?)"

    # Both agents should have completed
    assert "agent1" in result
    assert "agent2" in result


@pytest.mark.asyncio
async def test_timing_instrumentation():
    """Test that timing data is recorded."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    agent = MockAgent("test_agent", delay=0.1)
    result = await coordinator.execute_layer("layer", [agent], state)

    assert "agent_timings" in result
    assert "test_agent" in result["agent_timings"]
    assert "duration" in result["agent_timings"]["test_agent"]
    assert result["agent_timings"]["test_agent"]["duration"] >= 0.1


@pytest.mark.asyncio
async def test_layer_barrier_recorded():
    """Test that layer barriers are recorded."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    agent1 = MockAgent("agent1")
    agent2 = MockAgent("agent2")

    result = await coordinator.execute_layer("test_layer", [agent1, agent2], state)

    assert "layer_barriers" in result
    assert len(result["layer_barriers"]) > 0

    barrier = result["layer_barriers"][0]
    assert barrier["layer"] == "test_layer"
    assert set(barrier["agents"]) == {"agent1", "agent2"}
    assert "wait_time" in barrier


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test that coordinator handles timeouts gracefully."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    # Create agent that takes too long
    slow_agent = MockAgent("slow", delay=5.0)

    result = await coordinator.execute_layer(
        "timeout_layer",
        [slow_agent],
        state,
        timeout=0.1
    )

    # Should return state without blocking forever
    assert result == state  # Timeout returns original state


@pytest.mark.asyncio
async def test_state_merging():
    """Test that concurrent agent states are merged correctly."""
    coordinator = AsyncCoordinator()
    state = create_initial_state("test")

    agent1 = MockAgent("body")
    agent2 = MockAgent("conclusion")

    result = await coordinator.execute_layer(
        "content_layer",
        [agent1, agent2],
        state,
        timeout=5.0
    )

    # Both agents should have written to state
    assert "body" in result
    assert "conclusion" in result
    assert result["body"] == "Content from body"
    assert result["conclusion"] == "Content from conclusion"
