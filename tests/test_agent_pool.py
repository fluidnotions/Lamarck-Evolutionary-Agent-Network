"""
Tests for AgentPool class and agent pool management.
"""

import pytest
import random
from lean.agent_pool import AgentPool, create_agent_pool
from lean.agents import IntroAgent
from lean.memory import MemoryManager


@pytest.fixture
def sample_pool():
    """Create a sample agent pool for testing."""
    return create_agent_pool("intro", population_size=5)


def test_agent_pool_initialization():
    """Test AgentPool initialization with correct parameters."""
    pool = AgentPool(role="intro", population_size=5, min_size=3, max_size=8)

    assert pool.role == "intro"
    assert pool.min_size == 3
    assert pool.max_size == 8
    assert pool.generation == 0
    assert len(pool.agents) == 0  # Empty until agents added


def test_create_agent_pool():
    """Test factory function creates populated pool."""
    pool = create_agent_pool("intro", population_size=5)

    assert pool.size() == 5
    assert pool.role == "intro"
    assert all(a.role == "intro" for a in pool.agents)
    assert all(hasattr(a, 'agent_id') for a in pool.agents)
    assert all(hasattr(a, 'fitness_history') for a in pool.agents)


def test_create_agent_pool_all_roles():
    """Test pool creation for all three roles."""
    for role in ["intro", "body", "conclusion"]:
        pool = create_agent_pool(role, population_size=3)
        assert pool.size() == 3
        assert pool.role == role


def test_create_agent_pool_invalid_role():
    """Test pool creation fails with invalid role."""
    with pytest.raises(ValueError, match="Unknown role"):
        create_agent_pool("invalid_role", population_size=5)


def test_create_agent_pool_invalid_population_size():
    """Test pool creation fails with invalid population size."""
    with pytest.raises(ValueError, match="population_size"):
        create_agent_pool("intro", population_size=10, min_size=3, max_size=8)

    with pytest.raises(ValueError, match="population_size"):
        create_agent_pool("intro", population_size=2, min_size=3, max_size=8)


def test_add_agent(sample_pool):
    """Test adding agent to pool."""
    initial_size = sample_pool.size()

    # Create new agent
    memory = MemoryManager(collection_name="test_memories")
    new_agent = IntroAgent(role="intro", memory_manager=memory)
    new_agent.agent_id = "test_agent"
    new_agent.fitness_history = []
    new_agent.task_count = 0

    # Add method
    import types
    new_agent.avg_fitness = types.MethodType(
        lambda self: 0.0 if not self.fitness_history else sum(self.fitness_history) / len(self.fitness_history),
        new_agent
    )

    sample_pool.add_agent(new_agent)

    assert sample_pool.size() == initial_size + 1
    assert new_agent in sample_pool.agents


def test_add_agent_at_max_size():
    """Test adding agent fails when pool at max size."""
    pool = create_agent_pool("intro", population_size=8, max_size=8)

    memory = MemoryManager(collection_name="test_memories")
    new_agent = IntroAgent(role="intro", memory_manager=memory)

    with pytest.raises(ValueError, match="maximum size"):
        pool.add_agent(new_agent)


def test_remove_agent(sample_pool):
    """Test removing agent from pool."""
    initial_size = sample_pool.size()
    agent_to_remove = sample_pool.agents[0]

    result = sample_pool.remove_agent(agent_to_remove.agent_id)

    assert result is True
    assert sample_pool.size() == initial_size - 1
    assert agent_to_remove not in sample_pool.agents


def test_remove_agent_not_found(sample_pool):
    """Test removing nonexistent agent returns False."""
    result = sample_pool.remove_agent("nonexistent_agent")
    assert result is False


def test_remove_agent_at_min_size():
    """Test removing agent fails when pool at minimum size."""
    pool = create_agent_pool("intro", population_size=3, min_size=3)

    with pytest.raises(ValueError, match="minimum size"):
        pool.remove_agent(pool.agents[0].agent_id)


def test_select_best_agent(sample_pool):
    """Test selecting best agent by fitness."""
    # Set different fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i)]  # 0.0, 1.0, 2.0, 3.0, 4.0

    best_agent = sample_pool.select_agent(strategy="best")

    assert best_agent.avg_fitness() == 4.0


def test_select_epsilon_greedy_exploration(sample_pool):
    """Test epsilon-greedy with full exploration (epsilon=1.0)."""
    # Set fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i)]

    # With epsilon=1.0, should always explore (random)
    random.seed(42)
    selected = sample_pool.select_agent(strategy="epsilon_greedy", epsilon=1.0)

    # Just verify it returns an agent (randomness makes specific check hard)
    assert selected in sample_pool.agents


def test_select_epsilon_greedy_exploitation(sample_pool):
    """Test epsilon-greedy with full exploitation (epsilon=0.0)."""
    # Set fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i)]

    # With epsilon=0.0, should always exploit (best)
    selected = sample_pool.select_agent(strategy="epsilon_greedy", epsilon=0.0)

    assert selected.avg_fitness() == 4.0


def test_select_fitness_weighted(sample_pool):
    """Test fitness-weighted selection."""
    # Set fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i + 1)]  # 1.0, 2.0, 3.0, 4.0, 5.0

    # Run multiple times, check distribution makes sense
    random.seed(42)
    selections = [
        sample_pool.select_agent(strategy="fitness_weighted")
        for _ in range(100)
    ]

    # Higher fitness should be selected more often
    best_agent = sample_pool.agents[-1]
    worst_agent = sample_pool.agents[0]

    best_count = selections.count(best_agent)
    worst_count = selections.count(worst_agent)

    assert best_count > worst_count  # Higher fitness selected more


def test_select_fitness_weighted_zero_fitness(sample_pool):
    """Test fitness-weighted selection with all zero fitness."""
    # All agents have zero fitness
    for agent in sample_pool.agents:
        agent.fitness_history = []

    # Should not crash, should return random agent
    selected = sample_pool.select_agent(strategy="fitness_weighted")
    assert selected in sample_pool.agents


def test_select_tournament(sample_pool):
    """Test tournament selection."""
    # Set fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i)]  # 0.0, 1.0, 2.0, 3.0, 4.0

    # Tournament of size 3 should select best from 3 random agents
    random.seed(42)
    selected = sample_pool.select_agent(strategy="tournament", tournament_size=3)

    # Verify it's one of the agents
    assert selected in sample_pool.agents


def test_select_unknown_strategy(sample_pool):
    """Test unknown selection strategy raises error."""
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        sample_pool.select_agent(strategy="invalid_strategy")


def test_select_from_empty_pool():
    """Test selection from empty pool raises error."""
    pool = AgentPool(role="intro", population_size=5)

    with pytest.raises(ValueError, match="Cannot select from empty pool"):
        pool.select_agent(strategy="best")


def test_get_top_n(sample_pool):
    """Test getting top N agents."""
    # Set fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i)]  # 0.0, 1.0, 2.0, 3.0, 4.0

    top_2 = sample_pool.get_top_n(n=2)

    assert len(top_2) == 2
    assert top_2[0].avg_fitness() == 4.0  # Best
    assert top_2[1].avg_fitness() == 3.0  # Second best


def test_get_top_n_exceeds_population(sample_pool):
    """Test getting top N when N > population size."""
    top_10 = sample_pool.get_top_n(n=10)

    # Should return all agents
    assert len(top_10) == 5


def test_get_random_lower_half(sample_pool):
    """Test getting random agent from lower half."""
    # Set fitness scores
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i)]  # 0.0, 1.0, 2.0, 3.0, 4.0

    # Run multiple times to test distribution
    random.seed(42)
    selections = [sample_pool.get_random_lower_half() for _ in range(50)]

    # All selections should be from lower half (fitness 0.0, 1.0, 2.0)
    lower_half_agents = sample_pool.agents[:2]  # First 2 agents

    for selected in selections:
        assert selected in lower_half_agents


def test_get_random_lower_half_small_pool():
    """Test getting random lower half with small pool."""
    pool = create_agent_pool("intro", population_size=3, min_size=3)

    # With 3 agents, lower_half should still work
    selected = pool.get_random_lower_half()
    assert selected in pool.agents


def test_get_all_stats_empty_pool():
    """Test getting stats from empty pool."""
    pool = AgentPool(role="intro", population_size=5)

    stats = pool.get_all_stats()

    assert stats["role"] == "intro"
    assert stats["population_size"] == 0
    assert stats["generation"] == 0
    assert stats["agents"] == []


def test_get_all_stats_populated_pool(sample_pool):
    """Test getting stats from populated pool."""
    # Set fitness scores and task counts
    for i, agent in enumerate(sample_pool.agents):
        agent.fitness_history = [float(i + 5)]  # 5.0, 6.0, 7.0, 8.0, 9.0
        agent.task_count = i + 1

    stats = sample_pool.get_all_stats()

    assert stats["role"] == "intro"
    assert stats["population_size"] == 5
    assert stats["avg_fitness"] == 7.0  # Mean of 5,6,7,8,9
    assert stats["best_fitness"] == 9.0
    assert stats["worst_fitness"] == 5.0
    assert len(stats["agents"]) == 5

    # Check agent stats structure
    agent_stat = stats["agents"][0]
    assert "agent_id" in agent_stat
    assert "avg_fitness" in agent_stat
    assert "task_count" in agent_stat


def test_get_all_stats_no_tasks(sample_pool):
    """Test stats when no agents have completed tasks."""
    # All agents have empty fitness history
    for agent in sample_pool.agents:
        agent.fitness_history = []
        agent.task_count = 0

    stats = sample_pool.get_all_stats()

    assert stats["avg_fitness"] == 0.0
    assert stats["best_fitness"] == 0.0
    assert stats["worst_fitness"] == 0.0


def test_unique_agent_ids(sample_pool):
    """Test all agents have unique IDs."""
    agent_ids = [a.agent_id for a in sample_pool.agents]

    assert len(agent_ids) == len(set(agent_ids))  # No duplicates


def test_unique_memory_collections(sample_pool):
    """Test all agents have unique memory collections."""
    collection_names = [a.memory.collection_name for a in sample_pool.agents]

    assert len(collection_names) == len(set(collection_names))  # No duplicates
