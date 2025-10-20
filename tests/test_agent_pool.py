"""Tests for AgentPool - integration of Phase 1 utilities."""
import pytest
import tempfile
import shutil

from lean.agent_pool import AgentPool, create_agent_pools
from lean.base_agent_v2 import create_agents_v2
from lean.shared_rag import SharedRAG
from lean.compaction import HybridCompaction, ScoreBasedCompaction
from lean.selection import TournamentSelection, RankBasedSelection
from lean.reproduction import SexualReproduction, AsexualReproduction


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    reasoning_dir = tempfile.mkdtemp()
    shared_rag_dir = tempfile.mkdtemp()

    yield reasoning_dir, shared_rag_dir

    # Cleanup
    shutil.rmtree(reasoning_dir, ignore_errors=True)
    shutil.rmtree(shared_rag_dir, ignore_errors=True)


@pytest.fixture
def shared_rag(temp_dirs):
    """Create shared RAG for testing."""
    _, shared_rag_dir = temp_dirs
    return SharedRAG(persist_directory=shared_rag_dir)


@pytest.fixture
def initial_agents(temp_dirs):
    """Create initial agents for testing."""
    reasoning_dir, shared_rag_dir = temp_dirs

    agents = create_agents_v2(
        reasoning_dir=reasoning_dir,
        shared_rag_dir=shared_rag_dir,  # Fixed: use shared_rag_dir
        agent_ids={
            'intro': 'intro_test_1',
            'body': 'body_test_1',
            'conclusion': 'conclusion_test_1'
        }
    )

    return agents


def test_agent_pool_initialization(initial_agents):
    """Test basic pool initialization."""
    pool = AgentPool(
        role='intro',
        initial_agents=[initial_agents['intro']],
        max_size=5
    )

    assert pool.role == 'intro'
    assert len(pool.agents) == 1
    assert pool.max_size == 5
    assert pool.generation == 0
    assert len(pool.history) == 1  # Initial stats recorded


def test_agent_pool_select_agent(initial_agents):
    """Test agent selection from pool."""
    pool = AgentPool(
        role='intro',
        initial_agents=[initial_agents['intro']],
        max_size=5
    )

    # With single agent
    selected = pool.select_agent()
    assert selected == initial_agents['intro']

    # Test all selection strategies
    for strategy in ['fitness_proportionate', 'random', 'best']:
        selected = pool.select_agent(strategy=strategy)
        assert selected in pool.agents


def test_agent_pool_evolve_generation(initial_agents, shared_rag):
    """Test evolution cycle."""
    # Give agent some fitness scores first
    agent = initial_agents['intro']
    for _ in range(5):
        agent.record_fitness(score=7.5, domain='test')

    pool = AgentPool(
        role='intro',
        initial_agents=[agent],
        max_size=3,
        selection_strategy=TournamentSelection(tournament_size=2),
        compaction_strategy=HybridCompaction()
    )

    # Evolve
    reproduction_strategy = SexualReproduction(mutation_rate=0.1)
    pool.evolve_generation(
        reproduction_strategy=reproduction_strategy,
        shared_rag=shared_rag
    )

    # Verify evolution happened
    assert pool.generation == 1
    assert len(pool.agents) == 3  # Population size
    assert len(pool.history) == 2  # Initial + after evolution


def test_agent_pool_fitness_tracking(initial_agents, shared_rag):
    """Test fitness tracking across generations."""
    agent = initial_agents['intro']

    # Give some fitness
    for i in range(5):
        agent.record_fitness(score=6.0 + i * 0.5, domain='test')

    pool = AgentPool(
        role='intro',
        initial_agents=[agent],
        max_size=3
    )

    initial_fitness = pool.avg_fitness()
    assert initial_fitness > 0

    # Evolve multiple generations
    reproduction_strategy = SexualReproduction(mutation_rate=0.05)

    for _ in range(2):
        # Give offspring some fitness before next evolution
        for offspring in pool.agents:
            offspring.record_fitness(score=7.0, domain='test')

        pool.evolve_generation(
            reproduction_strategy=reproduction_strategy,
            shared_rag=shared_rag
        )

    assert pool.generation == 2
    assert len(pool.history) == 3  # Initial + 2 evolutions


def test_agent_pool_get_stats(initial_agents):
    """Test statistics retrieval."""
    agent = initial_agents['intro']

    for i in range(5):
        agent.record_fitness(score=7.0 + i, domain='test')

    pool = AgentPool(
        role='intro',
        initial_agents=[agent],
        max_size=3
    )

    stats = pool.get_stats()

    assert stats['role'] == 'intro'
    assert stats['generation'] == 0
    assert stats['size'] == 1
    assert stats['avg_fitness'] > 0
    assert 'fitness_range' in stats
    assert 'diversity' in stats
    assert 'agents' in stats
    assert len(stats['agents']) == 1


def test_agent_pool_get_top_n(initial_agents, shared_rag):
    """Test getting top N agents."""
    agent = initial_agents['intro']

    # Give fitness
    for i in range(5):
        agent.record_fitness(score=7.0, domain='test')

    pool = AgentPool(
        role='intro',
        initial_agents=[agent],
        max_size=5
    )

    # Evolve to get multiple agents
    pool.evolve_generation(
        reproduction_strategy=SexualReproduction(),
        shared_rag=shared_rag
    )

    # Give different fitness to agents
    for i, agent in enumerate(pool.agents):
        agent.record_fitness(score=5.0 + i, domain='test')

    top_2 = pool.get_top_n(2)
    assert len(top_2) == 2
    assert top_2[0].avg_fitness() >= top_2[1].avg_fitness()


def test_agent_pool_history(initial_agents, shared_rag):
    """Test evolution history tracking."""
    agent = initial_agents['intro']

    for i in range(5):
        agent.record_fitness(score=7.0, domain='test')

    pool = AgentPool(
        role='intro',
        initial_agents=[agent],
        max_size=3
    )

    # Evolve 3 generations
    for _ in range(3):
        # Give fitness to offspring
        for offspring in pool.agents:
            offspring.record_fitness(score=7.5, domain='test')

        pool.evolve_generation(
            reproduction_strategy=SexualReproduction(),
            shared_rag=shared_rag
        )

    history = pool.get_history()
    assert len(history) == 4  # Initial + 3 evolutions

    # Verify history structure
    for record in history:
        assert 'generation' in record
        assert 'avg_fitness' in record
        assert 'diversity' in record
        assert 'size' in record


def test_agent_pool_with_different_strategies(initial_agents, shared_rag):
    """Test pool with different strategy combinations."""
    agent = initial_agents['intro']

    for i in range(5):
        agent.record_fitness(score=8.0, domain='test')

    # Test with different strategies
    strategies = [
        (TournamentSelection(tournament_size=2), ScoreBasedCompaction(), AsexualReproduction()),
        (RankBasedSelection(elitism_count=1), HybridCompaction(), SexualReproduction())
    ]

    for selection, compaction, reproduction in strategies:
        pool = AgentPool(
            role='intro',
            initial_agents=[agent],
            max_size=3,
            selection_strategy=selection,
            compaction_strategy=compaction
        )

        pool.evolve_generation(
            reproduction_strategy=reproduction,
            shared_rag=shared_rag
        )

        assert pool.generation == 1
        assert len(pool.agents) == 3


def test_create_agent_pools_factory(initial_agents):
    """Test factory function for creating pools."""
    pools = create_agent_pools(
        agents=initial_agents,
        pool_size=5,
        selection_strategy=TournamentSelection(tournament_size=3),
        compaction_strategy=HybridCompaction()
    )

    assert len(pools) == 3  # intro, body, conclusion
    assert 'intro' in pools
    assert 'body' in pools
    assert 'conclusion' in pools

    for role, pool in pools.items():
        assert pool.role == role
        assert len(pool.agents) == 1  # Started with 1 agent
        assert pool.max_size == 5


def test_agent_pool_empty_agents():
    """Test pool with no agents."""
    pool = AgentPool(
        role='intro',
        initial_agents=[],
        max_size=5
    )

    assert pool.avg_fitness() == 0.0
    assert pool.measure_diversity() == 0.0

    stats = pool.get_stats()
    assert stats['size'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
