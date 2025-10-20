"""Tests for selection strategies - focused version."""
import pytest
import numpy as np
from lean.selection import (
    TournamentSelection, FitnessProportionateSelection,
    RankBasedSelection, DiversityAwareSelection, create_selection_strategy
)

class MockAgent:
    def __init__(self, fitness, agent_id=None):
        self.fitness_scores = [fitness] * 5
        self.agent_id = agent_id or id(self)
        self.reasoning_memory = MockReasoningMemory()
    
    def avg_fitness(self):
        return sum(self.fitness_scores) / len(self.fitness_scores)

class MockReasoningMemory:
    def get_all_reasoning(self):
        return [{'reasoning': 'test', 'embedding': np.random.rand(384).tolist()}]

class MockPool:
    def __init__(self, agents):
        self.agents = agents

@pytest.fixture
def sample_pool():
    return MockPool([
        MockAgent(9.0, 'best'),
        MockAgent(7.0, 'good'),
        MockAgent(5.0, 'ok'),
        MockAgent(3.0, 'poor')
    ])

def test_tournament_selection_basic(sample_pool):
    strategy = TournamentSelection(tournament_size=2)
    parents = strategy.select_parents(sample_pool, num_parents=4)
    assert len(parents) == 4
    avg_fitness = np.mean([p.avg_fitness() for p in parents])
    assert avg_fitness >= 5.0  # Should favor higher fitness

def test_fitness_proportionate_selection(sample_pool):
    strategy = FitnessProportionateSelection()
    parents = strategy.select_parents(sample_pool, num_parents=4)
    assert len(parents) == 4

def test_rank_based_selection_elitism(sample_pool):
    strategy = RankBasedSelection(elitism_count=1)
    parents = strategy.select_parents(sample_pool, num_parents=4)
    assert len(parents) == 4
    assert parents[0].agent_id == 'best'  # Elitism ensures best is selected

def test_diversity_aware_selection(sample_pool):
    strategy = DiversityAwareSelection(diversity_weight=0.3)
    parents = strategy.select_parents(sample_pool, num_parents=4)
    assert len(parents) == 4

def test_factory_creation():
    s1 = create_selection_strategy('tournament', tournament_size=3)
    assert isinstance(s1, TournamentSelection)
    
    s2 = create_selection_strategy('proportionate')
    assert isinstance(s2, FitnessProportionateSelection)
    
    s3 = create_selection_strategy('rank', elitism_count=2)
    assert isinstance(s3, RankBasedSelection)
    
    s4 = create_selection_strategy('diversity')
    assert isinstance(s4, DiversityAwareSelection)

def test_empty_pool():
    strategy = TournamentSelection()
    parents = strategy.select_parents(MockPool([]), num_parents=5)
    assert parents == []

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
