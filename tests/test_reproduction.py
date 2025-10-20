"""Tests for reproduction strategies - with mocks for unmerged dependencies."""
import pytest
from lean.reproduction import AsexualReproduction, SexualReproduction, create_reproduction_strategy

class MockCompaction:
    def compact(self, patterns, max_size):
        return patterns[:max_size]  # Simple truncate

class MockMemory:
    def __init__(self, patterns):
        self.patterns = patterns
    def get_all_reasoning(self):
        return self.patterns

class MockAgent:
    def __init__(self, role, patterns, shared_rag=None):
        self.role = role
        self.reasoning_memory = MockMemory(patterns)
        self.shared_rag = shared_rag or MockSharedRAG()

class MockSharedRAG:
    pass

@pytest.fixture
def mock_parent1():
    patterns = [{'reasoning': f'P1 pattern {i}', 'score': 8.0 + i*0.1} for i in range(10)]
    return MockAgent('intro', patterns)

@pytest.fixture
def mock_parent2():
    patterns = [{'reasoning': f'P2 pattern {i}', 'score': 7.0 + i*0.1} for i in range(10)]
    return MockAgent('intro', patterns)

def test_asexual_reproduction(mock_parent1):
    """Test asexual reproduction creates offspring."""
    strategy = AsexualReproduction(mutation_rate=0.0)
    compaction = MockCompaction()
    
    # Mock the offspring creation to avoid import issues
    import lean.reproduction
    original_create = AsexualReproduction._create_offspring
    
    def mock_create(self, role, generation, inherited_patterns, shared_rag):
        # Return a mock offspring
        return MockAgent(role, inherited_patterns, shared_rag)
    
    AsexualReproduction._create_offspring = mock_create
    
    try:
        offspring = strategy.reproduce(mock_parent1, None, compaction, generation=2)
        assert offspring.role == 'intro'
        assert len(offspring.reasoning_memory.get_all_reasoning()) <= 10
    finally:
        AsexualReproduction._create_offspring = original_create

def test_sexual_reproduction(mock_parent1, mock_parent2):
    """Test sexual reproduction combines parents."""
    strategy = SexualReproduction(mutation_rate=0.0, crossover_rate=0.5)
    compaction = MockCompaction()
    
    import lean.reproduction
    def mock_create(self, role, generation, inherited_patterns, shared_rag):
        return MockAgent(role, inherited_patterns, shared_rag)
    
    SexualReproduction._create_offspring = mock_create
    
    try:
        offspring = strategy.reproduce(mock_parent1, mock_parent2, compaction, generation=2)
        assert offspring.role == 'intro'
        # Should have patterns from both parents
        patterns = offspring.reasoning_memory.get_all_reasoning()
        assert len(patterns) > 0
    finally:
        pass

def test_factory_creation():
    s1 = create_reproduction_strategy('asexual', mutation_rate=0.1)
    assert isinstance(s1, AsexualReproduction)
    assert s1.mutation_rate == 0.1
    
    s2 = create_reproduction_strategy('sexual', mutation_rate=0.2, crossover_rate=0.5)
    assert isinstance(s2, SexualReproduction)
    assert s2.mutation_rate == 0.2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
