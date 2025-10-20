# M2 Selection Development
**Branch**: `m2-selection`
**Worktree**: `/home/justin/Documents/dev/workspaces/lean-selection`
**Base**: develop (58991bf)

---

## Objective

Implement parent selection algorithms to choose best agents for reproduction.

**Goal**: Select high-fitness agents while maintaining population diversity

**Architecture**: Pure Python utility classes (strategy pattern), NOT LangGraph nodes

---

## LangGraph Integration

**This feature is a pure Python utility:**
- SelectionStrategy and implementations are **Python classes**
- Called BY AgentPool.evolve_generation() (which is called by LangGraph `_evolve_node`)
- NOT LangGraph nodes themselves

**Usage in workflow:**
```python
# In AgentPool.evolve_generation() (called by _evolve_node)
parents = self.selection_strategy.select_parents(
    pool=self,
    num_parents=self.max_size // 2
)
```

See: `/home/justin/Documents/dev/workspaces/hvas-mini/docs/brainstorming/2025-10-20-M2-LANGGRAPH-ARCHITECTURE.md`

---

## Files to Create

1. `src/lean/selection.py` - Selection algorithms
2. `tests/test_selection.py` - Unit tests
3. `examples/selection_demo.py` - Demonstration

---

## Implementation Checklist

### Phase 1: Base Class
- [ ] Create `SelectionStrategy` abstract base class
- [ ] Define `select_parents()` interface
- [ ] Add diversity measurement support

### Phase 2: Tournament Selection
- [ ] Implement `TournamentSelection`
- [ ] k agents compete, best wins
- [ ] Configurable tournament size

### Phase 3: Fitness-Proportionate Selection
- [ ] Implement `FitnessProportionateSelection`
- [ ] Roulette wheel by fitness
- [ ] Handle zero/negative fitness

### Phase 4: Rank-Based Selection
- [ ] Implement `RankBasedSelection`
- [ ] Select top N by fitness rank
- [ ] Support elitism (preserve best)

### Phase 5: Diversity-Aware Selection
- [ ] Implement `DiversityAwareSelection`
- [ ] Balance fitness and diversity
- [ ] Use embedding distance

### Phase 6: Testing
- [ ] Unit tests for each strategy
- [ ] Test with mock agent pools
- [ ] Validate diversity preservation
- [ ] Test edge cases (single agent, all same fitness)

### Phase 7: Documentation
- [ ] Add docstrings
- [ ] Create demo script
- [ ] Document configuration options

---

## Quick Start

```bash
# You are in: /home/justin/Documents/dev/workspaces/lean-selection
cd /home/justin/Documents/dev/workspaces/lean-selection

# Create implementation file
touch src/lean/selection.py

# Start development
# Implement SelectionStrategy base class first
```

---

## Example Interface

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict

class SelectionStrategy(ABC):
    """Base class for parent selection."""

    @abstractmethod
    def select_parents(
        self,
        pool: 'AgentPool',
        num_parents: int,
        metadata: Optional[Dict] = None
    ) -> List['BaseAgentV2']:
        """Select parents from pool.

        Args:
            pool: AgentPool to select from
            num_parents: Number of parents to select
            metadata: Optional context

        Returns:
            List of selected agents
        """
        pass
```

---

## Testing Strategy

```python
# Test with mock agent pool
class MockAgent:
    def __init__(self, fitness):
        self.fitness_scores = [fitness] * 5

    def avg_fitness(self):
        return sum(self.fitness_scores) / len(self.fitness_scores)

class MockPool:
    def __init__(self, agents):
        self.agents = agents

# Create mock agents
agents = [MockAgent(8.5), MockAgent(7.0), MockAgent(9.0), MockAgent(6.5)]
pool = MockPool(agents)

# Test selection
strategy = TournamentSelection(tournament_size=2)
parents = strategy.select_parents(pool, num_parents=2)

assert len(parents) == 2
# Higher fitness agents should be selected more often
```

---

## Merge Criteria

Before merging to develop:
- ✅ All tests passing
- ✅ 4 strategies implemented (tournament, proportionate, rank, diversity)
- ✅ Diversity validation
- ✅ Documentation complete
- ✅ Demo script works

---

## Dependencies

**Requires**: Mock `AgentPool` for testing (or wait for `m2-agent-pools` merge)

Can develop using simplified mock for now.

---

## See Also

- Main plan: `/home/justin/Documents/dev/workspaces/hvas-mini/docs/planning/M2_EVOLUTION_PLAN.md`
- Base develop: `/home/justin/Documents/dev/workspaces/hvas-mini`
