# M2 Compaction Development
**Branch**: `m2-compaction`
**Worktree**: `/home/justin/Documents/dev/workspaces/lean-compaction`
**Base**: develop (58991bf)

---

## Objective

Implement reasoning pattern compaction strategies to forget unsuccessful patterns before inheritance.

**Goal**: Reduce 100 patterns → 20-30 best patterns for inheritance

**Architecture**: Pure Python utility classes (strategy pattern), NOT LangGraph nodes

---

## LangGraph Integration

**This feature is a pure Python utility:**
- CompactionStrategy and implementations are **Python classes**
- Called BY LangGraph nodes (specifically `_evolve_node` and `ReproductionStrategy`)
- NOT LangGraph nodes themselves

**Usage in workflow:**
```python
# In pipeline_v2.py _evolve_node (LangGraph node)
pool.evolve_generation(
    reproduction_strategy=self.reproduction_strategy,
    shared_rag=self.shared_rag
)

# Inside ReproductionStrategy.reproduce() (Python utility)
inherited = compaction_strategy.compact(combined_patterns, max_size=100)
```

See: `/home/justin/Documents/dev/workspaces/hvas-mini/docs/brainstorming/2025-10-20-M2-LANGGRAPH-ARCHITECTURE.md`

---

## Files to Create

1. `src/lean/compaction.py` - Core compaction strategies
2. `tests/test_compaction.py` - Unit tests
3. `examples/compaction_demo.py` - Demonstration

---

## Implementation Checklist

### Phase 1: Base Class
- [ ] Create `CompactionStrategy` abstract base class
- [ ] Define `compact()` interface
- [ ] Add metadata tracking (what was kept/removed)

### Phase 2: Score-Based Strategy
- [ ] Implement `ScoreBasedCompaction`
- [ ] Keep top N by score
- [ ] Test with sample patterns

### Phase 3: Frequency-Based Strategy
- [ ] Implement `FrequencyBasedCompaction`
- [ ] Keep most-retrieved patterns
- [ ] Weight by retrieval_count × score

### Phase 4: Diversity-Preserving Strategy
- [ ] Implement `DiversityPreservingCompaction`
- [ ] Cluster patterns by embedding
- [ ] Keep best from each cluster

### Phase 5: Hybrid Strategy
- [ ] Implement `HybridCompaction`
- [ ] Combine score + frequency + diversity
- [ ] Configurable weights

### Phase 6: Testing
- [ ] Unit tests for each strategy
- [ ] Test edge cases (0 patterns, 1 pattern)
- [ ] Benchmark performance (1000+ patterns)
- [ ] Validate diversity preservation

### Phase 7: Documentation
- [ ] Add docstrings
- [ ] Create demo script
- [ ] Document configuration options

---

## Quick Start

```bash
# You are in: /home/justin/Documents/dev/workspaces/lean-compaction
cd /home/justin/Documents/dev/workspaces/lean-compaction

# Create implementation file
touch src/lean/compaction.py

# Start development
# Implement CompactionStrategy base class first
```

---

## Example Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class CompactionStrategy(ABC):
    """Base class for reasoning pattern compaction."""

    @abstractmethod
    def compact(
        self,
        patterns: List[Dict],
        max_size: int,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Compact patterns to max_size.

        Args:
            patterns: List of reasoning pattern dicts
            max_size: Maximum number of patterns to keep
            metadata: Optional context (domain, generation, etc.)

        Returns:
            Compacted list of patterns (length <= max_size)
        """
        pass

    def get_stats(self) -> Dict:
        """Get compaction statistics."""
        return {
            'strategy': self.__class__.__name__,
            'kept': 0,
            'removed': 0,
            'kept_percentage': 0.0
        }
```

---

## Testing Strategy

```python
# Test with sample patterns
sample_patterns = [
    {'reasoning': 'Pattern 1', 'score': 8.5, 'retrieval_count': 10},
    {'reasoning': 'Pattern 2', 'score': 7.0, 'retrieval_count': 2},
    {'reasoning': 'Pattern 3', 'score': 9.0, 'retrieval_count': 15},
    # ... more patterns
]

strategy = ScoreBasedCompaction()
compacted = strategy.compact(sample_patterns, max_size=2)

assert len(compacted) == 2
assert compacted[0]['score'] >= compacted[1]['score']
```

---

## Merge Criteria

Before merging to develop:
- ✅ All tests passing
- ✅ 4 strategies implemented (score, frequency, diversity, hybrid)
- ✅ Performance: <100ms for 1000 patterns
- ✅ Documentation complete
- ✅ Demo script works

---

## Dependencies

This feature has NO dependencies on other M2 features. Can develop independently.

---

## See Also

- Main plan: `/home/justin/Documents/dev/workspaces/hvas-mini/docs/planning/M2_EVOLUTION_PLAN.md`
- Base develop: `/home/justin/Documents/dev/workspaces/hvas-mini`
