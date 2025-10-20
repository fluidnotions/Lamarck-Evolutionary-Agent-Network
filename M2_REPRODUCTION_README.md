# M2 Reproduction Development
**Branch**: `m2-reproduction`
**Worktree**: `/home/justin/Documents/dev/workspaces/lean-reproduction`
**Base**: develop (58991bf)

---

## Objective

Implement agent reproduction with reasoning pattern inheritance.

**Goal**: Create offspring agents that inherit compacted reasoning patterns from parent(s)

**Architecture**: Pure Python utility classes (strategy pattern), NOT LangGraph nodes

---

## LangGraph Integration

**This feature is a pure Python utility:**
- ReproductionStrategy and implementations are **Python classes**
- Called BY AgentPool.evolve_generation() (which is called by LangGraph `_evolve_node`)
- NOT LangGraph nodes themselves

**Usage in workflow:**
```python
# In AgentPool.evolve_generation() (called by _evolve_node)
child = reproduction_strategy.reproduce(
    parent1=parent1,
    parent2=parent2,
    compaction_strategy=self.compaction_strategy,
    generation=self.generation + 1
)
```

See: `/home/justin/Documents/dev/workspaces/hvas-mini/docs/brainstorming/2025-10-20-M2-LANGGRAPH-ARCHITECTURE.md`

---

## Files to Create

1. `src/lean/reproduction.py` - Reproduction strategies
2. `tests/test_reproduction.py` - Unit tests
3. `examples/reproduction_demo.py` - Demonstration

---

## Implementation Checklist

### Phase 1: Base Class
- [ ] Create `ReproductionStrategy` abstract base class
- [ ] Define `reproduce()` interface
- [ ] Add lineage tracking

### Phase 2: Asexual Reproduction
- [ ] Implement `AsexualReproduction`
- [ ] Single parent → offspring
- [ ] Optional mutation

### Phase 3: Sexual Reproduction
- [ ] Implement `SexualReproduction`
- [ ] Two parents → offspring
- [ ] Crossover (mix patterns)
- [ ] Optional mutation

### Phase 4: Factory Function
- [ ] Implement `create_offspring()`
- [ ] Integrate compaction
- [ ] Create child agent with inherited patterns
- [ ] Track parent → child lineage

### Phase 5: Mutation Support
- [ ] Add noise to inherited patterns
- [ ] Configurable mutation rate
- [ ] Preserve pattern structure

### Phase 6: Testing
- [ ] Unit tests for each strategy
- [ ] Test with real BaseAgentV2 agents
- [ ] Validate inheritance works
- [ ] Test lineage tracking

### Phase 7: Documentation
- [ ] Add docstrings
- [ ] Create demo script
- [ ] Document configuration options

---

## Quick Start

```bash
# You are in: /home/justin/Documents/dev/workspaces/lean-reproduction
cd /home/justin/Documents/dev/workspaces/lean-reproduction

# Create implementation file
touch src/lean/reproduction.py

# Start development
# Implement ReproductionStrategy base class first
```

---

## Example Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict
from lean.base_agent_v2 import BaseAgentV2
from lean.compaction import CompactionStrategy

class ReproductionStrategy(ABC):
    """Base class for agent reproduction."""

    @abstractmethod
    def reproduce(
        self,
        parent1: BaseAgentV2,
        parent2: Optional[BaseAgentV2],
        compaction_strategy: CompactionStrategy,
        generation: int
    ) -> BaseAgentV2:
        """Create offspring from parent(s).

        Args:
            parent1: First parent agent
            parent2: Second parent (None for asexual)
            compaction_strategy: How to compact inherited patterns
            generation: Generation number for offspring

        Returns:
            New agent with inherited reasoning patterns
        """
        pass
```

---

## Key Implementation: Factory Function

```python
def create_offspring(
    parent1: BaseAgentV2,
    parent2: Optional[BaseAgentV2],
    role: str,
    generation: int,
    compaction_strategy: CompactionStrategy,
    shared_rag: SharedRAG
) -> BaseAgentV2:
    """Create offspring agent with inherited patterns."""

    # 1. Get parent reasoning patterns
    parent1_patterns = parent1.reasoning_memory.get_all_reasoning()
    parent2_patterns = parent2.reasoning_memory.get_all_reasoning() if parent2 else []

    # 2. Compact (forget unsuccessful patterns)
    combined = parent1_patterns + parent2_patterns
    inherited = compaction_strategy.compact(
        patterns=combined,
        max_size=int(os.getenv('INHERITED_REASONING_SIZE', '100'))
    )

    # 3. Create child agent with inherited patterns
    child_id = f"{role}_gen{generation}_child{uuid.uuid4().hex[:6]}"
    child_memory = ReasoningMemory(
        collection_name=generate_reasoning_collection_name(role, child_id),
        inherited_reasoning=inherited  # KEY: Pass inherited patterns
    )

    # 4. Create agent
    if role == 'intro':
        child = IntroAgentV2(
            role=role,
            agent_id=child_id,
            reasoning_memory=child_memory,
            shared_rag=shared_rag
        )
    # ... other roles

    return child
```

---

## Testing Strategy

```python
# Test with real agents
from lean.base_agent_v2 import IntroAgentV2
from lean.reasoning_memory import ReasoningMemory
from lean.shared_rag import SharedRAG
from lean.compaction import ScoreBasedCompaction

# Create parent with patterns
parent_memory = ReasoningMemory(...)
parent_memory.store_reasoning_pattern(
    reasoning="Pattern 1",
    score=8.5,
    situation="test"
)
parent = IntroAgentV2(
    role='intro',
    agent_id='parent',
    reasoning_memory=parent_memory,
    shared_rag=shared_rag
)

# Reproduce
strategy = AsexualReproduction()
compaction = ScoreBasedCompaction()
child = strategy.reproduce(parent, None, compaction, generation=2)

# Verify inheritance
child_patterns = child.reasoning_memory.get_all_reasoning()
assert len(child_patterns) > 0
assert child_patterns[0]['reasoning'] == "Pattern 1"
```

---

## Merge Criteria

Before merging to develop:
- ✅ All tests passing
- ✅ 2 strategies implemented (asexual, sexual)
- ✅ Factory function working
- ✅ Inheritance validated
- ✅ Lineage tracking
- ✅ Documentation complete
- ✅ Demo script works

---

## Dependencies

**Requires**: `m2-compaction` merged (CompactionStrategy)

Can develop using mock compaction for now:
```python
class MockCompaction:
    def compact(self, patterns, max_size):
        return patterns[:max_size]  # Simple truncate
```

---

## See Also

- Main plan: `/home/justin/Documents/dev/workspaces/hvas-mini/docs/planning/M2_EVOLUTION_PLAN.md`
- Base develop: `/home/justin/Documents/dev/workspaces/hvas-mini`
- Compaction: `/home/justin/Documents/dev/workspaces/lean-compaction`
