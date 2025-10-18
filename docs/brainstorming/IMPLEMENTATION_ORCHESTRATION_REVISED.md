# HVAS Mini Implementation Orchestration Plan (REVISED)

## CRITICAL ARCHITECTURAL CHANGE

**Based on insights from ANOTHER_ASPECT.MD:**

### What Changed

**OLD APPROACH (Wrong)**:
- Mutate prompts (add/remove/modify instructions)
- Each agent starts with empty memories
- Evolution happens through prompt optimization

**NEW APPROACH (Correct)**:
- **Prompts stay stable** (minimal variation for diversity only)
- **Memories ARE inherited** (compacted from parents)
- **Evolution happens through WHICH memories get passed on**

### Why This is Better

1. **Prompts are the PROBLEM, not the solution** - We use RAG because prompts can't hold nuance
2. **Evolution should work on what matters** - The accumulated knowledge, not the instructions
3. **Lamarckian is better for AI** - Successful learning gets passed on directly
4. **Knowledge accumulates** - Each generation builds on the last

---

## Overview

This document orchestrates the implementation of the evolutionary multi-agent system with **memory inheritance** as the primary evolutionary mechanism.

**Implementation Philosophy**:
- Break into milestones
- Create feature branches for parallel work
- Use worktrees for concurrent development
- AGENT_TASK.md files guide each branch
- Sequential dependencies respected

---

## Milestone Structure

```
M1: Core System (foundational)
├─ M2: Memory Inheritance (depends on M1) ← CHANGED
│   └─ M3: Strategies (depends on M1, M2)
│       └─ M4: Experimentation (depends on M3)
└─ M5: Enhancement (parallel with M4)
```

---

## Milestone 1: Core System

**Goal**: Foundation for multi-agent populations with basic selection and context distribution

**Duration**: 3-4 days

### Feature Branches (Parallel Work Possible)

#### Branch 1.1: `agent-pool-infrastructure`
**Status**: ✅ Worktree created, AGENT_TASK.md complete
**Can start**: Immediately
**Depends on**: None

---

#### Branch 1.2: `individual-memory-collections`
**Can start**: Immediately (parallel with 1.1)
**Depends on**: None

**CHANGES**: Add support for **inherited memories**

**Tasks**:
- Modify `MemoryManager` to remove score threshold
- Support per-agent collections (`intro_agent_1_memories`)
- Implement weighted retrieval (`similarity × (score/10)`)
- Store all experiences with metadata
- **NEW**: Load inherited memories on agent initialization
- **NEW**: Separate personal vs inherited memories

**Files**:
- `src/hvas_mini/memory.py` (MODIFY)
- `tests/test_memory.py` (MODIFY)

**Key Implementation Change**:

```python
class MemoryManager:
    def __init__(self, collection_name, persist_directory, inherited_memories=None):
        """Initialize with optional inherited memories.

        Args:
            inherited_memories: List of memory dicts from parents (compacted)
        """
        self.collection = chromadb.get_collection(collection_name)
        self.inherited_count = 0

        # Load inherited memories into collection
        if inherited_memories:
            self._load_inherited_memories(inherited_memories)
            self.inherited_count = len(inherited_memories)

    def _load_inherited_memories(self, memories):
        """Load parent memories into agent's collection."""
        for mem in memories:
            self.collection.add(
                embeddings=[mem['embedding']],
                documents=[mem['content']],
                metadatas=[{
                    **mem['metadata'],
                    'inherited': True,  # Mark as inherited
                    'parent_ids': mem.get('parent_ids', [])
                }],
                ids=[f"inherited_{mem['id']}"]
            )

    def get_stats(self):
        """Return stats including inherited vs personal memories."""
        return {
            "total_memories": self.count(),
            "inherited_memories": self.inherited_count,
            "personal_memories": self.count() - self.inherited_count,
            ...
        }
```

**AGENT_TASK.md location**: `docs/feature-plans/individual-memory-collections/AGENT_TASK.md` (UPDATE)

---

#### Branch 1.3: `fitness-tracking`
**Can start**: Immediately (parallel with 1.1, 1.2)
**Depends on**: None

**NO CHANGES** - fitness tracking remains the same

---

#### Branch 1.4: `context-distribution`
**Can start**: After 1.1 completes (needs AgentPool)
**Depends on**: Branch 1.1

**NO CHANGES** - context distribution remains the same

---

#### Branch 1.5: `pipeline-integration`
**Can start**: After 1.1, 1.2, 1.3, 1.4 complete
**Depends on**: All M1 branches

**CHANGES**: Remove parameter evolution entirely (temperature stays fixed)

**Tasks**:
- Update `HVASMiniPipeline` to use agent pools
- Integrate context distribution
- Update LangGraph workflow
- **REMOVE**: Parameter evolution
- **REMOVE**: Temperature tweaking
- **REMOVE**: Trust manager

---

## Milestone 2: Memory Inheritance (REDESIGNED)

**Goal**: Enable Lamarckian evolution through memory inheritance

**Duration**: 2-3 days

**Starts**: After M1 complete

### Feature Branches (Parallel Work Possible)

#### Branch 2.1: `memory-compaction` ⭐ NEW
**Can start**: Immediately after M1
**Depends on**: M1 complete

**Tasks**:
- Implement memory compaction strategies
- Score-weighted selection
- Diversity preservation (clustering)
- Frequency-based (retrieval count × usefulness)
- LLM distillation (optional)

**Files**:
- `src/hvas_mini/evolution/memory_compaction.py` (NEW)
- `tests/test_memory_compaction.py` (NEW)

**Implementation**:

```python
class MemoryCompactor:
    """Compacts parent memories for inheritance."""

    def compact_memories(self,
                        parent1_memories: List[Dict],
                        parent2_memories: List[Dict],
                        target_size: int = 100,
                        strategy: str = "balanced") -> List[Dict]:
        """Merge and distill parent memories.

        Args:
            parent1_memories: All memories from parent 1
            parent2_memories: All memories from parent 2
            target_size: Maximum memories to inherit
            strategy: Compaction approach (score, diversity, frequency, balanced)

        Returns:
            Compacted list of memories to inherit
        """
        combined = parent1_memories + parent2_memories

        if strategy == "score":
            return self._compact_by_score(combined, target_size)
        elif strategy == "diversity":
            return self._compact_by_diversity(combined, target_size)
        elif strategy == "frequency":
            return self._compact_by_frequency(combined, target_size)
        elif strategy == "balanced":
            return self._compact_balanced(combined, target_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _compact_by_score(self, memories, target_size):
        """Keep highest-scoring memories."""
        sorted_mem = sorted(memories, key=lambda m: m['metadata']['score'], reverse=True)
        return sorted_mem[:target_size]

    def _compact_by_diversity(self, memories, target_size):
        """Keep diverse memories covering different topics."""
        # Cluster by embedding similarity
        clusters = self._cluster_memories(memories)

        # Take best representative from each cluster
        selected = []
        for cluster in clusters:
            best = max(cluster, key=lambda m: m['metadata']['score'])
            selected.append(best)
            if len(selected) >= target_size:
                break

        return selected

    def _compact_by_frequency(self, memories, target_size):
        """Keep memories that were frequently useful."""
        # Score = retrieval_count × score × recency
        scored = []
        for mem in memories:
            usefulness = (
                mem['metadata'].get('retrieval_count', 0) *
                mem['metadata']['score'] *
                self._recency_factor(mem['metadata']['timestamp'])
            )
            scored.append((usefulness, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:target_size]]

    def _compact_balanced(self, memories, target_size):
        """Balanced: 50% score, 30% diversity, 20% frequency."""
        by_score = self._compact_by_score(memories, int(target_size * 0.5))
        by_diversity = self._compact_by_diversity(memories, int(target_size * 0.3))
        by_frequency = self._compact_by_frequency(memories, int(target_size * 0.2))

        # Merge and deduplicate
        seen = set()
        result = []
        for mem in by_score + by_diversity + by_frequency:
            mem_id = mem['id']
            if mem_id not in seen:
                seen.add(mem_id)
                result.append(mem)

        return result[:target_size]
```

**AGENT_TASK.md location**: `docs/feature-plans/memory-compaction/AGENT_TASK.md` (CREATE)

---

#### Branch 2.2: `memory-inheritance-reproduction` ⭐ REDESIGNED
**Can start**: Parallel with 2.1
**Depends on**: M1 complete

**Tasks**:
- Implement reproduction with memory inheritance (NOT prompt mutation)
- Parent selection for reproduction
- Offspring initialization with inherited memories
- Lineage tracking
- **MINIMAL** prompt variation (optional diversity)

**Files**:
- `src/hvas_mini/evolution/reproduction.py` (NEW)
- `tests/test_reproduction.py` (NEW)

**Implementation**:

```python
class ReproductionManager:
    """Manages agent reproduction with memory inheritance."""

    def __init__(self, memory_compactor: MemoryCompactor):
        self.compactor = memory_compactor

    def reproduce(self,
                  parent1: BaseAgent,
                  parent2: BaseAgent,
                  prompt_variation: bool = False) -> BaseAgent:
        """Create offspring inheriting parents' compacted memories.

        Args:
            parent1: First parent agent
            parent2: Second parent agent
            prompt_variation: If True, slight prompt variation (5% chance)

        Returns:
            Offspring agent with inherited memories
        """
        # 1. Compact parent memories
        parent1_mems = parent1.memory.get_all_memories()
        parent2_mems = parent2.memory.get_all_memories()

        inherited_memories = self.compactor.compact_memories(
            parent1_mems,
            parent2_mems,
            target_size=100,
            strategy="balanced"
        )

        # 2. Determine child genome (mostly stable)
        if prompt_variation and random.random() < 0.05:
            # Rare variation: simple synonym replacement
            child_genome = self._minimal_variation(parent1.genome)
        else:
            # Default: inherit parent1's genome unchanged
            child_genome = parent1.genome

        # 3. Create offspring
        child_id = f"{parent1.role}_agent_{self._next_id()}"

        memory_manager = MemoryManager(
            collection_name=f"{child_id}_memories",
            persist_directory=self.persist_directory,
            inherited_memories=inherited_memories  # ← KEY: Pass inherited memories
        )

        child = parent1.__class__(
            role=parent1.role,
            memory_manager=memory_manager,
            agent_id=child_id,
            genome=child_genome
        )

        # 4. Track lineage
        child.lineage = {
            "parent1": parent1.agent_id,
            "parent2": parent2.agent_id,
            "generation": max(parent1.generation, parent2.generation) + 1,
            "inherited_memory_count": len(inherited_memories)
        }

        return child

    def _minimal_variation(self, genome: str) -> str:
        """Very minimal prompt variation (synonym replacement only)."""
        synonyms = {
            "engaging": ["compelling", "captivating"],
            "clear": ["straightforward", "unambiguous"],
            "concise": ["brief", "succinct"]
        }

        for word, alternatives in synonyms.items():
            if word in genome and random.random() < 0.5:
                replacement = random.choice(alternatives)
                genome = genome.replace(word, replacement, 1)
                break  # Only one replacement

        return genome
```

**AGENT_TASK.md location**: `docs/feature-plans/memory-inheritance-reproduction/AGENT_TASK.md` (CREATE)

---

#### Branch 2.3: `population-management`
**Can start**: After 2.1, 2.2 complete
**Depends on**: Branches 2.1, 2.2

**CHANGES**: Update to use memory inheritance instead of genome mutation

**Tasks**:
- Create `EvolutionManager` class
- Add agent triggers (population < min, diversity < threshold, stagnation)
- Remove agent triggers (fitness < 6.0, task_count ≥ 20)
- Evolution cycle (every 10 generations)
- **USE**: Memory inheritance for reproduction
- **REMOVE**: Genome mutation logic

---

#### Branch 2.4: `evolution-pipeline-integration`
**Can start**: After 2.3 completes
**Depends on**: All M2 branches

**NO MAJOR CHANGES** - just integrate the new reproduction system

---

### Milestone 2 Completion Criteria

- [ ] Memory compaction working (3+ strategies)
- [ ] Reproduction creates offspring with inherited memories
- [ ] Offspring start with 50-100 inherited memories
- [ ] Population adds/removes agents correctly
- [ ] Evolution runs automatically every 10 gens
- [ ] **Prompts remain stable** (minimal variation only)
- [ ] All M2 tests passing
- [ ] Can run 30 generations with memory inheritance

---

## Milestone 3: Strategies (UPDATED)

**Changes**: Strategies now focus on **memory compaction approach**, not genome mutation

### Strategy Component Updates

```python
class EvolutionaryStrategy:
    selection: SelectionMethod  # Unchanged
    context: ContextDistribution  # Unchanged
    memory_compaction: CompactionStrategy  # ← NEW: Which compaction strategy
    evolution: PopulationDynamics  # Unchanged
    memory_retrieval: RetrievalWeighting  # Unchanged
```

### Three Baseline Strategies (Updated)

#### Strategy A: Conservative Evolution
- **Selection**: ε-greedy (90/10)
- **Memory Compaction**: Score-weighted (keep highest-scoring) ← CHANGED
- **Evolution**: Slow (every 20 generations)
- **Memory Retrieval**: Heavy quality weighting
- **Hypothesis**: Best memories propagate, quality over diversity

#### Strategy B: Aggressive Evolution
- **Selection**: Tournament (top 3)
- **Memory Compaction**: Frequency-based (keep most-used) ← CHANGED
- **Evolution**: Fast (every 5 generations)
- **Memory Retrieval**: Pure similarity
- **Hypothesis**: Useful patterns propagate, rapid iteration

#### Strategy C: Balanced Adaptive
- **Selection**: Fitness-proportional
- **Memory Compaction**: Balanced (50% score, 30% diversity, 20% frequency) ← CHANGED
- **Evolution**: Adaptive
- **Memory Retrieval**: Balanced weighting
- **Hypothesis**: Self-regulating, maintains diversity

---

## Key Architectural Changes Summary

### What Changed

| Aspect | OLD (Wrong) | NEW (Correct) |
|--------|-------------|---------------|
| **Primary Evolution** | Mutate prompts | Inherit memories |
| **Genome (Prompt)** | Active evolution target | Mostly stable |
| **Memories** | Agent-specific, not inherited | Compacted and passed to offspring |
| **Offspring Start** | Empty memories | 50-100 inherited memories |
| **Knowledge Transfer** | Through prompt instructions | Through memory content |
| **Milestone 2 Focus** | Genome mutation | Memory compaction |
| **Strategy Variation** | Mutation rates | Compaction strategies |

### Why This is Better

1. **Prompts are bad at encoding nuance** - That's why we need RAG
2. **Evolution on what matters** - Successful patterns (memories), not instructions
3. **Lamarckian learning** - Acquired knowledge passes to offspring
4. **Knowledge accumulation** - Each generation builds on previous
5. **Avoids prompt engineering trap** - Stop trying to write better instructions

### What This Tests

**Core Question**: Does memory inheritance create agents that improve faster than agents starting from scratch each generation?

**Hypothesis**: Offspring inheriting compacted parent memories should:
- Start with higher baseline performance
- Reach peak performance faster
- Show cumulative knowledge across generations
- Demonstrate "cultural evolution" (knowledge passing between agents)

**Measurement**: Compare Strategy A/B/C (different compaction methods) to see which way of inheriting knowledge works best.

---

## Implementation Timeline (Updated)

### Week 1: Milestone 1 (Core System) - UNCHANGED
```
Day 1-2: Branches 1.1, 1.2, 1.3 (parallel) - 1.2 updated for inherited memories
Day 3:   Branch 1.4 (after 1.1)
Day 4:   Branch 1.5 (after all) - remove parameter evolution
```

### Week 2: Milestone 2 (Memory Inheritance) - REDESIGNED
```
Day 1-2: Branches 2.1 (compaction), 2.2 (inheritance) (parallel)
Day 3:   Branch 2.3 (after 2.1, 2.2)
Day 4:   Branch 2.4 (after 2.3)
```

### Week 3: Milestone 3 (Strategies) - UPDATED
```
Day 1-2: Update strategies for memory compaction focus
Day 3:   Parallel execution framework
```

### Week 4: Milestone 4 + 5 - UNCHANGED

---

## Critical Implementation Notes

### Memory Inheritance Mechanics

**Question**: How do offspring memories interact with personal experience?

**Answer**: They merge naturally in ChromaDB:
- Inherited memories marked with `metadata.inherited = True`
- Personal memories marked with `metadata.inherited = False`
- Retrieval doesn't distinguish (both searched)
- Over time, personal high-quality memories dominate through score weighting

**Question**: Does this create memory explosion?

**Answer**: No, because:
- Compaction keeps inherited memories at ~100
- Personal memories accumulate over lifetime (~100-200)
- Total per agent: ~200-300 memories max
- Multiply by 15 agents = ~4500 memories total (manageable)

### Prompt Stability

**Question**: Why keep prompts mostly stable?

**Answer**:
1. Prompts can't encode the nuanced knowledge we're learning
2. Evolution should act on the memories (where the knowledge is)
3. Prompt variation only for minimal diversity (5% chance, simple synonyms)
4. This avoids the "prompt engineering trap" of trying to word better instructions

**Question**: But won't all agents end up identical if prompts don't change?

**Answer**: No, because:
- Different inherited memories = different starting knowledge
- Different personal experiences = unique lifetime learning
- Different retrieval patterns = different response styles
- Memories ARE the differentiation, not prompts

---

## Next Steps

1. **Immediate**: Update existing AGENT_TASK.md files for M1.2 and M1.5
2. **New**: Create AGENT_TASK.md for M2.1 (memory-compaction) and M2.2 (memory-inheritance-reproduction)
3. **Week 1**: Implement M1 branches
4. **Week 2**: Implement M2 branches with memory inheritance
5. **Week 3-4**: Strategies and experiments testing compaction approaches

**Ready to proceed with this revised architecture?**
