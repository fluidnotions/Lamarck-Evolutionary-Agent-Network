## Migration Guide: BaseAgent → BaseAgentV2

**Date**: 2025-10-20
**Status**: Ready for migration
**Estimated Time**: 2-4 hours for full codebase

---

## Overview

This guide explains how to migrate from the old `BaseAgent` + `MemoryManager` architecture to the new `BaseAgentV2` + `ReasoningMemory` + `SharedRAG` architecture.

**Key Changes**:
- `MemoryManager` → `ReasoningMemory` (per-agent cognitive patterns)
- New: `SharedRAG` (shared domain knowledge)
- `generate_content()` → `generate_with_reasoning()` (with <think>/<final> tags)
- Storage: Reasoning patterns vs. domain knowledge separation

---

## Quick Comparison

### Old Architecture:
```python
from lean.memory import MemoryManager
from lean.agents import BaseAgent, IntroAgent

memory = MemoryManager(collection_name="intro_agent_1_memories")
agent = IntroAgent(role="intro", memory_manager=memory)

# Generate
content = await agent.generate_content(state, memories)

# Store
agent.store_memory(score)
```

### New Architecture:
```python
from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.shared_rag import SharedRAG
from lean.base_agent_v2 import IntroAgentV2

# Per-agent reasoning patterns
collection_name = generate_reasoning_collection_name("intro", "agent_1")
reasoning_memory = ReasoningMemory(collection_name=collection_name)

# Shared knowledge base
shared_rag = SharedRAG()  # Shared by ALL agents

agent = IntroAgentV2(
    role="intro",
    agent_id="intro_agent_1",
    reasoning_memory=reasoning_memory,
    shared_rag=shared_rag
)

# Generate with reasoning
result = agent.generate_with_reasoning(
    topic=topic,
    reasoning_patterns=reasoning_patterns,  # From retrieve_similar_reasoning()
    domain_knowledge=domain_knowledge,      # From shared_rag.retrieve()
    reasoning_context=reasoning_context
)

# Store reasoning pattern
agent.prepare_reasoning_storage(
    thinking=result['thinking'],
    output=result['output'],
    topic=topic,
    domain=domain,
    generation=generation,
    context_sources=context_sources
)
agent.record_fitness(score=score, domain=domain)
agent.store_reasoning_and_output(score=score)
```

---

## Migration Steps

### Step 1: Update Dependencies

**Install new requirements**:
```bash
pip install chromadb sentence-transformers
```

**Update imports**:
```python
# OLD
from lean.memory import MemoryManager
from lean.agents import BaseAgent, IntroAgent, BodyAgent, ConclusionAgent

# NEW
from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.shared_rag import SharedRAG
from lean.base_agent_v2 import IntroAgentV2, BodyAgentV2, ConclusionAgentV2
```

---

### Step 2: Update Agent Initialization

**OLD**:
```python
from lean.agents import create_agents

agents = create_agents(persist_directory="./data/memories")
# Returns: {'intro': IntroAgent, 'body': BodyAgent, 'conclusion': ConclusionAgent}
```

**NEW (Recommended - Using Factory)**:
```python
from lean.base_agent_v2 import create_agents_v2

agents = create_agents_v2(
    reasoning_dir="./data/reasoning",
    shared_rag_dir="./data/shared_rag",
    agent_ids={'intro': 'agent_1', 'body': 'agent_1', 'conclusion': 'agent_1'}
)
# Returns: {'intro': IntroAgentV2, 'body': BodyAgentV2, 'conclusion': ConclusionAgentV2}
```

**NEW (Manual - For Custom Setup)**:
```python
from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.shared_rag import SharedRAG
from lean.base_agent_v2 import IntroAgentV2

# Create shared RAG (ONE instance for all agents)
shared_rag = SharedRAG(persist_directory="./data/shared_rag")

# Create intro agent
collection_name = generate_reasoning_collection_name("intro", "agent_1")
reasoning_memory = ReasoningMemory(
    collection_name=collection_name,
    persist_directory="./data/reasoning"
)

intro_agent = IntroAgentV2(
    role="intro",
    agent_id="intro_agent_1",
    reasoning_memory=reasoning_memory,
    shared_rag=shared_rag
)
```

**Key Changes**:
- Use `create_agents_v2()` factory for simple migration (recommended)
- Or manually create agents for custom setup
- Use `generate_reasoning_collection_name()` for collection naming
- Create `ReasoningMemory` instance per agent
- Create ONE `SharedRAG` instance, pass to all agents
- Provide explicit `agent_id`

---

### Step 3: Update Workflow (8-Step Cycle)

**OLD Workflow**:
```python
# Retrieve memories
memories = agent.memory.retrieve(state['topic'])

# Generate
content = await agent.generate_content(state, memories)

# Evaluate
score = evaluator.evaluate(content)

# Store
agent.store_memory(score)
```

**NEW Workflow** (8-step cycle):
```python
# STEP 2: Plan approach
reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(
    query=topic,
    k=5
)

# STEP 3: Retrieve knowledge
domain_knowledge = shared_rag.retrieve(
    query=topic,
    k=3
)

# STEP 4: Receive context (from ContextManager)
reasoning_context = context_manager.assemble_context(
    current_agent=agent,
    hierarchy_context=hierarchy_context,
    all_pools=all_pools,
    workflow_state=state
)

# STEP 5: Generate
result = agent.generate_with_reasoning(
    topic=topic,
    reasoning_patterns=reasoning_patterns,
    domain_knowledge=domain_knowledge,
    reasoning_context=reasoning_context['content']
)

# STEP 6: Evaluate
score = evaluator.evaluate(result['output'])

# STEP 7: Store
agent.prepare_reasoning_storage(
    thinking=result['thinking'],
    output=result['output'],
    topic=topic,
    domain=domain,
    generation=generation,
    context_sources=reasoning_context['sources']
)
agent.record_fitness(score=score, domain=domain)
agent.store_reasoning_and_output(score=score)
```

---

### Step 4: Update State Management

**Add reasoning fields to state**:
```python
# OLD
state['intro_output'] = content
state['intro_score'] = score

# NEW
state['intro_output'] = result['output']  # <final> content
state['intro_reasoning'] = result['thinking']  # <think> content
state['intro_score'] = score
```

**Why**: Context distribution needs reasoning traces, not outputs.

---

### Step 5: Update Context Distribution

**OLD** (distributes outputs):
```python
context = peer_agent.pending_memory['content']  # Output content
```

**NEW** (distributes reasoning traces):
```python
# Use ContextManager (already implemented in context_manager.py)
from lean.context_manager import ContextManager

context_manager = ContextManager()
reasoning_context = context_manager.assemble_context(
    current_agent=agent,
    hierarchy_context=task_description,
    all_pools={'intro': intro_pool, 'body': body_pool, 'conclusion': conclusion_pool},
    workflow_state=state
)
```

---

### Step 6: Update Agent Pools

**Add required methods for ContextManager**:
```python
class AgentPool:
    def get_top_n(self, n: int = 2):
        """Get top N agents by fitness."""
        sorted_agents = sorted(
            self.agents,
            key=lambda a: a.avg_fitness(),
            reverse=True
        )
        return sorted_agents[:n]

    def get_random_lower_half(self):
        """Get random agent from lower half by fitness."""
        import random
        sorted_agents = sorted(self.agents, key=lambda a: a.avg_fitness())
        lower_half = sorted_agents[:len(sorted_agents)//2]
        return random.choice(lower_half) if lower_half else self.agents[0]

    def size(self):
        """Get pool size."""
        return len(self.agents)
```

---

### Step 7: Update Inheritance (M2)

**OLD** (content inheritance):
```python
inherited_memories = parent.memory.get_all_memories()
child_memory = MemoryManager(
    collection_name=child_collection,
    inherited_memories=inherited_memories
)
```

**NEW** (reasoning pattern inheritance):
```python
# Get parent's reasoning patterns
parent_reasoning = parent.reasoning_memory.get_all_reasoning()

# Compact/select best patterns (see M2 compaction strategies)
compacted_reasoning = compact_reasoning_patterns(parent_reasoning, max_size=100)

# Create child with inherited patterns
child_reasoning_memory = ReasoningMemory(
    collection_name=child_collection,
    inherited_reasoning=compacted_reasoning
)
```

---

## Data Migration

### Option 1: Fresh Start (Recommended)
- Start with clean `./data/reasoning/` and `./data/shared_rag/` directories
- Let agents build new reasoning patterns from scratch
- Faster, cleaner approach

### Option 2: Convert Existing Data
If you have valuable old memories:

```python
def migrate_memories_to_reasoning(old_collection_name: str, new_collection_name: str):
    """Convert old MemoryManager data to ReasoningMemory format."""

    # Load old memories
    old_memory = MemoryManager(collection_name=old_collection_name)
    old_data = old_memory.get_all_memories()

    # Create new reasoning memory
    new_memory = ReasoningMemory(collection_name=new_collection_name)

    for mem in old_data:
        # Extract reasoning pattern from old content
        # This is a heuristic - adjust based on your data
        content = mem['content']

        # Store as reasoning pattern
        new_memory.store_reasoning_pattern(
            reasoning=f"Generated content: {content[:200]}...",  # Simplified
            score=mem['score'],
            situation=f"Topic: {mem.get('topic', 'unknown')}",
            tactic="legacy_migration",
            metadata=mem.get('metadata', {})
        )

    print(f"Migrated {len(old_data)} memories → reasoning patterns")
```

**Note**: This is a simplified migration. True reasoning patterns should come from `<think>` tags.

---

## Testing Migration

### Test 1: Agent Creation
```python
def test_agent_creation():
    shared_rag = SharedRAG()
    collection_name = generate_reasoning_collection_name("intro", "test_1")
    reasoning_memory = ReasoningMemory(collection_name=collection_name)

    agent = IntroAgentV2(
        role="intro",
        agent_id="test_intro_1",
        reasoning_memory=reasoning_memory,
        shared_rag=shared_rag
    )

    assert agent.agent_id == "test_intro_1"
    assert agent.role == "intro"
    print("✅ Agent creation successful")
```

### Test 2: Storage Separation
```python
def test_storage_separation():
    # Should create separate directories
    assert os.path.exists("./data/reasoning/")
    assert os.path.exists("./data/shared_rag/")
    print("✅ Storage separation working")
```

### Test 3: Generation & Storage
```python
def test_generation_and_storage():
    # Generate
    result = agent.generate_with_reasoning(
        topic="Test topic",
        reasoning_patterns=[],
        domain_knowledge=[],
        reasoning_context=""
    )

    # Should have thinking and output
    assert 'thinking' in result
    assert 'output' in result

    # Store
    agent.prepare_reasoning_storage(
        thinking=result['thinking'],
        output=result['output'],
        topic="Test topic",
        domain="General",
        generation=1,
        context_sources=[]
    )
    agent.store_reasoning_and_output(score=8.5)

    # Verify reasoning stored
    assert agent.reasoning_memory.count() > 0
    # Verify output stored in shared RAG (score >= 8.0)
    assert agent.shared_rag.count() > 0

    print("✅ Generation and storage successful")
```

---

## Common Issues & Solutions

### Issue 1: Missing <think>/<final> tags
**Problem**: LLM doesn't always return tags.
**Solution**: BaseAgentV2 has fallback - treats entire response as output.
```python
# In _parse_response():
if not thinking and not output:
    thinking = "No reasoning provided"
    output = response_text
```

### Issue 2: Shared RAG not populating
**Problem**: No outputs stored in shared RAG.
**Solution**: Check score threshold (default 8.0).
```python
# Lower threshold for testing
os.environ['SHARED_RAG_MIN_SCORE'] = '7.0'
```

### Issue 3: Context distribution fails
**Problem**: `_get_agent_recent_reasoning()` returns empty.
**Solution**: Ensure agents have `reasoning_memory` attribute.
```python
# In context_manager.py, there's a try/except fallback
try:
    reasoning_patterns = agent.reasoning_memory.get_all_reasoning(...)
except AttributeError:
    print(f"[Warning] Agent {agent.agent_id} missing reasoning_memory")
    return ""
```

---

## Rollback Plan

If migration fails:

1. **Backup old data**:
   ```bash
   cp -r ./data/memories ./data/memories_backup
   ```

2. **Keep old code**:
   - Old `BaseAgent` still in `agents.py`
   - Old `MemoryManager` still in `memory.py`

3. **Revert imports**:
   ```python
   # Switch back
   from lean.agents import BaseAgent, IntroAgent
   from lean.memory import MemoryManager
   ```

---

## Verification Checklist

After migration:

- [ ] All agents use `BaseAgentV2` (or subclasses)
- [ ] Single `SharedRAG` instance created
- [ ] Each agent has own `ReasoningMemory` instance
- [ ] Workflow implements 8-step cycle
- [ ] State includes `{role}_reasoning` fields
- [ ] Context distribution uses `ContextManager`
- [ ] Storage separation: `./data/reasoning/` and `./data/shared_rag/`
- [ ] Tests pass
- [ ] Can run 5+ generations successfully

---

## Performance Considerations

### Token Usage
- `<think>` tags increase token count (~20-30% more)
- **Mitigation**: Monitor API costs, consider shorter reasoning

### Storage Growth
- Reasoning patterns accumulate (100-250 per agent)
- **Mitigation**: Implement compaction (M2)

### Retrieval Speed
- Large collections may slow down
- **Mitigation**: Optimize ChromaDB settings, limit retrieval count

---

## Next Steps After Migration

1. **Run multi-generation tests** (5-10 generations)
2. **Validate reasoning improvement** (compare early vs. late generations)
3. **Implement M2** (evolution, compaction, reproduction)
4. **Tune parameters**:
   - `MAX_REASONING_RETRIEVE`
   - `MAX_KNOWLEDGE_RETRIEVE`
   - `SHARED_RAG_MIN_SCORE`

---

## Support & Resources

- **Code**: `src/lean/base_agent_v2.py`, `reasoning_memory.py`, `shared_rag.py`
- **Tests**: `tests/test_reasoning_integration.py`
- **Examples**: `examples/reasoning_pattern_demo.py`, `examples/simple_workflow_demo.py`
- **Docs**: `docs/planning/CONSOLIDATED_PLAN.md`, `docs/brainstorming/REFINEMENT.md`

---

## Summary

**Key Changes**:
1. MemoryManager → ReasoningMemory (cognitive patterns)
2. New SharedRAG (domain knowledge)
3. generate_content() → generate_with_reasoning() (<think>/<final>)
4. 8-step cycle in workflow
5. Storage separation

**Benefits**:
- Evolve HOW agents think, not WHAT they produce
- Shared knowledge base prevents redundant learning
- Cleaner separation of concerns (3-layer architecture)
- Better lineage tracking (parent reasoning → child)

**Estimated Time**: 2-4 hours for full codebase migration

🚀 **Ready to migrate!**
