# Architecture Refinement Analysis

**Date**: 2025-10-20
**Status**: Architecture refinement complete, branch validity assessed

---

## The Key Conceptual Shift

### OLD (Fuzzy)
- **Memories = Content/Outputs**
- Agents stored successful outputs (paragraphs, content)
- Problem: Content only applies to similar topics
- Didn't make sense for diverse topics

### NEW (Clear)
- **Memories = Reasoning Patterns (Cognitive Strategies)**
- Agents store planning steps + execution traces
- Content goes to **shared RAG** (available to all)
- Three-layer separation: Prompts (WHAT to do) + RAG (WHAT to know) + Reasoning (HOW to think)

### Example of What Gets Stored NOW

**LLM Response with Externalized Reasoning:**
```xml
<think>
The task is writing an introduction about AI understanding evolution.
Based on past patterns, I should:
1. Start with a historical anchor (Turing seems perfect given the topic)
2. Add a statistic about modern AI interpretability research
3. Build tension by contrasting past and present
4. End with a question about future implications

I'll use Turing's imitation game as the hook, cite recent interpretability
progress statistics, then pose a question about whether machines can truly
understand themselves.
</think>

<final>
In 1950, Alan Turing proposed a test: if a machine could fool humans into
thinking it was conscious, did the distinction matter? Seventy-five years
later, as neural networks process billions of parameters, a new question
emerges—not whether machines can think, but whether they can explain how
they think. Can AI learn to interpret itself?
</final>
```

**What gets STORED (in per-agent reasoning collection):**
```python
{
    # Extracted from <think> tags
    "situation": "writing intro for AI evolution topic",
    "tactic": "historical anchor → statistic → tension → question",
    "reasoning": "<full <think> content>",

    # Metadata
    "score": 8.5,
    "retrieval_count": 12,
    "generation": 5,
    "provenance": {"agent_id": "intro_agent_2", "timestamp": "..."},
    "embedding": [...],  # Vector for similarity search
    "inherited_from": ["parent1_reasoning_087", "parent2_reasoning_134"]
}
```

**What does NOT get stored:**
```python
# Goes to shared RAG if needed (Layer 2):
"In 1950, Alan Turing proposed a test..."  # The actual output content
```

---

## Implementation: The 4-Step Process

**Key Challenge**: LLM APIs don't expose internal reasoning—only final outputs.

**Solution**: Induce the model to externalize reasoning via structured prompts.

### Step 1: Capture the Reasoning Trace

Add to system prompt:
```
SYSTEM: You are an intro writer. When generating content, include your
reasoning under <think> tags and your final output under <final> tags.
```

Provide context from:
- Retrieved reasoning patterns (from vector DB)
- Shared RAG knowledge (domain facts)
- Reasoning traces from other agents (40/30/20/10 distribution)

### Step 2: Extract and Store the Reasoning

Parse the LLM response:
1. Extract `<think>` section → this is the reasoning pattern
2. Extract `<final>` section → this is the output (optional: store in shared RAG)
3. Structure as reasoning unit with metadata:
   - `situation`, `tactic`, `reasoning`
   - `score`, `retrieval_count`, `generation`
   - `provenance`, `embedding`, `inherited_from`

### Step 3: Retrieval by Reasoning Similarity

Next generation, similar task:
1. Embed the new task description
2. Search vector DB for similar reasoning patterns (NOT content)
3. Filter by `score > 7.0` (only high-performing reasoning)
4. Return top-k reasoning patterns
5. Include in context for next generation

### Step 4: Scoring and Compaction

After evaluation:
1. Score the output (LLM evaluator: engagement, clarity, depth)
2. Update reasoning unit's score
3. Increment `retrieval_count` each time pattern is used
4. Periodically (every 10 generations):
   - Prune low performers (score < 6.0 after 20+ uses)
   - Merge similar patterns (cluster by embedding distance)
   - Abstract new patterns from successful combinations
   - Compact and pass to offspring

**This is the evolutionary loop**: Good reasoning patterns survive and propagate, bad ones get pruned.

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Layer 1: Fixed Prompts (Interface)     │
│  "You are an intro writer"              │
│  - Never changes                         │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Layer 2: Shared RAG (Knowledge)        │
│  Domain facts, content, references      │
│  - Available to all agents              │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│  Layer 3: Evolving Reasoning            │
│  Planning steps, cognitive strategies   │
│  - Per-agent, inherited from parents    │
└─────────────────────────────────────────┘
```

**Key Insight**: Models think in embeddings, not text. Vector search excels at finding similar reasoning structures. "Find planning sequences like mine" is exactly what embedding search was built for.

---

## Branch Validity Assessment

### ✅ STILL VALID (No Changes Needed)

#### M1.1: agent-pool-infrastructure (MERGED)
- Population management is independent of what's stored
- Selection strategies work the same way
- No changes needed

#### M1.3: fitness-tracking (WORKTREE ACTIVE)
- Tracks performance based on output quality
- Independent of storage architecture
- Can proceed as planned

#### M1.6: tavily-web-search (MERGED)
- Provides external knowledge for shared RAG
- Purpose unchanged
- No changes needed

#### visualization-v2 (WORKTREE ACTIVE)
- Visualizes outputs and performance
- Independent of storage architecture
- No changes needed

---

### ⚠️ NEEDS REVISION

#### M1.2: individual-memory-collections (MERGED - NEEDS REFACTOR)

**Current state**: Stores content/experiences
**Needs to become**: `ReasoningMemory` class storing cognitive patterns

**Required changes**:
1. Rename `MemoryManager` → `ReasoningMemory`
2. Update schema:
   - Add `planning_steps` field (list of strings)
   - Add `execution_trace` field (string)
   - Add `context_type` field (string)
   - Keep `score`, `retrieval_count`, `generation`
   - Remove content storage
3. Create separate `SharedRAG` class for domain knowledge
4. Update retrieval to query by reasoning pattern similarity

**Impact**: FOUNDATIONAL - other branches depend on this

---

#### M1.4: context-distribution (WORKTREE ACTIVE)

**Current plan**: Distribute content/outputs
**Needs to become**: Distribute reasoning traces

**From README**:
> "Receive context: Get reasoning traces from other agents (40% hierarchy, 30% high-performers, 20% random, 10% peer)"

**Required changes**:
1. Update AGENT_TASK.md to reflect reasoning trace distribution
2. Share planning steps + execution traces, NOT generated content
3. 40/30/20/10 distribution stays the same
4. Broadcast tracking concept stays the same

**Action**: Update task description, then implement

---

#### M1.5: pipeline-integration (WORKTREE ACTIVE)

**Current plan**: 7-step cycle with content storage
**Needs to become**: 8-step cycle with reasoning pattern storage

**New workflow** (from README):
1. **Start with inheritance**: 50-100 reasoning patterns from parents
2. **Plan approach**: Query reasoning patterns for similar tasks
3. **Retrieve knowledge**: Get domain facts from shared RAG
4. **Receive context**: Get reasoning traces from other agents (40/30/20/10)
5. **Generate**: Fixed prompt + reasoning patterns + knowledge + context
6. **Evaluate**: Score output quality
7. **Store reasoning pattern**: Planning steps + execution trace + score
8. **Evolve**: Every 10 generations

**Required changes**:
1. Update AGENT_TASK.md to reflect 8-step cycle
2. Integrate `ReasoningMemory` class (from refactored M1.2)
3. Integrate `SharedRAG` class (from refactored M1.2)
4. Add "Plan Approach" step (query reasoning patterns)
5. Separate "Retrieve Knowledge" (shared RAG) from reasoning retrieval
6. Update storage to save reasoning patterns, not content

**Action**: Major refactor, update task description first

---

#### core-concept-refactor (WORKTREE ACTIVE)

**Status**: REVIEW NEEDED
**Reason**: May be superseded by this README refinement

**Action**: Review design docs in worktree to see if still relevant

---

## Updated Priority Order

### Priority 0: Architecture Alignment (CRITICAL)

**Refactor M1.2** - MUST DO FIRST
- Everything depends on this
- Changes `MemoryManager` → `ReasoningMemory` + `SharedRAG`
- Updates storage schema for reasoning patterns
- **Implementation**: Use `<think>` tags to capture reasoning, `<final>` tags for output
- Extract `<think>` content → store as reasoning pattern
- Extract `<final>` content → optionally store in shared RAG

### Priority 1: Update M1 Branches

1. **M1.3 (fitness-tracking)** - ✅ Proceed as planned
2. **M1.4 (context-distribution)** - ⚠️ Update AGENT_TASK.md first, then implement
3. **M1.5 (pipeline-integration)** - ⚠️ Update AGENT_TASK.md first, then major refactor

### Priority 2: Verify M1 Complete

- Run 20 generations successfully
- Reasoning pattern retrieval works
- Shared RAG separation works
- Context distribution shares reasoning traces
- All tests passing

### Priority 3: Begin M2

- Reasoning pattern compaction strategies
- Cognitive strategy inheritance
- Population management

---

## Configuration Changes

```bash
# NEW: Three-layer architecture
INHERITED_REASONING_SIZE=100    # Max reasoning patterns from parents
PERSONAL_REASONING_SIZE=150     # Max personal reasoning patterns
USE_SHARED_RAG=true             # Separate domain knowledge layer
REASONING_SEARCH_ONLY=true      # Don't mix content with reasoning
PROMPTS_IMMUTABLE=true          # Never modify prompts

# UPDATED: Context distribution
CONTEXT_WEIGHTS=40,30,20,10     # Now shares reasoning traces, not content

# DEPRECATED: Memory threshold
# MEMORY_SCORE_THRESHOLD=7.0    # No longer relevant
```

---

## Research Question (Updated)

**OLD**: Can AI agents improve by inheriting their parents' learned knowledge?
**NEW**: Can AI agents improve by inheriting their parents' reasoning patterns?

**Key Difference**: We're evolving HOW agents think (cognitive strategies), not WHAT they know (content). Content is shared via RAG. Reasoning patterns are inherited.

---

## Success Criteria (Updated)

1. **Reasoning Improvement**: Average scores increase >0.5 points through better cognitive strategies
2. **Emergent Specialization**: Roles develop distinct reasoning patterns
3. **Pattern Effectiveness**: Retrieved reasoning correlates with performance
4. **Sustained Diversity**: Multiple problem-solving approaches coexist
5. **Strategy Winner**: One compaction approach demonstrably outperforms

---

## Summary

### What Changed
- Memories → Reasoning patterns (cognitive strategies)
- Content → Shared RAG (domain knowledge)
- Added three-layer architecture (Prompts/RAG/Reasoning)

### What Stayed the Same
- Population structure (15 agents, 3 roles)
- Selection mechanisms (ε-greedy, tournament, fitness-weighted)
- Evolution frequency (every 10 generations)
- 40/30/20/10 context distribution (now shares reasoning traces)
- Three compaction strategies (A/B/C)

### Branch Impact
- ✅ **Valid**: M1.1, M1.3, M1.6, visualization-v2
- ⚠️ **Revision**: M1.2 (refactor), M1.4 (update), M1.5 (major refactor)
- ❓ **Review**: core-concept-refactor

### Next Steps
1. Refactor M1.2 to ReasoningMemory + SharedRAG
2. Update AGENT_TASK.md for M1.4 and M1.5
3. Complete M1 branches
4. Verify integration
5. Proceed to M2 (reasoning pattern inheritance)
