# Agent Task Breakdown

## Task 1: Fix Visualization Tracking for Aggregate/Critique Nodes
**Branch:** `fix/visualization-tracking`
**Worktree:** `../lean-viz-tracking`

**Problem:** The coordinator activity panel doesn't properly track aggregate and critique phases. The phases rely on checking `stream_logs` for keywords, but the aggregate/critique nodes may not be adding consistent log messages.

**Tasks:**
- Add explicit stream_log messages in `_aggregate_node` and `_critique_node`
- Update visualization to properly detect these phases
- Ensure checkmarks (✓) appear correctly for completed phases

**Files to modify:**
- `src/lean/pipeline.py` (aggregate/critique nodes)
- `src/lean/visualization.py` (coordinator panel tracking)

---

## Task 2: Add Logging for Reasoning Patterns and Inheritance
**Branch:** `feat/reasoning-logs`
**Worktree:** `../lean-reasoning-logs`

**Problem:** Memory retrieval isn't being logged. Users can't see what reasoning patterns are being retrieved or inherited during agent execution.

**Tasks:**
- Add logger.info() calls in `_execute_ensemble` showing retrieved reasoning patterns
- Log inherited reasoning patterns from parent agents during reproduction
- Show pattern count, source (inherited vs learned), and similarity scores
- Update visualization to display this info in memory panel

**Files to modify:**
- `src/lean/pipeline.py` (_execute_ensemble method)
- `src/lean/reproduction.py` (pattern inheritance)
- `src/lean/base_agent.py` (generate_with_reasoning)
- `src/lean/visualization.py` (memory panel)

---

## Task 3: Fix Specialist Agent Invocation
**Branch:** `fix/specialist-execution`
**Worktree:** `../lean-specialists`

**Problem:** Specialist agents (Researcher, Fact Checker, Stylist) are never being invoked. They should run after winning agents are selected to improve outputs.

**Current behavior:** Specialists are only conditionally called within body generation
**Desired behavior:** Specialists should process winner outputs after ensemble selection

**Tasks:**
- Modify specialist invocation to run on ALL winning outputs (intro, body, conclusion)
- Add specialist nodes to LangGraph OR run specialists in post-processing after ensemble selection
- Ensure specialists also use reasoning memory and evolve over time
- Log specialist activity with improvements made

**Files to modify:**
- `src/lean/pipeline.py` (add specialist processing after ensemble)
- `src/lean/specialists.py` (ensure reasoning memory integration)
- `src/lean/visualization.py` (track specialist execution)

---

## Task 4: Add LangGraph Visualization/Tracing
**Branch:** `feat/langgraph-debug`
**Worktree:** `../lean-langgraph-debug`

**Problem:** No visibility into LangGraph execution flow. Hard to debug which nodes execute and in what order.

**Tasks:**
- Add LangGraph tracing/debugging output
- Create visualization of graph execution path
- Log node entry/exit with timing information
- Add graph structure export for debugging

**Files to modify:**
- `src/lean/pipeline.py` (_build_graph, generate methods)
- Create new file: `src/lean/graph_debugger.py` for tracing utilities

---

## Key Question to Investigate

**Memory Retrieval Mystery:**
- Scores ARE improving (7→8.5 for intro, 6→6.2 for conclusion)
- BUT visualization shows "No memories retrieved"
- Memory retrieval happens in `_execute_ensemble` via:
  ```python
  reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(query=topic, k=5)
  domain_knowledge = agent.shared_rag.retrieve(query=topic, k=3)
  ```

**Hypothesis:**
1. Memory IS being retrieved (that's why scores improve)
2. BUT state tracking fields aren't being updated:
   - `state['reasoning_patterns_used'][role]` 
   - `state['domain_knowledge_used'][role]`
3. Visualization relies on these state fields to display memory info

**Fix:** Update `_execute_ensemble` to populate these tracking fields.

**Where to look:**
- `src/lean/pipeline.py:246-259` (memory retrieval in ensemble)
- Check if `context_sources` are tracked correctly
- Verify state updates for memory tracking
