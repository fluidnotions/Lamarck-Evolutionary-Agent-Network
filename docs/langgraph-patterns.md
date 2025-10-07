# LangGraph Patterns - Customization Guide

This guide explains LangGraph patterns used in HVAS Mini and how to customize the workflow orchestration.

---

## Table of Contents

1. [LangGraph Basics](#langgraph-basics)
2. [Current Workflow Pattern](#current-workflow-pattern)
3. [Parallel Execution](#parallel-execution)
4. [Conditional Routing](#conditional-routing)
5. [Checkpointing & Memory](#checkpointing--memory)
6. [Streaming Patterns](#streaming-patterns)
7. [Advanced Patterns](#advanced-patterns)
8. [Examples](#examples)

---

## LangGraph Basics

### Core Concepts

**StateGraph**: Defines workflow as a directed graph
- **Nodes**: Functions that process state
- **Edges**: Connections determining execution flow
- **State**: TypedDict shared across all nodes

### Basic Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define state
class MyState(TypedDict):
    input: str
    output: str

# Create graph
workflow = StateGraph(MyState)

# Add nodes
workflow.add_node("process", my_function)

# Define flow
workflow.set_entry_point("process")
workflow.add_edge("process", END)

# Compile
app = workflow.compile()

# Execute
result = await app.ainvoke({"input": "hello"})
```

---

## Current Workflow Pattern

### Sequential Pipeline

HVAS Mini uses a linear sequential pattern:

```python
def _build_graph(self) -> StateGraph:
    """Current sequential workflow."""
    workflow = StateGraph(BlogState)

    # Add all nodes
    workflow.add_node("intro", self.agents["intro"])
    workflow.add_node("body", self.agents["body"])
    workflow.add_node("conclusion", self.agents["conclusion"])
    workflow.add_node("evaluate", self.evaluator)
    workflow.add_node("evolve", self._evolution_node)

    # Linear flow: A → B → C → D → E → END
    workflow.set_entry_point("intro")
    workflow.add_edge("intro", "body")
    workflow.add_edge("body", "conclusion")
    workflow.add_edge("conclusion", "evaluate")
    workflow.add_edge("evaluate", "evolve")
    workflow.add_edge("evolve", END)

    return workflow.compile(checkpointer=MemorySaver())
```

### Execution Flow

```
START
  ↓
intro (writes state["intro"])
  ↓
body (reads state["intro"], writes state["body"])
  ↓
conclusion (reads state["intro"] + state["body"], writes state["conclusion"])
  ↓
evaluate (reads all content, writes state["scores"])
  ↓
evolve (reads state["scores"], updates agent parameters)
  ↓
END
```

---

## Parallel Execution

### Pattern 1: Fan-Out / Fan-In

Execute multiple agents in parallel, then combine:

```python
def _build_parallel_graph(self) -> StateGraph:
    """Parallel agent execution."""
    workflow = StateGraph(BlogState)

    # Add nodes
    workflow.add_node("intro", self.agents["intro"])
    workflow.add_node("body", self.agents["body"])
    workflow.add_node("conclusion", self.agents["conclusion"])
    workflow.add_node("combine", self._combine_node)
    workflow.add_node("evaluate", self.evaluator)

    # Parallel execution
    workflow.set_entry_point("intro")

    # From intro, fan out to body AND conclusion simultaneously
    workflow.add_edge("intro", "body")
    workflow.add_edge("intro", "conclusion")

    # Both must complete before combine
    workflow.add_edge("body", "combine")
    workflow.add_edge("conclusion", "combine")

    workflow.add_edge("combine", "evaluate")
    workflow.add_edge("evaluate", END)

    return workflow.compile()

def _combine_node(self, state: BlogState) -> BlogState:
    """Combine parallel outputs."""
    # All sections now available
    state["stream_logs"].append("[Combiner] All sections complete")
    return state
```

### Pattern 2: Independent Parallel Branches

```python
def _build_independent_parallel_graph(self) -> StateGraph:
    """Truly independent parallel execution."""
    workflow = StateGraph(BlogState)

    # All agents can run simultaneously
    workflow.add_node("intro", self.agents["intro"])
    workflow.add_node("body", self.agents["body"])
    workflow.add_node("conclusion", self.agents["conclusion"])
    workflow.add_node("sync", self._sync_node)

    # Set all as entry points (LangGraph handles parallel execution)
    workflow.set_entry_point("intro")
    workflow.set_entry_point("body")
    workflow.set_entry_point("conclusion")

    # All converge to sync
    workflow.add_edge("intro", "sync")
    workflow.add_edge("body", "sync")
    workflow.add_edge("conclusion", "sync")

    workflow.add_edge("sync", END)

    return workflow.compile()
```

### Parallel Execution Trade-offs

**Pros**:
- Faster execution
- Independent failures
- Scalable

**Cons**:
- No context sharing (agents can't see each other's outputs)
- More complex error handling
- Potential race conditions

---

## Conditional Routing

### Pattern 1: Quality Check Branch

```python
def _build_conditional_graph(self) -> StateGraph:
    """Route based on quality scores."""
    workflow = StateGraph(BlogState)

    workflow.add_node("intro", self.agents["intro"])
    workflow.add_node("evaluate_intro", self._evaluate_intro)
    workflow.add_node("retry_intro", self._retry_intro)
    workflow.add_node("body", self.agents["body"])

    workflow.set_entry_point("intro")
    workflow.add_edge("intro", "evaluate_intro")

    # Conditional edge: route based on score
    workflow.add_conditional_edges(
        "evaluate_intro",
        self._should_retry_intro,
        {
            "retry": "retry_intro",
            "continue": "body"
        }
    )

    workflow.add_edge("retry_intro", "evaluate_intro")  # Loop back
    workflow.add_edge("body", END)

    return workflow.compile()

def _should_retry_intro(self, state: BlogState) -> str:
    """Decide whether to retry intro."""
    score = state["scores"].get("intro", 0)

    if score < 5.0 and state.get("retry_count", 0) < 3:
        return "retry"
    else:
        return "continue"

def _retry_intro(self, state: BlogState) -> BlogState:
    """Retry intro generation with modified parameters."""
    state["retry_count"] = state.get("retry_count", 0) + 1

    # Adjust temperature for retry
    self.agents["intro"].parameters["temperature"] += 0.1

    state["stream_logs"].append(f"[Retry] Attempting intro again ({state['retry_count']})")

    return state
```

### Pattern 2: Dynamic Agent Selection

```python
def _build_dynamic_graph(self) -> StateGraph:
    """Select agents based on topic."""
    workflow = StateGraph(BlogState)

    workflow.add_node("classifier", self._classify_topic)
    workflow.add_node("technical_agent", self.technical_agent)
    workflow.add_node("creative_agent", self.creative_agent)
    workflow.add_node("body", self.agents["body"])

    workflow.set_entry_point("classifier")

    # Route based on classification
    workflow.add_conditional_edges(
        "classifier",
        self._route_by_topic,
        {
            "technical": "technical_agent",
            "creative": "creative_agent"
        }
    )

    workflow.add_edge("technical_agent", "body")
    workflow.add_edge("creative_agent", "body")
    workflow.add_edge("body", END)

    return workflow.compile()

def _route_by_topic(self, state: BlogState) -> str:
    """Route based on topic classification."""
    topic = state["topic"].lower()

    technical_keywords = ["programming", "algorithm", "data", "ml", "ai"]
    if any(kw in topic for kw in technical_keywords):
        return "technical"
    else:
        return "creative"
```

---

## Checkpointing & Memory

### Pattern 1: Basic Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

def _build_checkpointed_graph(self) -> StateGraph:
    """Graph with execution checkpoints."""
    workflow = StateGraph(BlogState)

    # ... add nodes and edges ...

    # Compile with checkpointer
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Use with thread_id for persistence
async def generate(self, topic: str):
    config = {
        "configurable": {
            "thread_id": f"blog_{topic}"
        }
    }

    # Execution is checkpointed at each node
    result = await self.app.ainvoke(initial_state, config)
```

### Pattern 2: Resume from Checkpoint

```python
async def resume_generation(self, thread_id: str, updates: Dict):
    """Resume from a previous checkpoint."""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Get current state
    current_state = await self.app.aget_state(config)

    # Update state
    updated_state = {**current_state, **updates}

    # Continue execution
    result = await self.app.ainvoke(updated_state, config)

    return result
```

### Pattern 3: Persistent Checkpointing

```python
from langgraph.checkpoint.sqlite import SqliteSaver

def _build_persistent_graph(self) -> StateGraph:
    """Graph with SQLite persistence."""
    workflow = StateGraph(BlogState)

    # ... add nodes and edges ...

    # Use SQLite for persistence
    checkpointer = SqliteSaver.from_conn_string("./data/checkpoints.db")
    return workflow.compile(checkpointer=checkpointer)
```

---

## Streaming Patterns

### Pattern 1: Value Streaming (Current)

Stream complete state after each node:

```python
async def generate_with_streaming(self, topic: str):
    """Stream state updates."""
    initial_state = create_initial_state(topic)
    config = {"configurable": {"thread_id": "demo"}}

    # Stream mode: "values" (complete state after each node)
    async for state in self.app.astream(
        initial_state,
        config,
        stream_mode="values"
    ):
        # state is complete BlogState after each node
        self.visualizer.display(state)
        yield state
```

### Pattern 2: Update Streaming

Stream only changes to state:

```python
async def generate_with_updates(self, topic: str):
    """Stream only state changes."""
    initial_state = create_initial_state(topic)
    config = {"configurable": {"thread_id": "demo"}}

    # Stream mode: "updates" (only changes)
    async for update in self.app.astream(
        initial_state,
        config,
        stream_mode="updates"
    ):
        # update is Dict with only modified fields
        node_name = update.get("node")
        changes = update.get("updates")

        print(f"Node {node_name} updated: {changes.keys()}")
        yield update
```

### Pattern 3: Messages Streaming

Stream LLM token generation:

```python
async def generate_with_tokens(self, topic: str):
    """Stream LLM tokens as they're generated."""
    initial_state = create_initial_state(topic)
    config = {"configurable": {"thread_id": "demo"}}

    # Stream mode: "messages" (LLM tokens)
    async for chunk in self.app.astream(
        initial_state,
        config,
        stream_mode="messages"
    ):
        if hasattr(chunk, "content"):
            print(chunk.content, end="", flush=True)
        yield chunk
```

### Pattern 4: Custom Streaming

```python
async def generate_with_custom_streaming(self, topic: str):
    """Custom streaming with callbacks."""

    def on_node_start(node_name: str):
        print(f"Starting {node_name}...")

    def on_node_end(node_name: str, state: BlogState):
        print(f"Completed {node_name}")
        self.visualizer.update(state)

    initial_state = create_initial_state(topic)

    # Manual streaming with callbacks
    current_state = initial_state

    for node_name in ["intro", "body", "conclusion", "evaluate", "evolve"]:
        on_node_start(node_name)

        # Execute node
        node_func = self.app.nodes[node_name]
        current_state = await node_func(current_state)

        on_node_end(node_name, current_state)

        yield current_state
```

---

## Advanced Patterns

### Pattern 1: Subgraphs

```python
def _build_hierarchical_graph(self) -> StateGraph:
    """Main graph with subgraphs."""

    # Create subgraph for intro generation
    intro_subgraph = StateGraph(BlogState)
    intro_subgraph.add_node("draft", self.agents["intro"])
    intro_subgraph.add_node("refine", self._refine_intro)
    intro_subgraph.add_node("polish", self._polish_intro)
    intro_subgraph.set_entry_point("draft")
    intro_subgraph.add_edge("draft", "refine")
    intro_subgraph.add_edge("refine", "polish")
    intro_subgraph.add_edge("polish", END)
    compiled_intro = intro_subgraph.compile()

    # Main graph uses subgraph
    workflow = StateGraph(BlogState)
    workflow.add_node("intro_pipeline", compiled_intro)
    workflow.add_node("body", self.agents["body"])
    workflow.add_node("conclusion", self.agents["conclusion"])

    workflow.set_entry_point("intro_pipeline")
    workflow.add_edge("intro_pipeline", "body")
    workflow.add_edge("body", "conclusion")
    workflow.add_edge("conclusion", END)

    return workflow.compile()
```

### Pattern 2: Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

def _build_hitl_graph(self) -> StateGraph:
    """Graph with human review step."""
    workflow = StateGraph(BlogState)

    workflow.add_node("intro", self.agents["intro"])
    workflow.add_node("human_review", self._request_human_review)
    workflow.add_node("revise", self._revise_intro)
    workflow.add_node("body", self.agents["body"])

    workflow.set_entry_point("intro")
    workflow.add_edge("intro", "human_review")

    # Conditional: wait for human input
    workflow.add_conditional_edges(
        "human_review",
        self._check_human_approval,
        {
            "approved": "body",
            "revise": "revise"
        }
    )

    workflow.add_edge("revise", "human_review")
    workflow.add_edge("body", END)

    return workflow.compile(checkpointer=MemorySaver())

async def _request_human_review(self, state: BlogState) -> BlogState:
    """Pause for human review."""
    state["awaiting_human"] = True
    state["stream_logs"].append("[HITL] Awaiting human review...")
    return state

def _check_human_approval(self, state: BlogState) -> str:
    """Check if human approved."""
    return state.get("human_approved", "approved")

# Usage
async def generate_with_review(self, topic: str):
    config = {"configurable": {"thread_id": f"blog_{topic}"}}
    initial_state = create_initial_state(topic)

    # Start execution
    result = await self.app.ainvoke(initial_state, config)

    # ... system pauses at human_review node ...

    # Human provides feedback
    feedback = input("Approve intro? (yes/revise): ")

    # Update state with human decision
    updated_state = {
        **result,
        "human_approved": "approved" if feedback == "yes" else "revise",
        "awaiting_human": False
    }

    # Resume execution
    final_result = await self.app.ainvoke(updated_state, config)
    return final_result
```

### Pattern 3: Dynamic Graph Modification

```python
class AdaptivePipeline(HVASMiniPipeline):
    """Pipeline that modifies graph based on runtime conditions."""

    def _build_adaptive_graph(self, complexity: str) -> StateGraph:
        """Build graph based on complexity level."""
        workflow = StateGraph(BlogState)

        if complexity == "simple":
            # Simple: just intro and conclusion
            workflow.add_node("intro", self.agents["intro"])
            workflow.add_node("conclusion", self.agents["conclusion"])
            workflow.set_entry_point("intro")
            workflow.add_edge("intro", "conclusion")
            workflow.add_edge("conclusion", END)

        elif complexity == "standard":
            # Standard: intro, body, conclusion
            workflow.add_node("intro", self.agents["intro"])
            workflow.add_node("body", self.agents["body"])
            workflow.add_node("conclusion", self.agents["conclusion"])
            workflow.set_entry_point("intro")
            workflow.add_edge("intro", "body")
            workflow.add_edge("body", "conclusion")
            workflow.add_edge("conclusion", END)

        elif complexity == "detailed":
            # Detailed: add title, summary, SEO
            workflow.add_node("title", self.agents["title"])
            workflow.add_node("intro", self.agents["intro"])
            workflow.add_node("body", self.agents["body"])
            workflow.add_node("conclusion", self.agents["conclusion"])
            workflow.add_node("summary", self.agents["summary"])
            workflow.set_entry_point("title")
            workflow.add_edge("title", "intro")
            workflow.add_edge("intro", "body")
            workflow.add_edge("body", "conclusion")
            workflow.add_edge("conclusion", "summary")
            workflow.add_edge("summary", END)

        return workflow.compile()

    async def generate(self, topic: str, complexity: str = "standard"):
        """Generate with dynamic graph."""
        self.app = self._build_adaptive_graph(complexity)
        initial_state = create_initial_state(topic)
        return await self.app.ainvoke(initial_state)
```

---

## Examples

### Example 1: Multi-Stage Refinement

```python
def _build_refinement_graph(self) -> StateGraph:
    """Multi-stage refinement workflow."""
    workflow = StateGraph(BlogState)

    # Stage 1: Draft
    workflow.add_node("draft_intro", self.agents["intro"])
    workflow.add_node("draft_body", self.agents["body"])
    workflow.add_node("draft_conclusion", self.agents["conclusion"])

    # Stage 2: Evaluate drafts
    workflow.add_node("evaluate_drafts", self.evaluator)

    # Stage 3: Refine (conditional)
    workflow.add_node("refine_intro", self._refine_intro)
    workflow.add_node("refine_body", self._refine_body)
    workflow.add_node("refine_conclusion", self._refine_conclusion)

    # Stage 4: Final evaluation
    workflow.add_node("final_evaluate", self.evaluator)

    # Flow
    workflow.set_entry_point("draft_intro")
    workflow.add_edge("draft_intro", "draft_body")
    workflow.add_edge("draft_body", "draft_conclusion")
    workflow.add_edge("draft_conclusion", "evaluate_drafts")

    # Conditional refinement
    workflow.add_conditional_edges(
        "evaluate_drafts",
        self._needs_refinement,
        {
            "refine": "refine_intro",
            "done": "final_evaluate"
        }
    )

    workflow.add_edge("refine_intro", "refine_body")
    workflow.add_edge("refine_body", "refine_conclusion")
    workflow.add_edge("refine_conclusion", "final_evaluate")
    workflow.add_edge("final_evaluate", END)

    return workflow.compile()

def _needs_refinement(self, state: BlogState) -> str:
    """Check if any section needs refinement."""
    scores = state.get("scores", {})
    if any(score < 7.0 for score in scores.values()):
        return "refine"
    return "done"
```

### Example 2: A/B Testing Workflow

```python
def _build_ab_test_graph(self) -> StateGraph:
    """Generate multiple variants for A/B testing."""
    workflow = StateGraph(BlogState)

    # Generate variants
    workflow.add_node("variant_a_intro", self.agent_variant_a)
    workflow.add_node("variant_b_intro", self.agent_variant_b)
    workflow.add_node("combine_variants", self._combine_variants)
    workflow.add_node("evaluate_both", self._evaluate_variants)
    workflow.add_node("select_best", self._select_best_variant)

    # Parallel generation
    workflow.set_entry_point("variant_a_intro")
    workflow.set_entry_point("variant_b_intro")

    workflow.add_edge("variant_a_intro", "combine_variants")
    workflow.add_edge("variant_b_intro", "combine_variants")
    workflow.add_edge("combine_variants", "evaluate_both")
    workflow.add_edge("evaluate_both", "select_best")
    workflow.add_edge("select_best", END)

    return workflow.compile()

def _combine_variants(self, state: BlogState) -> BlogState:
    """Store both variants in state."""
    state["variants"] = {
        "a": state.get("variant_a", ""),
        "b": state.get("variant_b", "")
    }
    return state

def _select_best_variant(self, state: BlogState) -> BlogState:
    """Select variant with higher score."""
    scores = state.get("variant_scores", {})

    if scores.get("a", 0) > scores.get("b", 0):
        state["intro"] = state["variants"]["a"]
        state["selected_variant"] = "a"
    else:
        state["intro"] = state["variants"]["b"]
        state["selected_variant"] = "b"

    return state
```

---

## Best Practices

1. **Keep Nodes Focused**: Each node should have a single responsibility
2. **State Immutability**: Return new state instead of mutating
3. **Error Handling**: Add try/except in nodes with fallback logic
4. **Checkpointing**: Use for long-running workflows
5. **Streaming**: Use appropriate stream mode for your use case
6. **Testing**: Test graph structure separately from node logic

---

## Troubleshooting

### Graph Won't Compile

**Problem**: Compilation errors

**Solution**:
1. Check all edges connect valid nodes
2. Ensure entry point is set
3. Verify all paths reach END

### Parallel Execution Not Working

**Problem**: Nodes run sequentially

**Solution**:
- Use multiple `set_entry_point()` calls
- Remove unnecessary dependencies between nodes
- Check if nodes share stateful resources

### Checkpointing Not Persisting

**Problem**: State lost between runs

**Solution**:
1. Ensure same `thread_id` is used
2. Check checkpointer is passed to `compile()`
3. Verify checkpoint storage is accessible

---

## Next Steps

- Review [extending-agents.md](extending-agents.md) for creating agents compatible with these patterns
- See [custom-evaluation.md](custom-evaluation.md) for evaluation nodes
- Explore official LangGraph docs: https://python.langchain.com/docs/langgraph
