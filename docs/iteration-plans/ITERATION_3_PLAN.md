# Iteration 3: True Hierarchical Architecture - Implementation Plan

**Goal**: Transform HVAS Mini from peer-to-peer pipeline into true hierarchical system with bidirectional flow and closed-loop refinement.

**Status**: PLANNING
**Prerequisites**: M1-M4 complete ✅
**Timeline**: 11-16 days

---

## Executive Summary

### What We're Building

A **3-layer hierarchical agent system** that demonstrates:

1. **Downward Flow**: Intent and context flow from parent to child with semantic filtering
2. **Upward Flow**: Results flow back up through aggregation and abstraction
3. **Closed-Loop**: Top-level critiques outputs and triggers revisions
4. **Semantic Distance**: Context strength based on vector space distance
5. **Multi-Pass Refinement**: Recursive improvement until quality threshold

### Why This Matters

Current HVAS Mini is a sophisticated **peer-to-peer pipeline**. This transforms it into a **hierarchical cognitive system** that mirrors:
- Biological nervous systems (bottom-up perception, top-down prediction)
- Deep learning architectures (forward pass, backward pass, iterative refinement)
- Organizational intelligence (delegation down, reporting up, feedback loops)

### Architecture Comparison

**Before (Iteration 2)**:
```
Intro → [Body ∥ Conclusion] → Evaluate → Output
  ↕       ↕↕                    ↓
(trust) (trust)              (one-shot)
```

**After (Iteration 3)**:
```
                Coordinator ← critique
                 ↓ context ↑ results
            [Content Agents] ← revise
                 ↓ tasks ↑ results
            [Specialists] ← focus
                   ↓
               [output]
            (multi-pass until quality met)
```

---

## Milestones Overview

| ID | Milestone | Priority | Duration | Dependencies | Tests |
|----|-----------|----------|----------|--------------|-------|
| M6 | Hierarchical Structure | CRITICAL | 3-4 days | None | 8 |
| M7 | Bidirectional Flow | CRITICAL | 3-4 days | M6 | 12 |
| M8 | Closed-Loop Refinement | HIGH | 2-3 days | M6, M7 | 10 |
| M9 | Semantic Distance | HIGH | 2-3 days | M6 | 8 |
| M10 | Visualization | MEDIUM | 1-2 days | M6-M9 | 5 |

**Total**: 11-16 days, 43 tests

---

## M6: Hierarchical Structure Foundation

### Objective
Establish 3-layer hierarchy with parent-child relationships and layer-based execution.

### Architecture

```
Layer 1 (Orchestration):
  ├─ CoordinatorAgent
  │    Role: Parse intent, distribute context, integrate results, critique
  │    Children: intro, body, conclusion
  │    Output: Final integrated blog post

Layer 2 (Content):
  ├─ IntroAgent (existing, enhanced)
  │    Role: Write engaging introduction
  │    Children: researcher, stylist
  │    Output: Introduction + confidence
  │
  ├─ BodyAgent (existing, enhanced)
  │    Role: Write informative body
  │    Children: researcher, fact_checker
  │    Output: Body + confidence
  │
  └─ ConclusionAgent (existing, enhanced)
       Role: Write synthesis conclusion
       Children: stylist
       Output: Conclusion + confidence

Layer 3 (Specialists):
  ├─ ResearchAgent
  │    Role: Find relevant information
  │    Output: Research findings + confidence
  │
  ├─ FactCheckerAgent
  │    Role: Verify accuracy
  │    Output: Fact-check results + confidence
  │
  └─ StyleAgent
       Role: Enhance tone and style
       Output: Style suggestions + confidence
```

### Deliverables

#### 1. AgentHierarchy Configuration
```python
# src/hvas_mini/hierarchy/structure.py

from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class AgentNode:
    """Node in agent hierarchy."""
    role: str
    layer: int
    children: List[str]
    semantic_vector: List[float]

class AgentHierarchy:
    """Defines hierarchical relationships between agents."""

    def __init__(self):
        self.nodes = {
            # Layer 1
            "coordinator": AgentNode(
                role="coordinator",
                layer=1,
                children=["intro", "body", "conclusion"],
                semantic_vector=[0.0, 1.0, 0.0]  # Integration-focused
            ),

            # Layer 2
            "intro": AgentNode(
                role="intro",
                layer=2,
                children=["researcher", "stylist"],
                semantic_vector=[0.8, 0.5, 0.2]  # Engaging
            ),
            "body": AgentNode(
                role="body",
                layer=2,
                children=["researcher", "fact_checker"],
                semantic_vector=[0.5, 0.8, 0.9]  # Informative
            ),
            "conclusion": AgentNode(
                role="conclusion",
                layer=2,
                children=["stylist"],
                semantic_vector=[0.7, 0.6, 0.3]  # Synthesis
            ),

            # Layer 3
            "researcher": AgentNode(
                role="researcher",
                layer=3,
                children=[],
                semantic_vector=[0.3, 0.9, 1.0]  # Factual
            ),
            "fact_checker": AgentNode(
                role="fact_checker",
                layer=3,
                children=[],
                semantic_vector=[0.2, 0.8, 0.9]  # Accuracy
            ),
            "stylist": AgentNode(
                role="stylist",
                layer=3,
                children=[],
                semantic_vector=[0.9, 0.4, 0.2]  # Style
            ),
        }

    def get_children(self, agent_role: str) -> List[str]:
        """Get direct children of an agent."""
        return self.nodes[agent_role].children

    def get_parent(self, agent_role: str) -> str | None:
        """Get parent of an agent."""
        for role, node in self.nodes.items():
            if agent_role in node.children:
                return role
        return None

    def get_layer(self, agent_role: str) -> int:
        """Get layer number for agent."""
        return self.nodes[agent_role].layer

    def get_layer_agents(self, layer: int) -> List[str]:
        """Get all agents in a layer."""
        return [role for role, node in self.nodes.items() if node.layer == layer]
```

#### 2. HierarchicalState Extension
```python
# src/hvas_mini/state.py

from typing import TypedDict, Dict, List

class AgentOutput(TypedDict):
    """Structured output from an agent."""
    content: str
    confidence: float  # 0.0 - 1.0
    metadata: Dict

class HierarchicalState(TypedDict):
    """Extended state for hierarchical execution."""

    # Existing BlogState fields
    topic: str
    intro: str
    body: str
    conclusion: str
    scores: Dict[str, float]
    # ... (all existing fields)

    # NEW: Hierarchical execution
    hierarchy: AgentHierarchy
    current_layer: int
    current_pass: int
    max_passes: int

    # NEW: Layer outputs
    layer_outputs: Dict[int, Dict[str, AgentOutput]]
    # {1: {"coordinator": {...}}, 2: {"intro": {...}, ...}, ...}

    # NEW: Coordinator state
    coordinator_intent: str  # Parsed high-level intent
    coordinator_critique: Dict[str, str]  # {agent: feedback}
    revision_requested: bool
    quality_threshold_met: bool

    # NEW: Pass tracking
    pass_history: List[Dict]  # [{pass: 1, scores: {...}, ...}, ...]
```

#### 3. CoordinatorAgent
```python
# src/hvas_mini/hierarchy/coordinator.py

from hvas_mini.agents import BaseAgent
from typing import Dict, List

class CoordinatorAgent(BaseAgent):
    """Top-level orchestrator agent (Layer 1)."""

    def __init__(self, hierarchy: AgentHierarchy, *args, **kwargs):
        super().__init__(role="coordinator", *args, **kwargs)
        self.hierarchy = hierarchy

    @property
    def content_key(self) -> str:
        return "coordinator_output"

    async def generate_content(
        self, state: HierarchicalState, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Parse intent and set high-level goals."""

        prompt = f"""You are a high-level coordinator for a blog writing system.

Topic: {state['topic']}

Your role:
1. Parse the intent of this topic
2. Define what makes a good blog post on this topic
3. Set constraints and goals for the content agents

Provide a brief intent statement (2-3 sentences) capturing the essence of what
this blog post should accomplish.

Intent:"""

        response = await self.llm.ainvoke(prompt)
        return response.content

    def distribute_context(self, state: HierarchicalState) -> Dict[str, str]:
        """Create filtered context for each child."""
        contexts = {}

        for child_role in self.hierarchy.get_children(self.role):
            # For now, give full intent to all children
            # M9 will add semantic filtering
            contexts[child_role] = state["coordinator_intent"]

        return contexts

    def aggregate_results(
        self, state: HierarchicalState, layer: int
    ) -> AgentOutput:
        """Combine results from a layer."""

        outputs = state["layer_outputs"][layer]

        # Combine all content
        combined_content = "\n\n".join([
            f"## {role.title()}\n{output['content']}"
            for role, output in outputs.items()
        ])

        # Average confidence
        avg_confidence = sum(
            output['confidence'] for output in outputs.values()
        ) / len(outputs)

        return AgentOutput(
            content=combined_content,
            confidence=avg_confidence,
            metadata={"sources": list(outputs.keys())}
        )

    def critique_outputs(self, state: HierarchicalState) -> Dict[str, str]:
        """Generate critique for each content agent."""
        critiques = {}

        for role in ["intro", "body", "conclusion"]:
            output = state["layer_outputs"][2][role]

            # Simple heuristic critique (could be LLM-based)
            issues = []

            if len(output['content']) < 100:
                issues.append("too short")

            if output['confidence'] < 0.7:
                issues.append("low confidence")

            if not issues:
                critiques[role] = "Good quality"
            else:
                critiques[role] = f"Issues: {', '.join(issues)}"

        return critiques
```

#### 4. Specialist Agents
```python
# src/hvas_mini/hierarchy/specialists.py

from hvas_mini.agents import BaseAgent

class ResearchAgent(BaseAgent):
    """Finds relevant information for content (Layer 3)."""

    @property
    def content_key(self) -> str:
        return "research"

    async def generate_content(self, state, memories, weighted_context="") -> str:
        prompt = f"""Research task for: {state['topic']}

Context from parent: {weighted_context}

Find 2-3 key facts or insights relevant to this topic.

Research findings:"""

        response = await self.llm.ainvoke(prompt)
        return response.content


class FactCheckerAgent(BaseAgent):
    """Verifies accuracy of content (Layer 3)."""

    @property
    def content_key(self) -> str:
        return "fact_check"

    async def generate_content(self, state, memories, weighted_context="") -> str:
        prompt = f"""Fact-check task for: {state['topic']}

Content to verify: {weighted_context}

Identify any claims that need verification and rate confidence.

Fact-check results:"""

        response = await self.llm.ainvoke(prompt)
        return response.content


class StyleAgent(BaseAgent):
    """Enhances style and tone (Layer 3)."""

    @property
    def content_key(self) -> str:
        return "style"

    async def generate_content(self, state, memories, weighted_context="") -> str:
        prompt = f"""Style enhancement task for: {state['topic']}

Content: {weighted_context}

Suggest improvements to tone, flow, and engagement.

Style suggestions:"""

        response = await self.llm.ainvoke(prompt)
        return response.content
```

### Tests

```python
# test_hierarchy_structure.py

def test_hierarchy_definition():
    """Hierarchy should define 3 layers."""
    hierarchy = AgentHierarchy()

    assert hierarchy.get_layer("coordinator") == 1
    assert hierarchy.get_layer("intro") == 2
    assert hierarchy.get_layer("researcher") == 3


def test_parent_child_relationships():
    """Parents should know their children."""
    hierarchy = AgentHierarchy()

    assert "intro" in hierarchy.get_children("coordinator")
    assert "researcher" in hierarchy.get_children("intro")
    assert hierarchy.get_parent("body") == "coordinator"


def test_layer_agents():
    """Should retrieve all agents in a layer."""
    hierarchy = AgentHierarchy()

    layer_2 = hierarchy.get_layer_agents(2)
    assert set(layer_2) == {"intro", "body", "conclusion"}


def test_coordinator_creation():
    """CoordinatorAgent should initialize."""
    hierarchy = AgentHierarchy()
    memory = MemoryManager("coordinator_memories")

    coordinator = CoordinatorAgent(hierarchy, memory)

    assert coordinator.role == "coordinator"
    assert coordinator.hierarchy is not None
```

---

## M7: Bidirectional Flow

### Objective
Implement downward context distribution and upward result aggregation with confidence scoring.

### Deliverables

#### 1. Downward Context Distribution
```python
# src/hvas_mini/hierarchy/flow.py

class HierarchicalExecutor:
    """Executes hierarchical workflow with bidirectional flow."""

    async def execute_downward(
        self, state: HierarchicalState, layer: int
    ) -> HierarchicalState:
        """Execute all agents in a layer with parent context."""

        agents_in_layer = state["hierarchy"].get_layer_agents(layer)

        for agent_role in agents_in_layer:
            # Get parent
            parent_role = state["hierarchy"].get_parent(agent_role)

            if parent_role:
                # Get filtered context from parent
                parent_output = state["layer_outputs"][layer - 1][parent_role]
                context = parent_output["content"]
            else:
                # Top layer uses original topic
                context = state["topic"]

            # Execute agent with context
            agent = self.agents[agent_role]
            output = await agent(state, context)

            # Store structured output
            state["layer_outputs"][layer][agent_role] = AgentOutput(
                content=output,
                confidence=self._estimate_confidence(output),
                metadata={"parent": parent_role, "layer": layer}
            )

        return state

    def _estimate_confidence(self, content: str) -> float:
        """Estimate confidence in output quality."""
        # Simple heuristic based on length and structure
        length_score = min(len(content) / 500, 1.0)
        return length_score  # Can be more sophisticated
```

#### 2. Upward Result Aggregation
```python
async def execute_upward(
    self, state: HierarchicalState, layer: int
) -> HierarchicalState:
    """Aggregate results from children and pass to parents."""

    agents_in_layer = state["hierarchy"].get_layer_agents(layer)

    for agent_role in agents_in_layer:
        # Get all children
        children = state["hierarchy"].get_children(agent_role)

        if not children:
            # Leaf node, no aggregation needed
            continue

        # Gather child outputs
        child_outputs = []
        for child_role in children:
            child_output = state["layer_outputs"][layer + 1][child_role]
            child_outputs.append(child_output)

        # Aggregate
        aggregated = self._aggregate_outputs(child_outputs, agent_role)

        # Update agent's output with aggregated information
        state["layer_outputs"][layer][agent_role]["metadata"]["aggregated_children"] = aggregated

    return state

def _aggregate_outputs(
    self, outputs: List[AgentOutput], parent_role: str
) -> Dict:
    """Combine multiple child outputs."""

    # Weight by confidence
    weighted_content = []
    total_confidence = sum(o["confidence"] for o in outputs)

    for output in outputs:
        weight = output["confidence"] / total_confidence if total_confidence > 0 else 1.0 / len(outputs)
        weighted_content.append({
            "content": output["content"],
            "weight": weight,
            "confidence": output["confidence"]
        })

    return {
        "combined_content": "\n\n".join([o["content"] for o in outputs]),
        "weighted_confidence": sum(
            o["confidence"] * (o["confidence"] / total_confidence)
            for o in outputs
        ) if total_confidence > 0 else 0.5,
        "sources": [o["metadata"] for o in outputs]
    }
```

### Tests

```python
def test_downward_context_filtering():
    """Children should receive filtered context from parent."""
    # Test that context passed down is appropriate


def test_upward_aggregation():
    """Parent should aggregate child results."""
    # Test that multiple child outputs are combined


def test_confidence_propagation():
    """Confidence should flow upward correctly."""
    # Test that parent confidence reflects children
```

---

## M8: Closed-Loop Refinement

### Objective
Enable multi-pass execution with coordinator critique and revision requests.

### Deliverables

#### 1. Critique Generation
```python
async def critique_and_decide(self, state: HierarchicalState) -> bool:
    """Critique outputs and decide if revision needed."""

    # Generate critiques
    critiques = self.coordinator.critique_outputs(state)
    state["coordinator_critique"] = critiques

    # Check if quality threshold met
    avg_score = sum(
        state["layer_outputs"][2][role]["confidence"]
        for role in ["intro", "body", "conclusion"]
    ) / 3

    threshold = 0.8  # Configurable

    if avg_score >= threshold:
        state["quality_threshold_met"] = True
        return False  # No revision needed

    # Check if more passes allowed
    if state["current_pass"] >= state["max_passes"]:
        return False  # No more passes

    return True  # Revision needed
```

#### 2. Revision Request Mechanism
```python
async def request_revision(
    self, state: HierarchicalState
) -> HierarchicalState:
    """Generate revision feedback and prepare for next pass."""

    for role in ["intro", "body", "conclusion"]:
        critique = state["coordinator_critique"][role]

        # Generate specific revision instruction
        revision_prompt = f"""The {role} needs improvement: {critique}

Original: {state["layer_outputs"][2][role]["content"]}

Please revise to address these issues."""

        # Store revision request
        state["coordinator_critique"][role] = revision_prompt

    state["revision_requested"] = True
    state["current_pass"] += 1

    return state
```

#### 3. Multi-Pass Execution Loop
```python
async def execute_with_refinement(
    self, state: HierarchicalState
) -> HierarchicalState:
    """Execute with closed-loop refinement."""

    for pass_num in range(1, state["max_passes"] + 1):
        state["current_pass"] = pass_num

        # Execute all layers (downward then upward)
        for layer in [1, 2, 3]:
            state = await self.execute_downward(state, layer)

        for layer in [3, 2, 1]:
            state = await self.execute_upward(state, layer)

        # Record pass
        state["pass_history"].append({
            "pass": pass_num,
            "scores": {
                role: state["layer_outputs"][2][role]["confidence"]
                for role in ["intro", "body", "conclusion"]
            }
        })

        # Critique and decide
        needs_revision = await self.critique_and_decide(state)

        if not needs_revision:
            break

        # Request revision for next pass
        state = await self.request_revision(state)

    return state
```

### Tests

```python
def test_quality_threshold_detection():
    """Should detect when quality threshold is met."""


def test_revision_improves_quality():
    """Second pass should improve quality."""


def test_max_passes_limit():
    """Should stop at max passes."""


def test_early_exit_on_quality():
    """Should exit early if quality threshold met."""
```

---

## M9: Semantic Distance Weighting

### Objective
Implement vector-based semantic distance and use it to weight context flow.

### Deliverables

#### 1. Semantic Distance Calculation
```python
import numpy as np

def compute_semantic_distance(
    hierarchy: AgentHierarchy, agent_a: str, agent_b: str
) -> float:
    """Compute cosine distance between agent role vectors."""

    vec_a = np.array(hierarchy.nodes[agent_a].semantic_vector)
    vec_b = np.array(hierarchy.nodes[agent_b].semantic_vector)

    # Cosine similarity
    similarity = np.dot(vec_a, vec_b) / (
        np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    )

    # Convert to distance [0, 1]
    distance = (1.0 - similarity) / 2.0  # Normalize to [0, 1]

    return distance
```

#### 2. Distance-Based Context Filtering
```python
def filter_context_by_distance(
    context: str, distance: float, min_ratio: float = 0.3
) -> str:
    """Filter context based on semantic distance."""

    # Context strength inversely related to distance
    # distance=0.0 → strength=1.0 (full context)
    # distance=1.0 → strength=min_ratio (minimal context)
    strength = 1.0 - (distance * (1.0 - min_ratio))

    # Filter by keeping top strength% of content
    sentences = context.split(". ")
    keep_count = max(1, int(len(sentences) * strength))

    filtered = ". ".join(sentences[:keep_count])

    return filtered
```

### Tests

```python
def test_distance_calculation():
    """Similar agents should have small distance."""


def test_context_filtering():
    """Distant agents should receive filtered context."""


def test_distance_affects_flow():
    """Distance should impact context strength."""
```

---

## M10: Visualization

### Objective
Visualize hierarchical structure, bidirectional flow, and refinement cycles.

### Deliverables

1. Hierarchy tree diagram
2. Flow animation (down/up arrows)
3. Pass-by-pass comparison
4. Distance matrix heatmap

---

## Demonstration Scenarios

### Scenario 1: Happy Path (Single Pass)
```
Topic: "Machine Learning Basics"

Pass 1:
  Coordinator: "Explain ML concepts clearly for beginners"
  ↓ (filtered context)
  IntroAgent + [researcher, stylist]: confidence=0.9
  BodyAgent + [researcher, fact_checker]: confidence=0.85
  ConclusionAgent + [stylist]: confidence=0.88
  ↑ (aggregated)
  Coordinator: avg_confidence=0.88 > threshold(0.8)
  → Quality met, output

Result: Single-pass success, no revision needed
```

### Scenario 2: Revision Loop
```
Topic: "Quantum Entanglement"

Pass 1:
  Coordinator: "Explain quantum entanglement"
  ↓
  Agents: confidence=0.65 (too technical)
  ↑
  Coordinator: "Needs simplification"
  → Revision requested

Pass 2:
  Coordinator: "Use analogies, simplify terms"
  ↓ (with feedback)
  Agents: confidence=0.82 (improved)
  ↑
  Coordinator: threshold met
  → Output

Result: Two passes, quality improved +17%
```

### Scenario 3: Distance Filtering
```
ResearchAgent distance from Coordinator = 0.7

Coordinator context (100 sentences)
  ↓ filter_by_distance(0.7)
BodyAgent (55 sentences)
  ↓ filter_by_distance(0.7)
ResearchAgent (30 sentences)

Result: Context appropriately filtered by distance
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Hierarchy Depth | 3 layers | Layer count |
| Pass Convergence | +15% quality/pass | Score delta |
| Context Filtering | Child < 70% Parent | Token count |
| Aggregation Quality | Combined > Individual | Score comparison |
| Revision Effectiveness | Pass2 > Pass1 | Confidence delta |
| Distance Correlation | r > 0.7 | Context vs distance |

---

## Next Steps

1. **Review this plan** - Ensure it captures the hierarchical principles
2. **Create M6 branch** - Start with hierarchical structure
3. **Implement incrementally** - One milestone at a time
4. **Test thoroughly** - Each milestone has test suite
5. **Demonstrate clearly** - Show bidirectional flow and refinement

This plan transforms HVAS Mini into a true hierarchical cognitive system!
