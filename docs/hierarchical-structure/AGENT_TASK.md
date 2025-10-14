# Agent Task: M6 - Hierarchical Structure Foundation

## Branch: `feature/hierarchical-structure`

## Priority: CRITICAL

## Execution: SEQUENTIAL (Blocks M7, M8, M9)

## Objective

Establish a **3-layer hierarchical agent system** with parent-child relationships and layer-based execution. Transform HVAS Mini from a flat peer-to-peer pipeline into a true hierarchy where:

- **Layer 1** (Coordinator): Orchestrates, parses intent, critiques
- **Layer 2** (Content): Creates blog sections (intro, body, conclusion)
- **Layer 3** (Specialists): Provides focused sub-tasks (research, fact-check, style)

This milestone provides the **foundational structure** for bidirectional flow (M7), closed-loop refinement (M8), and semantic distance weighting (M9).

---

## Dependencies

- ✅ M1-M4 merged to master
- ✅ Current codebase on master branch

---

## Architecture Overview

### Current Structure (Iteration 2)
```
Sequential Pipeline with Peer Trust:

IntroAgent ────────────────────┐
                               ↓
BodyAgent ──→ (trust weighted)─┤
                               ├──→ Evaluator → Evolution
ConclusionAgent ──→ (trust)────┘
```

### Target Structure (M6)
```
3-Layer Hierarchy:

┌─────────────────────────┐
│   CoordinatorAgent      │  ← Layer 1: Orchestration
│   (parses intent)       │
└───────────┬─────────────┘
            │
    ┌───────┼────────┐
    ↓       ↓        ↓
┌────────┐ ┌────────┐ ┌────────┐
│ Intro  │ │  Body  │ │Concl.  │  ← Layer 2: Content
└───┬────┘ └───┬────┘ └───┬────┘
    │          │          │
    ↓          ↓          ↓
[research]  [research] [style]
[style]     [fact_check]       ← Layer 3: Specialists
```

---

## Tasks

### 1. Create AgentHierarchy Configuration Class

**File**: `src/hvas_mini/hierarchy/structure.py`

**Purpose**: Define the 3-layer structure and relationships.

**Implementation**:

```python
"""
Agent hierarchy structure definition.

Defines parent-child relationships and layer organization.
"""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AgentNode:
    """Single node in agent hierarchy."""
    role: str
    layer: int  # 1=coordinator, 2=content, 3=specialist
    children: List[str]  # Direct child roles
    semantic_vector: List[float]  # For M9 semantic distance

class AgentHierarchy:
    """Defines hierarchical relationships between agents."""

    def __init__(self):
        """Initialize 3-layer hierarchy."""
        self.nodes = {
            # Layer 1: Orchestration
            "coordinator": AgentNode(
                role="coordinator",
                layer=1,
                children=["intro", "body", "conclusion"],
                semantic_vector=[0.0, 1.0, 0.0]  # Integration-focused
            ),

            # Layer 2: Content Agents
            "intro": AgentNode(
                role="intro",
                layer=2,
                children=["researcher", "stylist"],
                semantic_vector=[0.8, 0.5, 0.2]  # Engaging, hook
            ),
            "body": AgentNode(
                role="body",
                layer=2,
                children=["researcher", "fact_checker"],
                semantic_vector=[0.5, 0.8, 0.9]  # Content-heavy
            ),
            "conclusion": AgentNode(
                role="conclusion",
                layer=2,
                children=["stylist"],
                semantic_vector=[0.7, 0.6, 0.3]  # Synthesis
            ),

            # Layer 3: Specialists
            "researcher": AgentNode(
                role="researcher",
                layer=3,
                children=[],  # Leaf node
                semantic_vector=[0.3, 0.9, 1.0]  # Factual, deep
            ),
            "fact_checker": AgentNode(
                role="fact_checker",
                layer=3,
                children=[],
                semantic_vector=[0.2, 0.8, 0.9]  # Accuracy-focused
            ),
            "stylist": AgentNode(
                role="stylist",
                layer=3,
                children=[],
                semantic_vector=[0.9, 0.4, 0.2]  # Style, tone
            ),
        }

    def get_children(self, agent_role: str) -> List[str]:
        """Get direct children of an agent.

        Args:
            agent_role: Parent agent role

        Returns:
            List of child agent roles
        """
        if agent_role not in self.nodes:
            return []
        return self.nodes[agent_role].children.copy()

    def get_parent(self, agent_role: str) -> str | None:
        """Get parent of an agent.

        Args:
            agent_role: Child agent role

        Returns:
            Parent role or None if top-level
        """
        for role, node in self.nodes.items():
            if agent_role in node.children:
                return role
        return None

    def get_layer(self, agent_role: str) -> int:
        """Get layer number for agent.

        Args:
            agent_role: Agent role

        Returns:
            Layer number (1-3)
        """
        if agent_role not in self.nodes:
            raise ValueError(f"Unknown agent: {agent_role}")
        return self.nodes[agent_role].layer

    def get_layer_agents(self, layer: int) -> List[str]:
        """Get all agents in a layer.

        Args:
            layer: Layer number (1-3)

        Returns:
            List of agent roles in that layer
        """
        return [
            role for role, node in self.nodes.items()
            if node.layer == layer
        ]

    def get_siblings(self, agent_role: str) -> List[str]:
        """Get sibling agents (same parent).

        Args:
            agent_role: Agent role

        Returns:
            List of sibling roles
        """
        parent = self.get_parent(agent_role)
        if not parent:
            return []

        siblings = self.get_children(parent)
        return [s for s in siblings if s != agent_role]

    def is_ancestor(self, potential_ancestor: str, agent_role: str) -> bool:
        """Check if one agent is an ancestor of another.

        Args:
            potential_ancestor: Potential ancestor role
            agent_role: Agent to check

        Returns:
            True if ancestor relationship exists
        """
        current = agent_role
        while current:
            parent = self.get_parent(current)
            if parent == potential_ancestor:
                return True
            current = parent
        return False

    def get_all_descendants(self, agent_role: str) -> List[str]:
        """Get all descendants (children, grandchildren, etc.).

        Args:
            agent_role: Root agent

        Returns:
            List of all descendant roles
        """
        descendants = []
        children = self.get_children(agent_role)

        for child in children:
            descendants.append(child)
            descendants.extend(self.get_all_descendants(child))

        return descendants
```

**Acceptance Criteria**:
- ✅ 3 layers defined (coordinator=1, content=2, specialists=3)
- ✅ Parent-child relationships correct
- ✅ Methods for traversal (get_parent, get_children)
- ✅ Layer queries work

---

### 2. Extend BlogState for Hierarchical Execution

**File**: `src/hvas_mini/state.py`

**Purpose**: Add fields to track hierarchical execution state.

**Changes**:

```python
from typing import TypedDict, Dict, List

# NEW: Import hierarchy
try:
    from hvas_mini.hierarchy.structure import AgentHierarchy
except ImportError:
    AgentHierarchy = None

# NEW: Structured agent output
class AgentOutput(TypedDict):
    """Output from an agent with metadata."""
    content: str
    confidence: float  # 0.0-1.0
    metadata: Dict  # Additional context

class HierarchicalState(BlogState):
    """Extended state for hierarchical execution."""

    # NEW: Hierarchy instance
    hierarchy: AgentHierarchy

    # NEW: Execution tracking
    current_layer: int  # Which layer is executing
    current_pass: int  # Which refinement pass (M8)
    max_passes: int  # Maximum refinement passes

    # NEW: Layer outputs
    layer_outputs: Dict[int, Dict[str, AgentOutput]]
    # {1: {"coordinator": AgentOutput}, 2: {"intro": AgentOutput, ...}, ...}

    # NEW: Coordinator state
    coordinator_intent: str  # High-level parsed intent
    coordinator_critique: Dict[str, str]  # {agent: feedback}
    revision_requested: bool
    quality_threshold_met: bool

    # NEW: Pass tracking
    pass_history: List[Dict]
    # [{pass: 1, scores: {...}, outputs: {...}}, ...]


def create_hierarchical_state(topic: str) -> HierarchicalState:
    """Create initial hierarchical state.

    Args:
        topic: Blog topic

    Returns:
        Initialized HierarchicalState
    """
    base_state = create_initial_state(topic)

    hierarchical_fields = {
        "hierarchy": AgentHierarchy() if AgentHierarchy else None,
        "current_layer": 0,
        "current_pass": 1,
        "max_passes": 3,
        "layer_outputs": {1: {}, 2: {}, 3: {}},
        "coordinator_intent": "",
        "coordinator_critique": {},
        "revision_requested": False,
        "quality_threshold_met": False,
        "pass_history": [],
    }

    return {**base_state, **hierarchical_fields}
```

**Acceptance Criteria**:
- ✅ HierarchicalState extends BlogState
- ✅ layer_outputs dictionary structure correct
- ✅ AgentOutput TypedDict defined
- ✅ create_hierarchical_state() works

---

### 3. Implement CoordinatorAgent (Layer 1)

**File**: `src/hvas_mini/hierarchy/coordinator.py`

**Purpose**: Top-level orchestrator that parses intent and critiques results.

**Implementation**:

```python
"""
Coordinator agent - top-level orchestrator (Layer 1).
"""

from hvas_mini.agents import BaseAgent
from hvas_mini.hierarchy.structure import AgentHierarchy
from typing import Dict, List

class CoordinatorAgent(BaseAgent):
    """Top-level orchestrator agent (Layer 1).

    Responsibilities:
    - Parse user intent from topic
    - Define high-level goals
    - Distribute context to content agents
    - Aggregate results from content agents
    - Critique outputs (used in M8)
    """

    def __init__(self, hierarchy: AgentHierarchy, memory_manager, trust_manager=None):
        """Initialize coordinator.

        Args:
            hierarchy: AgentHierarchy instance
            memory_manager: MemoryManager for coordinator
            trust_manager: Optional TrustManager
        """
        super().__init__(role="coordinator", memory_manager=memory_manager, trust_manager=trust_manager)
        self.hierarchy = hierarchy

    @property
    def content_key(self) -> str:
        """State key for coordinator output."""
        return "coordinator_output"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Parse intent and set high-level goals.

        Args:
            state: Current HierarchicalState
            memories: Retrieved memories
            weighted_context: Context from peers (unused at Layer 1)

        Returns:
            Intent statement
        """
        topic = state["topic"]

        prompt = f"""You are a high-level coordinator for a blog writing system.

Topic: {topic}

Your role:
1. Parse the intent of this topic
2. Define what makes a good blog post on this topic
3. Set high-level constraints and goals

Provide a brief intent statement (2-3 sentences) that captures:
- The core message to convey
- The target audience
- Key points to cover

Intent:"""

        response = await self.llm.ainvoke(prompt)
        return response.content

    def distribute_context(self, state) -> Dict[str, str]:
        """Create context for each direct child.

        For M6, this gives full intent to all children.
        M9 will add semantic distance filtering.

        Args:
            state: Current HierarchicalState

        Returns:
            {child_role: context_for_child}
        """
        contexts = {}
        intent = state["coordinator_intent"]

        for child_role in self.hierarchy.get_children(self.role):
            # Full intent for now (M9 will filter)
            contexts[child_role] = intent

        return contexts

    def aggregate_results(self, state, layer: int) -> AgentOutput:
        """Combine results from a layer.

        Args:
            state: Current HierarchicalState
            layer: Layer number to aggregate

        Returns:
            Aggregated AgentOutput
        """
        outputs = state["layer_outputs"][layer]

        if not outputs:
            return AgentOutput(
                content="No outputs to aggregate",
                confidence=0.0,
                metadata={}
            )

        # Combine all content
        combined_parts = []
        for role, output in outputs.items():
            combined_parts.append(f"## {role.title()}\n{output['content']}")

        combined_content = "\n\n".join(combined_parts)

        # Average confidence
        avg_confidence = sum(
            output['confidence'] for output in outputs.values()
        ) / len(outputs)

        return AgentOutput(
            content=combined_content,
            confidence=avg_confidence,
            metadata={
                "sources": list(outputs.keys()),
                "layer": layer
            }
        )

    def critique_outputs(self, state) -> Dict[str, str]:
        """Generate critique for each content agent.

        Used in M8 for closed-loop refinement.

        Args:
            state: Current HierarchicalState

        Returns:
            {agent_role: critique_message}
        """
        critiques = {}

        for role in ["intro", "body", "conclusion"]:
            if role not in state["layer_outputs"][2]:
                critiques[role] = "No output to critique"
                continue

            output = state["layer_outputs"][2][role]

            # Simple heuristic critique (can be LLM-based in future)
            issues = []

            if len(output['content']) < 100:
                issues.append("too short (< 100 chars)")

            if output['confidence'] < 0.7:
                issues.append(f"low confidence ({output['confidence']:.2f})")

            if not issues:
                critiques[role] = "Good quality"
            else:
                critiques[role] = f"Issues: {', '.join(issues)}"

        return critiques
```

**Acceptance Criteria**:
- ✅ Inherits from BaseAgent
- ✅ Parses intent from topic
- ✅ distribute_context() creates context for children
- ✅ aggregate_results() combines layer outputs
- ✅ critique_outputs() generates feedback

---

### 4. Implement Specialist Agents (Layer 3)

**File**: `src/hvas_mini/hierarchy/specialists.py`

**Purpose**: Focused agents that provide specialized services to content agents.

**Implementation**:

```python
"""
Specialist agents - Layer 3 focused sub-task agents.
"""

from hvas_mini.agents import BaseAgent
from typing import List, Dict

class ResearchAgent(BaseAgent):
    """Finds relevant information and facts (Layer 3)."""

    def __init__(self, memory_manager, trust_manager=None):
        super().__init__(role="researcher", memory_manager=memory_manager, trust_manager=trust_manager)

    @property
    def content_key(self) -> str:
        return "research"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Find relevant research for the topic.

        Args:
            state: Current state
            memories: Retrieved memories
            weighted_context: Context from parent

        Returns:
            Research findings
        """
        topic = state.get("topic", "")

        prompt = f"""You are a research specialist for blog content.

Topic: {topic}

Context from parent: {weighted_context}

Find 2-3 key facts, statistics, or insights relevant to this topic.
Focus on accurate, useful information.

Research findings:"""

        response = await self.llm.ainvoke(prompt)
        return response.content


class FactCheckerAgent(BaseAgent):
    """Verifies accuracy and flags issues (Layer 3)."""

    def __init__(self, memory_manager, trust_manager=None):
        super().__init__(role="fact_checker", memory_manager=memory_manager, trust_manager=trust_manager)

    @property
    def content_key(self) -> str:
        return "fact_check"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Verify content accuracy.

        Args:
            state: Current state
            memories: Retrieved memories
            weighted_context: Content to fact-check

        Returns:
            Fact-check results
        """
        topic = state.get("topic", "")

        prompt = f"""You are a fact-checking specialist.

Topic: {topic}

Content to verify: {weighted_context}

Review for:
1. Factual accuracy
2. Claims that need verification
3. Potential inaccuracies

Provide fact-check feedback:"""

        response = await self.llm.ainvoke(prompt)
        return response.content


class StyleAgent(BaseAgent):
    """Enhances tone, flow, and engagement (Layer 3)."""

    def __init__(self, memory_manager, trust_manager=None):
        super().__init__(role="stylist", memory_manager=memory_manager, trust_manager=trust_manager)

    @property
    def content_key(self) -> str:
        return "style"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Enhance style and tone.

        Args:
            state: Current state
            memories: Retrieved memories
            weighted_context: Content to enhance

        Returns:
            Style suggestions
        """
        topic = state.get("topic", "")

        prompt = f"""You are a style enhancement specialist.

Topic: {topic}

Content: {weighted_context}

Suggest improvements to:
1. Tone and voice
2. Flow and transitions
3. Engagement and hooks

Style suggestions:"""

        response = await self.llm.ainvoke(prompt)
        return response.content
```

**Acceptance Criteria**:
- ✅ 3 specialist agents created
- ✅ Each inherits from BaseAgent
- ✅ Each has distinct focus (research, fact-check, style)
- ✅ Accepts parent context via weighted_context

---

### 5. Create Agent Factory

**File**: `src/hvas_mini/hierarchy/factory.py`

**Purpose**: Factory to create all hierarchical agents.

**Implementation**:

```python
"""
Factory for creating hierarchical agent instances.
"""

from typing import Dict
from hvas_mini.agents import BaseAgent, IntroAgent, BodyAgent, ConclusionAgent
from hvas_mini.hierarchy.coordinator import CoordinatorAgent
from hvas_mini.hierarchy.specialists import ResearchAgent, FactCheckerAgent, StyleAgent
from hvas_mini.hierarchy.structure import AgentHierarchy
from hvas_mini.memory import MemoryManager

def create_hierarchical_agents(
    persist_directory: str = "./data/memories",
    trust_manager=None
) -> tuple[Dict[str, BaseAgent], AgentHierarchy]:
    """Create all agents in hierarchical structure.

    Args:
        persist_directory: Where to persist memories
        trust_manager: Optional TrustManager

    Returns:
        (agents_dict, hierarchy) tuple
    """
    hierarchy = AgentHierarchy()
    agents = {}

    # Layer 1: Coordinator
    coordinator_memory = MemoryManager(
        collection_name="coordinator_memories",
        persist_directory=persist_directory
    )
    agents["coordinator"] = CoordinatorAgent(
        hierarchy=hierarchy,
        memory_manager=coordinator_memory,
        trust_manager=trust_manager
    )

    # Layer 2: Content Agents (existing, but now aware of children)
    for role in ["intro", "body", "conclusion"]:
        memory = MemoryManager(
            collection_name=f"{role}_memories",
            persist_directory=persist_directory
        )

        if role == "intro":
            agents[role] = IntroAgent(role, memory, trust_manager)
        elif role == "body":
            agents[role] = BodyAgent(role, memory, trust_manager)
        elif role == "conclusion":
            agents[role] = ConclusionAgent(role, memory, trust_manager)

    # Layer 3: Specialists
    specialist_roles = {
        "researcher": ResearchAgent,
        "fact_checker": FactCheckerAgent,
        "stylist": StyleAgent
    }

    for role, AgentClass in specialist_roles.items():
        memory = MemoryManager(
            collection_name=f"{role}_memories",
            persist_directory=persist_directory
        )
        agents[role] = AgentClass(
            memory_manager=memory,
            trust_manager=trust_manager
        )

    return agents, hierarchy
```

**Acceptance Criteria**:
- ✅ Creates all 7 agents (1 coordinator, 3 content, 3 specialist)
- ✅ Returns agents dict and hierarchy
- ✅ Each agent has appropriate MemoryManager

---

## Testing

**File**: `test_hierarchical_structure.py`

**Purpose**: Comprehensive tests for hierarchy structure.

```python
"""
Test suite for hierarchical structure (M6).
"""

import pytest
from hvas_mini.hierarchy.structure import AgentHierarchy, AgentNode
from hvas_mini.hierarchy.coordinator import CoordinatorAgent
from hvas_mini.hierarchy.specialists import ResearchAgent, FactCheckerAgent, StyleAgent
from hvas_mini.hierarchy.factory import create_hierarchical_agents
from hvas_mini.state import create_hierarchical_state, AgentOutput
from hvas_mini.memory import MemoryManager

class TestAgentHierarchy:
    """Test hierarchy structure and relationships."""

    def test_three_layers_defined(self):
        """Hierarchy should have 3 distinct layers."""
        hierarchy = AgentHierarchy()

        assert hierarchy.get_layer("coordinator") == 1
        assert hierarchy.get_layer("intro") == 2
        assert hierarchy.get_layer("body") == 2
        assert hierarchy.get_layer("conclusion") == 2
        assert hierarchy.get_layer("researcher") == 3
        assert hierarchy.get_layer("fact_checker") == 3
        assert hierarchy.get_layer("stylist") == 3

    def test_parent_child_relationships(self):
        """Parent-child relationships should be correct."""
        hierarchy = AgentHierarchy()

        # Coordinator children
        assert set(hierarchy.get_children("coordinator")) == {"intro", "body", "conclusion"}

        # Content agent children
        assert set(hierarchy.get_children("intro")) == {"researcher", "stylist"}
        assert set(hierarchy.get_children("body")) == {"researcher", "fact_checker"}
        assert set(hierarchy.get_children("conclusion")) == {"stylist"}

        # Specialists have no children
        assert hierarchy.get_children("researcher") == []

    def test_get_parent(self):
        """Should correctly identify parent agents."""
        hierarchy = AgentHierarchy()

        assert hierarchy.get_parent("intro") == "coordinator"
        assert hierarchy.get_parent("body") == "coordinator"
        assert hierarchy.get_parent("researcher") == "intro" or hierarchy.get_parent("researcher") == "body"
        assert hierarchy.get_parent("coordinator") is None

    def test_get_layer_agents(self):
        """Should retrieve all agents in a layer."""
        hierarchy = AgentHierarchy()

        layer1 = hierarchy.get_layer_agents(1)
        assert layer1 == ["coordinator"]

        layer2 = hierarchy.get_layer_agents(2)
        assert set(layer2) == {"intro", "body", "conclusion"}

        layer3 = hierarchy.get_layer_agents(3)
        assert set(layer3) == {"researcher", "fact_checker", "stylist"}

    def test_get_siblings(self):
        """Should get sibling agents."""
        hierarchy = AgentHierarchy()

        intro_siblings = hierarchy.get_siblings("intro")
        assert set(intro_siblings) == {"body", "conclusion"}

    def test_is_ancestor(self):
        """Should detect ancestor relationships."""
        hierarchy = AgentHierarchy()

        assert hierarchy.is_ancestor("coordinator", "intro") is True
        assert hierarchy.is_ancestor("coordinator", "researcher") is True
        assert hierarchy.is_ancestor("intro", "researcher") is True
        assert hierarchy.is_ancestor("body", "intro") is False


class TestCoordinatorAgent:
    """Test coordinator agent functionality."""

    def test_coordinator_creation(self):
        """CoordinatorAgent should initialize."""
        hierarchy = AgentHierarchy()
        memory = MemoryManager("coordinator_test", persist_directory="./test_data")

        coordinator = CoordinatorAgent(hierarchy, memory)

        assert coordinator.role == "coordinator"
        assert coordinator.hierarchy is not None

    def test_distribute_context(self):
        """Should create context for all content agents."""
        hierarchy = AgentHierarchy()
        memory = MemoryManager("coordinator_test", persist_directory="./test_data")
        coordinator = CoordinatorAgent(hierarchy, memory)

        state = create_hierarchical_state("Machine Learning")
        state["coordinator_intent"] = "Explain ML basics clearly"

        contexts = coordinator.distribute_context(state)

        assert "intro" in contexts
        assert "body" in contexts
        assert "conclusion" in contexts
        assert contexts["intro"] == "Explain ML basics clearly"


class TestSpecialistAgents:
    """Test specialist agent creation."""

    def test_researcher_creation(self):
        """ResearchAgent should initialize."""
        memory = MemoryManager("researcher_test", persist_directory="./test_data")
        agent = ResearchAgent(memory)

        assert agent.role == "researcher"
        assert agent.content_key == "research"

    def test_fact_checker_creation(self):
        """FactCheckerAgent should initialize."""
        memory = MemoryManager("fact_checker_test", persist_directory="./test_data")
        agent = FactCheckerAgent(memory)

        assert agent.role == "fact_checker"
        assert agent.content_key == "fact_check"

    def test_stylist_creation(self):
        """StyleAgent should initialize."""
        memory = MemoryManager("stylist_test", persist_directory="./test_data")
        agent = StyleAgent(memory)

        assert agent.role == "stylist"
        assert agent.content_key == "style"


class TestAgentFactory:
    """Test agent factory."""

    def test_create_all_agents(self):
        """Factory should create all 7 agents."""
        agents, hierarchy = create_hierarchical_agents("./test_data")

        assert len(agents) == 7
        assert "coordinator" in agents
        assert "intro" in agents
        assert "researcher" in agents

    def test_hierarchy_returned(self):
        """Factory should return hierarchy instance."""
        agents, hierarchy = create_hierarchical_agents("./test_data")

        assert isinstance(hierarchy, AgentHierarchy)
        assert hierarchy.get_layer("coordinator") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Test Execution**:
```bash
cd worktrees/hierarchical-structure
uv run pytest test_hierarchical_structure.py -v
```

**Expected**: 8/8 tests passing

---

## Deliverables Checklist

- [ ] `src/hvas_mini/hierarchy/structure.py` - AgentHierarchy class
- [ ] `src/hvas_mini/hierarchy/coordinator.py` - CoordinatorAgent class
- [ ] `src/hvas_mini/hierarchy/specialists.py` - 3 specialist agents
- [ ] `src/hvas_mini/hierarchy/factory.py` - Agent factory
- [ ] `src/hvas_mini/hierarchy/__init__.py` - Package exports
- [ ] `src/hvas_mini/state.py` - Extended with HierarchicalState
- [ ] `test_hierarchical_structure.py` - 8 comprehensive tests
- [ ] All tests passing

---

## Acceptance Criteria

1. ✅ **3-layer hierarchy defined**
   - Layer 1: coordinator
   - Layer 2: intro, body, conclusion
   - Layer 3: researcher, fact_checker, stylist

2. ✅ **Parent-child relationships correct**
   - get_parent() returns correct parent
   - get_children() returns correct children
   - Traversal methods work

3. ✅ **CoordinatorAgent functional**
   - Parses intent from topic
   - Distributes context to children
   - Aggregates results from layer
   - Generates critiques (for M8)

4. ✅ **Specialist agents created**
   - ResearchAgent finds information
   - FactCheckerAgent verifies accuracy
   - StyleAgent enhances tone

5. ✅ **HierarchicalState extended**
   - layer_outputs structure correct
   - AgentOutput TypedDict defined
   - Pass tracking fields present

6. ✅ **All tests passing**
   - 8/8 tests pass
   - No regressions in existing tests

---

## Integration Points

### With M7 (Bidirectional Flow)
- Hierarchy provides structure for downward/upward flow
- AgentOutput format used for upward aggregation
- Parent-child relationships used for routing

### With M8 (Closed-Loop Refinement)
- critique_outputs() used for generating feedback
- pass_history tracks iterations
- revision_requested triggers re-execution

### With M9 (Semantic Distance)
- semantic_vector in AgentNode used for distance calculations
- distribute_context() will use distance for filtering

---

## Testing

```bash
# Run tests
cd worktrees/hierarchical-structure
uv run pytest test_hierarchical_structure.py -v

# Expected output
# 8 tests, all passing
```

---

## Next Steps

After M6 completion:
1. Commit to feature/hierarchical-structure branch
2. Move AGENT_TASK.md to docs/hierarchical-structure/
3. Create PR or prepare for merge
4. Begin M7 (Bidirectional Flow) - depends on M6

---

## Notes

- This milestone provides the **foundation** for true hierarchical architecture
- All subsequent milestones (M7-M10) depend on this structure
- Focus on **correctness** of relationships over performance
- Coordinator critique is simple heuristics for M6; can be enhanced later
- Semantic vectors are placeholders for M9

---

## Timeline

**Estimated**: 3-4 days

**Breakdown**:
- Day 1: AgentHierarchy structure + tests (4-5 hours)
- Day 2: CoordinatorAgent + specialists (4-5 hours)
- Day 3: Integration, factory, state extension (3-4 hours)
- Day 4: Testing, debugging, documentation (2-3 hours)
