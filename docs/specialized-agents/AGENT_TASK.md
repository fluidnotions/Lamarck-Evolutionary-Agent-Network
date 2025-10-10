# Agent Task: Specialized Agents

## Branch: `feature/specialized-agents`

## Priority: MEDIUM - Application layer

## Execution: SEQUENTIAL (after feature/base-agent)

## Objective
Implement three specialized agents (IntroAgent, BodyAgent, ConclusionAgent) that inherit from BaseAgent and provide domain-specific content generation with memory-informed prompts.

## Dependencies
- ✅ feature/project-foundation
- ✅ feature/state-management
- ✅ feature/memory-system
- ✅ feature/base-agent (must be merged)

## Tasks

### 1. Extend `src/hvas_mini/agents.py`

Add specialized agents according to spec (section 3.3):

```python
# Add to existing agents.py file

class IntroAgent(BaseAgent):
    """Agent specialized in writing introductions.

    Focuses on creating engaging hooks and setting expectations.
    """

    @property
    def content_key(self) -> str:
        return "intro"

    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict]
    ) -> str:
        """Generate introduction with memory examples.

        Args:
            state: Current workflow state
            memories: Retrieved successful introductions

        Returns:
            Generated introduction
        """
        # Format memory examples
        memory_examples = ""
        if memories:
            memory_examples = "\\n\\n".join([
                f"Example (score: {m['score']:.1f}):\\n{m['content']}"
                for m in memories[:2]
            ])
        else:
            memory_examples = "No previous examples available."

        # Construct prompt
        prompt = f\"\"\"Write an engaging introduction for a blog post about: {state['topic']}

Previous successful introductions on similar topics:
{memory_examples}

Requirements:
- 2-3 sentences
- Hook the reader immediately
- Mention the topic naturally
- Set expectations for what follows

Introduction:\"\"\"

        # Generate
        response = await self.llm.ainvoke(prompt)
        return response.content


class BodyAgent(BaseAgent):
    """Agent specialized in writing main body content.

    Focuses on detailed, informative content with examples.
    """

    @property
    def content_key(self) -> str:
        return "body"

    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict]
    ) -> str:
        """Generate body content with context from intro.

        Args:
            state: Current workflow state
            memories: Retrieved successful body sections

        Returns:
            Generated body content
        """
        # Can see the intro if it was generated
        context = f"Introduction: {state.get('intro', 'Not yet written')}"

        # Format memory examples (truncated for prompt)
        memory_examples = ""
        if memories:
            memory_examples = "\\n\\n".join([
                f"Example body (score: {m['score']:.1f}):\\n{m['content'][:200]}..."
                for m in memories[:2]
            ])

        # Construct prompt
        prompt = f\"\"\"Write the main body for a blog post about: {state['topic']}

{context}

Previous successful body sections:
{memory_examples}

Requirements:
- 3-4 paragraphs
- Informative and detailed
- Include specific examples or data
- Natural flow from the introduction

Body:\"\"\"

        # Generate
        response = await self.llm.ainvoke(prompt)
        return response.content


class ConclusionAgent(BaseAgent):
    """Agent specialized in writing conclusions.

    Focuses on summarizing and providing call-to-action.
    """

    @property
    def content_key(self) -> str:
        return "conclusion"

    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict]
    ) -> str:
        """Generate conclusion with full context.

        Args:
            state: Current workflow state
            memories: Retrieved successful conclusions

        Returns:
            Generated conclusion
        """
        # Can see intro and body if generated
        intro_preview = state.get('intro', 'Not yet written')
        body_preview = state.get('body', 'Not yet written')[:200]

        context = f\"\"\"
Introduction: {intro_preview}

Body preview: {body_preview}...
\"\"\"

        # Format memory examples
        memory_examples = ""
        if memories:
            memory_examples = "\\n".join([
                f"Example: {m['content']}"
                for m in memories[:1]
            ])

        # Construct prompt
        prompt = f\"\"\"Write a conclusion for this blog post about: {state['topic']}

{context}

Previous successful conclusions:
{memory_examples}

Requirements:
- 2-3 sentences
- Summarize key points
- End with memorable statement
- Call to action or thought to ponder

Conclusion:\"\"\"

        # Generate
        response = await self.llm.ainvoke(prompt)
        return response.content
```

### 2. Create Agent Factory

Add factory function to `agents.py`:

```python
def create_agents(persist_directory: str = "./data/memories") -> Dict[str, BaseAgent]:
    """Create all specialized agents.

    Args:
        persist_directory: Where to persist memories

    Returns:
        Dictionary of agent instances
    """
    agents = {}

    for role in ["intro", "body", "conclusion"]:
        # Create memory manager for this agent
        memory = MemoryManager(
            collection_name=f"{role}_memories",
            persist_directory=persist_directory
        )

        # Create appropriate agent
        if role == "intro":
            agents[role] = IntroAgent(role, memory)
        elif role == "body":
            agents[role] = BodyAgent(role, memory)
        elif role == "conclusion":
            agents[role] = ConclusionAgent(role, memory)

    return agents
```

### 3. Create Tests

Create `test_specialized_agents.py`:

```python
"""Tests for specialized agents."""

from hvas_mini.agents import IntroAgent, BodyAgent, ConclusionAgent, create_agents
from hvas_mini.state import create_initial_state
from hvas_mini.memory import MemoryManager
import tempfile
import shutil
import pytest


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def intro_agent(temp_dir):
    memory = MemoryManager("intro", persist_directory=temp_dir)
    return IntroAgent("intro", memory)


@pytest.fixture
def body_agent(temp_dir):
    memory = MemoryManager("body", persist_directory=temp_dir)
    return BodyAgent("body", memory)


@pytest.fixture
def conclusion_agent(temp_dir):
    memory = MemoryManager("conclusion", persist_directory=temp_dir)
    return ConclusionAgent("conclusion", memory)


@pytest.mark.asyncio
async def test_intro_agent(intro_agent):
    """Test intro agent generates content."""
    state = create_initial_state("machine learning")
    result = await intro_agent(state)

    assert result["intro"] != ""
    assert len(result["intro"]) > 0
    assert "intro" in result["retrieved_memories"]


@pytest.mark.asyncio
async def test_body_agent(body_agent):
    """Test body agent generates content."""
    state = create_initial_state("machine learning")
    state["intro"] = "This is the introduction."

    result = await body_agent(state)

    assert result["body"] != ""
    assert len(result["body"]) > 0


@pytest.mark.asyncio
async def test_conclusion_agent(conclusion_agent):
    """Test conclusion agent generates content."""
    state = create_initial_state("machine learning")
    state["intro"] = "This is the introduction."
    state["body"] = "This is the body content."

    result = await conclusion_agent(state)

    assert result["conclusion"] != ""
    assert len(result["conclusion"]) > 0


def test_create_agents(temp_dir):
    """Test agent factory."""
    agents = create_agents(persist_directory=temp_dir)

    assert "intro" in agents
    assert "body" in agents
    assert "conclusion" in agents
    assert isinstance(agents["intro"], IntroAgent)
    assert isinstance(agents["body"], BodyAgent)
    assert isinstance(agents["conclusion"], ConclusionAgent)


@pytest.mark.asyncio
async def test_agent_context_passing(intro_agent, body_agent, conclusion_agent):
    """Test agents can see each other's output."""
    state = create_initial_state("AI ethics")

    # Execute in sequence
    state = await intro_agent(state)
    state = await body_agent(state)
    state = await conclusion_agent(state)

    # All sections should be populated
    assert state["intro"] != ""
    assert state["body"] != ""
    assert state["conclusion"] != ""
```

## Deliverables Checklist

- [ ] `IntroAgent` in `agents.py`
- [ ] `BodyAgent` in `agents.py`
- [ ] `ConclusionAgent` in `agents.py`
- [ ] `create_agents()` factory function
- [ ] Memory-informed prompts for each agent
- [ ] Context passing between agents
- [ ] `test_specialized_agents.py` with passing tests
- [ ] Complete docstrings

## Acceptance Criteria

1. ✅ All three agents inherit from BaseAgent correctly
2. ✅ Each agent has appropriate `content_key`
3. ✅ Prompts include memory examples when available
4. ✅ BodyAgent can see intro in state
5. ✅ ConclusionAgent can see intro and body
6. ✅ All tests pass: `uv run pytest test_specialized_agents.py`
7. ✅ Agents generate reasonable content

## Testing

```bash
cd worktrees/specialized-agents
uv run pytest test_specialized_agents.py -v

# Note: Tests will call real Anthropic API
# Set ANTHROPIC_API_KEY in .env
```

## Integration Notes

These agents will be used by:
- LangGraph workflow as nodes
- Each maintains its own memory collection
- Sequential execution allows context passing

## Next Steps

After completion, merge to main. Can now integrate with:
- feature/evaluation-system (to score agent outputs)
- feature/langgraph-orchestration (to connect as nodes)
