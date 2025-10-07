# Extending Agents - Customization Guide

This guide explains how to create new specialized agents and customize existing ones in HVAS Mini.

---

## Table of Contents

1. [Creating a New Agent](#creating-a-new-agent)
2. [Customizing Agent Prompts](#customizing-agent-prompts)
3. [Adding Agent-Specific Parameters](#adding-agent-specific-parameters)
4. [Custom Memory Retrieval](#custom-memory-retrieval)
5. [Examples](#examples)

---

## Creating a New Agent

### Step 1: Define Your Agent Class

Create a new agent by inheriting from `BaseAgent`:

```python
from hvas_mini.agents import BaseAgent
from hvas_mini.state import BlogState
from typing import List, Dict

class TitleAgent(BaseAgent):
    """Agent specialized in generating catchy titles."""

    @property
    def content_key(self) -> str:
        """State key where this agent writes its output."""
        return "title"  # Will write to state["title"]

    async def generate_content(
        self,
        state: BlogState,
        memories: List[Dict]
    ) -> str:
        """Generate title based on intro and body."""
        # Access other agent outputs
        intro = state.get("intro", "")
        body_preview = state.get("body", "")[:200]

        # Format retrieved memories
        memory_examples = ""
        if memories:
            memory_examples = "\\n".join([
                f"Example title (score: {m['score']:.1f}): {m['content']}"
                for m in memories[:3]
            ])

        # Construct prompt
        prompt = f\"\"\"Create a compelling title for this blog post:

Topic: {state['topic']}

Introduction: {intro}

Body preview: {body_preview}...

Previous successful titles:
{memory_examples}

Requirements:
- 5-10 words
- Engaging and clickworthy
- Accurately reflects content
- SEO-friendly

Title:\"\"\"

        # Generate using LLM
        response = await self.llm.ainvoke(prompt)
        return response.content
```

### Step 2: Update State Definition

Add your new field to `BlogState` in `src/hvas_mini/state.py`:

```python
class BlogState(TypedDict):
    # Existing fields...
    topic: str
    intro: str
    body: str
    conclusion: str

    # Add your new field
    title: str  # <-- Add this

    # Rest of state...
    scores: Dict[str, float]
    # ...
```

### Step 3: Add to Agent Factory

Update `create_agents()` in `src/hvas_mini/agents.py`:

```python
def create_agents(persist_directory: str = "./data/memories") -> Dict[str, BaseAgent]:
    """Create all specialized agents."""
    agents = {}

    # Existing agents
    for role in ["intro", "body", "conclusion"]:
        # ... existing code ...

    # Add your new agent
    title_memory = MemoryManager(
        collection_name="title_memories",
        persist_directory=persist_directory
    )
    agents["title"] = TitleAgent("title", title_memory)

    return agents
```

### Step 4: Update Graph Workflow

Modify the graph in `src/hvas_mini/pipeline.py`:

```python
def _build_graph(self) -> StateGraph:
    """Construct LangGraph workflow."""
    workflow = StateGraph(BlogState)

    # Add nodes
    workflow.add_node("intro", self.agents["intro"])
    workflow.add_node("body", self.agents["body"])
    workflow.add_node("conclusion", self.agents["conclusion"])
    workflow.add_node("title", self.agents["title"])  # <-- Add this
    workflow.add_node("evaluate", self.evaluator)
    workflow.add_node("evolve", self._evolution_node)

    # Define flow
    workflow.set_entry_point("intro")
    workflow.add_edge("intro", "body")
    workflow.add_edge("body", "conclusion")
    workflow.add_edge("conclusion", "title")  # <-- Add this
    workflow.add_edge("title", "evaluate")    # <-- Modify this
    workflow.add_edge("evaluate", "evolve")
    workflow.add_edge("evolve", END)

    return workflow.compile(checkpointer=MemorySaver())
```

### Step 5: Update Evaluation

Add scoring for your new agent in `src/hvas_mini/evaluation.py`:

```python
class ContentEvaluator:
    def __call__(self, state: BlogState) -> BlogState:
        scores = {
            "intro": self._score_intro(state["intro"], state["topic"]),
            "body": self._score_body(state["body"], state["topic"]),
            "conclusion": self._score_conclusion(state["conclusion"], state["topic"], state["intro"]),
            "title": self._score_title(state["title"], state["topic"])  # <-- Add this
        }
        state["scores"] = scores
        return state

    def _score_title(self, title: str, topic: str) -> float:
        """Score title quality."""
        score = 5.0

        # Length check
        word_count = len(title.split())
        if 5 <= word_count <= 10:
            score += 2.0

        # Topic relevance
        if any(word.lower() in title.lower() for word in topic.split()):
            score += 2.0

        # Engaging words
        engaging = ["ultimate", "complete", "essential", "guide", "how to", "why", "what"]
        if any(word in title.lower() for word in engaging):
            score += 1.0

        return min(10.0, score)
```

---

## Customizing Agent Prompts

### Basic Prompt Customization

Modify the `generate_content()` method:

```python
async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
    # Custom prompt template
    prompt = f\"\"\"You are an expert technical writer specializing in {state['topic']}.

Your task: Write an introduction that:
1. Starts with a surprising fact or statistic
2. Clearly states what the reader will learn
3. Builds credibility by mentioning your expertise

Previous successful examples:
{self._format_memories(memories)}

Topic: {state['topic']}

Write the introduction now:\"\"\"

    response = await self.llm.ainvoke(prompt)
    return response.content
```

### Adding Few-Shot Examples

Include static examples in prompts:

```python
def _get_few_shot_examples(self) -> str:
    """Provide static few-shot examples."""
    return \"\"\"
Example 1:
Topic: "quantum computing"
Output: "Quantum computers just broke a record that would take classical computers 47 years to match. In this guide, you'll discover how quantum mechanics enables computation that defies our everyday logic."

Example 2:
Topic: "sustainable architecture"
Output: "Buildings account for 40% of global energy consumption—but innovative architects are changing that. Learn how sustainable design is revolutionizing the construction industry."
\"\"\"

async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
    prompt = f\"\"\"
{self._get_few_shot_examples()}

Now write for this topic: {state['topic']}
\"\"\"
    # ... rest of implementation
```

### Dynamic Prompt Adjustment

Adjust prompts based on agent state:

```python
async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
    # Adjust prompt based on performance
    if self.parameters["generation_count"] < 3:
        # Early generations: more guidance
        guidance = "Be creative but follow the structure closely."
    else:
        # Later generations: more freedom
        guidance = "Use your learned patterns to create compelling content."

    prompt = f\"\"\"{guidance}

Topic: {state['topic']}
\"\"\"
    # ... rest of implementation
```

---

## Adding Agent-Specific Parameters

### Extend Agent with Custom Parameters

```python
class CustomAgent(BaseAgent):
    def __init__(self, role: str, memory_manager: MemoryManager):
        super().__init__(role, memory_manager)

        # Add custom parameters
        self.parameters.update({
            "max_length": 200,
            "formality": 0.7,
            "creativity_boost": 1.0
        })

    def evolve_parameters(self, score: float, state: BlogState):
        """Evolve custom parameters based on performance."""
        # Call parent evolution
        super().evolve_parameters(score, state)

        # Custom evolution logic
        if score > 8.0:
            # Good performance: increase creativity
            self.parameters["creativity_boost"] *= 1.1
        elif score < 6.0:
            # Poor performance: reduce creativity
            self.parameters["creativity_boost"] *= 0.9

        # Clamp values
        self.parameters["creativity_boost"] = max(0.5, min(2.0, self.parameters["creativity_boost"]))

    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        # Use custom parameters in generation
        self.llm.temperature *= self.parameters["creativity_boost"]

        # ... rest of implementation
```

### Parameter Persistence

Store parameters in memory metadata:

```python
def store_memory(self, score: float):
    """Store memory with custom parameters."""
    if self.pending_memory is None:
        return

    # Include custom parameters
    metadata = {
        "timestamp": self.pending_memory["timestamp"],
        "parameters": json.dumps(self.parameters),  # All parameters
        "formality": self.parameters.get("formality"),  # Individual values
        "creativity_boost": self.parameters.get("creativity_boost")
    }

    self.memory.store(
        content=self.pending_memory["content"],
        topic=self.pending_memory["topic"],
        score=score,
        metadata=metadata
    )
```

---

## Custom Memory Retrieval

### Filtered Memory Retrieval

```python
async def retrieve_memories(self, topic: str) -> List[Dict]:
    """Retrieve memories with custom filtering."""
    # Get base memories
    memories = await super().retrieve_memories(topic)

    # Custom filtering
    filtered = []
    for mem in memories:
        # Only use memories with high creativity
        if mem.get("metadata", {}).get("creativity_boost", 0) > 0.8:
            filtered.append(mem)

    return filtered[:self.max_retrieve]
```

### Weighted Memory Selection

```python
def _select_diverse_memories(self, memories: List[Dict]) -> List[Dict]:
    """Select diverse memories instead of just top-scoring."""
    if len(memories) <= 3:
        return memories

    # Sort by score
    sorted_mems = sorted(memories, key=lambda x: x["score"], reverse=True)

    # Take top scorer
    selected = [sorted_mems[0]]

    # Add diverse examples
    for mem in sorted_mems[1:]:
        # Check if sufficiently different from selected
        if self._is_diverse(mem, selected):
            selected.append(mem)
        if len(selected) >= 3:
            break

    return selected

def _is_diverse(self, candidate: Dict, selected: List[Dict]) -> bool:
    """Check if candidate is diverse from already selected."""
    candidate_words = set(candidate["content"].lower().split())

    for mem in selected:
        mem_words = set(mem["content"].lower().split())
        overlap = len(candidate_words & mem_words) / len(candidate_words)

        if overlap > 0.7:  # Too similar
            return False

    return True
```

---

## Examples

### Example 1: Summary Agent

```python
class SummaryAgent(BaseAgent):
    """Generates executive summary of the blog post."""

    @property
    def content_key(self) -> str:
        return "summary"

    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        # Read all sections
        full_content = f\"\"\"{state.get('intro', '')}

{state.get('body', '')}

{state.get('conclusion', '')}\"\"\"

        prompt = f\"\"\"Create a 2-sentence executive summary of this blog post:

{full_content}

Requirements:
- Exactly 2 sentences
- Capture main value proposition
- Actionable and clear

Summary:\"\"\"

        response = await self.llm.ainvoke(prompt)
        return response.content
```

### Example 2: SEO Keywords Agent

```python
class SEOAgent(BaseAgent):
    """Extracts SEO keywords from content."""

    @property
    def content_key(self) -> str:
        return "seo_keywords"

    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        prompt = f\"\"\"Analyze this blog post and extract 5-7 SEO keywords:

Topic: {state['topic']}
Intro: {state.get('intro', '')}
Body: {state.get('body', '')[:500]}

Return keywords as comma-separated list:\"\"\"

        response = await self.llm.ainvoke(prompt)
        return response.content
```

### Example 3: Fact-Checker Agent

```python
class FactCheckerAgent(BaseAgent):
    """Verifies claims in generated content."""

    @property
    def content_key(self) -> str:
        return "fact_check_report"

    async def generate_content(self, state: BlogState, memories: List[Dict]) -> str:
        content = f\"\"\"{state.get('intro', '')}
{state.get('body', '')}
{state.get('conclusion', '')}\"\"\"

        prompt = f\"\"\"Review this content for factual claims that need verification:

{content}

For each claim:
1. Quote the claim
2. Assess confidence (high/medium/low)
3. Suggest verification source

Report:\"\"\"

        response = await self.llm.ainvoke(prompt)
        return response.content
```

---

## Best Practices

1. **Single Responsibility**: Each agent should have one clear purpose
2. **State Keys**: Use descriptive state keys (e.g., "seo_keywords" not "keywords")
3. **Memory Namespacing**: Give each agent its own ChromaDB collection
4. **Prompt Engineering**: Iterate on prompts based on actual outputs
5. **Parameter Bounds**: Always clamp evolved parameters to reasonable ranges
6. **Error Handling**: Handle cases where previous agents haven't run yet
7. **Testing**: Write tests for each new agent independently

---

## Troubleshooting

### Agent Not Executing

**Problem**: Agent added but not running

**Solution**: Check graph edges in pipeline.py - ensure your node is connected

### Memory Not Storing

**Problem**: Agent generates content but memories don't persist

**Solution**:
1. Check score meets threshold (default 7.0)
2. Verify `store_memory()` is called in evolution node
3. Check ChromaDB persistence directory exists

### Prompt Too Long

**Problem**: Prompts exceed context window

**Solution**:
1. Truncate retrieved memories
2. Summarize other agent outputs
3. Use only most relevant context

---

## Next Steps

- Explore [custom-evaluation.md](custom-evaluation.md) for scoring new agents
- See [langgraph-patterns.md](langgraph-patterns.md) for advanced workflows
- Review `src/hvas_mini/agents.py` for complete examples
