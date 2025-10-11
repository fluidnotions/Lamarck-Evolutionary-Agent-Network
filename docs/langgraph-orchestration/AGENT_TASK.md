# Agent Task: LangGraph Orchestration

## Branch: `feature/langgraph-orchestration`

## Priority: CRITICAL - Integration layer

## Execution: SEQUENTIAL (after ALL components complete)

## Objective
Implement the main pipeline using LangGraph to orchestrate agents, evaluation, evolution, and visualization in a streaming workflow.

## Dependencies
- ✅ feature/project-foundation
- ✅ feature/state-management
- ✅ feature/memory-system
- ✅ feature/base-agent
- ✅ feature/specialized-agents
- ✅ feature/evaluation-system
- ✅ feature/visualization

## Tasks

### 1. Create `src/hvas_mini/pipeline.py`

Implement the main HVAS pipeline:

```python
"""
Main pipeline orchestrator for HVAS Mini.

Coordinates agents, evaluation, and evolution using LangGraph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, AsyncIterator
import os
from dotenv import load_dotenv

from hvas_mini.state import BlogState, create_initial_state
from hvas_mini.agents import create_agents
from hvas_mini.evaluation import ContentEvaluator
from hvas_mini.visualization import StreamVisualizer

load_dotenv()


class HVASMiniPipeline:
    """Main pipeline orchestrator for HVAS Mini system."""

    def __init__(self, persist_directory: str = "./data/memories"):
        """Initialize pipeline.

        Args:
            persist_directory: Where to persist agent memories
        """
        # Initialize agents
        self.agents = create_agents(persist_directory)

        # Initialize evaluator and visualizer
        self.evaluator = ContentEvaluator()
        self.visualizer = StreamVisualizer()

        # Build LangGraph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow.

        Graph structure:
        START → intro → body → conclusion → evaluate → evolve → END

        Returns:
            Compiled LangGraph application
        """
        workflow = StateGraph(BlogState)

        # Add agent nodes
        workflow.add_node("intro", self.agents["intro"])
        workflow.add_node("body", self.agents["body"])
        workflow.add_node("conclusion", self.agents["conclusion"])

        # Add evaluation node
        workflow.add_node("evaluate", self.evaluator)

        # Add evolution node
        workflow.add_node("evolve", self._evolution_node)

        # Define execution flow
        workflow.set_entry_point("intro")
        workflow.add_edge("intro", "body")
        workflow.add_edge("body", "conclusion")
        workflow.add_edge("conclusion", "evaluate")
        workflow.add_edge("evaluate", "evolve")
        workflow.add_edge("evolve", END)

        # Compile with memory for checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _evolution_node(self, state: BlogState) -> BlogState:
        """Evolution node: store memories and update parameters.

        Args:
            state: Current workflow state with scores

        Returns:
            Updated state
        """
        for role, agent in self.agents.items():
            score = state["scores"].get(role, 0)

            # Store memory if score meets threshold
            agent.store_memory(score)

            # Evolve parameters based on score
            agent.evolve_parameters(score, state)

        state["stream_logs"].append(
            "[Evolution] Memories stored, parameters updated"
        )

        return state

    async def generate(
        self,
        topic: str,
        config: Dict | None = None
    ) -> BlogState:
        """Generate blog post with streaming visualization.

        Args:
            topic: The topic to write about
            config: Optional LangGraph configuration

        Returns:
            Final state with generated content and scores
        """
        # Initialize state
        initial_state = create_initial_state(topic)

        # Configure streaming
        if config is None:
            config = {
                "configurable": {
                    "thread_id": f"blog_{topic.replace(' ', '_')}"
                }
            }

        # Stream execution
        final_state = initial_state

        if os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true":
            # Stream with visualization
            async def state_stream() -> AsyncIterator[BlogState]:
                nonlocal final_state
                async for event in self.app.astream(
                    initial_state,
                    config,
                    stream_mode="values"
                ):
                    final_state = event
                    yield event

            await self.visualizer.display_stream(state_stream())
        else:
            # Run without visualization
            final_state = await self.app.ainvoke(initial_state, config)

        return final_state

    def get_memory_stats(self) -> Dict:
        """Get memory statistics for all agents.

        Returns:
            Dictionary of memory stats per agent
        """
        return {
            role: agent.memory.get_stats()
            for role, agent in self.agents.items()
        }

    def get_agent_parameters(self) -> Dict:
        """Get current parameters for all agents.

        Returns:
            Dictionary of parameters per agent
        """
        return {
            role: agent.parameters
            for role, agent in self.agents.items()
        }
```

### 2. Create `main.py`

Main entry point with demo:

```python
"""
HVAS Mini - Main entry point and demo.

Demonstrates learning over multiple generations.
"""

import asyncio
from hvas_mini.pipeline import HVASMiniPipeline
from hvas_mini.evaluation import calculate_overall_score
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Demo execution showing learning across multiple topics."""
    console = Console()
    pipeline = HVASMiniPipeline()

    # Test topics to show learning
    # Similar topics will benefit from stored memories
    topics = [
        "introduction to machine learning",
        "machine learning applications",  # Similar - will use memories
        "python programming basics",
        "python for data science",  # Similar - will use memories
        "artificial intelligence ethics"
    ]

    console.print("\\n[bold cyan]HVAS Mini - Hierarchical Vector Agent System[/bold cyan]")
    console.print("Demonstrating learning across multiple generations\\n")

    for i, topic in enumerate(topics, 1):
        console.print(f"\\n{'='*60}")
        console.print(f"[bold]Generation {i}: {topic}[/bold]")
        console.print('='*60)

        # Generate
        result = await pipeline.generate(topic)

        # Display summary
        pipeline.visualizer.print_summary(result, show_content=False)

        # Calculate overall score
        overall = calculate_overall_score(result["scores"])
        console.print(f"\\n[bold]Overall Score: {overall}/10[/bold]")

        # Show learning progress
        if i > 1:
            console.print("\\n[bold cyan]📈 Learning Progress:[/bold cyan]")
            stats = pipeline.get_memory_stats()
            for role, stat in stats.items():
                total = stat["total_memories"]
                avg = stat.get("avg_score", 0)
                console.print(
                    f"  {role}: {total} memories | "
                    f"avg quality: {avg:.1f}"
                )

        # Brief pause between generations
        await asyncio.sleep(1)

    # Final statistics
    console.print("\\n" + "="*60)
    console.print("[bold green]✓ All Generations Complete[/bold green]")
    console.print("="*60 + "\\n")

    console.print("[bold]🎓 Final System State:[/bold]\\n")

    # Memory statistics
    console.print("[bold cyan]Memory Statistics:[/bold cyan]")
    stats = pipeline.get_memory_stats()
    for role, stat in stats.items():
        console.print(f"\\n  {role.upper()}:")
        console.print(f"    Total memories: {stat['total_memories']}")
        console.print(f"    Avg score: {stat.get('avg_score', 0):.1f}")
        console.print(f"    Total retrievals: {stat.get('total_retrievals', 0)}")

    # Agent parameters
    console.print("\\n[bold yellow]Agent Parameters:[/bold yellow]")
    params = pipeline.get_agent_parameters()
    for role, param in params.items():
        console.print(f"\\n  {role.upper()}:")
        console.print(f"    Temperature: {param['temperature']:.2f}")
        console.print(f"    Generations: {param['generation_count']}")
        if param['score_history']:
            avg = sum(param['score_history']) / len(param['score_history'])
            console.print(f"    Avg score: {avg:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Create Tests

Create `test_pipeline.py`:

```python
"""Tests for pipeline orchestration."""

from hvas_mini.pipeline import HVASMiniPipeline
import pytest
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def pipeline(temp_dir):
    return HVASMiniPipeline(persist_directory=temp_dir)


@pytest.mark.asyncio
async def test_pipeline_creation(pipeline):
    """Test pipeline can be created."""
    assert pipeline.agents is not None
    assert "intro" in pipeline.agents
    assert "body" in pipeline.agents
    assert "conclusion" in pipeline.agents


@pytest.mark.asyncio
async def test_pipeline_generation(pipeline):
    """Test full generation pipeline."""
    result = await pipeline.generate("test topic")

    # Should have all content
    assert result["intro"] != ""
    assert result["body"] != ""
    assert result["conclusion"] != ""

    # Should have scores
    assert "intro" in result["scores"]
    assert "body" in result["scores"]
    assert "conclusion" in result["scores"]

    # Should have metadata
    assert result["generation_id"] != ""
    assert result["topic"] == "test topic"


@pytest.mark.asyncio
async def test_pipeline_learning(pipeline):
    """Test pipeline learns across generations."""
    # First generation
    result1 = await pipeline.generate("machine learning basics")
    score1 = result1["scores"]["intro"]

    # Second generation (similar topic)
    result2 = await pipeline.generate("machine learning concepts")

    # Should have retrieved memories
    assert len(result2["retrieved_memories"]["intro"]) > 0


def test_memory_stats(pipeline):
    """Test getting memory statistics."""
    stats = pipeline.get_memory_stats()

    assert "intro" in stats
    assert "body" in stats
    assert "conclusion" in stats


def test_agent_parameters(pipeline):
    """Test getting agent parameters."""
    params = pipeline.get_agent_parameters()

    assert "intro" in params
    assert "body" in params
    assert "conclusion" in params
    assert "temperature" in params["intro"]
```

### 4. Create Integration Test

Create `test_integration.py`:

```python
"""Integration test for complete system."""

import asyncio
import pytest
from hvas_mini.pipeline import HVASMiniPipeline
import tempfile
import shutil


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_learning_cycle():
    """Test complete learning cycle over multiple generations."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        pipeline = HVASMiniPipeline(persist_directory=temp_dir)

        topics = [
            "python programming",
            "python for beginners",  # Similar
            "python best practices"  # Similar
        ]

        results = []

        for topic in topics:
            result = await pipeline.generate(topic)
            results.append(result)

            # Brief pause
            await asyncio.sleep(0.5)

        # Verify learning occurred
        # Later generations should have memory retrievals
        assert len(results[1]["retrieved_memories"]["intro"]) > 0
        assert len(results[2]["retrieved_memories"]["intro"]) > 0

        # Memory should accumulate
        stats = pipeline.get_memory_stats()
        assert stats["intro"]["total_memories"] >= 2

        # Parameters should have evolved
        params = pipeline.get_agent_parameters()
        assert params["intro"]["generation_count"] == 3

    finally:
        shutil.rmtree(temp_dir)
```

## Deliverables Checklist

- [ ] `src/hvas_mini/pipeline.py` with:
  - [ ] `HVASMiniPipeline` class
  - [ ] `_build_graph()` method
  - [ ] `_evolution_node()` method
  - [ ] `generate()` method
  - [ ] Complete docstrings
- [ ] `main.py` with demo execution
- [ ] `test_pipeline.py` with passing tests
- [ ] `test_integration.py` with end-to-end test
- [ ] LangGraph workflow with correct edges

## Acceptance Criteria

1. ✅ LangGraph workflow executes correctly
2. ✅ Agents run in correct sequence
3. ✅ Evaluation occurs after all agents
4. ✅ Evolution stores memories and updates parameters
5. ✅ Streaming visualization works
6. ✅ Learning occurs across multiple generations
7. ✅ All tests pass: `uv run pytest test_pipeline.py test_integration.py`
8. ✅ Demo runs successfully: `uv run python main.py`

## Testing

```bash
cd worktrees/langgraph-orchestration

# Unit tests
uv run pytest test_pipeline.py -v

# Integration test
uv run pytest test_integration.py -v -m integration

# Run demo (requires ANTHROPIC_API_KEY)
uv run python main.py
```

## Integration Notes

This is the final integration layer that:
- Uses all agents, evaluator, visualizer
- Orchestrates with LangGraph
- Demonstrates complete HVAS learning cycle
- Provides main entry point

## Next Steps

After completion:
1. Merge to main
2. Test complete system
3. Create documentation in ./docs
4. Write README.md
