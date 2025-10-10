# Agent Task: Visualization System

## Branch: `feature/visualization`

## Priority: LOW - Enhancement feature

## Execution: PARALLEL with other features

## Objective
Implement real-time streaming visualization using Rich library to display agent execution, memory retrieval, and parameter evolution.

## Dependencies
- ✅ feature/project-foundation
- ✅ feature/state-management

## Tasks

### 1. Create `src/hvas_mini/visualization.py`

Implement according to spec (section 3.5):

```python
"""
Real-time visualization system for HVAS Mini.

Uses Rich library to display streaming execution updates.
"""

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from typing import Dict, List, AsyncIterator
import os
from dotenv import load_dotenv

from hvas_mini.state import BlogState

load_dotenv()


class StreamVisualizer:
    """Real-time visualization of agent execution.

    Displays:
    - Agent execution status
    - Memory retrieval
    - Parameter evolution
    - Activity logs
    """

    def __init__(self):
        """Initialize visualizer."""
        self.console = Console()
        self.show_visualization = (
            os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true"
        )

    def create_status_table(self, state: BlogState) -> Table:
        """Create status table for current execution.

        Args:
            state: Current workflow state

        Returns:
            Rich Table with agent status
        """
        table = Table(title="🤖 Agent Execution Status", show_header=True)
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Memories", style="green")
        table.add_column("Temperature", style="yellow")
        table.add_column("Score", style="blue")

        for role in ["intro", "body", "conclusion"]:
            # Get data
            memories = len(state.get("retrieved_memories", {}).get(role, []))
            param_update = state.get("parameter_updates", {}).get(role, {})
            temp = param_update.get("new_temperature", 0.7)
            score = state.get("scores", {}).get(role, 0.0)

            # Determine status
            if state.get(role):
                status = "✓ Complete"
            else:
                status = "⟳ Processing"

            table.add_row(
                role.capitalize(),
                status,
                str(memories),
                f"{temp:.2f}",
                f"{score:.1f}" if score > 0 else "-"
            )

        return table

    def create_memory_panel(self, state: BlogState) -> Panel:
        """Show retrieved memories.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with memory information
        """
        memories_text = ""

        for role, memories in state.get("retrieved_memories", {}).items():
            if memories:
                memories_text += f"[bold cyan]{role.upper()}:[/bold cyan]\\n"
                for i, mem in enumerate(memories[:2], 1):
                    preview = mem[:100] + "..." if len(mem) > 100 else mem
                    memories_text += f"  {i}. {preview}\\n"
                memories_text += "\\n"

        if not memories_text:
            memories_text = "[dim]No memories retrieved yet[/dim]"

        return Panel(
            memories_text,
            title="🧠 Retrieved Memories",
            border_style="green"
        )

    def create_evolution_panel(self, state: BlogState) -> Panel:
        """Show parameter evolution.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with evolution information
        """
        evolution_text = ""

        for role, updates in state.get("parameter_updates", {}).items():
            if updates:
                old_t = updates["old_temperature"]
                new_t = updates["new_temperature"]
                score = updates["score"]
                avg = updates["avg_score"]

                # Determine direction
                if new_t > old_t:
                    change = "↑"
                    change_style = "green"
                elif new_t < old_t:
                    change = "↓"
                    change_style = "red"
                else:
                    change = "→"
                    change_style = "yellow"

                evolution_text += (
                    f"[bold cyan]{role.upper()}:[/bold cyan]\\n"
                    f"  Temperature: {old_t:.2f} "
                    f"[{change_style}]{change}[/{change_style}] {new_t:.2f}\\n"
                    f"  Last Score: {score:.1f} | Avg: {avg:.1f}\\n\\n"
                )

        if not evolution_text:
            evolution_text = "[dim]No parameter updates yet[/dim]"

        return Panel(
            evolution_text,
            title="🔧 Parameter Evolution",
            border_style="yellow"
        )

    def create_logs_panel(self, state: BlogState) -> Panel:
        """Create activity log panel.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with recent logs
        """
        logs = state.get("stream_logs", [])[-5:]
        logs_text = "\\n".join(logs) if logs else "[dim]No activity yet[/dim]"

        return Panel(
            logs_text,
            title="📋 Activity Log",
            border_style="blue"
        )

    async def display_stream(
        self,
        state_stream: AsyncIterator[BlogState]
    ):
        """Display real-time execution updates.

        Args:
            state_stream: Async iterator of state updates
        """
        if not self.show_visualization:
            # Just consume the stream without displaying
            async for _ in state_stream:
                pass
            return

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=10),
            Layout(name="memories", size=10),
            Layout(name="evolution", size=8),
            Layout(name="logs", size=7)
        )

        # Stream with live updates
        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["memories"].update(self.create_memory_panel(state))
                layout["evolution"].update(self.create_evolution_panel(state))
                layout["logs"].update(self.create_logs_panel(state))

    def print_summary(
        self,
        state: BlogState,
        show_content: bool = False
    ):
        """Print final summary after generation.

        Args:
            state: Final workflow state
            show_content: Whether to show generated content
        """
        if not self.show_visualization:
            return

        self.console.print("\\n" + "=" * 60)
        self.console.print("[bold green]✓ Generation Complete[/bold green]")
        self.console.print("=" * 60 + "\\n")

        # Scores
        self.console.print("[bold]📊 Scores:[/bold]")
        for role, score in state.get("scores", {}).items():
            color = "green" if score >= 8.0 else "yellow" if score >= 6.0 else "red"
            self.console.print(f"  {role}: [{color}]{score:.1f}/10[/{color}]")

        # Memory stats
        self.console.print("\\n[bold]🧠 Memory Status:[/bold]")
        for role in ["intro", "body", "conclusion"]:
            count = len(state.get("retrieved_memories", {}).get(role, []))
            self.console.print(f"  {role}: {count} memories used")

        # Parameters
        self.console.print("\\n[bold]🔧 Final Parameters:[/bold]")
        for role, updates in state.get("parameter_updates", {}).items():
            if updates:
                temp = updates["new_temperature"]
                self.console.print(f"  {role}: temp={temp:.2f}")

        # Content (if requested)
        if show_content:
            self.console.print("\\n[bold]📝 Generated Content:[/bold]\\n")
            self.console.print(Panel(
                state.get("intro", ""),
                title="Introduction",
                border_style="cyan"
            ))
            self.console.print(Panel(
                state.get("body", "")[:500] + "...",
                title="Body (preview)",
                border_style="cyan"
            ))
            self.console.print(Panel(
                state.get("conclusion", ""),
                title="Conclusion",
                border_style="cyan"
            ))
```

### 2. Create Tests

Create `test_visualization.py`:

```python
"""Tests for visualization system."""

from hvas_mini.visualization import StreamVisualizer
from hvas_mini.state import create_initial_state
import pytest


@pytest.fixture
def visualizer():
    return StreamVisualizer()


def test_status_table_creation(visualizer):
    """Test status table can be created."""
    state = create_initial_state("test")
    state["intro"] = "Test intro"

    table = visualizer.create_status_table(state)

    assert table is not None
    assert table.title == "🤖 Agent Execution Status"


def test_memory_panel_creation(visualizer):
    """Test memory panel creation."""
    state = create_initial_state("test")
    state["retrieved_memories"] = {
        "intro": ["Memory 1", "Memory 2"]
    }

    panel = visualizer.create_memory_panel(state)

    assert panel is not None
    assert "INTRO" in str(panel)


def test_evolution_panel_creation(visualizer):
    """Test evolution panel creation."""
    state = create_initial_state("test")
    state["parameter_updates"] = {
        "intro": {
            "old_temperature": 0.7,
            "new_temperature": 0.8,
            "score": 8.5,
            "avg_score": 8.0
        }
    }

    panel = visualizer.create_evolution_panel(state)

    assert panel is not None


def test_logs_panel_creation(visualizer):
    """Test logs panel creation."""
    state = create_initial_state("test")
    state["stream_logs"] = [
        "[intro] Retrieved 2 memories",
        "[body] Generated content"
    ]

    panel = visualizer.create_logs_panel(state)

    assert panel is not None


def test_visualization_can_be_disabled(visualizer):
    """Test visualization can be disabled."""
    import os
    os.environ["ENABLE_VISUALIZATION"] = "false"

    viz = StreamVisualizer()
    assert viz.show_visualization is False
```

## Deliverables Checklist

- [ ] `src/hvas_mini/visualization.py` with:
  - [ ] `StreamVisualizer` class
  - [ ] `create_status_table()` method
  - [ ] `create_memory_panel()` method
  - [ ] `create_evolution_panel()` method
  - [ ] `create_logs_panel()` method
  - [ ] `display_stream()` async method
  - [ ] `print_summary()` method
  - [ ] Complete docstrings
- [ ] `test_visualization.py` with passing tests
- [ ] Rich UI components (tables, panels, layouts)

## Acceptance Criteria

1. ✅ Creates beautiful terminal UI with Rich
2. ✅ Status table shows all agents
3. ✅ Memory panel shows retrievals
4. ✅ Evolution panel shows parameter changes
5. ✅ Logs panel shows activity
6. ✅ All tests pass: `uv run pytest test_visualization.py`
7. ✅ Can be disabled via .env

## Testing

```bash
cd worktrees/visualization
uv run pytest test_visualization.py -v
```

## Integration Notes

The visualizer will be:
- Used by main pipeline to display streaming updates
- Optional (can be disabled)
- Reads all state fields for display

## Next Steps

After completion, merge to main and integrate with:
- feature/langgraph-orchestration (display during streaming)
