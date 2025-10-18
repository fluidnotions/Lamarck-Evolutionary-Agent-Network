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

load_dotenv()

# Import BlogState - will need state.py from state-management branch
try:
    from lean.state import BlogState
except ImportError:
    # For standalone development
    from typing import TypedDict

    class BlogState(TypedDict):
        """Temporary type definition."""

        topic: str
        intro: str
        body: str
        conclusion: str
        scores: Dict[str, float]
        retrieved_memories: Dict[str, List[str]]
        parameter_updates: Dict[str, Dict[str, float]]
        generation_id: str
        timestamp: str
        stream_logs: List[str]


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
                f"{score:.1f}" if score > 0 else "-",
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
                memories_text += f"[bold cyan]{role.upper()}:[/bold cyan]\n"
                for i, mem in enumerate(memories[:2], 1):
                    preview = mem[:100] + "..." if len(mem) > 100 else mem
                    memories_text += f"  {i}. {preview}\n"
                memories_text += "\n"

        if not memories_text:
            memories_text = "[dim]No memories retrieved yet[/dim]"

        return Panel(memories_text, title="🧠 Retrieved Memories", border_style="green")

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
                    f"[bold cyan]{role.upper()}:[/bold cyan]\n"
                    f"  Temperature: {old_t:.2f} "
                    f"[{change_style}]{change}[/{change_style}] {new_t:.2f}\n"
                    f"  Last Score: {score:.1f} | Avg: {avg:.1f}\n\n"
                )

        if not evolution_text:
            evolution_text = "[dim]No parameter updates yet[/dim]"

        return Panel(
            evolution_text, title="🔧 Parameter Evolution", border_style="yellow"
        )

    def create_logs_panel(self, state: BlogState) -> Panel:
        """Create activity log panel.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with recent logs
        """
        logs = state.get("stream_logs", [])[-5:]
        logs_text = "\n".join(logs) if logs else "[dim]No activity yet[/dim]"

        return Panel(logs_text, title="📋 Activity Log", border_style="blue")

    def create_concurrency_panel(self, state: BlogState) -> Panel:
        """Show agent execution timing and overlap.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with concurrency metrics
        """
        timings = state.get("agent_timings", {})

        if not timings:
            return Panel(
                "[dim]No timing data yet[/dim]",
                title="⏱️  Concurrent Execution",
                border_style="magenta"
            )

        timing_text = ""
        total_time = 0
        overlapping_time = 0

        for agent, timing in timings.items():
            if timing.get("duration"):
                duration = timing["duration"]
                total_time += duration
                timing_text += f"[cyan]{agent}:[/cyan] {duration:.2f}s\n"

        # Calculate concurrency percentage
        if state.get("layer_barriers"):
            last_barrier = state["layer_barriers"][-1]
            barrier_time = last_barrier.get("wait_time", 0)
            if barrier_time > 0 and total_time > 0:
                concurrency_pct = (1 - (barrier_time / total_time)) * 100
                timing_text += f"\n[bold green]Concurrency: {concurrency_pct:.1f}%[/bold green]"

        return Panel(
            timing_text,
            title="⏱️  Concurrent Execution",
            border_style="magenta"
        )

    async def display_stream(self, state_stream: AsyncIterator[BlogState]):
        """Display real-time execution updates.

        MODIFIED: Add concurrency panel

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
            Layout(name="concurrency", size=6),  # NEW
            Layout(name="memories", size=10),
            Layout(name="evolution", size=8),
            Layout(name="logs", size=7),
        )

        # Stream with live updates
        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["concurrency"].update(self.create_concurrency_panel(state))  # NEW
                layout["memories"].update(self.create_memory_panel(state))
                layout["evolution"].update(self.create_evolution_panel(state))
                layout["logs"].update(self.create_logs_panel(state))

    def print_summary(self, state: BlogState, show_content: bool = False):
        """Print final summary after generation.

        Args:
            state: Final workflow state
            show_content: Whether to show generated content
        """
        if not self.show_visualization:
            return

        self.console.print("\n" + "=" * 60)
        self.console.print("[bold green]✓ Generation Complete[/bold green]")
        self.console.print("=" * 60 + "\n")

        # Scores
        self.console.print("[bold]📊 Scores:[/bold]")
        for role, score in state.get("scores", {}).items():
            color = "green" if score >= 8.0 else "yellow" if score >= 6.0 else "red"
            self.console.print(f"  {role}: [{color}]{score:.1f}/10[/{color}]")

        # Memory stats
        self.console.print("\n[bold]🧠 Memory Status:[/bold]")
        for role in ["intro", "body", "conclusion"]:
            count = len(state.get("retrieved_memories", {}).get(role, []))
            self.console.print(f"  {role}: {count} memories used")

        # Parameters
        self.console.print("\n[bold]🔧 Final Parameters:[/bold]")
        for role, updates in state.get("parameter_updates", {}).items():
            if updates:
                temp = updates["new_temperature"]
                self.console.print(f"  {role}: temp={temp:.2f}")

        # Content (if requested)
        if show_content:
            self.console.print("\n[bold]📝 Generated Content:[/bold]\n")
            self.console.print(
                Panel(state.get("intro", ""), title="Introduction", border_style="cyan")
            )
            self.console.print(
                Panel(
                    state.get("body", "")[:500] + "...",
                    title="Body (preview)",
                    border_style="cyan",
                )
            )
            self.console.print(
                Panel(
                    state.get("conclusion", ""), title="Conclusion", border_style="cyan"
                )
            )
