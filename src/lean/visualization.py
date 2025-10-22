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


class HierarchicalVisualizer(StreamVisualizer):
    """Enhanced visualizer for hierarchical ensemble architecture.

    Displays:
    - 3-layer hierarchy (Coordinator → Content Agents → Specialists)
    - Agent pool evolution (population, fitness, diversity)
    - Coordinator workflow (research → distribute → critique)
    - Memory retrieval with inheritance tracking
    - Revision loop progress
    """

    def __init__(self, pipeline=None):
        """Initialize hierarchical visualizer.

        Args:
            pipeline: Pipeline instance (provides access to pools, coordinator, specialists)
        """
        super().__init__()
        self.pipeline = pipeline

    def create_status_table(self, state: BlogState) -> Table:
        """Create hierarchical status table showing all 3 layers.

        Args:
            state: Current workflow state

        Returns:
            Rich Table with hierarchical agent status
        """
        table = Table(title="🤖 Agent Execution Status (Hierarchical)", show_header=True)
        table.add_column("Layer", style="dim", no_wrap=True, width=5)
        table.add_column("Agent", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        table.add_column("Pool", style="yellow", width=7)
        table.add_column("Memories", style="green", width=8)
        table.add_column("Score", style="blue", width=6)

        # Layer 1: Coordinator
        coord_status = self._get_coordinator_status(state)
        coord_memories = state.get("reasoning_patterns_used", {}).get("coordinator", 0)
        table.add_row(
            "L1",
            "Coordinator",
            coord_status,
            "-",
            str(coord_memories) if coord_memories > 0 else "-",
            "-"
        )

        # Layer 2: Content Agents (in pools)
        for role in ["intro", "body", "conclusion"]:
            status = "✓ Done" if state.get(role) else "⟳ Gen" if state.get(f"{role}_reasoning") else "⏸ Wait"

            # Pool info
            pool_info = "-"
            if self.pipeline and hasattr(self.pipeline, 'agent_pools'):
                pool = self.pipeline.agent_pools.get(role)
                if pool:
                    pool_info = f"{pool.size()}"

            # Memory info
            reasoning_count = state.get("reasoning_patterns_used", {}).get(role, 0)
            knowledge_count = state.get("domain_knowledge_used", {}).get(role, 0)
            mem_info = f"R:{reasoning_count} K:{knowledge_count}" if (reasoning_count + knowledge_count) > 0 else "-"

            # Score
            score = state.get("scores", {}).get(role, 0.0)
            score_str = f"{score:.1f}" if score > 0 else "-"

            table.add_row(
                "L2",
                f"{role.capitalize()}",
                status,
                pool_info,
                mem_info,
                score_str
            )

        # Layer 3: Specialists (if enabled)
        if self.pipeline and hasattr(self.pipeline, 'specialists') and self.pipeline.specialists:
            for spec_name in ["researcher", "fact_checker", "stylist"]:
                if spec_name in self.pipeline.specialists:
                    # Check if specialist was used (placeholder - would need tracking in state)
                    status = "○ Ready"
                    table.add_row(
                        "L3",
                        spec_name.replace("_", " ").title(),
                        status,
                        "-",
                        "-",
                        "-"
                    )

        return table

    def _get_coordinator_status(self, state: BlogState) -> str:
        """Determine coordinator status from state.

        Args:
            state: Current workflow state

        Returns:
            Status string
        """
        # Check for coordinator critique
        if state.get("coordinator_critique"):
            return "✓ Critique"

        # Check if content is being generated
        if state.get("intro") or state.get("body") or state.get("conclusion"):
            return "⟳ Aggr"

        # Check for research/distribute phase
        if state.get("stream_logs"):
            recent_logs = state.get("stream_logs", [])[-3:]
            for log in recent_logs:
                if "research" in log.lower() or "tavily" in log.lower():
                    return "⟳ Research"
                if "distribute" in log.lower() or "context" in log.lower():
                    return "⟳ Dist"

        return "○ Ready"

    def create_pool_evolution_panel(self, state: BlogState) -> Panel:
        """Show agent pool evolution status.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with pool evolution information
        """
        if not self.pipeline or not hasattr(self.pipeline, 'agent_pools'):
            return Panel(
                "[dim]Pool information not available[/dim]",
                title="🧬 Agent Pool Evolution",
                border_style="magenta"
            )

        evo_text = ""

        # Generation info
        gen_num = state.get("generation_number", 0)
        if hasattr(self.pipeline, 'evolution_frequency'):
            evo_freq = self.pipeline.evolution_frequency
            next_evo = ((gen_num // evo_freq) + 1) * evo_freq
            evo_text += f"[bold]Generation:[/bold] {gen_num} (Next evolution: {next_evo})\n\n"

        # Pool stats for each role
        for role in ["intro", "body", "conclusion"]:
            pool = self.pipeline.agent_pools.get(role)
            if pool:
                active_agent = pool.select_active_agent() if pool.size() > 0 else None
                active_fitness = active_agent.avg_fitness() if active_agent else 0.0

                evo_text += f"[bold cyan]{role.upper()} POOL:[/bold cyan]\n"
                evo_text += f"  Active: {active_agent.agent_id if active_agent else 'None'} "
                evo_text += f"(fitness: {active_fitness:.1f})\n"
                evo_text += f"  Population: {pool.size()} agents | "
                evo_text += f"Avg Fitness: {pool.avg_fitness():.1f} | "
                evo_text += f"Diversity: {pool.measure_diversity():.2f}\n\n"

        # Check for recent evolution events in logs
        recent_logs = state.get("stream_logs", [])[-10:]
        for log in recent_logs:
            if "evolution" in log.lower() or "offspring" in log.lower():
                evo_text += f"[yellow]🧬 {log}[/yellow]\n"

        if not evo_text:
            evo_text = "[dim]No pool data yet[/dim]"

        return Panel(
            evo_text,
            title="🧬 Agent Pool Evolution",
            border_style="magenta"
        )

    def create_coordinator_panel(self, state: BlogState) -> Panel:
        """Show coordinator workflow progress.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with coordinator activity
        """
        coord_text = ""

        # Workflow phases
        phases = [
            ("1. RESEARCH", "✓" if "research" in str(state.get("stream_logs", [])).lower() else "○"),
            ("2. DISTRIBUTE", "✓" if "distribute" in str(state.get("stream_logs", [])).lower() else "○"),
            ("3-5. GENERATE", "⟳" if any(state.get(r) for r in ["intro", "body", "conclusion"]) else "○"),
            ("6. AGGREGATE", "✓" if all(state.get(r) for r in ["intro", "body", "conclusion"]) else "○"),
            ("7. CRITIQUE", "✓" if state.get("coordinator_critique") else "○"),
        ]

        for phase, status in phases:
            coord_text += f"[{phase}] {status}\n"

        # Revision loop info
        revision_count = state.get("revision_count", 0)
        max_revisions = 2  # Default, could get from pipeline
        if self.pipeline and hasattr(self.pipeline, 'max_revisions'):
            max_revisions = self.pipeline.max_revisions

        coord_text += f"\n[bold]Revision Loop:[/bold] {revision_count}/{max_revisions} iterations\n"

        # Critique summary if available
        if state.get("coordinator_critique"):
            critique = state["coordinator_critique"]
            if isinstance(critique, dict):
                scores = critique.get("scores", {})
                overall = scores.get("overall", 0)
                coord_text += f"\n[bold]Quality Score:[/bold] {overall:.1f}/10\n"
                if critique.get("feedback"):
                    feedback_preview = critique["feedback"][:80]
                    coord_text += f"[dim]{feedback_preview}...[/dim]\n"

        return Panel(
            coord_text,
            title="🎯 Coordinator Activity",
            border_style="cyan"
        )

    def create_memory_panel(self, state: BlogState) -> Panel:
        """Show retrieved memories with inheritance tracking.

        Args:
            state: Current workflow state

        Returns:
            Rich Panel with hierarchical memory information
        """
        mem_text = ""

        # Coordinator memories
        coord_reasoning = state.get("reasoning_patterns_used", {}).get("coordinator", 0)
        coord_knowledge = state.get("domain_knowledge_used", {}).get("coordinator", 0)
        if coord_reasoning + coord_knowledge > 0:
            mem_text += "[bold cyan]COORDINATOR (L1):[/bold cyan]\n"
            mem_text += f"  Reasoning: {coord_reasoning} patterns | Domain: {coord_knowledge} facts\n\n"

        # Content agent memories
        for role in ["intro", "body", "conclusion"]:
            reasoning_count = state.get("reasoning_patterns_used", {}).get(role, 0)
            knowledge_count = state.get("domain_knowledge_used", {}).get(role, 0)

            if reasoning_count + knowledge_count > 0:
                mem_text += f"[bold cyan]{role.upper()} (L2):[/bold cyan]\n"
                mem_text += f"  Reasoning: {reasoning_count} patterns | Domain: {knowledge_count} facts\n"

                # Show reasoning preview if available
                reasoning_preview = state.get(f"{role}_reasoning", "")
                if reasoning_preview:
                    preview_lines = reasoning_preview.split('\n')[:2]
                    for line in preview_lines:
                        if line.strip():
                            mem_text += f"  [dim]{line[:60]}...[/dim]\n"

                mem_text += "\n"

        if not mem_text:
            mem_text = "[dim]No memories retrieved yet[/dim]"

        return Panel(
            mem_text,
            title="🧠 Retrieved Memories (Hierarchical)",
            border_style="green"
        )

    async def display_stream(self, state_stream: AsyncIterator[BlogState]):
        """Display real-time hierarchical execution updates.

        Args:
            state_stream: Async iterator of state updates
        """
        if not self.show_visualization:
            # Just consume the stream without displaying
            async for _ in state_stream:
                pass
            return

        # Create hierarchical layout
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=12),       # Hierarchical status
            Layout(name="coordinator", size=10),  # Coordinator workflow
            Layout(name="pools", size=12),        # Pool evolution
            Layout(name="memories", size=12),     # Hierarchical memories
            Layout(name="logs", size=8),          # Activity logs
        )

        # Stream with live updates
        with Live(layout, console=self.console, refresh_per_second=4) as live:
            async for state in state_stream:
                layout["status"].update(self.create_status_table(state))
                layout["coordinator"].update(self.create_coordinator_panel(state))
                layout["pools"].update(self.create_pool_evolution_panel(state))
                layout["memories"].update(self.create_memory_panel(state))
                layout["logs"].update(self.create_logs_panel(state))
