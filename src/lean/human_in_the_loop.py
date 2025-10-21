"""
Human-in-the-Loop (HITL) Interface for LEAN

Provides interactive review and feedback capabilities at key points in the
evolutionary learning pipeline:
- Content review: Edit/approve agent outputs before scoring
- Manual scoring: Override or supplement automatic evaluation
- Evolution approval: Review and approve/reject offspring before replacement
- Pattern review: Approve reasoning patterns before storage

Usage:
    hitl = HumanInTheLoop(enabled=True, review_points=['content', 'scoring'])

    # Review content
    approved_content = hitl.review_content(
        role='intro',
        content=output,
        topic=topic,
        generation=gen_num
    )

    # Manual scoring
    score = hitl.manual_score(
        role='intro',
        content=output,
        auto_score=8.0  # Optional automatic score for reference
    )
"""

from typing import Dict, List, Optional, Literal, Tuple
from dataclasses import dataclass
from enum import Enum
import os
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, FloatPrompt
from rich.syntax import Syntax
from rich.table import Table
from rich.markdown import Markdown


class ReviewPoint(Enum):
    """Points in the pipeline where human review can occur."""
    CONTENT = "content"  # Review/edit generated content
    SCORING = "scoring"  # Manually score outputs
    EVOLUTION = "evolution"  # Approve evolution decisions
    PATTERNS = "patterns"  # Review reasoning patterns before storage


@dataclass
class ReviewResult:
    """Result of a human review."""
    approved: bool
    content: Optional[str] = None  # Modified content (if applicable)
    score: Optional[float] = None  # Manual score (if applicable)
    feedback: Optional[str] = None  # Human feedback/notes


class HumanInTheLoop:
    """Interactive human-in-the-loop interface for evolutionary pipeline.

    Enables human oversight and feedback at critical points in the learning cycle.
    Can be configured to pause execution for review, editing, and approval.

    Attributes:
        enabled: Whether HITL is active
        review_points: Which points in pipeline to pause for review
        auto_approve: If True, only pause when explicitly requested
        console: Rich console for terminal UI
    """

    def __init__(
        self,
        enabled: bool = True,
        review_points: Optional[List[str]] = None,
        auto_approve: bool = False,
        verbose: bool = True
    ):
        """Initialize human-in-the-loop interface.

        Args:
            enabled: Enable/disable HITL globally
            review_points: List of ReviewPoint values to enable
                          If None, enables all review points
            auto_approve: If True, only pause when explicitly requested
            verbose: Show detailed information during reviews
        """
        self.enabled = enabled
        self.auto_approve = auto_approve
        self.verbose = verbose
        self.console = Console()

        # Configure review points
        if review_points is None:
            self.review_points = {point.value for point in ReviewPoint}
        else:
            self.review_points = set(review_points)

        # Statistics
        self.stats = {
            'reviews_requested': 0,
            'reviews_approved': 0,
            'reviews_rejected': 0,
            'content_edits': 0,
            'manual_scores': 0
        }

    def should_review(self, review_point: str) -> bool:
        """Check if we should pause for review at this point.

        Args:
            review_point: ReviewPoint value

        Returns:
            True if review should occur
        """
        if not self.enabled:
            return False
        if self.auto_approve:
            return False
        return review_point in self.review_points

    def review_content(
        self,
        role: str,
        content: str,
        topic: str,
        generation: int,
        agent_id: Optional[str] = None,
        reasoning: Optional[str] = None
    ) -> ReviewResult:
        """Review and optionally edit generated content.

        Args:
            role: Agent role (intro, body, conclusion)
            content: Generated content to review
            topic: Current topic
            generation: Generation number
            agent_id: Optional agent identifier
            reasoning: Optional reasoning trace

        Returns:
            ReviewResult with approval status and possibly edited content
        """
        self.stats['reviews_requested'] += 1

        if not self.should_review(ReviewPoint.CONTENT.value):
            return ReviewResult(approved=True, content=content)

        self.console.print()
        self.console.print(Panel.fit(
            f"[bold cyan]Content Review - {role.upper()}[/bold cyan]\n"
            f"Generation: {generation} | Topic: {topic}",
            border_style="cyan"
        ))

        # Show content
        self.console.print(Panel(
            content,
            title=f"[yellow]{role.capitalize()} Output[/yellow]",
            border_style="yellow"
        ))

        # Show reasoning if available
        if reasoning and self.verbose:
            self.console.print(Panel(
                reasoning[:500] + "..." if len(reasoning) > 500 else reasoning,
                title="[dim]Reasoning (truncated)[/dim]",
                border_style="dim"
            ))

        # Review options
        self.console.print("\n[bold]Options:[/bold]")
        self.console.print("  [green]1.[/green] Approve as-is")
        self.console.print("  [yellow]2.[/yellow] Edit content")
        self.console.print("  [red]3.[/red] Reject (regenerate)")

        choice = Prompt.ask(
            "Choose action",
            choices=["1", "2", "3"],
            default="1"
        )

        if choice == "1":
            self.stats['reviews_approved'] += 1
            self.console.print("[green]✓ Content approved[/green]")
            return ReviewResult(approved=True, content=content)

        elif choice == "2":
            self.stats['content_edits'] += 1
            self.console.print("\n[yellow]Enter edited content (end with Ctrl+D or empty line):[/yellow]")
            edited_lines = []
            try:
                while True:
                    line = input()
                    if not line:
                        break
                    edited_lines.append(line)
            except EOFError:
                pass

            edited_content = "\n".join(edited_lines) if edited_lines else content
            self.stats['reviews_approved'] += 1
            self.console.print("[green]✓ Content edited and approved[/green]")
            return ReviewResult(approved=True, content=edited_content)

        else:  # choice == "3"
            self.stats['reviews_rejected'] += 1
            feedback = Prompt.ask("[dim]Rejection reason (optional)[/dim]", default="")
            self.console.print("[red]✗ Content rejected[/red]")
            return ReviewResult(approved=False, feedback=feedback)

    def manual_score(
        self,
        role: str,
        content: str,
        auto_score: Optional[float] = None,
        topic: Optional[str] = None,
        generation: Optional[int] = None
    ) -> Optional[float]:
        """Manually score agent output.

        Args:
            role: Agent role
            content: Content to score
            auto_score: Automatic score (for reference)
            topic: Optional topic
            generation: Optional generation number

        Returns:
            Manual score (0-10) or None if auto-score accepted
        """
        if not self.should_review(ReviewPoint.SCORING.value):
            return auto_score

        self.console.print()
        self.console.print(Panel.fit(
            f"[bold magenta]Manual Scoring - {role.upper()}[/bold magenta]",
            border_style="magenta"
        ))

        # Show content
        self.console.print(Panel(
            content[:300] + "..." if len(content) > 300 else content,
            title=f"[yellow]Content to Score[/yellow]",
            border_style="yellow"
        ))

        if auto_score is not None:
            self.console.print(f"\n[dim]Automatic score: {auto_score:.1f}/10[/dim]")

        # Score options
        use_manual = Confirm.ask(
            "Provide manual score?",
            default=False
        )

        if not use_manual:
            return auto_score

        self.stats['manual_scores'] += 1

        while True:
            score = FloatPrompt.ask(
                "Enter score (0-10)",
                default=auto_score if auto_score else 5.0
            )
            if 0 <= score <= 10:
                self.console.print(f"[green]✓ Score: {score:.1f}/10[/green]")
                return score
            else:
                self.console.print("[red]Score must be between 0 and 10[/red]")

    def review_evolution(
        self,
        role: str,
        parents: List[Dict],
        offspring: List[Dict],
        generation: int
    ) -> ReviewResult:
        """Review evolution event and approve/reject offspring.

        Args:
            role: Agent role
            parents: List of parent agent dicts
            offspring: List of offspring agent dicts
            generation: Generation number

        Returns:
            ReviewResult with approval status
        """
        if not self.should_review(ReviewPoint.EVOLUTION.value):
            return ReviewResult(approved=True)

        self.console.print()
        self.console.print(Panel.fit(
            f"[bold blue]Evolution Review - {role.upper()}[/bold blue]\n"
            f"Generation: {generation}",
            border_style="blue"
        ))

        # Parent summary
        table = Table(title="Parents", show_header=True)
        table.add_column("Agent ID", style="cyan")
        table.add_column("Fitness", style="green")
        table.add_column("Patterns", style="yellow")

        for parent in parents[:5]:  # Show top 5
            table.add_row(
                parent.get('agent_id', 'N/A')[:20],
                f"{parent.get('fitness', 0):.2f}",
                str(parent.get('pattern_count', 0))
            )

        self.console.print(table)

        # Offspring summary
        self.console.print(f"\n[bold]Offspring:[/bold] {len(offspring)} new agents")

        # Approve/reject
        approved = Confirm.ask(
            "\nApprove this evolution?",
            default=True
        )

        if approved:
            self.stats['reviews_approved'] += 1
            self.console.print("[green]✓ Evolution approved[/green]")
            return ReviewResult(approved=True)
        else:
            self.stats['reviews_rejected'] += 1
            feedback = Prompt.ask("[dim]Rejection reason (optional)[/dim]", default="")
            self.console.print("[red]✗ Evolution rejected[/red]")
            return ReviewResult(approved=False, feedback=feedback)

    def review_pattern(
        self,
        role: str,
        reasoning: str,
        score: float,
        situation: str,
        metadata: Optional[Dict] = None
    ) -> ReviewResult:
        """Review reasoning pattern before storage.

        Args:
            role: Agent role
            reasoning: Reasoning trace
            score: Quality score
            situation: Situation description
            metadata: Optional metadata

        Returns:
            ReviewResult with approval status
        """
        if not self.should_review(ReviewPoint.PATTERNS.value):
            return ReviewResult(approved=True)

        self.console.print()
        self.console.print(Panel.fit(
            f"[bold green]Pattern Review - {role.upper()}[/bold green]\n"
            f"Score: {score:.1f}/10",
            border_style="green"
        ))

        # Show pattern
        self.console.print(Panel(
            f"[bold]Situation:[/bold] {situation}\n\n"
            f"[bold]Reasoning:[/bold]\n{reasoning[:400]}{'...' if len(reasoning) > 400 else ''}",
            title="[yellow]Reasoning Pattern[/yellow]",
            border_style="yellow"
        ))

        # Approve/reject
        approved = Confirm.ask(
            "Store this pattern?",
            default=True
        )

        if approved:
            self.stats['reviews_approved'] += 1
            self.console.print("[green]✓ Pattern approved for storage[/green]")
            return ReviewResult(approved=True)
        else:
            self.stats['reviews_rejected'] += 1
            self.console.print("[red]✗ Pattern rejected[/red]")
            return ReviewResult(approved=False)

    def show_stats(self):
        """Display HITL statistics."""
        if not self.enabled:
            self.console.print("[dim]HITL is disabled[/dim]")
            return

        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]Human-in-the-Loop Statistics[/bold cyan]")
        self.console.print("=" * 60)

        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Reviews Requested", str(self.stats['reviews_requested']))
        table.add_row("Reviews Approved", str(self.stats['reviews_approved']))
        table.add_row("Reviews Rejected", str(self.stats['reviews_rejected']))
        table.add_row("Content Edits", str(self.stats['content_edits']))
        table.add_row("Manual Scores", str(self.stats['manual_scores']))

        if self.stats['reviews_requested'] > 0:
            approval_rate = (self.stats['reviews_approved'] /
                           self.stats['reviews_requested'] * 100)
            table.add_row("Approval Rate", f"{approval_rate:.1f}%")

        self.console.print(table)
        self.console.print("=" * 60 + "\n")


# Convenience function for environment-based HITL config
def create_hitl_from_env() -> HumanInTheLoop:
    """Create HITL instance from environment variables.

    Environment variables:
        ENABLE_HITL: "true" or "false" (default: false)
        HITL_REVIEW_POINTS: Comma-separated list (e.g., "content,scoring")
        HITL_AUTO_APPROVE: "true" or "false" (default: false)
        HITL_VERBOSE: "true" or "false" (default: true)

    Returns:
        Configured HumanInTheLoop instance
    """
    enabled = os.getenv("ENABLE_HITL", "false").lower() == "true"
    auto_approve = os.getenv("HITL_AUTO_APPROVE", "false").lower() == "true"
    verbose = os.getenv("HITL_VERBOSE", "true").lower() == "true"

    review_points_str = os.getenv("HITL_REVIEW_POINTS", "")
    review_points = None
    if review_points_str:
        review_points = [p.strip() for p in review_points_str.split(",")]

    return HumanInTheLoop(
        enabled=enabled,
        review_points=review_points,
        auto_approve=auto_approve,
        verbose=verbose
    )
