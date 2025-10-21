"""
Human-in-the-Loop Demo

Demonstrates the interactive review capabilities of LEAN's HITL interface.
Run this script to see how human oversight works at different pipeline stages.

Usage:
    # Interactive mode (pauses for review)
    ENABLE_HITL=true python examples/hitl_demo.py

    # Review only scoring
    ENABLE_HITL=true HITL_REVIEW_POINTS=scoring python examples/hitl_demo.py

    # Auto-approve mode (just shows what would be reviewed)
    ENABLE_HITL=true HITL_AUTO_APPROVE=true python examples/hitl_demo.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lean.human_in_the_loop import HumanInTheLoop, create_hitl_from_env, ReviewPoint
from rich.console import Console
from rich.panel import Panel

console = Console()


def demo_content_review():
    """Demonstrate content review functionality."""
    console.print("\n[bold cyan]=== Demo 1: Content Review ===[/bold cyan]")

    hitl = HumanInTheLoop(
        enabled=True,
        review_points=[ReviewPoint.CONTENT.value],
        verbose=True
    )

    # Simulate agent output
    sample_content = """
Artificial Intelligence represents a transformative technology that is reshaping
industries and society. From healthcare diagnostics to autonomous vehicles, AI
systems are demonstrating capabilities that were once thought to be exclusively human.

The field has evolved through several waves of innovation, from rule-based expert
systems to modern deep learning architectures that can process vast amounts of data
and identify complex patterns.
    """.strip()

    sample_reasoning = """
I should start with a broad overview of AI's impact, then narrow down to specific
applications. The hook needs to be engaging but not sensationalist. I'll mention
both the technical evolution and practical applications to appeal to different readers.
    """.strip()

    result = hitl.review_content(
        role="intro",
        content=sample_content,
        topic="The Future of Artificial Intelligence",
        generation=5,
        agent_id="intro_gen5_child1",
        reasoning=sample_reasoning
    )

    if result.approved:
        console.print(f"\n[green]Content was approved![/green]")
        if result.content != sample_content:
            console.print("[yellow]Content was edited by human[/yellow]")
    else:
        console.print(f"\n[red]Content was rejected![/red]")
        if result.feedback:
            console.print(f"[dim]Reason: {result.feedback}[/dim]")


def demo_manual_scoring():
    """Demonstrate manual scoring functionality."""
    console.print("\n[bold magenta]=== Demo 2: Manual Scoring ===[/bold magenta]")

    hitl = HumanInTheLoop(
        enabled=True,
        review_points=[ReviewPoint.SCORING.value]
    )

    sample_content = """
Machine learning algorithms excel at pattern recognition tasks that would be
impossible for humans to perform manually at scale. By training on millions
of examples, these systems can learn to classify images, translate languages,
and predict outcomes with remarkable accuracy.
    """.strip()

    score = hitl.manual_score(
        role="body",
        content=sample_content,
        auto_score=8.5,
        topic="Machine Learning Fundamentals",
        generation=3
    )

    if score is not None:
        console.print(f"\n[green]Final score: {score:.1f}/10[/green]")
    else:
        console.print("\n[yellow]Using automatic score[/yellow]")


def demo_evolution_review():
    """Demonstrate evolution approval functionality."""
    console.print("\n[bold blue]=== Demo 3: Evolution Review ===[/bold blue]")

    hitl = HumanInTheLoop(
        enabled=True,
        review_points=[ReviewPoint.EVOLUTION.value]
    )

    # Simulate parent/offspring data
    parents = [
        {'agent_id': 'intro_gen4_agent1', 'fitness': 7.8, 'pattern_count': 15},
        {'agent_id': 'intro_gen4_agent2', 'fitness': 7.5, 'pattern_count': 12},
        {'agent_id': 'intro_gen4_agent3', 'fitness': 7.2, 'pattern_count': 18},
        {'agent_id': 'intro_gen4_agent4', 'fitness': 6.9, 'pattern_count': 10},
        {'agent_id': 'intro_gen4_agent5', 'fitness': 6.5, 'pattern_count': 14},
    ]

    offspring = [
        {'agent_id': 'intro_gen5_child1', 'inherited_patterns': 20},
        {'agent_id': 'intro_gen5_child2', 'inherited_patterns': 20},
        {'agent_id': 'intro_gen5_child3', 'inherited_patterns': 20},
        {'agent_id': 'intro_gen5_child4', 'inherited_patterns': 20},
        {'agent_id': 'intro_gen5_child5', 'inherited_patterns': 20},
    ]

    result = hitl.review_evolution(
        role="intro",
        parents=parents,
        offspring=offspring,
        generation=5
    )

    if result.approved:
        console.print(f"\n[green]Evolution approved! Population replaced.[/green]")
    else:
        console.print(f"\n[red]Evolution rejected! Keeping current population.[/red]")


def demo_pattern_review():
    """Demonstrate pattern storage review functionality."""
    console.print("\n[bold green]=== Demo 4: Pattern Review ===[/bold green]")

    hitl = HumanInTheLoop(
        enabled=True,
        review_points=[ReviewPoint.PATTERN.value]
    )

    sample_reasoning = """
The key challenge here is balancing technical accuracy with accessibility.
I should explain the concept using analogies familiar to most readers, then
gradually introduce more technical terminology. Starting with a real-world
example (like spam filtering) makes the abstract concept more concrete.

I'll structure this with: 1) relatable example, 2) core mechanism explanation,
3) broader implications. This progression helps readers build understanding
incrementally.
    """.strip()

    result = hitl.review_pattern(
        role="body",
        reasoning=sample_reasoning,
        score=8.5,
        situation="Explain machine learning to general audience",
        metadata={'topic': 'ML Fundamentals', 'generation': 3}
    )

    if result.approved:
        console.print(f"\n[green]Pattern approved for storage![/green]")
    else:
        console.print(f"\n[red]Pattern rejected - will not be stored.[/red]")


def demo_statistics():
    """Demonstrate statistics tracking."""
    console.print("\n[bold yellow]=== Demo 5: Statistics ===[/bold yellow]")

    hitl = HumanInTheLoop(enabled=True)

    # Simulate some reviews
    hitl.stats = {
        'reviews_requested': 15,
        'reviews_approved': 12,
        'reviews_rejected': 3,
        'content_edits': 2,
        'manual_scores': 5
    }

    hitl.show_stats()


def main():
    """Run all demos."""
    console.print(Panel.fit(
        "[bold white]LEAN Human-in-the-Loop Demo[/bold white]\n\n"
        "This demo shows how HITL enables interactive oversight of the\n"
        "evolutionary learning pipeline. You can review content, provide\n"
        "manual scores, approve evolution events, and review patterns.\n\n"
        "[dim]Press Ctrl+C to skip any demo[/dim]",
        border_style="bold white"
    ))

    demos = [
        ("Content Review", demo_content_review),
        ("Manual Scoring", demo_manual_scoring),
        ("Evolution Approval", demo_evolution_review),
        ("Pattern Review", demo_pattern_review),
        ("Statistics", demo_statistics)
    ]

    for name, demo_func in demos:
        try:
            demo_func()
            input("\n[dim]Press Enter to continue to next demo...[/dim]")
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Skipping {name} demo[/yellow]")
            continue
        except Exception as e:
            console.print(f"\n[red]Error in {name} demo: {e}[/red]")
            continue

    console.print("\n[bold green]✓ Demo complete![/bold green]\n")

    # Show environment-based configuration
    console.print("[bold cyan]Environment Configuration Example:[/bold cyan]")
    console.print("""
To use HITL in your experiments:

    # Enable content and scoring review
    export ENABLE_HITL=true
    export HITL_REVIEW_POINTS=content,scoring

    # Run experiment
    python main_v2.py

Or inline:
    ENABLE_HITL=true HITL_REVIEW_POINTS=scoring python main_v2.py
    """)


if __name__ == "__main__":
    main()
