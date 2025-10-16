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
        "artificial intelligence ethics",
    ]

    console.print("\n[bold cyan]HVAS Mini - Hierarchical Vector Agent System[/bold cyan]")
    console.print("Demonstrating learning across multiple generations\n")

    # Track generation history for persistent display
    generation_history = []

    for i, topic in enumerate(topics, 1):
        # Display persistent history summary at top
        if generation_history:
            console.print("\n" + "=" * 80)
            console.print("[bold cyan]📊 GENERATION HISTORY[/bold cyan]")
            console.print("=" * 80)

            from rich.table import Table
            history_table = Table(show_header=True, header_style="bold magenta")
            history_table.add_column("#", style="dim", width=3)
            history_table.add_column("Topic", style="cyan", width=30)
            history_table.add_column("Intro", style="green", justify="center", width=6)
            history_table.add_column("Body", style="green", justify="center", width=6)
            history_table.add_column("Concl", style="green", justify="center", width=6)
            history_table.add_column("Overall", style="yellow", justify="center", width=8)
            history_table.add_column("Memories", style="blue", justify="center", width=10)

            for idx, gen in enumerate(generation_history, 1):
                topic_display = gen["topic"][:28] + "..." if len(gen["topic"]) > 28 else gen["topic"]
                history_table.add_row(
                    str(idx),
                    topic_display,
                    f"{gen['scores']['intro']:.1f}",
                    f"{gen['scores']['body']:.1f}",
                    f"{gen['scores']['conclusion']:.1f}",
                    f"[bold]{gen['overall']:.1f}[/bold]",
                    f"{gen['total_memories']}"
                )

            console.print(history_table)
            console.print()

        console.print(f"\n{'='*80}")
        console.print(f"[bold]🚀 Generation {i}/{len(topics)}: {topic}[/bold]")
        console.print("=" * 80)

        # Generate
        result = await pipeline.generate(topic)

        # Display summary
        pipeline.visualizer.print_summary(result, show_content=False)

        # Calculate overall score
        overall = calculate_overall_score(result["scores"])
        console.print(f"\n[bold]Overall Score: {overall}/10[/bold]")

        # Show learning progress
        stats = pipeline.get_memory_stats()
        total_memories = sum(stat.get("total_memories", 0) for stat in stats.values())

        if i > 1:
            console.print("\n[bold cyan]📈 Learning Progress:[/bold cyan]")
            for role, stat in stats.items():
                total = stat.get("total_memories", 0)
                avg = stat.get("avg_score", 0)
                console.print(
                    f"  {role}: {total} memories | " f"avg quality: {avg:.1f}"
                )

        # Store this generation in history
        generation_history.append({
            "topic": topic,
            "scores": result["scores"].copy(),
            "overall": overall,
            "total_memories": total_memories
        })

        # Brief pause between generations
        await asyncio.sleep(1)

    # Final statistics
    console.print("\n" + "=" * 60)
    console.print("[bold green]✓ All Generations Complete[/bold green]")
    console.print("=" * 60 + "\n")

    console.print("[bold]🎓 Final System State:[/bold]\n")

    # Memory statistics
    console.print("[bold cyan]Memory Statistics:[/bold cyan]")
    stats = pipeline.get_memory_stats()
    for role, stat in stats.items():
        console.print(f"\n  {role.upper()}:")
        console.print(f"    Total memories: {stat.get('total_memories', 0)}")
        console.print(f"    Avg score: {stat.get('avg_score', 0):.1f}")
        console.print(f"    Total retrievals: {stat.get('total_retrievals', 0)}")

    # Agent parameters
    console.print("\n[bold yellow]Agent Parameters:[/bold yellow]")
    params = pipeline.get_agent_parameters()
    for role, param in params.items():
        console.print(f"\n  {role.upper()}:")
        console.print(f"    Temperature: {param['temperature']:.2f}")
        console.print(f"    Generations: {param['generation_count']}")
        if param["score_history"]:
            avg = sum(param["score_history"]) / len(param["score_history"])
            console.print(f"    Avg score: {avg:.1f}")


if __name__ == "__main__":
    asyncio.run(main())
