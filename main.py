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

    for i, topic in enumerate(topics, 1):
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Generation {i}: {topic}[/bold]")
        console.print("=" * 60)

        # Generate
        result = await pipeline.generate(topic)

        # Display summary
        pipeline.visualizer.print_summary(result, show_content=False)

        # Calculate overall score
        overall = calculate_overall_score(result["scores"])
        console.print(f"\n[bold]Overall Score: {overall}/10[/bold]")

        # Show learning progress
        if i > 1:
            console.print("\n[bold cyan]📈 Learning Progress:[/bold cyan]")
            stats = pipeline.get_memory_stats()
            for role, stat in stats.items():
                total = stat["total_memories"]
                avg = stat.get("avg_score", 0)
                console.print(
                    f"  {role}: {total} memories | " f"avg quality: {avg:.1f}"
                )

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
        console.print(f"    Total memories: {stat['total_memories']}")
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
