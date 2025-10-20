"""
Compaction Strategies Demo

Demonstrates forgetting strategies for reasoning pattern evolution.

Shows:
1. Different compaction strategies
2. How they reduce 100 patterns → 30
3. Comparison of strategies
4. Performance metrics
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from lean.compaction import (
    ScoreBasedCompaction,
    FrequencyBasedCompaction,
    DiversityPreservingCompaction,
    HybridCompaction,
    create_compaction_strategy
)


console = Console()


def create_sample_patterns(n: int = 100) -> list:
    """Create sample reasoning patterns for demo."""
    patterns = []

    # Create 3 clusters of patterns
    for cluster_id in range(3):
        for i in range(n // 3):
            # Each cluster has different score distribution
            if cluster_id == 0:  # High-quality cluster
                base_score = 8.0
                base_freq = 15
            elif cluster_id == 1:  # Medium-quality cluster
                base_score = 6.5
                base_freq = 10
            else:  # Low-quality cluster
                base_score = 5.0
                base_freq = 3

            score = base_score + np.random.rand() * 2
            freq = base_freq + int(np.random.rand() * 10)

            # Create embedding (cluster-specific)
            embedding = np.zeros(384)
            embedding[cluster_id * 100:(cluster_id + 1) * 100] = np.random.rand(100)

            patterns.append({
                'reasoning': f'Reasoning pattern {cluster_id}-{i}: Strategy for handling...',
                'score': score,
                'retrieval_count': freq,
                'embedding': embedding.tolist(),
                'situation': f'cluster_{cluster_id}',
                'timestamp': 1000.0 + i
            })

    return patterns


def demo_compaction_strategies():
    """Demonstrate all compaction strategies."""

    console.print("\n" + "=" * 70, style="bold blue")
    console.print("  COMPACTION STRATEGIES DEMO", style="bold blue")
    console.print("=" * 70 + "\n", style="bold blue")

    console.print("Creating 99 reasoning patterns (3 clusters)...\n")

    # Create patterns
    patterns = create_sample_patterns(99)

    # Strategy instances
    strategies = {
        'Score-Based': ScoreBasedCompaction(),
        'Frequency-Based': FrequencyBasedCompaction(score_weight=0.5),
        'Diversity-Preserving': DiversityPreservingCompaction(min_clusters=3),
        'Hybrid (Recommended)': HybridCompaction(
            score_weight=0.4,
            frequency_weight=0.3,
            diversity_weight=0.3
        )
    }

    # Results storage
    results = {}

    console.print(Panel.fit(
        "Testing 4 compaction strategies:\n"
        "  • Score-Based: Keep highest-scoring patterns\n"
        "  • Frequency-Based: Keep most-retrieved patterns\n"
        "  • Diversity-Preserving: Keep diverse strategies (clustering)\n"
        "  • Hybrid: Balance score, frequency, and diversity",
        title="Strategies",
        border_style="green"
    ))
    console.print()

    # Run each strategy
    for name, strategy in strategies.items():
        console.print(f"Running [cyan]{name}[/cyan]...")

        compacted = strategy.compact(patterns, max_size=30)

        # Calculate metrics
        avg_score = np.mean([p['score'] for p in compacted])
        avg_freq = np.mean([p['retrieval_count'] for p in compacted])

        # Count patterns from each cluster
        cluster_counts = {}
        for p in compacted:
            cluster = p['situation']
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        diversity_score = len(cluster_counts)  # Simple diversity metric

        results[name] = {
            'compacted': compacted,
            'avg_score': avg_score,
            'avg_freq': avg_freq,
            'diversity': diversity_score,
            'cluster_counts': cluster_counts,
            'stats': strategy.get_stats()
        }

    console.print("\n" + "=" * 70 + "\n", style="bold green")

    # Display results table
    table = Table(title="Compaction Results (99 → 30 patterns)")

    table.add_column("Strategy", style="cyan", no_wrap=True)
    table.add_column("Avg Score", justify="right", style="magenta")
    table.add_column("Avg Frequency", justify="right", style="yellow")
    table.add_column("Diversity\n(clusters)", justify="right", style="green")
    table.add_column("Cluster Distribution", style="white")

    for name, result in results.items():
        cluster_dist = ", ".join(
            f"{k}: {v}" for k, v in result['cluster_counts'].items()
        )

        table.add_row(
            name,
            f"{result['avg_score']:.2f}",
            f"{result['avg_freq']:.1f}",
            f"{result['diversity']}/3",
            cluster_dist
        )

    console.print(table)
    console.print()

    # Analysis
    console.print(Panel.fit(
        "[bold]Analysis:[/bold]\n\n"
        "• [cyan]Score-Based[/cyan]: Highest avg score, but may lose diversity\n"
        "• [cyan]Frequency-Based[/cyan]: Keeps battle-tested patterns\n"
        "• [cyan]Diversity-Preserving[/cyan]: Best diversity, balanced quality\n"
        "• [cyan]Hybrid[/cyan]: Best overall balance (recommended for evolution)",
        title="Strategy Comparison",
        border_style="yellow"
    ))
    console.print()

    # Show sample patterns from best strategy
    console.print(Panel.fit(
        "[bold]Sample Compacted Patterns (Hybrid Strategy):[/bold]",
        border_style="green"
    ))

    hybrid_compacted = results['Hybrid (Recommended)']['compacted']
    for i, pattern in enumerate(hybrid_compacted[:5]):
        console.print(
            f"  {i+1}. Score: {pattern['score']:.1f}, "
            f"Freq: {pattern['retrieval_count']}, "
            f"Cluster: {pattern['situation']}"
        )

    console.print(f"  ... and {len(hybrid_compacted) - 5} more patterns\n")

    # Performance stats
    console.print(Panel.fit(
        "[bold]Compaction Statistics:[/bold]\n\n"
        f"• Before: 99 patterns\n"
        f"• After: 30 patterns\n"
        f"• Compaction rate: {30/99:.1%}\n"
        f"• Patterns forgotten: 69 (lowest-quality)",
        title="Summary",
        border_style="blue"
    ))


def demo_factory_usage():
    """Demonstrate factory function."""

    console.print("\n" + "=" * 70, style="bold blue")
    console.print("  FACTORY FUNCTION DEMO", style="bold blue")
    console.print("=" * 70 + "\n", style="bold blue")

    console.print("Creating strategies using factory function...\n")

    # Create strategies
    strategies = [
        ('score', {}),
        ('frequency', {'score_weight': 0.7}),
        ('diversity', {'min_clusters': 5}),
        ('hybrid', {'score_weight': 0.5, 'frequency_weight': 0.3, 'diversity_weight': 0.2})
    ]

    for name, kwargs in strategies:
        strategy = create_compaction_strategy(name, **kwargs)
        console.print(f"  ✓ Created [cyan]{name}[/cyan] strategy: {strategy.__class__.__name__}")
        if kwargs:
            console.print(f"    Parameters: {kwargs}")

    console.print()


def demo_performance():
    """Demonstrate performance with large dataset."""

    console.print("\n" + "=" * 70, style="bold blue")
    console.print("  PERFORMANCE DEMO", style="bold blue")
    console.print("=" * 70 + "\n", style="bold blue")

    import time

    console.print("Testing compaction performance with 1000 patterns...\n")

    # Create large dataset
    patterns = create_sample_patterns(1000)

    strategy = HybridCompaction()

    start = time.time()
    compacted = strategy.compact(patterns, max_size=30)
    elapsed = time.time() - start

    console.print(Panel.fit(
        f"[bold]Performance Results:[/bold]\n\n"
        f"• Dataset size: 1000 patterns\n"
        f"• Compacted to: 30 patterns\n"
        f"• Time: {elapsed:.3f} seconds\n"
        f"• Rate: {1000/elapsed:.0f} patterns/second",
        title="Performance",
        border_style="green"
    ))

    console.print()


if __name__ == '__main__':
    demo_compaction_strategies()
    demo_factory_usage()
    demo_performance()

    console.print("\n" + "=" * 70, style="bold green")
    console.print("  DEMO COMPLETE", style="bold green")
    console.print("=" * 70 + "\n", style="bold green")

    console.print(
        "[bold]Next Steps:[/bold]\n"
        "1. Use compaction in ReproductionStrategy\n"
        "2. Compact parent patterns before inheritance\n"
        "3. Pass only best 20-30 patterns to offspring\n"
        "4. Run 20-generation evolution experiment\n"
    )
