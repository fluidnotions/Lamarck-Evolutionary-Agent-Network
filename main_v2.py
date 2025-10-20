"""
Main entry point for LEAN V2 with reasoning pattern architecture.

This version uses:
- BaseAgentV2 agents with <think>/<final> externalization
- ReasoningMemory for cognitive pattern storage
- SharedRAG for domain knowledge
- ContextManager for reasoning trace distribution
- 8-step learning cycle

Run: python main_v2.py
"""

import asyncio
import os
from dotenv import load_dotenv

from src.lean.pipeline_v2 import PipelineV2

load_dotenv()


async def main():
    """Run V2 pipeline with reasoning patterns."""

    print("\n" + "="*70)
    print("  LEAN V2: Lamarck Evolutionary Agent Network")
    print("  Reasoning Pattern Architecture")
    print("="*70 + "\n")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not set")
        print("Please set your API key in the .env file")
        return

    # Initialize pipeline with M2 evolution
    print("Initializing evolutionary pipeline...")

    POPULATION_SIZE = 3  # Agents per role
    EVOLUTION_FREQUENCY = 5  # Evolve every N generations
    TOTAL_GENERATIONS = 20  # Total generations to run

    pipeline = PipelineV2(
        reasoning_dir="./data/reasoning",
        shared_rag_dir="./data/shared_rag",
        domain="General",
        population_size=POPULATION_SIZE,
        evolution_frequency=EVOLUTION_FREQUENCY
    )

    print("✅ Pipeline V2 initialized with M2 Evolution")
    print(f"  - Population: {POPULATION_SIZE} agents per role")
    print(f"  - Evolution frequency: every {EVOLUTION_FREQUENCY} generations")
    print(f"  - Total generations: {TOTAL_GENERATIONS}")
    print("  - Using reasoning patterns + shared RAG")
    print("  - Full 8-step learning cycle with inheritance")
    print()

    # Define topics - pairs of similar topics to test transfer learning
    # Evolution happens at gen 5, 10, 15, 20
    # Similar topics test if inherited reasoning helps on related content
    topics = [
        # Block 1: AI fundamentals (gens 1-4, before first evolution)
        "The Future of Artificial Intelligence",
        "Machine Learning Fundamentals",
        "Neural Networks Explained",
        "Deep Learning Basics",

        # Block 2: After first evolution (gen 5)
        "AI in Healthcare Applications",  # Similar to previous
        "Medical Diagnosis with Machine Learning",
        "Neural Networks for Patient Care",
        "Deep Learning in Medical Imaging",
        "AI-Powered Drug Discovery",

        # Block 3: After second evolution (gen 10)
        "The Ethics of Artificial Intelligence",
        "AI Safety and Alignment",
        "Machine Learning Bias and Fairness",
        "Responsible AI Development",
        "AI Governance Frameworks",

        # Block 4: After third evolution (gen 15)
        "Climate Change and AI Solutions",
        "Machine Learning for Environmental Science",
        "Neural Networks Predicting Weather",
        "Deep Learning for Sustainability",
        "AI-Driven Climate Modeling",
    ]

    # Limit to requested generations
    topics = topics[:TOTAL_GENERATIONS]

    print(f"Running {len(topics)} generations with evolutionary learning...")
    print(f"Evolution events at generations: {', '.join(str(i) for i in range(EVOLUTION_FREQUENCY, TOTAL_GENERATIONS + 1, EVOLUTION_FREQUENCY))}\n")

    results = []

    for gen_num, topic in enumerate(topics, start=1):
        print(f"{'='*70}")
        print(f"  Generation {gen_num}/{len(topics)}: {topic}")
        print(f"{'='*70}\n")

        # Run generation
        final_state = await pipeline.generate(
            topic=topic,
            generation_number=gen_num
        )

        # Collect results
        result = {
            'generation': gen_num,
            'topic': topic,
            'scores': final_state['scores'],
            'avg_score': sum(final_state['scores'].values()) / len(final_state['scores']),
            'reasoning_patterns_used': sum(final_state['reasoning_patterns_used'].values()),
            'domain_knowledge_used': sum(final_state['domain_knowledge_used'].values())
        }
        results.append(result)

        print(f"\nGeneration {gen_num} complete:")
        print(f"  Intro score: {final_state['scores']['intro']:.1f}")
        print(f"  Body score: {final_state['scores']['body']:.1f}")
        print(f"  Conclusion score: {final_state['scores']['conclusion']:.1f}")
        print(f"  Average: {result['avg_score']:.1f}")
        print(f"  Reasoning patterns used: {result['reasoning_patterns_used']}")
        print(f"  Domain knowledge used: {result['domain_knowledge_used']}")
        print()

    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70 + "\n")

    print("Generation Progress:")
    print("Gen | Topic                              | Avg Score | Patterns | Knowledge")
    print("-" * 85)
    for result in results:
        topic_short = result['topic'][:35].ljust(35)
        print(f" {result['generation']}  | {topic_short} |   {result['avg_score']:.2f}    |    {result['reasoning_patterns_used']}     |     {result['domain_knowledge_used']}")

    # Pool statistics (M2 evolution)
    print("\nEvolutionary Pool Statistics:")
    pool_stats = pipeline.get_agent_stats()
    for role, stats in pool_stats.items():
        print(f"\n{role.capitalize()} Pool:")
        print(f"  Generation: {stats['generation']}")
        print(f"  Pool size: {stats['pool_size']} agents")
        print(f"  Avg fitness: {stats['avg_fitness']:.2f}")
        print(f"  Top agent fitness: {stats['top_agent_fitness']:.2f}")
        print(f"  Diversity: {stats['diversity']:.3f}")

    # Shared RAG
    print("\nShared RAG:")
    rag_stats = pipeline.get_shared_rag_stats()
    print(f"  Total knowledge: {rag_stats['total_knowledge']}")
    print(f"  By source: {rag_stats['by_source']}")

    # Evolution impact analysis
    print("\nEvolutionary Learning Analysis:")

    # Compare performance before and after each evolution
    evolution_points = list(range(EVOLUTION_FREQUENCY, TOTAL_GENERATIONS + 1, EVOLUTION_FREQUENCY))

    for i, evo_gen in enumerate(evolution_points):
        if evo_gen < len(results):
            # Pre-evolution (last 2 gens before evolution)
            pre_start = max(0, evo_gen - EVOLUTION_FREQUENCY)
            pre_end = evo_gen
            pre_scores = [r['avg_score'] for r in results[pre_start:pre_end]]
            pre_avg = sum(pre_scores) / len(pre_scores) if pre_scores else 0

            # Post-evolution (first 2 gens after evolution)
            post_start = evo_gen
            post_end = min(len(results), evo_gen + 2)
            post_scores = [r['avg_score'] for r in results[post_start:post_end]]
            post_avg = sum(post_scores) / len(post_scores) if post_scores else 0

            improvement = post_avg - pre_avg

            print(f"\nEvolution {i+1} (generation {evo_gen}):")
            print(f"  Pre-evolution avg:  {pre_avg:.2f}")
            print(f"  Post-evolution avg: {post_avg:.2f}")
            print(f"  Change: {improvement:+.2f} points {'✅' if improvement > 0 else '⚠️'}")

    # Overall trend
    if len(results) >= 2:
        first_avg = results[0]['avg_score']
        last_avg = results[-1]['avg_score']
        total_improvement = last_avg - first_avg

        print(f"\nOverall Learning Trend:")
        print(f"  Generation 1 avg:  {first_avg:.2f}")
        print(f"  Generation {len(results)} avg: {last_avg:.2f}")
        print(f"  Total improvement: {total_improvement:+.2f} points")

        if total_improvement > 0.5:
            print(f"  ✅ Strong positive learning detected!")
        elif total_improvement > 0:
            print(f"  ✅ Modest improvement observed")
        elif total_improvement < -0.5:
            print(f"  ⚠️  Performance decreased (exploration phase?)")
        else:
            print(f"  → Stable performance")

    print("\n" + "="*70)
    print("  Experiment Complete")
    print("="*70 + "\n")

    print("Data stored in:")
    print("  - ./data/reasoning/ (reasoning patterns)")
    print("  - ./data/shared_rag/ (domain knowledge)")


if __name__ == "__main__":
    asyncio.run(main())
