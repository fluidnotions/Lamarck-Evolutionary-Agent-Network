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

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = PipelineV2(
        reasoning_dir="./data/reasoning",
        shared_rag_dir="./data/shared_rag",
        agent_ids={
            'intro': 'agent_1',
            'body': 'agent_1',
            'conclusion': 'agent_1'
        },
        domain="General"
    )

    print("✅ Pipeline V2 initialized")
    print("  - Using BaseAgentV2 agents")
    print("  - Reasoning patterns + Shared RAG")
    print("  - 8-step learning cycle")
    print()

    # Define topics (alternating similar topics to test learning)
    topics = [
        "The Future of Artificial Intelligence",
        "Machine Learning Fundamentals",
        "Neural Networks Explained",
        "Deep Learning Applications",
        "AI Safety and Ethics",
    ]

    print(f"Running {len(topics)} generations...\n")

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

    # Agent statistics
    print("\nAgent Statistics:")
    agent_stats = pipeline.get_agent_stats()
    for role, stats in agent_stats.items():
        print(f"\n{role.capitalize()}:")
        print(f"  Tasks: {stats['task_count']}, Avg Fitness: {stats['avg_fitness']:.2f}")
        print(f"  Reasoning patterns: {stats['reasoning_patterns']} "
              f"(personal: {stats['personal_patterns']}, inherited: {stats['inherited_patterns']})")

    # Shared RAG
    print("\nShared RAG:")
    rag_stats = pipeline.get_shared_rag_stats()
    print(f"  Total knowledge: {rag_stats['total_knowledge']}")
    print(f"  By source: {rag_stats['by_source']}")

    # Learning improvement
    if len(results) >= 2:
        first_avg = results[0]['avg_score']
        last_avg = results[-1]['avg_score']
        improvement = last_avg - first_avg

        print(f"\nLearning Improvement:")
        print(f"  First generation: {first_avg:.2f}")
        print(f"  Last generation: {last_avg:.2f}")
        print(f"  Change: {improvement:+.2f} points")

        if improvement > 0:
            print(f"  ✅ Agents improved!")
        elif improvement < 0:
            print(f"  ⚠️  Scores decreased (may indicate exploration)")
        else:
            print(f"  → Scores stable")

    print("\n" + "="*70)
    print("  Experiment Complete")
    print("="*70 + "\n")

    print("Data stored in:")
    print("  - ./data/reasoning/ (reasoning patterns)")
    print("  - ./data/shared_rag/ (domain knowledge)")


if __name__ == "__main__":
    asyncio.run(main())
