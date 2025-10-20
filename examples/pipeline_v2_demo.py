"""
Pipeline V2 Multi-Generation Demo

Demonstrates the complete 8-step learning cycle over multiple generations.
Shows reasoning pattern accumulation and fitness improvement.

Run: python examples/pipeline_v2_demo.py
"""

import os
import sys
from pathlib import Path
import tempfile
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.pipeline_v2 import PipelineV2
from dotenv import load_dotenv

load_dotenv()


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


async def main():
    """Run multi-generation pipeline demo."""

    print_section("PIPELINE V2: MULTI-GENERATION DEMO")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set. Please set it in .env file.")
        return

    # Setup temporary directories
    temp_base = tempfile.mkdtemp(prefix="pipeline_v2_demo_")
    reasoning_dir = os.path.join(temp_base, "reasoning")
    rag_dir = os.path.join(temp_base, "shared_rag")
    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)

    print(f"Storage directories:")
    print(f"  Reasoning: {reasoning_dir}")
    print(f"  Shared RAG: {rag_dir}")

    try:
        # Create pipeline
        print_section("INITIALIZING PIPELINE V2")

        pipeline = PipelineV2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=rag_dir,
            agent_ids={
                'intro': 'demo_1',
                'body': 'demo_1',
                'conclusion': 'demo_1'
            },
            domain="Technology"
        )

        print("✅ Pipeline initialized")
        print(f"  - Agents: intro, body, conclusion")
        print(f"  - Context distribution: 40/30/20/10 (hierarchy/high-cred/diversity/peer)")
        print(f"  - Quality threshold: ≥8.0 for shared RAG")

        # Seed shared RAG with initial knowledge
        print_section("SEEDING SHARED RAG")

        shared_rag = pipeline.agents['intro'].shared_rag
        shared_rag.store(
            content="Machine learning enables computers to learn from data without explicit programming.",
            metadata={'topic': 'ML', 'domain': 'Technology'},
            source='manual'
        )
        shared_rag.store(
            content="Neural networks are computational models inspired by biological neural networks.",
            metadata={'topic': 'ML', 'domain': 'Technology'},
            source='manual'
        )

        print(f"✅ Seeded {shared_rag.count()} knowledge items")

        # Run multiple generations
        print_section("RUNNING 5 GENERATIONS")

        topics = [
            "Understanding Neural Networks",
            "Deep Learning Fundamentals",
            "Neural Network Applications",
            "Training Deep Learning Models",
            "Optimizing Neural Network Performance"
        ]

        generation_results = []

        for gen_num, topic in enumerate(topics, start=1):
            print(f"\n--- GENERATION {gen_num}: {topic} ---")

            # Disable visualization for demo
            os.environ['ENABLE_VISUALIZATION'] = 'false'

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
                'reasoning_used': final_state['reasoning_patterns_used'],
                'knowledge_used': final_state['domain_knowledge_used']
            }
            generation_results.append(result)

            # Print summary
            print(f"  Scores: intro={final_state['scores']['intro']:.1f}, "
                  f"body={final_state['scores']['body']:.1f}, "
                  f"conclusion={final_state['scores']['conclusion']:.1f}")
            print(f"  Avg: {result['avg_score']:.1f}")
            print(f"  Reasoning used: {sum(final_state['reasoning_patterns_used'].values())} patterns")
            print(f"  Knowledge used: {sum(final_state['domain_knowledge_used'].values())} items")

        # Show progression
        print_section("GENERATION PROGRESSION")

        print("Generation | Avg Score | Reasoning Used | Knowledge Used")
        print("-" * 60)
        for result in generation_results:
            reasoning_count = sum(result['reasoning_used'].values())
            knowledge_count = sum(result['knowledge_used'].values())
            print(f"    {result['generation']}      |   {result['avg_score']:.2f}   |       {reasoning_count}        |       {knowledge_count}")

        # Show agent statistics
        print_section("FINAL AGENT STATISTICS")

        agent_stats = pipeline.get_agent_stats()

        for role, stats in agent_stats.items():
            print(f"\n{stats['agent_id']}:")
            print(f"  Role: {stats['role']}")
            print(f"  Tasks completed: {stats['task_count']}")
            print(f"  Avg fitness: {stats['avg_fitness']:.2f}")
            print(f"  Reasoning patterns: {stats['reasoning_patterns']}")
            print(f"    - Personal: {stats['personal_patterns']}")
            print(f"    - Inherited: {stats['inherited_patterns']}")

        # Show shared RAG statistics
        print_section("SHARED RAG STATISTICS")

        rag_stats = pipeline.get_shared_rag_stats()
        print(f"  Total knowledge: {rag_stats['total_knowledge']}")
        print(f"  By source: {rag_stats['by_source']}")

        # Show context flow statistics
        print_section("CONTEXT FLOW STATISTICS")

        context_stats = pipeline.get_context_flow_stats()
        print(f"  Diversity score: {context_stats.get('diversity_score', 0):.2f}")
        print(f"  Unique sources: {context_stats.get('unique_sources', 0)}")
        print(f"  Total sources: {context_stats.get('total_sources', 0)}")

        # Calculate improvement
        print_section("LEARNING IMPROVEMENT")

        if len(generation_results) >= 2:
            first_avg = generation_results[0]['avg_score']
            last_avg = generation_results[-1]['avg_score']
            improvement = last_avg - first_avg

            print(f"  First generation avg: {first_avg:.2f}")
            print(f"  Last generation avg: {last_avg:.2f}")
            print(f"  Improvement: {improvement:+.2f} points")

            if improvement > 0:
                print(f"  ✅ Agents improved over {len(generation_results)} generations")
            else:
                print(f"  ⚠️  No improvement (may need more generations)")

        print_section("DEMO COMPLETE")

        print(f"""
Key Observations:
1. Each agent accumulated {agent_stats['intro']['reasoning_patterns']} reasoning patterns
2. Shared RAG grew from 2 → {rag_stats['total_knowledge']} items
3. Later generations retrieved reasoning patterns from earlier ones
4. Context distribution used 40/30/20/10 weighting
5. Only outputs with score ≥8.0 went to shared RAG

8-Step Cycle Demonstrated:
✅ STEP 1: START - Agents initialized (with inheritance support)
✅ STEP 2: PLAN - Retrieved reasoning patterns each generation
✅ STEP 3: RETRIEVE - Queried shared RAG for knowledge
✅ STEP 4: CONTEXT - Assembled reasoning traces (40/30/20/10)
✅ STEP 5: GENERATE - Created content with <think>/<final> tags
✅ STEP 6: EVALUATE - Scored output quality
✅ STEP 7: STORE - Stored reasoning + high-quality outputs
⏳ STEP 8: EVOLVE - (M2 - selection, reproduction, compaction)

Files preserved at: {temp_base}
Inspect reasoning patterns and shared RAG data
""")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
