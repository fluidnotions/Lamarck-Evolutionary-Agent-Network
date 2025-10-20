"""
V2 Workflow Integration Example

Demonstrates using create_agents_v2() factory in a workflow.
This shows the migration path from old agents to V2 agents.

Run: python examples/v2_workflow_integration.py
"""

import os
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.base_agent_v2 import create_agents_v2
from dotenv import load_dotenv

load_dotenv()


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    """Run V2 workflow integration demo."""

    print_section("V2 WORKFLOW INTEGRATION DEMO")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set. Please set it in .env file.")
        return

    # Setup temporary directories
    temp_base = tempfile.mkdtemp(prefix="v2_workflow_")
    reasoning_dir = os.path.join(temp_base, "reasoning")
    rag_dir = os.path.join(temp_base, "shared_rag")
    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)

    print(f"Storage directories:")
    print(f"  Reasoning: {reasoning_dir}")
    print(f"  Shared RAG: {rag_dir}")

    try:
        # STEP 1: Create agents using factory
        print_section("STEP 1: Create Agents with Factory")

        agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=rag_dir,
            agent_ids={
                'intro': 'workflow_agent_1',
                'body': 'workflow_agent_1',
                'conclusion': 'workflow_agent_1'
            }
        )

        print("✅ Created agents:")
        for role, agent in agents.items():
            print(f"  - {role}: {agent.agent_id}")

        # STEP 2: Seed shared RAG with domain knowledge
        print_section("STEP 2: Seed Shared Knowledge Base")

        shared_rag = agents['intro'].shared_rag  # All agents share same instance

        shared_rag.store(
            content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={'topic': 'python', 'domain': 'programming'},
            source='manual'
        )
        shared_rag.store(
            content="Python's extensive standard library and third-party packages make it versatile for various applications.",
            metadata={'topic': 'python', 'domain': 'programming'},
            source='manual'
        )

        print(f"✅ Seeded {shared_rag.count()} knowledge items")

        # STEP 3: Run a simple generation cycle
        print_section("STEP 3: Run Generation Cycle")

        topic = "Getting Started with Python Programming"
        domain = "Programming"
        generation = 1

        # Intro agent
        print("--- INTRO AGENT ---")
        intro_agent = agents['intro']

        # Retrieve reasoning patterns
        reasoning_patterns = intro_agent.reasoning_memory.retrieve_similar_reasoning(
            query=topic, k=3
        )
        print(f"Retrieved {len(reasoning_patterns)} reasoning patterns")

        # Retrieve domain knowledge
        domain_knowledge = shared_rag.retrieve(query=topic, k=2)
        print(f"Retrieved {len(domain_knowledge)} knowledge items")

        # Generate
        result = intro_agent.generate_with_reasoning(
            topic=topic,
            reasoning_patterns=reasoning_patterns,
            domain_knowledge=domain_knowledge,
            reasoning_context=""
        )

        print(f"\n--- THINKING ---")
        print(result['thinking'][:150] + "...")
        print(f"\n--- OUTPUT ---")
        print(result['output'])

        # Store
        score = 8.5  # Simulated evaluation
        intro_agent.prepare_reasoning_storage(
            thinking=result['thinking'],
            output=result['output'],
            topic=topic,
            domain=domain,
            generation=generation,
            context_sources=['hierarchy']
        )
        intro_agent.record_fitness(score=score, domain=domain)
        intro_agent.store_reasoning_and_output(score=score)

        print(f"\n✅ Score: {score}/10.0")
        print(f"✅ Reasoning pattern stored")
        if score >= 8.0:
            print(f"✅ Output stored in shared RAG")

        # STEP 4: Verify storage
        print_section("STEP 4: Verify Storage")

        print("Reasoning Memory (per-agent):")
        for role, agent in agents.items():
            count = agent.reasoning_memory.count()
            print(f"  {role}: {count} patterns")

        print(f"\nShared RAG (global):")
        print(f"  Total knowledge: {shared_rag.count()}")
        stats = shared_rag.get_stats()
        print(f"  By source: {stats['by_source']}")

        # STEP 5: Show agent statistics
        print_section("STEP 5: Agent Statistics")

        for role, agent in agents.items():
            stats = agent.get_stats()
            print(f"\n{stats['agent_id']}:")
            print(f"  Tasks completed: {stats['task_count']}")
            print(f"  Avg fitness: {stats['avg_fitness']:.2f}")
            print(f"  Reasoning patterns: {stats['reasoning_patterns']}")

        print_section("INTEGRATION COMPLETE")

        print("""
Key Features Demonstrated:
1. ✅ Factory function creates agents with proper setup
2. ✅ Shared RAG instance across all agents
3. ✅ Per-agent reasoning memory
4. ✅ Generation with <think>/<final> tags
5. ✅ Storage separation (reasoning vs. knowledge)
6. ✅ Quality threshold for shared RAG

Migration Path:
- OLD: create_agents(persist_directory="./data/memories")
- NEW: create_agents_v2(reasoning_dir="./data/reasoning", shared_rag_dir="./data/shared_rag")

Next Steps:
- Integrate V2 agents into existing workflow
- Update pipeline to use 8-step cycle
- Run multi-generation tests

Files preserved at: {temp_base}
""".format(temp_base=temp_base))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
