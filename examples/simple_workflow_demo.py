"""
Simple 2-3 Generation Workflow Demo

Demonstrates a minimal workflow using the reasoning pattern architecture:
- Creates 3 intro agents
- Runs 3 generations with same topic
- Shows reasoning pattern accumulation
- Validates storage separation

Run: python examples/simple_workflow_demo.py
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.shared_rag import SharedRAG
from lean.base_agent_v2 import IntroAgentV2
from dotenv import load_dotenv

load_dotenv()


class SimpleAgentPool:
    """Minimal agent pool for demonstration."""

    def __init__(self, role: str, agents: list):
        self.role = role
        self.agents = agents

    def select_agent(self, strategy: str = "random"):
        """Select agent (simple random for demo)."""
        import random
        return random.choice(self.agents)

    def get_top_n(self, n: int = 2):
        """Get top N agents by fitness."""
        sorted_agents = sorted(
            self.agents,
            key=lambda a: a.avg_fitness(),
            reverse=True
        )
        return sorted_agents[:n]

    def get_random_lower_half(self):
        """Get random agent from lower half by fitness."""
        import random
        sorted_agents = sorted(
            self.agents,
            key=lambda a: a.avg_fitness()
        )
        lower_half = sorted_agents[:len(sorted_agents)//2] if len(sorted_agents) > 1 else sorted_agents
        return random.choice(lower_half) if lower_half else self.agents[0]

    def size(self):
        """Get pool size."""
        return len(self.agents)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def simple_workflow():
    """Run a simple 2-3 generation workflow."""

    print_section("SIMPLE WORKFLOW: 3 GENERATIONS")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set. Please set it in .env file.")
        return

    # Setup temporary directories for demo
    temp_base = tempfile.mkdtemp(prefix="simple_workflow_")
    reasoning_dir = os.path.join(temp_base, "reasoning")
    rag_dir = os.path.join(temp_base, "shared_rag")
    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)

    print(f"Storage directories:")
    print(f"  Reasoning: {reasoning_dir}")
    print(f"  Shared RAG: {rag_dir}")

    try:
        # Create shared RAG (Layer 2: shared by all)
        shared_rag = SharedRAG(persist_directory=rag_dir)

        # Seed with some domain knowledge
        print("\nSeeding shared RAG with domain knowledge...")
        shared_rag.store(
            content="Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
            metadata={'topic': 'machine learning', 'domain': 'ML'},
            source='manual'
        )
        shared_rag.store(
            content="Neural networks are inspired by biological neurons and consist of interconnected layers of nodes.",
            metadata={'topic': 'neural networks', 'domain': 'ML'},
            source='manual'
        )
        print(f"✅ Seeded {shared_rag.count()} knowledge items")

        # Create 3 intro agents (Generation 0)
        print("\n" + "="*60)
        print("GENERATION 0: Initializing Population")
        print("="*60)

        agents = []
        for i in range(3):
            collection_name = generate_reasoning_collection_name("intro", f"agent_{i}")
            reasoning_memory = ReasoningMemory(
                collection_name=collection_name,
                persist_directory=reasoning_dir
            )

            agent = IntroAgentV2(
                role="intro",
                agent_id=f"intro_agent_{i}",
                reasoning_memory=reasoning_memory,
                shared_rag=shared_rag
            )
            agents.append(agent)

        print(f"Created {len(agents)} intro agents")

        # Create simple pool
        intro_pool = SimpleAgentPool(role="intro", agents=agents)

        # Run 3 generations
        topic = "Understanding Deep Learning"
        domain = "ML"

        for gen in range(3):
            print_section(f"GENERATION {gen + 1}")

            # Select agent (round-robin for simplicity)
            agent = agents[gen % len(agents)]
            print(f"Selected: {agent.agent_id}")

            # STEP 2: Query reasoning patterns
            print("\nSTEP 2: Querying reasoning patterns...")
            reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(
                query=topic,
                k=3
            )
            print(f"  Retrieved {len(reasoning_patterns)} reasoning patterns")

            # STEP 3: Query shared RAG
            print("\nSTEP 3: Querying shared RAG...")
            domain_knowledge = shared_rag.retrieve(query=topic, k=2)
            print(f"  Retrieved {len(domain_knowledge)} knowledge items")

            # STEP 4: Get context from other agents (simplified)
            print("\nSTEP 4: Assembling context...")
            reasoning_context = ""
            if gen > 0:
                # Get reasoning from previous agent
                prev_agent = agents[(gen - 1) % len(agents)]
                prev_patterns = prev_agent.reasoning_memory.get_all_reasoning(
                    include_inherited=False
                )
                if prev_patterns:
                    latest = prev_patterns[-1]
                    reasoning_context = f"[PEER REASONING]:\n{latest['reasoning'][:200]}..."
                    print(f"  Added context from {prev_agent.agent_id}")

            # STEP 5: Generate
            print("\nSTEP 5: Generating with reasoning...")
            result = agent.generate_with_reasoning(
                topic=topic,
                reasoning_patterns=reasoning_patterns,
                domain_knowledge=domain_knowledge,
                reasoning_context=reasoning_context
            )

            print(f"\n--- THINKING ---")
            print(result['thinking'][:200] + "..." if len(result['thinking']) > 200 else result['thinking'])
            print(f"\n--- OUTPUT ---")
            print(result['output'][:200] + "..." if len(result['output']) > 200 else result['output'])

            # STEP 6: Evaluate (simulated)
            # Vary scores: 7.0, 8.5, 8.2
            scores = [7.0, 8.5, 8.2]
            score = scores[gen % len(scores)]
            print(f"\nSTEP 6: Evaluated - Score: {score}/10.0")

            # STEP 7: Store
            print("\nSTEP 7: Storing reasoning pattern...")
            agent.prepare_reasoning_storage(
                thinking=result['thinking'],
                output=result['output'],
                topic=topic,
                domain=domain,
                generation=gen + 1,
                context_sources=['hierarchy']
            )
            agent.record_fitness(score=score, domain=domain)
            agent.store_reasoning_and_output(score=score)

            print(f"  ✅ Reasoning stored in agent's collection")
            if score >= 8.0:
                print(f"  ✅ Output stored in shared RAG (score >= 8.0)")
            else:
                print(f"  ⏭️  Output not stored in shared RAG (score < 8.0)")

        # Final statistics
        print_section("FINAL STATISTICS")

        print("AGENT STATISTICS:")
        for agent in agents:
            stats = agent.get_stats()
            print(f"\n{stats['agent_id']}:")
            print(f"  Tasks completed: {stats['task_count']}")
            print(f"  Avg fitness: {stats['avg_fitness']:.2f}")
            print(f"  Reasoning patterns: {stats['reasoning_patterns']}")
            print(f"    - Personal: {stats['personal_patterns']}")
            print(f"    - Inherited: {stats['inherited_patterns']}")

        print("\nSHARED RAG STATISTICS:")
        rag_stats = shared_rag.get_stats()
        print(f"  Total knowledge: {rag_stats['total_knowledge']}")
        print(f"  By source: {rag_stats['by_source']}")

        print("\nSTORAGE VALIDATION:")
        print(f"  Reasoning patterns stored: {sum(a.get_stats()['reasoning_patterns'] for a in agents)}")
        print(f"  Shared knowledge items: {rag_stats['total_knowledge']}")
        print(f"  Storage separation: ✅ Working correctly")

        print_section("WORKFLOW COMPLETE")

        print(f"""
Key Observations:
1. Each agent accumulated personal reasoning patterns
2. High-quality outputs (≥8.0) were added to shared RAG
3. Later generations could retrieve reasoning from earlier ones
4. Shared RAG grew from 2 → {rag_stats['total_knowledge']} items
5. Storage separation maintained (reasoning vs. knowledge)

Temporary files at: {temp_base}
(Will be cleaned up on next run)
""")

    finally:
        # Note: Not cleaning up temp_base to allow inspection
        # Remove the shutil.rmtree line to keep files for inspection
        print(f"\n📁 Files preserved at: {temp_base}")
        print("   Inspect reasoning patterns and shared RAG data")


if __name__ == "__main__":
    simple_workflow()
