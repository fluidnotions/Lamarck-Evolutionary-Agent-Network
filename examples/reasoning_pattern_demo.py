"""
Demo: 8-Step Reasoning Pattern Cycle

This example demonstrates the new reasoning pattern architecture:
- Layer 1: Fixed prompts with <think>/<final> tags
- Layer 2: Shared RAG for domain knowledge
- Layer 3: ReasoningMemory for cognitive strategies

Run: python examples/reasoning_pattern_demo.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.shared_rag import SharedRAG
from lean.base_agent_v2 import IntroAgentV2
from dotenv import load_dotenv

load_dotenv()


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def demo_reasoning_pattern_cycle():
    """Demonstrate the complete 8-step reasoning pattern cycle."""

    print_section("8-STEP REASONING PATTERN CYCLE DEMO")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set. Please set it in .env file.")
        return

    # Setup directories
    reasoning_dir = "./demo_data/reasoning"
    rag_dir = "./demo_data/shared_rag"
    os.makedirs(reasoning_dir, exist_ok=True)
    os.makedirs(rag_dir, exist_ok=True)

    # ===== STEP 1: START WITH INHERITANCE =====
    print_section("STEP 1: Start with Inheritance")

    # Simulate parent reasoning patterns
    parent_patterns = [
        {
            'reasoning': """For introductions about technical topics, I should:
1. Start with a relatable analogy or historical context
2. Bridge to the modern application
3. Pose a question that creates intrigue

This approach scored 8.2 in the past.""",
            'score': 8.2,
            'situation': 'intro for ML topic',
            'tactic': 'analogy → modern → question',
            'metadata': {'generation': 0, 'parent_id': 'parent_agent_1'}
        },
        {
            'reasoning': """When writing for technical audiences:
- Start with concrete examples before abstractions
- Use statistics to build credibility early
- End with implications or future direction

Scored 7.8 previously.""",
            'score': 7.8,
            'situation': 'intro for technical topic',
            'tactic': 'concrete → credibility → implications',
            'metadata': {'generation': 0, 'parent_id': 'parent_agent_2'}
        }
    ]

    print(f"Agent inheriting {len(parent_patterns)} reasoning patterns from parents:")
    for i, pattern in enumerate(parent_patterns, 1):
        print(f"\n  Pattern {i}:")
        print(f"    Tactic: {pattern['tactic']}")
        print(f"    Score: {pattern['score']}")
        print(f"    Parent: {pattern['metadata']['parent_id']}")

    # Create agent with inherited patterns
    collection_name = generate_reasoning_collection_name("intro", "demo_agent_1")
    reasoning_memory = ReasoningMemory(
        collection_name=collection_name,
        persist_directory=reasoning_dir,
        inherited_reasoning=parent_patterns
    )

    shared_rag = SharedRAG(persist_directory=rag_dir)

    agent = IntroAgentV2(
        role="intro",
        agent_id="demo_agent_1",
        reasoning_memory=reasoning_memory,
        shared_rag=shared_rag,
        parent_ids=["parent_agent_1", "parent_agent_2"]
    )

    print(f"\n✅ Agent initialized with {reasoning_memory.count()} inherited reasoning patterns")

    # ===== STEP 2: PLAN APPROACH =====
    print_section("STEP 2: Plan Approach")

    topic = "Understanding Transformer Architecture in Modern AI"
    print(f"Topic: {topic}")
    print("\nQuerying reasoning patterns: 'How did I/my parents solve similar problems?'")

    reasoning_patterns = reasoning_memory.retrieve_similar_reasoning(
        query=topic,
        k=5,
        score_weight=0.5
    )

    print(f"\n✅ Retrieved {len(reasoning_patterns)} relevant reasoning patterns:")
    for i, pattern in enumerate(reasoning_patterns, 1):
        print(f"\n  Pattern {i}:")
        print(f"    Tactic: {pattern['tactic']}")
        print(f"    Score: {pattern['score']:.1f}")
        print(f"    Similarity: {pattern['similarity']:.2f}")
        print(f"    Weighted Relevance: {pattern['weighted_relevance']:.2f}")

    # ===== STEP 3: RETRIEVE KNOWLEDGE =====
    print_section("STEP 3: Retrieve Knowledge")

    # Add some domain knowledge to shared RAG
    print("Adding domain knowledge to shared RAG...")
    shared_rag.store(
        content="Transformers are neural network architectures that use self-attention mechanisms to process sequential data. Introduced in 2017 by Vaswani et al., they revolutionized NLP.",
        metadata={'topic': 'transformers', 'domain': 'ML', 'score': 9.0},
        source='manual'
    )
    shared_rag.store(
        content="The key innovation of transformers is the attention mechanism, which allows the model to weigh the importance of different parts of the input when making predictions.",
        metadata={'topic': 'attention mechanism', 'domain': 'ML', 'score': 8.5},
        source='manual'
    )

    print("\nQuerying shared RAG: 'What do I need to know?'")
    domain_knowledge = shared_rag.retrieve(
        query=topic,
        k=3
    )

    print(f"\n✅ Retrieved {len(domain_knowledge)} knowledge items:")
    for i, item in enumerate(domain_knowledge, 1):
        print(f"\n  Knowledge {i}:")
        print(f"    Source: {item['source']}")
        print(f"    Similarity: {item['similarity']:.2f}")
        print(f"    Content: {item['content'][:100]}...")

    # ===== STEP 4: RECEIVE CONTEXT =====
    print_section("STEP 4: Receive Context")

    # Simulate reasoning traces from other agents
    reasoning_context = """[HIGH_CREDIBILITY (30%)]:
Agent: body_agent_3 (fitness: 8.5)
Reasoning: "For transformer explanations, I used a layered approach: first the problem it solves, then mechanism, then impact"

[DIVERSITY (20%)]:
Agent: conclusion_agent_7 (fitness: 6.2)
Reasoning: "I tried starting with the result first, then working backwards to explain how we got there"

[PEER (10%)]:
Agent: intro_agent_2 (fitness: 7.8)
Reasoning: "Used a timeline approach: past challenges → transformer innovation → current state"
"""

    print("Reasoning traces from other agents (40/30/20/10 distribution):")
    print(reasoning_context)

    print("\n✅ Context assembled from multiple sources")

    # ===== STEP 5: GENERATE =====
    print_section("STEP 5: Generate with Reasoning")

    print("Generating with <think> (reasoning) + <final> (output) tags...")
    print("\n[Calling LLM...]")

    result = agent.generate_with_reasoning(
        topic=topic,
        reasoning_patterns=reasoning_patterns,
        domain_knowledge=domain_knowledge,
        reasoning_context=reasoning_context
    )

    print("\n✅ Generated response with reasoning:")
    print("\n--- THINKING (from <think> tags) ---")
    print(result['thinking'])
    print("\n--- OUTPUT (from <final> tags) ---")
    print(result['output'])

    # ===== STEP 6: EVALUATE =====
    print_section("STEP 6: Evaluate")

    # Simulate evaluation
    score = 8.7
    print(f"Output evaluated by LLMEvaluator")
    print(f"Score: {score}/10.0")
    print("Criteria: engagement, clarity, hook")

    print(f"\n✅ Quality score: {score}")

    # ===== STEP 7: STORE REASONING PATTERN =====
    print_section("STEP 7: Store Reasoning Pattern")

    print("Storing reasoning pattern (from <think> section)...")
    agent.prepare_reasoning_storage(
        thinking=result['thinking'],
        output=result['output'],
        topic=topic,
        domain="ML",
        generation=1,
        context_sources=['hierarchy', 'high_credibility', 'diversity']
    )

    agent.record_fitness(score=score, domain="ML")
    agent.store_reasoning_and_output(score=score)

    print(f"\n✅ Reasoning pattern stored in: {collection_name}")
    print(f"✅ Output stored in shared RAG: {score >= 8.0}")

    # ===== STEP 8: EVOLVE =====
    print_section("STEP 8: Evolve")

    print("Evolution happens every 10 generations (handled by M2):")
    print("  - Select best reasoners as parents")
    print("  - Compact reasoning patterns")
    print("  - Reproduce with inherited patterns")
    print("  - Manage population based on fitness")

    print("\n✅ Agent ready for next generation")

    # ===== FINAL STATS =====
    print_section("FINAL STATISTICS")

    reasoning_stats = reasoning_memory.get_stats()
    rag_stats = shared_rag.get_stats()
    agent_stats = agent.get_stats()

    print("Reasoning Memory:")
    print(f"  Total patterns: {reasoning_stats['total_patterns']}")
    print(f"  Inherited: {reasoning_stats['inherited_patterns']}")
    print(f"  Personal: {reasoning_stats['personal_patterns']}")
    print(f"  Avg score: {reasoning_stats['avg_score']:.2f}")

    print("\nShared RAG:")
    print(f"  Total knowledge: {rag_stats['total_knowledge']}")
    print(f"  By source: {rag_stats['by_source']}")

    print("\nAgent Performance:")
    print(f"  Agent ID: {agent_stats['agent_id']}")
    print(f"  Tasks completed: {agent_stats['task_count']}")
    print(f"  Avg fitness: {agent_stats['avg_fitness']:.2f}")

    print_section("DEMO COMPLETE")

    print("""
Key Takeaways:
1. Reasoning patterns (HOW to think) are inherited, not content
2. Domain knowledge (WHAT to know) is shared via RAG
3. <think> tags externalize reasoning for storage
4. High-quality outputs (≥8.0) contribute to shared knowledge
5. The 8-step cycle ensures systematic cognitive improvement

Next: Run multiple generations to see reasoning patterns evolve!
""")


if __name__ == "__main__":
    demo_reasoning_pattern_cycle()
