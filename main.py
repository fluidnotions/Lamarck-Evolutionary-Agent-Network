"""
Main entry point for LEAN with hierarchical coordinator architecture.

This version implements the 3-layer architecture:
- Layer 1: Coordinator (research, orchestration, critique)
- Layer 2: Content Agents (intro, body, conclusion)
- Layer 3: Specialist Agents (researcher, fact-checker, stylist)

Features:
- Tavily research integration
- Coordinator-driven workflow
- Specialist agent support
- Revision loop with coordinator critique
- Pool-based evolution
- Reasoning pattern inheritance
- YAML-based configuration

Run: python main_v3.py [--config CONFIG_NAME]
"""

import asyncio
import os
import argparse
from dotenv import load_dotenv
from loguru import logger

from src.lean.pipeline import Pipeline
from src.lean.config_loader import load_config
from src.lean.logger import setup_logger

load_dotenv()

# Initialize logging (file-based, no console spam)
setup_logger(log_dir="./logs")


async def main(config_name: str = "default"):
    """Run pipeline with hierarchical architecture.

    Args:
        config_name: Name of the experiment config file (without .yml)
    """

    print("\n" + "="*70)
    print("  LEAN: Lamarck Evolutionary Agent Network")
    print("  Hierarchical Coordinator Architecture")
    print("="*70 + "\n")

    logger.info("Starting LEAN experiment with hierarchical architecture")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not set")
        print("Please set your API key in the .env file")
        return

    # Load configuration from YAML
    print(f"Loading experiment configuration: {config_name}")
    try:
        exp_config, agent_prompts = load_config(config_name)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Available configs should be in ./config/experiments/")
        return
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return

    print(f"✅ Loaded: {exp_config.name}")
    print(f"   {exp_config.description}")
    print()

    # Initialize pipeline with hierarchical architecture
    print("Initializing hierarchical pipeline...")

    # Get pipeline-specific settings from configuration
    pipeline_config = exp_config.pipeline_config or {}
    enable_research = exp_config.research_config.get('enabled', True) and bool(os.getenv('TAVILY_API_KEY'))
    enable_specialists = bool(pipeline_config.get('enable_specialists', True))
    enable_revision = bool(pipeline_config.get('enable_revision', True))
    max_revisions = int(pipeline_config.get('max_revisions', 2))

    # Convert agent_prompts from Dict[str, AgentPromptConfig] to Dict[str, str]
    agent_prompts_dict = {
        role: config.system_prompt
        for role, config in agent_prompts.items()
    }

    pipeline = Pipeline(
        reasoning_dir=exp_config.reasoning_dir,
        shared_rag_dir=exp_config.shared_rag_dir,
        domain=exp_config.domain,
        population_size=exp_config.population_size,
        evolution_frequency=exp_config.evolution_frequency,
        enable_research=enable_research,
        enable_specialists=enable_specialists,
        enable_revision=enable_revision,
        max_revisions=max_revisions,
        agent_prompts=agent_prompts_dict,
        experiment_config=exp_config
    )

    print("✅ Pipeline V3 initialized with Hierarchical Architecture")
    print(f"  - Population: {exp_config.population_size} agents per role")
    print(f"  - Evolution frequency: every {exp_config.evolution_frequency} generations")
    print(f"  - Total generations: {exp_config.total_generations}")
    print()
    print("  Architecture:")
    print("    Layer 1: Coordinator (research, orchestration, critique)")
    print("    Layer 2: Content Agents (intro, body, conclusion)")
    print("    Layer 3: Specialist Agents (researcher, fact-checker, stylist)")
    print()

    # Show V3 features status
    print("  V3 Features:")
    if enable_research:
        tavily_status = "✅" if os.getenv('TAVILY_API_KEY') else "⚠️ (no API key)"
        print(f"    ✅ Tavily Research: {tavily_status}")
        print(f"       Max results: {exp_config.research_config.get('max_results', 5)}")
        print(f"       Search depth: {exp_config.research_config.get('search_depth', 'advanced')}")
    else:
        print("    ❌ Tavily Research: disabled")

    print(f"    {'✅' if enable_specialists else '❌'} Specialist Agents: {'enabled' if enable_specialists else 'disabled'}")
    print(f"    {'✅' if enable_revision else '❌'} Revision Loop: {'enabled' if enable_revision else 'disabled'}")
    if enable_revision:
        print(f"       Max revisions: {max_revisions}")
    print()

    # Get topics from configuration
    topics = exp_config.get_all_topics()
    TOTAL_GENERATIONS = exp_config.total_generations

    # Limit to requested generations
    topics = topics[:TOTAL_GENERATIONS]

    print(f"Running {len(topics)} generations with evolutionary learning...")
    print(f"Evolution events at generations: {', '.join(str(i) for i in range(exp_config.evolution_frequency, TOTAL_GENERATIONS + 1, exp_config.evolution_frequency))}\n")

    # Execute generations
    for gen_num, topic in enumerate(topics, start=1):
        # topic is a string (topic title from get_all_topics())
        print(f"\n{'='*70}")
        print(f"Generation {gen_num}/{len(topics)}")
        print(f"Topic: {topic}")

        # Show topic metadata
        metadata = exp_config.get_topic_metadata(topic)
        if metadata:
            if metadata.get('keywords'):
                print(f"Keywords: {', '.join(metadata['keywords'])}")
            if metadata.get('difficulty'):
                print(f"Difficulty: {metadata['difficulty']}")

        print(f"{'='*70}\n")

        # Generate with hierarchical pipeline
        try:
            final_state = await pipeline.generate(
                topic=topic,
                generation_number=gen_num
            )

            # Show results
            print(f"\n{'─'*70}")
            print("RESULTS")
            print(f"{'─'*70}\n")

            print(f"📝 INTRO:\n{final_state['intro']}\n")
            print(f"📝 BODY:\n{final_state['body']}\n")
            print(f"📝 CONCLUSION:\n{final_state['conclusion']}\n")

            # Show coordinator critique (V3 feature)
            if 'coordinator_critique' in final_state:
                critique = final_state['coordinator_critique']
                scores = critique.get('scores', {})
                print(f"{'─'*70}")
                print("COORDINATOR CRITIQUE")
                print(f"{'─'*70}")
                print(f"Coherence: {scores.get('coherence', 0):.1f}/10")
                print(f"Accuracy: {scores.get('accuracy', 0):.1f}/10")
                print(f"Depth: {scores.get('depth', 0):.1f}/10")
                print(f"Overall: {scores.get('overall', 0):.1f}/10")
                print(f"\nFeedback: {critique.get('feedback', 'N/A')}\n")

            # Show evaluation scores
            print(f"{'─'*70}")
            print("EVALUATION SCORES")
            print(f"{'─'*70}")
            for role, score in final_state['scores'].items():
                print(f"{role.capitalize()}: {score:.1f}/10")

            avg_score = sum(final_state['scores'].values()) / len(final_state['scores'])
            print(f"\nAverage: {avg_score:.1f}/10\n")

            # Show revision info if any
            if final_state.get('revision_count', 0) > 0:
                print(f"🔄 Revisions performed: {final_state['revision_count']}\n")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in generation {gen_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final statistics
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}\n")

    # Get agent pool stats
    print("Agent Pool Statistics:")
    for role, pool in pipeline.agent_pools.items():
        print(f"\n{role.capitalize()} Pool:")
        print(f"  Generation: {pool.generation}")
        print(f"  Population: {pool.size()}")
        print(f"  Avg Fitness: {pool.avg_fitness():.2f}")
        print(f"  Diversity: {pool.measure_diversity():.3f}")

    # Get shared RAG stats
    print("\nShared Knowledge Base:")
    rag_stats = pipeline.shared_rag.get_stats()
    print(f"  Total items: {rag_stats.get('total_knowledge', 0)}")
    print(f"  Sources: {', '.join(rag_stats.get('sources', []))}")

    print(f"\n{'='*70}")
    print("Data saved to:")
    print(f"  - Reasoning patterns: {exp_config.reasoning_dir}")
    print(f"  - Shared knowledge: {exp_config.shared_rag_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEAN: Hierarchical Coordinator Architecture"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Name of experiment config file (without .yml extension)"
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main(config_name=args.config))
