"""
Pipeline V2 with Reasoning Pattern Architecture

Implements the 8-step learning cycle:
1. START → Agent has inherited reasoning patterns
2. PLAN → Retrieve similar reasoning patterns
3. RETRIEVE → Query shared RAG for domain knowledge
4. CONTEXT → Assemble reasoning traces from other agents (40/30/20/10)
5. GENERATE → Create content with <think>/<final> tags
6. EVALUATE → Score output quality
7. STORE → Store reasoning pattern and high-quality outputs
8. EVOLVE → (M2) Selection, compaction, reproduction

This pipeline uses BaseAgentV2 agents with ReasoningMemory + SharedRAG.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, AsyncIterator, Optional
import os
from dotenv import load_dotenv
import time

load_dotenv()

from lean.state import BlogState, create_initial_state
from lean.base_agent_v2 import create_agents_v2
from lean.context_manager import ContextManager
from lean.evaluation import ContentEvaluator
from lean.visualization import StreamVisualizer
from lean.agent_pool import AgentPool
from lean.reproduction import SexualReproduction


class PipelineV2:
    """Pipeline V2 with reasoning pattern architecture and 8-step learning cycle."""

    def __init__(
        self,
        reasoning_dir: str = "./data/reasoning",
        shared_rag_dir: str = "./data/shared_rag",
        agent_ids: Optional[Dict[str, str]] = None,
        domain: str = "General",
        population_size: int = 5,
        evolution_frequency: int = 10
    ):
        """Initialize V2 pipeline.

        Args:
            reasoning_dir: Directory for per-agent reasoning patterns
            shared_rag_dir: Directory for shared knowledge base
            agent_ids: Optional dict mapping role → agent_id
            domain: Domain category for this pipeline instance
            population_size: Number of agents per role in evolutionary pool
            evolution_frequency: Trigger evolution every N generations
        """
        self.domain = domain
        self.generation_counter = 0
        self.evolution_frequency = evolution_frequency

        # Create initial V2 agents with factory
        initial_agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=shared_rag_dir,
            agent_ids=agent_ids
        )

        # Store shared_rag reference for evolution
        self.shared_rag = initial_agents['intro'].shared_rag

        # Initialize reproduction strategy for evolution
        self.reproduction_strategy = SexualReproduction()

        # Initialize context manager for reasoning trace distribution
        self.context_manager = ContextManager(
            hierarchy_weight=0.40,
            high_credibility_weight=0.30,
            diversity_weight=0.20,
            peer_weight=0.10
        )

        # Initialize evaluator and visualizer
        self.evaluator = ContentEvaluator()
        self.visualizer = StreamVisualizer()

        # Create agent pools with M2 evolution (replaces SimpleAgentPool)
        self.agent_pools = {
            'intro': AgentPool(
                role='intro',
                initial_agents=[initial_agents['intro']],
                max_size=population_size
            ),
            'body': AgentPool(
                role='body',
                initial_agents=[initial_agents['body']],
                max_size=population_size
            ),
            'conclusion': AgentPool(
                role='conclusion',
                initial_agents=[initial_agents['conclusion']],
                max_size=population_size
            )
        }

        # Build LangGraph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow with 8-step cycle.

        Graph structure:
        START → intro → body → conclusion → evaluate → evolve → END

        Each agent node executes steps 2-5 internally.
        Evaluate and evolve nodes execute steps 6-7.

        Returns:
            Compiled LangGraph application
        """
        workflow = StateGraph(BlogState)

        # Agent nodes (each executes steps 2-5)
        workflow.add_node("intro", self._intro_node)
        workflow.add_node("body", self._body_node)
        workflow.add_node("conclusion", self._conclusion_node)

        # Evaluation node (step 6)
        workflow.add_node("evaluate", self._evaluate_node)

        # Evolution node (step 7)
        workflow.add_node("evolve", self._evolve_node)

        # Define execution flow
        workflow.set_entry_point("intro")
        workflow.add_edge("intro", "body")
        workflow.add_edge("body", "conclusion")
        workflow.add_edge("conclusion", "evaluate")
        workflow.add_edge("evaluate", "evolve")
        workflow.add_edge("evolve", END)

        # Compile with memory for checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _intro_node(self, state: BlogState) -> BlogState:
        """Execute intro agent with 8-step cycle (steps 2-5).

        Steps:
        2. PLAN → Retrieve similar reasoning patterns
        3. RETRIEVE → Query shared RAG
        4. CONTEXT → Assemble reasoning traces (hierarchy only for intro)
        5. GENERATE → Create content with reasoning

        Args:
            state: Current workflow state

        Returns:
            Updated state with intro content and reasoning
        """
        # Select agent from pool (fitness-proportionate)
        agent = self.agent_pools['intro'].select_agent(strategy="fitness_proportionate")
        topic = state['topic']

        start_time = time.time()

        # STEP 2: Retrieve reasoning patterns
        reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(
            query=topic,
            k=int(os.getenv('MAX_REASONING_RETRIEVE', '5'))
        )

        # STEP 3: Retrieve domain knowledge
        domain_knowledge = agent.shared_rag.retrieve(
            query=topic,
            k=int(os.getenv('MAX_KNOWLEDGE_RETRIEVE', '3'))
        )

        # STEP 4: Assemble context (hierarchy only for intro - it's first)
        hierarchy_context = f"Write an engaging introduction for: {topic}"
        reasoning_context = self.context_manager.assemble_context(
            current_agent=agent,
            hierarchy_context=hierarchy_context,
            all_pools=self.agent_pools,
            workflow_state=state
        )

        # STEP 5: Generate with reasoning
        result = agent.generate_with_reasoning(
            topic=topic,
            reasoning_patterns=reasoning_patterns,
            domain_knowledge=domain_knowledge,
            reasoning_context=reasoning_context['content']
        )

        # Store in state
        state['intro'] = result['output']
        state['intro_reasoning'] = result['thinking']
        state['reasoning_patterns_used']['intro'] = len(reasoning_patterns)
        state['domain_knowledge_used']['intro'] = len(domain_knowledge)

        # Prepare for storage (will be stored in evolve node after evaluation)
        agent.prepare_reasoning_storage(
            thinking=result['thinking'],
            output=result['output'],
            topic=topic,
            domain=self.domain,
            generation=state['generation_number'],
            context_sources=reasoning_context['sources']
        )

        # Track timing
        end_time = time.time()
        state['agent_timings']['intro'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        # Log
        state['stream_logs'].append(
            f"[intro] Generated ({len(reasoning_patterns)} patterns, "
            f"{len(domain_knowledge)} knowledge items)"
        )

        return state

    async def _body_node(self, state: BlogState) -> BlogState:
        """Execute body agent with 8-step cycle (steps 2-5).

        Steps:
        2. PLAN → Retrieve similar reasoning patterns
        3. RETRIEVE → Query shared RAG
        4. CONTEXT → Assemble reasoning traces (includes intro)
        5. GENERATE → Create content with reasoning

        Args:
            state: Current workflow state

        Returns:
            Updated state with body content and reasoning
        """
        # Select agent from pool (fitness-proportionate)
        agent = self.agent_pools['body'].select_agent(strategy="fitness_proportionate")
        topic = state['topic']

        start_time = time.time()

        # STEP 2: Retrieve reasoning patterns
        reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(
            query=topic,
            k=int(os.getenv('MAX_REASONING_RETRIEVE', '5'))
        )

        # STEP 3: Retrieve domain knowledge
        domain_knowledge = agent.shared_rag.retrieve(
            query=topic,
            k=int(os.getenv('MAX_KNOWLEDGE_RETRIEVE', '3'))
        )

        # STEP 4: Assemble context (includes intro reasoning)
        hierarchy_context = f"Write detailed body content for: {topic}. Intro: {state['intro'][:100]}..."
        reasoning_context = self.context_manager.assemble_context(
            current_agent=agent,
            hierarchy_context=hierarchy_context,
            all_pools=self.agent_pools,
            workflow_state=state
        )

        # STEP 5: Generate with reasoning
        result = agent.generate_with_reasoning(
            topic=topic,
            reasoning_patterns=reasoning_patterns,
            domain_knowledge=domain_knowledge,
            reasoning_context=reasoning_context['content']
        )

        # Store in state
        state['body'] = result['output']
        state['body_reasoning'] = result['thinking']
        state['reasoning_patterns_used']['body'] = len(reasoning_patterns)
        state['domain_knowledge_used']['body'] = len(domain_knowledge)

        # Prepare for storage
        agent.prepare_reasoning_storage(
            thinking=result['thinking'],
            output=result['output'],
            topic=topic,
            domain=self.domain,
            generation=state['generation_number'],
            context_sources=reasoning_context['sources']
        )

        # Track timing
        end_time = time.time()
        state['agent_timings']['body'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        # Log
        state['stream_logs'].append(
            f"[body] Generated ({len(reasoning_patterns)} patterns, "
            f"{len(domain_knowledge)} knowledge items)"
        )

        return state

    async def _conclusion_node(self, state: BlogState) -> BlogState:
        """Execute conclusion agent with 8-step cycle (steps 2-5).

        Steps:
        2. PLAN → Retrieve similar patterns
        3. RETRIEVE → Query shared RAG
        4. CONTEXT → Assemble reasoning traces (includes intro + body)
        5. GENERATE → Create content with reasoning

        Args:
            state: Current workflow state

        Returns:
            Updated state with conclusion content and reasoning
        """
        # Select agent from pool (fitness-proportionate)
        agent = self.agent_pools['conclusion'].select_agent(strategy="fitness_proportionate")
        topic = state['topic']

        start_time = time.time()

        # STEP 2: Retrieve reasoning patterns
        reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(
            query=topic,
            k=int(os.getenv('MAX_REASONING_RETRIEVE', '5'))
        )

        # STEP 3: Retrieve domain knowledge
        domain_knowledge = agent.shared_rag.retrieve(
            query=topic,
            k=int(os.getenv('MAX_KNOWLEDGE_RETRIEVE', '3'))
        )

        # STEP 4: Assemble context (includes intro + body reasoning)
        hierarchy_context = f"Write conclusion for: {topic}. Content so far: {state['intro'][:50]}... {state['body'][:50]}..."
        reasoning_context = self.context_manager.assemble_context(
            current_agent=agent,
            hierarchy_context=hierarchy_context,
            all_pools=self.agent_pools,
            workflow_state=state
        )

        # STEP 5: Generate with reasoning
        result = agent.generate_with_reasoning(
            topic=topic,
            reasoning_patterns=reasoning_patterns,
            domain_knowledge=domain_knowledge,
            reasoning_context=reasoning_context['content']
        )

        # Store in state
        state['conclusion'] = result['output']
        state['conclusion_reasoning'] = result['thinking']
        state['reasoning_patterns_used']['conclusion'] = len(reasoning_patterns)
        state['domain_knowledge_used']['conclusion'] = len(domain_knowledge)

        # Prepare for storage
        agent.prepare_reasoning_storage(
            thinking=result['thinking'],
            output=result['output'],
            topic=topic,
            domain=self.domain,
            generation=state['generation_number'],
            context_sources=reasoning_context['sources']
        )

        # Track timing
        end_time = time.time()
        state['agent_timings']['conclusion'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        # Log
        state['stream_logs'].append(
            f"[conclusion] Generated ({len(reasoning_patterns)} patterns, "
            f"{len(domain_knowledge)} knowledge items)"
        )

        return state

    def _evaluate_node(self, state: BlogState) -> BlogState:
        """Evaluate generated content (STEP 6).

        Args:
            state: Current workflow state with generated content

        Returns:
            State with scores added
        """
        # Use existing evaluator
        state = self.evaluator(state)

        # Log
        scores_str = ", ".join([f"{k}: {v:.1f}" for k, v in state['scores'].items()])
        state['stream_logs'].append(f"[evaluate] Scores: {scores_str}")

        return state

    def _evolve_node(self, state: BlogState) -> BlogState:
        """Store reasoning patterns and evolve agents (STEPS 7-8).

        Steps:
        7. STORE → Store reasoning patterns and outputs
        8. EVOLVE → Trigger population evolution (every N generations)

        Evolution cycle (when triggered):
        - Selection: Choose best agents as parents
        - Compaction: Forget unsuccessful patterns
        - Reproduction: Create offspring with inherited patterns
        - Population replacement: New generation replaces old

        Args:
            state: Current workflow state with scores

        Returns:
            Updated state
        """
        # STEP 7: Store reasoning and outputs for ALL agents in ALL pools
        for role, pool in self.agent_pools.items():
            score = state['scores'].get(role, 0.0)

            # Store for all agents in pool
            for agent in pool.agents:
                # Record fitness
                agent.record_fitness(score=score, domain=self.domain)

                # Store reasoning pattern and conditionally store output
                agent.store_reasoning_and_output(score=score)

        # Get pool statistics for logging
        intro_pool_avg = self.agent_pools['intro'].avg_fitness()
        body_pool_avg = self.agent_pools['body'].avg_fitness()
        conclusion_pool_avg = self.agent_pools['conclusion'].avg_fitness()

        state['stream_logs'].append(
            f"[evolve] Pool avg fitness: "
            f"Intro: {intro_pool_avg:.1f}, "
            f"Body: {body_pool_avg:.1f}, "
            f"Conclusion: {conclusion_pool_avg:.1f}"
        )

        # STEP 8: Trigger evolution every N generations
        if self.generation_counter > 0 and self.generation_counter % self.evolution_frequency == 0:
            state['stream_logs'].append(
                f"[evolve] 🧬 EVOLUTION TRIGGERED (generation {self.generation_counter})"
            )

            # Evolve each pool
            for role, pool in self.agent_pools.items():
                pool.evolve_generation(
                    reproduction_strategy=self.reproduction_strategy,
                    shared_rag=self.shared_rag
                )

                state['stream_logs'].append(
                    f"[evolve] {role.capitalize()} pool → Generation {pool.generation} "
                    f"(avg fitness: {pool.avg_fitness():.1f})"
                )

        # Log shared RAG growth (using any agent from any pool)
        shared_rag_stats = self.shared_rag.get_stats()
        state['stream_logs'].append(
            f"[evolve] Shared RAG: {shared_rag_stats['total_knowledge']} items"
        )

        return state

    async def generate(
        self,
        topic: str,
        config: Dict | None = None,
        generation_number: int = 0
    ) -> BlogState:
        """Generate blog post with 8-step learning cycle.

        Args:
            topic: The topic to write about
            config: Optional LangGraph configuration
            generation_number: Which generation this is (for tracking)

        Returns:
            Final state with generated content, reasoning, and scores
        """
        # Initialize state
        initial_state = create_initial_state(topic)
        initial_state['generation_number'] = generation_number
        self.generation_counter += 1

        # Configure streaming
        if config is None:
            config = {
                "configurable": {"thread_id": f"blog_v2_{topic.replace(' ', '_')}"}
            }

        # Stream execution
        final_state = initial_state

        if os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true":
            # Stream with visualization
            async def state_stream() -> AsyncIterator[BlogState]:
                nonlocal final_state
                async for event in self.app.astream(
                    initial_state, config, stream_mode="values"
                ):
                    final_state = event
                    yield event

            await self.visualizer.display_stream(state_stream())
        else:
            # Run without visualization
            final_state = await self.app.ainvoke(initial_state, config)

        return final_state

    def get_agent_stats(self) -> Dict:
        """Get statistics for all agent pools.

        Returns:
            Dictionary of pool stats
        """
        return {
            role: {
                'generation': pool.generation,
                'pool_size': len(pool.agents),
                'avg_fitness': pool.avg_fitness(),
                'top_agent_fitness': pool.agents[0].avg_fitness() if pool.agents else 0.0,
                'diversity': pool.measure_diversity()
            }
            for role, pool in self.agent_pools.items()
        }

    def get_shared_rag_stats(self) -> Dict:
        """Get shared RAG statistics.

        Returns:
            Shared RAG statistics dict
        """
        return self.shared_rag.get_stats()

    def get_context_flow_stats(self) -> Dict:
        """Get context flow diversity statistics.

        Returns:
            Context flow diversity metrics
        """
        return self.context_manager.measure_diversity(recent_n=10)
