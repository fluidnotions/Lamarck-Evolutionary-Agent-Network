"""
Pipeline V3 with Hierarchical Coordinator Architecture

Implements the 3-layer architecture:
- Layer 1: Coordinator (research, orchestration, critique)
- Layer 2: Content Agents (intro, body, conclusion)
- Layer 3: Specialist Agents (researcher, fact-checker, stylist)

Flow:
1. Coordinator researches topic
2. Coordinator distributes context to content agents
3. Content agents generate (with optional specialist support)
4. Coordinator aggregates and critiques
5. Revision loop if needed
6. Evaluate and evolve
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, AsyncIterator, Optional, List
import os
from dotenv import load_dotenv
import time

load_dotenv()

from lean.state import BlogState, create_initial_state
from lean.base_agent import create_agents, IntroAgent, BodyAgent, ConclusionAgent
from lean.coordinator import CoordinatorAgent
from lean.specialists import create_specialist_agents
from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
from lean.context_manager import ContextManager
from lean.evaluation import ContentEvaluator
from lean.visualization import HierarchicalVisualizer
from lean.agent_pool import AgentPool
from lean.reproduction import SexualReproduction
from loguru import logger



class Pipeline:
    """LEAN Pipeline with hierarchical coordinator architecture."""

    def __init__(
        self,
        reasoning_dir: str = "./data/reasoning",
        shared_rag_dir: str = "./data/shared_rag",
        agent_ids: Optional[Dict[str, str]] = None,
        domain: str = "General",
        population_size: int = 5,
        evolution_frequency: int = 10,
        enable_research: bool = True,
        enable_specialists: bool = True,
        enable_revision: bool = True,
        max_revisions: int = 2,
        agent_prompts: Optional[Dict[str, str]] = None
    ):
        """Initialize LEAN pipeline with hierarchical architecture.

        Args:
            reasoning_dir: Directory for per-agent reasoning patterns
            shared_rag_dir: Directory for shared knowledge base
            agent_ids: Optional dict mapping role → agent_id
            domain: Domain category for this pipeline instance
            population_size: Number of agents per role in evolutionary pool
            evolution_frequency: Trigger evolution every N generations
            enable_research: Enable Tavily research
            enable_specialists: Enable specialist agents
            enable_revision: Enable revision loop
            max_revisions: Maximum number of revision iterations
            agent_prompts: Optional dict mapping role → system_prompt from YAML
        """
        self.domain = domain
        self.generation_counter = 0
        self.evolution_frequency = evolution_frequency
        self.enable_research = enable_research
        self.enable_specialists = enable_specialists
        self.enable_revision = enable_revision
        self.max_revisions = max_revisions
        self.agent_prompts = agent_prompts or {}

        # Create content agents
        content_agents = create_agents(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=shared_rag_dir,
            agent_ids=agent_ids,
            agent_prompts=self.agent_prompts
        )

        # Store shared_rag reference for evolution and specialists
        self.shared_rag = content_agents['intro'].shared_rag

        # Create coordinator agent
        coordinator_collection = generate_reasoning_collection_name(
            'coordinator',
            agent_ids.get('coordinator', 'coordinator_1') if agent_ids else 'coordinator_1'
        )
        coordinator_memory = ReasoningMemory(
            collection_name=coordinator_collection,
            persist_directory=reasoning_dir
        )
        self.coordinator = CoordinatorAgent(
            agent_id='coordinator_1',
            reasoning_memory=coordinator_memory,
            shared_rag=self.shared_rag,
            enable_research=enable_research,
            system_prompt=self.agent_prompts.get('coordinator')
        )

        # Create specialist agents (if enabled)
        if enable_specialists:
            self.specialists = create_specialist_agents(
                reasoning_dir=reasoning_dir,
                shared_rag=self.shared_rag,
                agent_prompts=self.agent_prompts
            )
        else:
            self.specialists = None

        # Initialize reproduction strategy
        self.reproduction_strategy = SexualReproduction()

        # Initialize context manager
        self.context_manager = ContextManager(
            hierarchy_weight=0.40,
            high_credibility_weight=0.30,
            diversity_weight=0.20,
            peer_weight=0.10
        )

        # Initialize evaluator and visualizer
        self.evaluator = ContentEvaluator()
        self.visualizer = HierarchicalVisualizer(pipeline=self)

        # Create agent pools with initial population
        # For ensemble competition, we need multiple agents per role
        intro_agents = [content_agents['intro']]
        body_agents = [content_agents['body']]
        conclusion_agents = [content_agents['conclusion']]

        # Create additional agents to fill the initial population
        for i in range(1, population_size):
            # Intro agents
            intro_collection = generate_reasoning_collection_name('intro', f'agent_{i+1}')
            intro_memory = ReasoningMemory(
                collection_name=intro_collection,
                persist_directory=reasoning_dir
            )
            intro_agents.append(IntroAgent(
                role='intro',
                agent_id=f'intro_agent_{i+1}',
                reasoning_memory=intro_memory,
                shared_rag=self.shared_rag,
                system_prompt=self.agent_prompts.get('intro', '')
            ))

            # Body agents
            body_collection = generate_reasoning_collection_name('body', f'agent_{i+1}')
            body_memory = ReasoningMemory(
                collection_name=body_collection,
                persist_directory=reasoning_dir
            )
            body_agents.append(BodyAgent(
                role='body',
                agent_id=f'body_agent_{i+1}',
                reasoning_memory=body_memory,
                shared_rag=self.shared_rag,
                system_prompt=self.agent_prompts.get('body', '')
            ))

            # Conclusion agents
            conclusion_collection = generate_reasoning_collection_name('conclusion', f'agent_{i+1}')
            conclusion_memory = ReasoningMemory(
                collection_name=conclusion_collection,
                persist_directory=reasoning_dir
            )
            conclusion_agents.append(ConclusionAgent(
                role='conclusion',
                agent_id=f'conclusion_agent_{i+1}',
                reasoning_memory=conclusion_memory,
                shared_rag=self.shared_rag,
                system_prompt=self.agent_prompts.get('conclusion', '')
            ))

        # Create agent pools with full population
        self.agent_pools = {
            'intro': AgentPool(
                role='intro',
                initial_agents=intro_agents,
                max_size=population_size
            ),
            'body': AgentPool(
                role='body',
                initial_agents=body_agents,
                max_size=population_size
            ),
            'conclusion': AgentPool(
                role='conclusion',
                initial_agents=conclusion_agents,
                max_size=population_size
            )
        }

        # Build LangGraph
        self.app = self._build_graph()

    def _execute_ensemble(
        self,
        role: str,
        topic: str,
        reasoning_context: str,
        additional_context: str,
        generation_number: int,
        domain_knowledge: Optional[List[Dict]] = None
    ) -> Dict:
        """Execute all agents in a pool and select the best output.

        This implements true ensemble execution:
        1. All agents in the pool generate outputs in parallel
        2. Each output is scored individually
        3. Best output is selected for use
        4. All agents store their reasoning with individual scores

        Args:
            role: Agent role (intro, body, conclusion)
            topic: Topic to generate about
            reasoning_context: Context from previous agents
            additional_context: Coordinator or specialist context
            generation_number: Current generation number
            domain_knowledge: Optional domain knowledge to inject

        Returns:
            Dict with:
                - output: Best output selected
                - thinking: Best agent's thinking
                - all_results: List of (agent, result, score) tuples
                - winner_id: ID of winning agent
        """
        pool = self.agent_pools[role]
        results = []

        logger.info(f"[ENSEMBLE] Executing all {pool.size()} agents in {role} pool")

        # Execute all agents in the pool
        for agent in pool.agents:
            # Retrieve reasoning patterns for this agent
            reasoning_patterns = agent.reasoning_memory.retrieve_similar_reasoning(
                query=topic,
                k=int(os.getenv('MAX_REASONING_RETRIEVE', '5'))
            )

            # Retrieve domain knowledge
            if domain_knowledge is None:
                domain_knowledge = agent.shared_rag.retrieve(
                    query=topic,
                    k=int(os.getenv('MAX_KNOWLEDGE_RETRIEVE', '3'))
                )

            # Generate with reasoning
            result = agent.generate_with_reasoning(
                topic=topic,
                reasoning_patterns=reasoning_patterns,
                domain_knowledge=domain_knowledge,
                reasoning_context=reasoning_context,
                additional_context=additional_context
            )

            # Score this output
            score = self.evaluator.score_section(
                content=result['output'],
                section_type=role,
                topic=topic
            )

            logger.info(f"[ENSEMBLE] {agent.agent_id} scored {score:.1f}/10")

            # Prepare reasoning for storage (don't store yet, wait for selection)
            agent.prepare_reasoning_storage(
                thinking=result['thinking'],
                output=result['output'],
                topic=topic,
                domain=self.domain,
                generation=generation_number,
                context_sources=['coordinator', 'ensemble']
            )

            # Store serializable data only (no agent objects)
            results.append({
                'agent_id': agent.agent_id,
                'agent': agent,  # Keep for internal use
                'result': result,
                'score': score
            })

        # Select best result
        best = max(results, key=lambda x: x['score'])
        winner_agent = best['agent']
        winner_result = best['result']
        winner_score = best['score']

        logger.info(f"[ENSEMBLE] Winner: {winner_agent.agent_id} with score {winner_score:.1f}")

        # Create serializable version of results (without agent objects)
        serializable_results = [
            {
                'agent_id': r['agent_id'],
                'score': r['score'],
                'output_length': len(r['result']['output'])
            }
            for r in results
        ]

        return {
            'output': winner_result['output'],
            'thinking': winner_result['thinking'],
            'all_results': results,  # For internal use (has agent objects)
            'serializable_results': serializable_results,  # For state storage
            'winner_id': winner_agent.agent_id,
            'winner_score': winner_score
        }

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow with hierarchical architecture.

        Graph structure:
        START → research → distribute → intro → body → conclusion →
        aggregate → critique → [revise OR evaluate] → evolve → END

        Returns:
            Compiled LangGraph application
        """
        workflow = StateGraph(BlogState)

        # Coordinator nodes
        workflow.add_node("research", self._research_node)
        workflow.add_node("distribute", self._distribute_node)
        workflow.add_node("aggregate", self._aggregate_node)
        workflow.add_node("critique", self._critique_node)

        # Content agent nodes
        workflow.add_node("intro", self._intro_node)
        workflow.add_node("body", self._body_node)
        workflow.add_node("conclusion", self._conclusion_node)

        # Evaluation and evolution nodes
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("evolve", self._evolve_node)

        # Revision node (conditional)
        if self.enable_revision:
            workflow.add_node("revise", self._revise_node)

        # Define execution flow
        workflow.set_entry_point("research")
        workflow.add_edge("research", "distribute")
        workflow.add_edge("distribute", "intro")
        workflow.add_edge("intro", "body")
        workflow.add_edge("body", "conclusion")
        workflow.add_edge("conclusion", "aggregate")
        workflow.add_edge("aggregate", "critique")

        # Conditional: revise OR evaluate
        if self.enable_revision:
            workflow.add_conditional_edges(
                "critique",
                self._should_revise,
                {
                    "revise": "revise",
                    "evaluate": "evaluate"
                }
            )
            workflow.add_edge("revise", "intro")  # Loop back to intro for revision
        else:
            workflow.add_edge("critique", "evaluate")

        workflow.add_edge("evaluate", "evolve")
        workflow.add_edge("evolve", END)

        # Compile with memory for checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _research_node(self, state: BlogState) -> BlogState:
        """Coordinator researches topic using Tavily.

        Args:
            state: Current workflow state

        Returns:
            Updated state with research results
        """
        topic = state['topic']
        start_time = time.time()

        # Research topic
        research_results = self.coordinator.research_topic(
            topic=topic,
            max_results=int(os.getenv('TAVILY_MAX_RESULTS', '5')),
            search_depth=os.getenv('TAVILY_SEARCH_DEPTH', 'advanced')
        )

        # Store research results in state
        state['research_results'] = research_results
        state['stream_logs'].append(
            f"[research] Found {len(research_results.get('results', []))} sources"
        )

        # Track timing
        end_time = time.time()
        state['agent_timings']['research'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        return state

    async def _distribute_node(self, state: BlogState) -> BlogState:
        """Coordinator synthesizes research and distributes context.

        Args:
            state: Current workflow state

        Returns:
            Updated state with distributed contexts
        """
        start_time = time.time()

        # Get research results
        research_results = state.get('research_results', {})

        # Retrieve reasoning patterns for coordinator
        reasoning_patterns = self.coordinator.reasoning_memory.retrieve_similar_reasoning(
            query=state['topic'],
            k=int(os.getenv('MAX_REASONING_RETRIEVE', '5'))
        )

        # Retrieve domain knowledge
        domain_knowledge = self.coordinator.shared_rag.retrieve(
            query=state['topic'],
            k=int(os.getenv('MAX_KNOWLEDGE_RETRIEVE', '3'))
        )

        # Synthesize research into contexts
        synthesis = self.coordinator.synthesize_research(
            research_results=research_results,
            reasoning_patterns=reasoning_patterns,
            domain_knowledge=domain_knowledge
        )

        # Store distributed contexts
        state['coordinator_synthesis'] = synthesis
        state['intro_coordinator_context'] = synthesis['intro_context']
        state['body_coordinator_context'] = synthesis['body_context']
        state['conclusion_coordinator_context'] = synthesis['conclusion_context']

        state['stream_logs'].append("[distribute] Context distributed to content agents")

        # Track timing
        end_time = time.time()
        state['agent_timings']['distribute'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        return state

    async def _intro_node(self, state: BlogState) -> BlogState:
        """Execute intro agent ensemble with coordinator context.

        All agents in the intro pool compete, best output is selected.

        Args:
            state: Current workflow state

        Returns:
            Updated state with intro content
        """
        topic = state['topic']
        start_time = time.time()

        # Get coordinator context
        coordinator_context = state.get('intro_coordinator_context', '')

        # Execute ensemble: all agents compete
        ensemble_result = self._execute_ensemble(
            role='intro',
            topic=topic,
            reasoning_context="",
            additional_context=coordinator_context,
            generation_number=state['generation_number']
        )

        # Store winning output in state
        state['intro'] = ensemble_result['output']
        state['intro_reasoning'] = ensemble_result['thinking']

        # Store ensemble metadata for tracking (serializable only)
        state['intro_ensemble_results'] = ensemble_result['serializable_results']
        state['intro_winner_id'] = ensemble_result['winner_id']

        # Track timing
        end_time = time.time()
        state['agent_timings']['intro'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        state['stream_logs'].append(
            f"[intro] Ensemble complete - Winner: {ensemble_result['winner_id']} "
            f"(score: {ensemble_result['winner_score']:.1f})"
        )

        return state

    async def _body_node(self, state: BlogState) -> BlogState:
        """Execute body agent ensemble with coordinator context and optional specialist support.

        All agents in the body pool compete, best output is selected.

        Args:
            state: Current workflow state

        Returns:
            Updated state with body content
        """
        topic = state['topic']
        start_time = time.time()

        # Get coordinator context
        coordinator_context = state.get('body_coordinator_context', '')

        # Optionally invoke specialists (if enabled)
        specialist_context = ""
        if self.enable_specialists and self.specialists:
            specialist_context = await self._invoke_specialists(
                topic=topic,
                content_context=coordinator_context
            )

        # Combine coordinator and specialist contexts
        full_context = f"{coordinator_context}\n\nSPECIALIST INSIGHTS:\n{specialist_context}" if specialist_context else coordinator_context

        # Execute ensemble: all agents compete
        ensemble_result = self._execute_ensemble(
            role='body',
            topic=topic,
            reasoning_context=state.get('intro_reasoning', ''),
            additional_context=full_context,
            generation_number=state['generation_number']
        )

        # Store winning output in state
        state['body'] = ensemble_result['output']
        state['body_reasoning'] = ensemble_result['thinking']

        # Store ensemble metadata (serializable only)
        state['body_ensemble_results'] = ensemble_result['serializable_results']
        state['body_winner_id'] = ensemble_result['winner_id']

        # Track timing
        end_time = time.time()
        state['agent_timings']['body'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        state['stream_logs'].append(
            f"[body] Ensemble complete - Winner: {ensemble_result['winner_id']} "
            f"(score: {ensemble_result['winner_score']:.1f})"
        )

        return state

    async def _invoke_specialists(self, topic: str, content_context: str) -> str:
        """Invoke specialist agents for support.

        Args:
            topic: Topic being written about
            content_context: Context from coordinator

        Returns:
            Combined specialist insights
        """
        insights = []

        # Researcher insights
        if 'researcher' in self.specialists:
            research = self.specialists['researcher'].research_claim(
                claim=f"Research insights for: {topic}",
                content_context=content_context
            )
            insights.append(f"RESEARCH: {research['findings'][:300]}...")

        return "\n\n".join(insights) if insights else ""

    async def _conclusion_node(self, state: BlogState) -> BlogState:
        """Execute conclusion agent ensemble with coordinator context.

        All agents in the conclusion pool compete, best output is selected.

        Args:
            state: Current workflow state

        Returns:
            Updated state with conclusion content
        """
        topic = state['topic']
        start_time = time.time()

        # Get coordinator context
        coordinator_context = state.get('conclusion_coordinator_context', '')

        # Execute ensemble: all agents compete
        ensemble_result = self._execute_ensemble(
            role='conclusion',
            topic=topic,
            reasoning_context=f"{state.get('intro_reasoning', '')}\n\n{state.get('body_reasoning', '')}",
            additional_context=coordinator_context,
            generation_number=state['generation_number']
        )

        # Store winning output in state
        state['conclusion'] = ensemble_result['output']
        state['conclusion_reasoning'] = ensemble_result['thinking']

        # Store ensemble metadata (serializable only)
        state['conclusion_ensemble_results'] = ensemble_result['serializable_results']
        state['conclusion_winner_id'] = ensemble_result['winner_id']

        # Track timing
        end_time = time.time()
        state['agent_timings']['conclusion'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        state['stream_logs'].append(
            f"[conclusion] Ensemble complete - Winner: {ensemble_result['winner_id']} "
            f"(score: {ensemble_result['winner_score']:.1f})"
        )

        return state

    async def _aggregate_node(self, state: BlogState) -> BlogState:
        """Coordinator aggregates outputs from content agents.

        Args:
            state: Current workflow state

        Returns:
            Updated state (no changes, just pass-through)
        """
        state['stream_logs'].append("[aggregate] Content aggregated by coordinator")
        return state

    async def _critique_node(self, state: BlogState) -> BlogState:
        """Coordinator critiques aggregated output.

        Args:
            state: Current workflow state

        Returns:
            Updated state with critique
        """
        start_time = time.time()

        # Get content
        intro = state.get('intro', '')
        body = state.get('body', '')
        conclusion = state.get('conclusion', '')
        topic = state['topic']
        research_context = state.get('research_results')

        # Critique
        critique = self.coordinator.critique_output(
            intro=intro,
            body=body,
            conclusion=conclusion,
            topic=topic,
            research_context=research_context
        )

        # Store critique
        state['coordinator_critique'] = critique
        state['revision_count'] = state.get('revision_count', 0)

        # Log
        overall_score = critique.get('scores', {}).get('overall', 0)
        revision_needed = critique.get('revision_needed', False)

        state['stream_logs'].append(
            f"[critique] Overall score: {overall_score:.1f}, "
            f"Revision needed: {revision_needed}"
        )

        # Track timing
        end_time = time.time()
        state['agent_timings']['critique'] = {
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        }

        return state

    def _should_revise(self, state: BlogState) -> str:
        """Conditional edge: decide whether to revise or evaluate.

        Args:
            state: Current workflow state

        Returns:
            "revise" or "evaluate"
        """
        critique = state.get('coordinator_critique', {})
        revision_needed = critique.get('revision_needed', False)
        revision_count = state.get('revision_count', 0)

        # Revise if needed and under max revisions
        if revision_needed and revision_count < self.max_revisions:
            state['revision_count'] = revision_count + 1
            state['stream_logs'].append(f"[revise] Starting revision {revision_count + 1}/{self.max_revisions}")
            return "revise"

        return "evaluate"

    async def _revise_node(self, state: BlogState) -> BlogState:
        """Prepare state for revision (pass critique feedback to agents).

        Args:
            state: Current workflow state

        Returns:
            Updated state with revision feedback
        """
        critique = state.get('coordinator_critique', {})
        feedback = critique.get('feedback', '')

        # Add feedback to coordinator contexts for next iteration
        state['intro_coordinator_context'] = f"{state.get('intro_coordinator_context', '')}\n\nREVISION FEEDBACK:\n{feedback}"
        state['body_coordinator_context'] = f"{state.get('body_coordinator_context', '')}\n\nREVISION FEEDBACK:\n{feedback}"
        state['conclusion_coordinator_context'] = f"{state.get('conclusion_coordinator_context', '')}\n\nREVISION FEEDBACK:\n{feedback}"

        state['stream_logs'].append("[revise] Feedback added to contexts for revision")

        return state

    def _evaluate_node(self, state: BlogState) -> BlogState:
        """Evaluate generated content.

        Args:
            state: Current workflow state

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
        """Store reasoning patterns and evolve agents.

        In ensemble mode, each agent has individual scores from competition.

        Args:
            state: Current workflow state

        Returns:
            Updated state
        """
        # Store reasoning with INDIVIDUAL scores from ensemble competition
        for role in ['intro', 'body', 'conclusion']:
            # Use agent pool directly to access agents
            pool = self.agent_pools[role]
            ensemble_results_key = f'{role}_ensemble_results'

            if ensemble_results_key in state and state[ensemble_results_key]:
                # Ensemble mode: match agents with their scores
                serializable_results = state[ensemble_results_key]

                # Create agent_id to score mapping
                score_map = {r['agent_id']: r['score'] for r in serializable_results}

                # Record fitness and store reasoning for all agents
                for agent in pool.agents:
                    if agent.agent_id in score_map:
                        score = score_map[agent.agent_id]

                        # Record individual fitness
                        agent.record_fitness(score=score, domain=self.domain)

                        # Store reasoning (all agents executed, all have pending_reasoning)
                        if agent.pending_reasoning is not None:
                            agent.store_reasoning_and_output(score=score)

                logger.info(f"[EVOLVE] {role} pool: stored reasoning for {len(serializable_results)} agents")
            else:
                # Fallback: legacy single-agent mode (shouldn't happen with new code)
                logger.warning(f"[EVOLVE] No ensemble results for {role}, using fallback")
                pool = self.agent_pools[role]
                score = state['scores'].get(role, 0.0)

                for agent in pool.agents:
                    agent.record_fitness(score=score, domain=self.domain)
                    if agent.pending_reasoning is not None:
                        agent.store_reasoning_and_output(score=score)

        # Get pool statistics
        intro_pool_avg = self.agent_pools['intro'].avg_fitness()
        body_pool_avg = self.agent_pools['body'].avg_fitness()
        conclusion_pool_avg = self.agent_pools['conclusion'].avg_fitness()

        state['stream_logs'].append(
            f"[evolve] Pool avg fitness: "
            f"Intro: {intro_pool_avg:.1f}, "
            f"Body: {body_pool_avg:.1f}, "
            f"Conclusion: {conclusion_pool_avg:.1f}"
        )

        # Trigger evolution every N generations
        if self.generation_counter > 0 and self.generation_counter % self.evolution_frequency == 0:
            state['stream_logs'].append(
                f"[evolve] 🧬 EVOLUTION TRIGGERED (generation {self.generation_counter})"
            )

            for role, pool in self.agent_pools.items():
                pool.evolve_generation(
                    reproduction_strategy=self.reproduction_strategy,
                    shared_rag=self.shared_rag
                )

                state['stream_logs'].append(
                    f"[evolve] {role.capitalize()} pool → Generation {pool.generation} "
                    f"(avg fitness: {pool.avg_fitness():.1f})"
                )

        return state

    async def generate(
        self,
        topic: str,
        config: Dict | None = None,
        generation_number: int = 0
    ) -> BlogState:
        """Generate blog post with hierarchical architecture.

        Args:
            topic: The topic to write about
            config: Optional LangGraph configuration
            generation_number: Which generation this is

        Returns:
            Final state with generated content
        """
        # Initialize state
        initial_state = create_initial_state(topic)
        initial_state['generation_number'] = generation_number
        self.generation_counter += 1

        # Configure streaming
        if config is None:
            config = {
                "configurable": {"thread_id": f"blog_v3_{topic.replace(' ', '_')}"}
            }

        # Stream execution
        final_state = initial_state

        if os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true":
            async def state_stream() -> AsyncIterator[BlogState]:
                nonlocal final_state
                async for event in self.app.astream(
                    initial_state, config, stream_mode="values"
                ):
                    final_state = event
                    yield event

            await self.visualizer.display_stream(state_stream())
        else:
            final_state = await self.app.ainvoke(initial_state, config)

        return final_state
