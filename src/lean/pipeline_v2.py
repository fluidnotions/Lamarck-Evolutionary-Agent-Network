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


class PipelineV2:
    """Pipeline V2 with reasoning pattern architecture and 8-step learning cycle."""

    def __init__(
        self,
        reasoning_dir: str = "./data/reasoning",
        shared_rag_dir: str = "./data/shared_rag",
        agent_ids: Optional[Dict[str, str]] = None,
        domain: str = "General"
    ):
        """Initialize V2 pipeline.

        Args:
            reasoning_dir: Directory for per-agent reasoning patterns
            shared_rag_dir: Directory for shared knowledge base
            agent_ids: Optional dict mapping role → agent_id
            domain: Domain category for this pipeline instance
        """
        self.domain = domain
        self.generation_counter = 0

        # Create V2 agents with factory
        self.agents = create_agents_v2(
            reasoning_dir=reasoning_dir,
            shared_rag_dir=shared_rag_dir,
            agent_ids=agent_ids
        )

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

        # Create agent pools for context manager
        # For now, just wrap agents in simple structure
        # In full M2 implementation, these would be proper AgentPool instances
        self.agent_pools = {
            'intro': SimpleAgentPool('intro', [self.agents['intro']]),
            'body': SimpleAgentPool('body', [self.agents['body']]),
            'conclusion': SimpleAgentPool('conclusion', [self.agents['conclusion']])
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
        agent = self.agents['intro']
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
        agent = self.agents['body']
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
        2. PLAN → Retrieve similar reasoning patterns
        3. RETRIEVE → Query shared RAG
        4. CONTEXT → Assemble reasoning traces (includes intro + body)
        5. GENERATE → Create content with reasoning

        Args:
            state: Current workflow state

        Returns:
            Updated state with conclusion content and reasoning
        """
        agent = self.agents['conclusion']
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
        """Store reasoning patterns and evolve agents (STEP 7).

        Steps:
        - Record fitness for each agent
        - Store reasoning patterns (all patterns, no threshold)
        - Store outputs in shared RAG (only if score >= 8.0)

        Note: STEP 8 (reproduction, selection) is handled separately in M2.

        Args:
            state: Current workflow state with scores

        Returns:
            Updated state
        """
        # Store reasoning and outputs for each agent
        for role, agent in self.agents.items():
            score = state['scores'].get(role, 0.0)

            # Record fitness
            agent.record_fitness(score=score, domain=self.domain)

            # Store reasoning pattern and conditionally store output
            agent.store_reasoning_and_output(score=score)

        # Get statistics for logging
        intro_stats = self.agents['intro'].get_stats()
        body_stats = self.agents['body'].get_stats()
        conclusion_stats = self.agents['conclusion'].get_stats()

        # Log
        state['stream_logs'].append(
            f"[evolve] Reasoning patterns stored. "
            f"Intro: {intro_stats['reasoning_patterns']} patterns (avg: {intro_stats['avg_fitness']:.1f}), "
            f"Body: {body_stats['reasoning_patterns']} patterns (avg: {body_stats['avg_fitness']:.1f}), "
            f"Conclusion: {conclusion_stats['reasoning_patterns']} patterns (avg: {conclusion_stats['avg_fitness']:.1f})"
        )

        # Log shared RAG growth
        shared_rag_stats = self.agents['intro'].shared_rag.get_stats()
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
        """Get statistics for all agents.

        Returns:
            Dictionary of agent stats
        """
        return {
            role: agent.get_stats()
            for role, agent in self.agents.items()
        }

    def get_shared_rag_stats(self) -> Dict:
        """Get shared RAG statistics.

        Returns:
            Shared RAG statistics dict
        """
        return self.agents['intro'].shared_rag.get_stats()

    def get_context_flow_stats(self) -> Dict:
        """Get context flow diversity statistics.

        Returns:
            Context flow diversity metrics
        """
        return self.context_manager.measure_diversity(recent_n=10)


class SimpleAgentPool:
    """Simple agent pool wrapper for ContextManager compatibility."""

    def __init__(self, role: str, agents: list):
        self.role = role
        self.agents = agents

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
        sorted_agents = sorted(self.agents, key=lambda a: a.avg_fitness())
        lower_half = sorted_agents[:len(sorted_agents)//2] if len(sorted_agents) > 1 else sorted_agents
        return random.choice(lower_half) if lower_half else self.agents[0]

    def size(self):
        """Get pool size."""
        return len(self.agents)
