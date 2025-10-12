"""
Main pipeline orchestrator for HVAS Mini.

Coordinates agents, evaluation, and evolution using LangGraph.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, AsyncIterator
import os
from dotenv import load_dotenv

load_dotenv()

# Import dependencies from other branches
try:
    from hvas_mini.state import BlogState, create_initial_state
    from hvas_mini.agents import create_agents
    from hvas_mini.evaluation import ContentEvaluator
    from hvas_mini.visualization import StreamVisualizer
    from hvas_mini.orchestration.async_coordinator import AsyncCoordinator
    from hvas_mini.weighting.trust_manager import TrustManager
    from hvas_mini.weighting.weight_updates import update_all_weights
except ImportError:
    # For standalone development
    print("Warning: Some dependencies not available in standalone mode")


class HVASMiniPipeline:
    """Main pipeline orchestrator for HVAS Mini system."""

    def __init__(self, persist_directory: str = "./data/memories"):
        """Initialize pipeline.

        Args:
            persist_directory: Where to persist agent memories
        """
        # NEW: Initialize trust manager for agent weighting
        self.trust_manager = TrustManager(
            initial_weight=float(os.getenv("INITIAL_TRUST_WEIGHT", "0.5")),
            learning_rate=float(os.getenv("TRUST_LEARNING_RATE", "0.1")),
        )

        # Initialize agents with trust manager
        self.agents = create_agents(persist_directory, self.trust_manager)

        # Initialize evaluator and visualizer
        self.evaluator = ContentEvaluator()
        self.visualizer = StreamVisualizer()

        # NEW: Initialize async coordinator
        self.coordinator = AsyncCoordinator()

        # Build LangGraph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow with concurrent execution.

        NEW Graph structure:
        START → intro → [body ∥ conclusion] → evaluate → evolve → END

        Agents in brackets execute concurrently.

        Returns:
            Compiled LangGraph application
        """
        workflow = StateGraph(BlogState)

        # Layer 1: Intro (sequential - needs topic context)
        workflow.add_node("intro", self.agents["intro"])

        # Layer 2: Body & Conclusion (CONCURRENT)
        # Both can read intro, but don't depend on each other
        workflow.add_node("body_and_conclusion", self._concurrent_layer_2)

        # Layer 3: Evaluation (sequential - needs all content)
        workflow.add_node("evaluate", self.evaluator)

        # Layer 4: Evolution (sequential)
        workflow.add_node("evolve", self._evolution_node)

        # Define execution flow
        workflow.set_entry_point("intro")
        workflow.add_edge("intro", "body_and_conclusion")
        workflow.add_edge("body_and_conclusion", "evaluate")
        workflow.add_edge("evaluate", "evolve")
        workflow.add_edge("evolve", END)

        # Compile with memory for checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _concurrent_layer_2(self, state: BlogState) -> BlogState:
        """Execute body and conclusion agents concurrently.

        This is a LangGraph node that internally runs multiple agents in parallel.
        """
        # Execute both agents concurrently using coordinator
        agents = [self.agents["body"], self.agents["conclusion"]]

        updated_state = await self.coordinator.execute_layer(
            layer_name="layer_2_content",
            agents=agents,
            state=state,
            timeout=60.0
        )

        # Log concurrent execution
        updated_state["stream_logs"].append(
            "[Pipeline] Body and Conclusion executed concurrently"
        )

        return updated_state

    async def _evolution_node(self, state: BlogState) -> BlogState:
        """Evolution node: update weights, store memories, and update parameters.

        Args:
            state: Current workflow state with scores

        Returns:
            Updated state
        """
        # 1. NEW: Update trust weights based on performance signals
        weight_updates = update_all_weights(
            self.trust_manager, state, state["scores"]
        )

        # Store updated weights in state
        state["agent_weights"] = self.trust_manager.get_all_weights()
        state["weight_history"].extend(weight_updates)

        # 2. Store memories and evolve parameters (existing logic)
        for role, agent in self.agents.items():
            score = state["scores"].get(role, 0)

            # Store memory if score meets threshold
            agent.store_memory(score)

            # Evolve parameters based on score
            agent.evolve_parameters(score, state)

        # 3. Log with weight update count
        state["stream_logs"].append(
            f"[Evolution] Weights updated ({len(weight_updates)} relationships), "
            f"memories stored, parameters evolved"
        )

        return state

    async def generate(
        self, topic: str, config: Dict | None = None
    ) -> BlogState:
        """Generate blog post with streaming visualization.

        Args:
            topic: The topic to write about
            config: Optional LangGraph configuration

        Returns:
            Final state with generated content and scores
        """
        # Initialize state
        initial_state = create_initial_state(topic)

        # Configure streaming
        if config is None:
            config = {
                "configurable": {"thread_id": f"blog_{topic.replace(' ', '_')}"}
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

    def get_memory_stats(self) -> Dict:
        """Get memory statistics for all agents.

        Returns:
            Dictionary of memory stats per agent
        """
        return {role: agent.memory.get_stats() for role, agent in self.agents.items()}

    def get_agent_parameters(self) -> Dict:
        """Get current parameters for all agents.

        Returns:
            Dictionary of parameters per agent
        """
        return {role: agent.parameters for role, agent in self.agents.items()}
