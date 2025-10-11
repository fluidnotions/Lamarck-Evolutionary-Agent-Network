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
        # Initialize agents
        self.agents = create_agents(persist_directory)

        # Initialize evaluator and visualizer
        self.evaluator = ContentEvaluator()
        self.visualizer = StreamVisualizer()

        # Build LangGraph
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct LangGraph workflow.

        Graph structure:
        START → intro → body → conclusion → evaluate → evolve → END

        Returns:
            Compiled LangGraph application
        """
        workflow = StateGraph(BlogState)

        # Add agent nodes
        workflow.add_node("intro", self.agents["intro"])
        workflow.add_node("body", self.agents["body"])
        workflow.add_node("conclusion", self.agents["conclusion"])

        # Add evaluation node
        workflow.add_node("evaluate", self.evaluator)

        # Add evolution node
        workflow.add_node("evolve", self._evolution_node)

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

    async def _evolution_node(self, state: BlogState) -> BlogState:
        """Evolution node: store memories and update parameters.

        Args:
            state: Current workflow state with scores

        Returns:
            Updated state
        """
        for role, agent in self.agents.items():
            score = state["scores"].get(role, 0)

            # Store memory if score meets threshold
            agent.store_memory(score)

            # Evolve parameters based on score
            agent.evolve_parameters(score, state)

        state["stream_logs"].append(
            "[Evolution] Memories stored, parameters updated"
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
