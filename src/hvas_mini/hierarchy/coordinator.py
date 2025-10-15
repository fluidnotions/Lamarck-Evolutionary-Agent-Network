"""
Coordinator agent - top-level orchestrator (Layer 1).
"""

from hvas_mini.agents import BaseAgent
from hvas_mini.hierarchy.structure import AgentHierarchy
from hvas_mini.state import AgentOutput
from typing import Dict, List, Any


class CoordinatorAgent(BaseAgent):
    """Top-level orchestrator agent (Layer 1).

    Responsibilities:
    - Parse user intent from topic
    - Define high-level goals
    - Distribute context to content agents
    - Aggregate results from content agents
    - Critique outputs (used in M8)
    """

    def __init__(self, hierarchy: AgentHierarchy, memory_manager, trust_manager=None):
        """Initialize coordinator.

        Args:
            hierarchy: AgentHierarchy instance
            memory_manager: MemoryManager for coordinator
            trust_manager: Optional TrustManager
        """
        super().__init__(role="coordinator", memory_manager=memory_manager, trust_manager=trust_manager)
        self.hierarchy = hierarchy

    @property
    def content_key(self) -> str:
        """State key for coordinator output."""
        return "coordinator_output"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Parse intent and set high-level goals.

        Args:
            state: Current HierarchicalState
            memories: Retrieved memories
            weighted_context: Context from peers (unused at Layer 1)

        Returns:
            Intent statement
        """
        topic = state["topic"]

        prompt = f"""You are a high-level coordinator for a blog writing system.

Topic: {topic}

Your role:
1. Parse the intent of this topic
2. Define what makes a good blog post on this topic
3. Set high-level constraints and goals

Provide a brief intent statement (2-3 sentences) that captures:
- The core message to convey
- The target audience
- Key points to cover

Intent:"""

        response = await self.llm.ainvoke(prompt)
        return response.content

    def distribute_context(self, state, use_semantic_filtering: bool = True) -> Dict[str, str]:
        """Create context for each direct child.

        With M9, uses semantic distance to filter context appropriately.

        Args:
            state: Current HierarchicalState
            use_semantic_filtering: Whether to apply semantic distance filtering

        Returns:
            {child_role: context_for_child}
        """
        from hvas_mini.hierarchy.semantic import compute_semantic_distance, filter_context_by_distance

        contexts = {}
        intent = state["coordinator_intent"]

        for child_role in self.hierarchy.get_children(self.role):
            if use_semantic_filtering:
                # Calculate semantic distance
                distance = compute_semantic_distance(self.hierarchy, self.role, child_role)

                # Filter context based on distance
                filtered_context = filter_context_by_distance(intent, distance, min_ratio=0.3)
                contexts[child_role] = filtered_context
            else:
                # Full context (M6/M7 behavior)
                contexts[child_role] = intent

        return contexts

    def aggregate_results(self, state, layer: int) -> AgentOutput:
        """Combine results from a layer.

        Args:
            state: Current HierarchicalState
            layer: Layer number to aggregate

        Returns:
            Aggregated AgentOutput
        """
        outputs = state["layer_outputs"][layer]

        if not outputs:
            return AgentOutput(
                content="No outputs to aggregate",
                confidence=0.0,
                metadata={}
            )

        # Combine all content
        combined_parts = []
        for role, output in outputs.items():
            combined_parts.append(f"## {role.title()}\n{output['content']}")

        combined_content = "\n\n".join(combined_parts)

        # Average confidence
        avg_confidence = sum(
            output['confidence'] for output in outputs.values()
        ) / len(outputs)

        return AgentOutput(
            content=combined_content,
            confidence=avg_confidence,
            metadata={
                "sources": list(outputs.keys()),
                "layer": layer
            }
        )

    def critique_outputs(self, state) -> Dict[str, str]:
        """Generate critique for each content agent.

        Used in M8 for closed-loop refinement.

        Args:
            state: Current HierarchicalState

        Returns:
            {agent_role: critique_message}
        """
        critiques = {}

        for role in ["intro", "body", "conclusion"]:
            if role not in state["layer_outputs"][2]:
                critiques[role] = "No output to critique"
                continue

            output = state["layer_outputs"][2][role]

            # Simple heuristic critique (can be LLM-based in future)
            issues = []

            if len(output['content']) < 100:
                issues.append("too short (< 100 chars)")

            if output['confidence'] < 0.7:
                issues.append(f"low confidence ({output['confidence']:.2f})")

            if not issues:
                critiques[role] = "Good quality"
            else:
                critiques[role] = f"Issues: {', '.join(issues)}"

        return critiques
