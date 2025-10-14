"""
Hierarchical execution engine with bidirectional flow.

Manages context distribution (downward) and result aggregation (upward).
"""

from typing import Dict, List
from hvas_mini.state import HierarchicalState, AgentOutput
from hvas_mini.agents import BaseAgent


class HierarchicalExecutor:
    """Executes workflow with bidirectional flow through hierarchy.

    Downward flow: Context passes from parent to children
    Upward flow: Results aggregate from children to parent
    """

    def __init__(self, agents: Dict[str, BaseAgent]):
        """Initialize executor with agent dict.

        Args:
            agents: Dictionary of {role: agent_instance}
        """
        self.agents = agents

    async def execute_downward(self, state: HierarchicalState, layer: int):
        """Execute layer with downward context flow.

        Distributes context from parent layer to current layer agents,
        then executes all agents in the layer.

        Args:
            state: Current hierarchical state
            layer: Layer number to execute (1-3)
        """
        hierarchy = state["hierarchy"]

        for agent_role in hierarchy.get_layer_agents(layer):
            parent_role = hierarchy.get_parent(agent_role)

            # Get context from parent (or topic if no parent)
            if parent_role and parent_role in state["layer_outputs"].get(layer - 1, {}):
                context = state["layer_outputs"][layer - 1][parent_role]["content"]
            else:
                # Layer 1 (coordinator) gets topic as context
                context = state.get("coordinator_intent", state["topic"])

            # Execute agent with parent context
            agent = self.agents[agent_role]

            # For now, call agent's generate_content directly
            # (Full integration with agent.__call__ will come in pipeline)
            memories = agent.memory.retrieve(state["topic"])
            output = await agent.generate_content(state, memories, weighted_context=context)

            # Store with confidence
            state["layer_outputs"][layer][agent_role] = AgentOutput(
                content=output,
                confidence=self._estimate_confidence(output),
                metadata={"parent": parent_role, "layer": layer}
            )

    async def execute_upward(self, state: HierarchicalState, layer: int):
        """Aggregate child results upward to parent.

        Collects outputs from child layer and attaches aggregated
        information to parent's metadata.

        Args:
            state: Current hierarchical state
            layer: Layer number to aggregate from (1-2, children in layer+1)
        """
        hierarchy = state["hierarchy"]

        for agent_role in hierarchy.get_layer_agents(layer):
            children = hierarchy.get_children(agent_role)

            if not children:
                continue

            # Gather child outputs from layer below
            child_layer = layer + 1
            child_outputs = []

            for child in children:
                if child in state["layer_outputs"].get(child_layer, {}):
                    child_outputs.append(state["layer_outputs"][child_layer][child])

            if not child_outputs:
                continue

            # Aggregate child results
            aggregated = self._aggregate_outputs(child_outputs)

            # Update parent's metadata with child aggregation
            if agent_role in state["layer_outputs"][layer]:
                state["layer_outputs"][layer][agent_role]["metadata"]["children"] = aggregated

    def _estimate_confidence(self, content: str) -> float:
        """Estimate output confidence based on heuristics.

        Simple heuristic based on length and structure.
        Can be enhanced with LLM-based quality scoring.

        Args:
            content: Generated content string

        Returns:
            Confidence score in [0, 1]
        """
        if not content:
            return 0.1

        # Length score (500 chars = 1.0)
        length_score = min(len(content) / 500, 1.0)

        # Structure score (has paragraphs?)
        paragraphs = content.count("\n\n")
        structure_score = min(paragraphs / 3, 1.0)

        # Combined (weight length more heavily)
        confidence = (length_score * 0.7) + (structure_score * 0.3)

        return max(0.1, min(1.0, confidence))

    def _aggregate_outputs(self, outputs: List[AgentOutput]) -> Dict:
        """Combine multiple child outputs with confidence weighting.

        Args:
            outputs: List of AgentOutput from children

        Returns:
            Aggregated dictionary with combined content and weighted confidence
        """
        if not outputs:
            return {
                "combined_content": "",
                "weighted_confidence": 0.0,
                "source_count": 0
            }

        total_confidence = sum(o["confidence"] for o in outputs)

        # Calculate weights based on confidence
        if total_confidence == 0:
            weights = [1.0 / len(outputs)] * len(outputs)
        else:
            weights = [o["confidence"] / total_confidence for o in outputs]

        # Combine content from all children
        combined_parts = []
        for output in outputs:
            combined_parts.append(output["content"])

        return {
            "combined_content": "\n\n".join(combined_parts),
            "weighted_confidence": sum(
                o["confidence"] * w for o, w in zip(outputs, weights)
            ),
            "source_count": len(outputs),
            "individual_confidences": [o["confidence"] for o in outputs]
        }

    async def execute_full_cycle(self, state: HierarchicalState) -> HierarchicalState:
        """Execute full downward + upward cycle through all layers.

        This is a single pass through the hierarchy:
        1. Downward: Layer 1 → 2 → 3 (context distribution)
        2. Upward: Layer 3 → 2 → 1 (result aggregation)

        Args:
            state: Current hierarchical state

        Returns:
            Updated state after full cycle
        """
        # Phase 1: Downward execution (distribute context)
        for layer in [1, 2, 3]:
            await self.execute_downward(state, layer)

        # Phase 2: Upward aggregation (combine results)
        for layer in [3, 2, 1]:
            await self.execute_upward(state, layer)

        return state
