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
                confidence=float(self._estimate_confidence(output)),
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
            weights = [float(1.0 / len(outputs))] * len(outputs)
        else:
            weights = [float(o["confidence"] / total_confidence) for o in outputs]

        # Combine content from all children
        combined_parts = []
        for output in outputs:
            combined_parts.append(output["content"])

        return {
            "combined_content": "\n\n".join(combined_parts),
            "weighted_confidence": float(sum(
                o["confidence"] * w for o, w in zip(outputs, weights)
            )),
            "source_count": len(outputs),
            "individual_confidences": [float(o["confidence"]) for o in outputs]
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

    async def execute_with_refinement(self, state: HierarchicalState) -> HierarchicalState:
        """Execute with closed-loop refinement (M8).

        Multi-pass execution with coordinator critique and revision:
        - Executes full cycle (down + up)
        - Coordinator critiques outputs
        - If quality threshold not met, requests revision and repeats
        - Early exit if quality threshold met or max passes reached

        Args:
            state: Current hierarchical state

        Returns:
            Updated state after refinement process
        """
        import os

        for pass_num in range(1, state["max_passes"] + 1):
            state["current_pass"] = pass_num

            # Execute full cycle (downward + upward)
            for layer in [1, 2, 3]:
                await self.execute_downward(state, layer)

            for layer in [3, 2, 1]:
                await self.execute_upward(state, layer)

            # Record pass results
            pass_record = {
                "pass": pass_num,
                "scores": {
                    role: state["layer_outputs"][2][role]["confidence"]
                    for role in ["intro", "body", "conclusion"]
                    if role in state["layer_outputs"][2]
                }
            }
            state["pass_history"].append(pass_record)

            # Critique and decide if revision needed
            needs_revision = await self.critique_and_decide(state)

            if not needs_revision:
                state["quality_threshold_met"] = True
                break

            # Prepare for next pass if needed
            if pass_num < state["max_passes"]:
                await self.request_revision(state)

        return state

    async def critique_and_decide(self, state: HierarchicalState) -> bool:
        """Critique outputs and decide if revision is needed.

        Uses coordinator agent to critique each content agent's output.
        Checks if average confidence meets quality threshold.

        Args:
            state: Current hierarchical state

        Returns:
            True if revision is needed, False if quality is sufficient
        """
        import os

        # Get coordinator's critique
        coordinator = self.agents.get("coordinator")
        if coordinator:
            critiques = coordinator.critique_outputs(state)
            state["coordinator_critique"] = critiques

        # Calculate average confidence from Layer 2 (content agents)
        confidences = []
        for role in ["intro", "body", "conclusion"]:
            if role in state["layer_outputs"][2]:
                confidences.append(state["layer_outputs"][2][role]["confidence"])

        if not confidences:
            return False  # No outputs to critique

        avg_confidence = float(sum(confidences) / len(confidences))

        # Check quality threshold
        threshold = float(os.getenv("QUALITY_THRESHOLD", "0.8"))

        if avg_confidence >= threshold:
            return False  # Quality met, no revision needed

        # Check if we have passes remaining
        if state["current_pass"] >= state["max_passes"]:
            return False  # No more attempts

        return True  # Revision needed

    async def request_revision(self, state: HierarchicalState):
        """Generate specific revision feedback for content agents.

        Creates detailed revision instructions based on coordinator critique.
        Updates state to indicate revision is requested.

        Args:
            state: Current hierarchical state
        """
        state["revision_requested"] = True

        for role in ["intro", "body", "conclusion"]:
            if role not in state["coordinator_critique"]:
                continue

            critique = state["coordinator_critique"][role]

            # Skip if already good quality
            if "Good quality" in critique:
                continue

            # Get current output
            if role not in state["layer_outputs"][2]:
                continue

            current_output = state["layer_outputs"][2][role]["content"]

            # Generate detailed revision instruction
            revision_prompt = f"""Revision needed for {role}:

Issue: {critique}

Original output:
{current_output}

Please revise to address the issues identified. Focus on improving:
- Length and completeness
- Structure and organization
- Quality and coherence"""

            # Store revision instruction for next pass
            state["coordinator_critique"][role] = revision_prompt
