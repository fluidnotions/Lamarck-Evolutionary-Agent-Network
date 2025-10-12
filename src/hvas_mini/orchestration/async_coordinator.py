"""
Coordinates concurrent agent execution with layer synchronization.
"""

import asyncio
from typing import List, Dict, Callable, Awaitable
from datetime import datetime
import time


class AsyncCoordinator:
    """Manages concurrent execution of agents within layers."""

    def __init__(self):
        self.active_agents = {}
        self.layer_results = {}

    async def execute_layer(
        self,
        layer_name: str,
        agents: List[Callable],
        state: Dict,
        timeout: float = 30.0
    ) -> Dict:
        """Execute multiple agents concurrently in a layer.

        Args:
            layer_name: Identifier for this execution layer
            agents: List of async callables (agent instances)
            state: Shared state dict
            timeout: Max time to wait for layer completion

        Returns:
            Updated state after all agents complete
        """
        layer_start = time.time()

        # Record start times for each agent
        for agent in agents:
            agent_name = agent.role if hasattr(agent, 'role') else str(agent)
            state["agent_timings"][agent_name] = {
                "start": layer_start,
                "end": None,
                "duration": None
            }

        # Execute all agents concurrently
        try:
            tasks = [agent(state) for agent in agents]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )

            # Merge results (all agents update same state)
            # Last write wins for conflicting keys
            final_state = state
            for result in results:
                if isinstance(result, Exception):
                    print(f"[AsyncCoordinator] Agent error: {result}")
                    continue
                if isinstance(result, dict):
                    final_state = result  # Agents return updated state

            # Record end times
            layer_end = time.time()
            for agent in agents:
                agent_name = agent.role if hasattr(agent, 'role') else str(agent)
                if agent_name in final_state["agent_timings"]:
                    final_state["agent_timings"][agent_name]["end"] = layer_end
                    final_state["agent_timings"][agent_name]["duration"] = (
                        layer_end - final_state["agent_timings"][agent_name]["start"]
                    )

            # Record barrier
            final_state["layer_barriers"].append({
                "layer": layer_name,
                "agents": [a.role if hasattr(a, 'role') else str(a) for a in agents],
                "wait_time": layer_end - layer_start
            })

            return final_state

        except asyncio.TimeoutError:
            print(f"[AsyncCoordinator] Layer {layer_name} timed out after {timeout}s")
            return state


class LayerBarrier:
    """Synchronization point between hierarchical layers."""

    def __init__(self, layer_name: str):
        self.layer_name = layer_name
        self.waiting_agents = set()
        self.completed_agents = set()
        self.lock = asyncio.Lock()

    async def wait(self, agent_name: str, expected_count: int):
        """Wait for all agents in layer to complete.

        Args:
            agent_name: Name of agent waiting
            expected_count: Total agents expected in this layer
        """
        async with self.lock:
            self.completed_agents.add(agent_name)

        # Wait until all agents complete
        while len(self.completed_agents) < expected_count:
            await asyncio.sleep(0.01)

    def reset(self):
        """Reset barrier for next use."""
        self.completed_agents.clear()
