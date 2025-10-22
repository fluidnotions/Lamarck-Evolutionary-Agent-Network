"""
LangGraph debugging and tracing utilities.

Provides instrumentation for LangGraph execution flow visualization and debugging.
"""

import time
from typing import Any, Dict, Callable, Optional
from functools import wraps
from loguru import logger
import json


class GraphDebugger:
    """Instruments LangGraph nodes for debugging and tracing."""

    def __init__(self, enabled: bool = True):
        """Initialize graph debugger.

        Args:
            enabled: Whether debugging is enabled
        """
        self.enabled = enabled
        self.node_timings: Dict[str, list] = {}
        self.execution_order: list = []

    def wrap_node(self, node_name: str):
        """Decorator to wrap a LangGraph node with timing and logging.

        Args:
            node_name: Name of the node being wrapped

        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def async_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
                if not self.enabled:
                    return await func(state)

                # Log node entry
                logger.info(f"[LANGGRAPH] → Entering node: {node_name}")

                # Track execution order
                self.execution_order.append({
                    'node': node_name,
                    'timestamp': time.time(),
                    'type': 'entry'
                })

                start_time = time.time()

                try:
                    # Execute the node
                    result = await func(state)

                    # Calculate timing
                    duration = time.time() - start_time

                    # Store timing
                    if node_name not in self.node_timings:
                        self.node_timings[node_name] = []
                    self.node_timings[node_name].append(duration)

                    # Log node exit
                    logger.info(
                        f"[LANGGRAPH] ← Exiting node: {node_name} "
                        f"(duration: {duration:.2f}s)"
                    )

                    # Track execution order
                    self.execution_order.append({
                        'node': node_name,
                        'timestamp': time.time(),
                        'type': 'exit',
                        'duration': duration
                    })

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"[LANGGRAPH] ✗ Error in node: {node_name} "
                        f"(duration: {duration:.2f}s) - {str(e)}"
                    )
                    raise

            @wraps(func)
            def sync_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
                if not self.enabled:
                    return func(state)

                # Log node entry
                logger.info(f"[LANGGRAPH] → Entering node: {node_name}")

                # Track execution order
                self.execution_order.append({
                    'node': node_name,
                    'timestamp': time.time(),
                    'type': 'entry'
                })

                start_time = time.time()

                try:
                    # Execute the node
                    result = func(state)

                    # Calculate timing
                    duration = time.time() - start_time

                    # Store timing
                    if node_name not in self.node_timings:
                        self.node_timings[node_name] = []
                    self.node_timings[node_name].append(duration)

                    # Log node exit
                    logger.info(
                        f"[LANGGRAPH] ← Exiting node: {node_name} "
                        f"(duration: {duration:.2f}s)"
                    )

                    # Track execution order
                    self.execution_order.append({
                        'node': node_name,
                        'timestamp': time.time(),
                        'type': 'exit',
                        'duration': duration
                    })

                    return result

                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"[LANGGRAPH] ✗ Error in node: {node_name} "
                        f"(duration: {duration:.2f}s) - {str(e)}"
                    )
                    raise

            # Return async or sync wrapper based on function type
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics for all nodes.

        Returns:
            Dictionary mapping node names to timing stats
        """
        stats = {}
        for node_name, timings in self.node_timings.items():
            stats[node_name] = {
                'count': len(timings),
                'total': sum(timings),
                'avg': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings)
            }
        return stats

    def get_execution_path(self) -> list:
        """Get the execution path through the graph.

        Returns:
            List of node execution events in order
        """
        return self.execution_order

    def export_execution_trace(self, filepath: str):
        """Export execution trace to JSON file.

        Args:
            filepath: Path to write JSON file
        """
        trace = {
            'execution_order': self.execution_order,
            'timing_stats': self.get_timing_stats(),
            'total_nodes': len(self.node_timings),
            'total_executions': sum(len(t) for t in self.node_timings.values())
        }

        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)

        logger.info(f"[LANGGRAPH] Exported execution trace to {filepath}")

    def print_summary(self):
        """Print a summary of graph execution."""
        logger.info("[LANGGRAPH] ═══ Execution Summary ═══")

        stats = self.get_timing_stats()
        for node_name, node_stats in sorted(stats.items()):
            logger.info(
                f"[LANGGRAPH]   {node_name}: "
                f"{node_stats['count']} executions, "
                f"avg={node_stats['avg']:.2f}s, "
                f"total={node_stats['total']:.2f}s"
            )

        logger.info("[LANGGRAPH] ═══════════════════════")

    def reset(self):
        """Reset all tracking data."""
        self.node_timings = {}
        self.execution_order = []
