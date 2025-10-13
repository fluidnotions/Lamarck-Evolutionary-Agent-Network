"""
Metrics monitoring for meta-agent system.

Collects and analyzes performance data to identify graph optimization opportunities.
"""

from typing import Dict, List, Optional
from collections import deque
import statistics


class MetricsMonitor:
    """Monitors agent performance metrics and identifies optimization opportunities."""

    def __init__(
        self,
        history_window: int = 10,
        low_score_threshold: float = 6.0,
        high_variance_threshold: float = 1.5,
    ):
        """Initialize metrics monitor.

        Args:
            history_window: Number of generations to track
            low_score_threshold: Threshold for identifying underperforming agents
            high_variance_threshold: Threshold for identifying unstable agents
        """
        self.history_window = history_window
        self.low_score_threshold = low_score_threshold
        self.high_variance_threshold = high_variance_threshold

        # Track metrics per agent
        self.score_history: Dict[str, deque] = {}
        self.timing_history: Dict[str, deque] = {}
        self.generation_count = 0

    def record_generation(
        self, scores: Dict[str, float], timings: Dict[str, Dict[str, float]]
    ):
        """Record metrics from a generation.

        Args:
            scores: Agent scores {agent: score}
            timings: Agent timing data {agent: {start, end, duration}}
        """
        self.generation_count += 1

        # Record scores
        for agent, score in scores.items():
            if agent not in self.score_history:
                self.score_history[agent] = deque(maxlen=self.history_window)
            self.score_history[agent].append(score)

        # Record timings
        for agent, timing in timings.items():
            if agent not in self.timing_history:
                self.timing_history[agent] = deque(maxlen=self.history_window)
            self.timing_history[agent].append(timing["duration"])

    def analyze_agent_performance(self, agent: str) -> Dict:
        """Analyze performance for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Analysis dictionary with metrics and recommendations
        """
        if agent not in self.score_history:
            return {
                "agent": agent,
                "status": "insufficient_data",
                "recommendations": [],
            }

        scores = list(self.score_history[agent])
        timings = list(self.timing_history.get(agent, []))

        if len(scores) < 3:
            return {
                "agent": agent,
                "status": "insufficient_data",
                "recommendations": [],
            }

        # Calculate statistics
        avg_score = statistics.mean(scores)
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0
        avg_timing = statistics.mean(timings) if timings else 0

        # Identify issues
        issues = []
        recommendations = []

        if avg_score < self.low_score_threshold:
            issues.append("low_performance")
            recommendations.append(
                f"Agent consistently scores below {self.low_score_threshold}"
            )

        if score_variance > self.high_variance_threshold:
            issues.append("high_variance")
            recommendations.append(
                f"Agent has high variance (σ²={score_variance:.2f}), indicating instability"
            )

        if avg_timing > 5.0:  # More than 5 seconds
            issues.append("slow_execution")
            recommendations.append(
                f"Agent takes {avg_timing:.2f}s on average, consider optimization"
            )

        return {
            "agent": agent,
            "avg_score": avg_score,
            "score_variance": score_variance,
            "avg_timing": avg_timing,
            "recent_scores": scores[-5:],  # Last 5 scores
            "issues": issues,
            "recommendations": recommendations,
            "status": "analyzed" if not issues else "needs_attention",
        }

    def identify_optimization_opportunities(self) -> List[Dict]:
        """Identify system-wide optimization opportunities.

        Returns:
            List of optimization recommendations
        """
        opportunities = []

        # Check each agent
        all_agents = set(self.score_history.keys())

        for agent in all_agents:
            analysis = self.analyze_agent_performance(agent)

            if analysis["status"] == "needs_attention":
                opportunities.append(
                    {
                        "type": "agent_performance",
                        "agent": agent,
                        "issues": analysis["issues"],
                        "recommendations": analysis["recommendations"],
                    }
                )

        # Check for parallelization opportunities
        # If multiple agents have similar timings and no dependencies, suggest parallel execution
        if len(all_agents) >= 2:
            agents_with_timing = [
                a for a in all_agents if a in self.timing_history
            ]

            if len(agents_with_timing) >= 2:
                # Simple heuristic: if agents exist and have timing data, parallelization may help
                opportunities.append(
                    {
                        "type": "parallelization",
                        "suggestion": "Consider executing independent agents in parallel",
                        "agents": list(agents_with_timing),
                    }
                )

        return opportunities

    def get_summary_statistics(self) -> Dict:
        """Get summary statistics across all agents.

        Returns:
            Summary dictionary
        """
        if not self.score_history:
            return {
                "generation_count": self.generation_count,
                "agents_tracked": 0,
            }

        all_scores = []
        for scores in self.score_history.values():
            all_scores.extend(scores)

        all_timings = []
        for timings in self.timing_history.values():
            all_timings.extend(timings)

        return {
            "generation_count": self.generation_count,
            "agents_tracked": len(self.score_history),
            "overall_avg_score": statistics.mean(all_scores) if all_scores else 0,
            "overall_score_variance": (
                statistics.variance(all_scores) if len(all_scores) > 1 else 0
            ),
            "overall_avg_timing": statistics.mean(all_timings) if all_timings else 0,
            "total_samples": len(all_scores),
        }

    def reset(self):
        """Reset all tracked metrics."""
        self.score_history.clear()
        self.timing_history.clear()
        self.generation_count = 0
