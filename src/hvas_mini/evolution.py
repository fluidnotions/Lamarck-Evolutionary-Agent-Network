"""
Parameter evolution utilities for HVAS Mini.
"""

from typing import Dict, List


def calculate_temperature_adjustment(
    score_history: List[float],
    current_temp: float,
    learning_rate: float = 0.1,
    min_temp: float = 0.5,
    max_temp: float = 1.0,
) -> float:
    """Calculate new temperature based on score history.

    Args:
        score_history: Recent scores
        current_temp: Current temperature value
        learning_rate: Learning rate for adjustments
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature

    Returns:
        New temperature value
    """
    if not score_history:
        return current_temp

    # Use last 5 scores
    recent = score_history[-5:]
    avg = sum(recent) / len(recent)

    # Calculate delta
    if avg < 6.0:
        delta = -learning_rate
    elif avg > 8.0:
        delta = learning_rate
    else:
        delta = (7.0 - avg) * learning_rate * 0.5

    # Apply bounds
    new_temp = current_temp + delta
    return max(min_temp, min(max_temp, new_temp))


def get_evolution_stats(parameters: Dict) -> Dict:
    """Get evolution statistics for an agent.

    Args:
        parameters: Agent parameters dictionary

    Returns:
        Statistics dictionary
    """
    history = parameters.get("score_history", [])

    if not history:
        return {
            "generations": 0,
            "avg_score": 0.0,
            "current_temp": parameters.get("temperature", 0.7),
        }

    return {
        "generations": parameters.get("generation_count", 0),
        "avg_score": sum(history) / len(history),
        "recent_avg": sum(history[-5:]) / len(history[-5:]),
        "best_score": max(history),
        "worst_score": min(history),
        "current_temp": parameters.get("temperature", 0.7),
    }
