"""
Weight update logic based on evaluation scores.
"""

from typing import Dict, List


def calculate_performance_signal(
    agent_score: float, peer_score: float, max_score: float = 10.0
) -> float:
    """Calculate performance signal for weight update.

    Signal represents how useful peer's context was for agent's performance.

    Args:
        agent_score: Score received by agent using peer's context
        peer_score: Score received by peer itself
        max_score: Maximum possible score

    Returns:
        Performance signal (0-1)
    """
    # Normalize scores
    agent_norm = agent_score / max_score
    peer_norm = peer_score / max_score

    # High agent score + high peer score = strong signal
    # (peer's quality helped agent succeed)
    signal = (agent_norm + peer_norm) / 2

    return max(0.0, min(1.0, signal))


def update_all_weights(
    trust_manager, state: Dict, scores: Dict[str, float]
) -> List[Dict]:
    """Update all agent weights based on current scores.

    Args:
        trust_manager: TrustManager instance
        state: Current BlogState
        scores: Current evaluation scores

    Returns:
        Weight update history entries
    """
    updates = []
    agents = list(scores.keys())

    for agent in agents:
        for peer in agents:
            if agent == peer:
                continue

            # Calculate signal
            signal = calculate_performance_signal(
                agent_score=scores[agent],
                peer_score=scores[peer],
            )

            # Update weight
            old_weight = trust_manager.get_weight(agent, peer)
            new_weight = trust_manager.update_weight(agent, peer, signal)

            # Record
            updates.append({
                "agent": agent,
                "peer": peer,
                "old_weight": old_weight,
                "new_weight": new_weight,
                "delta": new_weight - old_weight,
                "signal": signal,
            })

    return updates
