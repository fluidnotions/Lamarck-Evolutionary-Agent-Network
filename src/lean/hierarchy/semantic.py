"""
Semantic distance calculation for context filtering.

Uses cosine similarity between agent semantic vectors to determine
how much context should be shared between agents in the hierarchy.
"""

import numpy as np
from typing import List
from lean.hierarchy.structure import AgentHierarchy


def compute_semantic_distance(
    hierarchy: AgentHierarchy, agent_a: str, agent_b: str
) -> float:
    """Compute cosine distance between agent semantic vectors.

    Semantic distance indicates how different two agents' focuses are:
    - 0.0 = identical focus (share all context)
    - 1.0 = opposite focus (share minimal context)

    Args:
        hierarchy: AgentHierarchy instance
        agent_a: First agent role
        agent_b: Second agent role

    Returns:
        Distance in [0, 1] where 0=identical, 1=opposite

    Raises:
        ValueError: If agents not found in hierarchy
    """
    if agent_a not in hierarchy.nodes:
        raise ValueError(f"Agent '{agent_a}' not found in hierarchy")
    if agent_b not in hierarchy.nodes:
        raise ValueError(f"Agent '{agent_b}' not found in hierarchy")

    vec_a = np.array(hierarchy.nodes[agent_a].semantic_vector)
    vec_b = np.array(hierarchy.nodes[agent_b].semantic_vector)

    # Normalize vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.5  # Default distance for zero vectors

    # Cosine similarity: dot product / (norm_a * norm_b)
    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    # Clamp similarity to [-1, 1] (floating point safety)
    similarity = max(-1.0, min(1.0, similarity))

    # Convert to distance [0, 1]
    # similarity=1 → distance=0, similarity=-1 → distance=1
    distance = float((1.0 - similarity) / 2.0)

    return max(0.0, min(1.0, distance))


def compute_similarity_matrix(hierarchy: AgentHierarchy) -> dict:
    """Compute pairwise semantic distances for all agents.

    Useful for analysis and debugging.

    Args:
        hierarchy: AgentHierarchy instance

    Returns:
        Dictionary of {(agent_a, agent_b): distance}
    """
    matrix = {}
    agent_roles = list(hierarchy.nodes.keys())

    for i, agent_a in enumerate(agent_roles):
        for agent_b in agent_roles[i:]:  # Only compute upper triangle
            distance = compute_semantic_distance(hierarchy, agent_a, agent_b)
            matrix[(agent_a, agent_b)] = distance
            # Symmetric
            matrix[(agent_b, agent_a)] = distance

    return matrix


def filter_context_by_distance(
    context: str,
    distance: float,
    min_ratio: float = 0.3
) -> str:
    """Filter context based on semantic distance.

    Closer semantic distance (lower value) means more context shared.
    Further distance means less context shared.

    Args:
        context: Full context string
        distance: Semantic distance [0, 1]
        min_ratio: Minimum context to retain (default: 30%)

    Returns:
        Filtered context string

    Examples:
        >>> filter_context_by_distance("Sent1. Sent2. Sent3.", distance=0.0)
        "Sent1. Sent2. Sent3."  # Full context (distance=0)

        >>> filter_context_by_distance("Sent1. Sent2. Sent3.", distance=1.0, min_ratio=0.3)
        "Sent1."  # Only 30% (distance=1)
    """
    if not context:
        return ""

    # Calculate sharing strength (inverse of distance)
    # distance=0 → strength=1.0 (share 100%)
    # distance=1 → strength=min_ratio (share min_ratio%)
    strength = 1.0 - (distance * (1.0 - min_ratio))

    # Split into sentences
    sentences = [s.strip() for s in context.split(".") if s.strip()]

    if not sentences:
        return context

    # Calculate how many sentences to keep
    keep_count = max(1, int(len(sentences) * strength))

    # Keep top N sentences (prioritize beginning)
    filtered_sentences = sentences[:keep_count]

    return ". ".join(filtered_sentences) + "."


def compute_context_weights(
    hierarchy: AgentHierarchy,
    parent_role: str,
    child_roles: List[str]
) -> dict:
    """Compute semantic distance weights for children.

    Args:
        hierarchy: AgentHierarchy instance
        parent_role: Parent agent role
        child_roles: List of child agent roles

    Returns:
        Dictionary of {child_role: weight} where weight in [0, 1]
    """
    weights = {}

    for child in child_roles:
        distance = compute_semantic_distance(hierarchy, parent_role, child)
        # Convert distance to weight (inverse relationship)
        # distance=0 → weight=1.0, distance=1 → weight=0.0
        weight = float(1.0 - distance)
        weights[child] = weight

    return weights


def get_contextual_relevance(
    hierarchy: AgentHierarchy,
    source_role: str,
    target_role: str
) -> float:
    """Get relevance score for context from source to target.

    Higher score means more relevant context.

    Args:
        hierarchy: AgentHierarchy instance
        source_role: Agent providing context
        target_role: Agent receiving context

    Returns:
        Relevance score in [0, 1]
    """
    distance = compute_semantic_distance(hierarchy, source_role, target_role)

    # Relevance is inverse of distance
    relevance = float(1.0 - distance)

    return max(0.0, min(1.0, relevance))
