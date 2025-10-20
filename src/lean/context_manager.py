"""
Context distribution manager for reasoning traces (M1.4).

**CRITICAL**: Distributes REASONING TRACES (cognitive strategies), NOT content/outputs.

Implements 40/30/20/10 weighted distribution:
- 40%: Hierarchy/parent context
- 30%: High-credibility cross-role agents (reasoning traces)
- 20%: Random low-performer (forced diversity)
- 10%: Same-role peer
"""

from typing import List, Dict, Optional
import random
import time


class ContextManager:
    """Manages weighted reasoning trace distribution across agent populations."""

    def __init__(
        self,
        hierarchy_weight: float = 0.40,
        high_credibility_weight: float = 0.30,
        diversity_weight: float = 0.20,
        peer_weight: float = 0.10
    ):
        """Initialize context manager.

        Args:
            hierarchy_weight: Weight for parent/coordinator context
            high_credibility_weight: Weight for top performers' reasoning
            diversity_weight: Weight for random low-performer reasoning
            peer_weight: Weight for same-role peer reasoning
        """
        self.hierarchy_weight = hierarchy_weight
        self.high_credibility_weight = high_credibility_weight
        self.diversity_weight = diversity_weight
        self.peer_weight = peer_weight

        # Validate weights sum to 1.0
        total = sum([
            hierarchy_weight,
            high_credibility_weight,
            diversity_weight,
            peer_weight
        ])
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        # Track context flow for analysis
        self.context_flow_history = []

    def assemble_context(
        self,
        current_agent,
        hierarchy_context: str,
        all_pools: Dict,
        workflow_state: Dict
    ) -> Dict:
        """Assemble weighted reasoning context from multiple sources.

        **CRITICAL**: Assembles REASONING TRACES, not outputs.

        Args:
            current_agent: Agent receiving context (BaseAgentV2)
            hierarchy_context: Parent/coordinator context (task description, intent)
            all_pools: Dict mapping role → AgentPool
            workflow_state: Current workflow state with reasoning traces

        Returns:
            Dict with assembled context and metadata
        """
        context_pieces = []

        # 40% Hierarchy context
        context_pieces.append({
            'source': 'hierarchy',
            'weight': self.hierarchy_weight,
            'content': hierarchy_context,
            'metadata': {'type': 'task_description'}
        })

        # 30% High-credibility cross-role agents (REASONING TRACES)
        high_cred_context = self._get_high_credibility_context(
            current_agent, all_pools, workflow_state
        )
        if high_cred_context:
            context_pieces.append({
                'source': 'high_credibility',
                'weight': self.high_credibility_weight,
                'content': high_cred_context['content'],
                'metadata': high_cred_context['metadata']
            })

        # 20% Random low-performer (forced diversity in reasoning approaches)
        diversity_context = self._get_diversity_context(
            current_agent, all_pools, workflow_state
        )
        if diversity_context:
            context_pieces.append({
                'source': 'diversity',
                'weight': self.diversity_weight,
                'content': diversity_context['content'],
                'metadata': diversity_context['metadata']
            })

        # 10% Same-role peer (colleague reasoning patterns)
        peer_context = self._get_peer_context(
            current_agent, all_pools, workflow_state
        )
        if peer_context:
            context_pieces.append({
                'source': 'peer',
                'weight': self.peer_weight,
                'content': peer_context['content'],
                'metadata': peer_context['metadata']
            })

        # Combine weighted context
        combined_content = self._combine_context_pieces(context_pieces)

        # Track context flow
        self.context_flow_history.append({
            'receiver': current_agent.agent_id,
            'sources': [p['source'] for p in context_pieces],
            'timestamp': time.time(),
            'context_pieces': context_pieces
        })

        return {
            'content': combined_content,
            'context_pieces': context_pieces,
            'sources': [p['source'] for p in context_pieces],
            'metadata': {
                'agent_id': current_agent.agent_id,
                'role': current_agent.role,
                'num_sources': len(context_pieces)
            }
        }

    def _get_high_credibility_context(
        self,
        current_agent,
        all_pools: Dict,
        workflow_state: Dict
    ) -> Optional[Dict]:
        """Get REASONING TRACE from top performers in OTHER roles.

        **CRITICAL**: Returns reasoning patterns (<think> content), NOT outputs.

        Args:
            current_agent: Agent receiving context
            all_pools: All agent pools
            workflow_state: Workflow state

        Returns:
            Dict with reasoning trace and metadata, or None
        """
        # Get cross-role pools (not current agent's role)
        other_roles = [role for role in all_pools.keys() if role != current_agent.role]

        if not other_roles:
            return None

        # Get top performers from other roles
        top_performers = []
        for role in other_roles:
            pool = all_pools[role]
            top_2 = pool.get_top_n(n=2)
            top_performers.extend(top_2)

        if not top_performers:
            return None

        # Select one randomly (to avoid always using the same agent)
        selected = random.choice(top_performers)

        # Get their most recent REASONING TRACE (from <think> tags, NOT output)
        reasoning_trace = self._get_agent_recent_reasoning(selected, workflow_state)

        return {
            'content': reasoning_trace,  # This is reasoning, not output
            'metadata': {
                'agent_id': selected.agent_id,
                'role': selected.role,
                'fitness': selected.avg_fitness(),
                'type': 'reasoning_trace'  # Mark as reasoning
            }
        }

    def _get_diversity_context(
        self,
        current_agent,
        all_pools: Dict,
        workflow_state: Dict
    ) -> Optional[Dict]:
        """Get REASONING TRACE from random low-performer (forced diversity).

        **CRITICAL**: Returns alternative cognitive approaches.

        Args:
            current_agent: Agent receiving context
            all_pools: All agent pools
            workflow_state: Workflow state

        Returns:
            Dict with reasoning trace and metadata, or None
        """
        # Pick random role (could be same or different)
        roles = list(all_pools.keys())
        selected_role = random.choice(roles)
        pool = all_pools[selected_role]

        # Get random agent from lower half by fitness
        low_performer = pool.get_random_lower_half()

        # Get their recent REASONING TRACE (alternative cognitive approach)
        reasoning_trace = self._get_agent_recent_reasoning(low_performer, workflow_state)

        return {
            'content': reasoning_trace,  # Alternative reasoning strategy
            'metadata': {
                'agent_id': low_performer.agent_id,
                'role': low_performer.role,
                'fitness': low_performer.avg_fitness(),
                'type': 'diversity_reasoning'  # Different cognitive approach
            }
        }

    def _get_peer_context(
        self,
        current_agent,
        all_pools: Dict,
        workflow_state: Dict
    ) -> Optional[Dict]:
        """Get REASONING TRACE from same-role peer.

        Args:
            current_agent: Agent receiving context
            all_pools: All agent pools
            workflow_state: Workflow state

        Returns:
            Dict with reasoning trace and metadata, or None
        """
        # Get pool for current agent's role
        pool = all_pools.get(current_agent.role)
        if not pool or pool.size() < 2:
            return None

        # Get different agent from same role
        peers = [a for a in pool.agents if a.agent_id != current_agent.agent_id]
        if not peers:
            return None

        peer = random.choice(peers)

        # Get their recent REASONING TRACE (similar-role cognitive strategy)
        reasoning_trace = self._get_agent_recent_reasoning(peer, workflow_state)

        return {
            'content': reasoning_trace,  # Peer reasoning strategy
            'metadata': {
                'agent_id': peer.agent_id,
                'role': peer.role,
                'fitness': peer.avg_fitness(),
                'type': 'peer_reasoning'  # Same-role cognitive approach
            }
        }

    def _get_agent_recent_reasoning(
        self,
        agent,
        workflow_state: Dict
    ) -> str:
        """Get agent's most recent REASONING TRACE from workflow state or memory.

        **CRITICAL**: This retrieves reasoning patterns (<think> content), NOT outputs.

        Args:
            agent: Agent to get reasoning from (BaseAgentV2)
            workflow_state: Current workflow state

        Returns:
            Recent reasoning trace or empty string
        """
        # Try workflow state first (current generation)
        # Look for reasoning trace, NOT output
        reasoning_key = f"{agent.role}_reasoning"
        if reasoning_key in workflow_state:
            return workflow_state[reasoning_key]

        # Fall back to agent's reasoning memory (previous generation)
        try:
            reasoning_patterns = agent.reasoning_memory.get_all_reasoning(
                include_inherited=False
            )
            if reasoning_patterns:
                # Get most recent personal reasoning pattern
                recent = max(
                    reasoning_patterns,
                    key=lambda m: m['metadata'].get('timestamp', 0)
                )
                # Return the reasoning content (from <think> tags)
                return recent['reasoning']  # NOT 'content' - this is the <think> section
        except AttributeError:
            # Fallback if agent doesn't have reasoning_memory
            print(f"[Warning] Agent {agent.agent_id} missing reasoning_memory")
            return ""

        return ""

    def _combine_context_pieces(self, context_pieces: List[Dict]) -> str:
        """Combine weighted context pieces into single string.

        Args:
            context_pieces: List of context dicts with weight and content

        Returns:
            Combined context string
        """
        if not context_pieces:
            return ""

        # Build weighted context string
        sections = []

        for piece in context_pieces:
            source = piece['source']
            weight = piece['weight']
            content = piece['content']

            if not content:
                continue

            # Format section with source label
            section = f"[{source.upper()} ({weight*100:.0f}%)]:\n{content}\n"
            sections.append(section)

        return "\n".join(sections)

    def get_broadcast_stats(self, agent_id: str) -> Dict:
        """Get statistics on how widely an agent's reasoning broadcasts.

        Args:
            agent_id: Agent to analyze

        Returns:
            Dict with broadcast statistics
        """
        # Count how many times this agent was used as context source
        broadcasts = []
        for entry in self.context_flow_history:
            for piece in entry.get('context_pieces', []):
                if piece.get('metadata', {}).get('agent_id') == agent_id:
                    broadcasts.append(entry)
                    break

        unique_receivers = set(entry['receiver'] for entry in broadcasts)

        return {
            'agent_id': agent_id,
            'total_broadcasts': len(broadcasts),
            'unique_receivers': len(unique_receivers),
            'receiver_ids': list(unique_receivers)
        }

    def measure_diversity(self, recent_n: int = 10) -> Dict:
        """Measure reasoning context diversity.

        Args:
            recent_n: Number of recent context flows to analyze

        Returns:
            Dict with diversity metrics
        """
        recent_flows = self.context_flow_history[-recent_n:]

        if not recent_flows:
            return {'diversity_score': 0.0, 'unique_sources': 0}

        # Extract all source agent IDs
        all_sources = []
        for flow in recent_flows:
            for piece in flow.get('context_pieces', []):
                source_id = piece.get('metadata', {}).get('agent_id')
                if source_id:
                    all_sources.append(source_id)

        if not all_sources:
            return {'diversity_score': 0.0, 'unique_sources': 0}

        # Calculate diversity (unique sources / total sources)
        unique_sources = len(set(all_sources))
        diversity_score = unique_sources / len(all_sources)

        return {
            'diversity_score': diversity_score,
            'unique_sources': unique_sources,
            'total_sources': len(all_sources)
        }

    def export_context_flow(self, filepath: str = "context_flow.json"):
        """Export context flow history for analysis.

        Args:
            filepath: Output file path
        """
        import json

        # Serialize context flow (remove non-serializable items)
        serializable_history = []
        for entry in self.context_flow_history:
            serializable_entry = {
                'receiver': entry['receiver'],
                'sources': entry['sources'],
                'timestamp': entry['timestamp']
            }
            serializable_history.append(serializable_entry)

        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)

        print(f"Context flow data exported to {filepath}")
