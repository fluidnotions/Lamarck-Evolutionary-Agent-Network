"""
Web search integration using Tavily API.

Provides web search capability to agents with fitness-based quota management.
Higher-performing agents get more search quota.
"""

from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import hashlib
import json
from datetime import datetime, timedelta

load_dotenv()

# Conditional import - tavily may not be installed yet
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    TavilyClient = None


class SearchQuotaManager:
    """Manages search quota allocation based on agent fitness."""

    def __init__(self):
        """Initialize quota manager."""
        self.quotas = {}  # agent_id → remaining quota
        self.last_reset = datetime.now()
        self.reset_interval = timedelta(hours=1)  # Reset quotas hourly

    def allocate_quota(self, agent_id: str, fitness_rank: str) -> int:
        """Allocate search quota based on fitness rank.

        Args:
            agent_id: Unique agent identifier
            fitness_rank: 'top', 'middle', 'bottom'

        Returns:
            Number of searches allocated
        """
        # Check if need to reset quotas
        if datetime.now() - self.last_reset > self.reset_interval:
            self.quotas = {}
            self.last_reset = datetime.now()

        # Allocate based on rank
        quota_map = {
            'top': 5,
            'middle': 3,
            'bottom': 1
        }

        quota = quota_map.get(fitness_rank, 1)
        self.quotas[agent_id] = quota
        return quota

    def consume_quota(self, agent_id: str, amount: int = 1) -> bool:
        """Attempt to consume search quota.

        Args:
            agent_id: Agent requesting search
            amount: Number of searches to consume

        Returns:
            True if quota available and consumed, False otherwise
        """
        current = self.quotas.get(agent_id, 0)
        if current >= amount:
            self.quotas[agent_id] = current - amount
            return True
        return False

    def get_remaining(self, agent_id: str) -> int:
        """Get remaining quota for agent."""
        return self.quotas.get(agent_id, 0)


class TavilySearchManager:
    """Manages web searches with Tavily API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily client.

        Args:
            api_key: Tavily API key (defaults to env var)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

        if not TAVILY_AVAILABLE:
            print("[Warning] Tavily package not installed. Web search disabled.")
            self.client = None
        elif not self.api_key:
            print("[Warning] TAVILY_API_KEY not found. Web search disabled.")
            self.client = None
        else:
            self.client = TavilyClient(api_key=self.api_key)

        self.quota_manager = SearchQuotaManager()

        # Simple cache to avoid duplicate searches
        self.cache = {}
        self.cache_ttl = timedelta(hours=1)

    def search(
        self,
        query: str,
        agent_id: str,
        max_results: int = 3,
        search_depth: str = "basic"
    ) -> List[Dict]:
        """Perform web search with quota checking.

        Args:
            query: Search query
            agent_id: Agent performing search
            max_results: Maximum results to return
            search_depth: 'basic' or 'advanced'

        Returns:
            List of search results with title, content, url
        """
        # Check if Tavily is available
        if not self.client:
            return [{
                'title': 'Search Unavailable',
                'content': 'Tavily API not configured. Set TAVILY_API_KEY to enable web search.',
                'url': '',
                'error': True
            }]

        # Check cache first
        cache_key = self._cache_key(query)
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_result

        # Check quota
        if not self.quota_manager.consume_quota(agent_id, amount=1):
            return [{
                'title': 'Quota Exceeded',
                'content': f'Agent {agent_id} has exceeded search quota. Results may be limited.',
                'url': '',
                'error': True
            }]

        try:
            # Perform search
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True,
                include_raw_content=False
            )

            # Format results
            results = []
            for result in response.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0.0)
                })

            # Add AI-generated answer if available
            if response.get('answer'):
                results.insert(0, {
                    'title': 'AI Summary',
                    'content': response['answer'],
                    'url': '',
                    'is_summary': True
                })

            # Cache results
            self.cache[cache_key] = (results, datetime.now())

            return results

        except Exception as e:
            # Fallback on error
            return [{
                'title': 'Search Error',
                'content': f'Search failed: {str(e)}',
                'url': '',
                'error': True
            }]

    def _cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        return hashlib.md5(query.lower().encode()).hexdigest()

    def format_search_context(self, results: List[Dict], max_length: int = 1000) -> str:
        """Format search results into context string for LLM.

        Args:
            results: Search results
            max_length: Maximum character length

        Returns:
            Formatted context string
        """
        if not results:
            return "No search results available."

        context_parts = ["WEB SEARCH RESULTS:\n"]

        for i, result in enumerate(results[:5], 1):
            if result.get('error'):
                context_parts.append(f"{i}. {result['content']}\n")
                continue

            title = result.get('title', 'Untitled')
            content = result.get('content', '')[:200]  # Truncate content
            url = result.get('url', '')

            context_parts.append(f"{i}. {title}\n")
            context_parts.append(f"   {content}...\n")
            if url:
                context_parts.append(f"   Source: {url}\n")
            context_parts.append("\n")

        full_context = "".join(context_parts)

        # Truncate if too long
        if len(full_context) > max_length:
            return full_context[:max_length] + "...\n[Results truncated]"

        return full_context
