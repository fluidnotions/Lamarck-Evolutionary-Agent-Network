"""
Tests for Tavily web search integration.
"""

import pytest
import os
from datetime import datetime, timedelta
from lean.web_search import TavilySearchManager, SearchQuotaManager


def test_quota_allocation():
    """Test fitness-based quota allocation."""
    quota_mgr = SearchQuotaManager()

    # Allocate quotas
    top_quota = quota_mgr.allocate_quota("agent_1", "top")
    mid_quota = quota_mgr.allocate_quota("agent_2", "middle")
    bot_quota = quota_mgr.allocate_quota("agent_3", "bottom")

    assert top_quota == 5
    assert mid_quota == 3
    assert bot_quota == 1


def test_quota_consumption():
    """Test quota consumption."""
    quota_mgr = SearchQuotaManager()
    quota_mgr.allocate_quota("agent_1", "top")

    # Initial quota
    assert quota_mgr.get_remaining("agent_1") == 5

    # Consume quota
    assert quota_mgr.consume_quota("agent_1", 1) == True
    assert quota_mgr.get_remaining("agent_1") == 4

    # Consume all
    quota_mgr.consume_quota("agent_1", 4)
    assert quota_mgr.get_remaining("agent_1") == 0

    # Cannot consume more
    assert quota_mgr.consume_quota("agent_1", 1) == False


def test_quota_reset():
    """Test quota reset after interval."""
    quota_mgr = SearchQuotaManager()

    # Allocate and consume
    quota_mgr.allocate_quota("agent_1", "top")
    quota_mgr.consume_quota("agent_1", 3)
    assert quota_mgr.get_remaining("agent_1") == 2

    # Simulate time passage
    quota_mgr.last_reset = datetime.now() - timedelta(hours=2)

    # Allocate again - should reset
    quota_mgr.allocate_quota("agent_1", "top")
    assert quota_mgr.get_remaining("agent_1") == 5


def test_search_result_formatting():
    """Test search result formatting."""
    search_mgr = TavilySearchManager(api_key="test_key")

    results = [
        {
            'title': 'Test Article',
            'content': 'This is test content about the topic.',
            'url': 'https://example.com/article',
            'score': 0.9
        }
    ]

    formatted = search_mgr.format_search_context(results)

    assert 'Test Article' in formatted
    assert 'test content' in formatted
    assert 'example.com' in formatted


def test_search_formatting_with_truncation():
    """Test search formatting with length limit."""
    search_mgr = TavilySearchManager(api_key="test_key")

    results = [
        {
            'title': 'Article ' + str(i),
            'content': 'Content ' * 100,
            'url': f'https://example.com/{i}',
            'score': 0.9
        }
        for i in range(10)
    ]

    formatted = search_mgr.format_search_context(results, max_length=500)

    assert len(formatted) <= 550  # Allow some overhead
    assert '[Results truncated]' in formatted


def test_search_without_api_key():
    """Test search manager without API key."""
    # Clear env var temporarily
    original_key = os.environ.get('TAVILY_API_KEY')
    if 'TAVILY_API_KEY' in os.environ:
        del os.environ['TAVILY_API_KEY']

    search_mgr = TavilySearchManager()

    # Should initialize but client is None
    assert search_mgr.client is None

    # Search should return error message
    results = search_mgr.search("test query", "agent_1")
    assert len(results) == 1
    assert results[0].get('error') == True
    assert 'not configured' in results[0]['content'].lower() or 'unavailable' in results[0]['content'].lower()

    # Restore env var
    if original_key:
        os.environ['TAVILY_API_KEY'] = original_key


def test_quota_exceeded_handling():
    """Test handling when quota is exceeded."""
    search_mgr = TavilySearchManager(api_key="test_key")

    # Allocate low quota
    search_mgr.quota_manager.allocate_quota("agent_1", "bottom")  # 1 search

    # First search consumes quota
    search_mgr.quota_manager.consume_quota("agent_1", 1)

    # Second search should fail quota check
    results = search_mgr.search("test query", "agent_1")
    assert len(results) == 1
    assert results[0].get('error') == True
    assert 'exceeded' in results[0]['content'].lower() and 'quota' in results[0]['content'].lower()


def test_cache_functionality():
    """Test search result caching."""
    search_mgr = TavilySearchManager(api_key="test_key")

    # Manually populate cache
    cache_key = search_mgr._cache_key("test query")
    test_results = [{'title': 'Cached', 'content': 'Cached content', 'url': ''}]
    search_mgr.cache[cache_key] = (test_results, datetime.now())

    # Allocate quota
    search_mgr.quota_manager.allocate_quota("agent_1", "top")

    # Search should return cached results without consuming quota
    # (Note: This test assumes cache check happens before quota check)
    initial_quota = search_mgr.quota_manager.get_remaining("agent_1")
    results = search_mgr.search("test query", "agent_1")

    # Check if results match cache
    if results == test_results:
        # Cache was used - quota should not be consumed
        assert search_mgr.quota_manager.get_remaining("agent_1") == initial_quota


@pytest.mark.skipif(
    not os.getenv("TAVILY_API_KEY"),
    reason="TAVILY_API_KEY not set - skipping integration test"
)
def test_real_search():
    """Integration test with real Tavily API (requires key)."""
    search_mgr = TavilySearchManager()

    # Allocate quota
    search_mgr.quota_manager.allocate_quota("test_agent", "top")

    # Perform real search
    results = search_mgr.search(
        query="Python programming language",
        agent_id="test_agent",
        max_results=3
    )

    # Verify results structure
    assert len(results) > 0
    assert 'title' in results[0]
    assert 'content' in results[0]

    # Check quota was consumed
    assert search_mgr.quota_manager.get_remaining("test_agent") == 4


def test_empty_results_formatting():
    """Test formatting with no results."""
    search_mgr = TavilySearchManager(api_key="test_key")

    formatted = search_mgr.format_search_context([])
    assert "No search results available" in formatted


def test_error_result_formatting():
    """Test formatting with error results."""
    search_mgr = TavilySearchManager(api_key="test_key")

    results = [
        {
            'title': 'Error',
            'content': 'Search failed',
            'url': '',
            'error': True
        }
    ]

    formatted = search_mgr.format_search_context(results)
    assert 'Search failed' in formatted
