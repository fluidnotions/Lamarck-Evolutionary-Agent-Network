"""
Tests for compaction strategies.

Tests the forgetting component of Step 8 (EVOLVE):
- ScoreBasedCompaction
- FrequencyBasedCompaction
- DiversityPreservingCompaction
- HybridCompaction
"""

import pytest
import numpy as np
from lean.compaction import (
    CompactionStrategy,
    ScoreBasedCompaction,
    FrequencyBasedCompaction,
    DiversityPreservingCompaction,
    HybridCompaction,
    create_compaction_strategy
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_patterns_no_embeddings():
    """Sample reasoning patterns without embeddings."""
    return [
        {
            'reasoning': 'Pattern 1: High score, high freq',
            'score': 9.0,
            'retrieval_count': 15,
            'situation': 'test',
            'timestamp': 1000.0
        },
        {
            'reasoning': 'Pattern 2: High score, low freq',
            'score': 8.5,
            'retrieval_count': 2,
            'situation': 'test',
            'timestamp': 1001.0
        },
        {
            'reasoning': 'Pattern 3: Med score, high freq',
            'score': 7.0,
            'retrieval_count': 20,
            'situation': 'test',
            'timestamp': 1002.0
        },
        {
            'reasoning': 'Pattern 4: Low score, low freq',
            'score': 5.0,
            'retrieval_count': 1,
            'situation': 'test',
            'timestamp': 1003.0
        },
        {
            'reasoning': 'Pattern 5: Med score, med freq',
            'score': 7.5,
            'retrieval_count': 10,
            'situation': 'test',
            'timestamp': 1004.0
        }
    ]


@pytest.fixture
def sample_patterns_with_embeddings():
    """Sample reasoning patterns with embeddings for diversity testing."""
    # Create diverse embeddings (3 clusters)
    patterns = []

    # Cluster 1: High-performing intro strategies
    for i in range(10):
        patterns.append({
            'reasoning': f'Intro strategy {i}',
            'score': 8.0 + np.random.rand(),
            'retrieval_count': 10 + int(np.random.rand() * 10),
            'embedding': [1.0 + np.random.rand() * 0.1, 0.1, 0.1],  # Near [1,0,0]
            'situation': 'intro',
            'timestamp': 1000.0 + i
        })

    # Cluster 2: High-performing body strategies
    for i in range(10):
        patterns.append({
            'reasoning': f'Body strategy {i}',
            'score': 7.0 + np.random.rand(),
            'retrieval_count': 5 + int(np.random.rand() * 10),
            'embedding': [0.1, 1.0 + np.random.rand() * 0.1, 0.1],  # Near [0,1,0]
            'situation': 'body',
            'timestamp': 2000.0 + i
        })

    # Cluster 3: Low-performing misc strategies
    for i in range(10):
        patterns.append({
            'reasoning': f'Misc strategy {i}',
            'score': 5.0 + np.random.rand(),
            'retrieval_count': 1 + int(np.random.rand() * 5),
            'embedding': [0.1, 0.1, 1.0 + np.random.rand() * 0.1],  # Near [0,0,1]
            'situation': 'misc',
            'timestamp': 3000.0 + i
        })

    return patterns


# ============================================================================
# Test ScoreBasedCompaction
# ============================================================================

def test_score_based_compaction_basic(sample_patterns_no_embeddings):
    """Test basic score-based compaction."""
    strategy = ScoreBasedCompaction()

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    assert len(compacted) == 3
    # Should keep highest scores: 9.0, 8.5, 7.5
    scores = [p['score'] for p in compacted]
    assert scores == [9.0, 8.5, 7.5]


def test_score_based_compaction_no_compaction_needed(sample_patterns_no_embeddings):
    """Test when patterns already under limit."""
    strategy = ScoreBasedCompaction()

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=10)

    assert len(compacted) == 5  # All patterns kept
    assert compacted == sample_patterns_no_embeddings


def test_score_based_compaction_stats(sample_patterns_no_embeddings):
    """Test statistics tracking."""
    strategy = ScoreBasedCompaction()

    strategy.compact(sample_patterns_no_embeddings, max_size=3)
    stats = strategy.get_stats()

    assert stats['strategy'] == 'ScoreBasedCompaction'
    assert stats['total_compactions'] == 1
    assert stats['total_patterns_before'] == 5
    assert stats['total_patterns_after'] == 3
    assert stats['avg_compaction_rate'] == 3 / 5


# ============================================================================
# Test FrequencyBasedCompaction
# ============================================================================

def test_frequency_based_compaction_pure_frequency(sample_patterns_no_embeddings):
    """Test frequency-based with score_weight=0 (pure frequency)."""
    strategy = FrequencyBasedCompaction(score_weight=0.0)

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    assert len(compacted) == 3
    # Should keep highest frequency: 20, 15, 10
    freqs = [p['retrieval_count'] for p in compacted]
    assert freqs == [20, 15, 10]


def test_frequency_based_compaction_balanced(sample_patterns_no_embeddings):
    """Test frequency-based with balanced weighting."""
    strategy = FrequencyBasedCompaction(score_weight=0.5)

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    assert len(compacted) == 3
    # Exact order depends on formula, but should be balanced
    # Pattern 1 (score=9.0, freq=15) should be in top 3
    # Pattern 3 (score=7.0, freq=20) should be in top 3
    reasoning_texts = [p['reasoning'] for p in compacted]
    assert any('Pattern 1' in text for text in reasoning_texts)
    assert any('Pattern 3' in text for text in reasoning_texts)


def test_frequency_based_compaction_pure_score(sample_patterns_no_embeddings):
    """Test frequency-based with score_weight=1 (effectively score-based)."""
    strategy = FrequencyBasedCompaction(score_weight=1.0)

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    assert len(compacted) == 3
    # Should behave like score-based
    scores = [p['score'] for p in compacted]
    assert scores[0] >= scores[1] >= scores[2]


# ============================================================================
# Test DiversityPreservingCompaction
# ============================================================================

def test_diversity_preserving_compaction_with_embeddings(sample_patterns_with_embeddings):
    """Test diversity-preserving compaction with embeddings."""
    strategy = DiversityPreservingCompaction(min_clusters=3)

    compacted = strategy.compact(sample_patterns_with_embeddings, max_size=10)

    assert len(compacted) == 10

    # Check that compacted patterns come from multiple clusters
    situations = [p['situation'] for p in compacted]
    unique_situations = set(situations)

    # Should have patterns from multiple clusters (intro, body, misc)
    assert len(unique_situations) >= 2  # At least 2 different types


def test_diversity_preserving_compaction_fallback_no_embeddings(sample_patterns_no_embeddings):
    """Test fallback to score-based when no embeddings."""
    strategy = DiversityPreservingCompaction()

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    assert len(compacted) == 3
    # Should fallback to score-based
    scores = [p['score'] for p in compacted]
    assert scores == [9.0, 8.5, 7.5]


def test_diversity_preserving_compaction_quality_within_clusters(sample_patterns_with_embeddings):
    """Test that best patterns are selected within each cluster."""
    strategy = DiversityPreservingCompaction(min_clusters=3)

    compacted = strategy.compact(sample_patterns_with_embeddings, max_size=6)

    # For intro cluster (first 10 patterns), should select highest scores
    intro_compacted = [p for p in compacted if p['situation'] == 'intro']
    if intro_compacted:
        intro_scores = [p['score'] for p in intro_compacted]
        # Scores should be in descending order (best from cluster)
        assert all(intro_scores[i] >= intro_scores[i+1]
                  for i in range(len(intro_scores) - 1))


# ============================================================================
# Test HybridCompaction
# ============================================================================

def test_hybrid_compaction_with_embeddings(sample_patterns_with_embeddings):
    """Test hybrid compaction with all components."""
    strategy = HybridCompaction(
        score_weight=0.4,
        frequency_weight=0.3,
        diversity_weight=0.3,
        min_clusters=3
    )

    compacted = strategy.compact(sample_patterns_with_embeddings, max_size=10)

    assert len(compacted) == 10

    # Should have diversity
    situations = [p['situation'] for p in compacted]
    unique_situations = set(situations)
    assert len(unique_situations) >= 2

    # Should prefer higher scores
    avg_score = np.mean([p['score'] for p in compacted])
    all_avg_score = np.mean([p['score'] for p in sample_patterns_with_embeddings])
    assert avg_score > all_avg_score  # Compacted should have higher avg score


def test_hybrid_compaction_without_embeddings(sample_patterns_no_embeddings):
    """Test hybrid compaction fallback without embeddings."""
    strategy = HybridCompaction(
        score_weight=0.5,
        frequency_weight=0.5,
        diversity_weight=0.0  # No diversity component possible
    )

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    assert len(compacted) == 3

    # Should balance score and frequency
    # Pattern 3 (score=7.0, freq=20) or Pattern 1 (score=9.0, freq=15) should be top
    # Depends on normalization, but both should be in top 3
    reasoning_texts = [p['reasoning'] for p in compacted]
    assert any('Pattern 1' in text for text in reasoning_texts)
    assert any('Pattern 3' in text for text in reasoning_texts)


def test_hybrid_compaction_weight_normalization():
    """Test that weights are normalized automatically."""
    strategy = HybridCompaction(
        score_weight=2.0,
        frequency_weight=1.0,
        diversity_weight=1.0
    )

    # Weights should sum to 1.0 after normalization
    total = strategy.score_weight + strategy.frequency_weight + strategy.diversity_weight
    assert abs(total - 1.0) < 0.001


def test_hybrid_compaction_extreme_weights(sample_patterns_no_embeddings):
    """Test hybrid with extreme weights (all score)."""
    strategy = HybridCompaction(
        score_weight=1.0,
        frequency_weight=0.0,
        diversity_weight=0.0
    )

    compacted = strategy.compact(sample_patterns_no_embeddings, max_size=3)

    # Should behave like score-based
    scores = [p['score'] for p in compacted]
    assert scores == [9.0, 8.5, 7.5]


# ============================================================================
# Test Factory Function
# ============================================================================

def test_create_compaction_strategy_score():
    """Test factory creates ScoreBasedCompaction."""
    strategy = create_compaction_strategy('score')

    assert isinstance(strategy, ScoreBasedCompaction)


def test_create_compaction_strategy_frequency():
    """Test factory creates FrequencyBasedCompaction."""
    strategy = create_compaction_strategy('frequency', score_weight=0.7)

    assert isinstance(strategy, FrequencyBasedCompaction)
    assert strategy.score_weight == 0.7


def test_create_compaction_strategy_diversity():
    """Test factory creates DiversityPreservingCompaction."""
    strategy = create_compaction_strategy('diversity', min_clusters=10)

    assert isinstance(strategy, DiversityPreservingCompaction)
    assert strategy.min_clusters == 10


def test_create_compaction_strategy_hybrid():
    """Test factory creates HybridCompaction."""
    strategy = create_compaction_strategy('hybrid')

    assert isinstance(strategy, HybridCompaction)


def test_create_compaction_strategy_invalid():
    """Test factory raises error for invalid strategy."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_compaction_strategy('invalid')


# ============================================================================
# Integration Tests
# ============================================================================

def test_compaction_reduces_100_to_30(sample_patterns_with_embeddings):
    """Test realistic scenario: 100 patterns → 30."""
    # Extend to ~100 patterns
    patterns = sample_patterns_with_embeddings * 3  # 30 * 3 = 90
    patterns.extend(sample_patterns_with_embeddings[:10])  # Total ~100

    strategy = HybridCompaction()
    compacted = strategy.compact(patterns, max_size=30)

    assert len(compacted) == 30

    # Should keep higher quality patterns
    avg_score_compacted = np.mean([p['score'] for p in compacted])
    avg_score_all = np.mean([p['score'] for p in patterns])
    assert avg_score_compacted >= avg_score_all


def test_different_strategies_produce_different_results(sample_patterns_with_embeddings):
    """Test that different strategies produce different results."""
    strategies = [
        ScoreBasedCompaction(),
        FrequencyBasedCompaction(score_weight=0.0),
        DiversityPreservingCompaction(),
        HybridCompaction()
    ]

    results = []
    for strategy in strategies:
        compacted = strategy.compact(sample_patterns_with_embeddings, max_size=10)
        # Get reasoning texts as identifier
        texts = tuple(p['reasoning'] for p in compacted)
        results.append(texts)

    # At least some strategies should produce different results
    unique_results = len(set(results))
    assert unique_results >= 2  # At least 2 different outcomes


def test_compaction_preserves_pattern_structure():
    """Test that compacted patterns maintain required fields."""
    patterns = [
        {
            'reasoning': f'Pattern {i}',
            'score': 5.0 + i,
            'retrieval_count': i,
            'situation': 'test',
            'timestamp': 1000.0 + i
        }
        for i in range(20)
    ]

    strategy = ScoreBasedCompaction()
    compacted = strategy.compact(patterns, max_size=5)

    for pattern in compacted:
        assert 'reasoning' in pattern
        assert 'score' in pattern
        assert 'retrieval_count' in pattern
        assert 'situation' in pattern
        assert 'timestamp' in pattern


def test_compaction_performance_large_dataset():
    """Test compaction performance with large dataset."""
    import time

    # Create 1000 patterns
    patterns = [
        {
            'reasoning': f'Pattern {i}',
            'score': 5.0 + (i % 6),  # Scores 5.0-10.0
            'retrieval_count': i % 25,
            'embedding': np.random.rand(384).tolist(),  # 384-dim embedding
            'situation': 'test',
            'timestamp': 1000.0 + i
        }
        for i in range(1000)
    ]

    strategy = HybridCompaction()

    start = time.time()
    compacted = strategy.compact(patterns, max_size=30)
    elapsed = time.time() - start

    assert len(compacted) == 30
    assert elapsed < 5.0  # Should complete in < 5 seconds


# ============================================================================
# Edge Cases
# ============================================================================

def test_compaction_empty_list():
    """Test compaction with empty pattern list."""
    strategy = ScoreBasedCompaction()

    compacted = strategy.compact([], max_size=10)

    assert compacted == []


def test_compaction_single_pattern():
    """Test compaction with single pattern."""
    patterns = [{
        'reasoning': 'Only pattern',
        'score': 8.0,
        'retrieval_count': 5,
        'situation': 'test',
        'timestamp': 1000.0
    }]

    strategy = ScoreBasedCompaction()
    compacted = strategy.compact(patterns, max_size=10)

    assert len(compacted) == 1
    assert compacted[0] == patterns[0]


def test_compaction_max_size_zero():
    """Test compaction with max_size=0."""
    patterns = [{
        'reasoning': 'Pattern',
        'score': 8.0,
        'retrieval_count': 5,
        'situation': 'test',
        'timestamp': 1000.0
    }]

    strategy = ScoreBasedCompaction()
    compacted = strategy.compact(patterns, max_size=0)

    assert len(compacted) == 0


def test_compaction_missing_fields():
    """Test compaction handles missing optional fields gracefully."""
    patterns = [
        {
            'reasoning': 'Pattern 1',
            # Missing score, retrieval_count
        },
        {
            'reasoning': 'Pattern 2',
            'score': 8.0,
            # Missing retrieval_count
        }
    ]

    # Should not crash - use defaults
    strategy = ScoreBasedCompaction()
    compacted = strategy.compact(patterns, max_size=1)

    assert len(compacted) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
