# Agent Task: Evaluation System

## Branch: `feature/evaluation-system`

## Priority: MEDIUM - Needed for learning

## Execution: PARALLEL with other features

## Objective
Implement multi-factor content evaluation system that scores each section (intro, body, conclusion) based on quality metrics.

## Dependencies
- ✅ feature/project-foundation
- ✅ feature/state-management

## Tasks

### 1. Create `src/hvas_mini/evaluation.py`

Implement according to spec (section 3.4):

```python
"""
Content evaluation system for HVAS Mini.

Scores generated content based on multiple quality factors.
"""

from typing import Dict
import re
import os
from dotenv import load_dotenv

from hvas_mini.state import BlogState

load_dotenv()


class ContentEvaluator:
    """Multi-factor content scoring system.

    Evaluates intro, body, and conclusion sections on a 0-10 scale
    based on length, structure, topic relevance, and engagement.
    """

    def __call__(self, state: BlogState) -> BlogState:
        """Score each section based on multiple factors.

        Args:
            state: Current workflow state with generated content

        Returns:
            State with scores populated
        """
        scores = {
            "intro": self._score_intro(state["intro"], state["topic"]),
            "body": self._score_body(state["body"], state["topic"]),
            "conclusion": self._score_conclusion(
                state["conclusion"],
                state["topic"],
                state["intro"]
            )
        }

        state["scores"] = scores

        # Log scores for visualization
        state["stream_logs"].append(
            f"[Evaluator] Scores - "
            f"Intro: {scores['intro']:.1f}, "
            f"Body: {scores['body']:.1f}, "
            f"Conclusion: {scores['conclusion']:.1f}"
        )

        return state

    def _score_intro(self, intro: str, topic: str) -> float:
        """Score introduction section.

        Factors:
        - Length (20-60 words optimal)
        - Topic relevance (mentions topic)
        - Engagement hooks (question, engaging words)
        - Question marks (engaging questions)

        Args:
            intro: Introduction text
            topic: Blog topic

        Returns:
            Score from 0-10
        """
        score = 5.0  # Base score

        # Length check
        word_count = len(intro.split())
        if 20 <= word_count <= 60:
            score += 1.5

        # Topic relevance
        if topic.lower() in intro.lower():
            score += 1.5

        # Engagement hooks
        hooks = [
            "discover", "learn", "imagine", "what if",
            "have you ever", "wondering", "explore"
        ]
        if any(hook in intro.lower() for hook in hooks):
            score += 1.0

        # Question mark (engaging questions)
        if "?" in intro:
            score += 1.0

        return min(10.0, score)

    def _score_body(self, body: str, topic: str) -> float:
        """Score body section.

        Factors:
        - Length (>150 words)
        - Paragraph structure (2-5 paragraphs)
        - Topic coverage (mentions topic words)
        - Specific examples or data (numbers, "example")

        Args:
            body: Body text
            topic: Blog topic

        Returns:
            Score from 0-10
        """
        score = 5.0

        # Length and structure
        word_count = len(body.split())
        if word_count > 150:
            score += 1.5

        # Paragraph structure
        paragraphs = [p for p in body.split('\\n\\n') if p.strip()]
        if 2 <= len(paragraphs) <= 5:
            score += 1.0

        # Topic coverage
        topic_words = topic.lower().split()
        topic_coverage = sum(
            1 for word in topic_words
            if word in body.lower()
        )
        if topic_coverage >= len(topic_words) * 0.7:
            score += 1.5

        # Specific examples or data
        has_numbers = any(char.isdigit() for char in body)
        has_examples = "example" in body.lower()
        if has_numbers or has_examples:
            score += 1.0

        return min(10.0, score)

    def _score_conclusion(
        self,
        conclusion: str,
        topic: str,
        intro: str
    ) -> float:
        """Score conclusion section.

        Factors:
        - Length (20-50 words)
        - Summarization keywords
        - Call to action
        - Echoes intro theme

        Args:
            conclusion: Conclusion text
            topic: Blog topic
            intro: Introduction text (for theme checking)

        Returns:
            Score from 0-10
        """
        score = 5.0

        # Length check
        word_count = len(conclusion.split())
        if 20 <= word_count <= 50:
            score += 1.5

        # Summarization keywords
        summary_words = [
            "summary", "remember", "key", "learned",
            "important", "conclude", "takeaway"
        ]
        if any(word in conclusion.lower() for word in summary_words):
            score += 1.5

        # Call to action
        cta_words = [
            "try", "start", "begin", "explore",
            "consider", "think about", "apply"
        ]
        if any(word in conclusion.lower() for word in cta_words):
            score += 1.0

        # Echoes intro theme (common words)
        intro_words = set(w.lower() for w in intro.split() if len(w) > 4)
        conclusion_words = set(w.lower() for w in conclusion.split() if len(w) > 4)
        common_words = intro_words & conclusion_words
        if len(common_words) >= 2:
            score += 1.0

        return min(10.0, score)


def calculate_overall_score(scores: Dict[str, float]) -> float:
    """Calculate weighted overall score.

    Args:
        scores: Dictionary of section scores

    Returns:
        Weighted average score
    """
    weights = {
        "intro": 0.2,
        "body": 0.5,
        "conclusion": 0.3
    }

    total = sum(
        scores.get(section, 0) * weight
        for section, weight in weights.items()
    )

    return round(total, 2)
```

### 2. Create Tests

Create `test_evaluation.py`:

```python
"""Tests for evaluation system."""

from hvas_mini.evaluation import ContentEvaluator, calculate_overall_score
from hvas_mini.state import create_initial_state
import pytest


@pytest.fixture
def evaluator():
    return ContentEvaluator()


def test_intro_scoring(evaluator):
    """Test introduction scoring."""
    state = create_initial_state("machine learning")

    # Good intro
    state["intro"] = (
        "Have you ever wondered how computers learn from data? "
        "Machine learning enables systems to improve automatically. "
        "Let's explore this fascinating field."
    )

    result = evaluator(state)
    assert result["scores"]["intro"] >= 7.0


def test_body_scoring(evaluator):
    """Test body scoring."""
    state = create_initial_state("machine learning")

    # Good body with examples
    state["body"] = \"\"\"Machine learning is a subset of artificial intelligence.
It focuses on building systems that learn from data.

For example, recommendation systems use ML to suggest products.
In 2023, over 80% of companies adopted ML technologies.

There are three main types: supervised, unsupervised, and reinforcement learning.
Each type serves different purposes and use cases.

The applications are vast, from healthcare to finance.
Machine learning continues to transform industries worldwide.\"\"\"

    state["intro"] = "Introduction"
    state["conclusion"] = "Conclusion"

    result = evaluator(state)
    assert result["scores"]["body"] >= 7.0


def test_conclusion_scoring(evaluator):
    """Test conclusion scoring."""
    state = create_initial_state("machine learning")

    state["intro"] = "Machine learning helps computers learn from data automatically."
    state["body"] = "Body content here."

    # Good conclusion
    state["conclusion"] = (
        "Remember, machine learning transforms how computers process data. "
        "Start exploring ML in your own projects today. "
        "The key is practice and continuous learning."
    )

    result = evaluator(state)
    assert result["scores"]["conclusion"] >= 7.0


def test_poor_content_scores_low(evaluator):
    """Test that poor content gets low scores."""
    state = create_initial_state("test topic")

    # Poor content: too short, no structure
    state["intro"] = "This is intro."
    state["body"] = "Short body."
    state["conclusion"] = "Done."

    result = evaluator(state)

    assert result["scores"]["intro"] < 7.0
    assert result["scores"]["body"] < 7.0
    assert result["scores"]["conclusion"] < 7.0


def test_overall_score_calculation():
    """Test weighted overall score."""
    scores = {
        "intro": 8.0,
        "body": 9.0,
        "conclusion": 7.0
    }

    overall = calculate_overall_score(scores)

    # Should be weighted: 0.2*8 + 0.5*9 + 0.3*7 = 8.2
    assert overall == 8.2


def test_evaluator_logs_scores(evaluator):
    """Test that evaluator logs scores."""
    state = create_initial_state("test")
    state["intro"] = "Test intro content here."
    state["body"] = "Test body content here."
    state["conclusion"] = "Test conclusion content."

    result = evaluator(state)

    # Should add log entry
    assert len(result["stream_logs"]) > 0
    assert "[Evaluator]" in result["stream_logs"][-1]
```

## Deliverables Checklist

- [ ] `src/hvas_mini/evaluation.py` with:
  - [ ] `ContentEvaluator` class
  - [ ] `_score_intro()` method
  - [ ] `_score_body()` method
  - [ ] `_score_conclusion()` method
  - [ ] `calculate_overall_score()` function
  - [ ] Complete docstrings
- [ ] `test_evaluation.py` with passing tests
- [ ] Multi-factor scoring logic

## Acceptance Criteria

1. ✅ Evaluator scores all three sections
2. ✅ Good content scores >= 7.0
3. ✅ Poor content scores < 7.0
4. ✅ Scores written to state["scores"]
5. ✅ Logs added to stream_logs
6. ✅ All tests pass: `uv run pytest test_evaluation.py`
7. ✅ Score calculation is deterministic

## Testing

```bash
cd worktrees/evaluation-system
uv run pytest test_evaluation.py -v
```

## Integration Notes

The evaluator will be:
- Called as a LangGraph node after content generation
- Used to determine which memories to store (threshold)
- Used to trigger parameter evolution
- Displayed in visualization

## Next Steps

After completion, merge to main and integrate with:
- feature/langgraph-orchestration (as evaluation node)
- feature/base-agent (for evolution triggers)
