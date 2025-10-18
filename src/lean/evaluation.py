"""
Content evaluation system for HVAS Mini.

Scores generated content based on multiple quality factors.
"""

from typing import Dict
import re
import os
from dotenv import load_dotenv

load_dotenv()

# Import BlogState - will need state.py from state-management branch
try:
    from lean.state import BlogState
except ImportError:
    # For standalone development
    from typing import TypedDict, List

    class BlogState(TypedDict):
        """Temporary type definition."""

        topic: str
        intro: str
        body: str
        conclusion: str
        scores: Dict[str, float]
        retrieved_memories: Dict[str, List[str]]
        parameter_updates: Dict[str, Dict[str, float]]
        generation_id: str
        timestamp: str
        stream_logs: List[str]


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
                state["conclusion"], state["topic"], state["intro"]
            ),
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
            "discover",
            "learn",
            "imagine",
            "what if",
            "have you ever",
            "wondering",
            "explore",
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
        paragraphs = [p for p in body.split("\n\n") if p.strip()]
        if 2 <= len(paragraphs) <= 5:
            score += 1.0

        # Topic coverage
        topic_words = topic.lower().split()
        topic_coverage = sum(1 for word in topic_words if word in body.lower())
        if topic_coverage >= len(topic_words) * 0.7:
            score += 1.5

        # Specific examples or data
        has_numbers = any(char.isdigit() for char in body)
        has_examples = "example" in body.lower()
        if has_numbers or has_examples:
            score += 1.0

        return min(10.0, score)

    def _score_conclusion(self, conclusion: str, topic: str, intro: str) -> float:
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
            "summary",
            "remember",
            "key",
            "learned",
            "important",
            "conclude",
            "takeaway",
        ]
        if any(word in conclusion.lower() for word in summary_words):
            score += 1.5

        # Call to action
        cta_words = [
            "try",
            "start",
            "begin",
            "explore",
            "consider",
            "think about",
            "apply",
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
    weights = {"intro": 0.2, "body": 0.5, "conclusion": 0.3}

    total = sum(scores.get(section, 0) * weight for section, weight in weights.items())

    return round(total, 2)
