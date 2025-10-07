# Custom Evaluation - Customization Guide

This guide explains how to customize the evaluation system to score agent outputs according to your specific quality criteria.

---

## Table of Contents

1. [Understanding the Evaluation System](#understanding-the-evaluation-system)
2. [Customizing Scoring Functions](#customizing-scoring-functions)
3. [Adding New Evaluation Criteria](#adding-new-evaluation-criteria)
4. [LLM-Based Evaluation](#llm-based-evaluation)
5. [Multi-Objective Scoring](#multi-objective-scoring)
6. [Examples](#examples)

---

## Understanding the Evaluation System

### Current Architecture

```python
ContentEvaluator
  ├── __call__(state) → state with scores
  ├── _score_intro(intro, topic) → float (0-10)
  ├── _score_body(body, topic) → float (0-10)
  └── _score_conclusion(conclusion, topic, intro) → float (0-10)
```

### Scoring Principles

1. **Scale**: 0-10 (higher is better)
2. **Threshold**: Default 7.0 for memory storage
3. **Multi-Factor**: Combines multiple quality dimensions
4. **Deterministic**: Same input → same score (for reproducibility)

### How Scores Are Used

```
Generation → Evaluation → Decision
     ↓            ↓           ↓
  Content      Scores     Store in memory? (≥ 7.0)
                 ↓           ↓
              Agent      Evolve parameters
```

---

## Customizing Scoring Functions

### Modifying Existing Scores

Edit `src/hvas_mini/evaluation.py`:

```python
def _score_intro(self, intro: str, topic: str) -> float:
    """Score introduction with custom criteria."""
    score = 5.0  # Base score

    # Custom Criterion 1: Opening Impact
    strong_openings = ["imagine", "what if", "did you know", "statistics show"]
    first_words = intro.lower()[:50]
    if any(opening in first_words for opening in strong_openings):
        score += 2.0  # High weight for strong opening

    # Custom Criterion 2: Emotional Appeal
    emotion_words = ["exciting", "fascinating", "transformative", "revolutionary"]
    if any(word in intro.lower() for word in emotion_words):
        score += 1.0

    # Custom Criterion 3: Length (adjusted)
    word_count = len(intro.split())
    if 30 <= word_count <= 80:  # Prefer longer intros
        score += 1.5

    # Custom Criterion 4: Question Engagement
    if intro.count("?") >= 2:  # Multiple questions
        score += 1.5

    return min(10.0, score)
```

### Adjusting Score Weights

```python
def _score_body(self, body: str, topic: str) -> float:
    """Score body with weighted criteria."""
    criteria_scores = {}

    # Criterion 1: Length (weight: 0.2)
    word_count = len(body.split())
    if word_count > 150:
        criteria_scores["length"] = 10.0
    else:
        criteria_scores["length"] = (word_count / 150) * 10

    # Criterion 2: Structure (weight: 0.3)
    paragraphs = [p for p in body.split('\n\n') if p.strip()]
    if 3 <= len(paragraphs) <= 6:
        criteria_scores["structure"] = 10.0
    else:
        criteria_scores["structure"] = 5.0

    # Criterion 3: Examples (weight: 0.25)
    has_examples = body.lower().count("example") + body.lower().count("for instance")
    criteria_scores["examples"] = min(10.0, has_examples * 3.0)

    # Criterion 4: Data/Numbers (weight: 0.25)
    numbers = sum(1 for char in body if char.isdigit())
    criteria_scores["data"] = min(10.0, (numbers / 5) * 10)

    # Weighted average
    weights = {
        "length": 0.2,
        "structure": 0.3,
        "examples": 0.25,
        "data": 0.25
    }

    final_score = sum(
        criteria_scores[k] * weights[k]
        for k in weights.keys()
    )

    return round(final_score, 2)
```

---

## Adding New Evaluation Criteria

### Example: Readability Score

```python
import textstat  # pip install textstat

class ContentEvaluator:
    def _score_readability(self, text: str) -> float:
        """Score based on readability metrics."""
        # Flesch Reading Ease (0-100, higher = easier)
        ease = textstat.flesch_reading_ease(text)

        # Convert to 0-10 scale
        # Target: 60-70 (standard reading level)
        if 60 <= ease <= 70:
            score = 10.0
        elif 50 <= ease < 60:
            score = 8.0
        elif 70 < ease <= 80:
            score = 8.0
        else:
            score = 6.0

        return score

    def _score_body(self, body: str, topic: str) -> float:
        base_score = 5.0

        # Existing criteria...
        # ...

        # Add readability
        readability_score = self._score_readability(body)
        if readability_score >= 8.0:
            base_score += 1.5

        return min(10.0, base_score)
```

### Example: Sentiment Analysis

```python
from textblob import TextBlob  # pip install textblob

class ContentEvaluator:
    def _score_sentiment(self, text: str, desired: str = "positive") -> float:
        """Score based on sentiment alignment."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1

        if desired == "positive":
            # Prefer positive sentiment
            score = (polarity + 1) * 5  # Convert to 0-10
        elif desired == "neutral":
            # Prefer neutral sentiment
            score = (1 - abs(polarity)) * 10
        else:
            score = 5.0

        return score

    def _score_intro(self, intro: str, topic: str) -> float:
        base_score = 5.0

        # Existing criteria...
        # ...

        # Add sentiment
        sentiment_score = self._score_sentiment(intro, desired="positive")
        if sentiment_score >= 7.0:
            base_score += 1.0

        return min(10.0, base_score)
```

### Example: Keyword Density

```python
from collections import Counter

class ContentEvaluator:
    def _score_keyword_density(self, text: str, topic: str) -> float:
        """Score based on optimal keyword usage."""
        # Extract topic keywords
        topic_words = set(topic.lower().split())

        # Count words in text
        words = text.lower().split()
        word_count = len(words)

        # Count topic word occurrences
        topic_occurrences = sum(1 for word in words if word in topic_words)

        # Calculate density (percentage)
        density = (topic_occurrences / word_count) * 100 if word_count > 0 else 0

        # Optimal density: 2-5%
        if 2 <= density <= 5:
            score = 10.0
        elif 1 <= density < 2:
            score = 7.0
        elif 5 < density <= 7:
            score = 7.0
        else:
            score = 5.0  # Too low or too high

        return score
```

---

## LLM-Based Evaluation

### Basic LLM Evaluator

```python
from langchain_anthropic import ChatAnthropic

class LLMContentEvaluator(ContentEvaluator):
    """Evaluator using LLM for scoring."""

    def __init__(self):
        super().__init__()
        self.eval_llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.0  # Deterministic evaluation
        )

    async def _llm_score(
        self,
        content: str,
        criteria: str,
        scale: str = "0-10"
    ) -> float:
        """Score content using LLM."""
        prompt = f\"\"\"Evaluate this content on the following criteria:

{criteria}

Content:
{content}

Provide a score on a {scale} scale where:
- 0-3: Poor quality
- 4-6: Average quality
- 7-8: Good quality
- 9-10: Excellent quality

Respond with ONLY a number (e.g., "7.5"). No explanation.

Score:\"\"\"

        response = await self.eval_llm.ainvoke(prompt)

        # Parse score
        try:
            score = float(response.content.strip())
            return max(0.0, min(10.0, score))
        except ValueError:
            return 5.0  # Default if parsing fails

    async def _score_intro(self, intro: str, topic: str) -> float:
        """Score intro using LLM."""
        criteria = \"\"\"
1. Does it hook the reader immediately?
2. Is the topic clearly introduced?
3. Does it set appropriate expectations?
4. Is it concise (2-3 sentences)?
        \"\"\"

        return await self._llm_score(intro, criteria)
```

### Multi-Aspect LLM Evaluation

```python
class MultiAspectEvaluator(ContentEvaluator):
    """Evaluate multiple aspects with LLM."""

    async def _detailed_llm_score(self, content: str, aspect: str) -> Dict[str, float]:
        """Get detailed scoring for multiple aspects."""
        prompt = f\"\"\"Evaluate this content on multiple dimensions:

Content:
{content}

Provide scores (0-10) for:
1. Clarity: How clear and understandable is it?
2. Engagement: How engaging and interesting is it?
3. Accuracy: How accurate and factual does it seem?
4. Relevance: How relevant is it to the topic?
5. Style: How well-written is it?

Respond in JSON format:
{{
  "clarity": X.X,
  "engagement": X.X,
  "accuracy": X.X,
  "relevance": X.X,
  "style": X.X
}}\"\"\"

        response = await self.eval_llm.ainvoke(prompt)

        try:
            import json
            scores = json.loads(response.content)
            return scores
        except (json.JSONDecodeError, ValueError):
            return {
                "clarity": 5.0,
                "engagement": 5.0,
                "accuracy": 5.0,
                "relevance": 5.0,
                "style": 5.0
            }

    async def _score_body(self, body: str, topic: str) -> float:
        """Score body with detailed LLM evaluation."""
        aspects = await self._detailed_llm_score(body, "body")

        # Weighted average
        weights = {
            "clarity": 0.25,
            "engagement": 0.15,
            "accuracy": 0.25,
            "relevance": 0.20,
            "style": 0.15
        }

        final_score = sum(
            aspects[k] * weights[k]
            for k in weights.keys()
        )

        return round(final_score, 2)
```

---

## Multi-Objective Scoring

### Pareto Optimization

```python
class MultiObjectiveEvaluator(ContentEvaluator):
    """Evaluate against multiple objectives."""

    def _score_intro(self, intro: str, topic: str) -> float:
        """Multi-objective intro scoring."""
        objectives = {
            "brevity": self._score_brevity(intro),
            "engagement": self._score_engagement(intro),
            "clarity": self._score_clarity(intro),
            "relevance": self._score_relevance(intro, topic)
        }

        # Store individual objective scores
        self.objective_scores = objectives

        # Aggregate: minimum (worst case)
        # This ensures all objectives are met
        return min(objectives.values())

        # Alternative: weighted average
        # weights = {"brevity": 0.2, "engagement": 0.3, "clarity": 0.3, "relevance": 0.2}
        # return sum(objectives[k] * weights[k] for k in weights)

    def _score_brevity(self, text: str) -> float:
        """Score based on conciseness."""
        words = len(text.split())
        if words <= 50:
            return 10.0
        elif words <= 80:
            return 7.0
        else:
            return 5.0

    def _score_engagement(self, text: str) -> float:
        """Score based on engagement factors."""
        score = 5.0
        if "?" in text:
            score += 2.0
        if any(word in text.lower() for word in ["you", "your"]):
            score += 1.5
        if text[0].isupper() and text[0] not in ["T", "I", "A"]:
            score += 1.5
        return min(10.0, score)

    def _score_clarity(self, text: str) -> float:
        """Score based on clarity."""
        # Simple heuristic: average sentence length
        sentences = text.split(".")
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0

        if 10 <= avg_length <= 20:
            return 10.0
        elif 8 <= avg_length < 10 or 20 < avg_length <= 25:
            return 7.0
        else:
            return 5.0

    def _score_relevance(self, text: str, topic: str) -> float:
        """Score based on topic relevance."""
        topic_words = set(topic.lower().split())
        text_words = set(text.lower().split())

        overlap = len(topic_words & text_words) / len(topic_words) if topic_words else 0

        return overlap * 10
```

### Dynamic Weighting

```python
class AdaptiveEvaluator(ContentEvaluator):
    """Evaluator with dynamic weights based on context."""

    def __init__(self):
        super().__init__()
        self.generation_count = 0

    def _get_adaptive_weights(self, generation: int) -> Dict[str, float]:
        """Get weights that change over generations."""
        if generation < 3:
            # Early: prioritize structure
            return {
                "structure": 0.4,
                "content": 0.3,
                "creativity": 0.3
            }
        else:
            # Later: prioritize creativity
            return {
                "structure": 0.2,
                "content": 0.3,
                "creativity": 0.5
            }

    def _score_intro(self, intro: str, topic: str) -> float:
        """Score with adaptive weights."""
        self.generation_count += 1

        # Calculate sub-scores
        structure_score = self._score_structure(intro)
        content_score = self._score_content_quality(intro, topic)
        creativity_score = self._score_creativity(intro)

        # Get adaptive weights
        weights = self._get_adaptive_weights(self.generation_count)

        # Weighted combination
        final_score = (
            structure_score * weights["structure"] +
            content_score * weights["content"] +
            creativity_score * weights["creativity"]
        )

        return round(final_score, 2)
```

---

## Examples

### Example 1: Technical Accuracy Evaluator

```python
class TechnicalEvaluator(ContentEvaluator):
    """Evaluator for technical content."""

    def _score_body(self, body: str, topic: str) -> float:
        score = 5.0

        # Technical terms usage
        technical_indicators = [
            "algorithm", "data structure", "implementation",
            "performance", "optimization", "architecture"
        ]
        tech_count = sum(1 for term in technical_indicators if term in body.lower())
        score += min(2.0, tech_count * 0.5)

        # Code examples
        if "```" in body or "code" in body.lower():
            score += 1.5

        # Precise language
        vague_words = ["maybe", "possibly", "might", "could be"]
        vague_count = sum(1 for word in vague_words if word in body.lower())
        if vague_count == 0:
            score += 1.5
        else:
            score -= vague_count * 0.3

        # Citations/references
        if "according to" in body.lower() or "[" in body:
            score += 1.0

        return min(10.0, score)
```

### Example 2: SEO-Focused Evaluator

```python
class SEOEvaluator(ContentEvaluator):
    """Evaluator prioritizing SEO factors."""

    def _score_intro(self, intro: str, topic: str) -> float:
        score = 5.0

        # Keyword in first 50 characters
        if any(word.lower() in intro[:50].lower() for word in topic.split()):
            score += 2.0

        # Length for featured snippet (40-60 words)
        word_count = len(intro.split())
        if 40 <= word_count <= 60:
            score += 2.0

        # Natural language questions
        if intro.startswith(("How", "What", "Why", "When", "Where")):
            score += 1.0

        return min(10.0, score)

    def _score_body(self, body: str, topic: str) -> float:
        score = 5.0

        # Header structure (H2, H3 simulation via newlines)
        headers = body.count('\n\n')
        if 3 <= headers <= 6:
            score += 1.5

        # Keyword density (1-3%)
        words = body.lower().split()
        keyword_count = sum(1 for word in words if word in topic.lower().split())
        density = (keyword_count / len(words)) * 100 if words else 0

        if 1 <= density <= 3:
            score += 2.0

        # Internal links simulation (mentions of related topics)
        link_indicators = ["see also", "learn more", "related", "check out"]
        if any(indicator in body.lower() for indicator in link_indicators):
            score += 1.0

        # Long-form content (>300 words)
        if len(words) > 300:
            score += 1.5

        return min(10.0, score)
```

### Example 3: Storytelling Evaluator

```python
class StorytellingEvaluator(ContentEvaluator):
    """Evaluator for narrative content."""

    def _score_intro(self, intro: str, topic: str) -> float:
        score = 5.0

        # Starts with anecdote or story
        story_starts = [
            "once", "imagine", "picture this", "i remember",
            "a few years ago", "when i first"
        ]
        if any(intro.lower().startswith(start) for start in story_starts):
            score += 2.5

        # Sensory details
        sensory_words = [
            "saw", "heard", "felt", "smelled", "tasted",
            "bright", "loud", "rough", "sweet"
        ]
        sensory_count = sum(1 for word in sensory_words if word in intro.lower())
        score += min(1.5, sensory_count * 0.5)

        # Emotional language
        emotions = ["excited", "worried", "curious", "amazed", "frustrated"]
        if any(emotion in intro.lower() for emotion in emotions):
            score += 1.0

        return min(10.0, score)

    def _score_conclusion(self, conclusion: str, topic: str, intro: str) -> float:
        score = 5.0

        # Callback to intro
        intro_words = set(intro.lower().split())
        conclusion_words = set(conclusion.lower().split())
        callback = len(intro_words & conclusion_words)

        if callback >= 5:
            score += 2.0

        # Lesson or takeaway
        takeaway_words = ["lesson", "learned", "realize", "understand", "now i know"]
        if any(word in conclusion.lower() for word in takeaway_words):
            score += 2.0

        # Forward-looking
        future_words = ["will", "going to", "future", "next", "continue"]
        if any(word in conclusion.lower() for word in future_words):
            score += 1.0

        return min(10.0, score)
```

---

## Best Practices

1. **Calibrate Thresholds**: Test your scoring on sample content to calibrate thresholds
2. **Document Criteria**: Clearly document what each score range means
3. **Validate Scores**: Compare LLM scores with human judgment
4. **Avoid Over-Optimization**: Don't make criteria too specific or gameable
5. **Monitor Distribution**: Track score distribution over time
6. **A/B Test Changes**: Test evaluation changes on historical data first

---

## Troubleshooting

### Scores Always Too High/Low

**Problem**: All content scores near 9-10 or near 3-4

**Solution**:
1. Adjust base score and increments
2. Check if criteria are too easy/hard to meet
3. Normalize scores across generations

### LLM Evaluator Not Deterministic

**Problem**: Same content gets different scores

**Solution**:
1. Set `temperature=0` on evaluation LLM
2. Use structured output (JSON)
3. Average multiple evaluation runs

### Memory Not Accumulating

**Problem**: No content meets threshold

**Solution**:
1. Lower `MEMORY_SCORE_THRESHOLD` in .env
2. Make scoring criteria more achievable
3. Ensure base scores are reasonable (start at 5.0)

---

## Next Steps

- Explore [extending-agents.md](extending-agents.md) for creating new agents
- See [langgraph-patterns.md](langgraph-patterns.md) for evaluation node patterns
- Review `src/hvas_mini/evaluation.py` for complete implementation
