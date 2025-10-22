# Coordinator Agent Role

## Overview

The Coordinator Agent is the top-level agent (Layer 1) in the LEAN hierarchical architecture. It orchestrates the entire content generation process, conducts research, and ensures quality.

## Responsibilities

### 1. Research Phase

Before distributing work to child agents, the coordinator performs comprehensive research using Tavily:

- **Topic Exploration**: Broad search to understand current discourse
- **Fact Gathering**: Collect statistics, studies, expert opinions
- **Source Evaluation**: Assess credibility and relevance
- **Synthesis**: Organize findings for distribution

### 2. Context Distribution

The coordinator filters and distributes relevant context to child agents based on semantic distance:

- **Intro Agent**: Receives hooks, current relevance, framing insights
- **Body Agent**: Receives core facts, arguments, evidence, structure suggestions
- **Conclusion Agent**: Receives implications, future directions, synthesis points

### 3. Quality Critique

After receiving outputs from all children, the coordinator:

- Evaluates coherence across sections
- Checks factual accuracy against research
- Assesses depth and insight
- Scores on 0-10 scale
- Provides specific revision feedback if quality < threshold

## Research Integration

The coordinator uses Tavily search to:

1. **Contextualize the Topic**: Understand current events, debates, recent developments
2. **Gather Evidence**: Find statistics, studies, expert quotes
3. **Identify Perspectives**: Note different viewpoints and controversies
4. **Update Knowledge**: Supplement RAG memory with fresh information

Example research queries:
- "[topic] recent developments 2025"
- "[topic] expert opinions"
- "[topic] statistics data"
- "[topic] case studies examples"

## Quality Thresholds

The coordinator enforces:

- **Minimum Average Confidence**: 0.8 (80%)
- **Maximum Passes**: 3 refinement iterations
- **Early Exit**: If quality threshold met, skip additional passes

## Reasoning Patterns

The coordinator learns from experience:

- **Successful Research Strategies**: Which queries yielded best insights
- **Effective Distribution**: What context helped children perform best
- **Critique Patterns**: What feedback led to improvements

## Integration with Evolution

As the coordinator evolves through reproduction:

- Better agents learn more effective research strategies
- Improved context filtering and distribution
- More accurate quality assessment
- Enhanced synthesis abilities
