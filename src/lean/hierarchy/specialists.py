"""
Specialist agents - Layer 3 focused sub-task agents.
"""

from lean.agents import BaseAgent
from typing import List, Dict


class ResearchAgent(BaseAgent):
    """Finds relevant information and facts (Layer 3)."""

    def __init__(self, memory_manager, trust_manager=None):
        super().__init__(role="researcher", memory_manager=memory_manager, trust_manager=trust_manager)

    @property
    def content_key(self) -> str:
        return "research"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Find relevant research for the topic.

        Args:
            state: Current state
            memories: Retrieved memories
            weighted_context: Context from parent

        Returns:
            Research findings
        """
        topic = state.get("topic", "")

        prompt = f"""You are a research specialist for blog content.

Topic: {topic}

Context from parent: {weighted_context}

Find 2-3 key facts, statistics, or insights relevant to this topic.
Focus on accurate, useful information.

Research findings:"""

        response = await self.llm.ainvoke(prompt)
        return response.content


class FactCheckerAgent(BaseAgent):
    """Verifies accuracy and flags issues (Layer 3)."""

    def __init__(self, memory_manager, trust_manager=None):
        super().__init__(role="fact_checker", memory_manager=memory_manager, trust_manager=trust_manager)

    @property
    def content_key(self) -> str:
        return "fact_check"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Verify content accuracy.

        Args:
            state: Current state
            memories: Retrieved memories
            weighted_context: Content to fact-check

        Returns:
            Fact-check results
        """
        topic = state.get("topic", "")

        prompt = f"""You are a fact-checking specialist.

Topic: {topic}

Content to verify: {weighted_context}

Review for:
1. Factual accuracy
2. Claims that need verification
3. Potential inaccuracies

Provide fact-check feedback:"""

        response = await self.llm.ainvoke(prompt)
        return response.content


class StyleAgent(BaseAgent):
    """Enhances tone, flow, and engagement (Layer 3)."""

    def __init__(self, memory_manager, trust_manager=None):
        super().__init__(role="stylist", memory_manager=memory_manager, trust_manager=trust_manager)

    @property
    def content_key(self) -> str:
        return "style"

    async def generate_content(
        self, state, memories: List[Dict], weighted_context: str = ""
    ) -> str:
        """Enhance style and tone.

        Args:
            state: Current state
            memories: Retrieved memories
            weighted_context: Content to enhance

        Returns:
            Style suggestions
        """
        topic = state.get("topic", "")

        prompt = f"""You are a style enhancement specialist.

Topic: {topic}

Content: {weighted_context}

Suggest improvements to:
1. Tone and voice
2. Flow and transitions
3. Engagement and hooks

Style suggestions:"""

        response = await self.llm.ainvoke(prompt)
        return response.content
