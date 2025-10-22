"""
CoordinatorAgent - Hierarchical coordinator with research and critique capabilities.

This agent implements the coordinator role from the 3-layer architecture:
- Layer 1: Coordinator orchestrates workflow
- Layer 2: Content agents (intro, body, conclusion)
- Layer 3: Specialist agents (researcher, fact-checker, stylist)
"""

from typing import List, Dict, Optional, Any
from lean.base_agent import BaseAgent
from lean.reasoning_memory import ReasoningMemory
from lean.shared_rag import SharedRAG
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()



class CoordinatorAgent(BaseAgent):
    """Coordinator agent that orchestrates research, distribution, and critique.

    Responsibilities:
    1. Research topics using Tavily (if enabled)
    2. Distribute context to child agents (intro, body, conclusion)
    3. Aggregate outputs from children
    4. Critique quality and request revisions if needed
    """

    def __init__(
        self,
        agent_id: str,
        reasoning_memory: ReasoningMemory,
        shared_rag: SharedRAG,
        parent_ids: Optional[List[str]] = None,
        enable_research: bool = True,
        system_prompt: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize coordinator agent.

        Args:
            agent_id: Unique identifier
            reasoning_memory: Reasoning pattern storage
            shared_rag: Shared knowledge base
            parent_ids: Parent agent IDs
            enable_research: Whether to enable Tavily research
            system_prompt: Base system prompt from YAML config
        """
        super().__init__(
            role='coordinator',
            agent_id=agent_id,
            reasoning_memory=reasoning_memory,
            shared_rag=shared_rag,
            parent_ids=parent_ids,
            system_prompt=system_prompt,
            llm_config=llm_config
        )

        self.enable_research = enable_research
        self.tavily_client = None

        # Initialize Tavily if enabled and API key available
        if self.enable_research:
            self._init_tavily()

    def _init_tavily(self):
        """Initialize Tavily research client."""
        tavily_api_key = os.getenv('TAVILY_API_KEY')

        if not tavily_api_key:
            logger.warning("TAVILY_API_KEY not found, research disabled")
            self.enable_research = False
            return

        try:
            from tavily import TavilyClient
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("Tavily research enabled for coordinator")
        except ImportError:
            logger.warning("Tavily package not installed, research disabled")
            self.enable_research = False

    def research_topic(
        self,
        topic: str,
        max_results: int = 5,
        search_depth: str = "advanced"
    ) -> Dict[str, any]:
        """Research topic using Tavily.

        Args:
            topic: Topic to research
            max_results: Maximum number of search results
            search_depth: "basic" or "advanced"

        Returns:
            Dict with research results and summary
        """
        if not self.enable_research or not self.tavily_client:
            return {
                'query': topic,
                'results': [],
                'summary': 'Research disabled or unavailable'
            }

        try:
            # Perform Tavily search
            response = self.tavily_client.search(
                query=topic,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=True
            )

            results = []
            for item in response.get('results', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'content': item.get('content', ''),
                    'score': item.get('score', 0.0)
                })

            return {
                'query': topic,
                'results': results,
                'summary': response.get('answer', ''),
                'raw_response': response
            }

        except Exception as e:
            logger.error(f"Tavily research error: {e}")
            return {
                'query': topic,
                'results': [],
                'summary': f'Research failed: {str(e)}'
            }

    def synthesize_research(
        self,
        research_results: Dict[str, any],
        reasoning_patterns: List[Dict],
        domain_knowledge: List[Dict]
    ) -> Dict[str, str]:
        """Synthesize research findings into structured context.

        Args:
            research_results: Results from research_topic()
            reasoning_patterns: Retrieved reasoning patterns
            domain_knowledge: Retrieved domain knowledge

        Returns:
            Dict with synthesized context for distribution
        """
        # Build prompt for synthesis
        results_text = self._format_research_results(research_results)
        reasoning_examples = self._format_reasoning_patterns(reasoning_patterns)
        knowledge_context = self._format_knowledge(domain_knowledge)

        prompt = f"""
{self._get_role_instruction()}

RESEARCH RESULTS:
{results_text}

PREVIOUS REASONING PATTERNS:
{reasoning_examples}

DOMAIN KNOWLEDGE:
{knowledge_context}

INSTRUCTIONS:
Synthesize the research findings into structured context for distribution to child agents.

Use <think> tags to plan your synthesis strategy:
- What are the key themes?
- What facts and statistics are most relevant?
- Are there multiple perspectives to note?
- What context will each agent type need?

Then use <final> tags to provide:
1. INTRO CONTEXT: Hooks, current relevance, framing insights
2. BODY CONTEXT: Core facts, arguments, evidence, structure suggestions
3. CONCLUSION CONTEXT: Implications, future directions, synthesis points

Format:
<think>
[Your synthesis planning]
</think>

<final>
INTRO CONTEXT:
[Context for intro agent]

BODY CONTEXT:
[Context for body agent]

CONCLUSION CONTEXT:
[Context for conclusion agent]
</final>
"""

        # Generate synthesis
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse response
        thinking, output = self._parse_response(response_text)

        # Extract section-specific contexts
        contexts = self._extract_agent_contexts(output)

        return {
            'thinking': thinking,
            'intro_context': contexts.get('intro', ''),
            'body_context': contexts.get('body', ''),
            'conclusion_context': contexts.get('conclusion', ''),
            'full_synthesis': output
        }

    def _format_research_results(self, research_results: Dict[str, any]) -> str:
        """Format research results for prompt.

        Args:
            research_results: Results from research_topic()

        Returns:
            Formatted string
        """
        if not research_results.get('results'):
            return "No research results available."

        formatted = []
        summary = research_results.get('summary', '')

        if summary:
            formatted.append(f"SUMMARY: {summary}\n")

        formatted.append("SOURCES:")
        for i, result in enumerate(research_results['results'][:5], 1):
            title = result.get('title', 'Untitled')
            content = result.get('content', '')[:300]
            score = result.get('score', 0.0)

            formatted.append(
                f"\n{i}. {title} (relevance: {score:.2f})\n"
                f"   {content}..."
            )

        return "\n".join(formatted)

    def _extract_agent_contexts(self, synthesis: str) -> Dict[str, str]:
        """Extract section-specific contexts from synthesis.

        Args:
            synthesis: Full synthesis text

        Returns:
            Dict mapping agent type to context
        """
        import re

        contexts = {}

        # Extract INTRO CONTEXT
        intro_pattern = r'INTRO CONTEXT:?\s*(.+?)(?=BODY CONTEXT:|CONCLUSION CONTEXT:|$)'
        intro_match = re.search(intro_pattern, synthesis, re.DOTALL | re.IGNORECASE)
        if intro_match:
            contexts['intro'] = intro_match.group(1).strip()

        # Extract BODY CONTEXT
        body_pattern = r'BODY CONTEXT:?\s*(.+?)(?=CONCLUSION CONTEXT:|$)'
        body_match = re.search(body_pattern, synthesis, re.DOTALL | re.IGNORECASE)
        if body_match:
            contexts['body'] = body_match.group(1).strip()

        # Extract CONCLUSION CONTEXT
        conclusion_pattern = r'CONCLUSION CONTEXT:?\s*(.+?)$'
        conclusion_match = re.search(conclusion_pattern, synthesis, re.DOTALL | re.IGNORECASE)
        if conclusion_match:
            contexts['conclusion'] = conclusion_match.group(1).strip()

        return contexts

    def critique_output(
        self,
        intro: str,
        body: str,
        conclusion: str,
        topic: str,
        research_context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """Critique aggregated output from child agents.

        Args:
            intro: Introduction content
            body: Body content
            conclusion: Conclusion content
            topic: Original topic
            research_context: Optional research context for fact-checking

        Returns:
            Dict with scores, feedback, and revision_needed flag
        """
        research_text = ""
        if research_context and research_context.get('summary'):
            research_text = f"\nRESEARCH CONTEXT:\n{research_context['summary']}"

        prompt = f"""
{self._get_role_instruction()}

TOPIC: {topic}
{research_text}

INTRO:
{intro}

BODY:
{body}

CONCLUSION:
{conclusion}

INSTRUCTIONS:
Critique this complete piece. Use <think> tags for your evaluation process, then <final> tags for structured feedback.

Evaluate:
1. COHERENCE: Do sections flow together?
2. ACCURACY: Are facts correct (check against research if available)?
3. DEPTH: Is there meaningful insight?
4. QUALITY: Overall score 0-10

Format:
<think>
[Your evaluation reasoning]
</think>

<final>
SCORES:
- Coherence: [0-10]
- Accuracy: [0-10]
- Depth: [0-10]
- Overall: [0-10]

FEEDBACK:
[Specific constructive feedback]

REVISION_NEEDED: [YES/NO]
</final>
"""

        # Generate critique
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse response
        thinking, output = self._parse_response(response_text)

        # Extract scores and feedback
        evaluation = self._parse_critique(output)
        evaluation['thinking'] = thinking

        return evaluation

    def _parse_critique(self, critique_text: str) -> Dict[str, any]:
        """Parse critique into structured format.

        Args:
            critique_text: Critique text from LLM

        Returns:
            Dict with scores, feedback, revision_needed
        """
        import re

        result = {
            'scores': {},
            'feedback': '',
            'revision_needed': False
        }

        # Extract scores
        score_patterns = {
            'coherence': r'Coherence:?\s*(\d+(?:\.\d+)?)',
            'accuracy': r'Accuracy:?\s*(\d+(?:\.\d+)?)',
            'depth': r'Depth:?\s*(\d+(?:\.\d+)?)',
            'overall': r'Overall:?\s*(\d+(?:\.\d+)?)'
        }

        for key, pattern in score_patterns.items():
            match = re.search(pattern, critique_text, re.IGNORECASE)
            if match:
                result['scores'][key] = float(match.group(1))

        # Extract feedback
        feedback_pattern = r'FEEDBACK:?\s*(.+?)(?=REVISION_NEEDED:|$)'
        feedback_match = re.search(feedback_pattern, critique_text, re.DOTALL | re.IGNORECASE)
        if feedback_match:
            result['feedback'] = feedback_match.group(1).strip()

        # Extract revision needed
        revision_pattern = r'REVISION_NEEDED:?\s*(YES|NO)'
        revision_match = re.search(revision_pattern, critique_text, re.IGNORECASE)
        if revision_match:
            result['revision_needed'] = revision_match.group(1).upper() == 'YES'

        return result
