"""
Specialist Agents - Layer 3 support agents for content enhancement.

These agents provide specialized support to content agents:
- ResearcherAgent: Deep research and evidence validation
- FactCheckerAgent: Claim verification and accuracy checking
- StylistAgent: Clarity and readability enhancement
"""

from typing import List, Dict, Optional, Any
from lean.base_agent import BaseAgent
from lean.reasoning_memory import ReasoningMemory
from lean.shared_rag import SharedRAG


class ResearcherAgent(BaseAgent):
    """Research specialist for deep research and evidence support.

    Provides:
    - Source discovery and validation
    - Evidence gathering
    - Knowledge gap identification
    - Credibility assessment
    """
    # Uses base class _get_role_instruction() with YAML prompt

    def research_claim(
        self,
        claim: str,
        content_context: str = "",
        reasoning_patterns: Optional[List[Dict]] = None,
        domain_knowledge: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """Research and validate a specific claim.

        Args:
            claim: Claim to research
            content_context: Context from content being developed
            reasoning_patterns: Retrieved reasoning patterns
            domain_knowledge: Retrieved domain knowledge

        Returns:
            Dict with research findings and validation
        """
        if reasoning_patterns is None:
            reasoning_patterns = []
        if domain_knowledge is None:
            domain_knowledge = []

        # Format context
        reasoning_examples = self._format_reasoning_patterns(reasoning_patterns)
        knowledge_context = self._format_knowledge(domain_knowledge)

        prompt = f"""
{self._get_role_instruction()}

CLAIM TO RESEARCH: {claim}

CONTENT CONTEXT:
{content_context}

PREVIOUS RESEARCH PATTERNS:
{reasoning_examples}

DOMAIN KNOWLEDGE:
{knowledge_context}

INSTRUCTIONS:
Research this claim and provide evidence-based insights.

Use <think> tags to plan your research approach, then <final> tags for findings.

Format:
<think>
[Your research strategy]
</think>

<final>
VALIDATION: [SUPPORTED/UNSUPPORTED/UNCERTAIN]

EVIDENCE:
[Relevant evidence for or against the claim]

CREDIBILITY: [HIGH/MEDIUM/LOW]

RECOMMENDATIONS:
[How to strengthen this claim with better evidence]
</final>
"""

        # Generate research
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse response
        thinking, output = self._parse_response(response_text)

        return {
            'claim': claim,
            'thinking': thinking,
            'findings': output,
            'raw_response': response_text
        }


class FactCheckerAgent(BaseAgent):
    """Fact-checking specialist for accuracy verification.

    Provides:
    - Claim verification
    - Error detection
    - Source validation
    - Correction suggestions
    """
    # Uses base class _get_role_instruction() with YAML prompt

    def check_content(
        self,
        content: str,
        research_context: Optional[str] = "",
        reasoning_patterns: Optional[List[Dict]] = None,
        domain_knowledge: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """Fact-check content for accuracy.

        Args:
            content: Content to fact-check
            research_context: Research context for verification
            reasoning_patterns: Retrieved reasoning patterns
            domain_knowledge: Retrieved domain knowledge

        Returns:
            Dict with fact-check results and corrections
        """
        if reasoning_patterns is None:
            reasoning_patterns = []
        if domain_knowledge is None:
            domain_knowledge = []

        # Format context
        reasoning_examples = self._format_reasoning_patterns(reasoning_patterns)
        knowledge_context = self._format_knowledge(domain_knowledge)

        research_text = ""
        if research_context:
            research_text = f"\nRESEARCH CONTEXT:\n{research_context}"

        prompt = f"""
{self._get_role_instruction()}

CONTENT TO CHECK:
{content}
{research_text}

PREVIOUS FACT-CHECKING PATTERNS:
{reasoning_examples}

DOMAIN KNOWLEDGE:
{knowledge_context}

INSTRUCTIONS:
Fact-check this content for accuracy.

Use <think> tags for your verification process, then <final> tags for findings.

Format:
<think>
[Your fact-checking approach]
</think>

<final>
ACCURACY SCORE: [0-10]

ISSUES FOUND:
[List any factual errors, unsupported claims, or misleading statements]

CORRECTIONS:
[Specific suggestions for fixing issues]

STATUS: [VERIFIED/NEEDS_REVISION/UNCERTAIN]
</final>
"""

        # Generate fact-check
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse response
        thinking, output = self._parse_response(response_text)

        return {
            'content_checked': content[:200] + "...",
            'thinking': thinking,
            'findings': output,
            'raw_response': response_text
        }


class StylistAgent(BaseAgent):
    """Style specialist for clarity and readability enhancement.

    Provides:
    - Clarity improvement
    - Readability enhancement
    - Style refinement
    - Error correction
    """
    # Uses base class _get_role_instruction() with YAML prompt

    def improve_style(
        self,
        content: str,
        target_tone: str = "professional",
        reasoning_patterns: Optional[List[Dict]] = None,
        domain_knowledge: Optional[List[Dict]] = None
    ) -> Dict[str, any]:
        """Improve content style and readability.

        Args:
            content: Content to improve
            target_tone: Desired tone (professional, casual, academic, etc.)
            reasoning_patterns: Retrieved reasoning patterns
            domain_knowledge: Retrieved domain knowledge

        Returns:
            Dict with style improvements and suggestions
        """
        if reasoning_patterns is None:
            reasoning_patterns = []
        if domain_knowledge is None:
            domain_knowledge = []

        # Format context
        reasoning_examples = self._format_reasoning_patterns(reasoning_patterns)
        knowledge_context = self._format_knowledge(domain_knowledge)

        prompt = f"""
{self._get_role_instruction()}

CONTENT TO IMPROVE:
{content}

TARGET TONE: {target_tone}

PREVIOUS STYLE PATTERNS:
{reasoning_examples}

DOMAIN KNOWLEDGE:
{knowledge_context}

INSTRUCTIONS:
Improve the style and clarity of this content while preserving its meaning.

Use <think> tags for your editing approach, then <final> tags for improvements.

Format:
<think>
[Your style improvement strategy]
</think>

<final>
READABILITY SCORE: [0-10]

IMPROVEMENTS:
[Specific style improvements with before/after examples]

REVISED VERSION:
[Improved version of the content]

TONE CONSISTENCY: [GOOD/NEEDS_WORK]
</final>
"""

        # Generate style improvements
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse response
        thinking, output = self._parse_response(response_text)

        return {
            'original_content': content[:200] + "...",
            'thinking': thinking,
            'improvements': output,
            'raw_response': response_text
        }


def create_specialist_agents(
    reasoning_dir: str = "./data/reasoning",
    shared_rag: Optional[SharedRAG] = None,
    agent_ids: Optional[Dict[str, str]] = None,
    agent_prompts: Optional[Dict[str, str]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    memory_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, BaseAgent]:
    """Create specialist agents.

    Args:
        reasoning_dir: Directory for reasoning patterns
        shared_rag: Shared knowledge base (reuse from main agents)
        agent_ids: Optional dict mapping role → agent_id
        agent_prompts: Optional dict mapping role → system_prompt from YAML

    Returns:
        Dictionary mapping role → specialist agent instance
    """
    from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name

    # Use provided shared_rag or create new one
    model_config = model_config or {}
    memory_config = memory_config or {}

    if shared_rag is None:
        shared_rag = SharedRAG(
            persist_directory="./data/shared_rag",
            embedding_model=model_config.get('embedding_model'),
            max_retrieve=memory_config.get('max_knowledge_retrieve'),
            min_quality_score=memory_config.get('shared_rag_min_score'),
        )

    # Default agent IDs
    if agent_ids is None:
        agent_ids = {
            'researcher': 'specialist_1',
            'fact_checker': 'specialist_1',
            'stylist': 'specialist_1'
        }

    # Default to empty prompts dict if not provided
    if agent_prompts is None:
        agent_prompts = {}

    specialists = {}

    # Create researcher agent
    researcher_collection = generate_reasoning_collection_name('researcher', agent_ids['researcher'])
    researcher_memory = ReasoningMemory(
        collection_name=researcher_collection,
        persist_directory=reasoning_dir,
        embedding_model=model_config.get('embedding_model'),
        max_retrieve=memory_config.get('max_reasoning_retrieve'),
    )
    specialists['researcher'] = ResearcherAgent(
        role='researcher',
        agent_id=f"researcher_{agent_ids['researcher']}",
        reasoning_memory=researcher_memory,
        shared_rag=shared_rag,
        system_prompt=agent_prompts.get('researcher'),
        llm_config=model_config
    )

    # Create fact-checker agent
    fact_checker_collection = generate_reasoning_collection_name('fact_checker', agent_ids['fact_checker'])
    fact_checker_memory = ReasoningMemory(
        collection_name=fact_checker_collection,
        persist_directory=reasoning_dir,
        embedding_model=model_config.get('embedding_model'),
        max_retrieve=memory_config.get('max_reasoning_retrieve'),
    )
    specialists['fact_checker'] = FactCheckerAgent(
        role='fact_checker',
        agent_id=f"fact_checker_{agent_ids['fact_checker']}",
        reasoning_memory=fact_checker_memory,
        shared_rag=shared_rag,
        system_prompt=agent_prompts.get('fact_checker'),
        llm_config=model_config
    )

    # Create stylist agent
    stylist_collection = generate_reasoning_collection_name('stylist', agent_ids['stylist'])
    stylist_memory = ReasoningMemory(
        collection_name=stylist_collection,
        persist_directory=reasoning_dir,
        embedding_model=model_config.get('embedding_model'),
        max_retrieve=memory_config.get('max_reasoning_retrieve'),
    )
    specialists['stylist'] = StylistAgent(
        role='stylist',
        agent_id=f"stylist_{agent_ids['stylist']}",
        reasoning_memory=stylist_memory,
        shared_rag=shared_rag,
        system_prompt=agent_prompts.get('stylist'),
        llm_config=model_config
    )

    return specialists
