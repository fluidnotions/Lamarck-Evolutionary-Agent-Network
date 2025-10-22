"""
Updated BaseAgent for reasoning pattern architecture.

**NEW**: Uses ReasoningMemory for cognitive patterns and SharedRAG for domain knowledge.
This implements the 8-step reasoning cycle from the architecture refactor.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from datetime import datetime
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Import new memory classes
try:
    from lean.reasoning_memory import ReasoningMemory
    from lean.shared_rag import SharedRAG
except ImportError:
    # Fallback for development
    print("[Warning] ReasoningMemory/SharedRAG not found, using placeholders")

    class ReasoningMemory:
        def retrieve_similar(self, query: str, k: int = 5) -> List[Dict]:
            return []

        def store_reasoning_pattern(self, *args, **kwargs) -> str:
            return ""

    class SharedRAG:
        def retrieve(self, query: str, k: int = 3) -> List[Dict]:
            return []

        def store_if_high_quality(self, *args, **kwargs) -> Optional[str]:
            return None


class BaseAgent(ABC):
    """Base agent with reasoning pattern memory and shared knowledge base.

    **Architecture**:
    - Layer 1 (Prompts): Fixed, adds <think>/<final> requirement
    - Layer 2 (Shared RAG): Domain knowledge via shared_rag
    - Layer 3 (Reasoning): Cognitive patterns via reasoning_memory
    """

    def __init__(
        self,
        role: str,
        agent_id: str,
        reasoning_memory: ReasoningMemory,
        shared_rag: SharedRAG,
        parent_ids: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ):
        """Initialize agent with reasoning pattern memory.

        Args:
            role: Agent role (intro, body, conclusion)
            agent_id: Unique identifier
            reasoning_memory: Reasoning pattern storage (Layer 3)
            shared_rag: Shared knowledge base (Layer 2)
            parent_ids: Parent agent IDs (for lineage tracking)
            system_prompt: Base system prompt from YAML config (Layer 1)
        """
        self.role = role
        self.agent_id = agent_id
        self.reasoning_memory = reasoning_memory
        self.shared_rag = shared_rag
        self.parent_ids = parent_ids or []
        self.system_prompt = system_prompt  # Store YAML prompt

        # Initialize LLM based on provider
        self.llm = self._initialize_llm()

        # Fitness tracking (output quality)
        self.fitness_history: List[float] = []
        self.domain_fitness: Dict[str, List[float]] = {}
        self.task_count = 0

        # Pending storage
        self.pending_reasoning: Optional[Dict] = None
        self.pending_output: Optional[Dict] = None

    def _initialize_llm(self):
        """Initialize LLM based on LLM_PROVIDER environment variable.

        Returns:
            Initialized LLM client (ChatAnthropic or ChatOpenAI)
        """
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        temperature = float(os.getenv("BASE_TEMPERATURE", "0.7"))

        if provider == "openai":
            model_name = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif provider == "anthropic":
            model_name = os.getenv("MODEL_NAME", "claude-3-5-sonnet-20241022")
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. Use 'anthropic' or 'openai'.")

    def generate_with_reasoning(
        self,
        topic: str,
        reasoning_patterns: List[Dict],
        domain_knowledge: List[Dict],
        reasoning_context: str = "",
        additional_context: str = ""
    ) -> Dict[str, str]:
        """Generate content with externalized reasoning.

        **STEP 5 of 8-step cycle**: Generate with <think> and <final> tags.

        Args:
            topic: The task/topic
            reasoning_patterns: Retrieved reasoning patterns (Step 2)
            domain_knowledge: Retrieved domain facts (Step 3)
            reasoning_context: Reasoning traces from other agents (Step 4)
            additional_context: Any additional context

        Returns:
            Dict with 'thinking' (<think> content) and 'output' (<final> content)
        """
        # Format reasoning patterns
        reasoning_examples = self._format_reasoning_patterns(reasoning_patterns)

        # Format domain knowledge
        knowledge_context = self._format_knowledge(domain_knowledge)

        # Build prompt with <think>/<final> requirement
        prompt = self._build_prompt_with_reasoning(
            topic=topic,
            reasoning_examples=reasoning_examples,
            knowledge_context=knowledge_context,
            reasoning_context=reasoning_context,
            additional_context=additional_context
        )

        # Generate response
        response = self.llm.invoke(prompt)
        response_text = response.content

        # Parse <think> and <final> sections
        thinking, output = self._parse_response(response_text)

        return {
            'thinking': thinking,
            'output': output,
            'raw_response': response_text
        }

    def _format_reasoning_patterns(self, patterns: List[Dict]) -> str:
        """Format retrieved reasoning patterns for prompt.

        Args:
            patterns: List of reasoning pattern dicts

        Returns:
            Formatted string
        """
        if not patterns:
            return "No previous reasoning patterns available."

        formatted = []
        for i, pattern in enumerate(patterns[:3], 1):  # Top 3
            tactic = pattern.get('tactic', 'No tactic')
            score = pattern.get('score', 0.0)
            reasoning = pattern.get('reasoning', '')[:200]  # Truncate

            formatted.append(
                f"Pattern {i} (score: {score:.1f}):\n"
                f"Approach: {tactic}\n"
                f"Reasoning: {reasoning}..."
            )

        return "\n\n".join(formatted)

    def _format_knowledge(self, knowledge: List[Dict]) -> str:
        """Format domain knowledge for prompt.

        Args:
            knowledge: List of knowledge dicts from shared RAG

        Returns:
            Formatted string
        """
        if not knowledge:
            return "No domain knowledge available."

        formatted = []
        for item in knowledge:
            content = item.get('content', '')[:300]  # Truncate
            source = item.get('source', 'unknown')
            formatted.append(f"[{source}] {content}")

        return "\n\n".join(formatted)

    def _build_prompt_with_reasoning(
        self,
        topic: str,
        reasoning_examples: str,
        knowledge_context: str,
        reasoning_context: str,
        additional_context: str
    ) -> str:
        """Build prompt that requires <think> and <final> tags.

        **CRITICAL**: This is Layer 1 (fixed prompt) + requirement for externalized reasoning.

        Args:
            topic: Task description
            reasoning_examples: Formatted reasoning patterns
            knowledge_context: Formatted domain knowledge
            reasoning_context: Reasoning traces from other agents
            additional_context: Any additional context

        Returns:
            Complete prompt string
        """
        # Base role instruction (Layer 1: Fixed)
        role_instruction = self._get_role_instruction()

        # Construct full prompt
        prompt = f"""{role_instruction}

TOPIC: {topic}

PREVIOUS REASONING PATTERNS (how you/your parents approached similar tasks):
{reasoning_examples}

DOMAIN KNOWLEDGE (what you need to know):
{knowledge_context}"""

        if reasoning_context:
            prompt += f"""

REASONING FROM OTHER AGENTS (how they approached their tasks):
{reasoning_context}"""

        if additional_context:
            prompt += f"""

ADDITIONAL CONTEXT:
{additional_context}"""

        prompt += """

INSTRUCTIONS:
1. First, think through your approach using <think> tags
2. Then, provide your final output using <final> tags

Example format:
<think>
My approach for this task:
1. [Step 1]
2. [Step 2]
3. [Step 3]

Based on past patterns, I should...
</think>

<final>
[Your actual output here]
</final>

Now complete the task:"""

        return prompt

    def _get_role_instruction(self) -> str:
        """Get role-specific instruction (Layer 1: Fixed).

        Returns from YAML config if available, otherwise falls back to default.

        Returns:
            Role instruction string
        """
        if self.system_prompt:
            return self.system_prompt

        # Fallback to default generic prompt if no YAML prompt provided
        return f"You are a {self.role} agent. Generate high-quality content for your assigned role."

    def _parse_response(self, response_text: str) -> Tuple[str, str]:
        """Parse <think> and <final> sections from LLM response.

        Args:
            response_text: Raw LLM response

        Returns:
            Tuple of (thinking, output)
        """
        # Extract <think> content
        think_pattern = r'<think>(.*?)</think>'
        think_match = re.search(think_pattern, response_text, re.DOTALL | re.IGNORECASE)
        thinking = think_match.group(1).strip() if think_match else ""

        # Extract <final> content
        final_pattern = r'<final>(.*?)</final>'
        final_match = re.search(final_pattern, response_text, re.DOTALL | re.IGNORECASE)
        output = final_match.group(1).strip() if final_match else ""

        # Fallback: if no tags found, treat entire response as output
        if not thinking and not output:
            print(f"[Warning] No <think>/<final> tags found in response for {self.role}")
            thinking = "No reasoning provided"
            output = response_text

        return thinking, output

    def prepare_reasoning_storage(
        self,
        thinking: str,
        output: str,
        topic: str,
        domain: str,
        generation: int,
        context_sources: List[str]
    ):
        """Prepare reasoning pattern for storage (called before evaluation).

        **STEP 7 (part 1)**: Prepare reasoning for storage, but don't store yet.

        Args:
            thinking: The <think> content
            output: The <final> content
            topic: Topic/task
            domain: Domain category
            generation: Generation number
            context_sources: Context sources used
        """
        # Extract tactic from thinking (first line or first 100 chars)
        tactic = self._extract_tactic(thinking)

        self.pending_reasoning = {
            'reasoning': thinking,
            'tactic': tactic,
            'situation': f"writing {self.role} for {domain} topic",
            'metadata': {
                'topic': topic,
                'domain': domain,
                'generation': generation,
                'context_sources': ','.join(context_sources),  # Convert list to comma-separated string
                'agent_id': self.agent_id,
                'role': self.role
            }
        }

        self.pending_output = {
            'output': output,
            'metadata': {
                'topic': topic,
                'domain': domain,
                'role': self.role,
                'agent_id': self.agent_id
            }
        }

    def store_reasoning_and_output(self, score: float):
        """Store reasoning pattern and optionally output after evaluation.

        **STEP 7 (part 2)**: Store after receiving score.

        Args:
            score: Quality score from evaluator
        """
        if self.pending_reasoning is None:
            print(f"[Warning] No pending reasoning to store for {self.agent_id}")
            return

        # Store reasoning pattern (always, no threshold)
        reasoning_id = self.reasoning_memory.store_reasoning_pattern(
            reasoning=self.pending_reasoning['reasoning'],
            score=score,
            situation=self.pending_reasoning['situation'],
            tactic=self.pending_reasoning['tactic'],
            metadata=self.pending_reasoning['metadata']
        )

        # Store output in shared RAG (only if high quality)
        if self.pending_output and score >= 8.0:
            self.shared_rag.store_if_high_quality(
                content=self.pending_output['output'],
                score=score,
                metadata=self.pending_output['metadata']
            )

        # Clear pending
        self.pending_reasoning = None
        self.pending_output = None

    def _extract_tactic(self, thinking: str) -> str:
        """Extract tactical approach from reasoning.

        Args:
            thinking: The <think> content

        Returns:
            Brief tactic description
        """
        lines = thinking.strip().split('\n')
        first_line = lines[0] if lines else thinking[:100]
        return first_line[:100].strip()

    # Fitness tracking methods
    def record_fitness(self, score: float, domain: str = ""):
        """Record fitness score (output quality).

        Args:
            score: Evaluation score (0-10)
            domain: Topic category
        """
        self.fitness_history.append(score)
        self.task_count += 1

        if domain:
            if domain not in self.domain_fitness:
                self.domain_fitness[domain] = []
            self.domain_fitness[domain].append(score)

    def avg_fitness(self, recent_n: Optional[int] = None) -> float:
        """Calculate average fitness.

        Args:
            recent_n: If provided, average only last N scores

        Returns:
            Average fitness score
        """
        if not self.fitness_history:
            return 0.0

        if recent_n:
            recent_scores = self.fitness_history[-recent_n:]
            return sum(recent_scores) / len(recent_scores)

        return sum(self.fitness_history) / len(self.fitness_history)

    def get_stats(self) -> Dict:
        """Get agent statistics.

        Returns:
            Dict with agent stats
        """
        reasoning_stats = self.reasoning_memory.get_stats()

        return {
            'agent_id': self.agent_id,
            'role': self.role,
            'task_count': self.task_count,
            'avg_fitness': self.avg_fitness(),
            'reasoning_patterns': reasoning_stats['total_patterns'],
            'inherited_patterns': reasoning_stats['inherited_patterns'],
            'personal_patterns': reasoning_stats['personal_patterns'],
            'parent_ids': self.parent_ids
        }


class IntroAgent(BaseAgent):
    """Intro agent with reasoning pattern architecture."""
    pass  # Uses base class _get_role_instruction() with YAML prompt


class BodyAgent(BaseAgent):
    """Body agent with reasoning pattern architecture."""
    pass  # Uses base class _get_role_instruction() with YAML prompt


class ConclusionAgent(BaseAgent):
    """Conclusion agent with reasoning pattern architecture."""
    pass  # Uses base class _get_role_instruction() with YAML prompt


def create_agents(
    reasoning_dir: str = "./data/reasoning",
    shared_rag_dir: str = "./data/shared_rag",
    agent_ids: Optional[Dict[str, str]] = None,
    agent_prompts: Optional[Dict[str, str]] = None
) -> Dict[str, BaseAgent]:
    """Create agents with reasoning pattern architecture.

    Args:
        reasoning_dir: Directory for per-agent reasoning patterns
        shared_rag_dir: Directory for shared knowledge base
        agent_ids: Optional dict mapping role → agent_id (e.g., {'intro': 'agent_1'})
        agent_prompts: Optional dict mapping role → system_prompt from YAML

    Returns:
        Dictionary mapping role → agent instance

    Example:
        agents = create_agents(
            reasoning_dir="./data/reasoning",
            shared_rag_dir="./data/shared_rag",
            agent_ids={'intro': 'agent_1', 'body': 'agent_1', 'conclusion': 'agent_1'},
            agent_prompts={'intro': 'You are...', 'body': 'You are...', 'conclusion': 'You are...'}
        )
    """
    from lean.reasoning_memory import ReasoningMemory, generate_reasoning_collection_name
    from lean.shared_rag import SharedRAG

    # Create single SharedRAG instance (shared by ALL agents)
    shared_rag = SharedRAG(persist_directory=shared_rag_dir)

    # Default agent IDs
    if agent_ids is None:
        agent_ids = {
            'intro': 'agent_1',
            'body': 'agent_1',
            'conclusion': 'agent_1'
        }

    # Default to empty prompts dict if not provided
    if agent_prompts is None:
        agent_prompts = {}

    agents = {}

    # Create intro agent
    intro_collection = generate_reasoning_collection_name('intro', agent_ids['intro'])
    intro_memory = ReasoningMemory(
        collection_name=intro_collection,
        persist_directory=reasoning_dir
    )
    agents['intro'] = IntroAgent(
        role='intro',
        agent_id=f"intro_{agent_ids['intro']}",
        reasoning_memory=intro_memory,
        shared_rag=shared_rag,
        system_prompt=agent_prompts.get('intro')
    )

    # Create body agent
    body_collection = generate_reasoning_collection_name('body', agent_ids['body'])
    body_memory = ReasoningMemory(
        collection_name=body_collection,
        persist_directory=reasoning_dir
    )
    agents['body'] = BodyAgent(
        role='body',
        agent_id=f"body_{agent_ids['body']}",
        reasoning_memory=body_memory,
        shared_rag=shared_rag,
        system_prompt=agent_prompts.get('body')
    )

    # Create conclusion agent
    conclusion_collection = generate_reasoning_collection_name('conclusion', agent_ids['conclusion'])
    conclusion_memory = ReasoningMemory(
        collection_name=conclusion_collection,
        persist_directory=reasoning_dir
    )
    agents['conclusion'] = ConclusionAgent(
        role='conclusion',
        agent_id=f"conclusion_{agent_ids['conclusion']}",
        reasoning_memory=conclusion_memory,
        shared_rag=shared_rag,
        system_prompt=agent_prompts.get('conclusion')
    )

    return agents
