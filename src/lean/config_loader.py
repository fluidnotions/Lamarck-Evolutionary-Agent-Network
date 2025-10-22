"""
Configuration loader for LEAN experiments.

Loads experiment topics, agent prompts, and documentation from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Topic:
    """A single topic/subject for experimentation."""
    title: str
    keywords: List[str]
    difficulty: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TopicBlock:
    """A block of related topics for transfer learning."""
    name: str
    description: str
    generation_range: List[int]
    documentation_path: Optional[str]
    topics: List[Topic]

    def load_documentation(self) -> Optional[str]:
        """Load the markdown documentation for this block."""
        if not self.documentation_path:
            return None

        doc_path = Path(self.documentation_path)
        if doc_path.exists():
            return doc_path.read_text()
        return None


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str
    population_size: int
    evolution_frequency: int
    total_generations: int
    reasoning_dir: str
    shared_rag_dir: str
    domain: str
    topic_blocks: List[TopicBlock]
    research_config: Dict[str, Any]
    quality_config: Dict[str, Any]

    # Model & Agent Configuration
    model_config: Optional[Dict[str, Any]] = None

    # Memory Configuration
    memory_config: Optional[Dict[str, Any]] = None

    # Evolution Configuration
    evolution_config: Optional[Dict[str, Any]] = None

    # Human-in-the-Loop Configuration
    hitl_config: Optional[Dict[str, Any]] = None

    # Visualization Configuration
    visualization_config: Optional[Dict[str, Any]] = None

    # Logging Configuration
    logging_config: Optional[Dict[str, Any]] = None

    def get_config_value(self, section: str, key: str, default: Any = None, env_var: str = None) -> Any:
        """Get configuration value from YAML with fallback to environment variable.

        Args:
            section: Config section (e.g., 'model', 'memory', 'hitl')
            key: Key within section
            default: Default value if not found
            env_var: Environment variable name for fallback

        Returns:
            Config value, env value, or default
        """
        import os

        # Try to get from YAML config
        config_section = getattr(self, f'{section}_config', None)
        if config_section and isinstance(config_section, dict):
            value = config_section.get(key)
            if value is not None:
                return value

        # Fallback to environment variable
        if env_var:
            return os.getenv(env_var, default)

        return default

    def get_all_topics(self) -> List[str]:
        """Extract all topic titles in order."""
        topics = []
        for block in self.topic_blocks:
            for topic in block.topics:
                topics.append(topic.title)
        return topics

    def get_topic_metadata(self, topic_title: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific topic."""
        for block in self.topic_blocks:
            for topic in block.topics:
                if topic.title == topic_title:
                    return {
                        'keywords': topic.keywords,
                        'difficulty': topic.difficulty,
                        'block': block.name,
                        'block_description': block.description,
                        'metadata': topic.metadata or {}
                    }
        return None

    def get_block_for_generation(self, generation: int) -> Optional[TopicBlock]:
        """Get the topic block for a given generation number."""
        for block in self.topic_blocks:
            start, end = block.generation_range
            if start <= generation <= end:
                return block
        return None


@dataclass
class AgentPromptConfig:
    """Configuration for a single agent's prompts."""
    role: str
    system_prompt: str
    reasoning_focus: Optional[str] = None
    documentation_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def load_documentation(self) -> Optional[str]:
        """Load the markdown documentation for this agent role."""
        if not self.documentation_path:
            return None

        doc_path = Path(self.documentation_path)
        if doc_path.exists():
            return doc_path.read_text()
        return None


class ConfigLoader:
    """Loads LEAN configuration from YAML files."""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)

    def load_experiment_config(self, config_name: str = "default") -> ExperimentConfig:
        """Load experiment configuration from YAML."""
        config_path = self.config_dir / "experiments" / f"{config_name}.yml"

        if not config_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {config_path}")

        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)

        # Parse topic blocks
        topic_blocks = []
        for block_data in data['topic_blocks']:
            topics = [
                Topic(
                    title=t['title'],
                    keywords=t.get('keywords', []),
                    difficulty=t.get('difficulty', 'intermediate'),
                    metadata=t.get('metadata')
                )
                for t in block_data['topics']
            ]

            topic_blocks.append(TopicBlock(
                name=block_data['name'],
                description=block_data['description'],
                generation_range=block_data['generation_range'],
                documentation_path=block_data.get('documentation'),
                topics=topics
            ))

        exp_data = data['experiment']

        return ExperimentConfig(
            name=exp_data['name'],
            description=exp_data['description'],
            population_size=exp_data['population_size'],
            evolution_frequency=exp_data['evolution_frequency'],
            total_generations=exp_data['total_generations'],
            reasoning_dir=exp_data['reasoning_dir'],
            shared_rag_dir=exp_data['shared_rag_dir'],
            domain=exp_data['domain'],
            topic_blocks=topic_blocks,
            research_config=data.get('research', {}),
            quality_config=data.get('quality', {}),
            model_config=data.get('model', {}),
            memory_config=data.get('memory', {}),
            evolution_config=data.get('evolution', {}),
            hitl_config=data.get('hitl', {}),
            visualization_config=data.get('visualization', {}),
            logging_config=data.get('logging', {})
        )

    def load_agent_prompts(self) -> Dict[str, AgentPromptConfig]:
        """Load all agent prompt configurations from YAML."""
        prompts_path = self.config_dir / "prompts" / "agents.yml"

        if not prompts_path.exists():
            raise FileNotFoundError(f"Agent prompts config not found: {prompts_path}")

        with open(prompts_path, 'r') as f:
            data = yaml.safe_load(f)

        # Remove evaluation section - it's metadata, not an agent
        evaluation_config = data.pop('evaluation', None)

        agent_configs = {}
        for role, config_data in data.items():
            agent_configs[role] = AgentPromptConfig(
                role=role,
                system_prompt=config_data['system_prompt'],
                reasoning_focus=config_data.get('reasoning_focus'),
                documentation_path=config_data.get('documentation'),
                metadata={'evaluation': evaluation_config} if role == 'coordinator' and evaluation_config else None
            )

        return agent_configs

    def get_evaluation_criteria(self) -> Optional[Dict[str, List[str]]]:
        """Extract evaluation criteria from agent prompts config."""
        prompts_path = self.config_dir / "prompts" / "agents.yml"

        if not prompts_path.exists():
            return None

        with open(prompts_path, 'r') as f:
            data = yaml.safe_load(f)

        return data.get('evaluation')


def load_config(config_name: str = "default", config_dir: str = "./config") -> tuple[ExperimentConfig, Dict[str, AgentPromptConfig]]:
    """
    Convenience function to load both experiment and agent configs.

    Returns:
        (experiment_config, agent_prompts)
    """
    loader = ConfigLoader(config_dir)
    exp_config = loader.load_experiment_config(config_name)
    agent_prompts = loader.load_agent_prompts()
    return exp_config, agent_prompts


# Example usage
if __name__ == "__main__":
    # Load configuration
    exp_config, agent_prompts = load_config("default")

    print(f"Experiment: {exp_config.name}")
    print(f"Total generations: {exp_config.total_generations}")
    print(f"Topic blocks: {len(exp_config.topic_blocks)}")
    print(f"Total topics: {len(exp_config.get_all_topics())}")
    print()

    # Show first topic from each block
    for block in exp_config.topic_blocks:
        print(f"{block.name} ({block.generation_range[0]}-{block.generation_range[1]}):")
        if block.topics:
            first_topic = block.topics[0]
            print(f"  - {first_topic.title}")
            print(f"    Keywords: {', '.join(first_topic.keywords)}")
            print(f"    Difficulty: {first_topic.difficulty}")
        print()

    # Show agent prompts
    print("Agent Prompts:")
    for role, config in agent_prompts.items():
        print(f"  - {role}: {len(config.system_prompt)} chars")
        if config.reasoning_focus:
            print(f"    Focus: {config.reasoning_focus}")
