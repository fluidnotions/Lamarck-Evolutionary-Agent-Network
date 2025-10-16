"""Configuration management utilities."""
import os
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    provider: str = Field(default="openai", description="LLM provider: openai or anthropic")
    model: str = Field(default="gpt-4", description="Model name")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum tokens to generate")
    timeout: float = Field(default=30.0, gt=0, description="Request timeout in seconds")


class ValidationConfig(BaseModel):
    """Configuration for validation behavior."""

    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    timeout_seconds: float = Field(default=30.0, gt=0, description="Validation timeout")
    parallel_validators: bool = Field(default=True, description="Run validators in parallel")
    fail_fast: bool = Field(default=False, description="Stop on first failure")
    min_confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence to pass"
    )


class Config(BaseModel):
    """Main application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    log_level: str = Field(default="INFO", description="Logging level")
    langchain_tracing: bool = Field(default=False, description="Enable LangChain tracing")

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.

        Returns:
            Config instance populated from environment
        """
        return cls(
            llm=LLMConfig(
                provider=os.getenv("LLM_PROVIDER", "openai"),
                model=os.getenv("LLM_MODEL", "gpt-4"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
                timeout=float(os.getenv("LLM_TIMEOUT", "30.0")),
            ),
            validation=ValidationConfig(
                max_retries=int(os.getenv("MAX_RETRIES", "3")),
                timeout_seconds=float(os.getenv("TIMEOUT_SECONDS", "30.0")),
                parallel_validators=os.getenv("PARALLEL_VALIDATORS", "true").lower() == "true",
                fail_fast=os.getenv("FAIL_FAST", "false").lower() == "true",
                min_confidence_threshold=float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5")),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            langchain_tracing=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return self.model_dump()


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance.

    Args:
        config: Configuration to set
    """
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to default."""
    global _config
    _config = None
