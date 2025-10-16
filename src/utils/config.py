"""Configuration management for HVAS-Mini.

This module provides configuration loading, validation, and access for the
validation system. It supports environment variables, default values, and
runtime configuration updates.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Configuration for Language Model settings.

    Attributes:
        provider: LLM provider (openai or anthropic)
        model: Model identifier
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
        api_key: API key for authentication (loaded from env)
    """
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        """Load API key from environment if not provided."""
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
    """
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class TimeoutConfig:
    """Configuration for timeout values.

    Attributes:
        agent_execution: Timeout for individual agent execution (seconds)
        llm_request: Timeout for LLM API requests (seconds)
        total_workflow: Timeout for entire validation workflow (seconds)
    """
    agent_execution: float = 300.0  # 5 minutes
    llm_request: float = 60.0  # 1 minute
    total_workflow: float = 1800.0  # 30 minutes


@dataclass
class ValidationConfig:
    """Configuration for validation behavior.

    Attributes:
        min_confidence_threshold: Minimum confidence to pass validation
        enable_parallel_validation: Whether to run validators in parallel
        max_parallel_validators: Maximum number of parallel validators
        fail_fast: Whether to stop on first validation failure
    """
    min_confidence_threshold: float = 0.7
    enable_parallel_validation: bool = True
    max_parallel_validators: int = 5
    fail_fast: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format string
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        log_file: Path to log file
        enable_json: Whether to use JSON format
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_console: bool = True
    enable_file: bool = False
    log_file: str = "hvas-mini.log"
    enable_json: bool = False


@dataclass
class Config:
    """Main configuration class for HVAS-Mini.

    This class aggregates all configuration sections and provides
    validation and access methods.

    Attributes:
        llm: LLM configuration
        retry: Retry configuration
        timeout: Timeout configuration
        validation: Validation configuration
        logging: Logging configuration
        debug: Enable debug mode
        metadata: Additional custom configuration
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate LLM config
        if self.llm.temperature < 0.0 or self.llm.temperature > 1.0:
            raise ValueError("LLM temperature must be between 0.0 and 1.0")

        if self.llm.max_tokens < 1:
            raise ValueError("LLM max_tokens must be positive")

        if not self.llm.api_key:
            raise ValueError(f"API key not found for provider: {self.llm.provider}")

        # Validate retry config
        if self.retry.max_attempts < 1:
            raise ValueError("Retry max_attempts must be at least 1")

        if self.retry.initial_delay < 0:
            raise ValueError("Retry initial_delay must be non-negative")

        # Validate timeout config
        if self.timeout.agent_execution < 0:
            raise ValueError("Timeout agent_execution must be non-negative")

        if self.timeout.llm_request < 0:
            raise ValueError("Timeout llm_request must be non-negative")

        if self.timeout.total_workflow < 0:
            raise ValueError("Timeout total_workflow must be non-negative")

        # Validate validation config
        if not 0.0 <= self.validation.min_confidence_threshold <= 1.0:
            raise ValueError("Validation min_confidence_threshold must be between 0.0 and 1.0")

        if self.validation.max_parallel_validators < 1:
            raise ValueError("Validation max_parallel_validators must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "api_key": "***" if self.llm.api_key else None,
            },
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "initial_delay": self.retry.initial_delay,
                "max_delay": self.retry.max_delay,
                "exponential_base": self.retry.exponential_base,
                "jitter": self.retry.jitter,
            },
            "timeout": {
                "agent_execution": self.timeout.agent_execution,
                "llm_request": self.timeout.llm_request,
                "total_workflow": self.timeout.total_workflow,
            },
            "validation": {
                "min_confidence_threshold": self.validation.min_confidence_threshold,
                "enable_parallel_validation": self.validation.enable_parallel_validation,
                "max_parallel_validators": self.validation.max_parallel_validators,
                "fail_fast": self.validation.fail_fast,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "enable_console": self.logging.enable_console,
                "enable_file": self.logging.enable_file,
                "log_file": self.logging.log_file,
                "enable_json": self.logging.enable_json,
            },
            "debug": self.debug,
            "metadata": self.metadata,
        }


def load_config(
    env_file: Optional[str] = ".env",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    debug: bool = False,
) -> Config:
    """Load configuration from environment and defaults.

    Args:
        env_file: Path to .env file (None to skip loading)
        llm_provider: Override LLM provider
        llm_model: Override LLM model
        debug: Enable debug mode

    Returns:
        Loaded and validated configuration

    Raises:
        ValueError: If configuration is invalid
    """
    # Load environment variables
    if env_file:
        load_dotenv(env_file)

    # Create LLM config
    llm_config = LLMConfig()

    if llm_provider:
        if llm_provider not in ["openai", "anthropic"]:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")
        llm_config.provider = llm_provider

    if llm_model:
        llm_config.model = llm_model

    # Override from environment variables
    if os.getenv("LLM_PROVIDER"):
        provider = os.getenv("LLM_PROVIDER", "")
        if provider in ["openai", "anthropic"]:
            llm_config.provider = provider

    if os.getenv("LLM_MODEL"):
        llm_config.model = os.getenv("LLM_MODEL", llm_config.model)

    if os.getenv("LLM_TEMPERATURE"):
        llm_config.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    if os.getenv("LLM_MAX_TOKENS"):
        llm_config.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # Create retry config
    retry_config = RetryConfig()
    if os.getenv("RETRY_MAX_ATTEMPTS"):
        retry_config.max_attempts = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))

    # Create timeout config
    timeout_config = TimeoutConfig()
    if os.getenv("TIMEOUT_AGENT_EXECUTION"):
        timeout_config.agent_execution = float(os.getenv("TIMEOUT_AGENT_EXECUTION", "300"))

    # Create validation config
    validation_config = ValidationConfig()
    if os.getenv("MIN_CONFIDENCE_THRESHOLD"):
        validation_config.min_confidence_threshold = float(
            os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7")
        )

    # Create logging config
    logging_config = LoggingConfig()
    if os.getenv("LOG_LEVEL"):
        logging_config.level = os.getenv("LOG_LEVEL", "INFO")

    if os.getenv("LOG_FILE"):
        logging_config.log_file = os.getenv("LOG_FILE", "hvas-mini.log")
        logging_config.enable_file = True

    # Create main config
    config = Config(
        llm=llm_config,
        retry=retry_config,
        timeout=timeout_config,
        validation=validation_config,
        logging=logging_config,
        debug=debug or os.getenv("DEBUG", "").lower() == "true",
    )

    # Validate configuration
    config.validate()

    return config


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Returns:
        Global configuration instance

    Raises:
        RuntimeError: If configuration has not been initialized
    """
    global _config
    if _config is None:
        raise RuntimeError("Configuration not initialized. Call load_config() first.")
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance.

    Args:
        config: Configuration instance to set as global
    """
    global _config
    config.validate()
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance.

    Useful for testing.
    """
    global _config
    _config = None
