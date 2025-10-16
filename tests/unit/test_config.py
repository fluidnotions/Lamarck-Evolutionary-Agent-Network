"""Unit tests for configuration module."""

import os
import pytest
from unittest.mock import patch

from src.utils.config import (
    Config,
    LLMConfig,
    LoggingConfig,
    RetryConfig,
    TimeoutConfig,
    ValidationConfig,
    get_config,
    load_config,
    reset_config,
    set_config,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_llm_config_defaults(self):
        """Test LLM config default values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = LLMConfig()

            assert config.provider == "openai"
            assert config.model == "gpt-4"
            assert config.temperature == 0.7
            assert config.max_tokens == 2000
            assert config.api_key == "test_key"

    def test_llm_config_anthropic(self):
        """Test LLM config for Anthropic."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):
            config = LLMConfig(provider="anthropic")

            assert config.provider == "anthropic"
            assert config.api_key == "test_key"


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_retry_config_defaults(self):
        """Test retry config default values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True


class TestTimeoutConfig:
    """Tests for TimeoutConfig."""

    def test_timeout_config_defaults(self):
        """Test timeout config default values."""
        config = TimeoutConfig()

        assert config.agent_execution == 300.0
        assert config.llm_request == 60.0
        assert config.total_workflow == 1800.0


class TestValidationConfig:
    """Tests for ValidationConfig."""

    def test_validation_config_defaults(self):
        """Test validation config default values."""
        config = ValidationConfig()

        assert config.min_confidence_threshold == 0.7
        assert config.enable_parallel_validation is True
        assert config.max_parallel_validators == 5
        assert config.fail_fast is False


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_logging_config_defaults(self):
        """Test logging config default values."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.enable_console is True
        assert config.enable_file is False
        assert config.enable_json is False


class TestConfig:
    """Tests for main Config class."""

    def test_config_initialization(self):
        """Test basic config initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()

            assert isinstance(config.llm, LLMConfig)
            assert isinstance(config.retry, RetryConfig)
            assert isinstance(config.timeout, TimeoutConfig)
            assert isinstance(config.validation, ValidationConfig)
            assert isinstance(config.logging, LoggingConfig)
            assert config.debug is False
            assert config.metadata == {}

    def test_config_validate_success(self):
        """Test config validation with valid values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()

            # Should not raise
            config.validate()

    def test_config_validate_invalid_temperature(self):
        """Test config validation with invalid temperature."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            config.llm.temperature = 1.5

            with pytest.raises(ValueError, match="temperature must be between"):
                config.validate()

    def test_config_validate_missing_api_key(self):
        """Test config validation with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            with pytest.raises(ValueError, match="API key not found"):
                config.validate()

    def test_config_validate_invalid_retry_attempts(self):
        """Test config validation with invalid retry attempts."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            config.retry.max_attempts = 0

            with pytest.raises(ValueError, match="max_attempts must be at least 1"):
                config.validate()

    def test_config_validate_invalid_timeout(self):
        """Test config validation with negative timeout."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            config.timeout.agent_execution = -1.0

            with pytest.raises(ValueError, match="must be non-negative"):
                config.validate()

    def test_config_validate_invalid_confidence(self):
        """Test config validation with invalid confidence threshold."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            config.validation.min_confidence_threshold = 1.5

            with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
                config.validate()

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            config_dict = config.to_dict()

            assert "llm" in config_dict
            assert "retry" in config_dict
            assert "timeout" in config_dict
            assert "validation" in config_dict
            assert "logging" in config_dict
            assert config_dict["llm"]["api_key"] == "***"  # Masked


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_defaults(self):
        """Test loading config with defaults."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = load_config(env_file=None)

            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-4"

    def test_load_config_with_overrides(self):
        """Test loading config with overrides."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = load_config(
                env_file=None,
                llm_provider="openai",
                llm_model="gpt-3.5-turbo",
                debug=True,
            )

            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-3.5-turbo"
            assert config.debug is True

    def test_load_config_from_env_vars(self):
        """Test loading config from environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "test_key",
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-3.5-turbo",
            "LLM_TEMPERATURE": "0.5",
            "LLM_MAX_TOKENS": "1000",
            "RETRY_MAX_ATTEMPTS": "5",
            "TIMEOUT_AGENT_EXECUTION": "600",
            "MIN_CONFIDENCE_THRESHOLD": "0.8",
            "LOG_LEVEL": "DEBUG",
            "DEBUG": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = load_config(env_file=None)

            assert config.llm.provider == "openai"
            assert config.llm.model == "gpt-3.5-turbo"
            assert config.llm.temperature == 0.5
            assert config.llm.max_tokens == 1000
            assert config.retry.max_attempts == 5
            assert config.timeout.agent_execution == 600.0
            assert config.validation.min_confidence_threshold == 0.8
            assert config.logging.level == "DEBUG"
            assert config.debug is True

    def test_load_config_invalid_provider(self):
        """Test loading config with invalid provider."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            with pytest.raises(ValueError, match="Invalid LLM provider"):
                load_config(env_file=None, llm_provider="invalid")


class TestConfigGlobal:
    """Tests for global config functions."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_set_and_get_config(self):
        """Test setting and getting global config."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            set_config(config)

            retrieved_config = get_config()

            assert retrieved_config is config

    def test_get_config_not_initialized(self):
        """Test getting config before initialization."""
        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            get_config()

    def test_set_config_validates(self):
        """Test that set_config validates configuration."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            config.llm.temperature = 2.0  # Invalid

            with pytest.raises(ValueError):
                set_config(config)

    def test_reset_config(self):
        """Test resetting global config."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            config = Config()
            set_config(config)

            reset_config()

            with pytest.raises(RuntimeError):
                get_config()
