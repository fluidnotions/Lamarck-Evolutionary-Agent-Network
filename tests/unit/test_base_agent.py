"""Unit tests for base agent module."""

import pytest
from unittest.mock import Mock, patch

from src.agents.base import BaseAgent, ValidationAgent
from src.graph.state import ValidationResult, ValidationState, create_initial_state


class MockAgent(BaseAgent):
    """Mock agent for testing BaseAgent."""

    def __init__(self, name="test_agent", raise_error=False):
        super().__init__(
            name=name,
            description="Test agent",
            capabilities=["test_capability"],
        )
        self.raise_error = raise_error
        self.executed = False

    def _execute(self, state: ValidationState) -> ValidationResult:
        """Mock execute implementation."""
        self.executed = True

        if self.raise_error:
            raise ValueError("Test error")

        return ValidationResult(
            validator_name=self.name,
            status="passed",
            confidence=0.9,
            findings=["Test finding"],
        )


class TestBaseAgent:
    """Tests for BaseAgent class."""

    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = MockAgent(name="test_agent")

        assert agent.name == "test_agent"
        assert agent.description == "Test agent"
        assert agent.capabilities == ["test_capability"]
        assert agent.metadata == {}

    def test_agent_initialization_with_metadata(self):
        """Test agent initialization with metadata."""
        metadata = {"version": "1.0", "author": "test"}
        agent = MockAgent(name="test")
        agent.metadata = metadata

        assert agent.metadata == metadata

    def test_execute_success(self):
        """Test successful agent execution."""
        agent = MockAgent(name="validator1")
        state = create_initial_state({"key": "value"}, {"type": "test"})

        new_state = agent.execute(state)

        assert agent.executed is True
        assert len(new_state["validation_results"]) == 1
        assert new_state["validation_results"][0].validator_name == "validator1"
        assert new_state["validation_results"][0].status == "passed"

    def test_execute_handles_errors(self):
        """Test that execute handles errors gracefully."""
        agent = MockAgent(name="validator1", raise_error=True)
        state = create_initial_state({"key": "value"}, {"type": "test"})

        new_state = agent.execute(state)

        assert len(new_state["errors"]) == 1
        assert new_state["errors"][0].validator_name == "validator1"
        assert new_state["errors"][0].error_type == "ValueError"
        assert "Test error" in new_state["errors"][0].message

    def test_execute_adds_execution_time(self):
        """Test that execute adds execution time to result."""
        agent = MockAgent(name="validator1")
        state = create_initial_state({"key": "value"}, {"type": "test"})

        new_state = agent.execute(state)

        result = new_state["validation_results"][0]
        assert result.execution_time > 0

    def test_validate_state_success(self):
        """Test state validation with valid state."""
        agent = MockAgent()
        state = create_initial_state({"key": "value"}, {"type": "test"})

        # Should not raise
        agent._validate_state(state)

    def test_validate_state_missing_fields(self):
        """Test state validation with missing fields."""
        agent = MockAgent()
        state: ValidationState = {"input_data": {}}  # type: ignore

        with pytest.raises(ValueError, match="Missing required state field"):
            agent._validate_state(state)

    def test_is_recoverable_error(self):
        """Test error recoverability detection."""
        agent = MockAgent()

        # Non-recoverable
        assert agent._is_recoverable_error(ValueError()) is False
        assert agent._is_recoverable_error(TypeError()) is False
        assert agent._is_recoverable_error(KeyError()) is False

        # Recoverable
        assert agent._is_recoverable_error(TimeoutError()) is True
        assert agent._is_recoverable_error(ConnectionError()) is True

    def test_get_info(self):
        """Test getting agent info."""
        agent = MockAgent(name="test_agent")
        info = agent.get_info()

        assert info["name"] == "test_agent"
        assert info["description"] == "Test agent"
        assert info["capabilities"] == ["test_capability"]
        assert "metadata" in info

    def test_supports_capability(self):
        """Test capability support check."""
        agent = MockAgent(name="test_agent")

        assert agent.supports_capability("test_capability") is True
        assert agent.supports_capability("other_capability") is False

    def test_repr(self):
        """Test string representation."""
        agent = MockAgent(name="test_agent")
        repr_str = repr(agent)

        assert "MockAgent" in repr_str
        assert "test_agent" in repr_str


class MockValidationAgent(ValidationAgent):
    """Mock validation agent for testing ValidationAgent."""

    def _execute(self, state: ValidationState) -> ValidationResult:
        """Mock execute implementation."""
        return ValidationResult(
            validator_name=self.name,
            status="passed",
            confidence=0.9,
        )


class TestValidationAgent:
    """Tests for ValidationAgent class."""

    def test_validation_agent_initialization(self):
        """Test ValidationAgent initialization."""
        agent = MockValidationAgent(
            name="test_validator",
            description="Test validation agent",
            min_confidence=0.8,
        )

        assert agent.name == "test_validator"
        assert agent.min_confidence == 0.8

    def test_calculate_confidence_no_findings(self):
        """Test confidence calculation with no findings."""
        agent = MockValidationAgent(
            name="test", description="Test", min_confidence=0.5
        )

        confidence = agent.calculate_confidence([])

        assert confidence == 1.0

    def test_calculate_confidence_with_findings(self):
        """Test confidence calculation with findings."""
        agent = MockValidationAgent(
            name="test", description="Test", min_confidence=0.5
        )

        findings = ["Finding 1", "Finding 2"]
        confidence = agent.calculate_confidence(findings)

        # Confidence reduces with findings
        assert confidence < 1.0
        assert confidence >= 0.0

    def test_calculate_confidence_with_severity(self):
        """Test confidence calculation with severity scores."""
        agent = MockValidationAgent(
            name="test", description="Test", min_confidence=0.5
        )

        findings = ["Critical", "Minor"]
        severity_scores = [0.9, 0.2]  # Critical has high severity

        confidence = agent.calculate_confidence(findings, severity_scores)

        # High severity should reduce confidence significantly
        expected = 1.0 - ((0.9 + 0.2) / 2)
        assert abs(confidence - expected) < 0.01

    def test_calculate_confidence_severity_mismatch(self):
        """Test that mismatched severity raises error."""
        agent = MockValidationAgent(
            name="test", description="Test", min_confidence=0.5
        )

        findings = ["Finding 1", "Finding 2"]
        severity_scores = [0.9]  # Only one score for two findings

        with pytest.raises(ValueError, match="severity_scores must match"):
            agent.calculate_confidence(findings, severity_scores)

    def test_categorize_findings(self):
        """Test finding categorization."""
        agent = MockValidationAgent(
            name="test", description="Test", min_confidence=0.5
        )

        findings = [
            "Critical error in module",
            "Warning: potential issue",
            "Informational message",
            "Must fix this immediately",
            "Should consider improvement",
        ]

        categorized = agent.categorize_findings(findings)

        assert len(categorized["critical"]) == 2  # error, must
        assert len(categorized["warning"]) == 2  # warning, should
        assert len(categorized["info"]) == 1  # info message

    def test_categorize_findings_empty(self):
        """Test categorization with empty findings."""
        agent = MockValidationAgent(
            name="test", description="Test", min_confidence=0.5
        )

        categorized = agent.categorize_findings([])

        assert categorized["critical"] == []
        assert categorized["warning"] == []
        assert categorized["info"] == []
