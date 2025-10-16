"""Integration tests for complete validation workflows."""
import pytest
from unittest.mock import Mock, patch

from src.graph.workflow import ValidationWorkflow
from src.models.validation_result import ValidationResult


@pytest.fixture
def mock_llm_for_workflow():
    """Create a mock LLM for workflow testing."""
    mock = Mock()
    mock.invoke = Mock(return_value=Mock(content="Mocked response"))
    return mock


class TestValidationWorkflowIntegration:
    """Integration tests for the validation workflow."""

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_runs_all_validators(self, mock_llm_factory, sample_data, sample_config):
        """Test that workflow runs all requested validators."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema", "business", "quality"],
            config=sample_config,
        )

        # Check that result was generated
        assert result is not None
        assert hasattr(result, "validation_results")
        assert len(result.validation_results) > 0

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_with_single_validator(self, mock_llm_factory, sample_data):
        """Test workflow with only schema validator."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema"],
            config={"schema": {"schema": {"type": "object"}}},
        )

        assert result is not None
        assert result.overall_status in ["passed", "failed", "partial"]

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_handles_validation_errors(self, mock_llm_factory, invalid_data, sample_schema):
        """Test that workflow properly handles validation errors."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=invalid_data,
            validators=["schema"],
            config={"schema": {"schema": sample_schema}},
        )

        assert result is not None
        assert result.overall_status in ["failed", "partial"]
        assert result.total_errors > 0

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_calculates_confidence(self, mock_llm_factory, sample_data):
        """Test that workflow calculates overall confidence score."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema", "quality"],
            config={},
        )

        assert result is not None
        assert 0.0 <= result.confidence_score <= 1.0

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_generates_report(self, mock_llm_factory, sample_data):
        """Test that workflow generates a comprehensive report."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema"],
            config={},
        )

        # Test different report formats
        text_report = result.get_report("text")
        assert isinstance(text_report, str)
        assert len(text_report) > 0
        assert "VALIDATION REPORT" in text_report

        markdown_report = result.get_report("markdown")
        assert isinstance(markdown_report, str)
        assert "# Validation Report" in markdown_report

        json_report = result.get_report("json")
        assert isinstance(json_report, str)
        assert "{" in json_report

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_preserves_execution_time(self, mock_llm_factory, sample_data):
        """Test that workflow tracks execution time."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema"],
            config={},
        )

        assert result.execution_time_ms >= 0

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_workflow_with_custom_workflow_id(self, mock_llm_factory, sample_data):
        """Test workflow with custom workflow ID."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()
        custom_id = "test-workflow-123"

        result = workflow.run(
            data=sample_data,
            validators=["schema"],
            config={},
            workflow_id=custom_id,
        )

        assert result is not None


class TestWorkflowRouting:
    """Tests for workflow routing logic."""

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_supervisor_routes_to_validators(self, mock_llm_factory, sample_data):
        """Test that supervisor correctly routes to validators."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        # Run workflow and check routing
        result = workflow.run(
            data=sample_data,
            validators=["schema", "quality"],
            config={},
        )

        # Should have results from both validators
        assert result is not None
        validator_names = [v.validator_name for v in result.validation_results]
        assert "schema_validator" in validator_names
        assert "data_quality" in validator_names

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_aggregator_collects_all_results(self, mock_llm_factory, sample_data):
        """Test that aggregator collects results from all validators."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema", "business", "quality"],
            config={},
        )

        # Aggregator should have collected results from all validators
        assert result is not None
        assert len(result.validation_results) >= 3


class TestWorkflowStateManagement:
    """Tests for workflow state management."""

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_state_tracks_completed_validators(self, mock_llm_factory, sample_data):
        """Test that state properly tracks completed validators."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=sample_data,
            validators=["schema", "quality"],
            config={},
        )

        # All requested validators should have completed
        assert result is not None
        assert len(result.validation_results) == 2

    @patch('src.agents.base.BaseAgent._create_default_llm')
    def test_state_accumulates_errors(self, mock_llm_factory, invalid_data, sample_schema):
        """Test that state accumulates errors across validators."""
        mock_llm_factory.return_value = Mock()

        workflow = ValidationWorkflow()

        result = workflow.run(
            data=invalid_data,
            validators=["schema", "quality"],
            config={
                "schema": {"schema": sample_schema},
                "data_quality": {"required_fields": ["user", "order"]},
            },
        )

        # Should have accumulated errors
        assert result is not None
        assert result.total_errors > 0
