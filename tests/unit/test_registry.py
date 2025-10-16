"""Unit tests for validator registry."""

import pytest

from src.agents.base import BaseAgent
from src.agents.registry import ValidatorRegistry
from src.graph.state import ValidationState


class MockValidator(BaseAgent):
    """Mock validator for testing."""

    def __init__(self, name: str, capabilities: list[str]):
        super().__init__(name=name, description=f"Mock {name}", capabilities=capabilities)

    def execute(self, state: ValidationState) -> ValidationState:
        return state


class TestValidatorRegistry:
    """Test validator registry functionality."""

    def test_registry_initialization(self) -> None:
        """Test registry initializes empty."""
        registry = ValidatorRegistry()
        assert len(registry) == 0
        assert registry.get_all_validators() == []

    def test_register_validator(self) -> None:
        """Test registering a validator."""
        registry = ValidatorRegistry()
        validator = MockValidator("test_validator", ["test_capability"])

        registry.register(validator)

        assert len(registry) == 1
        assert registry.is_registered("test_validator")
        assert validator in registry.get_all_validators()

    def test_register_multiple_validators(self) -> None:
        """Test registering multiple validators."""
        registry = ValidatorRegistry()
        v1 = MockValidator("validator1", ["cap1"])
        v2 = MockValidator("validator2", ["cap2"])

        registry.register(v1)
        registry.register(v2)

        assert len(registry) == 2
        assert registry.is_registered("validator1")
        assert registry.is_registered("validator2")

    def test_register_overwrites_existing(self) -> None:
        """Test that registering same name overwrites."""
        registry = ValidatorRegistry()
        v1 = MockValidator("validator", ["cap1"])
        v2 = MockValidator("validator", ["cap2"])

        registry.register(v1)
        registry.register(v2)

        assert len(registry) == 1
        retrieved = registry.get_validator("validator")
        assert retrieved is v2

    def test_unregister_validator(self) -> None:
        """Test unregistering a validator."""
        registry = ValidatorRegistry()
        validator = MockValidator("test_validator", ["cap"])

        registry.register(validator)
        assert registry.is_registered("test_validator")

        registry.unregister("test_validator")
        assert not registry.is_registered("test_validator")
        assert len(registry) == 0

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent validator doesn't error."""
        registry = ValidatorRegistry()
        registry.unregister("nonexistent")  # Should not raise

    def test_get_validator(self) -> None:
        """Test retrieving validator by name."""
        registry = ValidatorRegistry()
        validator = MockValidator("test", ["cap"])

        registry.register(validator)
        retrieved = registry.get_validator("test")

        assert retrieved is validator

    def test_get_nonexistent_validator(self) -> None:
        """Test retrieving nonexistent validator returns None."""
        registry = ValidatorRegistry()
        assert registry.get_validator("nonexistent") is None

    def test_get_validators_by_capability(self) -> None:
        """Test finding validators by capability."""
        registry = ValidatorRegistry()
        v1 = MockValidator("v1", ["schema", "json"])
        v2 = MockValidator("v2", ["schema", "xml"])
        v3 = MockValidator("v3", ["business"])

        registry.register(v1)
        registry.register(v2)
        registry.register(v3)

        schema_validators = registry.get_validators_by_capability("schema")
        assert len(schema_validators) == 2
        assert v1 in schema_validators
        assert v2 in schema_validators

        business_validators = registry.get_validators_by_capability("business")
        assert len(business_validators) == 1
        assert v3 in business_validators

    def test_get_validators_by_nonexistent_capability(self) -> None:
        """Test querying nonexistent capability returns empty list."""
        registry = ValidatorRegistry()
        v = MockValidator("v", ["cap1"])
        registry.register(v)

        result = registry.get_validators_by_capability("nonexistent")
        assert result == []

    def test_capabilities_summary(self) -> None:
        """Test getting capabilities summary."""
        registry = ValidatorRegistry()
        v1 = MockValidator("v1", ["cap1", "cap2"])
        v2 = MockValidator("v2", ["cap2", "cap3"])

        registry.register(v1)
        registry.register(v2)

        summary = registry.get_capabilities_summary()

        assert "cap1" in summary
        assert "cap2" in summary
        assert "cap3" in summary
        assert summary["cap1"] == ["v1"]
        assert set(summary["cap2"]) == {"v1", "v2"}
        assert summary["cap3"] == ["v2"]

    def test_get_validator_metadata(self) -> None:
        """Test getting all validator metadata."""
        registry = ValidatorRegistry()
        v1 = MockValidator("v1", ["cap1"])
        v2 = MockValidator("v2", ["cap2"])

        registry.register(v1)
        registry.register(v2)

        metadata = registry.get_validator_metadata()

        assert len(metadata) == 2
        names = [m["name"] for m in metadata]
        assert "v1" in names
        assert "v2" in names

    def test_contains_operator(self) -> None:
        """Test 'in' operator works."""
        registry = ValidatorRegistry()
        v = MockValidator("test", ["cap"])

        registry.register(v)

        assert "test" in registry
        assert "nonexistent" not in registry

    def test_capability_index_updated_on_unregister(self) -> None:
        """Test capability index is properly updated when unregistering."""
        registry = ValidatorRegistry()
        v1 = MockValidator("v1", ["cap1"])
        v2 = MockValidator("v2", ["cap1"])

        registry.register(v1)
        registry.register(v2)

        validators = registry.get_validators_by_capability("cap1")
        assert len(validators) == 2

        registry.unregister("v1")

        validators = registry.get_validators_by_capability("cap1")
        assert len(validators) == 1
        assert validators[0].name == "v2"
