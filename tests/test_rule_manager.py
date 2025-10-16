"""Tests for rule manager (rule_manager.py)."""
import pytest
from datetime import datetime
from src.validators.rule_manager import (
    RuleManager,
    RuleStatus,
    ConflictType,
    RuleVersion,
)
from src.validators.rule_engine_v2 import RuleV2, Condition, ComparisonOperator


class TestRuleManager:
    """Test RuleManager class."""

    def test_add_rule(self):
        """Test adding a rule."""
        manager = RuleManager()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        version = manager.add_rule(rule, created_by="test_user")

        assert version.version == 1
        assert version.status == RuleStatus.ACTIVE
        assert version.created_by == "test_user"
        assert "test_rule" in manager.versions

    def test_add_new_version(self):
        """Test adding new version of existing rule."""
        manager = RuleManager()

        rule_v1 = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Must be positive"
        )

        manager.add_rule(rule_v1)

        # Add new version
        rule_v2 = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 10),
            error_message="Must be greater than 10"
        )

        version = manager.add_rule(rule_v2, changelog="Increased threshold")

        assert version.version == 2
        assert version.changelog == "Increased threshold"
        assert len(manager.versions["test_rule"]) == 2

        # Old version should be deprecated
        old_version = manager.versions["test_rule"][0]
        assert old_version.status == RuleStatus.DEPRECATED

    def test_get_rule(self):
        """Test getting a rule."""
        manager = RuleManager()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Test"
        )

        manager.add_rule(rule)

        # Get latest
        version = manager.get_rule("test_rule")
        assert version is not None
        assert version.version == 1

        # Get specific version
        version = manager.get_rule("test_rule", version=1)
        assert version is not None

        # Non-existent rule
        version = manager.get_rule("nonexistent")
        assert version is None

    def test_activate_deactivate_rule(self):
        """Test activating and deactivating rules."""
        manager = RuleManager()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Test"
        )

        manager.add_rule(rule)

        # Deactivate
        result = manager.deactivate_rule("test_rule")
        assert result is True
        assert manager.get_rule("test_rule", status=RuleStatus.ACTIVE) is None

        # Activate
        result = manager.activate_rule("test_rule")
        assert result is True
        version = manager.get_rule("test_rule", status=RuleStatus.ACTIVE)
        assert version is not None

    def test_delete_rule(self):
        """Test deleting rules."""
        manager = RuleManager()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Test"
        )

        manager.add_rule(rule)
        manager.add_rule(rule)  # Add version 2

        # Delete specific version
        result = manager.delete_rule("test_rule", version=2)
        assert result is True
        assert len(manager.versions["test_rule"]) == 1

        # Delete all versions
        result = manager.delete_rule("test_rule")
        assert result is True
        assert "test_rule" not in manager.versions

    def test_detect_priority_conflicts(self):
        """Test detecting priority conflicts."""
        manager = RuleManager()

        rule1 = RuleV2(
            name="rule1",
            condition=Condition("a", ComparisonOperator.GT, 0),
            error_message="Test",
            priority=5
        )

        rule2 = RuleV2(
            name="rule2",
            condition=Condition("b", ComparisonOperator.GT, 0),
            error_message="Test",
            priority=5
        )

        manager.add_rule(rule1)
        manager.add_rule(rule2)

        conflicts = manager.detect_conflicts()
        assert len(conflicts) > 0
        assert any(c.conflict_type == ConflictType.PRIORITY_CONFLICT for c in conflicts)

    def test_get_dependency_tree(self):
        """Test getting dependency tree."""
        manager = RuleManager()

        rule1 = RuleV2(
            name="rule1",
            condition=lambda d: True,
            error_message="Test"
        )

        rule2 = RuleV2(
            name="rule2",
            condition=lambda d: True,
            error_message="Test",
            dependencies={"rule1"}
        )

        rule3 = RuleV2(
            name="rule3",
            condition=lambda d: True,
            error_message="Test",
            dependencies={"rule2"}
        )

        manager.add_rule(rule1)
        manager.add_rule(rule2)
        manager.add_rule(rule3)

        tree = manager.get_dependency_tree("rule3")
        assert tree["name"] == "rule3"
        assert len(tree["dependencies"]) == 1

    def test_export_rules(self):
        """Test exporting rules."""
        manager = RuleManager()

        rule = RuleV2(
            name="test_rule",
            condition=Condition("value", ComparisonOperator.GT, 0),
            error_message="Test",
            tags={"validation"}
        )

        manager.add_rule(rule)

        export_data = manager.export_rules()

        assert "rules" in export_data
        assert "test_rule" in export_data["rules"]
        assert len(export_data["rules"]["test_rule"]["versions"]) == 1

    def test_get_statistics(self):
        """Test getting statistics."""
        manager = RuleManager()

        rule1 = RuleV2(
            name="rule1",
            condition=Condition("a", ComparisonOperator.GT, 0),
            error_message="Test"
        )

        manager.add_rule(rule1)

        stats = manager.get_statistics()

        assert stats["total_rules"] == 1
        assert stats["total_versions"] == 1
        assert "status_counts" in stats
        assert "engine_stats" in stats
