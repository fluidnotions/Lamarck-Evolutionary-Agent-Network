"""Rule management system with versioning, conflict detection, and dependencies."""
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
import hashlib

from src.validators.rule_engine_v2 import RuleV2, RuleEngineV2

logger = logging.getLogger(__name__)


class RuleStatus(Enum):
    """Status of a rule version."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ConflictType(Enum):
    """Types of rule conflicts."""
    CONTRADICTORY = "contradictory"  # Rules that can never both pass
    REDUNDANT = "redundant"  # One rule makes another obsolete
    PRIORITY_CONFLICT = "priority_conflict"  # Same priority, overlapping conditions
    DEPENDENCY_CYCLE = "dependency_cycle"  # Circular dependencies


@dataclass
class RuleVersion:
    """Represents a versioned rule."""

    rule: RuleV2
    version: int
    status: RuleStatus
    created_at: datetime
    created_by: str
    changelog: str = ""
    checksum: str = field(default="")

    def __post_init__(self) -> None:
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate checksum of rule configuration."""
        data = {
            "name": self.rule.name,
            "error_message": self.rule.error_message,
            "severity": self.rule.severity,
            "priority": self.rule.priority,
            "tags": sorted(list(self.rule.tags)),
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


@dataclass
class RuleConflict:
    """Represents a conflict between rules."""

    conflict_type: ConflictType
    rule1_name: str
    rule2_name: str
    description: str
    severity: str = "warning"
    suggested_resolution: Optional[str] = None


@dataclass
class RuleDependency:
    """Represents a dependency between rules."""

    dependent_rule: str
    required_rule: str
    reason: str = ""


class RuleManager:
    """Manages rule lifecycle, versioning, and dependencies."""

    def __init__(self, engine: Optional[RuleEngineV2] = None):
        """Initialize rule manager.

        Args:
            engine: Optional rule engine to manage
        """
        self.engine = engine or RuleEngineV2()
        self.versions: Dict[str, List[RuleVersion]] = {}  # rule_name -> versions
        self.dependencies: Dict[str, Set[str]] = {}  # rule_name -> set of dependencies
        self.conflicts: List[RuleConflict] = []

    def add_rule(
        self,
        rule: RuleV2,
        created_by: str = "system",
        changelog: str = "",
        status: RuleStatus = RuleStatus.ACTIVE,
    ) -> RuleVersion:
        """Add a new rule or create a new version.

        Args:
            rule: Rule to add
            created_by: Who created the rule
            changelog: Description of changes
            status: Initial status

        Returns:
            RuleVersion instance
        """
        # Get or create version list
        if rule.name not in self.versions:
            self.versions[rule.name] = []
            version_num = 1
        else:
            # New version
            version_num = len(self.versions[rule.name]) + 1
            # Deprecate previous active version
            for v in self.versions[rule.name]:
                if v.status == RuleStatus.ACTIVE:
                    v.status = RuleStatus.DEPRECATED

        # Create version
        version = RuleVersion(
            rule=rule,
            version=version_num,
            status=status,
            created_at=datetime.now(),
            created_by=created_by,
            changelog=changelog,
        )

        self.versions[rule.name].append(version)

        # Add to engine if active
        if status == RuleStatus.ACTIVE:
            self.engine.add_rule(rule)

        # Track dependencies
        if rule.dependencies:
            self.dependencies[rule.name] = rule.dependencies.copy()

        # Check for conflicts
        self._detect_conflicts_for_rule(rule)

        logger.info(f"Added rule {rule.name} v{version_num} with status {status.value}")

        return version

    def get_rule(
        self,
        name: str,
        version: Optional[int] = None,
        status: Optional[RuleStatus] = None,
    ) -> Optional[RuleVersion]:
        """Get a rule by name and optionally version.

        Args:
            name: Rule name
            version: Specific version number (None for latest)
            status: Filter by status (None for any)

        Returns:
            RuleVersion or None if not found
        """
        if name not in self.versions:
            return None

        versions = self.versions[name]

        # Filter by status
        if status:
            versions = [v for v in versions if v.status == status]

        if not versions:
            return None

        # Get specific version or latest
        if version is not None:
            for v in versions:
                if v.version == version:
                    return v
            return None
        else:
            return versions[-1]  # Latest version

    def get_active_rule(self, name: str) -> Optional[RuleV2]:
        """Get the active version of a rule.

        Args:
            name: Rule name

        Returns:
            Active RuleV2 or None
        """
        version = self.get_rule(name, status=RuleStatus.ACTIVE)
        return version.rule if version else None

    def get_all_versions(self, name: str) -> List[RuleVersion]:
        """Get all versions of a rule.

        Args:
            name: Rule name

        Returns:
            List of RuleVersion objects
        """
        return self.versions.get(name, [])

    def activate_rule(self, name: str, version: Optional[int] = None) -> bool:
        """Activate a rule version.

        Args:
            name: Rule name
            version: Version to activate (None for latest)

        Returns:
            True if successful
        """
        rule_version = self.get_rule(name, version=version)
        if not rule_version:
            logger.warning(f"Rule {name} v{version} not found")
            return False

        # Deactivate other versions
        for v in self.versions[name]:
            if v.status == RuleStatus.ACTIVE:
                v.status = RuleStatus.DEPRECATED

        # Activate this version
        rule_version.status = RuleStatus.ACTIVE
        self.engine.add_rule(rule_version.rule)

        logger.info(f"Activated rule {name} v{rule_version.version}")
        return True

    def deactivate_rule(self, name: str) -> bool:
        """Deactivate a rule.

        Args:
            name: Rule name

        Returns:
            True if successful
        """
        if name not in self.versions:
            return False

        # Deactivate all versions
        for v in self.versions[name]:
            if v.status == RuleStatus.ACTIVE:
                v.status = RuleStatus.DEPRECATED

        # Remove from engine
        self.engine.remove_rule(name)

        logger.info(f"Deactivated rule {name}")
        return True

    def delete_rule(self, name: str, version: Optional[int] = None) -> bool:
        """Delete a rule version.

        Args:
            name: Rule name
            version: Specific version to delete (None for all)

        Returns:
            True if successful
        """
        if name not in self.versions:
            return False

        if version is not None:
            # Delete specific version
            self.versions[name] = [v for v in self.versions[name] if v.version != version]
            if not self.versions[name]:
                del self.versions[name]
                self.engine.remove_rule(name)
        else:
            # Delete all versions
            del self.versions[name]
            self.engine.remove_rule(name)
            if name in self.dependencies:
                del self.dependencies[name]

        logger.info(f"Deleted rule {name}" + (f" v{version}" if version else ""))
        return True

    def detect_conflicts(self) -> List[RuleConflict]:
        """Detect conflicts between all active rules.

        Returns:
            List of detected conflicts
        """
        self.conflicts = []

        active_rules = [
            v.rule for versions in self.versions.values()
            for v in versions if v.status == RuleStatus.ACTIVE
        ]

        # Check for priority conflicts
        self._detect_priority_conflicts(active_rules)

        # Check for dependency cycles
        self._detect_dependency_cycles()

        # TODO: Add semantic conflict detection (contradictory/redundant)
        # This would require analyzing rule conditions

        return self.conflicts

    def _detect_conflicts_for_rule(self, rule: RuleV2) -> None:
        """Detect conflicts for a specific rule.

        Args:
            rule: Rule to check
        """
        active_rules = [
            v.rule for versions in self.versions.values()
            for v in versions if v.status == RuleStatus.ACTIVE and v.rule.name != rule.name
        ]

        # Check priority conflicts
        for other in active_rules:
            if other.priority == rule.priority and rule.priority != 0:
                conflict = RuleConflict(
                    conflict_type=ConflictType.PRIORITY_CONFLICT,
                    rule1_name=rule.name,
                    rule2_name=other.name,
                    description=f"Both rules have priority {rule.priority}",
                    severity="warning",
                    suggested_resolution="Adjust rule priorities to ensure deterministic execution order",
                )
                self.conflicts.append(conflict)

    def _detect_priority_conflicts(self, rules: List[RuleV2]) -> None:
        """Detect priority conflicts among rules.

        Args:
            rules: List of rules to check
        """
        priority_groups: Dict[int, List[str]] = {}

        for rule in rules:
            if rule.priority not in priority_groups:
                priority_groups[rule.priority] = []
            priority_groups[rule.priority].append(rule.name)

        for priority, rule_names in priority_groups.items():
            if len(rule_names) > 1 and priority != 0:
                conflict = RuleConflict(
                    conflict_type=ConflictType.PRIORITY_CONFLICT,
                    rule1_name=rule_names[0],
                    rule2_name=rule_names[1],
                    description=f"Multiple rules share priority {priority}: {', '.join(rule_names)}",
                    severity="warning",
                    suggested_resolution="Assign unique priorities to rules",
                )
                self.conflicts.append(conflict)

    def _detect_dependency_cycles(self) -> None:
        """Detect circular dependencies between rules."""
        def has_cycle(rule_name: str, visited: Set[str], stack: Set[str]) -> bool:
            visited.add(rule_name)
            stack.add(rule_name)

            # Check dependencies
            for dep in self.dependencies.get(rule_name, set()):
                if dep not in visited:
                    if has_cycle(dep, visited, stack):
                        return True
                elif dep in stack:
                    # Found cycle
                    conflict = RuleConflict(
                        conflict_type=ConflictType.DEPENDENCY_CYCLE,
                        rule1_name=rule_name,
                        rule2_name=dep,
                        description=f"Circular dependency detected: {rule_name} -> {dep}",
                        severity="error",
                        suggested_resolution="Remove circular dependencies",
                    )
                    self.conflicts.append(conflict)
                    return True

            stack.remove(rule_name)
            return False

        visited: Set[str] = set()
        for rule_name in self.dependencies.keys():
            if rule_name not in visited:
                has_cycle(rule_name, visited, set())

    def get_dependency_tree(self, rule_name: str) -> Dict[str, Any]:
        """Get dependency tree for a rule.

        Args:
            rule_name: Rule name

        Returns:
            Nested dictionary representing dependency tree
        """
        def build_tree(name: str, visited: Set[str]) -> Dict[str, Any]:
            if name in visited:
                return {"name": name, "circular": True}

            visited.add(name)
            deps = self.dependencies.get(name, set())

            return {
                "name": name,
                "dependencies": [build_tree(dep, visited.copy()) for dep in deps]
            }

        return build_tree(rule_name, set())

    def export_rules(self, include_inactive: bool = False) -> Dict[str, Any]:
        """Export all rules to a dictionary.

        Args:
            include_inactive: Include inactive rules

        Returns:
            Dictionary with all rule data
        """
        export_data: Dict[str, Any] = {
            "exported_at": datetime.now().isoformat(),
            "rules": {},
        }

        for rule_name, versions in self.versions.items():
            rule_data = {
                "versions": []
            }

            for version in versions:
                if not include_inactive and version.status != RuleStatus.ACTIVE:
                    continue

                rule_data["versions"].append({
                    "version": version.version,
                    "status": version.status.value,
                    "created_at": version.created_at.isoformat(),
                    "created_by": version.created_by,
                    "changelog": version.changelog,
                    "checksum": version.checksum,
                    "rule": {
                        "name": version.rule.name,
                        "error_message": version.rule.error_message,
                        "severity": version.rule.severity,
                        "priority": version.rule.priority,
                        "enabled": version.rule.enabled,
                        "tags": list(version.rule.tags),
                        "dependencies": list(version.rule.dependencies),
                        "metadata": version.rule.metadata,
                    }
                })

            if rule_data["versions"]:
                export_data["rules"][rule_name] = rule_data

        return export_data

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about managed rules.

        Returns:
            Dictionary with statistics
        """
        total_rules = len(self.versions)
        total_versions = sum(len(versions) for versions in self.versions.values())

        status_counts = {status.value: 0 for status in RuleStatus}
        for versions in self.versions.values():
            for version in versions:
                status_counts[version.status.value] += 1

        return {
            "total_rules": total_rules,
            "total_versions": total_versions,
            "status_counts": status_counts,
            "conflicts": len(self.conflicts),
            "dependencies": len(self.dependencies),
            "engine_stats": self.engine.get_analytics(),
        }
