"""Validator registry for managing available validators."""

from typing import Optional, Any
import logging

from src.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class ValidatorRegistry:
    """
    Registry for managing available validators and their capabilities.

    The supervisor uses this registry to:
    1. Discover available validators
    2. Query validator capabilities
    3. Select appropriate validators for tasks
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._validators: dict[str, BaseAgent] = {}
        self._capabilities_index: dict[str, list[str]] = {}

    def register(self, validator: BaseAgent) -> None:
        """
        Register a validator in the registry.

        Args:
            validator: Validator agent to register
        """
        if validator.name in self._validators:
            logger.warning(f"Overwriting existing validator: {validator.name}")

        self._validators[validator.name] = validator

        # Index capabilities for fast lookup
        for capability in validator.capabilities:
            if capability not in self._capabilities_index:
                self._capabilities_index[capability] = []
            if validator.name not in self._capabilities_index[capability]:
                self._capabilities_index[capability].append(validator.name)

        logger.info(f"Registered validator: {validator.name} with capabilities: {validator.capabilities}")

    def unregister(self, validator_name: str) -> None:
        """
        Unregister a validator from the registry.

        Args:
            validator_name: Name of validator to unregister
        """
        if validator_name not in self._validators:
            logger.warning(f"Validator not found: {validator_name}")
            return

        validator = self._validators[validator_name]

        # Remove from capabilities index
        for capability in validator.capabilities:
            if capability in self._capabilities_index:
                if validator_name in self._capabilities_index[capability]:
                    self._capabilities_index[capability].remove(validator_name)
                if not self._capabilities_index[capability]:
                    del self._capabilities_index[capability]

        del self._validators[validator_name]
        logger.info(f"Unregistered validator: {validator_name}")

    def get_validator(self, name: str) -> Optional[BaseAgent]:
        """
        Get a validator by name.

        Args:
            name: Validator name

        Returns:
            Validator agent or None if not found
        """
        return self._validators.get(name)

    def get_validators_by_capability(self, capability: str) -> list[BaseAgent]:
        """
        Get all validators that provide a specific capability.

        Args:
            capability: Capability to search for

        Returns:
            List of validators with that capability
        """
        validator_names = self._capabilities_index.get(capability, [])
        return [self._validators[name] for name in validator_names]

    def get_all_validators(self) -> list[BaseAgent]:
        """
        Get all registered validators.

        Returns:
            List of all validators
        """
        return list(self._validators.values())

    def get_capabilities_summary(self) -> dict[str, list[str]]:
        """
        Get summary of all capabilities and their validators.

        Returns:
            Dictionary mapping capabilities to validator names
        """
        return dict(self._capabilities_index)

    def get_validator_metadata(self) -> list[dict[str, Any]]:
        """
        Get metadata for all validators.

        This is used by the supervisor to understand available validators.

        Returns:
            List of validator metadata dictionaries
        """
        return [validator.get_metadata() for validator in self._validators.values()]

    def is_registered(self, validator_name: str) -> bool:
        """
        Check if a validator is registered.

        Args:
            validator_name: Name of validator to check

        Returns:
            True if registered, False otherwise
        """
        return validator_name in self._validators

    def __len__(self) -> int:
        """Return number of registered validators."""
        return len(self._validators)

    def __contains__(self, validator_name: str) -> bool:
        """Check if validator is in registry."""
        return validator_name in self._validators

    def __repr__(self) -> str:
        return f"ValidatorRegistry(validators={len(self._validators)})"
