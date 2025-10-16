"""Recovery mechanisms for workflow failures including checkpoints and rollback."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints for workflow recovery.

    Checkpoints allow the workflow to be restored to a previous
    good state in case of failures.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints (None for in-memory only)
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints: dict[str, dict[str, Any]] = {}

        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        checkpoint_id: str,
        state: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Create a checkpoint of the current state.

        Args:
            checkpoint_id: Unique identifier for this checkpoint
            state: State to checkpoint
            metadata: Additional metadata about the checkpoint
        """
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "state": self._serialize_state(state),
            "metadata": metadata or {},
        }

        # Store in memory
        self.checkpoints[checkpoint_id] = checkpoint

        # Store to disk if configured
        if self.checkpoint_dir:
            self._save_to_disk(checkpoint_id, checkpoint)

        logger.info(f"Created checkpoint: {checkpoint_id}")

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """Restore state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            Restored state, or None if checkpoint not found
        """
        # Try memory first
        checkpoint = self.checkpoints.get(checkpoint_id)

        # Try disk if not in memory
        if not checkpoint and self.checkpoint_dir:
            checkpoint = self._load_from_disk(checkpoint_id)

        if checkpoint:
            logger.info(f"Restored checkpoint: {checkpoint_id}")
            return self._deserialize_state(checkpoint["state"])
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_id}")
            return None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            List of checkpoint metadata
        """
        checkpoints = []
        for checkpoint_id, checkpoint in self.checkpoints.items():
            checkpoints.append({
                "checkpoint_id": checkpoint_id,
                "timestamp": checkpoint["timestamp"],
                "metadata": checkpoint.get("metadata", {}),
            })
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        # Delete from memory
        if checkpoint_id in self.checkpoints:
            del self.checkpoints[checkpoint_id]

        # Delete from disk
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            if checkpoint_path.exists():
                checkpoint_path.unlink()

        logger.info(f"Deleted checkpoint: {checkpoint_id}")
        return True

    def _serialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Serialize state for storage.

        Args:
            state: State to serialize

        Returns:
            Serialized state
        """
        try:
            # Use JSON serialization to ensure it's serializable
            return json.loads(json.dumps(state, default=str))
        except Exception as e:
            logger.error(f"Failed to serialize state: {e}")
            return {"error": "serialization_failed"}

    def _deserialize_state(self, serialized: dict[str, Any]) -> dict[str, Any]:
        """Deserialize state from storage.

        Args:
            serialized: Serialized state

        Returns:
            Deserialized state
        """
        return serialized

    def _save_to_disk(self, checkpoint_id: str, checkpoint: dict[str, Any]) -> None:
        """Save checkpoint to disk.

        Args:
            checkpoint_id: Checkpoint ID
            checkpoint: Checkpoint data
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)
            logger.debug(f"Saved checkpoint to disk: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to disk: {e}")

    def _load_from_disk(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """Load checkpoint from disk.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint data or None
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
            if checkpoint_path.exists():
                with open(checkpoint_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from disk: {e}")
        return None


class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""

    def __init__(self, name: str):
        """Initialize recovery strategy.

        Args:
            name: Name of the strategy
        """
        self.name = name

    @abstractmethod
    def recover(
        self,
        state: dict[str, Any],
        checkpoint_manager: CheckpointManager,
    ) -> dict[str, Any]:
        """Attempt to recover from failure.

        Args:
            state: Current (failed) state
            checkpoint_manager: Checkpoint manager for recovery

        Returns:
            Recovered state
        """
        pass


class ResumeFromLastCheckpoint(RecoveryStrategy):
    """Resume workflow from the last successful checkpoint."""

    def __init__(self):
        """Initialize resume strategy."""
        super().__init__("resume_from_checkpoint")

    def recover(
        self,
        state: dict[str, Any],
        checkpoint_manager: CheckpointManager,
    ) -> dict[str, Any]:
        """Recover by resuming from last checkpoint.

        Args:
            state: Current (failed) state
            checkpoint_manager: Checkpoint manager

        Returns:
            Recovered state
        """
        checkpoints = checkpoint_manager.list_checkpoints()

        if not checkpoints:
            logger.warning("No checkpoints available for recovery")
            return state

        # Get the most recent checkpoint
        latest_checkpoint = checkpoints[0]
        checkpoint_id = latest_checkpoint["checkpoint_id"]

        logger.info(f"Recovering from checkpoint: {checkpoint_id}")
        restored_state = checkpoint_manager.restore_checkpoint(checkpoint_id)

        if restored_state:
            # Merge metadata about recovery
            restored_state.setdefault("metadata", {})
            restored_state["metadata"]["recovered_from"] = checkpoint_id
            restored_state["metadata"]["recovery_timestamp"] = datetime.now().isoformat()
            restored_state["metadata"]["recovery_strategy"] = self.name
            return restored_state
        else:
            logger.error("Failed to restore checkpoint")
            return state


class ReplayFailedValidators(RecoveryStrategy):
    """Replay only the validators that failed."""

    def __init__(self):
        """Initialize replay strategy."""
        super().__init__("replay_failed")

    def recover(
        self,
        state: dict[str, Any],
        checkpoint_manager: CheckpointManager,
    ) -> dict[str, Any]:
        """Recover by replaying failed validators.

        Args:
            state: Current (failed) state
            checkpoint_manager: Checkpoint manager

        Returns:
            Recovered state
        """
        failed_validators = state.get("failed_validators", [])

        if not failed_validators:
            logger.info("No failed validators to replay")
            return state

        logger.info(f"Replaying {len(failed_validators)} failed validators")

        # Reset failed validators to active
        state.setdefault("active_validators", []).extend(failed_validators)
        state["failed_validators"] = []

        # Remove failed validation results
        results = state.get("validation_results", [])
        filtered_results = [
            r for r in results
            if r.validator_name not in failed_validators
        ]
        state["validation_results"] = filtered_results

        # Mark as recovery in progress
        state.setdefault("metadata", {})
        state["metadata"]["recovery_in_progress"] = True
        state["metadata"]["replaying_validators"] = failed_validators
        state["metadata"]["recovery_strategy"] = self.name

        return state


class ResetToKnownGoodState(RecoveryStrategy):
    """Reset workflow to a known good state.

    This strategy finds the last checkpoint with all validators
    passed and restarts from there.
    """

    def __init__(self):
        """Initialize reset strategy."""
        super().__init__("reset_to_good_state")

    def recover(
        self,
        state: dict[str, Any],
        checkpoint_manager: CheckpointManager,
    ) -> dict[str, Any]:
        """Recover by resetting to last known good state.

        Args:
            state: Current (failed) state
            checkpoint_manager: Checkpoint manager

        Returns:
            Recovered state
        """
        checkpoints = checkpoint_manager.list_checkpoints()

        # Find last checkpoint with no failures
        for checkpoint_info in checkpoints:
            checkpoint_id = checkpoint_info["checkpoint_id"]
            restored_state = checkpoint_manager.restore_checkpoint(checkpoint_id)

            if restored_state and not restored_state.get("failed_validators", []):
                logger.info(f"Found good state at checkpoint: {checkpoint_id}")

                # Mark as recovered
                restored_state.setdefault("metadata", {})
                restored_state["metadata"]["recovered_from"] = checkpoint_id
                restored_state["metadata"]["recovery_strategy"] = self.name
                restored_state["metadata"]["recovery_timestamp"] = datetime.now().isoformat()

                return restored_state

        logger.warning("No good state found, returning current state")
        return state


class WorkflowRecovery:
    """Main recovery coordinator for workflow failures.

    This class coordinates recovery attempts using multiple strategies
    and checkpoint management.
    """

    def __init__(
        self,
        checkpoint_manager: Optional[CheckpointManager] = None,
        strategies: Optional[list[RecoveryStrategy]] = None,
    ):
        """Initialize workflow recovery.

        Args:
            checkpoint_manager: Checkpoint manager to use
            strategies: List of recovery strategies to try (in order)
        """
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.strategies = strategies or [
            ResumeFromLastCheckpoint(),
            ReplayFailedValidators(),
            ResetToKnownGoodState(),
        ]

    def create_checkpoint(
        self,
        state: dict[str, Any],
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """Create a checkpoint for the current state.

        Args:
            state: Current state
            checkpoint_name: Optional name for checkpoint

        Returns:
            Checkpoint ID
        """
        if not checkpoint_name:
            # Generate checkpoint ID based on completion
            completed = len(state.get("completed_validators", []))
            total = completed + len(state.get("active_validators", []))
            checkpoint_name = f"checkpoint_{completed}_{total}"

        metadata = {
            "completed_validators": state.get("completed_validators", []),
            "active_validators": state.get("active_validators", []),
            "overall_status": state.get("overall_status", "unknown"),
        }

        self.checkpoint_manager.create_checkpoint(
            checkpoint_id=checkpoint_name,
            state=state,
            metadata=metadata,
        )

        return checkpoint_name

    def recover(self, state: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Attempt to recover from failure.

        Args:
            state: Current (failed) state

        Returns:
            Tuple of (recovered_state, success)
        """
        logger.info(f"Attempting recovery with {len(self.strategies)} strategies")

        for strategy in self.strategies:
            try:
                logger.info(f"Trying recovery strategy: {strategy.name}")
                recovered_state = strategy.recover(state, self.checkpoint_manager)

                # Check if recovery was successful
                if self._is_recovery_successful(recovered_state, state):
                    logger.info(f"Recovery successful with strategy: {strategy.name}")
                    return recovered_state, True
                else:
                    logger.warning(f"Strategy {strategy.name} did not improve state")

            except Exception as e:
                logger.error(f"Recovery strategy {strategy.name} failed: {e}")
                continue

        logger.error("All recovery strategies failed")
        return state, False

    def _is_recovery_successful(
        self,
        recovered_state: dict[str, Any],
        original_state: dict[str, Any],
    ) -> bool:
        """Check if recovery improved the state.

        Args:
            recovered_state: State after recovery attempt
            original_state: Original failed state

        Returns:
            True if recovery was successful
        """
        # Check if we have active validators to continue
        has_active = len(recovered_state.get("active_validators", [])) > 0

        # Check if failed validators decreased
        original_failed = len(original_state.get("failed_validators", []))
        recovered_failed = len(recovered_state.get("failed_validators", []))
        failures_decreased = recovered_failed < original_failed

        return has_active or failures_decreased

    def auto_checkpoint(
        self,
        state: dict[str, Any],
        interval: int = 5,
    ) -> Optional[str]:
        """Automatically create checkpoint based on progress.

        Args:
            state: Current state
            interval: Create checkpoint every N completed validators

        Returns:
            Checkpoint ID if created, None otherwise
        """
        completed_count = len(state.get("completed_validators", []))

        # Create checkpoint at regular intervals
        if completed_count > 0 and completed_count % interval == 0:
            return self.create_checkpoint(state)

        return None
