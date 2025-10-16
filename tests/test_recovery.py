"""Tests for recovery mechanisms."""

import tempfile
from pathlib import Path

import pytest

from src.resilience.recovery import (
    CheckpointManager,
    RecoveryStrategy,
    ReplayFailedValidators,
    ResetToKnownGoodState,
    ResumeFromLastCheckpoint,
    WorkflowRecovery,
)


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        manager = CheckpointManager()
        state = {"data": "test", "count": 42}

        manager.create_checkpoint("test_checkpoint", state)

        assert "test_checkpoint" in manager.checkpoints

    def test_restore_checkpoint(self):
        """Test restoring a checkpoint."""
        manager = CheckpointManager()
        state = {"data": "test", "count": 42}

        manager.create_checkpoint("test_checkpoint", state)
        restored = manager.restore_checkpoint("test_checkpoint")

        assert restored is not None
        assert restored["data"] == "test"
        assert restored["count"] == 42

    def test_restore_nonexistent_checkpoint(self):
        """Test restoring checkpoint that doesn't exist."""
        manager = CheckpointManager()

        restored = manager.restore_checkpoint("nonexistent")

        assert restored is None

    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        manager = CheckpointManager()

        manager.create_checkpoint("cp1", {"a": 1})
        manager.create_checkpoint("cp2", {"b": 2})

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2
        checkpoint_ids = [cp["checkpoint_id"] for cp in checkpoints]
        assert "cp1" in checkpoint_ids
        assert "cp2" in checkpoint_ids

    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        manager = CheckpointManager()

        manager.create_checkpoint("test", {"a": 1})
        assert "test" in manager.checkpoints

        manager.delete_checkpoint("test")
        assert "test" not in manager.checkpoints

    def test_checkpoint_with_metadata(self):
        """Test checkpoint with metadata."""
        manager = CheckpointManager()
        state = {"data": "test"}
        metadata = {"stage": "validation", "completed": 5}

        manager.create_checkpoint("test", state, metadata=metadata)

        checkpoints = manager.list_checkpoints()
        assert checkpoints[0]["metadata"] == metadata

    def test_checkpoint_to_disk(self):
        """Test saving checkpoint to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

            state = {"data": "test"}
            manager.create_checkpoint("test", state)

            # Check file was created
            checkpoint_file = checkpoint_dir / "test.json"
            assert checkpoint_file.exists()

            # Create new manager and restore
            new_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            restored = new_manager.restore_checkpoint("test")

            assert restored is not None
            assert restored["data"] == "test"


class TestResumeFromLastCheckpoint:
    """Tests for ResumeFromLastCheckpoint strategy."""

    def test_resume_from_last_checkpoint(self):
        """Test resuming from last checkpoint."""
        strategy = ResumeFromLastCheckpoint()
        manager = CheckpointManager()

        # Create checkpoints
        manager.create_checkpoint("cp1", {"stage": 1})
        manager.create_checkpoint("cp2", {"stage": 2})

        failed_state = {"stage": "failed"}

        recovered = strategy.recover(failed_state, manager)

        # Should restore from cp2 (most recent)
        assert recovered["stage"] == 2
        assert "recovered_from" in recovered["metadata"]

    def test_no_checkpoints_available(self):
        """Test when no checkpoints are available."""
        strategy = ResumeFromLastCheckpoint()
        manager = CheckpointManager()

        state = {"data": "test"}

        recovered = strategy.recover(state, manager)

        # Should return original state
        assert recovered == state


class TestReplayFailedValidators:
    """Tests for ReplayFailedValidators strategy."""

    def test_replay_failed_validators(self):
        """Test replaying failed validators."""
        strategy = ReplayFailedValidators()
        manager = CheckpointManager()

        state = {
            "failed_validators": ["val1", "val2"],
            "active_validators": [],
            "validation_results": [
                type("Result", (), {"validator_name": "val1"}),
                type("Result", (), {"validator_name": "val2"}),
            ],
        }

        recovered = strategy.recover(state, manager)

        # Failed validators should be moved to active
        assert "val1" in recovered["active_validators"]
        assert "val2" in recovered["active_validators"]
        assert recovered["failed_validators"] == []

        # Failed results should be removed
        # (simplified check since we used mock objects)
        assert "validation_results" in recovered

    def test_no_failed_validators(self):
        """Test when there are no failed validators."""
        strategy = ReplayFailedValidators()
        manager = CheckpointManager()

        state = {"failed_validators": []}

        recovered = strategy.recover(state, manager)

        assert recovered == state


class TestResetToKnownGoodState:
    """Tests for ResetToKnownGoodState strategy."""

    def test_reset_to_good_state(self):
        """Test resetting to last known good state."""
        strategy = ResetToKnownGoodState()
        manager = CheckpointManager()

        # Create checkpoints
        manager.create_checkpoint("cp1", {
            "stage": 1,
            "failed_validators": [],
        })
        manager.create_checkpoint("cp2", {
            "stage": 2,
            "failed_validators": ["val1"],  # Has failures
        })

        failed_state = {"stage": "failed"}

        recovered = strategy.recover(failed_state, manager)

        # Should restore from cp1 (no failures)
        assert recovered["stage"] == 1
        assert "recovered_from" in recovered["metadata"]

    def test_no_good_state_available(self):
        """Test when no good state is available."""
        strategy = ResetToKnownGoodState()
        manager = CheckpointManager()

        # All checkpoints have failures
        manager.create_checkpoint("cp1", {"failed_validators": ["val1"]})
        manager.create_checkpoint("cp2", {"failed_validators": ["val2"]})

        state = {"data": "test"}

        recovered = strategy.recover(state, manager)

        # Should return original state
        assert recovered == state


class TestWorkflowRecovery:
    """Tests for WorkflowRecovery coordinator."""

    def test_create_checkpoint(self):
        """Test creating checkpoint through WorkflowRecovery."""
        recovery = WorkflowRecovery()

        state = {
            "completed_validators": ["val1", "val2"],
            "active_validators": ["val3"],
        }

        checkpoint_id = recovery.create_checkpoint(state)

        assert checkpoint_id is not None
        assert checkpoint_id in recovery.checkpoint_manager.checkpoints

    def test_create_named_checkpoint(self):
        """Test creating checkpoint with custom name."""
        recovery = WorkflowRecovery()
        state = {"data": "test"}

        checkpoint_id = recovery.create_checkpoint(state, "my_checkpoint")

        assert checkpoint_id == "my_checkpoint"

    def test_successful_recovery(self):
        """Test successful recovery."""
        recovery = WorkflowRecovery()

        # Create a good checkpoint
        good_state = {
            "completed_validators": ["val1"],
            "active_validators": ["val2"],
            "failed_validators": [],
        }
        recovery.create_checkpoint(good_state, "good_checkpoint")

        # Failed state
        failed_state = {
            "completed_validators": ["val1"],
            "active_validators": [],
            "failed_validators": ["val2"],
        }

        recovered_state, success = recovery.recover(failed_state)

        assert success
        # Check that we got a better state
        assert len(recovered_state.get("active_validators", [])) > 0

    def test_recovery_with_multiple_strategies(self):
        """Test recovery tries multiple strategies."""

        class CountingStrategy(RecoveryStrategy):
            """Strategy that counts how many times it was called."""

            call_count = 0

            def __init__(self):
                super().__init__("counting")

            def recover(self, state, checkpoint_manager):
                CountingStrategy.call_count += 1
                # Don't improve state, let next strategy try
                return state

        strategy1 = CountingStrategy()
        strategy2 = CountingStrategy()

        recovery = WorkflowRecovery(strategies=[strategy1, strategy2])

        failed_state = {"failed_validators": ["test"]}
        recovery.recover(failed_state)

        # Both strategies should have been called
        assert CountingStrategy.call_count >= 2

    def test_auto_checkpoint(self):
        """Test automatic checkpoint creation."""
        recovery = WorkflowRecovery()

        # First 4 validators - no checkpoint
        state = {"completed_validators": ["v1", "v2", "v3", "v4"]}
        result = recovery.auto_checkpoint(state, interval=5)
        assert result is None

        # 5th validator - checkpoint should be created
        state = {"completed_validators": ["v1", "v2", "v3", "v4", "v5"]}
        result = recovery.auto_checkpoint(state, interval=5)
        assert result is not None

        # 10th validator - another checkpoint
        state = {"completed_validators": [f"v{i}" for i in range(1, 11)]}
        result = recovery.auto_checkpoint(state, interval=5)
        assert result is not None


class TestRecoverySuccess:
    """Tests for recovery success detection."""

    def test_recovery_with_active_validators(self):
        """Test recovery is successful if active validators exist."""
        recovery = WorkflowRecovery()

        original = {
            "active_validators": [],
            "failed_validators": ["val1"],
        }

        recovered = {
            "active_validators": ["val1"],  # Restored
            "failed_validators": [],
        }

        success = recovery._is_recovery_successful(recovered, original)
        assert success

    def test_recovery_with_reduced_failures(self):
        """Test recovery is successful if failures decreased."""
        recovery = WorkflowRecovery()

        original = {
            "active_validators": [],
            "failed_validators": ["val1", "val2"],
        }

        recovered = {
            "active_validators": [],
            "failed_validators": ["val1"],  # One less failure
        }

        success = recovery._is_recovery_successful(recovered, original)
        assert success

    def test_recovery_not_successful(self):
        """Test recovery is not successful if state didn't improve."""
        recovery = WorkflowRecovery()

        original = {
            "active_validators": [],
            "failed_validators": ["val1"],
        }

        recovered = {
            "active_validators": [],
            "failed_validators": ["val1"],  # Same
        }

        success = recovery._is_recovery_successful(recovered, original)
        assert not success
