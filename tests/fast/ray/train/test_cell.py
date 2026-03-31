from unittest.mock import MagicMock, patch

from miles.ray.train.cell import (
    RayTrainCell,
    _StateAllocatedAlive,
    _StateAllocatedErrored,
    _StateAllocatedUninitialized,
    _StatePending,
    _StateStopped,
)
from miles.utils.indep_dp import IndepDPInfo


def _make_cell_with_state(state) -> RayTrainCell:
    """Create a RayTrainCell with a pre-set state, bypassing __init__."""
    cell = object.__new__(RayTrainCell)
    cell.cell_index = 0
    cell._state = state
    return cell


def _make_indep_dp_info(*, cell_index: int = 0, alive_cell_indices: list[int] | None = None) -> IndepDPInfo:
    if alive_cell_indices is None:
        alive_cell_indices = [0]
    return IndepDPInfo(
        cell_index=cell_index,
        num_cells=3,
        alive_rank=alive_cell_indices.index(cell_index),
        alive_size=len(alive_cell_indices),
        quorum_id=0,
        alive_cell_indices=alive_cell_indices,
    )


def _make_alive_state() -> _StateAllocatedAlive:
    return _StateAllocatedAlive(actor_handles=[], indep_dp_info=_make_indep_dp_info())


class TestStopIdempotent:
    def test_stop_already_stopped_is_noop(self):
        """Calling stop() on an already-stopped cell does not crash and state remains Stopped."""
        cell = _make_cell_with_state(_StateStopped())

        cell.stop()

        assert cell.is_stopped

    def test_stop_pending_transitions_to_stopped(self):
        """Calling stop() on a Pending cell transitions to Stopped (normal path)."""
        cell = _make_cell_with_state(_StatePending())

        cell.stop()

        assert cell.is_stopped


class TestMarkAsPendingIdempotent:
    def test_mark_as_pending_already_pending_is_noop(self):
        """Calling mark_as_pending() on an already-pending cell does not crash."""
        cell = _make_cell_with_state(_StatePending())

        cell.mark_as_pending()

        assert cell.is_pending

    def test_mark_as_pending_already_allocated_is_noop(self):
        """Calling mark_as_pending() on an alive cell does not crash and state stays."""
        cell = _make_cell_with_state(_make_alive_state())

        cell.mark_as_pending()

        assert cell.is_alive


class TestStateTransitions:
    def test_uninitialized_is_allocated_but_not_alive(self):
        """Uninitialized state is allocated but not alive or errored."""
        cell = _make_cell_with_state(_StateAllocatedUninitialized(actor_handles=[]))

        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored

    def test_mark_as_alive_transitions_uninitialized_to_alive(self):
        """_mark_as_alive transitions from Uninitialized to Alive."""
        cell = _make_cell_with_state(_StateAllocatedUninitialized(actor_handles=[]))
        info = _make_indep_dp_info()

        cell._mark_as_alive(indep_dp_info=info)

        assert cell.is_alive
        assert not cell.is_errored
        assert cell.indep_dp_info == info

    def test_mark_as_errored_transitions_alive_to_errored(self):
        """_mark_as_errored transitions from Alive to Errored."""
        cell = _make_cell_with_state(_make_alive_state())

        cell._mark_as_errored()

        assert cell.is_errored
        assert not cell.is_alive
        assert cell.is_allocated

    @patch("miles.ray.train.cell.ray")
    def test_stop_from_alive_kills_actors_and_transitions_to_stopped(self, mock_ray):
        """stop() from Alive kills all actors and transitions to Stopped."""
        actor_0, actor_1 = MagicMock(), MagicMock()
        state = _StateAllocatedAlive(actor_handles=[actor_0, actor_1], indep_dp_info=_make_indep_dp_info())
        cell = _make_cell_with_state(state)

        cell.stop()

        assert cell.is_stopped
        assert not cell.is_alive
        assert not cell.is_allocated
        mock_ray.kill.assert_any_call(actor_0)
        mock_ray.kill.assert_any_call(actor_1)
        assert mock_ray.kill.call_count == 2

    def test_update_indep_dp_info_replaces_info(self):
        """_update_indep_dp_info updates the stored IndepDPInfo on an alive cell."""
        old_info = _make_indep_dp_info(alive_cell_indices=[0, 1, 2])
        cell = _make_cell_with_state(_StateAllocatedAlive(actor_handles=[], indep_dp_info=old_info))

        new_info = _make_indep_dp_info(alive_cell_indices=[0, 2])
        cell._update_indep_dp_info(new_info)

        assert cell.indep_dp_info == new_info
        assert cell.indep_dp_info.alive_size == 2


class TestStatePredicates:
    """Verify is_pending/is_allocated/is_alive/is_stopped/is_errored for each state."""

    def test_pending(self):
        cell = _make_cell_with_state(_StatePending())
        assert cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_stopped

    def test_uninitialized(self):
        cell = _make_cell_with_state(_StateAllocatedUninitialized(actor_handles=[]))
        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_stopped

    def test_alive(self):
        cell = _make_cell_with_state(_make_alive_state())
        assert not cell.is_pending
        assert cell.is_allocated
        assert cell.is_alive
        assert not cell.is_stopped
        assert not cell.is_errored

    def test_errored(self):
        cell = _make_cell_with_state(
            _StateAllocatedErrored(actor_handles=[], indep_dp_info=_make_indep_dp_info())
        )
        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_stopped
        assert cell.is_errored

    def test_stopped(self):
        cell = _make_cell_with_state(_StateStopped())
        assert not cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert cell.is_stopped
