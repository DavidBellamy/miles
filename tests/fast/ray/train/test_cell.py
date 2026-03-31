import asyncio
from unittest.mock import MagicMock, patch

import pytest

from miles.ray.train.cell import RayTrainCell
from miles.utils.indep_dp import IndepDPInfo


def _make_indep_dp_info(
    *,
    cell_index: int = 0,
    alive_cell_indices: list[int] | None = None,
    quorum_id: int = 1,
) -> IndepDPInfo:
    if alive_cell_indices is None:
        alive_cell_indices = [0]
    return IndepDPInfo(
        cell_index=cell_index,
        num_cells=3,
        alive_rank=alive_cell_indices.index(cell_index),
        alive_size=len(alive_cell_indices),
        quorum_id=quorum_id,
        alive_cell_indices=alive_cell_indices,
    )


def _make_mock_actors(count: int = 2) -> list[MagicMock]:
    return [MagicMock(name=f"actor_{i}") for i in range(count)]


def _make_cell(*, cell_index: int = 0, actor_count: int = 2) -> RayTrainCell:
    """Create a real RayTrainCell with a mock actor factory."""
    actors = _make_mock_actors(actor_count)
    cell = RayTrainCell(
        args=MagicMock(),
        role="actor",
        with_ref=False,
        cell_index=cell_index,
        actor_factory=lambda: list(actors),
        rollout_manager=None,
    )
    return cell


class TestInitialState:
    def test_starts_as_uninitialized_after_init(self):
        """After __init__, cell is allocated (uninitialized) — actors created but not init'd."""
        cell = _make_cell()

        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_pending
        assert not cell.is_stopped

    def test_actor_handles_available_after_init(self):
        """Actor handles from the factory are accessible."""
        cell = _make_cell(actor_count=3)

        handles = cell._get_actor_handles()
        assert len(handles) == 3


class TestStopTransitions:
    @patch("miles.ray.train.cell.ray")
    def test_stop_from_uninitialized_kills_actors(self, mock_ray):
        """stop() from Uninitialized kills all actors and transitions to Stopped."""
        cell = _make_cell(actor_count=2)
        handles = cell._get_actor_handles()

        cell.stop()

        assert cell.is_stopped
        assert not cell.is_allocated
        assert mock_ray.kill.call_count == 2
        mock_ray.kill.assert_any_call(handles[0])
        mock_ray.kill.assert_any_call(handles[1])

    @patch("miles.ray.train.cell.ray")
    def test_stop_from_alive_kills_actors(self, mock_ray):
        """stop() from Alive kills all actors and transitions to Stopped."""
        cell = _make_cell(actor_count=2)
        handles = cell._get_actor_handles()
        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())

        cell.stop()

        assert cell.is_stopped
        assert mock_ray.kill.call_count == 2

    def test_stop_from_pending_transitions_to_stopped(self):
        """stop() from Pending (no actors) transitions to Stopped without killing."""
        cell = _make_cell()
        cell.stop()  # Uninitialized → Stopped
        cell.mark_as_pending()  # Stopped → Pending

        cell.stop()  # Pending → Stopped

        assert cell.is_stopped

    def test_stop_already_stopped_is_idempotent(self):
        """Calling stop() on a stopped cell is a no-op."""
        cell = _make_cell()
        cell.stop()
        assert cell.is_stopped

        cell.stop()

        assert cell.is_stopped


class TestMarkAsPending:
    def test_from_stopped(self):
        """mark_as_pending from Stopped transitions to Pending."""
        cell = _make_cell()
        cell.stop()

        cell.mark_as_pending()

        assert cell.is_pending

    def test_idempotent_when_pending(self):
        """mark_as_pending on Pending is a no-op."""
        cell = _make_cell()
        cell.stop()
        cell.mark_as_pending()

        cell.mark_as_pending()

        assert cell.is_pending

    def test_idempotent_when_allocated(self):
        """mark_as_pending on an allocated cell is a no-op."""
        cell = _make_cell()  # Uninitialized (allocated)

        cell.mark_as_pending()

        assert cell.is_allocated


class TestAllocateForPending:
    def test_reallocate_after_stop_start(self):
        """After stop → mark_as_pending → allocate_for_pending, cell has fresh actors."""
        call_count = 0
        actors_v1 = _make_mock_actors(2)
        actors_v2 = _make_mock_actors(2)

        def factory():
            nonlocal call_count
            call_count += 1
            return list(actors_v1 if call_count == 1 else actors_v2)

        cell = RayTrainCell(
            args=MagicMock(),
            role="actor",
            with_ref=False,
            cell_index=0,
            actor_factory=factory,
            rollout_manager=None,
        )
        assert call_count == 1

        cell.stop()
        cell.mark_as_pending()
        cell.allocate_for_pending()

        assert call_count == 2
        assert cell.is_allocated
        assert cell._get_actor_handles() == actors_v2


class TestMarkAsAlive:
    def test_transitions_uninitialized_to_alive(self):
        """_mark_as_alive transitions from Uninitialized to Alive with indep_dp_info."""
        cell = _make_cell()
        info = _make_indep_dp_info(alive_cell_indices=[0, 1, 2])

        cell._mark_as_alive(indep_dp_info=info)

        assert cell.is_alive
        assert cell.indep_dp_info == info

    def test_preserves_actor_handles(self):
        """_mark_as_alive preserves the actor handles from Uninitialized state."""
        cell = _make_cell(actor_count=3)
        handles_before = cell._get_actor_handles()

        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())

        assert cell._get_actor_handles() == handles_before

    def test_rejects_from_alive(self):
        """_mark_as_alive from Alive raises AssertionError."""
        cell = _make_cell()
        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())

        with pytest.raises(AssertionError):
            cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())


class TestUpdateIndepDPInfo:
    def test_updates_stored_info(self):
        """_update_indep_dp_info replaces the IndepDPInfo on an alive cell."""
        cell = _make_cell()
        old_info = _make_indep_dp_info(alive_cell_indices=[0, 1, 2])
        cell._mark_as_alive(indep_dp_info=old_info)

        new_info = _make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._update_indep_dp_info(new_info)

        assert cell.indep_dp_info == new_info
        assert cell.indep_dp_info.alive_size == 2
        assert cell.indep_dp_info.quorum_id == 2

    def test_preserves_actor_handles(self):
        """_update_indep_dp_info preserves actor handles."""
        cell = _make_cell(actor_count=3)
        handles = cell._get_actor_handles()
        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())

        cell._update_indep_dp_info(_make_indep_dp_info(quorum_id=5))

        assert cell._get_actor_handles() == handles

    def test_rejects_from_uninitialized(self):
        """_update_indep_dp_info from Uninitialized raises AssertionError."""
        cell = _make_cell()

        with pytest.raises(AssertionError):
            cell._update_indep_dp_info(_make_indep_dp_info())


class TestMarkAsErrored:
    def test_transitions_alive_to_errored(self):
        """_mark_as_errored from Alive transitions to Errored."""
        cell = _make_cell()
        info = _make_indep_dp_info()
        cell._mark_as_alive(indep_dp_info=info)

        cell._mark_as_errored()

        assert cell.is_errored
        assert not cell.is_alive
        assert cell.is_allocated
        assert cell.indep_dp_info == info

    def test_errored_is_idempotent(self):
        """_mark_as_errored from Errored is accepted (stays Errored)."""
        cell = _make_cell()
        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())
        cell._mark_as_errored()

        cell._mark_as_errored()

        assert cell.is_errored


class TestAsyncInit:
    def test_dispatches_init_and_marks_alive(self):
        """async_init dispatches remote init calls and marks cell as alive."""
        cell = _make_cell(actor_count=2)
        info = _make_indep_dp_info()

        refs = cell.async_init(indep_dp_info=info)

        assert len(refs) == 2
        assert cell.is_alive
        assert cell.indep_dp_info == info


class TestPrepareIndepDPModeAlive:
    def test_reconfigure_and_update_info(self):
        """prepare_indep_dp_mode_alive reconfigures actors and updates local indep_dp_info."""
        cell = _make_cell(actor_count=1)
        old_info = _make_indep_dp_info(alive_cell_indices=[0, 1, 2])
        cell._mark_as_alive(indep_dp_info=old_info)

        new_info = _make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        asyncio.get_event_loop().run_until_complete(
            cell.prepare_indep_dp_mode_alive(indep_dp_info=new_info, send_ckpt_dst_ranks=[])
        )

        assert cell.indep_dp_info == new_info
        assert cell.is_alive


class TestStatePredicates:
    """Verify is_pending/is_allocated/is_alive/is_errored/is_stopped across all states."""

    def test_pending(self):
        cell = _make_cell()
        cell.stop()
        cell.mark_as_pending()

        assert cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_uninitialized(self):
        cell = _make_cell()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_alive(self):
        cell = _make_cell()
        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())

        assert not cell.is_pending
        assert cell.is_allocated
        assert cell.is_alive
        assert not cell.is_errored
        assert not cell.is_stopped

    def test_errored(self):
        cell = _make_cell()
        cell._mark_as_alive(indep_dp_info=_make_indep_dp_info())
        cell._mark_as_errored()

        assert not cell.is_pending
        assert cell.is_allocated
        assert not cell.is_alive
        assert cell.is_errored
        assert not cell.is_stopped

    def test_stopped(self):
        cell = _make_cell()
        cell.stop()

        assert not cell.is_pending
        assert not cell.is_allocated
        assert not cell.is_alive
        assert not cell.is_errored
        assert cell.is_stopped


class TestFullLifecycle:
    @patch("miles.ray.train.cell.ray")
    def test_full_stop_start_cycle(self, mock_ray):
        """Full lifecycle: init → alive → stop → pending → allocate → alive."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return _make_mock_actors(2)

        # Step 1: Create cell (Pending → Uninitialized)
        cell = RayTrainCell(
            args=MagicMock(),
            role="actor",
            with_ref=False,
            cell_index=0,
            actor_factory=factory,
            rollout_manager=None,
        )
        assert cell.is_allocated and not cell.is_alive
        assert call_count == 1

        # Step 2: Mark as alive (Uninitialized → Alive)
        info_v1 = _make_indep_dp_info(alive_cell_indices=[0, 1, 2], quorum_id=1)
        cell._mark_as_alive(indep_dp_info=info_v1)
        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 1

        # Step 3: Stop (Alive → Stopped, actors killed)
        cell.stop()
        assert cell.is_stopped
        assert mock_ray.kill.call_count == 2

        # Step 4: Start (Stopped → Pending)
        cell.mark_as_pending()
        assert cell.is_pending

        # Step 5: Allocate (Pending → Uninitialized, new actors)
        cell.allocate_for_pending()
        assert cell.is_allocated and not cell.is_alive
        assert call_count == 2

        # Step 6: Mark as alive again with new config (Uninitialized → Alive)
        info_v2 = _make_indep_dp_info(alive_cell_indices=[0, 2], quorum_id=2)
        cell._mark_as_alive(indep_dp_info=info_v2)
        assert cell.is_alive
        assert cell.indep_dp_info.quorum_id == 2
        assert cell.indep_dp_info.alive_size == 2
