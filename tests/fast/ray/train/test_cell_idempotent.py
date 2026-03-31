from miles.ray.train.cell import RayTrainCell, _StatePending, _StateRunning, _StateStopped


def _make_cell_with_state(state) -> RayTrainCell:
    """Create a RayTrainCell with a pre-set state, bypassing __init__."""
    cell = object.__new__(RayTrainCell)
    cell.cell_id = 0
    cell._state = state
    return cell


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

    def test_mark_as_pending_already_running_is_noop(self):
        """Calling mark_as_pending() on a running cell does not crash and state stays Running."""
        cell = _make_cell_with_state(_StateRunning(actor_handles=[]))

        cell.mark_as_pending()

        assert cell.is_running
