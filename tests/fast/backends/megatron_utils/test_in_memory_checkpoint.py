from unittest.mock import patch

import pytest

from miles.backends.megatron_utils.in_memory_checkpoint import InMemoryCheckpointManager


@pytest.fixture()
def manager():
    with patch("miles.backends.megatron_utils.in_memory_checkpoint.get_args") as mock_get_args:
        mock_args = mock_get_args.return_value
        mock_args.non_persistent_ckpt_type = "local"
        mock_args.non_persistent_local_ckpt_algo = "fully_parallel"
        yield InMemoryCheckpointManager()


class TestInMemoryCheckpointManager:
    def test_find_latest_returns_minus_one_initially(self, manager: InMemoryCheckpointManager):
        assert manager.find_latest() == -1

    def test_load_before_save_raises(self, manager: InMemoryCheckpointManager):
        with pytest.raises(AssertionError, match="No in-memory checkpoint"):
            manager.load()

    def test_save_then_load_returns_same_object(self, manager: InMemoryCheckpointManager):
        sentinel = object()
        manager.save(state_dict=sentinel, iteration=5)

        assert manager.find_latest() == 5

        result, name = manager.load()
        assert result is sentinel
        assert "5" in name

    def test_load_clears_state_so_second_load_raises(self, manager: InMemoryCheckpointManager):
        manager.save(state_dict=object(), iteration=1)
        manager.load()

        with pytest.raises(AssertionError):
            manager.load()

    def test_save_twice_without_load_raises(self, manager: InMemoryCheckpointManager):
        manager.save(state_dict=object(), iteration=1)

        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=2)

    def test_save_load_save_load_cycle(self, manager: InMemoryCheckpointManager):
        obj1 = object()
        manager.save(state_dict=obj1, iteration=1)
        result1, _ = manager.load()
        assert result1 is obj1

        obj2 = object()
        manager.save(state_dict=obj2, iteration=2)
        result2, _ = manager.load()
        assert result2 is obj2

    def test_async_save_raises(self, manager: InMemoryCheckpointManager):
        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=1, is_async=True)
