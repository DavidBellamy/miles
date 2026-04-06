"""Tests for miles.utils.witness.module: _DataWitness, install_witness, _record_and_log_witness_param, clear_witness_stale_rows."""

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from miles.utils.witness.allocator import WitnessInfo
from miles.utils.witness.module import (
    _DataWitness,
    _record_and_log_witness_param,
    _zero_witness_rows,
    install_witness,
    witness_dump_and_clear_stale,
)


class TestDataWitnessForward:
    def test_forward_does_not_change_hidden_states(self) -> None:
        """Witness output is zero, so hidden_states should be unchanged."""
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[0, 1, 2, 3]])  # [1, 4]
        hidden = torch.randn(4, 1, 8)  # [s, b, h] Megatron SBH layout
        result = witness(ids, hidden)
        assert torch.equal(result, hidden)

    def test_forward_unchanged_after_optimizer_step(self) -> None:
        witness = _DataWitness(buffer_size=10)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.1)

        ids = torch.tensor([[0, 1, 2]])
        hidden = torch.randn(3, 1, 8)
        result = witness(ids, hidden)
        result.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        # After optimizer update, weights are nonzero, but hidden_states still unchanged
        assert not torch.all(witness.witness.weight == 0.0)
        result2 = witness(ids, hidden)
        assert torch.equal(result2, hidden)

    def test_backward_records_gradient_on_witness_weight(self) -> None:
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[2, 5]])
        hidden = torch.randn(2, 1, 4, requires_grad=True)

        result = witness(ids, hidden)
        result.sum().backward()

        grad = witness.witness.weight.grad
        assert grad is not None
        nonzero_rows = grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()
        assert set(nonzero_rows) == {2, 5}

    def test_no_effect_on_main_model_gradients(self) -> None:
        """Witness should not alter gradients for main model parameters."""
        torch.manual_seed(42)
        linear = nn.Linear(8, 1)

        hidden = torch.randn(4, 1, 8, requires_grad=True)

        # Compute loss without witness
        out_no_witness = linear(hidden).sum()
        out_no_witness.backward()
        grad_linear_no = linear.weight.grad.clone()

        linear.zero_grad()
        if hidden.grad is not None:
            hidden.grad.zero_()

        # Compute loss with witness
        witness = _DataWitness(buffer_size=10)
        ids = torch.tensor([[0, 0, 0, 0]])
        h = witness(ids, hidden)  # hidden unchanged (witness output is zero)
        out_with_witness = linear(h).sum()
        out_with_witness.backward()

        assert torch.equal(grad_linear_no, linear.weight.grad)


class TestRecordAndLogWitnessParam:
    def test_logs_nonzero_weight_rows(self) -> None:
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[3] = 1.0
        witness.witness.weight.data[7] = 2.0

        with patch("miles.utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.head", stale_ids=[])

            mock_logger.log.assert_called_once()
            # New API: log(event_cls, partial_dict)
            partial = mock_logger.log.call_args[0][1]
            assert set(partial["nonzero_witness_ids"]) == {3, 7}
            assert partial["instance_id"] == "pp0.head"

    def test_record_and_log_event_fields(self) -> None:
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[1] = 0.5
        witness.witness.weight.data[4] = -0.3

        with patch("miles.utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.tail", stale_ids=[])

            mock_logger.log.assert_called_once()
            from miles.utils.event_logger.models import WitnessSnapshotParamEvent

            assert mock_logger.log.call_args[0][0] is WitnessSnapshotParamEvent
            partial = mock_logger.log.call_args[0][1]
            assert partial["instance_id"] == "pp0.tail"
            assert set(partial["nonzero_witness_ids"]) == {1, 4}


# ---------------------------------------------------------------------------
# Fake GPTModel for install_witness / forward integration tests
# ---------------------------------------------------------------------------


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_tensor: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return hidden_states


class _FakeGPTModel(nn.Module):
    def __init__(self, *, pre_process: bool = True) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.decoder = _FakeDecoder()
        self.embedding = nn.Embedding(100, 16)

    def forward(self, input_ids: torch.Tensor, witness_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_process:
            decoder_input = self.embedding(input_ids)
        else:
            decoder_input = None

        if hasattr(self, "local_head_witness") and witness_ids is not None:
            if decoder_input is not None:
                decoder_input = self.local_head_witness(witness_ids, decoder_input)
            else:
                self.decoder.input_tensor = self.local_head_witness(witness_ids, self.decoder.input_tensor)

        if decoder_input is None:
            decoder_input = self.decoder.input_tensor

        return self.decoder(hidden_states=decoder_input)


class TestInstallWitness:
    def test_witness_is_submodule(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        assert "local_head_witness" in dict(model.named_modules())
        assert "local_tail_witness" in dict(model.named_modules())

    def test_witness_in_parameters(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        param_names = [name for name, _ in model.named_parameters()]
        assert any("local_head_witness" in name for name in param_names)
        assert any("local_tail_witness" in name for name in param_names)

    def test_forward_adds_zero(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out_no = model(tokens)
        out_with = model(tokens, witness_ids=torch.tensor([[0, 1, 2]]))
        assert torch.equal(out_no, out_with)

    def test_forward_produces_gradient(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        tokens = torch.tensor([[1, 2, 3]])
        out = model(tokens, witness_ids=torch.tensor([[5, 5, 5]]))
        out.sum().backward()
        grad = model.local_head_witness.witness.weight.grad
        assert grad is not None
        assert 5 in grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_no_witness_ids_no_effect(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        out = model(torch.tensor([[1, 2, 3]]))
        assert out is not None

    def test_witness_in_state_dict(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        sd = model.state_dict()
        assert any("local_head_witness" in k for k in sd)
        assert any("local_tail_witness" in k for k in sd)

    def test_checkpoint_roundtrip(self) -> None:
        model = _FakeGPTModel()
        install_witness(model, buffer_size=10)
        model.local_head_witness.witness.weight.data.fill_(42.0)
        sd = model.state_dict()

        model2 = _FakeGPTModel()
        install_witness(model2, buffer_size=10)
        model2.load_state_dict(sd)
        assert torch.equal(model2.local_head_witness.witness.weight.data, model.local_head_witness.witness.weight.data)

    def test_disabled_no_submodule(self) -> None:
        model = _FakeGPTModel()
        assert not hasattr(model, "local_head_witness")

    def test_middle_pp_stage_modifies_input_tensor(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        hidden = torch.randn(1, 4, 16)
        model.decoder.input_tensor = hidden.clone()
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[0, 1, 2, 3]]))
        assert torch.equal(out, hidden)

    def test_middle_pp_stage_produces_gradient(self) -> None:
        model = _FakeGPTModel(pre_process=False)
        install_witness(model, buffer_size=10)
        model.decoder.input_tensor = torch.randn(1, 4, 16, requires_grad=True)
        out = model(torch.tensor([[1, 2, 3, 4]]), witness_ids=torch.tensor([[5, 5, 5, 5]]))
        out.sum().backward()
        assert 5 in model.local_head_witness.witness.weight.grad.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    def test_forward_bitwise_zero_bf16(self) -> None:
        witness = _DataWitness(buffer_size=10).to(dtype=torch.bfloat16)
        witness.witness.weight.data.fill_(3.14)
        ids = torch.tensor([[0, 1, 2]])
        hidden = torch.randn(3, 1, 8, dtype=torch.bfloat16)
        result = witness(ids, hidden)
        assert torch.equal(result, hidden)


class TestZeroWitnessRows:
    def test_weight_is_zeroed(self) -> None:
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data.fill_(1.0)
        optimizer = torch.optim.Adam(witness.parameters(), lr=0.01)

        idx = torch.tensor([2, 5, 7])
        _zero_witness_rows(witness=witness, idx=idx, optimizer=optimizer)

        for i in [2, 5, 7]:
            assert witness.witness.weight.data[i].item() == 0.0
        for i in [0, 1, 3, 4, 6, 8, 9]:
            assert witness.witness.weight.data[i].item() == 1.0

    def test_optimizer_state_is_zeroed(self) -> None:
        """After an optimizer step, exp_avg and exp_avg_sq should be zeroed for stale rows."""
        witness = _DataWitness(buffer_size=10)

        optimizer = torch.optim.Adam(witness.parameters(), lr=0.01)
        ids = torch.arange(10)
        out = witness(ids, ids)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        weight = witness.witness.weight
        state = optimizer.state[weight]
        assert not torch.all(state["exp_avg"] == 0.0)

        stale_idx = torch.tensor([3, 6])
        _zero_witness_rows(witness=witness, idx=stale_idx, optimizer=optimizer)

        assert weight.data[3].item() == 0.0
        assert weight.data[6].item() == 0.0

        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key][3].item() == 0.0
            assert state[key][6].item() == 0.0
            non_stale = [i for i in range(10) if i not in [3, 6]]
            assert not torch.all(state[key][non_stale] == 0.0)

    def test_zero_witness_rows_clears_main_param(self) -> None:
        """When weight has a main_param attribute (Megatron mixed precision), _zero_witness_rows zeroes main_param.data and optimizer state keyed by main_param."""
        witness = _DataWitness(buffer_size=10)

        # Step 1: Set nonzero weight
        witness.witness.weight.data[3] = 1.0

        # Step 2: Create main_param and attach to weight
        main_param = torch.ones(10, 1)
        witness.witness.weight.main_param = main_param

        # Step 3: Build optimizer keyed on main_param (as Megatron does)
        optimizer = torch.optim.Adam([main_param], lr=0.01)
        main_param.grad = torch.ones_like(main_param)
        optimizer.step()  # initialize optimizer state
        optimizer.zero_grad()

        # Step 4: Call _zero_witness_rows
        idx = torch.tensor([3])
        _zero_witness_rows(witness=witness, idx=idx, optimizer=optimizer)

        # Step 5: Verify main_param.data is zeroed at idx
        assert main_param.data[3].item() == 0.0
        assert main_param.data[0].item() != 0.0

        # Step 6: Verify optimizer state keyed by main_param is zeroed at idx
        state = optimizer.state[main_param]
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key][3].item() == 0.0
            assert state[key][0].item() != 0.0


# ---------------------------------------------------------------------------
# Helpers for witness_dump_and_clear_stale tests
# ---------------------------------------------------------------------------


def _make_fake_chunk(buffer_size: int = 10) -> nn.Module:
    """Create a fake model chunk with .module.local_head_witness and .module.local_tail_witness."""
    inner = nn.Module()
    inner.local_head_witness = _DataWitness(buffer_size=buffer_size)
    inner.local_tail_witness = _DataWitness(buffer_size=buffer_size)
    chunk = nn.Module()
    chunk.module = inner
    return chunk


class TestWitnessDumpAndClearStale:
    def test_witness_dump_and_clear_stale_logs_all_witnesses(self) -> None:
        """2 chunks x 2 witnesses = 4 log events with correct instance_ids."""
        chunk0 = _make_fake_chunk()
        chunk1 = _make_fake_chunk()
        chunk0.module.local_head_witness.witness.weight.data[1] = 1.0
        chunk0.module.local_tail_witness.witness.weight.data[2] = 1.0
        chunk1.module.local_head_witness.witness.weight.data[3] = 1.0
        chunk1.module.local_tail_witness.witness.weight.data[4] = 1.0

        model = [chunk0, chunk1]
        all_params = list(chunk0.parameters()) + list(chunk1.parameters())
        optimizer = torch.optim.Adam(all_params, lr=0.01)
        witness_info = WitnessInfo(witness_ids=[1, 2, 3, 4], stale_ids=[5, 6])

        with patch("miles.utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

            assert mock_logger.log.call_count == 4
            logged_instance_ids = [call[0][1]["instance_id"] for call in mock_logger.log.call_args_list]
            assert logged_instance_ids == ["pp0.head", "pp0.tail", "pp1.head", "pp1.tail"]

            logged_stale_ids = [call[0][1]["stale_ids"] for call in mock_logger.log.call_args_list]
            for stale in logged_stale_ids:
                assert stale == [5, 6]

    def test_witness_dump_and_clear_stale_clears_stale_rows(self) -> None:
        """Stale IDs should have their weight rows zeroed after the call."""
        chunk = _make_fake_chunk(buffer_size=10)
        chunk.module.local_head_witness.witness.weight.data.fill_(1.0)
        chunk.module.local_tail_witness.witness.weight.data.fill_(1.0)

        model = [chunk]
        optimizer = torch.optim.Adam(chunk.parameters(), lr=0.01)
        witness_info = WitnessInfo(witness_ids=[0], stale_ids=[3, 7])

        with patch("miles.utils.witness.module.get_event_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        for witness_attr in ("local_head_witness", "local_tail_witness"):
            witness = getattr(chunk.module, witness_attr)
            assert witness.witness.weight.data[3].item() == 0.0
            assert witness.witness.weight.data[7].item() == 0.0
            assert witness.witness.weight.data[0].item() == 1.0

    def test_witness_dump_and_clear_stale_empty_stale_ids(self) -> None:
        """Empty stale_ids should not trigger any zeroing."""
        chunk = _make_fake_chunk(buffer_size=10)
        chunk.module.local_head_witness.witness.weight.data.fill_(1.0)
        chunk.module.local_tail_witness.witness.weight.data.fill_(1.0)

        model = [chunk]
        optimizer = torch.optim.Adam(chunk.parameters(), lr=0.01)
        witness_info = WitnessInfo(witness_ids=[0], stale_ids=[])

        with patch("miles.utils.witness.module.get_event_logger") as mock_get_logger:
            mock_get_logger.return_value = MagicMock()
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        for witness_attr in ("local_head_witness", "local_tail_witness"):
            witness = getattr(chunk.module, witness_attr)
            assert torch.all(witness.witness.weight.data == 1.0)

    def test_record_and_log_witness_param_includes_stale_ids(self) -> None:
        """Log event should contain the correct stale_ids field."""
        witness = _DataWitness(buffer_size=10)
        witness.witness.weight.data[2] = 1.0

        with patch("miles.utils.witness.module.get_event_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            _record_and_log_witness_param(witness=witness, instance_id="pp0.head", stale_ids=[8, 9])

            mock_logger.log.assert_called_once()
            partial = mock_logger.log.call_args[0][1]
            assert partial["stale_ids"] == [8, 9]
