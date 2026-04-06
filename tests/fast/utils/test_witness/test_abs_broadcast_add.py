import pytest
import torch

from miles.utils.witness.module import _AbsBroadcastAdd, witness_broadcast_add


class TestAbsBroadcastAddForward:
    def test_forward_value_matches_plain_addition(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 2, 1)
        result = witness_broadcast_add(hidden, addend)
        expected = hidden + addend
        assert torch.equal(result, expected)

    def test_forward_preserves_zero_addend(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.zeros(4, 2, 1)
        result = witness_broadcast_add(hidden, addend)
        assert torch.equal(result, hidden)

    def test_forward_assert_addend_last_dim_must_be_1(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 2, 3)
        with pytest.raises(AssertionError, match="addend last dim must be 1"):
            witness_broadcast_add(hidden, addend)

    def test_forward_assert_leading_dims_must_match(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(4, 3, 1)  # second dim differs
        with pytest.raises(AssertionError, match="must match on all dims except last"):
            witness_broadcast_add(hidden, addend)

    def test_forward_assert_ndim_must_match(self) -> None:
        hidden = torch.randn(4, 2, 8)
        addend = torch.randn(2, 1)  # 2D vs 3D
        with pytest.raises(AssertionError, match="must match on all dims except last"):
            witness_broadcast_add(hidden, addend)


class TestAbsBroadcastAddBackwardHiddenStates:
    def test_hidden_states_gradient_is_pass_through(self) -> None:
        hidden = torch.randn(4, 2, 8, requires_grad=True)
        addend = torch.randn(4, 2, 1, requires_grad=True)
        result = witness_broadcast_add(hidden, addend)
        loss = result.sum()
        loss.backward()
        # hidden gradient = all ones (pass-through from sum)
        assert torch.equal(hidden.grad, torch.ones_like(hidden))


class TestAbsBroadcastAddBackwardAddend:
    def test_addend_gradient_is_abs_sum_over_last_dim(self) -> None:
        hidden = torch.randn(3, 2, 4, requires_grad=True)
        addend = torch.zeros(3, 2, 1, requires_grad=True)

        result = witness_broadcast_add(hidden, addend)
        # Use a loss that produces a known gradient at result
        upstream_grad = torch.tensor([[[1.0, -2.0, 3.0, -4.0],
                                       [0.5, -0.5, 0.5, -0.5]],
                                      [[1.0, 1.0, 1.0, 1.0],
                                       [-1.0, -1.0, -1.0, -1.0]],
                                      [[0.0, 0.0, 0.0, 0.0],
                                       [2.0, -1.0, 0.5, -0.3]]])
        result.backward(upstream_grad)

        # Expected addend grad: abs(upstream_grad).sum(dim=-1, keepdim=True)
        expected = upstream_grad.abs().sum(dim=-1, keepdim=True)
        assert torch.allclose(addend.grad, expected)

    def test_addend_gradient_no_cancellation_with_mixed_signs(self) -> None:
        """The key property: mixed-sign gradients don't cancel to zero."""
        hidden = torch.randn(1, 1, 8, requires_grad=True)
        addend = torch.zeros(1, 1, 1, requires_grad=True)

        result = witness_broadcast_add(hidden, addend)
        # Gradient with equal positive and negative values (sums to zero normally)
        upstream_grad = torch.tensor([[[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]])
        result.backward(upstream_grad)

        # Plain broadcast backward would give sum = 0
        assert upstream_grad.sum(dim=-1, keepdim=True).item() == 0.0
        # But abs broadcast gives sum of absolute values = 8
        assert addend.grad.item() == 8.0

    def test_addend_gradient_matches_plain_sum_when_all_positive(self) -> None:
        """When all gradients are positive, abs().sum() == sum()."""
        hidden = torch.randn(2, 1, 4, requires_grad=True)
        addend = torch.zeros(2, 1, 1, requires_grad=True)

        result = witness_broadcast_add(hidden, addend)
        upstream_grad = torch.abs(torch.randn(2, 1, 4))  # all positive
        result.backward(upstream_grad)

        expected = upstream_grad.sum(dim=-1, keepdim=True)
        assert torch.allclose(addend.grad, expected)

    def test_addend_gradient_always_non_negative(self) -> None:
        """abs().sum() is always >= 0."""
        hidden = torch.randn(10, 5, 16, requires_grad=True)
        addend = torch.zeros(10, 5, 1, requires_grad=True)

        result = witness_broadcast_add(hidden, addend)
        upstream_grad = torch.randn(10, 5, 16)
        result.backward(upstream_grad)

        assert (addend.grad >= 0).all()


class TestAbsBroadcastAddGradientFlow:
    def test_gradient_flows_through_to_embedding(self) -> None:
        """End-to-end: gradient reaches embedding weight via abs broadcast add."""
        vocab_size = 16
        hidden_dim = 8
        seq_len = 4

        embedding = torch.nn.Embedding(vocab_size, 1)
        torch.nn.init.zeros_(embedding.weight)
        output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        witness_ids = torch.tensor([[0, 1, 2, 3]])  # [1, seq_len]
        w = embedding(witness_ids)  # [1, 4, 1]
        out = w - w.detach()

        hidden_states = torch.randn(seq_len, 1, hidden_dim, requires_grad=True)
        tail_out = out.transpose(0, 1).contiguous()  # [4, 1, 1]

        combined = witness_broadcast_add(hidden_states, tail_out)
        logits = output_layer(combined)
        loss = logits.sum()
        loss.backward()

        # All 4 witness_ids should have nonzero gradient
        nonzero_rows = (embedding.weight.grad.abs() > 0).squeeze(-1)
        assert nonzero_rows[:4].all(), f"Expected rows 0-3 nonzero, got {embedding.weight.grad[:4]}"

    def test_gradient_nonzero_even_when_plain_broadcast_cancels(self) -> None:
        """Simulates the exact scenario: output_layer gradient cancels under plain broadcast."""
        vocab_size = 4
        hidden_dim = 4
        seq_len = 2

        embedding = torch.nn.Embedding(8, 1)
        torch.nn.init.zeros_(embedding.weight)

        # Construct output_layer weight where column sums are constant
        # → plain broadcast gradient is exactly zero (softmax sum-to-zero property)
        W = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0]])
        output_layer = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        output_layer.weight.data = W

        witness_ids = torch.tensor([[0, 1]])
        w = embedding(witness_ids)
        out = w - w.detach()

        hidden_states = torch.randn(seq_len, 1, hidden_dim, requires_grad=True)
        tail_out = out.transpose(0, 1).contiguous()

        # With abs broadcast add, gradient should be nonzero
        combined = witness_broadcast_add(hidden_states, tail_out)
        logits = output_layer(combined)
        targets = torch.tensor([[0, 1]])
        log_probs = torch.nn.functional.log_softmax(logits.squeeze(1), dim=-1)
        loss = -log_probs.gather(1, targets.T).sum()
        loss.backward()

        assert embedding.weight.grad is not None
        assert (embedding.weight.grad[:2].abs() > 0).all(), (
            f"Expected nonzero grad for witness rows 0,1, got {embedding.weight.grad[:2]}"
        )


class TestAbsBroadcastAddDoubleBackward:
    def test_gradcheck(self) -> None:
        """Verify numerical gradient correctness with torch.autograd.gradcheck."""
        hidden = torch.randn(2, 2, 4, dtype=torch.float64, requires_grad=True)
        addend = torch.randn(2, 2, 1, dtype=torch.float64, requires_grad=True)
        # gradcheck only tests hidden_states gradient (which is pass-through)
        # For addend, abs() is not differentiable at 0, so we test separately
        assert torch.autograd.gradcheck(
            lambda h: _AbsBroadcastAdd.apply(h, addend.detach()),
            (hidden,),
        )
