"""Tests for DPO loss function."""

import torch

from ludwig.modules.dpo_loss import compute_token_log_probs, dpo_loss


class TestComputeTokenLogProbs:
    def test_basic_shape(self):
        logits = torch.randn(2, 10, 100)  # batch=2, seq_len=10, vocab=100
        labels = torch.randint(0, 100, (2, 10))
        result = compute_token_log_probs(logits, labels)
        assert result.shape == (2,)

    def test_ignores_minus_100(self):
        logits = torch.randn(1, 5, 100)
        labels = torch.tensor([[-100, -100, 3, 4, 5]])
        result = compute_token_log_probs(logits, labels)
        assert torch.isfinite(result).all()

    def test_all_ignored_gives_zero(self):
        logits = torch.randn(1, 5, 100)
        labels = torch.full((1, 5), -100)
        result = compute_token_log_probs(logits, labels)
        assert result.item() == 0.0


class TestDPOLoss:
    def test_sigmoid_loss_basic(self):
        batch, seq_len, vocab = 4, 20, 100
        chosen_logits = torch.randn(batch, seq_len, vocab)
        rejected_logits = torch.randn(batch, seq_len, vocab)
        chosen_labels = torch.randint(0, vocab, (batch, seq_len))
        rejected_labels = torch.randint(0, vocab, (batch, seq_len))

        loss, chosen_rewards, rejected_rewards = dpo_loss(
            chosen_logits, rejected_logits, chosen_labels, rejected_labels, beta=0.1
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss > 0  # DPO loss is always positive

    def test_ipo_loss(self):
        batch, seq_len, vocab = 2, 10, 50
        chosen_logits = torch.randn(batch, seq_len, vocab)
        rejected_logits = torch.randn(batch, seq_len, vocab)
        chosen_labels = torch.randint(0, vocab, (batch, seq_len))
        rejected_labels = torch.randint(0, vocab, (batch, seq_len))

        loss, _, _ = dpo_loss(chosen_logits, rejected_logits, chosen_labels, rejected_labels, beta=0.1, loss_type="ipo")
        assert torch.isfinite(loss)

    def test_label_smoothing(self):
        batch, seq_len, vocab = 2, 10, 50
        chosen_logits = torch.randn(batch, seq_len, vocab)
        rejected_logits = torch.randn(batch, seq_len, vocab)
        chosen_labels = torch.randint(0, vocab, (batch, seq_len))
        rejected_labels = torch.randint(0, vocab, (batch, seq_len))

        loss_no_smooth, _, _ = dpo_loss(
            chosen_logits, rejected_logits, chosen_labels, rejected_labels, beta=0.1, label_smoothing=0.0
        )
        loss_smooth, _, _ = dpo_loss(
            chosen_logits, rejected_logits, chosen_labels, rejected_labels, beta=0.1, label_smoothing=0.1
        )
        # With smoothing, loss should be different
        assert not torch.allclose(loss_no_smooth, loss_smooth)

    def test_rewards_are_detached(self):
        batch, seq_len, vocab = 2, 10, 50
        chosen_logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        rejected_logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        chosen_labels = torch.randint(0, vocab, (batch, seq_len))
        rejected_labels = torch.randint(0, vocab, (batch, seq_len))

        _, chosen_rewards, rejected_rewards = dpo_loss(chosen_logits, rejected_logits, chosen_labels, rejected_labels)
        assert not chosen_rewards.requires_grad
        assert not rejected_rewards.requires_grad

    def test_gradient_flow(self):
        batch, seq_len, vocab = 2, 10, 50
        chosen_logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        rejected_logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
        chosen_labels = torch.randint(0, vocab, (batch, seq_len))
        rejected_labels = torch.randint(0, vocab, (batch, seq_len))

        loss, _, _ = dpo_loss(chosen_logits, rejected_logits, chosen_labels, rejected_labels)
        loss.backward()
        assert chosen_logits.grad is not None
        assert rejected_logits.grad is not None

    def test_invalid_loss_type_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown DPO loss type"):
            dpo_loss(
                torch.randn(1, 5, 10),
                torch.randn(1, 5, 10),
                torch.randint(0, 10, (1, 5)),
                torch.randint(0, 10, (1, 5)),
                loss_type="invalid",
            )
