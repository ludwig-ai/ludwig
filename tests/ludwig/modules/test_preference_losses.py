"""Tests for KTO, ORPO, and GRPO loss functions."""

import torch

from ludwig.modules.dpo_loss import compute_token_log_probs
from ludwig.modules.preference_losses import grpo_loss, kto_loss, orpo_loss

BATCH = 4
SEQ_LEN = 20
VOCAB = 100


def _random_inputs():
    chosen_logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    rejected_logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    chosen_labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    rejected_labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    return chosen_logits, rejected_logits, chosen_labels, rejected_labels


class TestKTOLoss:
    def test_basic(self):
        loss, cr, rr = kto_loss(*_random_inputs())
        assert torch.isfinite(loss)
        assert loss.shape == ()

    def test_gradient_flow(self):
        chosen_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
        rejected_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        loss, _, _ = kto_loss(chosen_logits, rejected_logits, labels, labels)
        loss.backward()
        assert chosen_logits.grad is not None
        assert rejected_logits.grad is not None


class TestORPOLoss:
    def test_basic(self):
        loss, cr, rr = orpo_loss(*_random_inputs())
        assert torch.isfinite(loss)
        assert loss.shape == ()

    def test_gradient_flow(self):
        chosen_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
        rejected_logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        loss, _, _ = orpo_loss(chosen_logits, rejected_logits, labels, labels)
        loss.backward()
        assert chosen_logits.grad is not None


class TestGRPOLoss:
    def test_basic(self):
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        rewards = torch.randn(BATCH)
        loss, adv = grpo_loss(logits, labels, rewards)
        assert torch.isfinite(loss)
        assert loss.shape == ()

    def test_with_old_log_probs(self):
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        rewards = torch.randn(BATCH)
        old_lp = torch.randn(BATCH)
        loss, _ = grpo_loss(logits, labels, rewards, old_log_probs=old_lp)
        assert torch.isfinite(loss)

    def test_with_reference(self):
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        rewards = torch.randn(BATCH)
        ref_lp = torch.randn(BATCH)
        loss, _ = grpo_loss(logits, labels, rewards, reference_log_probs=ref_lp)
        assert torch.isfinite(loss)

    def test_gradient_flow(self):
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB, requires_grad=True)
        labels = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        # Rewards must be detached (they come from a reward model, not the policy)
        # but old_log_probs connect to the policy graph
        rewards = torch.randn(BATCH)
        old_lp = compute_token_log_probs(logits.detach(), labels)
        loss, _ = grpo_loss(logits, labels, rewards, old_log_probs=old_lp.detach())
        loss.backward()
        assert logits.grad is not None
