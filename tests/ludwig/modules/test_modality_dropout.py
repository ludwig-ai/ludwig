"""Tests for modality dropout."""

import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.modules.modality_dropout import ModalityDropout


def _make_encoder_outputs(batch_size=8, hidden_size=16, num_features=3):
    outputs = {}
    for i in range(num_features):
        name = f"feature_{i}"
        outputs[name] = {ENCODER_OUTPUT: torch.randn(batch_size, hidden_size)}
    return outputs


def _make_feature_shapes(hidden_size=16, num_features=3):
    return {f"feature_{i}": torch.Size([hidden_size]) for i in range(num_features)}


class TestModalityDropout:
    def test_eval_mode_passthrough(self):
        """In eval mode, outputs should be unchanged."""
        md = ModalityDropout(_make_feature_shapes(), dropout_prob=1.0)
        md.eval()
        inputs = _make_encoder_outputs()
        outputs = md(inputs)
        for name in inputs:
            assert torch.equal(inputs[name][ENCODER_OUTPUT], outputs[name][ENCODER_OUTPUT])

    def test_train_mode_zero_prob_passthrough(self):
        """With dropout_prob=0, outputs should be unchanged even in train mode."""
        md = ModalityDropout(_make_feature_shapes(), dropout_prob=0.0)
        md.train()
        inputs = _make_encoder_outputs()
        outputs = md(inputs)
        for name in inputs:
            assert torch.equal(inputs[name][ENCODER_OUTPUT], outputs[name][ENCODER_OUTPUT])

    def test_train_mode_full_dropout(self):
        """With dropout_prob=1.0 in train mode, all features should be replaced."""
        md = ModalityDropout(_make_feature_shapes(), dropout_prob=1.0)
        md.train()
        inputs = _make_encoder_outputs()
        outputs = md(inputs)
        for name in inputs:
            # Output should be the learned missing embedding, not the original
            assert not torch.equal(inputs[name][ENCODER_OUTPUT], outputs[name][ENCODER_OUTPUT])

    def test_missing_embeddings_are_learnable(self):
        """Missing embeddings should be in named_parameters and receive gradients."""
        md = ModalityDropout(_make_feature_shapes(), dropout_prob=1.0)
        md.train()

        param_names = [n for n, _ in md.named_parameters()]
        assert len(param_names) == 3
        assert all("missing_embeddings" in n for n in param_names)

        inputs = _make_encoder_outputs()
        outputs = md(inputs)
        loss = sum(o[ENCODER_OUTPUT].sum() for o in outputs.values())
        loss.backward()
        for name, param in md.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_output_shapes_preserved(self):
        """Output shapes should match input shapes."""
        md = ModalityDropout(_make_feature_shapes(hidden_size=32), dropout_prob=0.5)
        md.train()
        inputs = _make_encoder_outputs(batch_size=4, hidden_size=32)
        outputs = md(inputs)
        for name in inputs:
            assert outputs[name][ENCODER_OUTPUT].shape == inputs[name][ENCODER_OUTPUT].shape

    def test_partial_dropout(self):
        """With intermediate dropout_prob, some features should be dropped and others kept."""
        torch.manual_seed(42)
        md = ModalityDropout(_make_feature_shapes(num_features=10), dropout_prob=0.5)
        md.train()
        inputs = _make_encoder_outputs(num_features=10)
        outputs = md(inputs)
        changed = sum(
            1 for name in inputs if not torch.equal(inputs[name][ENCODER_OUTPUT], outputs[name][ENCODER_OUTPUT])
        )
        # With 10 features and p=0.5, expect roughly 5 changed (but stochastic)
        assert 0 < changed < 10
