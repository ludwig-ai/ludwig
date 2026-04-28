"""Unit tests for the Flamingo-style gated cross-attention module."""

from __future__ import annotations

import pytest
import torch

from ludwig.modules.gated_cross_attention import GatedCrossAttention


class TestGatedCrossAttention:
    def test_identity_at_init(self):
        """Zero-init gates must make the block an identity at step 0."""
        torch.manual_seed(0)
        block = GatedCrossAttention(d_model=32, num_heads=4)
        x = torch.randn(2, 7, 32)
        kv = torch.randn(2, 5, 32)

        out = block(x, kv)
        torch.testing.assert_close(out, x, atol=1e-6, rtol=1e-6)

    def test_non_identity_after_gate_nudge(self):
        torch.manual_seed(0)
        block = GatedCrossAttention(d_model=32, num_heads=4)
        with torch.no_grad():
            block.attn_gate.fill_(1.0)
            block.ffn_gate.fill_(1.0)
        x = torch.randn(2, 7, 32)
        kv = torch.randn(2, 5, 32)

        out = block(x, kv)
        # Shape preserved...
        assert out.shape == x.shape
        # ...but values changed by a meaningful amount.
        assert not torch.allclose(out, x, atol=1e-3)

    def test_different_kv_dim(self):
        """kv_dim != d_model should project keys/values to d_model."""
        block = GatedCrossAttention(d_model=32, num_heads=4, kv_dim=64)
        x = torch.randn(2, 4, 32)
        kv = torch.randn(2, 6, 64)
        out = block(x, kv)
        assert out.shape == x.shape

    def test_key_padding_mask(self):
        """Padding mask is passed through to the inner attention layer without crashing."""
        block = GatedCrossAttention(d_model=16, num_heads=4)
        x = torch.randn(2, 3, 16)
        kv = torch.randn(2, 4, 16)
        mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
        out = block(x, kv, key_padding_mask=mask)
        assert out.shape == x.shape

    @pytest.mark.parametrize("tanh_gate", [True, False])
    def test_tanh_gate_toggle(self, tanh_gate):
        block = GatedCrossAttention(d_model=16, num_heads=4, tanh_gate=tanh_gate)
        assert block.tanh_gate is tanh_gate

    def test_backward_pass(self):
        block = GatedCrossAttention(d_model=32, num_heads=4)
        x = torch.randn(2, 7, 32, requires_grad=True)
        kv = torch.randn(2, 5, 32)
        out = block(x, kv)
        out.sum().backward()
        assert x.grad is not None
        assert block.attn_gate.grad is not None
        assert block.ffn_gate.grad is not None


class TestLLMIsMultimodalSchema:
    """The VLM flag on LLMModelConfig should parse, default to False, and have no ill effects."""

    def _base(self) -> dict:
        return {
            "model_type": "llm",
            "base_model": "hf-internal-testing/tiny-random-GPTJForCausalLM",
            "input_features": [{"name": "prompt", "type": "text"}],
            "output_features": [{"name": "output", "type": "text"}],
        }

    def test_default_false(self):
        from ludwig.schema.model_types.base import ModelConfig

        cfg = ModelConfig.from_dict(self._base())
        assert cfg.is_multimodal is False

    def test_explicit_true(self):
        from ludwig.schema.model_types.base import ModelConfig

        cfg = ModelConfig.from_dict({**self._base(), "is_multimodal": True})
        assert cfg.is_multimodal is True


class TestMultimodalCollator:
    def test_collator_uses_processor_path(self):
        """The collator should call the processor with text + images and return its batch."""
        from ludwig.data.multimodal_collator import MultimodalCollator

        class _FakeProcessor:
            def __init__(self):
                self.tokenizer = None
                self.last_call = None

            def __call__(self, *, text, images, return_tensors, padding, **kwargs):
                self.last_call = {"text": text, "images": images, "padding": padding, **kwargs}
                return {"input_ids": torch.zeros(len(text), 4, dtype=torch.long)}

        proc = _FakeProcessor()
        collator = MultimodalCollator(proc)
        examples = [
            {"image": "img0", "text": "hi"},
            {"image": "img1", "text": "hello"},
        ]
        batch = collator(examples)
        assert "input_ids" in batch
        assert proc.last_call["text"] == ["hi", "hello"]
        assert proc.last_call["images"] == ["img0", "img1"]

    def test_collator_labels_path_masks_padding(self):
        from ludwig.data.multimodal_collator import MultimodalCollator

        class _FakeTokenizer:
            pad_token_id = 0

            def __call__(self, texts, return_tensors, padding, truncation, max_length=None):
                # Toy tokenizer: right-pad with 0s, all ids are 1.
                max_len = max(len(t) for t in texts)
                ids = torch.zeros(len(texts), max_len, dtype=torch.long)
                for i, t in enumerate(texts):
                    ids[i, : len(t)] = 1
                return {"input_ids": ids}

        class _FakeProcessor:
            def __init__(self):
                self.tokenizer = _FakeTokenizer()

            def __call__(self, *, text, images, return_tensors, padding, **kwargs):
                return {"input_ids": torch.zeros(len(text), 3, dtype=torch.long)}

        collator = MultimodalCollator(_FakeProcessor(), max_length=8)
        examples = [
            {"image": "img0", "text": "hi", "labels": "y"},
            {"image": "img1", "text": "hey", "labels": "longer"},
        ]
        batch = collator(examples)
        assert "labels" in batch
        # Padding positions replaced with -100; valid token positions kept as 1.
        assert (batch["labels"] == -100).any()
        assert (batch["labels"] == 1).any()
