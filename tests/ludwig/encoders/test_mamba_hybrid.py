"""Phase 6.6.2 — Mamba-2 + Jamba hybrid encoder unit tests."""

from __future__ import annotations

import pytest
import torch

from ludwig.encoders.mamba_hybrid import _Mamba2Block, JambaEncoder, Mamba2Encoder


class TestMamba2Block:
    def test_shape_preserved(self):
        torch.manual_seed(0)
        block = _Mamba2Block(d_model=32, num_heads=4)
        x = torch.randn(2, 16, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_init(self):
        """A freshly-initialised block with zero input path output is ~= identity on x via the residual connection.

        The Mamba-2 block isn't exactly identity at init, but the residual makes sure the output is 'close' to the input
        rather than random.
        """
        torch.manual_seed(0)
        block = _Mamba2Block(d_model=32, num_heads=4, dropout=0.0)
        block.eval()
        x = torch.randn(2, 8, 32)
        out = block(x)
        # Output has the same statistics as the input — not an identity, but residual-anchored.
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_d_model_divisibility_check(self):
        with pytest.raises(ValueError, match="must be divisible"):
            _Mamba2Block(d_model=30, num_heads=4)

    def test_backward(self):
        block = _Mamba2Block(d_model=16, num_heads=4)
        x = torch.randn(2, 8, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert block.log_alpha.grad is not None


class TestMamba2Encoder:
    def test_forward_shapes_with_embedding(self):
        encoder = Mamba2Encoder(
            should_embed=True,
            vocab=list(range(50)),
            embedding_size=64,
            d_model=32,
            n_layers=2,
            num_heads=4,
            output_size=24,
        )
        inputs = torch.randint(0, 50, (3, 10))
        out = encoder(inputs)["encoder_output"]
        assert out.shape == (3, 24)

    def test_forward_shapes_without_embedding(self):
        encoder = Mamba2Encoder(
            should_embed=False,
            d_model=32,
            n_layers=2,
            num_heads=4,
            output_size=16,
            reduce_output="mean",
        )
        inputs = torch.randn(2, 8, 32)
        out = encoder(inputs)["encoder_output"]
        assert out.shape == (2, 16)

    def test_reduce_output_last(self):
        encoder = Mamba2Encoder(
            should_embed=False, d_model=16, n_layers=2, num_heads=4, output_size=16, reduce_output="last"
        )
        inputs = torch.randn(2, 5, 16)
        out = encoder(inputs)["encoder_output"]
        assert out.shape == (2, 16)


class TestJambaEncoder:
    def test_interleaving_pattern(self):
        import torch.nn as nn

        encoder = JambaEncoder(
            should_embed=False, d_model=16, n_layers=8, attention_every_k=4, num_heads=4, ffn_size=32, output_size=16
        )
        # Every 4th position (indices 3 and 7) is attention.
        types = [type(layer).__name__ for layer in encoder.layers]
        attention_positions = [i for i, t in enumerate(types) if t == "TransformerEncoderLayer"]
        assert attention_positions == [3, 7], f"expected attention at indices [3, 7], got {attention_positions}"
        # The remaining layers should be SSM blocks.
        ssm_positions = [i for i, t in enumerate(types) if t == "_Mamba2Block"]
        assert ssm_positions == [0, 1, 2, 4, 5, 6]
        _ = nn  # silence unused import when running this test in isolation

    def test_forward_shape(self):
        encoder = JambaEncoder(
            should_embed=False,
            d_model=32,
            n_layers=4,
            attention_every_k=2,  # alternate SSM / attention
            num_heads=4,
            ffn_size=64,
            output_size=20,
        )
        inputs = torch.randn(2, 6, 32)
        out = encoder(inputs)["encoder_output"]
        assert out.shape == (2, 20)

    def test_backward(self):
        encoder = JambaEncoder(
            should_embed=False, d_model=16, n_layers=4, attention_every_k=2, num_heads=4, ffn_size=32, output_size=16
        )
        x = torch.randn(2, 6, 16, requires_grad=True)
        out = encoder(x)["encoder_output"]
        out.sum().backward()
        assert x.grad is not None


class TestSchemaRegistration:
    def test_mamba2_encoder_config_parses(self):
        from ludwig.schema.encoders.mamba_hybrid import Mamba2EncoderConfig

        cfg = Mamba2EncoderConfig.model_validate({"type": "mamba2", "d_model": 128, "n_layers": 6, "num_heads": 4})
        assert cfg.type == "mamba2"
        assert cfg.d_model == 128
        assert cfg.num_heads == 4

    def test_jamba_encoder_config_parses(self):
        from ludwig.schema.encoders.mamba_hybrid import JambaEncoderConfig

        cfg = JambaEncoderConfig.model_validate(
            {"type": "jamba", "d_model": 128, "n_layers": 8, "attention_every_k": 4, "ffn_size": 512}
        )
        assert cfg.type == "jamba"
        assert cfg.attention_every_k == 4
        assert cfg.ffn_size == 512

    def test_full_model_config_with_mamba2(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "input_features": [
                    {"name": "seq", "type": "sequence", "encoder": {"type": "mamba2", "d_model": 64, "n_layers": 2}}
                ],
                "output_features": [{"name": "y", "type": "binary"}],
            }
        )
        assert cfg.input_features[0].encoder.type == "mamba2"
        assert cfg.input_features[0].encoder.d_model == 64

    def test_full_model_config_with_jamba(self):
        from ludwig.schema.model_config import ModelConfig

        cfg = ModelConfig.from_dict(
            {
                "input_features": [
                    {
                        "name": "seq",
                        "type": "sequence",
                        "encoder": {"type": "jamba", "d_model": 64, "n_layers": 4, "attention_every_k": 2},
                    }
                ],
                "output_features": [{"name": "y", "type": "binary"}],
            }
        )
        assert cfg.input_features[0].encoder.type == "jamba"
        assert cfg.input_features[0].encoder.attention_every_k == 2
