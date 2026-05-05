"""Tests for timeseries-specific encoders (PatchTST, N-BEATS) and metrics (MASE, sMAPE)."""

import torch


class TestPatchTSTEncoder:
    def test_forward_1d(self):
        from ludwig.encoders.timeseries_encoders import PatchTSTEncoder

        enc = PatchTSTEncoder(
            max_sequence_length=64,
            patch_size=8,
            patch_stride=4,
            d_model=32,
            num_heads=4,
            num_layers=1,
            ffn_dim=64,
            output_size=16,
        )
        x = torch.randn(4, 64)
        out = enc(x)
        assert out["encoder_output"].shape == (4, 16)

    def test_forward_multichannel(self):
        from ludwig.encoders.timeseries_encoders import PatchTSTEncoder

        enc = PatchTSTEncoder(
            max_sequence_length=64,
            patch_size=8,
            patch_stride=4,
            d_model=32,
            num_heads=4,
            num_layers=1,
            ffn_dim=64,
            output_size=16,
        )
        x = torch.randn(4, 64, 3)
        out = enc(x)
        assert out["encoder_output"].shape == (4, 16)

    def test_schema_roundtrip(self):
        from ludwig.schema.encoders.timeseries_encoders import PatchTSTConfig

        cfg = PatchTSTConfig.model_validate({"type": "patchtst", "patch_size": 32, "num_layers": 2})
        assert cfg.patch_size == 32
        assert cfg.num_layers == 2


class TestNBEATSEncoder:
    def test_forward_1d(self):
        from ludwig.encoders.timeseries_encoders import NBEATSEncoder

        enc = NBEATSEncoder(
            max_sequence_length=64,
            num_stacks=2,
            num_blocks=2,
            num_layers=2,
            layer_size=64,
            output_size=32,
        )
        x = torch.randn(4, 64)
        out = enc(x)
        assert out["encoder_output"].shape == (4, 32)

    def test_forward_multichannel(self):
        from ludwig.encoders.timeseries_encoders import NBEATSEncoder

        enc = NBEATSEncoder(
            max_sequence_length=64,
            num_stacks=2,
            num_blocks=2,
            num_layers=2,
            layer_size=64,
            output_size=32,
        )
        x = torch.randn(4, 64, 3)
        out = enc(x)
        assert out["encoder_output"].shape == (4, 32)

    def test_schema_roundtrip(self):
        from ludwig.schema.encoders.timeseries_encoders import NBEATSConfig

        cfg = NBEATSConfig.model_validate({"type": "nbeats", "num_stacks": 3})
        assert cfg.num_stacks == 3


class TestTimeseriesMetrics:
    def test_mase_metric(self):
        from ludwig.constants import MEAN_ABSOLUTE_SCALED_ERROR
        from ludwig.modules.metric_registry import get_metric_registry

        registry = get_metric_registry()
        assert MEAN_ABSOLUTE_SCALED_ERROR in registry

    def test_smape_metric(self):
        from ludwig.constants import SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR
        from ludwig.modules.metric_registry import get_metric_registry

        registry = get_metric_registry()
        assert SYMMETRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR in registry

    def test_mase_forward(self):
        from ludwig.modules.metric_modules import MASEMetric

        metric = MASEMetric()
        preds = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[1.1, 2.1, 3.1, 4.1]])
        val = metric.get_current_value(preds, target)
        assert val.item() > 0

    def test_smape_forward(self):
        from ludwig.modules.metric_modules import SMAPEMetric

        metric = SMAPEMetric()
        preds = torch.tensor([[1.0, 2.0, 3.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        val = metric.get_current_value(preds, target)
        assert val.item() < 1e-6  # perfect forecast -> 0
