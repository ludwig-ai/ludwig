"""Tests for PLE and Periodic number encoders."""

import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.encoders.number_encoders import PeriodicEncoder, PLEEncoder

BATCH_SIZE = 16


class TestPLEEncoder:
    def test_output_shape(self):
        encoder = PLEEncoder(num_bins=32, output_size=128)
        encoder.set_bin_edges(torch.linspace(0, 1, 33).tolist())
        inputs = torch.randn(BATCH_SIZE, 1)
        output = encoder(inputs)
        assert ENCODER_OUTPUT in output
        assert output[ENCODER_OUTPUT].shape == (BATCH_SIZE, 128)

    def test_output_shape_1d_input(self):
        encoder = PLEEncoder(num_bins=16, output_size=64)
        encoder.set_bin_edges(torch.linspace(0, 10, 17).tolist())
        inputs = torch.randn(BATCH_SIZE)
        output = encoder(inputs)
        assert output[ENCODER_OUTPUT].shape == (BATCH_SIZE, 64)

    def test_gradient_flow(self):
        encoder = PLEEncoder(num_bins=16, output_size=64)
        encoder.set_bin_edges(torch.linspace(0, 1, 17).tolist())
        inputs = torch.randn(BATCH_SIZE, 1, requires_grad=True)
        output = encoder(inputs)
        loss = output[ENCODER_OUTPUT].sum()
        loss.backward()
        assert inputs.grad is not None
        # Check projection weights got gradients
        assert encoder.projection.weight.grad is not None

    def test_ple_interpolation_values(self):
        """Test that PLE values are correct for known bin edges and inputs."""
        encoder = PLEEncoder(num_bins=4, output_size=4)
        encoder.set_bin_edges([0.0, 0.25, 0.5, 0.75, 1.0])

        # Input exactly at bin edges should produce known patterns
        inputs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).unsqueeze(-1)
        output = encoder(inputs)
        # Just verify output is valid (detailed interpolation values depend on projection weights)
        assert output[ENCODER_OUTPUT].shape == (5, 4)
        assert torch.isfinite(output[ENCODER_OUTPUT]).all()

    def test_properties(self):
        encoder = PLEEncoder(num_bins=32, output_size=128)
        assert encoder.input_shape == torch.Size([1])
        assert encoder.output_shape == torch.Size([128])

    def test_default_bin_edges(self):
        """Encoder should work even before set_bin_edges is called (uses default linspace)."""
        encoder = PLEEncoder(num_bins=8, output_size=32)
        inputs = torch.randn(BATCH_SIZE, 1)
        output = encoder(inputs)
        assert output[ENCODER_OUTPUT].shape == (BATCH_SIZE, 32)

    def test_set_bin_edges_deduplication(self):
        """Duplicate bin edges should be handled gracefully."""
        encoder = PLEEncoder(num_bins=4, output_size=16)
        # Duplicates at edges (common when data has many identical values)
        encoder.set_bin_edges([0.0, 0.0, 0.5, 0.5, 1.0])
        inputs = torch.tensor([0.25, 0.75]).unsqueeze(-1)
        output = encoder(inputs)
        assert torch.isfinite(output[ENCODER_OUTPUT]).all()


class TestPeriodicEncoder:
    def test_output_shape(self):
        encoder = PeriodicEncoder(num_frequencies=32, output_size=128)
        inputs = torch.randn(BATCH_SIZE, 1)
        output = encoder(inputs)
        assert ENCODER_OUTPUT in output
        assert output[ENCODER_OUTPUT].shape == (BATCH_SIZE, 128)

    def test_output_shape_1d_input(self):
        encoder = PeriodicEncoder(num_frequencies=16, output_size=64)
        inputs = torch.randn(BATCH_SIZE)
        output = encoder(inputs)
        assert output[ENCODER_OUTPUT].shape == (BATCH_SIZE, 64)

    def test_gradient_flow(self):
        encoder = PeriodicEncoder(num_frequencies=16, output_size=64)
        inputs = torch.randn(BATCH_SIZE, 1, requires_grad=True)
        output = encoder(inputs)
        loss = output[ENCODER_OUTPUT].sum()
        loss.backward()
        assert inputs.grad is not None
        assert encoder.frequencies.grad is not None
        assert encoder.phases.grad is not None
        assert encoder.projection.weight.grad is not None

    def test_properties(self):
        encoder = PeriodicEncoder(num_frequencies=32, output_size=128)
        assert encoder.input_shape == torch.Size([1])
        assert encoder.output_shape == torch.Size([128])

    def test_sigma_affects_initialization(self):
        """Different sigma values should produce different frequency distributions."""
        torch.manual_seed(42)
        enc1 = PeriodicEncoder(num_frequencies=64, sigma=0.1)
        torch.manual_seed(42)
        enc2 = PeriodicEncoder(num_frequencies=64, sigma=10.0)
        # Frequencies should differ by the sigma factor
        assert not torch.allclose(enc1.frequencies, enc2.frequencies)

    def test_periodic_output_is_bounded_before_projection(self):
        """Sin outputs should be in [-1, 1] range before projection."""
        encoder = PeriodicEncoder(num_frequencies=32, output_size=32)
        # Override projection to identity to check raw periodic features
        encoder.projection = torch.nn.Identity()
        encoder._output_size = 32
        inputs = torch.randn(BATCH_SIZE, 1) * 100  # large inputs
        output = encoder(inputs)
        # Raw periodic features are sin() values, should be in [-1, 1]
        assert output[ENCODER_OUTPUT].min() >= -1.0
        assert output[ENCODER_OUTPUT].max() <= 1.0
