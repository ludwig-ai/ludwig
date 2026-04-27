"""Number-specific encoders: PLE (Piecewise Linear Encoding) and Periodic.

Based on:
- PLE: "On Embeddings for Numerical Features in Tabular Deep Learning" (Gorishniy et al., NeurIPS 2022)
- Periodic: Same paper, Section 3.2
"""

import logging
import math

import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, NUMBER
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict

logger = logging.getLogger(__name__)


@DeveloperAPI
@register_encoder("ple", [NUMBER])
class PLEEncoder(Encoder):
    """Piecewise Linear Encoding for numerical features.

    Computes quantile-based bin edges from training data, then for each input value produces a num_bins-dimensional
    vector where each element is a piecewise-linear interpolation within that bin. A learned linear projection maps this
    to the output embedding space.

    This encoding makes even simple MLPs competitive with attention-based tabular models.
    """

    def __init__(self, input_size=1, num_bins=64, output_size=256, encoder_config=None, **kwargs):
        super().__init__()
        self.config = encoder_config
        self.num_bins = num_bins
        self._output_size = output_size

        # Bin edges are set from training data metadata via update_config_with_metadata
        self.register_buffer("bin_edges", torch.linspace(0, 1, num_bins + 1))
        self.projection = nn.Linear(num_bins, output_size)

    def set_bin_edges(self, bin_edges: list[float]):
        """Set bin edges from training data statistics."""
        edges = torch.tensor(bin_edges, dtype=torch.float32)
        # Ensure edges are strictly increasing by adding small epsilon to duplicates
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8
        self.bin_edges.copy_(edges)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        # inputs: [batch, 1]
        x = inputs.squeeze(-1) if inputs.dim() > 1 else inputs  # [batch]

        edges = self.bin_edges  # [num_bins + 1]
        # Compute piecewise linear encoding: for each bin, a value in [0, 1]
        # representing how far through the bin the input value is
        left_edges = edges[:-1]  # [num_bins]
        right_edges = edges[1:]  # [num_bins]
        widths = right_edges - left_edges  # [num_bins]

        # [batch, num_bins]: linear interpolation within each bin, clamped to [0, 1]
        ple = torch.clamp((x.unsqueeze(-1) - left_edges) / (widths + 1e-8), 0.0, 1.0)

        # Learned projection to output size
        output = self.projection(ple)  # [batch, output_size]
        return {ENCODER_OUTPUT: output}

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.encoders.number_encoders import PLEEncoderConfig

        return PLEEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_size])


@DeveloperAPI
@register_encoder("periodic", [NUMBER])
class PeriodicEncoder(Encoder):
    """Periodic encoding for numerical features.

    Uses learned sinusoidal features: sin(2*pi*f*x + phi) where f and phi are learnable per-frequency parameters.
    A linear projection maps the periodic features to the output embedding space.
    """

    def __init__(self, input_size=1, num_frequencies=64, output_size=256, sigma=1.0, encoder_config=None, **kwargs):
        super().__init__()
        self.config = encoder_config
        self.num_frequencies = num_frequencies
        self._output_size = output_size

        # Learnable frequencies and phases
        self.frequencies = nn.Parameter(torch.randn(num_frequencies) * sigma)
        self.phases = nn.Parameter(torch.zeros(num_frequencies))
        self.projection = nn.Linear(num_frequencies, output_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        # inputs: [batch, 1] or [batch]
        x = inputs.squeeze(-1) if inputs.dim() > 1 else inputs  # [batch]

        # Compute periodic features: sin(2*pi*f*x + phi)
        periodic = torch.sin(2 * math.pi * x.unsqueeze(-1) * self.frequencies + self.phases)  # [batch, num_freq]

        output = self.projection(periodic)  # [batch, output_size]
        return {ENCODER_OUTPUT: output}

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.encoders.number_encoders import PeriodicEncoderConfig

        return PeriodicEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_size])


@DeveloperAPI
@register_encoder("bins", [NUMBER])
class BinsEncoder(Encoder):
    """Binning encoder: discretize numbers into equal-width or equal-frequency bins.

    Simpler alternative to PLE. Good for small/medium datasets where PLE may overfit. Each bin gets a learned embedding.
    """

    def __init__(self, input_size: int = 1, num_bins: int = 32, output_size: int = 256, encoder_config=None, **kwargs):
        super().__init__()
        self.config = encoder_config
        self.num_bins = num_bins
        self._input_size = input_size
        self._output_size = output_size
        # Bin edges will be set from training data metadata, default: uniform [0,1]
        self.register_buffer("bin_edges", torch.linspace(0, 1, num_bins + 1))
        self.bin_embeddings = nn.Embedding(num_bins, output_size)

    def set_bin_edges(self, bin_edges: list[float]):
        """Set bin edges from training data statistics."""
        edges = torch.tensor(bin_edges, dtype=torch.float32)
        # Ensure edges are strictly increasing by adding small epsilon to duplicates
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-8
        self.bin_edges.copy_(edges)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        # inputs: [batch, 1] or [batch]
        x = inputs.float()
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        # Digitize: find bin index for each input value using searchsorted, clamp to valid range
        bin_idx = torch.searchsorted(self.bin_edges[1:-1], x.squeeze(-1)).clamp(0, self.num_bins - 1)
        return {ENCODER_OUTPUT: self.bin_embeddings(bin_idx)}

    @staticmethod
    def get_schema_cls():
        from ludwig.schema.encoders.number_encoders import BinsEncoderConfig

        return BinsEncoderConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self._output_size])
