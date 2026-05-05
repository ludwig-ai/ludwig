import torch
import torch.nn as nn

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ENCODER_OUTPUT, TIMESERIES
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.encoders.types import EncoderOutputDict
from ludwig.schema.encoders.timeseries_encoders import NBEATSConfig, PatchTSTConfig


@DeveloperAPI
@register_encoder("patchtst", [TIMESERIES])
class PatchTSTEncoder(Encoder):
    """PatchTST encoder.

    Splits the input time series into fixed-length patches and encodes them with a Transformer.
    Channel-independent: each feature dimension is processed independently.
    Reference: "A Time Series is Worth 64 Words" (Nie et al., 2023). https://arxiv.org/abs/2211.14730
    """

    def __init__(
        self,
        max_sequence_length: int,
        patch_size: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        output_size: int = 256,
        reduce_output: str = "mean",
        encoder_config: PatchTSTConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.reduce_output = reduce_output
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length

        # Patch projection
        self.patch_proj = nn.Linear(patch_size, d_model)

        # Positional encoding (learnable)
        num_patches = max(1, (max_sequence_length - patch_size) // patch_stride + 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        # inputs: [batch, seq_len] or [batch, seq_len, channels]
        if inputs.dim() == 2:
            x = inputs.unsqueeze(-1)  # [B, T, 1]
        else:
            x = inputs  # [B, T, C]

        B, T, C = x.shape

        # Process each channel independently by merging into batch dimension
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = x.reshape(B * C, T)  # [B*C, T]

        # Extract patches via unfold: [B*C, num_patches, patch_size]
        x_patched = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)

        # Project patches to d_model
        x_patched = self.patch_proj(x_patched)  # [B*C, num_patches, d_model]

        # Add positional embedding (clip to actual num_patches)
        num_patches = x_patched.shape[1]
        x_patched = x_patched + self.pos_embed[:, :num_patches, :]

        # Transformer encoder
        x_enc = self.transformer(x_patched)  # [B*C, num_patches, d_model]
        x_enc = self.norm(x_enc)

        # Reduce across patches
        if self.reduce_output == "mean":
            x_rep = x_enc.mean(dim=1)  # [B*C, d_model]
        elif self.reduce_output == "last":
            x_rep = x_enc[:, -1, :]
        else:  # first
            x_rep = x_enc[:, 0, :]

        # Average over channels to get per-sample representation
        x_rep = x_rep.reshape(B, C, -1).mean(dim=1)  # [B, d_model]
        out = self.head(x_rep)  # [B, output_size]

        return {ENCODER_OUTPUT: out}

    @staticmethod
    def get_schema_cls():
        return PatchTSTConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_size])


class NBEATSBlock(nn.Module):
    """A single N-BEATS block with backcast and forecast branches."""

    def __init__(self, input_size: int, theta_size: int, layer_size: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_size = input_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_size, layer_size), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = layer_size
        self.fc = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(layer_size, input_size)
        self.forecast_head = nn.Linear(layer_size, theta_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        backcast = self.backcast_head(h)
        forecast = self.forecast_head(h)
        return backcast, forecast


@DeveloperAPI
@register_encoder("nbeats", [TIMESERIES])
class NBEATSEncoder(Encoder):
    """N-BEATS encoder.

    Pure MLP architecture with doubly residual stacking. Each block produces a backcast (reconstruction)
    and a forecast contribution. Residuals are passed between blocks. The encoder returns a fixed-size
    learned representation from the aggregated forecast vectors.
    Reference: "N-BEATS" (Oreshkin et al., 2020). https://arxiv.org/abs/1905.10437
    """

    def __init__(
        self,
        max_sequence_length: int,
        num_stacks: int = 2,
        num_blocks: int = 3,
        num_layers: int = 4,
        layer_size: int = 256,
        output_size: int = 256,
        dropout: float = 0.0,
        encoder_config: NBEATSConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length
        input_size = max_sequence_length
        theta_size = output_size

        self.stacks = nn.ModuleList()
        for _ in range(num_stacks):
            stack = nn.ModuleList()
            for _ in range(num_blocks):
                block = NBEATSBlock(input_size, theta_size, layer_size, num_layers, dropout)
                stack.append(block)
            self.stacks.append(stack)

        self.output_proj = nn.Linear(theta_size, output_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> EncoderOutputDict:
        # inputs: [batch, seq_len] or [batch, seq_len, C]
        if inputs.dim() == 3:
            B, T, C = inputs.shape
            x = inputs.permute(0, 2, 1).reshape(B * C, T)
        else:
            B, T = inputs.shape
            C = 1
            x = inputs  # [B, T]

        residual = x
        forecast = torch.zeros(x.shape[0], self.output_size, device=x.device, dtype=x.dtype)

        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast

        # Average over channels if multi-channel input
        if C > 1:
            forecast = forecast.reshape(B, C, -1).mean(dim=1)  # [B, output_size]

        out = self.output_proj(forecast)  # [B, output_size]
        return {ENCODER_OUTPUT: out}

    @staticmethod
    def get_schema_cls():
        return NBEATSConfig

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.max_sequence_length])

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.output_size])
