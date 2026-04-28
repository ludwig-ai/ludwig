"""Mamba-2 and Jamba-style hybrid sequence encoders (Phase 6.6.2).

``Mamba2Encoder``
    Successor to Mamba-1.  The key architectural differences:
      * multi-head parametrisation with a per-head scalar decay (SSD simplification —
        Dao & Gu, "State Space Duality", 2024);
      * wider inner expansion with group-wise output gating;
      * slightly cheaper per-step compute because the state update is scalar per head
        instead of the full rank-1 Mamba-1 update.
    Like ``MambaEncoder`` in this repo, this is a pure-PyTorch approximation that
    keeps the architectural pattern without needing the CUDA ``mamba_ssm`` kernel.

``JambaEncoder``
    Hybrid encoder that interleaves Mamba-style SSM blocks with attention blocks.
    Following the Jamba architecture (Lieber et al., 2024), every
    ``attention_every_k``-th layer is a TransformerEncoderLayer (attention + MLP) and
    the remaining layers are Mamba-2 SSM blocks.  Default is ``attention_every_k=4``
    to match the 1:3 attention:SSM ratio reported in the Jamba paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import AUDIO, SEQUENCE, TEXT, TIMESERIES
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.schema.encoders.mamba_hybrid import JambaEncoderConfig, Mamba2EncoderConfig
from ludwig.utils.torch_utils import initializer_registry


class _Mamba2Block(nn.Module):
    """Single Mamba-2 SSM block.

    Input: ``(batch, seq_len, d_model)``.
    Output: ``(batch, seq_len, d_model)``.

    Multi-head SSD approximation: split ``d_model`` into ``num_heads`` channels, give
    each head a learnable scalar decay, mix along the sequence with a depthwise 1D
    convolution, then gate + project back.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        d_inner = d_model * expand_factor
        head_dim = d_inner // num_heads

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Per-head scalar decay, parametrised in log space for positivity.
        self.log_alpha = nn.Parameter(torch.empty(num_heads).uniform_(math.log(0.1), math.log(0.99)))
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_path, gate = xz.chunk(2, dim=-1)  # each (batch, seq_len, d_inner)

        # Local depthwise convolution (trim trailing padding to preserve causal length).
        batch, seq_len, _ = x_path.shape
        xt = x_path.transpose(1, 2)  # (batch, d_inner, seq_len)
        xt = self.conv1d(xt)[:, :, :seq_len]
        x_path = xt.transpose(1, 2)  # (batch, seq_len, d_inner)

        # Per-head scalar decay mixing (SSD-style): y_t = alpha_h * y_{t-1} + x_t.
        x_path = x_path.view(batch, seq_len, self.num_heads, self.head_dim)
        # (1, num_heads, 1) — hoisted outside the loop; shape unchanged per step.
        alpha = torch.sigmoid(self.log_alpha).view(self.num_heads, 1)
        outputs = torch.empty_like(x_path)
        y = torch.zeros(batch, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        for t in range(seq_len):
            y = alpha * y + x_path[:, t]
            outputs[:, t] = y
        x_path = outputs.view(batch, seq_len, self.d_inner)

        # Gated SiLU output — shared across heads.
        x_path = F.silu(x_path) * gate

        out = self.out_proj(x_path)
        out = self.dropout(out)
        return out + residual


class _Mamba2Stack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_conv: int,
        expand_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            _Mamba2Block(d_model, num_heads=num_heads, d_conv=d_conv, expand_factor=expand_factor, dropout=dropout)
            for _ in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@DeveloperAPI
@register_encoder("mamba2", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class Mamba2Encoder(Encoder):
    """Mamba-2 SSM encoder (Dao & Gu, 2024).

    Multi-head selective SSM with per-head scalar decay.  Linear-time in sequence length like Mamba-1 but with a more
    expressive hidden state and slightly cheaper per-step compute thanks to the state-space duality (SSD)
    simplification.
    """

    def __init__(
        self,
        max_sequence_length: int = 256,
        should_embed: bool = True,
        vocab=None,
        embedding_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        num_heads: int = 8,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        output_size: int = 256,
        reduce_output: str = "mean",
        encoder_config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = encoder_config
        self.should_embed = should_embed
        self.reduce_output = reduce_output
        self.max_sequence_length = max_sequence_length

        if should_embed:
            vocab_size = len(vocab) if vocab is not None else 1
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            embed_dim = embedding_size
            if embed_dim != d_model:
                self.embed_proj = nn.Linear(embed_dim, d_model)
            else:
                self.embed_proj = nn.Identity()
        else:
            self.embedding = None
            self.embed_proj = nn.Identity()

        self.stack = _Mamba2Stack(d_model, n_layers, num_heads, d_conv, expand_factor, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)

        self._output_size = output_size

    @property
    def input_dtype(self):
        return torch.int32 if self.should_embed else torch.float32

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output in (None, "none"):
            return torch.Size([self.max_sequence_length, self._output_size])
        return torch.Size([self._output_size])

    def forward(self, inputs: torch.Tensor, mask=None) -> dict[str, torch.Tensor]:
        if self.should_embed:
            x = self.embedding(inputs.long())
        else:
            x = inputs
        x = self.embed_proj(x)
        x = self.stack(x)
        x = self.final_norm(x)
        if self.reduce_output in (None, "none"):
            pass
        elif self.reduce_output == "mean":
            x = x.mean(dim=1)
        elif self.reduce_output == "sum":
            x = x.sum(dim=1)
        elif self.reduce_output == "max":
            x = x.max(dim=1).values
        elif self.reduce_output == "last":
            x = x[:, -1]
        else:
            raise ValueError(
                f"Unknown reduce_output={self.reduce_output!r}. "
                "Valid options: None, 'none', 'mean', 'sum', 'max', 'last'."
            )
        x = self.output_proj(x)
        return {"encoder_output": x}

    @staticmethod
    def get_schema_cls():
        return Mamba2EncoderConfig


@DeveloperAPI
@register_encoder("jamba", [AUDIO, SEQUENCE, TEXT, TIMESERIES])
class JambaEncoder(Encoder):
    """Jamba-style hybrid encoder (Lieber et al., 2024).

    Interleaves Mamba-2 SSM blocks with TransformerEncoderLayer attention blocks.
    Every ``attention_every_k``-th layer is an attention block; the rest are SSM
    blocks.  With ``attention_every_k=4`` (default) and ``n_layers=8`` the pattern is
    ``[S S S A S S S A]`` — 1:3 attention:SSM as reported in the Jamba paper.

    The attention block is a standard pre-norm Transformer encoder layer so the
    hybrid drops straight in as a Ludwig sequence encoder; the SSM block is the same
    ``_Mamba2Block`` used above.
    """

    def __init__(
        self,
        max_sequence_length: int = 256,
        should_embed: bool = True,
        vocab=None,
        embedding_size: int = 256,
        d_model: int = 256,
        n_layers: int = 8,
        attention_every_k: int = 4,
        num_heads: int = 8,
        ffn_size: int = 1024,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        output_size: int = 256,
        reduce_output: str = "mean",
        encoder_config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = encoder_config
        self.should_embed = should_embed
        self.reduce_output = reduce_output
        self.max_sequence_length = max_sequence_length

        if should_embed:
            vocab_size = len(vocab) if vocab is not None else 1
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            if embedding_size != d_model:
                self.embed_proj = nn.Linear(embedding_size, d_model)
            else:
                self.embed_proj = nn.Identity()
        else:
            self.embedding = None
            self.embed_proj = nn.Identity()

        layers: list[nn.Module] = []
        for i in range(n_layers):
            if (i + 1) % attention_every_k == 0:
                layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=ffn_size,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                    )
                )
            else:
                layers.append(
                    _Mamba2Block(
                        d_model, num_heads=num_heads, d_conv=d_conv, expand_factor=expand_factor, dropout=dropout
                    )
                )
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_size)

        self._output_size = output_size

    @property
    def input_dtype(self):
        return torch.int32 if self.should_embed else torch.float32

    @property
    def output_shape(self) -> torch.Size:
        if self.reduce_output in (None, "none"):
            return torch.Size([self.max_sequence_length, self._output_size])
        return torch.Size([self._output_size])

    def forward(self, inputs: torch.Tensor, mask=None) -> dict[str, torch.Tensor]:
        if self.should_embed:
            x = self.embedding(inputs.long())
        else:
            x = inputs
        x = self.embed_proj(x)
        for layer in self.layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                x = layer(x, src_key_padding_mask=mask)
            else:
                x = layer(x)
        x = self.final_norm(x)
        if self.reduce_output in (None, "none"):
            pass
        elif self.reduce_output == "mean":
            x = x.mean(dim=1)
        elif self.reduce_output == "sum":
            x = x.sum(dim=1)
        elif self.reduce_output == "max":
            x = x.max(dim=1).values
        elif self.reduce_output == "last":
            x = x[:, -1]
        else:
            raise ValueError(
                f"Unknown reduce_output={self.reduce_output!r}. "
                "Valid options: None, 'none', 'mean', 'sum', 'max', 'last'."
            )
        x = self.output_proj(x)
        return {"encoder_output": x}

    @staticmethod
    def get_schema_cls():
        return JambaEncoderConfig


# Silence unused-import warning — initializer_registry is imported so subclasses can use
# Ludwig's standard weight init if ever extended.
_ = initializer_registry
