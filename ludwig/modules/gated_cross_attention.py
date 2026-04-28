"""Gated cross-attention module for vision-language fusion (Flamingo-style).

Based on the Flamingo paper (Alayrac et al., NeurIPS 2022).  A gated cross-attention block
conditions a text (or other query) representation on visual (or other key/value) tokens.
The gate is initialised to zero so that the block is an identity at the start of training:
the pretrained language model's behaviour is preserved until the gate learns to attend to
the visual features.  Stable fine-tuning of VLMs on new modalities typically requires this
zero-init gating, otherwise the random cross-attention outputs wreck the LM head's calibration
on the first step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedCrossAttention(nn.Module):
    """Flamingo-style gated cross-attention block.

    Args:
        d_model: hidden size of the query stream (usually the LM hidden size).
        num_heads: number of attention heads.
        kv_dim: hidden size of the key/value stream (usually the vision encoder output).
            Defaults to ``d_model``.
        ffn_size: feed-forward network width.  Defaults to ``4 * d_model``.
        dropout: dropout probability applied after attention and FFN.
        tanh_gate: if True, wrap the gates in ``tanh`` so they're bounded in ``(-1, 1)``.

    The module owns two learnable scalar gates, ``attn_gate`` and ``ffn_gate``, both
    initialised to zero.  A forward pass over ``(x, kv)`` computes:

    .. code:: text

        x = x + tanh(attn_gate) * CrossAttn(x, kv)
        x = x + tanh(ffn_gate)  * FFN(x)

    so at step 0 the module is exactly the identity.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        kv_dim: int | None = None,
        ffn_size: int | None = None,
        dropout: float = 0.0,
        tanh_gate: bool = True,
    ) -> None:
        super().__init__()
        kv_dim = kv_dim if kv_dim is not None else d_model
        ffn_size = ffn_size if ffn_size is not None else 4 * d_model

        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.kv_proj = nn.Linear(kv_dim, d_model) if kv_dim != d_model else nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, d_model),
        )

        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ffn_gate = nn.Parameter(torch.zeros(1))
        self.tanh_gate = tanh_gate

    def _gate(self, g: torch.Tensor) -> torch.Tensor:
        return torch.tanh(g) if self.tanh_gate else g

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply gated cross-attention.

        Args:
            x: query stream, shape ``(batch, seq_len_q, d_model)``.
            kv: key/value stream, shape ``(batch, seq_len_kv, kv_dim)``.
            key_padding_mask: optional bool mask with ``True`` at padded positions in ``kv``,
                shape ``(batch, seq_len_kv)``.

        Returns:
            Updated query stream, same shape as ``x``.
        """
        kv_proj = self.kv_proj(self.kv_norm(kv))
        q_norm = self.q_norm(x)
        attn_out, _ = self.cross_attn(q_norm, kv_proj, kv_proj, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self._gate(self.attn_gate) * attn_out

        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self._gate(self.ffn_gate) * ffn_out
        return x
