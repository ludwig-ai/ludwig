"""Contrastive pre-alignment between encoders and combiner (Phase 6.4.2).

Before end-to-end training, a brief contrastive pre-training phase can align the output
spaces of different per-feature encoders so the combiner sees already-comparable
representations.  This mirrors the contrastive stage used in multimodal works like
CLIP (Radford et al., ICML 2021) and HyperFusion (Mansour & Shkolnisky, 2024), but
adapted to Ludwig's multi-encoder ECD architecture where every input feature has its
own encoder.

The module here is small and model-agnostic: given a dict of per-feature embeddings
``{feature_name: (batch, dim)}`` it projects each into a shared aligned space and
computes a symmetric multi-view InfoNCE loss across every pair of features in the
batch.  The aligned space is learnable (one linear projection per feature) and is
discarded after pre-training — only the updated encoder weights carry forward.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAlignmentLoss(nn.Module):
    """Symmetric multi-view InfoNCE loss over per-feature encoder outputs.

    Args:
        feature_dims: mapping ``{feature_name: encoder_output_dim}`` — each feature's
            unreduced embedding width.  A linear projection into the shared alignment
            space is created per feature.
        projection_dim: width of the shared alignment space.
        temperature: InfoNCE temperature.  Lower values sharpen the softmax.  CLIP
            uses a learnable log-temperature initialised to ``log(1/0.07)``; we follow
            that convention and expose the initial value as a constructor arg.
        learnable_temperature: when True, the (log) temperature is a trainable parameter;
            otherwise it's fixed.

    The forward returns a scalar loss summed over all ordered pairs
    ``(feature_i, feature_j)`` with ``i != j``.  Each pair contributes a symmetric
    InfoNCE term (row-wise + column-wise cross entropy), so permuting features
    leaves the loss value unchanged.
    """

    def __init__(
        self,
        feature_dims: dict[str, int],
        projection_dim: int = 128,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        if len(feature_dims) < 2:
            raise ValueError(f"ContrastiveAlignmentLoss requires at least 2 input features, got {len(feature_dims)}")
        self.feature_names = list(feature_dims.keys())
        self.projections = nn.ModuleDict({name: nn.Linear(dim, projection_dim) for name, dim in feature_dims.items()})
        init_log_t = math.log(1.0 / temperature)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(init_log_t, dtype=torch.float32))
        else:
            self.register_buffer("log_temperature", torch.tensor(init_log_t, dtype=torch.float32))

    def _project(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        projected = {}
        for name in self.feature_names:
            if name not in embeddings:
                raise KeyError(
                    f"ContrastiveAlignmentLoss expected feature {name!r} in batch; got {list(embeddings.keys())}"
                )
            z = self.projections[name](embeddings[name])
            projected[name] = F.normalize(z, dim=-1)
        return projected

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the pairwise symmetric InfoNCE loss.

        Args:
            embeddings: ``{feature_name: (batch, dim)}``.  Every key in
                ``self.feature_names`` must be present; batch size must match across
                features.

        Returns:
            Scalar loss tensor.
        """
        projected = self._project(embeddings)
        # ``log_temperature`` stores log(1/T) following the CLIP convention, so
        # exp(log_temperature) is the *inverse* temperature / logit scale.  Multiplying
        # the cosine similarity by this scale therefore sharpens the softmax when T is
        # small (and log_temperature is large positive).
        logit_scale = torch.exp(self.log_temperature).clamp(max=100.0)

        loss = projected[self.feature_names[0]].new_zeros(())
        num_pairs = 0
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                z_i = projected[self.feature_names[i]]
                z_j = projected[self.feature_names[j]]
                batch = z_i.shape[0]
                logits = (z_i @ z_j.T) * logit_scale
                targets = torch.arange(batch, device=logits.device)
                # Symmetric InfoNCE: each example should identify its positive in both directions.
                loss = loss + 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))
                num_pairs += 1

        return loss / max(num_pairs, 1)
