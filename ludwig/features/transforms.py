"""Feature transform protocol for Ludwig's lazy preprocessing pipeline.

All feature-specific data transforms should extend ``FeatureTransform`` so they
are composable, testable, and can be moved to GPU when that becomes beneficial.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class FeatureTransform(nn.Module):
    """Base class for all per-sample feature transforms.

    Subclasses implement ``forward(x: Tensor) -> Tensor``.  They are
    ``nn.Module`` subclasses so they can be:

    * Composed with ``nn.Sequential``
    * Saved / loaded with ``torch.save`` / ``torch.load``
    * Moved to GPU with ``.to(device)``
    * JIT-compiled with ``torch.jit.script`` (where supported)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityTransform(FeatureTransform):
    """Pass-through — useful as a placeholder or in tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NormalizationTransform(FeatureTransform):
    """Subtract mean, divide by std.  Both are registered as buffers so the
    transform travels correctly through ``torch.save`` / ``.to(device)``."""

    def __init__(self, mean: float, std: float) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = self.std.clamp(min=1e-8)
        return (x - self.mean) / std
