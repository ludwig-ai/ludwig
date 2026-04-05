"""TabM-style BatchEnsemble for parameter-efficient ensembling.

Implements the BatchEnsemble technique from Wen et al. (ICLR 2020) adapted for
tabular deep learning as described in TabM (Gorishniy et al., ICLR 2025).

A single MLP efficiently imitates an ensemble by sharing a backbone and using
per-member affine transforms (non-shared scaling vectors). This provides
ensemble-level performance at single-model inference cost.

Usage:
    from ludwig.modules.batch_ensemble import BatchEnsembleLinear

    # Replace nn.Linear with BatchEnsembleLinear
    layer = BatchEnsembleLinear(in_features=128, out_features=64, num_members=4)
"""

import torch
import torch.nn as nn


class BatchEnsembleLinear(nn.Module):
    """Linear layer with BatchEnsemble for parameter-efficient ensembling.

    Shares the main weight matrix across ensemble members, but each member has
    its own rank-1 scaling factors (r_i and s_i):
        output_i = (s_i * (W @ (r_i * x))) + b

    This adds only O(in + out) parameters per member instead of O(in * out).
    """

    def __init__(self, in_features: int, out_features: int, num_members: int = 4, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_members = num_members

        # Shared backbone
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / in_features**0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        # Per-member scaling vectors (rank-1 perturbations)
        self.r = nn.Parameter(torch.ones(num_members, in_features))  # input scaling
        self.s = nn.Parameter(torch.ones(num_members, out_features))  # output scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with implicit ensemble.

        During training, randomly selects an ensemble member per sample.
        During eval, averages predictions across all members.

        Args:
            x: [batch, in_features]

        Returns:
            [batch, out_features]
        """
        if self.training:
            # Random member assignment per sample
            member_idx = torch.randint(0, self.num_members, (x.shape[0],), device=x.device)
            r = self.r[member_idx]  # [batch, in_features]
            s = self.s[member_idx]  # [batch, out_features]

            # Apply: s * (W @ (r * x)) + b
            x_scaled = x * r
            out = torch.nn.functional.linear(x_scaled, self.weight, self.bias)
            return out * s
        else:
            # Average over all members at eval time
            outputs = []
            for i in range(self.num_members):
                x_scaled = x * self.r[i]
                out = torch.nn.functional.linear(x_scaled, self.weight, self.bias)
                outputs.append(out * self.s[i])
            return torch.stack(outputs).mean(dim=0)
