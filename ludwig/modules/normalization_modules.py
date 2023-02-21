import logging
from typing import Optional

import numpy as np
import torch
from torch.nn import BatchNorm1d, BatchNorm2d, LayerNorm, Module

from ludwig.utils.torch_utils import LudwigModule

logger = logging.getLogger(__name__)


# implementation adapted from https://github.com/dreamquark-ai/tabnet
class GhostBatchNormalization(LudwigModule):
    def __init__(
        self, num_features: int, momentum: float = 0.05, epsilon: float = 1e-3, virtual_batch_size: Optional[int] = 128
    ):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.bn = torch.nn.BatchNorm1d(num_features, momentum=momentum, eps=epsilon)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        if self.training and self.virtual_batch_size:
            splits = inputs.chunk(int(np.ceil(batch_size / self.virtual_batch_size)), 0)
            if batch_size == 1:
                logger.warning(
                    "Batch size is 1, but batch normalization requires batch size >= 2. Skipping batch normalization."
                    "Make sure to set `batch_size` to a value greater than 1."
                )
                # Skip batch normalization if the batch size is 1.
                return torch.cat(splits, 0)
            if batch_size % self.virtual_batch_size == 1:
                # Skip batch normalization for the last chunk if it is size 1.
                logger.warning(
                    f"Virtual batch size `{self.virtual_batch_size}` is not a factor of the batch size `{batch_size}`, "
                    "resulting in a chunk of size 1. Skipping batch normalization for the last chunk of size 1."
                )
            splits_with_bn = [self.bn(x) if x.shape[0] > 1 else x for x in splits]
            return torch.cat(splits_with_bn, 0)

        if batch_size != 1 or not self.training:
            return self.bn(inputs)
        return inputs

    @property
    def moving_mean(self) -> torch.Tensor:
        return self.bn.running_mean

    @property
    def moving_variance(self) -> torch.Tensor:
        return self.bn.running_var

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.num_features])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.num_features])


norm_registry = {
    "batch_1d": BatchNorm1d,
    "batch_2d": BatchNorm2d,
    "layer": LayerNorm,
    "ghost": GhostBatchNormalization,
}


def create_norm_layer(norm: str, input_rank: int, num_features: int, **norm_params) -> Module:
    if norm == "batch":
        # We use a different batch norm depending on the input_rank.
        # TODO(travis): consider moving this behind a general BatchNorm interface to avoid this kludge.
        if input_rank not in {2, 3}:
            ValueError(f"`input_rank` parameter expected to be either 2 or 3, but found {input_rank}.")
        norm = f"{norm}_{input_rank-1}d"

    norm_cls = norm_registry.get(norm)
    if norm_cls is None:
        raise ValueError(
            f"Unsupported value for `norm` param: {norm}. Supported values are: {list(norm_registry.keys())}"
        )

    return norm_cls(num_features, **norm_params)
