import pytest
from typing import Optional

import torch

from ludwig.modules.normalization_modules import GhostBatchNormalization

BATCH_SIZE = 16
FC_SIZE = 8


@pytest.mark.parametrize(
    'virtual_batch_size',
    [
        None,
        BATCH_SIZE // 2,
        BATCH_SIZE - 2
    ]
)
@pytest.mark.parametrize('mode',
                         [True, False])  # training (True) or eval(False)
def test_ghostbatchnormalization(
        mode: bool,
        virtual_batch_size: Optional[int]
) -> None:
    # setup up GhostBatchNormalization layer
    ghost_batch_norm = GhostBatchNormalization(
        FC_SIZE,
        virtual_batch_size=virtual_batch_size
    )

    # set training or eval mode
    ghost_batch_norm.train(mode=mode)

    # setup inputs to test
    inputs = torch.randn([BATCH_SIZE, FC_SIZE], dtype=torch.float32)

    # run tensor through
    norm_tensor = ghost_batch_norm(inputs)

    # check for correctness
    assert isinstance(norm_tensor, torch.Tensor)

    print(ghost_batch_norm.moving_mean, ghost_batch_norm.moving_variance)
