from typing import Optional, List
import pytest

import torch

from ludwig.modules.tabnet_modules import Sparsemax
from ludwig.modules.tabnet_modules import TabNet
from ludwig.modules.tabnet_modules import FeatureTransformer, FeatureBlock
from ludwig.modules.tabnet_modules import AttentiveTransformer

RANDOM_SEED = 67
BATCH_SIZE = 16
HIDDEN_SIZE = 8


def test_sparsemax():
    input_tensor = torch.tensor(
        [[-1.0, 0.0, 1.0], [5.01, 4.0, -2.0]],
        dtype=torch.float32
    )

    sparsemax = Sparsemax()

    output_tensor = sparsemax(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.equal(torch.tensor([[0, 0, 1], [1, 0, 0]],
                                            dtype=torch.float32))


@pytest.mark.parametrize(
    'external_shared_fc_layer', [True, False]
)
@pytest.mark.parametrize('apply_glu', [True, False])
@pytest.mark.parametrize('output_size', [4, 12])
def test_feature_block(
        output_size: int,
        apply_glu: bool,
        external_shared_fc_layer: bool
) -> None:
    # setup synthetic tensor
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, HIDDEN_SIZE], dtype=torch.float32)

    if external_shared_fc_layer:
        shared_fc_layer = torch.nn.Linear(
            HIDDEN_SIZE,
            output_size * 2 if apply_glu else output_size,
            bias=False
        )
    else:
        shared_fc_layer = None

    feature_block = FeatureBlock(
        HIDDEN_SIZE,
        size=output_size,
        apply_glu=apply_glu,
        shared_fc_layer=shared_fc_layer
    )

    output_tensor = feature_block(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, output_size)


@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('output_size', [4, 12])
def test_feature_transfomer(
        output_size: int,
        virtual_batch_size: Optional[int]
) -> None:
    # setup synthetic tensor
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, HIDDEN_SIZE], dtype=torch.float32)

    feature_transformer = FeatureTransformer(
        HIDDEN_SIZE,
        size=output_size,
        bn_virtual_bs=virtual_batch_size
    )

    output_tensor = feature_transformer(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, output_size)


@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('size', [2, 4, 8])
@pytest.mark.parametrize('output_dim', [2, 4, 12])
@pytest.mark.parametrize('input_dim', [2])
def test_tabnet(
        input_dim: int,
        output_dim: int,
        size: int,
        virtual_batch_size: Optional[int]
) -> None:
    # setup synthetic tensor
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, input_dim], dtype=torch.float32)

    feature_transformer = TabNet(
        input_dim,
        size,
        output_dim,
        num_steps=3,
        num_total_blocks=4,
        num_shared_blocks=2
    )

    output = feature_transformer(input_tensor)

    assert isinstance(output, tuple)
    assert output[0].shape == (BATCH_SIZE, output_dim)
