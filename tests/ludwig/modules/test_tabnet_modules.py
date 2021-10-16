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


@pytest.mark.parametrize('bn_virtual_bs', [None, 7])
@pytest.mark.parametrize(
    'external_shared_fc_layer', [True, False]
)
@pytest.mark.parametrize('apply_glu', [True, False])
@pytest.mark.parametrize('size', [4, 12])
@pytest.mark.parametrize('input_size', [2, 6])
def test_feature_block(
        input_size,
        size: int,
        apply_glu: bool,
        external_shared_fc_layer: bool,
        bn_virtual_bs: Optional[int]
) -> None:
    # setup synthetic tensor
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, input_size], dtype=torch.float32)

    if external_shared_fc_layer:
        shared_fc_layer = torch.nn.Linear(
            input_size,
            size * 2 if apply_glu else size,
            bias=False
        )
    else:
        shared_fc_layer = None

    feature_block = FeatureBlock(
        input_size,
        size,
        apply_glu=apply_glu,
        shared_fc_layer=shared_fc_layer,
        bn_virtual_bs=bn_virtual_bs
    )

    output_tensor = feature_block(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, size)


@pytest.mark.parametrize(
    'num_total_blocks, num_shared_blocks',
    [(4, 2), (6, 4), (3, 1)]
)
@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('size', [4, 12])
@pytest.mark.parametrize('input_size', [2, 6])
def test_feature_transformer(
        input_size: int,
        size: int,
        virtual_batch_size: Optional[int],
        num_total_blocks: int,
        num_shared_blocks: int
) -> None:
    # setup synthetic tensor
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, input_size], dtype=torch.float32)

    feature_transformer = FeatureTransformer(
        input_size,
        size,
        bn_virtual_bs=virtual_batch_size,
        num_total_blocks=num_total_blocks,
        num_shared_blocks=num_shared_blocks
    )

    output_tensor = feature_transformer(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, size)


@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('output_size', [10, 12])
@pytest.mark.parametrize('size', [4, 8])
@pytest.mark.parametrize('input_size', [2, 6])
def test_attentive_transformer(
        input_size: int,
        size: int,
        output_size: int,
        virtual_batch_size: Optional[int]
) -> None:
    # setup synthetic tensors
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, input_size], dtype=torch.float32)
    prior_scales = torch.ones([BATCH_SIZE, input_size])

    # setup required trasnformers for test
    feature_transformer = FeatureTransformer(
        input_size,
        size + output_size,
        bn_virtual_bs=virtual_batch_size
    )
    attentive_transformer = AttentiveTransformer(
        size,
        input_size,
        bn_virtual_bs=virtual_batch_size
    )

    # process synthetic tensor through transformers
    x = feature_transformer(input_tensor)
    output_tensor = attentive_transformer(x[:, output_size:], prior_scales)

    # check for correctness
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, input_size)


@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('size', [2, 4, 8])
@pytest.mark.parametrize('output_size', [2, 4, 12])
@pytest.mark.parametrize('input_size', [2])
def test_tabnet(
        input_size: int,
        output_size: int,
        size: int,
        virtual_batch_size: Optional[int]
) -> None:
    # setup synthetic tensor
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([BATCH_SIZE, input_size], dtype=torch.float32)

    feature_transformer = TabNet(
        input_size,
        size,
        output_size,
        num_steps=3,
        num_total_blocks=4,
        num_shared_blocks=2
    )

    output = feature_transformer(input_tensor)

    assert isinstance(output, tuple)
    assert output[0].shape == (BATCH_SIZE, output_size)
