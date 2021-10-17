import torch
import pytest
from typing import Optional

from ludwig.modules.fully_connected_modules import FCLayer, FCStack
from tests.integration_tests.utils import assert_model_parameters_updated

BATCH_SIZE = 2
INPUT_SIZE = 8
OUTPUT_SIZE = 4
RANDOM_SEED = 1919


@pytest.mark.parametrize('norm', [None, 'batch', 'layer'])
@pytest.mark.parametrize('use_bias', [True, False])
def test_fc_layer(
        use_bias: bool,
        norm: Optional[str]
) -> None:
    torch.manual_seed(RANDOM_SEED)
    batch = torch.randn([BATCH_SIZE, INPUT_SIZE], dtype=torch.float32)
    target_tensor = torch.ones([BATCH_SIZE, OUTPUT_SIZE], dtype=torch.float32)

    # setup layer to test
    fc_layer = FCLayer(
        INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        norm=norm,
        use_bias=use_bias
    )

    output_tensor = fc_layer(batch)

    # check for correct output type and shape
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, OUTPUT_SIZE)

    # check to confirm parameter updates
    assert_model_parameters_updated(fc_layer, batch, target_tensor)


@pytest.mark.parametrize('residual', [True, False])
@pytest.mark.parametrize('num_layers', [1, 3])
def test_fc_stack(
        num_layers: int,
        residual: bool
) -> None:
    torch.manual_seed(RANDOM_SEED)
    batch = torch.randn([BATCH_SIZE, INPUT_SIZE], dtype=torch.float32)
    target_tensor = torch.ones([BATCH_SIZE, OUTPUT_SIZE], dtype=torch.float32)

    # setup layer to test
    fc_stack = FCStack(
        INPUT_SIZE,
        default_fc_size=OUTPUT_SIZE,
        num_layers=num_layers
    )

    # confirm correct number of layers
    assert len(fc_stack.layers) == num_layers

    output_tensor = fc_stack(batch)

    # check for correct output type and shape
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (BATCH_SIZE, OUTPUT_SIZE)

    # check to confirm parameter updates
    assert_model_parameters_updated(fc_stack, batch, target_tensor)
