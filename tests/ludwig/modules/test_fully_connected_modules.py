from typing import List, Optional

import numpy as np
import pytest
import torch

from ludwig.modules.fully_connected_modules import FCLayer, FCStack
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device

BATCH_SIZE = 2
DEVICE = get_torch_device()
RANDOM_SEED = 1919


@pytest.mark.parametrize("input_size", [2, 3])
@pytest.mark.parametrize("output_size", [3, 4])
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dropout", [0.0, 0.6])
def test_fc_layer(
    input_size: int,
    output_size: int,
    activation: str,
    dropout: float,
):
    set_random_seed(RANDOM_SEED)  # make repeatable
    fc_layer = FCLayer(input_size=input_size, output_size=output_size, activation=activation, dropout=dropout).to(
        DEVICE
    )
    input_tensor = torch.randn(BATCH_SIZE, input_size, device=DEVICE)
    output_tensor = fc_layer(input_tensor)
    assert output_tensor.shape[1:] == fc_layer.output_shape


@pytest.mark.parametrize(
    "first_layer_input_size,layers,num_layers",
    [
        (2, None, 3),
        (2, [{"output_size": 4}, {"output_size": 8}], None),
        (2, [{"input_size": 2, "output_size": 4}, {"output_size": 8}], None),
    ],
)
def test_fc_stack(
    first_layer_input_size: Optional[int],
    layers: Optional[List],
    num_layers: Optional[int],
):
    set_random_seed(RANDOM_SEED)
    fc_stack = FCStack(first_layer_input_size=first_layer_input_size, layers=layers, num_layers=num_layers).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape[1:] == fc_stack.output_shape


def test_fc_stack_input_size_mismatch_fails():
    first_layer_input_size = 10
    layers = [{"input_size": 2, "output_size": 4}, {"output_size": 8}]

    fc_stack = FCStack(
        first_layer_input_size=first_layer_input_size,
        layers=layers,
    ).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)

    with pytest.raises(RuntimeError):
        fc_stack(input_tensor)


def test_fc_stack_no_layers_behaves_like_passthrough():
    first_layer_input_size = 10
    layers = None
    num_layers = 0
    output_size = 15

    fc_stack = FCStack(
        first_layer_input_size=first_layer_input_size,
        layers=layers,
        num_layers=num_layers,
        default_output_size=output_size,
    ).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)
    output_tensor = fc_stack(input_tensor)

    assert list(output_tensor.shape[1:]) == [first_layer_input_size]
    assert output_tensor.shape[1:] == fc_stack.output_shape
    assert np.all(np.isclose(input_tensor, output_tensor))
