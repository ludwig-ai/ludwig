from typing import List, Optional

import pytest
import torch

from ludwig.modules.fully_connected_modules import FCLayer, FCStack

BATCH_SIZE = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("input_size", [2, 3])
@pytest.mark.parametrize("output_size", [3, 4])
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_fc_layer(
    input_size: int,
    output_size: int,
    activation: str,
    dropout: float,
):
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
        (None, [{"input_size": 2, "output_size": 4}, {"output_size": 8}], None),
    ],
)
def test_fc_stack(first_layer_input_size: Optional[int], layers: Optional[List], num_layers: Optional[int]):
    if first_layer_input_size is None:
        first_layer_input_size = layers[0]["input_size"]
    fc_stack = FCStack(first_layer_input_size=first_layer_input_size, layers=layers, num_layers=num_layers).to(DEVICE)
    input_tensor = torch.randn(BATCH_SIZE, first_layer_input_size, device=DEVICE)
    output_tensor = fc_stack(input_tensor)
    assert output_tensor.shape[1:] == fc_stack.output_shape
