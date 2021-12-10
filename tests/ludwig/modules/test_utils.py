from typing import Tuple

import torch

from ludwig.utils.torch_utils import LudwigModule


def assert_output_shapes(module: LudwigModule, input_shape: Tuple[int]):
    """Runs a unit test to confirm that the out shape matches expected output.

    module: Module to be tested.
    input_shape: List of integers of the expected input shape (w/o batch dim).
    """

    inputs = torch.rand(2, *input_shape, dtype=module.input_dtype)
    output_tensor = module(inputs)
    assert output_tensor.shape[1:] == module.output_shape
