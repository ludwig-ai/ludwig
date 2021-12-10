import logging
from typing import Tuple

import pytest
import torch

from ludwig.modules import reduction_modules

logger = logging.getLogger(__name__)


# @pytest.mark.parametrize("reduce_mode", ["last", "sum", "mean", "avg", "max", "concat", None])
@pytest.mark.parametrize("reduce_mode", ["last", "concat"])
@pytest.mark.parametrize("test_input_shape", [(16, 1, 4), (8, 10, 16)])
def test_sequence_reducer(reduce_mode: str, test_input_shape: Tuple[int, ...]):
    batch_size = test_input_shape[0]
    max_sequence_length = test_input_shape[1]
    sequence_reducer = reduction_modules.SequenceReducer(reduce_mode=reduce_mode)
    inputs = torch.zeros(test_input_shape)
    # Generates random sequence of random length for each instance in batch.
    for batch_index in range(batch_size):
        sequence_length = torch.randint(max_sequence_length, (1,))
        inputs[batch_index, :sequence_length] = torch.rand((sequence_length,) + test_input_shape[2:])
    outputs = sequence_reducer(inputs)
    assert outputs.shape == sequence_reducer.infer_output_shape(test_input_shape)

