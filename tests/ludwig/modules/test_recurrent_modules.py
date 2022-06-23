import logging

import pytest
import torch

from ludwig.modules import recurrent_modules

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("max_sequence_length,expected_output_shape", [(19, [19, 256]), (None, [256])])
def test_recurrent_stack(max_sequence_length, expected_output_shape):
    recurrent_stack = recurrent_modules.RecurrentStack(
        input_size=10, max_sequence_length=max_sequence_length, hidden_size=256
    )
    assert recurrent_stack.output_shape == torch.Size(expected_output_shape)

    # Batch (N), Length (L), Input (H)
    inputs = torch.rand(2, 19, 10)
    hidden, final_state = recurrent_stack(inputs)

    assert hidden.shape == torch.Size([2, 19, 256])
    assert final_state.shape == torch.Size([2, 256])
