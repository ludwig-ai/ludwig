import logging
import torch

from ludwig.modules import recurrent_modules

logger = logging.getLogger(__name__)


def test_recurrent_stack():
    recurrent_stack = recurrent_modules.RecurrentStack(
        input_size=10,
        sequence_size=19,
        hidden_size=256
    )
    assert recurrent_stack.output_shape == torch.Size([19, 256])

    # Batch (N), Length (L), Input (H)
    inputs = torch.rand(2, 19, 10)
    hidden, final_state = recurrent_stack(inputs)

    assert hidden.shape == torch.Size([2, 19, 256])
    assert final_state.shape == torch.Size([2, 256])


def test_recurrent_stack_NoSequenceSize():
    recurrent_stack = recurrent_modules.RecurrentStack(
        input_size=10,
        hidden_size=256
    )
    assert recurrent_stack.output_shape == torch.Size([256])

    # Batch (N), Length (L), Input (H)
    inputs = torch.rand(2, 19, 10)
    hidden, final_state = recurrent_stack(inputs)
    assert hidden.shape == torch.Size([2, 19, 256])
    assert final_state.shape == torch.Size([2, 256])
