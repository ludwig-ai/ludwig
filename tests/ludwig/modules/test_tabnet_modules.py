import pytest
import torch

from ludwig.modules.tabnet_modules import Sparsemax

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
