import pytest
import torch

from ludwig.modules.tabnet_modules import Sparsemax

BATCH_SIZE = 16
HIDDEN_SIZE = 8


# x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
# tfa.activations.sparsemax(x)
# < tf.Tensor: shape = (2, 3), dtype = float32, numpy =
# array([[0., 0., 1.],
#        [0., 0., 1.]], dtype=float32) >

def test_sparsemax():
    input_tensor = torch.tensor(
        [[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]],
        dtype=torch.float32
    )

    sparsemax = Sparsemax()

    output_tensor = sparsemax(input_tensor)

    assert isinstance(output_tensor, torch.Tensor)
