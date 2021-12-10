import torch

from ludwig.encoders.binary_encoders import BinaryPassthroughEncoder


def test_binary_passthrough_encoder():
    binary_encoder = BinaryPassthroughEncoder()
    inputs = torch.rand(2, 1)
    outputs = binary_encoder(inputs)
    assert outputs.shape[1:] == binary_encoder.output_shape
