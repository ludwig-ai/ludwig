import pytest
import torch

from ludwig.encoders.generic_encoders import DenseEncoder, PassthroughEncoder


@pytest.mark.parametrize("input_size", [1, 2, 10])
@pytest.mark.parametrize("categorical", [True, False])
def test_generic_passthrough_encoder(input_size, categorical):
    passthrough_encoder = PassthroughEncoder(input_size)
    # Passthrough encoder allows categorical input feature (int), dense encoder's input must be float.
    if categorical:
        inputs = torch.randint(10, (10, input_size))
    else:
        inputs = torch.rand((10, input_size))
    outputs = passthrough_encoder(inputs)
    assert "encoder_output" in outputs
    encoder_output = outputs["encoder_output"]
    # Ensures output shape matches encoder expected output shape.
    assert encoder_output.shape[1:] == passthrough_encoder.output_shape


@pytest.mark.parametrize("input_size", [1, 2, 10])
@pytest.mark.parametrize("num_layers", [1, 3, 6])
@pytest.mark.parametrize("fc_size", [1, 2, 10, 256])
def test_generic_dense_encoder(input_size, num_layers, fc_size):
    dense_encoder = DenseEncoder(input_size, num_layers=num_layers, fc_size=fc_size)
    inputs = torch.rand((10, input_size))
    outputs = dense_encoder(inputs)
    assert "encoder_output" in outputs
    encoder_output = outputs["encoder_output"]
    # Ensures output shape matches encoder expected output shape.
    assert encoder_output.shape[1:] == dense_encoder.output_shape
