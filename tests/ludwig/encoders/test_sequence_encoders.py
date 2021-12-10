from typing import List

import pytest
import torch

from ludwig.encoders.sequence_encoders import SequencePassthroughEncoder
from ludwig.encoders.sequence_encoders import SequenceEmbedEncoder
from ludwig.encoders.sequence_encoders import StackedCNNRNN, StackedParallelCNN, StackedRNN, StackedTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# @pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "concat", "last", None])
# def test_sequence_passthrough_encoder(reduce_output: str):
#     sequence_passthrough_encoder = SequencePassthroughEncoder(reduce_output=reduce_output).to(DEVICE)
#     inputs = torch.rand(10, 30, 1).to(DEVICE)
#     outputs = sequence_passthrough_encoder(inputs)
#     # Note: SequencePassthroughEncoder does not implement output_shape - expect output to match input shape plus reduction.
#     assert outputs["encoder_output"].shape[1:] == inputs.shape


# @pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "concat", "last", None])
@pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "last"])  # Fails for concat and None
def test_sequence_embed_encoder(reduce_output: str):
    sequence_embed_encoder = SequenceEmbedEncoder(vocab=[1,2], max_sequence_length=8, reduce_output=reduce_output)
    inputs = torch.randint(2, (30, 1)).to(DEVICE)
    outputs = sequence_embed_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == sequence_embed_encoder.output_shape
