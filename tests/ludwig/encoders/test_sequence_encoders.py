from typing import List

import pytest
import torch

from ludwig.encoders.sequence_encoders import (
    SequenceEmbedEncoder,
    SequencePassthroughEncoder,
    StackedCNNRNN,
    StackedParallelCNN,
    StackedRNN,
    StackedTransformer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "last", "concat", "attention", None])
def test_sequence_passthrough_encoder(reduce_output: str):
    sequence_passthrough_encoder = SequencePassthroughEncoder(
        reduce_output=reduce_output,
        max_sequence_length=10,
        encoding_size=8
    ).to(DEVICE)
    inputs = torch.rand(20, 10, 8).to(DEVICE)
    outputs = sequence_passthrough_encoder(inputs)
    # SequencePassthroughEncoder does not implement output_shape, expect output to match input shape after reduce.
    assert outputs["encoder_output"].shape[1:] == sequence_passthrough_encoder.reduce_sequence.output_shape


@pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "last", "concat", "attention", None])
def test_sequence_embed_encoder(reduce_output: str):
    sequence_embed_encoder = SequenceEmbedEncoder(
        vocab=[1,2],
        max_sequence_length=10,
        reduce_output=reduce_output
    ).to(DEVICE)
    inputs = torch.randint(2, (20, 10)).to(DEVICE)
    outputs = sequence_embed_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == sequence_embed_encoder.output_shape

# Remaining Encoders
# ParallelCNN
# StackedCNN
# StackedParallelCNN
# StackedRNN
# StackedCNNRNN
# StackedTransformer


