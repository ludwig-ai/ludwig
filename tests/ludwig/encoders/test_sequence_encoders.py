from typing import Type

import pytest
import torch

from ludwig.encoders.sequence_encoders import (
    ParallelCNN,
    SequenceEmbedEncoder,
    SequencePassthroughEncoder,
    StackedCNN,
    StackedCNNRNN,
    StackedParallelCNN,
    StackedRNN,
    StackedTransformer,
)
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

DEVICE = get_torch_device()
RANDOM_SEED = 1919


@pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "last", "concat", "attention", None])
def test_sequence_passthrough_encoder(reduce_output: str):
    batch_size = 10
    sequence_length = 32
    sequence_passthrough_encoder = SequencePassthroughEncoder(
        reduce_output=reduce_output, max_sequence_length=sequence_length, encoding_size=8
    ).to(DEVICE)
    inputs = torch.rand(batch_size, sequence_length, 8).to(DEVICE)
    outputs = sequence_passthrough_encoder(inputs)
    # SequencePassthroughEncoder does not implement output_shape, expect output to match input shape after reduce.
    assert outputs["encoder_output"].shape[1:] == sequence_passthrough_encoder.reduce_sequence.output_shape


@pytest.mark.parametrize(
    "encoder_type",
    [SequenceEmbedEncoder, ParallelCNN, StackedCNN, StackedParallelCNN, StackedRNN, StackedCNNRNN, StackedTransformer],
)
@pytest.mark.parametrize("reduce_output", ["mean", "avg", "max", "last", "concat", "attention", None])
@pytest.mark.parametrize("vocab_size", [2, 1024])  # Uses vocabularies smaller than (and larger than) embedding size.
def test_sequence_encoders(encoder_type: Type, reduce_output: str, vocab_size: int):
    # make repeatable
    torch.manual_seed(RANDOM_SEED)

    batch_size = 10
    sequence_length = 32
    sequence_encoder = encoder_type(
        vocab=list(range(1, vocab_size + 1)), max_sequence_length=sequence_length, reduce_output=reduce_output
    ).to(DEVICE)
    inputs = torch.randint(2, (batch_size, sequence_length)).to(DEVICE)
    outputs = sequence_encoder(inputs)
    assert outputs["encoder_output"].shape[1:] == sequence_encoder.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sequence_encoder, (inputs,), target)

    assert (
        upc == tpc
    ), f"Not all parameters updated.  Parameters not updated: {not_updated}.\nModule: {sequence_encoder}"
