import logging
from typing import Union

import numpy as np
import pytest
import torch

from ludwig.constants import SEQUENCE
from ludwig.encoders.registry import get_encoder_cls
from tests.integration_tests.utils import ENCODERS

logger = logging.getLogger(__name__)

TEST_VOCAB_SIZE = 132
TEST_HIDDEN_SIZE = 32
TEST_STATE_SIZE = 16
TEST_EMBEDDING_SIZE = 64
TEST_NUM_FILTERS = 24
BATCH_SIZE = 2
SEQ_SIZE = 10
PARALLEL_CNN_LAYERS = 4

# encoder parameters combinations tested
encoder_parameters = {
    "vocab": [str(i) for i in range(TEST_VOCAB_SIZE)],
    "embedding_size": TEST_EMBEDDING_SIZE,
    "hidden_size": TEST_HIDDEN_SIZE,
    "num_filters": TEST_NUM_FILTERS,
    "num_layers": 1,
    "max_sequence_length": SEQ_SIZE,
    "state_size": TEST_STATE_SIZE,
    "cell_type": "rnn",
    "should_embed": True,
    "dropout": 0.0,
    "norm": None,
    "reduce_output": None,
}


@pytest.fixture(scope="module")
def input_sequence() -> torch.Tensor:
    # generates a realistic looking synthetic sequence tensor, i.e.
    # each sequence will have non-zero tokens at the beginning with
    # trailing zero tokens, including a max length token with a single
    # zero token at the end.  Example:
    # [
    #   [3, 5, 6, 0, 0, 0],
    #   [10, 11, 12, 13, 14, 0],   # max length sequence
    #   [32, 0, 0, 0, 0, 0]        # minimum length sequence
    # ]
    input_tensor = torch.zeros([BATCH_SIZE, SEQ_SIZE], dtype=torch.int32)
    sequence_lengths = np.random.randint(1, SEQ_SIZE, size=BATCH_SIZE)
    for i in range(input_tensor.shape[0]):
        input_tensor[i, : sequence_lengths[i]] = torch.tensor(
            np.random.randint(2, TEST_VOCAB_SIZE, size=sequence_lengths[i])
        )

    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    return input_tensor


@pytest.mark.parametrize("enc_reduce_output", [None, "sum"])
@pytest.mark.parametrize("enc_norm", [None, "batch", "layer"])
@pytest.mark.parametrize("enc_num_layers", [1, 2])
@pytest.mark.parametrize("enc_dropout", [0, 0.2])
@pytest.mark.parametrize("enc_cell_type", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("enc_encoder", ENCODERS + ["passthrough"])
def test_sequence_encoders(
    enc_encoder: str,
    enc_cell_type: str,
    enc_dropout: float,
    enc_num_layers: int,
    enc_norm: Union[None, str],
    enc_reduce_output: Union[None, str],
    input_sequence: torch.Tensor,
):
    # update encoder parameters for specific unit test case
    encoder_parameters["cell_type"] = enc_cell_type
    encoder_parameters["dropout"] = enc_dropout
    encoder_parameters["num_layers"] = enc_num_layers
    encoder_parameters["norm"] = enc_norm
    encoder_parameters["reduce_output"] = enc_reduce_output

    # retrieve encoder to test
    encoder_obj = get_encoder_cls(SEQUENCE, enc_encoder)(**encoder_parameters)
    if torch.cuda.is_available():
        encoder_obj = encoder_obj.cuda()

    encoder_out = encoder_obj(input_sequence)

    assert "encoder_output" in encoder_out
    assert isinstance(encoder_out["encoder_output"], torch.Tensor)

    if enc_encoder == "parallel_cnn":
        number_parallel_cnn_layers = PARALLEL_CNN_LAYERS
        output_dimension = encoder_parameters["num_filters"] * number_parallel_cnn_layers
        assert (
            encoder_out["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, output_dimension)
            if enc_reduce_output is None
            else (BATCH_SIZE, output_dimension)
        )

    elif enc_encoder == "stacked_parallel_cnn":
        number_parallel_cnn_layers = PARALLEL_CNN_LAYERS
        output_dimension = encoder_parameters["num_filters"] * number_parallel_cnn_layers
        assert (
            encoder_out["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, output_dimension)
            if enc_reduce_output is None
            else (BATCH_SIZE, output_dimension)
        )

    elif enc_encoder == "rnn":
        assert (
            encoder_out["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, TEST_STATE_SIZE)
            if enc_reduce_output is None
            else (BATCH_SIZE, TEST_STATE_SIZE)
        )

        assert "encoder_output_state" in encoder_out
        if enc_cell_type == "lstm":
            assert isinstance(encoder_out["encoder_output_state"], tuple)
            assert isinstance(encoder_out["encoder_output_state"][0], torch.Tensor)
            assert isinstance(encoder_out["encoder_output_state"][1], torch.Tensor)
            assert encoder_out["encoder_output_state"][0].shape == (BATCH_SIZE, TEST_STATE_SIZE)
            assert encoder_out["encoder_output_state"][1].shape == (BATCH_SIZE, TEST_STATE_SIZE)
        else:
            assert isinstance(encoder_out["encoder_output_state"], torch.Tensor)
            assert encoder_out["encoder_output_state"].shape == (BATCH_SIZE, TEST_STATE_SIZE)

    elif enc_encoder == "cnnrnn":
        assert encoder_out["encoder_output"].shape[1:] == encoder_obj.output_shape
        assert "encoder_output_state" in encoder_out

        if enc_cell_type == "lstm":
            assert isinstance(encoder_out["encoder_output_state"], tuple)
            assert encoder_out["encoder_output_state"][0].shape == (BATCH_SIZE, TEST_STATE_SIZE)
            assert encoder_out["encoder_output_state"][1].shape == (BATCH_SIZE, TEST_STATE_SIZE)
        else:
            assert isinstance(encoder_out["encoder_output_state"], torch.Tensor)
            assert encoder_out["encoder_output_state"].shape == (BATCH_SIZE, TEST_STATE_SIZE)

    elif enc_encoder == "stacked_cnn":
        assert encoder_out["encoder_output"].shape[1:] == encoder_obj.output_shape

    elif enc_encoder == "embed":
        assert (
            encoder_out["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, TEST_EMBEDDING_SIZE)
            if enc_reduce_output is None
            else (BATCH_SIZE, TEST_EMBEDDING_SIZE)
        )

    elif enc_encoder == "transformer":
        assert encoder_out["encoder_output"].shape[1:] == encoder_obj.output_shape

    elif enc_encoder == "passthrough":
        assert (
            encoder_out["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, 1)
            if enc_reduce_output is None
            else (BATCH_SIZE, 1)
        )

    else:
        raise ValueError(f"{enc_encoder} is an invalid encoder specification")


@pytest.mark.parametrize("enc_reduce_output", [None, "sum", "last", "mean", "max", "concat"])
def test_passthrough_encoder(enc_reduce_output, input_sequence):
    encoder_parameters = {"reduce_output": enc_reduce_output}

    # retrieve encoder to test
    encoder_obj = get_encoder_cls(SEQUENCE, "passthrough")(**encoder_parameters)

    encoder_out = encoder_obj(input_sequence)

    assert "encoder_output" in encoder_out
    assert (
        encoder_out["encoder_output"].shape == (BATCH_SIZE, SEQ_SIZE, 1)
        if enc_reduce_output is None
        else (BATCH_SIZE, 1)
    )


# test to ensure correct handling of vocab_size and embedding_size specifications
@pytest.mark.parametrize("enc_embedding_size", [TEST_VOCAB_SIZE - 8, TEST_VOCAB_SIZE, TEST_VOCAB_SIZE + 8])
def test_sequence_embed_encoder(enc_embedding_size: int, input_sequence: torch.Tensor) -> None:
    encoder_parameters["embedding_size"] = enc_embedding_size

    encoder_obj = get_encoder_cls(SEQUENCE, "embed")(**encoder_parameters)

    encoder_out = encoder_obj(input_sequence)

    assert encoder_out["encoder_output"].size()[1:] == encoder_obj.output_shape
