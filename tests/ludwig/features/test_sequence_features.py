from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import LENGTHS, PREDICTIONS, PROBABILITIES, PROBABILITY
from ludwig.features.sequence_feature import SequenceInputFeature, SequenceOutputFeature
from ludwig.features.text_feature import TextInputFeature, TextOutputFeature
from ludwig.utils.data_utils import from_numpy_dataset
from tests.integration_tests.utils import ENCODERS, sequence_feature

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
SEQ_SIZE = 6
VOCAB_SIZE = 64


@pytest.fixture(scope="module")
def input_sequence() -> Tuple[torch.Tensor, List]:
    # generates a realistic looking synthetic sequence tensor, i.e.
    # each sequence will have non-zero tokens at the beginning with
    # trailing zero tokens, including a max length token with a single
    # zero token at the end.  Example:
    # [
    #   [3, 5, 6, 0, 0, 0],
    #   [10, 11, 12, 13, 14, 0],   # max length sequence
    #   [32, 0, 0, 0, 0, 0]        # minimum length sequence
    # ]
    input_tensor = torch.zeros([BATCH_SIZE, SEQ_SIZE], dtype=torch.int32).to(DEVICE)
    sequence_lengths = np.random.randint(1, SEQ_SIZE, size=BATCH_SIZE)
    for i in range(input_tensor.shape[0]):
        input_tensor[i, : sequence_lengths[i]] = torch.tensor(
            np.random.randint(2, VOCAB_SIZE, size=sequence_lengths[i])
        )

    # emulate idx2str structure
    idx2str = ["<PAD>", "<UNK>"] + [str(i) for i in range(2, VOCAB_SIZE)]

    return input_tensor, idx2str


@pytest.fixture(scope="module")
def output_result():
    """Generates a realistic looking `result` DataFrame for postprocessing."""
    logits = torch.randn([BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE], dtype=torch.float32).to(DEVICE)
    probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()

    # sentence level probability
    probability_expected = np.prod(np.max(probabilities, axis=-1), axis=-1)

    # token level predictions
    predictions = np.argmax(probabilities, axis=-1)
    lengths = np.full([BATCH_SIZE, 1], SEQ_SIZE)

    # TODO: include test for LAST_PREDICTIONS functionality
    output_dict = {LENGTHS: lengths, PREDICTIONS: predictions, PROBABILITIES: probabilities}
    idx2str = ["<PAD>", "<UNK>"] + [str(i) for i in range(2, VOCAB_SIZE)]

    result = from_numpy_dataset(output_dict)
    return result, idx2str, probability_expected


@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("sequence_type", [SequenceInputFeature, TextInputFeature])
def test_sequence_input_feature(
    input_sequence: tuple, encoder: str, sequence_type: Union[SequenceInputFeature, TextInputFeature]
) -> None:
    # test assumes "sequence data" has been tokenized and converted to
    # numeric representation.  Focus of this test is primarily on
    # integration with encoder with correctly sized encoder tensor and
    # required properties are present

    input_sequence, idx2str = input_sequence

    # setup input sequence feature definition
    # use sequence_feature() to generate baseline
    # sequence definition and then augment with
    # pre-processing metadata parameters
    input_feature_defn = sequence_feature(
        encoder=encoder,
        max_len=SEQ_SIZE,
        # augment with emulated pre-processing metadata
        max_sequence_length=SEQ_SIZE,
        vocab=idx2str,
    )

    # create sequence input feature object
    input_feature_obj = sequence_type(input_feature_defn).to(DEVICE)

    # confirm dtype property
    assert input_feature_obj.input_dtype == torch.int32

    # confirm input_shape property
    assert input_feature_obj.input_shape == (SEQ_SIZE,)

    # confirm output_shape property default output shape
    # from sequence_feature() function
    encoder_output = input_feature_obj(input_sequence)
    assert encoder_output["encoder_output"].shape == (BATCH_SIZE, *input_feature_obj.output_shape)


@pytest.mark.parametrize("sequence_type", [SequenceOutputFeature, TextOutputFeature])
def test_sequence_output_feature(sequence_type: Union[SequenceInputFeature, TextInputFeature]):
    output_feature_defn = sequence_feature(
        max_len=SEQ_SIZE, max_sequence_length=SEQ_SIZE, vocab_size=VOCAB_SIZE, input_size=VOCAB_SIZE
    )
    output_feature_obj = sequence_type(output_feature_defn, {}).to(DEVICE)
    combiner_outputs = {}
    combiner_outputs["combiner_output"] = torch.randn([BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE], dtype=torch.float32).to(
        DEVICE
    )

    text_output = output_feature_obj(combiner_outputs, {})

    assert "last_hidden" in text_output
    assert "logits" in text_output
    assert text_output["logits"].size() == torch.Size([BATCH_SIZE, SEQ_SIZE, VOCAB_SIZE])


def test_text_feature_postprocess_predictions(output_result: Tuple[pd.DataFrame, List, np.ndarray]) -> None:
    result, idx2str, probability_expected = output_result
    output_feature_defn = sequence_feature(
        max_len=SEQ_SIZE, max_sequence_length=SEQ_SIZE, vocab_size=VOCAB_SIZE, input_size=VOCAB_SIZE
    )
    output_feature_obj = TextOutputFeature(output_feature_defn, {}).to(DEVICE)
    result = result.rename({col: f"{output_feature_obj.feature_name}_{col}" for col in result.columns}, axis=1)

    postprocessed_result = output_feature_obj.postprocess_predictions(result, {"idx2str": idx2str}, "", LOCAL_BACKEND)
    probability = np.stack(postprocessed_result[f"{output_feature_obj.feature_name}_{PROBABILITY}"].to_numpy())

    assert np.allclose(probability, probability_expected)
