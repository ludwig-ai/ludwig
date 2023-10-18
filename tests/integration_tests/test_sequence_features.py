import contextlib
import copy
from io import StringIO

import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import DECODER, ENCODER_OUTPUT_STATE, LOGITS
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from tests.integration_tests.utils import generate_data, run_experiment, sequence_feature

#
# this test is focused on testing input sequence features with all encoders
# and output sequence feature with Generator decoder.  Except for specified
# configuration parameters all other parameters assume default values.
#

TEST_VOCAB_SIZE = 132
TEST_HIDDEN_SIZE = 32
TEST_STATE_SIZE = 8
TEST_EMBEDDING_SIZE = 64
TEST_NUM_FILTERS = 24


# generates dataset that can be used for rest of test
@pytest.fixture(scope="module")
def generate_sequence_training_data():
    input_features = [
        sequence_feature(
            encoder={
                "vocab_size": TEST_VOCAB_SIZE,
                "embedding_size": TEST_EMBEDDING_SIZE,
                "state_size": TEST_STATE_SIZE,
                "hidden_size": TEST_HIDDEN_SIZE,
                "num_filters": TEST_NUM_FILTERS,
                "min_len": 5,
                "max_len": 10,
                "type": "rnn",
                "cell_type": "lstm",
            }
        )
    ]

    output_features = [
        sequence_feature(
            decoder={"type": "generator", "min_len": 5, "max_len": 10, "cell_type": "lstm", "attention": "bahdanau"}
        )
    ]

    # generate synthetic data set testing
    dataset = build_synthetic_dataset(150, copy.deepcopy(input_features) + copy.deepcopy(output_features))
    raw_data = "\n".join([r[0] + "," + r[1] for r in dataset])
    df = pd.read_csv(StringIO(raw_data))

    return df, input_features, output_features


# setups up minimal number of data structures required to support initialized
# input and output features.  The function returns initialized LudwigModel
# and batcher for training dataset
@contextlib.contextmanager
def setup_model_scaffolding(raw_df, input_features, output_features):
    # setup input feature for testing
    config = {"input_features": input_features, "output_features": output_features}

    # setup model scaffolding to for testing
    model = LudwigModel(config)
    training_set, _, _, training_set_metadata = preprocess_for_training(
        model.config, training_set=raw_df, skip_save_processed_input=True
    )
    model.training_set_metadata = training_set_metadata
    update_config_with_metadata(model.config_obj, training_set_metadata)
    model.model = model.create_model(model.config_obj)

    # setup batcher to go through synthetic data
    with training_set.initialize_batcher() as batcher:
        yield model, batcher


# TODO(#1333): Refactor this test once torch sequence generator work is complete.
# - Tests may be covered by other smaller scoped unit tests.
#
# tests output feature sequence with `Generator` decoder
# pytest parameters
#   dec_cell_type: decoder cell type
#   combiner_output_shapes: is a 2-tuple specifies the possible types of
#     tensors that the combiner may generate for sequences.
#     combiner_output_shapes[0]: specifies shape for hidden key
#     combiner_output_shapes[1]: is either None or 1 or 2-tuple representing
#       the encoder_output_state key. None: no encoder_output_state key,
#       1-tuple: generate tf.Tensor, 2-tuple: generate list with 2 tf.Tensors
# TODO(Justin): Move these to test_sequence_generator unit tests, and reintroduce decoder attention, beam_width, and
# num_layers when these are reimplemented.
@pytest.mark.parametrize("dec_cell_type", ["lstm", "rnn", "gru"])
@pytest.mark.parametrize(
    "combiner_output_shapes",
    [
        ((128, 10, TEST_STATE_SIZE), None),
        ((128, 10, TEST_STATE_SIZE), ((128, TEST_STATE_SIZE), (128, TEST_STATE_SIZE))),
        ((128, 10, TEST_STATE_SIZE), ((128, TEST_STATE_SIZE),)),
    ],
)
def test_sequence_decoders(
    dec_cell_type,
    combiner_output_shapes,
    generate_sequence_training_data,
):
    # retrieve pre-computed dataset and features
    raw_df = generate_sequence_training_data[0]
    input_features = generate_sequence_training_data[1]
    output_features = generate_sequence_training_data[2]
    output_feature_name = output_features[0]["name"]
    output_features[0][DECODER]["cell_type"] = dec_cell_type

    with setup_model_scaffolding(raw_df, input_features, output_features) as (model, _):
        # generate synthetic encoder_output tensors and make it look like
        # it came out of the combiner
        encoder_output = torch.randn(combiner_output_shapes[0])
        combiner_outputs = {"hidden": encoder_output}

        if combiner_output_shapes[1] is not None:
            if len(combiner_output_shapes[1]) > 1:
                encoder_output_state = (
                    torch.randn(combiner_output_shapes[1][0]),
                    torch.randn(combiner_output_shapes[1][1]),
                )
            else:
                encoder_output_state = torch.randn(combiner_output_shapes[1][0])

            combiner_outputs[ENCODER_OUTPUT_STATE] = encoder_output_state

        decoder = model.model.output_features.get(output_feature_name).decoder_obj
        decoder_out = decoder(combiner_outputs)

        # gather expected components of the shape
        batch_size = combiner_outputs["hidden"].shape[0]
        seq_size = output_features[0][DECODER]["max_len"] + 2  # For start and stop symbols.
        vocab_size = model.config_obj.output_features.to_list()[0][DECODER]["vocab_size"]

        # confirm shape and format of decoder output
        assert list(decoder_out[LOGITS].size()) == [batch_size, seq_size, vocab_size]


# final sanity test.  Checks a subset of sequence parameters
@pytest.mark.parametrize("dec_cell_type", ["lstm", "rnn", "gru"])
@pytest.mark.parametrize("enc_cell_type", ["lstm", "rnn", "gru"])
@pytest.mark.parametrize("enc_encoder", ["embed", "rnn"])
def test_sequence_generator(enc_encoder, enc_cell_type, dec_cell_type, csv_filename):
    # Define input and output features
    input_features = [
        sequence_feature(encoder={"type": enc_encoder, "min_len": 5, "max_len": 10, "cell_type": enc_cell_type})
    ]
    output_features = [
        sequence_feature(decoder={"type": "generator", "min_len": 5, "max_len": 10, "cell_type": dec_cell_type})
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)
