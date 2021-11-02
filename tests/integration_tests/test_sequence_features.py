import contextlib
import copy
from io import StringIO

import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import run_experiment
from tests.integration_tests.utils import sequence_feature, numerical_feature

#
# this test is focused on testing input sequence features with all encoders
# and output sequence feature with Generator decoder.  Except for specified
# configuration parameters all other parameters assume default values.
#

TEST_VOCAB_SIZE = 132
TEST_HIDDEN_SIZE = 32
TEST_STATE_SIZE = 16
TEST_EMBEDDING_SIZE = 64
TEST_NUM_FILTERS = 24


# generates dataset that can be used for rest of test
@pytest.fixture(scope='module')
def generate_sequence_training_data():
    input_features = [
        sequence_feature(
            vocab_size=TEST_VOCAB_SIZE,
            embedding_size=TEST_EMBEDDING_SIZE,
            state_size=TEST_STATE_SIZE,
            hidden_size=TEST_HIDDEN_SIZE,
            num_filters=TEST_NUM_FILTERS,
            min_len=5,
            max_len=10,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        )
    ]

    output_features = [
        sequence_feature(
            min_len=5,
            max_len=10,
            decoder='generator',
            cell_type='lstm',
            attention='bahdanau',
            reduce_input=None
        )
    ]

    # generate synthetic data set testing
    dataset = build_synthetic_dataset(
        150,
        copy.deepcopy(input_features) + copy.deepcopy(output_features)
    )
    raw_data = '\n'.join([r[0] + ',' + r[1] for r in dataset])
    df = pd.read_csv(StringIO(raw_data))

    return df, input_features, output_features


# setups up minimal number of data structures required to support initialized
# input and output features.  The function returns initialized LudwigModel
# and batcher for training dataset
@contextlib.contextmanager
def setup_model_scaffolding(
        raw_df,
        input_features,
        output_features
):
    # setup input feature for testing
    config = {'input_features': input_features,
              'output_features': output_features}

    # setup model scaffolding to for testing
    model = LudwigModel(config)
    training_set, _, _, training_set_metadata = preprocess_for_training(
        config,
        training_set=raw_df,
        skip_save_processed_input=True
    )
    model.training_set_metadata = training_set_metadata
    update_config_with_metadata(
        model.config,
        training_set_metadata
    )
    model.model = model.create_model(model.config)

    # setup batcher to go through synthetic data
    with training_set.initialize_batcher() as batcher:
        yield model, batcher


# TODO(#1333): refactor test once torch sequence generator work is complete
#
# tests output feature sequence with `Generator` decoder
# pytest parameters
#   dec_cell_type: decoder cell type
#   dec_attention: decoder's attention mechanism
#   dec_beam_width: decoder's beam search width
#   combiner_output_shapes: is a 2-tuple specifies the possible types of
#     tensors that the combiner may generate for sequences.
#     combiner_output_shapes[0]: specifies shape for hidden key
#     combiner_output_shapes[1]: is either None or 1 or 2-tuple representing
#       the encoder_output_state key. None: no encoder_output_state key,
#       1-tuple: generate tf.Tensor, 2-tuple: generate list with 2 tf.Tensors
#
@pytest.mark.parametrize('dec_num_layers', [1, 2])
@pytest.mark.parametrize('dec_beam_width', [1, 2])
@pytest.mark.parametrize('dec_attention', ['bahdanau', 'luong', None])
@pytest.mark.parametrize('dec_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize(
    'combiner_output_shapes',
    [
        ((128, 10, 8), None),
        ((128, 10, 32), None),
        ((128, 10, 8), ((128, 8), (128, 8))),
        ((128, 10, 8), ((128, 8),))

    ]
)
def test_sequence_decoders(
        dec_cell_type,
        dec_attention,
        dec_beam_width,
        dec_num_layers,
        combiner_output_shapes,
        generate_sequence_training_data
):
    # retrieve pre-computed dataset and features
    raw_df = generate_sequence_training_data[0]
    input_features = generate_sequence_training_data[1]
    output_features = generate_sequence_training_data[2]
    output_feature_name = output_features[0]['name']
    output_features[0]['cell_type'] = dec_cell_type
    output_features[0]['attention'] = dec_attention
    output_features[0]['beam_width'] = dec_beam_width
    output_features[0]['num_layers'] = dec_num_layers

    with setup_model_scaffolding(
        raw_df,
        input_features,
        output_features
    ) as (model, _):

        # generate synthetic encoder_output tensors and make it look like
        # it came out of the combiner
        encoder_output = torch.randn(combiner_output_shapes[0])
        combiner_outputs = {'hidden': encoder_output}

        if combiner_output_shapes[1] is not None:
            if len(combiner_output_shapes[1]) > 1:
                encoder_output_state = [
                    torch.randn(combiner_output_shapes[1][0]),
                    torch.randn(combiner_output_shapes[1][1])
                ]
            else:
                encoder_output_state = \
                    torch.randn(combiner_output_shapes[1][0])

            combiner_outputs['encoder_output_state'] = encoder_output_state

        decoder = model.model.output_features[output_feature_name].decoder_obj
        decoder_out = decoder(combiner_outputs)

        # gather expected components of the shape
        batch_size = combiner_outputs['hidden'].shape[0]
        seq_size = output_features[0]['max_len']
        num_classes = model.config['output_features'][0]['num_classes']

        # confirm output is what is expected
        assert len(decoder_out) == 5
        logits, lengths, preds, last_preds, probs = decoder_out

        # confirm shape and format of deocoder output
        if dec_beam_width > 1:
            assert logits is None
        else:
            assert isinstance(logits, torch.Tensor)
            assert logits.shape.as_list() == [
                batch_size, seq_size, num_classes]

        assert isinstance(lengths, torch.Tensor)
        assert lengths.shape.as_list() == [batch_size]

        assert isinstance(preds, torch.Tensor)
        assert preds.shape.as_list() == [batch_size, seq_size]

        assert isinstance(last_preds, torch.Tensor)
        assert last_preds.shape.as_list() == [batch_size]

        assert isinstance(probs, torch.Tensor)
        assert probs.shape.as_list() == [batch_size, seq_size, num_classes]


# todo: refactor test once torch sequence generator work is complete
#
# final sanity test.  Checks a subset of sequence parameters
#
@pytest.mark.parametrize('dec_num_layers', [1, 2])
@pytest.mark.parametrize('dec_beam_width', [1, 2])
@pytest.mark.parametrize('dec_attention', ['bahdanau', 'luong', None])
@pytest.mark.parametrize('dec_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('enc_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('enc_encoder', ['embed', 'rnn'])
def test_sequence_generator(
        enc_encoder,
        enc_cell_type,
        dec_cell_type,
        dec_attention,
        dec_beam_width,
        dec_num_layers,
        csv_filename
):
    # Define input and output features
    input_features = [
        sequence_feature(
            min_len=5,
            max_len=10,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        )
    ]
    output_features = [
        sequence_feature(
            min_len=5,
            max_len=10,
            decoder='generator',
            cell_type='lstm',
            attention='bahdanau',
            reduce_input=None
        )
    ]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)

    # setup encoder specification
    input_features[0]['encoder'] = enc_encoder
    input_features[0]['cell_type'] = enc_cell_type

    # setup decoder specification
    output_features[0]['cell_type'] = dec_cell_type
    output_features[0]['attention'] = dec_attention
    output_features[0]['beam_width'] = dec_beam_width
    output_features[0]['num_layers'] = dec_num_layers

    # run the experiment
    run_experiment(input_features, output_features, dataset=rel_path)
