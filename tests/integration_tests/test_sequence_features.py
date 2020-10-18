import copy
from io import StringIO

import pandas as pd
import pytest

import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from ludwig.utils.batcher import initialize_batcher
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import ENCODERS

DEFAULT_HIDDEN_SIZE = 8


@pytest.fixture(scope='module')
def generate_sequence_training_data():
    input_features = [
        sequence_feature(
            min_len=5,
            max_len=10,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        )
        # {'name': 'in_sequence', 'type': 'sequence', 'min_len': 2,
        #  'max_len': 10},
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

        # {'name': 'out_sequence', 'type': 'sequence', 'min_len': 3,
        #  'max_len': 10}
    ]

    dataset = build_synthetic_dataset(
        150,
        copy.deepcopy(input_features) + copy.deepcopy(output_features)
    )

    raw_data = '\n'.join([r[0] + ',' + r[1] for r in dataset])
    df = pd.read_csv(StringIO(raw_data))

    return df, input_features, output_features

@pytest.mark.parametrize('enc_cell_type', ['lstm', 'rnn', 'gru'])
@pytest.mark.parametrize('enc_encoder', ENCODERS)
def test_sequence_encoders(
    enc_encoder,
    enc_cell_type,
    generate_sequence_training_data
):
    # retrieve pre-computed dataset and features
    raw_df = generate_sequence_training_data[0]
    input_features = generate_sequence_training_data[1]
    output_features = generate_sequence_training_data[2]

    input_feature_name = input_features[0]['name']

    # setup input feature for testing
    input_features[0]['encoder'] = enc_encoder
    input_features[0]['cell_type'] = enc_cell_type
    config = {'input_features': input_features,
              'output_features': output_features}

    # setup model scaffolding to test feature
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
    batcher = initialize_batcher(
        training_set
    )
    while not batcher.last_batch():
        batch = batcher.next_batch()
        inputs = {
            i_feat.feature_name: batch[i_feat.feature_name]
            for i_feat in model.model.input_features.values()
        }
        # targets = {
        #     o_feat.feature_name: batch[o_feat.feature_name]
        #     for o_feat in model.model.output_features.values()
        # }
        # retrieve encoder to test
        encoder = model.model.input_features[input_feature_name].encoder_obj
        encoder_out = encoder(tf.cast(inputs[input_feature_name], dtype=tf.int32))

        # check encoder output for proper content, type and shape
        batch_size = batch[input_feature_name].shape[0]
        seq_size = input_features[0]['max_len']

        assert 'encoder_output' in encoder_out
        assert isinstance(encoder_out['encoder_output'], tf.Tensor)

        if enc_encoder == 'parallel_cnn':
            number_parallel_cnn_layers = \
                len(model.model.input_features[input_feature_name].encoder_obj.conv_layers)
            hidden_size = input_features[0]['state_size'] * number_parallel_cnn_layers
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, hidden_size]

        elif enc_encoder == 'stacked_parallel_cnn':
            number_parallel_cnn_layers = \
                len(model.model.input_features[input_feature_name]\
                    .encoder_obj.parallel_conv1d_stack\
                    .stacked_parallel_layers[0])
            hidden_size = input_features[0]['state_size'] * number_parallel_cnn_layers
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, hidden_size]

        elif enc_encoder == 'rnn':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, DEFAULT_HIDDEN_SIZE]

            assert 'encoder_output_state' in encoder_out
            if enc_cell_type == 'lstm':
                assert isinstance(encoder_out['encoder_output_state'], list)
                assert encoder_out['encoder_output_state'][0].shape.as_list() == \
                       [batch_size, DEFAULT_HIDDEN_SIZE]
                assert encoder_out['encoder_output_state'][1].shape.as_list() == \
                       [batch_size, DEFAULT_HIDDEN_SIZE]
            else:
                assert isinstance(encoder_out['encoder_output_state'], tf.Tensor)
                assert encoder_out['encoder_output_state'].shape.as_list() == \
                       [batch_size, DEFAULT_HIDDEN_SIZE]

        elif enc_encoder == 'cnnrnn':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, 1, DEFAULT_HIDDEN_SIZE]
            assert 'encoder_output_state' in encoder_out
            if enc_cell_type == 'lstm':
                assert isinstance(encoder_out['encoder_output_state'], list)
                assert encoder_out['encoder_output_state'][0].shape.as_list() == \
                       [batch_size, DEFAULT_HIDDEN_SIZE]
                assert encoder_out['encoder_output_state'][1].shape.as_list() == \
                       [batch_size, DEFAULT_HIDDEN_SIZE]
            else:
                assert isinstance(encoder_out['encoder_output_state'], tf.Tensor)
                assert encoder_out['encoder_output_state'].shape.as_list() == \
                       [batch_size, DEFAULT_HIDDEN_SIZE]

        elif enc_encoder == 'stacked_cnn':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, 1, DEFAULT_HIDDEN_SIZE]

        elif enc_encoder == 'transformer':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, 256]

        elif enc_encoder == 'embed':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, DEFAULT_HIDDEN_SIZE]

        else:
            raise ValueError('{} is an invalid encoder specification'\
                             .format(enc_encoder))
