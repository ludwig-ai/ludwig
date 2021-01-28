import copy
from io import StringIO

import pandas as pd
import pytest
import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_registries import update_config_with_metadata
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import run_experiment
from tests.integration_tests.utils import sequence_feature

#
# this test is focused on testing input sequence features with all encoders
# and output sequence feature with Generator decoder.  Except for specified
# configuration parameters all other parameters assume default values.
#

TEST_HIDDEN_SIZE = 8


# generates dataset that can be used for rest of test
# Note: sequence_feature function specifies hidden size to be 8
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
    batcher = training_set.initialize_batcher()

    return model, batcher


# test sequence input features
# pytest parameters:
#   enc_cell_type: encoder cell types
#   enc_encoder: sequence input feature encoder
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
    input_features[0]['encoder'] = enc_encoder
    input_features[0]['cell_type'] = enc_cell_type

    model, batcher = setup_model_scaffolding(
        raw_df,
        input_features,
        output_features
    )

    while not batcher.last_batch():
        batch = batcher.next_batch()
        inputs = {
            i_feat.feature_name: batch[i_feat.proc_column]
            for i_feat in model.model.input_features.values()
        }

        # retrieve encoder to test
        encoder = model.model.input_features[input_feature_name].encoder_obj
        encoder_out = encoder(
            tf.cast(inputs[input_feature_name], dtype=tf.int32))

        # check encoder output for proper content, type and shape
        proc_column = model.model.input_features[
            input_feature_name].proc_column
        batch_size = batch[proc_column].shape[0]
        seq_size = input_features[0]['max_len']

        assert 'encoder_output' in encoder_out
        assert isinstance(encoder_out['encoder_output'], tf.Tensor)

        if enc_encoder == 'parallel_cnn':
            number_parallel_cnn_layers = \
                len(model.model.input_features[
                        input_feature_name].encoder_obj.conv_layers)
            hidden_size = input_features[0][
                              'state_size'] * number_parallel_cnn_layers
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, hidden_size]

        elif enc_encoder == 'stacked_parallel_cnn':
            number_parallel_cnn_layers = \
                len(model.model.input_features[input_feature_name] \
                    .encoder_obj.parallel_conv1d_stack \
                    .stacked_parallel_layers[0])
            hidden_size = input_features[0][
                              'state_size'] * number_parallel_cnn_layers
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, hidden_size]

        elif enc_encoder == 'rnn':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, seq_size, TEST_HIDDEN_SIZE]

            assert 'encoder_output_state' in encoder_out
            if enc_cell_type == 'lstm':
                assert isinstance(encoder_out['encoder_output_state'], list)
                assert encoder_out['encoder_output_state'][
                           0].shape.as_list() == \
                       [batch_size, TEST_HIDDEN_SIZE]
                assert encoder_out['encoder_output_state'][
                           1].shape.as_list() == \
                       [batch_size, TEST_HIDDEN_SIZE]
            else:
                assert isinstance(encoder_out['encoder_output_state'],
                                  tf.Tensor)
                assert encoder_out['encoder_output_state'].shape.as_list() == \
                       [batch_size, TEST_HIDDEN_SIZE]

        elif enc_encoder == 'cnnrnn':
            assert encoder_out['encoder_output'].shape.as_list() == \
                   [batch_size, 1, TEST_HIDDEN_SIZE]
            assert 'encoder_output_state' in encoder_out

            if enc_cell_type == 'lstm':
                assert isinstance(encoder_out['encoder_output_state'], list)
                assert encoder_out['encoder_output_state'][0].shape.as_list() \
                       == [batch_size, TEST_HIDDEN_SIZE]
                assert encoder_out['encoder_output_state'][1].shape.as_list() \
                       == [batch_size, TEST_HIDDEN_SIZE]
            else:
                assert isinstance(encoder_out['encoder_output_state'],
                                  tf.Tensor)
                assert encoder_out['encoder_output_state'].shape.as_list() \
                       == [batch_size, TEST_HIDDEN_SIZE]

        elif enc_encoder == 'stacked_cnn':
            assert encoder_out['encoder_output'].shape.as_list() \
                   == [batch_size, 1, TEST_HIDDEN_SIZE]

        elif enc_encoder == 'transformer' or enc_encoder == 'embed':
            assert encoder_out['encoder_output'].shape.as_list() \
                   == [batch_size, seq_size, TEST_HIDDEN_SIZE]

        else:
            raise ValueError('{} is an invalid encoder specification' \
                             .format(enc_encoder))


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

    model, _ = setup_model_scaffolding(
        raw_df,
        input_features,
        output_features
    )

    # generate synthetic encoder_output tensors and make it look like
    # it came out of the combiner
    encoder_output = tf.random.normal(combiner_output_shapes[0])
    combiner_outputs = {'hidden': encoder_output}

    if combiner_output_shapes[1] is not None:
        if len(combiner_output_shapes[1]) > 1:
            encoder_output_state = [
                tf.random.normal(combiner_output_shapes[1][0]),
                tf.random.normal(combiner_output_shapes[1][1])
            ]
        else:
            encoder_output_state = \
                tf.random.normal(combiner_output_shapes[1][0])

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
        assert isinstance(logits, tf.Tensor)
        assert logits.shape.as_list() == [batch_size, seq_size, num_classes]

    assert isinstance(lengths, tf.Tensor)
    assert lengths.shape.as_list() == [batch_size]

    assert isinstance(preds, tf.Tensor)
    assert preds.shape.as_list() == [batch_size, seq_size]

    assert isinstance(last_preds, tf.Tensor)
    assert last_preds.shape.as_list() == [batch_size]

    assert isinstance(probs, tf.Tensor)
    assert probs.shape.as_list() == [batch_size, seq_size, num_classes]


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
