import pytest
import torch

from ludwig.utils.misc_utils import get_from_registry
from tests.integration_tests.utils import ENCODERS
from ludwig.encoders.sequence_encoders import \
    ENCODER_REGISTRY as SEQUENCE_ENCODER_REGISTRY

TEST_VOCAB_SIZE = 132
TEST_HIDDEN_SIZE = 32
TEST_STATE_SIZE = 16
TEST_EMBEDDING_SIZE = 64
TEST_NUM_FILTERS = 24
BATCH_SIZE = 128
SEQ_SIZE = 10
PARALLEL_CNN_LAYERS = 4

# todo(jmt): fine tune this parameter list for the test
encoder_parameters = {
    # "should_embed": True,
    "vocab": [str(i) for i in range(TEST_VOCAB_SIZE)],
    # "representation": 'dense',
    "embedding_size": TEST_EMBEDDING_SIZE,
    "hidden_size": TEST_HIDDEN_SIZE,
    "num_filters": TEST_NUM_FILTERS,
    # "embeddings_trainable": True,
    # "pretrained_embeddings": None,
    # "embeddings_on_cpu": False,
    "num_layers": 1,
    "max_sequence_length": SEQ_SIZE,
    "state_size": TEST_STATE_SIZE,
    "cell_type": 'rnn',
    # "bidirectional": False,
    # "activation": 'tanh',
    # "recurrent_activation": 'sigmoid',
    # "unit_forget_bias": True,
    # "recurrent_initializer": 'orthogonal',
    # "recurrent_regularizer": None,
    # "# recurrent_constraint": None,
    "dropout": 0.0,
    # "recurrent_dropout": 0.0,
    # "fc_layers": None,
    "num_fc_layers": 0,  # todo: need to assess impact of this parameter
    # "fc_size": 256,
    # "use_bias": True,
    # "weights_initializer": 'xavier_uniform',
    # "bias_initializer": 'zeros',
    # "weights_regularizer": None,
    # "bias_regularizer": None,
    # "activity_regularizer": None,
    # "weights_constraint": None,
    # "bias_constraint": None,
    "norm": None,
    # "norm_params": None,
    # "fc_activation": 'relu',
    # "fc_dropout": 0,
    "reduce_output": None,
}

@pytest.mark.parametrize('enc_reduce_output', [None, 'sum'])
@pytest.mark.parametrize('enc_norm', [None, 'batch', 'layer'])
@pytest.mark.parametrize('enc_num_layers', [1, 2])
@pytest.mark.parametrize('enc_dropout', [0, 0.2])
@pytest.mark.parametrize('enc_cell_type', ['rnn', 'gru', 'lstm'])
@pytest.mark.parametrize('enc_encoder', ENCODERS)
def test_sequence_encoders(
        enc_encoder,
        enc_cell_type,
        enc_dropout,
        enc_num_layers,
        enc_norm,
        enc_reduce_output
):
    # update encoder parameters for specific unit test case
    encoder_parameters['cell_type'] = enc_cell_type
    encoder_parameters['dropout'] = enc_dropout
    encoder_parameters['num_layers'] = enc_num_layers
    encoder_parameters['norm'] = enc_norm
    encoder_parameters['reduce_output'] = enc_reduce_output

    # retrieve encoder to test
    encoder_obj = get_from_registry(enc_encoder, SEQUENCE_ENCODER_REGISTRY)(
        **encoder_parameters
    )

    INPUT_SEQUENCE = torch.zeros([BATCH_SIZE, SEQ_SIZE], dtype=torch.int32)

    encoder_out = encoder_obj(INPUT_SEQUENCE)

    assert 'encoder_output' in encoder_out
    assert isinstance(encoder_out['encoder_output'], torch.Tensor)

    if enc_encoder == 'parallel_cnn':
        number_parallel_cnn_layers = PARALLEL_CNN_LAYERS
        output_dimension = encoder_parameters['num_filters'] \
                           * number_parallel_cnn_layers
        assert encoder_out['encoder_output'].shape == \
               (BATCH_SIZE, SEQ_SIZE, output_dimension) \
            if enc_reduce_output is None else (BATCH_SIZE, output_dimension)

    elif enc_encoder == 'stacked_parallel_cnn':
        number_parallel_cnn_layers = PARALLEL_CNN_LAYERS
        output_dimension = encoder_parameters['num_filters'] \
                           * number_parallel_cnn_layers
        assert encoder_out['encoder_output'].shape == \
               (BATCH_SIZE, SEQ_SIZE, output_dimension) \
            if enc_reduce_output is None else (BATCH_SIZE, output_dimension)

    elif enc_encoder == 'rnn':
        assert encoder_out['encoder_output'].shape == \
               (BATCH_SIZE, SEQ_SIZE, TEST_STATE_SIZE) \
            if enc_reduce_output is None else (BATCH_SIZE, TEST_STATE_SIZE)

        assert 'encoder_output_state' in encoder_out
        if enc_cell_type == 'lstm':
            assert isinstance(encoder_out['encoder_output_state'], tuple)
            assert isinstance(encoder_out['encoder_output_state'][0],
                              torch.Tensor)
            assert isinstance(encoder_out['encoder_output_state'][1],
                              torch.Tensor)
            assert encoder_out['encoder_output_state'][0].shape == \
                   (BATCH_SIZE, TEST_STATE_SIZE)
            assert encoder_out['encoder_output_state'][1].shape == \
                   (BATCH_SIZE, TEST_STATE_SIZE)
        else:
            assert isinstance(encoder_out['encoder_output_state'],
                              torch.Tensor)
            assert encoder_out['encoder_output_state'].shape == \
                   (BATCH_SIZE, TEST_STATE_SIZE)

    elif enc_encoder == 'cnnrnn':
        assert encoder_out['encoder_output'].shape == \
               (BATCH_SIZE, 1, TEST_STATE_SIZE) \
            if enc_reduce_output is None else (BATCH_SIZE, TEST_STATE_SIZE)

        assert 'encoder_output_state' in encoder_out

        if enc_cell_type == 'lstm':
            assert isinstance(encoder_out['encoder_output_state'],
                              tuple)
            assert encoder_out['encoder_output_state'][0].shape \
                   == (BATCH_SIZE, TEST_STATE_SIZE)
            assert encoder_out['encoder_output_state'][1].shape \
                   == (BATCH_SIZE, TEST_STATE_SIZE)
        else:
            assert isinstance(encoder_out['encoder_output_state'],
                              torch.Tensor)
            assert encoder_out['encoder_output_state'].shape \
                   == (BATCH_SIZE, TEST_STATE_SIZE)

    elif enc_encoder == 'stacked_cnn':
        assert encoder_out['encoder_output'].shape \
               == (BATCH_SIZE, 1, TEST_NUM_FILTERS) \
            if enc_reduce_output is None else (BATCH_SIZE, TEST_NUM_FILTERS)

    elif enc_encoder == 'embed':
        assert encoder_out['encoder_output'].shape \
               == (BATCH_SIZE, SEQ_SIZE, TEST_EMBEDDING_SIZE) \
            if enc_reduce_output is None else (BATCH_SIZE, TEST_EMBEDDING_SIZE)

    elif enc_encoder == 'transformer':
        assert encoder_out['encoder_output'].shape \
               == (BATCH_SIZE, SEQ_SIZE, TEST_HIDDEN_SIZE) \
            if enc_reduce_output is None else (BATCH_SIZE, TEST_HIDDEN_SIZE)

    else:
        raise ValueError('{} is an invalid encoder specification'
                         .format(enc_encoder))
