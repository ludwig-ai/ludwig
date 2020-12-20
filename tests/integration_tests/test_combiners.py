import logging

import pytest
import tensorflow as tf

from ludwig.combiners.combiners import \
    ConcatCombiner, SequenceConcatCombiner, SequenceCombiner, \
    sequence_encoder_registry, TabNetCombiner

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

BATCH_SIZE = 16
SEQ_SIZE = 12
HIDDEN_SIZE = 128
OTHER_HIDDEN_SIZE = 32
FC_SIZE = 64


# set up simulated encoder outputs
@pytest.fixture
def encoder_outputs():
    # generates simulated encoder outputs dictionary:
    #   feature_1: shape [b, h1] tensor
    #   feature_2: shape [b, h2] tensor
    #   feature_3: shape [b, s, h1] tensor
    #   feature_4: shape [b, sh, h2] tensor

    encoder_outputs = {}
    shapes_list = [
        [BATCH_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, OTHER_HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]
    ]
    feature_names = ['feature_' + str(i + 1) for i in range(len(shapes_list))]

    for feature_name, batch_shape in zip(feature_names, shapes_list):
        encoder_outputs[feature_name] = \
            {
                'encoder_output': tf.random.normal(
                    batch_shape,
                    dtype=tf.float32
                )
            }
        if len(batch_shape) > 2:
            encoder_outputs[feature_name]['encoder_output_state'] = \
                tf.random.normal(
                    [batch_shape[0], batch_shape[2]],
                    dtype=tf.float32
                )

    return encoder_outputs


# test for simple concatenation combiner
@pytest.mark.parametrize(
    'fc_layer',
    [None, [{'fc_size': 64}, {'fc_size': 64}]]
)
def test_concat_combiner(encoder_outputs, fc_layer):
    # clean out unneeded encoder outputs
    del encoder_outputs['feature_3']
    del encoder_outputs['feature_4']

    # setup combiner to test
    combiner = ConcatCombiner(
        fc_layers=fc_layer
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results

    # confirm correct output shapes
    if fc_layer:
        assert results['combiner_output'].shape.as_list() == [BATCH_SIZE,
                                                              FC_SIZE]
    else:
        # calculate expected hidden size for concatenated tensors
        hidden_size = 0
        for k in encoder_outputs:
            hidden_size += encoder_outputs[k]['encoder_output'].shape[1]

        assert results['combiner_output'].shape.as_list() == [BATCH_SIZE,
                                                              hidden_size]


# test for sequence concatenation combiner
@pytest.mark.parametrize('reduce_output', [None, 'sum'])
@pytest.mark.parametrize('main_sequence_feature', [None, 'feature_3'])
def test_sequence_concat_combiner(encoder_outputs, main_sequence_feature,
                                  reduce_output):
    combiner = SequenceConcatCombiner(
        main_sequence_feature=main_sequence_feature,
        reduce_output=reduce_output
    )

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs:
        hidden_size += encoder_outputs[k]['encoder_output'].shape[-1]

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results

    # confirm correct shape
    if reduce_output is None:
        assert results['combiner_output'].shape.as_list() == \
               [BATCH_SIZE, SEQ_SIZE, hidden_size]
    else:
        assert results['combiner_output'].shape.as_list() == \
               [BATCH_SIZE, hidden_size]


# test for sequence combiner
@pytest.mark.parametrize('reduce_output', [None, 'sum'])
@pytest.mark.parametrize('encoder', sequence_encoder_registry)
@pytest.mark.parametrize('main_sequence_feature', [None, 'feature_3'])
def test_sequence_combiner(encoder_outputs, main_sequence_feature,
                           encoder, reduce_output):
    combiner = SequenceCombiner(
        main_sequence_feature=main_sequence_feature,
        encoder=encoder,
        reduce_output=reduce_output
    )

    # calculate expected hidden size for concatenated tensors
    hidden_size = 0
    for k in encoder_outputs:
        hidden_size += encoder_outputs[k]['encoder_output'].shape[-1]

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results

    combiner_shape = results['combiner_output'].shape
    # test for correct dimension
    if reduce_output:
        assert len(combiner_shape) == 2
    else:
        assert len(combiner_shape) == 3

    # Shape test assumes on Ludwig sequence encoder defaults
    #   parallel encoders: # layers = 4, fc_size=256
    #   non-parallel encoders: fc_size=256
    # if defaults change, then this test has to be updated
    default_layer = 4
    default_fc_size = 256

    if 'parallel' in encoder:
        combiner_shape[-1] == default_layer * default_fc_size
    else:
        combiner_shape[-1] == default_fc_size


def test_tabnet_combiner(encoder_outputs):
    # clean out unneeded encoder outputs
    encoder_outputs = {}
    encoder_outputs['feature_1'] = {
        'encoder_output': tf.random.normal(
            [128, 1],
            dtype=tf.float32
        )
    }
    encoder_outputs['feature_2'] = {
        'encoder_output': tf.random.normal(
            [128, 1],
            dtype=tf.float32
        )
    }

    # setup combiner to test
    combiner = TabNetCombiner(
        [encoder_outputs['feature_1'], encoder_outputs['feature_2']],
        size=2,
        output_size=2,
        num_steps=3,
        num_total_blocks=4,
        num_shared_blocks=2,
        dropout=0.1
    )

    # concatenate encoder outputs
    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results
    assert 'attention_masks' in results
