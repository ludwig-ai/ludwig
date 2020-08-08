import logging

import numpy as np

import pytest

import tensorflow as tf

from ludwig.combiners.combiners import ConcatCombiner, SequenceConcatCombiner


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

    encoder_outputs = {}
    shapes_list = [
        [BATCH_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, OTHER_HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]
    ]
    feature_names = ['feature_'+str(i+1) for i in range(len(shapes_list))]

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


@pytest.mark.parametrize(
    'fc_layer',
    [None, [{'fc_size': 64}, {'fc_size':64}]]
)
def test_concat_combiner(encoder_outputs, fc_layer):

    # clean out unneeded encoder outputs
    del encoder_outputs['feature_3']
    del encoder_outputs['feature_4']

    # setup combiner to test
    combiner = ConcatCombiner(
        fc_layers=fc_layer
    )

    results = combiner(encoder_outputs)

    # required key present
    assert 'combiner_output' in results

    # confirm correct output shapes
    if fc_layer:
        assert results['combiner_output'].shape.as_list() == [BATCH_SIZE, FC_SIZE]
    else:
        hidden_size = 0
        for k in encoder_outputs:
            hidden_size += encoder_outputs[k]['encoder_output'].shape[1]
        assert results['combiner_output'].shape.as_list() == [BATCH_SIZE, hidden_size]

@pytest.mark.parametrize('reduce_output', [None, 'sum'])
@pytest.mark.parametrize('main_sequence_feature', [None, 'feature_3'])
def test_sequence_concat_combiner(encoder_outputs, main_sequence_feature,
                                  reduce_output):

    combiner = SequenceConcatCombiner(
        main_sequence_feature=main_sequence_feature,
        reduce_output=reduce_output
    )

    hidden_size = 0
    for k in encoder_outputs:
        hidden_size += encoder_outputs[k]['encoder_output'].shape[-1]

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
