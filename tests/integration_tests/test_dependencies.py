import logging
import os
import shutil

import pytest
import yaml

import tensorflow as tf

from ludwig.features.numerical_feature import NumericalOutputFeature
from tests.integration_tests.utils import numerical_feature


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

BATCH_SIZE = 16
SEQ_SIZE = 12
HIDDEN_SIZE = 128
OTHER_HIDDEN_SIZE = 32
OTHER_HIDDEN_SIZE2 = 18

# unit test for single dependency
@pytest.mark.parametrize(
    'dependent_hidden_shape', [
        [BATCH_SIZE, OTHER_HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]

    ]
)
@pytest.mark.parametrize(
    'hidden_shape', [
        [BATCH_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE]
    ]
)
def test_single_dependencies(hidden_shape, dependent_hidden_shape):
    hidden_layer = tf.random.normal(
        hidden_shape,
        dtype=tf.float32
    )
    other_hidden_layer = tf.random.normal(
        dependent_hidden_shape,
        dtype=tf.float32
    )

    other_dependencies = {'feature_name': other_hidden_layer}

    num_feature_defn = numerical_feature()
    num_feature_defn['loss'] = {'type': 'mean_squared_error'}
    num_feature_defn['dependencies'] = ['feature_name']
    if len(dependent_hidden_shape) > 2:
        num_feature_defn['reduce_dependencies'] = 'sum'

    out_feature = NumericalOutputFeature(num_feature_defn)

    results = out_feature.concat_dependencies(
        hidden_layer,
        other_dependencies
    )

    if len(hidden_shape) > 2:
        assert results.shape.as_list() == \
               [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE + OTHER_HIDDEN_SIZE]
    else:
        assert results.shape.as_list() == \
               [BATCH_SIZE, HIDDEN_SIZE + OTHER_HIDDEN_SIZE]


# unit test for multiple dependencies
@pytest.mark.parametrize(
    'dependent_hidden_shape2', [
        [BATCH_SIZE, OTHER_HIDDEN_SIZE2],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE2]

    ]
)
@pytest.mark.parametrize(
    'dependent_hidden_shape', [
        [BATCH_SIZE, OTHER_HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, OTHER_HIDDEN_SIZE]

    ]
)
@pytest.mark.parametrize(
    'hidden_shape', [
        [BATCH_SIZE, HIDDEN_SIZE],
        [BATCH_SIZE, SEQ_SIZE, HIDDEN_SIZE]
    ]
)
def test_multiple_dependencies(hidden_shape, dependent_hidden_shape,
                               dependent_hidden_shape2):
    hidden_layer = tf.random.normal(
        hidden_shape,
        dtype=tf.float32
    )
    other_hidden_layer = tf.random.normal(
        dependent_hidden_shape,
        dtype=tf.float32
    )
    other_hidden_layer2 = tf.random.normal(
        dependent_hidden_shape2,
        dtype=tf.float32
    )

    other_dependencies = {
        'feature_name': other_hidden_layer,
        'feature_name2': other_hidden_layer2
    }

    num_feature_defn = numerical_feature()
    num_feature_defn['loss'] = {'type': 'mean_squared_error'}
    num_feature_defn['dependencies'] = ['feature_name', 'feature_name2']
    if len(dependent_hidden_shape) > 2 or len(dependent_hidden_shape2) > 2:
        num_feature_defn['reduce_dependencies'] = 'sum'

    out_feature = NumericalOutputFeature(num_feature_defn)

    results = out_feature.concat_dependencies(
        hidden_layer,
        other_dependencies
    )

    if len(hidden_shape) > 2:
        assert results.shape.as_list() == \
               [BATCH_SIZE, SEQ_SIZE,
                HIDDEN_SIZE + OTHER_HIDDEN_SIZE + OTHER_HIDDEN_SIZE2]
    else:
        assert results.shape.as_list() == \
               [BATCH_SIZE,
                HIDDEN_SIZE + OTHER_HIDDEN_SIZE + OTHER_HIDDEN_SIZE2]



