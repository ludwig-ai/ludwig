from collections import namedtuple
import os
import shutil

import pandas as pd
import numpy as np
import pytest

import tensorflow as tf

from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.encoders.generic_encoders import DenseEncoder
from ludwig.encoders.sequence_encoders import ParallelCNN
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_utils import SEQUENCE_TYPES
from ludwig.models.ecd import build_single_input
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import image_feature


BATCH_SIZE = 128
RANDOM_SEED = 42
IMAGE_DIR = '/tmp/images'

# SyntheticData namedtuple structure:
# batch_size: Number of records to generate for a batch
# feature_generator: Ludwwig synthentic generator class
# feature_generator_args: tuple of required positional arguments
# feature_generator_kwargs: dictionary of required keyword arguments
SyntheticData = namedtuple(
    'SyntheticData',
    'batch_size feature_generator feature_generator_args feature_generator_kwargs'
)

# TestCase namedtuple structure:
# inputs: SyntheticData namedtuple of data to create
# XCoder_other_parms: dictionary for required encoder/decoder parameters
# regularizer_parm_names: list of regularizer keyword parameter names
TestCase = namedtuple('TestCase', 'inputs XCoder_other_parms regularizer_parm_names')


#
# Regularization Encoder Tests
#
@pytest.mark.parametrize(
    'test_case',
    [
        # # DenseEncoder
        TestCase(
            SyntheticData(BATCH_SIZE, numerical_feature,(), {}),
            {'num_layers': 2, 'encoder': 'dense', 'preprocessing':{'normalization': 'zscore'}},
            ['activity_regularizer', 'weights_regularizer', 'bias_regularizer']
        ),

        # Image Encoders
        TestCase(
            SyntheticData(BATCH_SIZE, image_feature, (IMAGE_DIR,), {}),
            {'encoder': 'stacked_cnn'},
            [
                'conv_activity_regularizer', 'conv_weights_regularizer',
                'conv_bias_regularizer',
                'fc_activity_regularizer', 'fc_weights_regularizer',
                'fc_bias_regularizer',
            ]
        ),
        TestCase(
            SyntheticData(BATCH_SIZE, image_feature, (IMAGE_DIR,), {}),
            {'encoder': 'resnet'},
            [
                'activity_regularizer', 'weights_regularizer',
                'bias_regularizer',
            ]
        ),

        # ParallelCNN Encoder
        TestCase(
            SyntheticData(BATCH_SIZE, sequence_feature, (), {}),
            {'encoder': 'parallel_cnn', 'cell_type': 'gru'},
            ['activity_regularizer', 'weights_regularizer', 'bias_regularizer']
        ),

    ]

)
def test_encoder(test_case):
    # set up required directories for images if needed
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)
    os.mkdir(IMAGE_DIR)


    # reproducible synthetic data set
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # create synthetic data for the test
    features = [
        test_case.inputs.feature_generator(
            *test_case.inputs.feature_generator_args,
            **test_case.inputs.feature_generator_kwargs
        )
    ]
    feature_name = features[0]['name']
    data_generator = build_synthetic_dataset(BATCH_SIZE, features)
    data_list = list(data_generator)
    raw_data = [x[0] for x in data_list[1:]]
    df = pd.DataFrame({data_list[0][0]: raw_data})

    # minimal model definition sufficient to create the input feature
    model_definition = {'input_features': features, 'output_features': []}
    training_set, _, _, train_set_metadata = preprocess_for_training(
        model_definition,
        data_train_df=df,
        skip_save_processed_input=True,
        random_seed=RANDOM_SEED
    )

    # run through each type of regularizer for the encoder
    regularizer_losses = []
    for regularizer in [None, 'l1', 'l2', 'l1_l2']:
        # start with clean slate and make reproducible
        tf.keras.backend.clear_session()
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        # setup kwarg for regularizer parms
        x_coder_kwargs = dict(
            zip(test_case.regularizer_parm_names,
                len(test_case.regularizer_parm_names)*[regularizer])
        )

        # combine other other keyword parameters
        x_coder_kwargs.update(test_case.XCoder_other_parms)

        features[0].update(x_coder_kwargs)
        if features[0]['type'] in SEQUENCE_TYPES:
            features[0]['vocab'] = train_set_metadata[feature_name]['idx2str']
            training_set.dataset[feature_name] = \
                training_set.dataset[feature_name].astype(np.int32)

        input_def_obj = build_single_input(features[0], None)

        inputs = training_set.dataset[feature_name]
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)

        input_def_obj.encoder_obj(inputs)
        regularizer_loss = tf.reduce_sum(input_def_obj.encoder_obj.losses)
        regularizer_losses.append(regularizer_loss)

    # check loss regularization loss values
    # None should be zero
    assert regularizer_losses[0] == 0

    # l1, l2 and l1_l2 should be greater than zero
    assert np.all([t > 0.0 for t in regularizer_losses[1:]])

    # # using default setting l1 + l2 == l1_l2 losses
    assert np.isclose(regularizer_losses[1].numpy() + regularizer_losses[2].numpy(),
                      regularizer_losses[3].numpy())

    # cleanup
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)









