import os
import shutil
import tempfile
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.constants import PROC_COLUMN, NAME
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.data.preprocessing import preprocess_for_training
from ludwig.features.feature_utils import SEQUENCE_TYPES, compute_feature_hash
from ludwig.models.ecd import build_single_input, build_single_output
from tests.integration_tests.utils import category_feature, LocalTestBackend
from tests.integration_tests.utils import date_feature
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature

BATCH_SIZE = 32
HIDDEN_SIZE = 128
SEQ_SIZE = 10
RANDOM_SEED = 42
IMAGE_DIR = tempfile.mkdtemp()

# SyntheticData namedtuple structure:
# batch_size: Number of records to generate for a batch
# feature_generator: Ludwig synthetic generator class
# feature_generator_args: tuple of required positional arguments
# feature_generator_kwargs: dictionary of required keyword arguments
SyntheticData = namedtuple(
    'SyntheticData',
    'batch_size feature_generator feature_generator_args feature_generator_kwargs'
)

# TestCase namedtuple structure:
# syn_data: SyntheticData namedtuple of data to create
# XCoder_other_parms: dictionary for required encoder/decoder parameters
# regularizer_parm_names: list of regularizer keyword parameter names
TestCase = namedtuple('TestCase',
                      'syn_data XCoder_other_parms regularizer_parm_names')


#
# Regularization Encoder Tests, commenting set of parametrized tests out for now and keeping one
#
@pytest.mark.parametrize(
    'test_case',
    [
        # DenseEncoder
        TestCase(
            SyntheticData(BATCH_SIZE, numerical_feature, (), {}),
            {'num_layers': 2, 'encoder': 'dense',
             'preprocessing': {'normalization': 'zscore'}},
            ['activity_regularizer', 'weights_regularizer', 'bias_regularizer']
        )
    ]
)
def test_encoder(test_case):
    # set up required directories for images if needed
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)
    os.mkdir(IMAGE_DIR)

    # reproducible synthetic data set
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # create synthetic data for the test
    features = [
        test_case.syn_data.feature_generator(
            *test_case.syn_data.feature_generator_args,
            **test_case.syn_data.feature_generator_kwargs
        )
    ]
    name = features[0][NAME]
    proc_column = compute_feature_hash(features[0])
    features[0][PROC_COLUMN] = proc_column

    data_generator = build_synthetic_dataset(BATCH_SIZE, features)
    data_list = list(data_generator)
    raw_data = [x[0] for x in data_list[1:]]
    df = pd.DataFrame({data_list[0][0]: raw_data})

    # minimal config sufficient to create the input feature
    config = {'input_features': features, 'output_features': []}
    training_set, _, _, training_set_metadata = preprocess_for_training(
        config,
        training_set=df,
        skip_save_processed_input=True,
        random_seed=RANDOM_SEED,
        backend=LocalTestBackend(),
    )

    # run through each type of regularizer for the encoder
    regularizer_losses = []
    for regularizer in [None, 'l1', 'l2', 'l1_l2']:
        # start with clean slate and make reproducible
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        # setup kwarg for regularizer parms
        x_coder_kwargs = dict(
            zip(test_case.regularizer_parm_names,
                len(test_case.regularizer_parm_names) * [regularizer])
        )

        # combine other other keyword parameters
        x_coder_kwargs.update(test_case.XCoder_other_parms)
        features[0].update(x_coder_kwargs)

        # shim code to support sequence/sequence like features
        if features[0]['type'] in SEQUENCE_TYPES.union({'category', 'set'}):
            features[0]['vocab'] = training_set_metadata[name][
                'idx2str']
            training_set.dataset[proc_column] = \
                training_set.dataset[proc_column].astype(np.int32)

        input_def_obj = build_single_input(features[0], None)

        inputs = training_set.dataset[proc_column]
        # make sure we are at least rank 2 tensor
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(-1, 1)

        # special handling for image feature
        if features[0]['type'] == 'image':
            inputs = torch.tensor(inputs.type(torch.float32)/255)

        input_def_obj.encoder_obj(torch.tensor(inputs))
        regularizer_loss = torch.sum(input_def_obj.encoder_obj.losses)
        regularizer_losses.append(regularizer_loss)

    # check loss regularization loss values
    # None should be zero
    assert regularizer_losses[0] == 0

    # l1, l2 and l1_l2 should be greater than zero
    assert np.all([t > 0.0 for t in regularizer_losses[1:]])

    # # using default setting l1 + l2 == l1_l2 losses
    assert np.isclose(
        regularizer_losses[1].numpy() + regularizer_losses[2].numpy(),
        regularizer_losses[3].numpy())

    # cleanup
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)
