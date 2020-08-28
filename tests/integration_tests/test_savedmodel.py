# -*- coding: utf-8 -*-
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import shutil
from copy import deepcopy

import numpy as np
import pytest
import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.constants import FULL
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


@pytest.mark.parametrize('should_load_model', [True, False])
def test_savedmodel(csv_filename, should_load_model):
    #######
    # Setup
    #######
    dir_path = os.path.dirname(csv_filename)

    # Single sequence input, single category output
    sf = sequence_feature()
    sf['encoder'] = 'parallel_cnn'
    input_features = [sf]

    output_features = [category_feature(vocab_size=2)]

    predictions_column_name = '{}_predictions'.format(
        output_features[0]['name'])

    # Generate test data
    data_csv_path = generate_data(input_features, output_features,
                                  csv_filename)

    #############
    # Train model
    #############
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {'epochs': 2}
    }
    ludwig_model = LudwigModel(model_definition)
    ludwig_model.train(
        data_csv=data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    ###################
    # save Ludwig model
    ###################
    ludwigmodel_path = os.path.join(dir_path, 'ludwigmodel')
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    ludwig_model.save(ludwigmodel_path)

    ###################
    # load Ludwig model
    ###################
    if should_load_model:
        ludwig_model = LudwigModel.load(ludwigmodel_path)

    #################
    # save savedmodel
    #################
    savedmodel_path = os.path.join(dir_path, 'savedmodel')
    shutil.rmtree(savedmodel_path, ignore_errors=True)
    ludwig_model.model.save_savedmodel(savedmodel_path)

    ##############################
    # collect weight tensors names
    ##############################
    original_predictions_df = ludwig_model.predict(data_csv=data_csv_path)
    original_weights = deepcopy(ludwig_model.model.model.trainable_variables)
    ludwig_model.close()

    ###################################################
    # load Ludwig model, obtain predictions and weights
    ###################################################
    ludwig_model = LudwigModel.load(ludwigmodel_path)
    loaded_prediction_df = ludwig_model.predict(data_csv=data_csv_path)
    loaded_weights = deepcopy(ludwig_model.model.model.trainable_variables)

    #################################################
    # restore savedmodel, obtain predictions and weights
    #################################################
    train_set_metadata_json_fp = os.path.join(
        ludwigmodel_path,
        TRAIN_SET_METADATA_FILE_NAME
    )

    dataset, train_set_metadata = preprocess_for_prediction(
        ludwigmodel_path,
        split=FULL,
        data_csv=data_csv_path,
        train_set_metadata=train_set_metadata_json_fp,
        evaluate_performance=False
    )

    restored_model = tf.saved_model.load(savedmodel_path)

    if_name = list(ludwig_model.model.model.input_features.keys())[0]
    of_name = list(ludwig_model.model.model.output_features.keys())[0]

    data_to_predict = {
        if_name: tf.convert_to_tensor(dataset.dataset[if_name], dtype=tf.int32)
    }

    logits = restored_model(data_to_predict, False, None)

    restored_predictions = tf.argmax(
        logits[of_name]['logits'],
        -1,
        name='predictions_{}'.format(of_name)
    )
    restored_predictions = tf.map_fn(
        lambda idx: train_set_metadata[of_name]['idx2str'][idx],
        restored_predictions,
        dtype=tf.string
    )

    restored_weights = deepcopy(restored_model.trainable_variables)

    #########
    # Cleanup
    #########
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    shutil.rmtree(savedmodel_path, ignore_errors=True)

    ###############################################
    # Check if weights and predictions are the same
    ###############################################

    # check for same number of weights as original model
    assert len(original_weights) == len(loaded_weights)
    assert len(original_weights) == len(restored_weights)

    # check to ensure weight valuess match the original model
    loaded_weights_match = np.all(
        [np.all(np.isclose(original_weights[i].numpy(),
                           loaded_weights[i].numpy())) for i in
         range(len(original_weights))]
    )
    restored_weights_match = np.all(
        [np.all(np.isclose(original_weights[i].numpy(),
                           restored_weights[i].numpy())) for i in
         range(len(original_weights))]
    )

    assert loaded_weights_match and restored_weights_match

    #  Are predictions identical to original ones?
    loaded_predictions_match = np.all(
        original_predictions_df[predictions_column_name] ==
        loaded_prediction_df[predictions_column_name]
    )

    restored_predictions_match = np.all(
        original_predictions_df[predictions_column_name] ==
        restored_predictions.numpy().astype('str')
    )

    assert loaded_predictions_match and restored_predictions_match
