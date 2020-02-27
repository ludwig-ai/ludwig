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

import numpy as np
import pandas as pd
import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.constants import FULL
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


def test_savedmodel(csv_filename):
    #######
    # Setup
    #######
    dir_path = os.path.dirname(csv_filename)

    # Single sequence input, single category output
    sf = sequence_feature()
    sf['encoder'] = 'parallel_cnn'
    input_features = [sf]
    input_feature_name = input_features[0]['name']
    input_feature_tensor_name = '{}/{}_placeholder:0'.format(
        input_feature_name,
        input_feature_name
    )
    output_features = [category_feature(vocab_size=2)]
    output_feature_name = output_features[0]['name']
    output_feature_tensor_name = '{}/predictions_{}/predictions_{}:0'.format(
        output_feature_name,
        output_feature_name,
        output_feature_name
    )
    predictions_column_name = '{}_predictions'.format(output_feature_name)
    weight_tensor_name = '{}/fc_0/weights:0'.format(input_feature_name)

    # Generate test data
    data_csv_path = generate_data(input_features, output_features, csv_filename)

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
    original_predictions_df = ludwig_model.predict(data_csv=data_csv_path)

    ###################
    # save Ludwig model
    ###################
    ludwigmodel_path = os.path.join(dir_path, 'ludwigmodel')
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    ludwig_model.save(ludwigmodel_path)

    #################
    # save savedmodel
    #################
    savedmodel_path = os.path.join(dir_path, 'savedmodel')
    shutil.rmtree(savedmodel_path, ignore_errors=True)
    ludwig_model.model.save_savedmodel(savedmodel_path)

    ##############################
    # collect weight tensors names
    ##############################
    with ludwig_model.model.session as sess:
        all_variables = tf.compat.v1.trainable_variables()
        all_variables_names = [v.name for v in all_variables]
    ludwig_model.close()

    ###################################################
    # load Ludwig model, obtain predictions and weights
    ###################################################
    ludwig_model = LudwigModel.load(ludwigmodel_path)
    ludwig_prediction_df = ludwig_model.predict(data_csv=data_csv_path)
    ludwig_weights = ludwig_model.model.collect_weights(all_variables_names)
    ludwig_model.close()

    #################################################
    # load savedmodel, obtain predictions and weights
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

    with tf.compat.v1.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.SERVING],
            savedmodel_path
        )

        predictions = sess.run(
            output_feature_tensor_name,
            feed_dict={
                input_feature_tensor_name: dataset.get(input_feature_name),
            }
        )

        savedmodel_prediction_df = pd.DataFrame(
            data=[train_set_metadata[output_feature_name]["idx2str"][p] for p in
                  predictions], columns=[predictions_column_name])

        savedmodel_weights = sess.run({n: n for n in all_variables_names})

    #########
    # Cleanup
    #########
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    shutil.rmtree(savedmodel_path, ignore_errors=True)

    ###############################################
    # Check if weights and predictions are the same
    ###############################################

    for var in all_variables_names:
        print(
            "Are the weights in {} identical?".format(var),
            np.all(ludwig_weights[var] == savedmodel_weights[var])
        )
    print(
        "Are loaded model predictions identical to original ones?",
        np.all(
            original_predictions_df[predictions_column_name] == \
            ludwig_prediction_df[predictions_column_name]
        )
    )
    print(
        "Are savedmodel predictions identical to loaded model?",
        np.all(
            ludwig_prediction_df[predictions_column_name] == \
            savedmodel_prediction_df[predictions_column_name]
        )
    )

    for var in all_variables_names:
        assert np.all(ludwig_weights[var] == savedmodel_weights[var])
    assert np.all(
        original_predictions_df[predictions_column_name] == \
        ludwig_prediction_df[predictions_column_name]
    )
    assert np.all(
        ludwig_prediction_df[predictions_column_name] == \
        savedmodel_prediction_df[predictions_column_name]
    )
