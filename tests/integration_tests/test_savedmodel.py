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
import tempfile
from copy import deepcopy

import numpy as np
import pytest
import tensorflow as tf

from ludwig.api import LudwigModel
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from tests.integration_tests.utils import category_feature, binary_feature, \
    numerical_feature, text_feature, vector_feature, image_feature, \
    audio_feature, timeseries_feature, date_feature, h3_feature, set_feature, \
    bag_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


@pytest.mark.parametrize('should_load_model', [True, False])
def test_savedmodel(csv_filename, should_load_model):
    #######
    # Setup
    #######
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = tmpdir
        data_csv_path = os.path.join(tmpdir, csv_filename)
        image_dest_folder = os.path.join(tmpdir, 'generated_images')
        audio_dest_folder = os.path.join(tmpdir, 'generated_audio')

        # Single sequence input, single category output
        input_features = [
            binary_feature(),
            numerical_feature(),
            category_feature(vocab_size=3),
            sequence_feature(vocab_size=3),
            text_feature(vocab_size=3),
            vector_feature(),
            image_feature(image_dest_folder),
            audio_feature(audio_dest_folder),
            timeseries_feature(),
            date_feature(),
            h3_feature(),
            set_feature(vocab_size=3),
            bag_feature(vocab_size=3),
        ]

        output_features = [
            category_feature(vocab_size=3),
            binary_feature(),
            numerical_feature(),
            sequence_feature(vocab_size=3),
            text_feature(vocab_size=3),
            set_feature(vocab_size=3),
            vector_feature()
        ]

        predictions_column_name = '{}_predictions'.format(
            output_features[0]['name'])

        # Generate test data
        data_csv_path = generate_data(input_features, output_features,
                                      data_csv_path)

        #############
        # Train model
        #############
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'training': {'epochs': 2}
        }
        ludwig_model = LudwigModel(config)
        ludwig_model.train(
            dataset=data_csv_path,
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

        ##############################
        # collect weight tensors names
        ##############################
        original_predictions_df, _ = ludwig_model.predict(
            dataset=data_csv_path)
        original_weights = deepcopy(ludwig_model.model.trainable_variables)

        #################
        # save savedmodel
        #################
        savedmodel_path = os.path.join(dir_path, 'savedmodel')
        shutil.rmtree(savedmodel_path, ignore_errors=True)
        ludwig_model.model.save_savedmodel(savedmodel_path)

        ###################################################
        # load Ludwig model, obtain predictions and weights
        ###################################################
        ludwig_model = LudwigModel.load(ludwigmodel_path)
        loaded_prediction_df, _ = ludwig_model.predict(dataset=data_csv_path)
        loaded_weights = deepcopy(ludwig_model.model.trainable_variables)

        #################################################
        # restore savedmodel, obtain predictions and weights
        #################################################
        training_set_metadata_json_fp = os.path.join(
            ludwigmodel_path,
            TRAIN_SET_METADATA_FILE_NAME
        )

        dataset, training_set_metadata = preprocess_for_prediction(
            ludwig_model.config,
            dataset=data_csv_path,
            training_set_metadata=training_set_metadata_json_fp
        )

        restored_model = tf.saved_model.load(savedmodel_path)

        # Check the outputs for one of the features for correctness
        # Here we choose the first output feature (categorical)
        of_name = list(ludwig_model.model.output_features.keys())[0]

        data_to_predict = {
            name: tf.convert_to_tensor(
                dataset.dataset[feature.proc_column],
                dtype=feature.get_input_dtype()
            )
            for name, feature in ludwig_model.model.input_features.items()
        }

        logits = restored_model(data_to_predict, False, None)

        restored_predictions = tf.argmax(
            logits[of_name]['logits'],
            -1,
            name='predictions_{}'.format(of_name)
        )
        restored_predictions = tf.map_fn(
            lambda idx: training_set_metadata[of_name]['idx2str'][idx],
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

        original_weights = sorted(original_weights, key=lambda w: w.name)
        restored_weights = sorted(restored_weights, key=lambda w: w.name)

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
