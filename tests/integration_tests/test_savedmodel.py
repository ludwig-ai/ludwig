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
import collections
import os
import shutil
import tempfile
from copy import deepcopy

import numpy as np
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.constants import *
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils import forward_utils
from tests.integration_tests.utils import category_feature, binary_feature, \
    numerical_feature, text_feature, vector_feature, image_feature, \
    audio_feature, timeseries_feature, date_feature, h3_feature, set_feature, \
    bag_feature, LocalTestBackend
from tests.integration_tests import utils
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


# Currently fails, which is ok if the tracing method works.
def test_torchscript_script(csv_filename):
    # Features don't matter since we're scripting the entire module wholesale.
    input_features = [date_feature()]
    output_features = [category_feature(vocab_size=3)]
    backend = LocalTestBackend()
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {'epochs': 2}
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        ludwig_model = LudwigModel(config, backend=backend)
        data_csv_path = os.path.join(tmpdir, csv_filename)
        data_csv_path = generate_data(input_features, output_features,
                                      data_csv_path)
        # Necessary in order to instantiate ludwig_model.model as an ECD object.
        ludwig_model.train(
            dataset=data_csv_path,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
        )

        torchscript_path = os.path.join(tmpdir, 'torchscript')
        ludwig_model.model.save_torchscript_script(torchscript_path)


@pytest.mark.distributed
# @pytest.mark.parametrize('should_load_model', [True, False])
@pytest.mark.parametrize('should_load_model', [True])
def test_torchscript(csv_filename, should_load_model):
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
            # combiners: Dimension out of range (expected to be in range of [-1, 0], but got 1)
            # binary_feature(),

            # combiners: Dimension out of range (expected to be in range of [-1, 0], but got 1)
            # numerical_feature(),

            category_feature(vocab_size=3),
            sequence_feature(vocab_size=3),

            # reduction_modules: IndexError: amax(): Expected reduction dim 1 to have non-zero size.
            # text_feature(vocab_size=3),

            vector_feature(),
            image_feature(image_dest_folder),

            # reduction_modules: IndexError: amax(): Expected reduction dim 1 to have non-zero size.
            # audio_feature(audio_dest_folder),

            # reduction_modules: IndexError: amax(): Expected reduction dim 1 to have non-zero size.
            # timeseries_feature(),
            date_feature(),
            date_feature(),
            h3_feature(),
            set_feature(vocab_size=3),
            bag_feature(vocab_size=3),
        ]

        output_features = [
            category_feature(vocab_size=3),
            binary_feature(),
            numerical_feature(),
            # sequence_feature(vocab_size=3),
            # text_feature(vocab_size=3),
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
        backend = LocalTestBackend()
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'training': {'epochs': 2}
        }
        ludwig_model = LudwigModel(config, backend=backend)
        ludwig_model.train(
            dataset=data_csv_path,
            skip_save_training_description=True,
            skip_save_training_statistics=True,
            skip_save_model=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
        )

        print('Finished modeling')

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
            ludwig_model = LudwigModel.load(ludwigmodel_path, backend=backend)

        ##############################
        # collect weight tensors names
        ##############################
        original_predictions_df, _ = ludwig_model.predict(
            dataset=data_csv_path)
        original_weights = deepcopy(list(ludwig_model.model.parameters()))

        print('Finished prediction')
        print('Starting saving')

        #################
        # save torchscript
        #################
        torchscript_path = os.path.join(dir_path, 'torchscript')
        shutil.rmtree(torchscript_path, ignore_errors=True)
        ludwig_model.model.save_torchscript(torchscript_path)

        print('Finished saving')

        ###################################################
        # load Ludwig model, obtain predictions and weights
        ###################################################
        ludwig_model = LudwigModel.load(ludwigmodel_path, backend=backend)
        loaded_prediction_df, _ = ludwig_model.predict(dataset=data_csv_path)
        loaded_weights = deepcopy(list(ludwig_model.model.parameters()))

        print('Loaded predictions')

        #################################################
        # restore torchscript, obtain predictions and weights
        #################################################
        training_set_metadata_json_fp = os.path.join(
            ludwigmodel_path,
            TRAIN_SET_METADATA_FILE_NAME
        )

        dataset, training_set_metadata = preprocess_for_prediction(
            ludwig_model.config,
            dataset=data_csv_path,
            training_set_metadata=training_set_metadata_json_fp,
            backend=backend,
        )

        restored_model = torch.jit.load(torchscript_path)

        print('Loaded torchscript')

        # Check the outputs for one of the features for correctness
        # Here we choose the first output feature (categorical)
        of_name = list(ludwig_model.model.output_features.keys())[0]

        data_to_predict = {
            name: torch.from_numpy(dataset.dataset[feature.proc_column])
            for name, feature in ludwig_model.model.input_features.items()
        }

        logits = restored_model(data_to_predict)

        print('Got torchscript predictions')

        # Restoring from torchscript drops the names of NamedTuples.
        # restored_predictions = torch.argmax(logits[of_name][1], -1)
        restored_predictions = torch.argmax(
            forward_utils.get_output_feature_tensor(logits, of_name, 'logits'), -1)

        print(
            f'original_predictions_df[predictions_column_name]: {original_predictions_df[predictions_column_name]}')
        print(
            f"training_set_metadata[of_name]: {training_set_metadata[of_name]}")
        print(f'restored_predictions: {restored_predictions}')
        restored_predictions = [training_set_metadata[of_name]
                                ['idx2str'][idx] for idx in restored_predictions]

        restored_weights = deepcopy(list(restored_model.parameters()))

        #########
        # Cleanup
        #########
        shutil.rmtree(ludwigmodel_path, ignore_errors=True)
        shutil.rmtree(torchscript_path, ignore_errors=True)

        ###############################################
        # Check if weights and predictions are the same
        ###############################################

        # Check to weight values match the original model.
        assert utils.is_all_close(original_weights, loaded_weights)
        assert utils.is_all_close(original_weights, restored_weights)

        # Check that predictions are identical to the original model.
        assert np.all(
            original_predictions_df[predictions_column_name] ==
            loaded_prediction_df[predictions_column_name]
        )

        assert np.all(
            original_predictions_df[predictions_column_name] ==
            restored_predictions
        )
