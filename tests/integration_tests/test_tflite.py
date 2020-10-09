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
import pytest
import tensorflow as tf

from ludwig.api import LudwigModel
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


@pytest.mark.parametrize('should_load_model', [True, False])
def test_tflite(csv_filename, should_load_model):
    #######
    # Setup
    #######
    dir_path = os.path.dirname(csv_filename)

    # Single category input, single category output
    input_features = [numerical_feature()]
    output_features = [numerical_feature()]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features,
                                  csv_filename)

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

    #################
    # save savedmodel
    #################
    savedmodel_path = os.path.join(dir_path, 'savedmodel')
    shutil.rmtree(savedmodel_path, ignore_errors=True)
    ludwig_model.model.save_savedmodel(savedmodel_path)

    #############################
    # convert the model to TFLite
    #############################
    tflite_converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
    tflite_model = tflite_converter.convert()

    #######################
    # save the TFLite model
    #######################
    tflite_path = os.path.join(dir_path, 'tflite')
    shutil.rmtree(tflite_path, ignore_errors=True)
    os.mkdir(tflite_path)
    tflite_filename = os.path.join(tflite_path, 'model.tflite')
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)

    ######################################################
    # restore TFLite model, obtain predictions and weights
    ######################################################
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # FIXME: This is "fake" random data, not consistent with actual test data
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

    ################################################################################
    # Note: output data cannot be compared between LudwigModel\SavedModel and TFLite
    # due to a precision loss caused by the optimization\compression itself
    ################################################################################

    #########
    # Cleanup
    #########
    shutil.rmtree(ludwigmodel_path, ignore_errors=True)
    shutil.rmtree(savedmodel_path, ignore_errors=True)
    shutil.rmtree(tflite_path, ignore_errors=True)
