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
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME

from tests.integration_tests.utils import category_feature, binary_feature, \
    numerical_feature, text_feature, vector_feature, image_feature, \
    audio_feature, timeseries_feature, date_feature, h3_feature, set_feature, \
    bag_feature, LocalTestBackend
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature

# TODO: test different tokenizers, (maybe) layer-by-layer validation
def test_pure_tf_model(csv_filename, tmpdir):
    dir_path = tmpdir
    data_csv_path = os.path.join(tmpdir, csv_filename)
    # image_dest_folder = os.path.join(tmpdir, 'generated_images')
    # audio_dest_folder = os.path.join(tmpdir, 'generated_audio')

    # Configure features to be tested:
    input_features = [
        binary_feature(),
        numerical_feature(),
        category_feature(vocab_size=3),
        # sequence_feature(vocab_size=3),
        text_feature(vocab_size=3),
        # vector_feature(),
        # image_feature(image_dest_folder),
        # audio_feature(audio_dest_folder),
        # timeseries_feature(),
        # date_feature(),
        # h3_feature(),
        # set_feature(vocab_size=3),
        # bag_feature(vocab_size=3),
    ]
    output_features = [
        category_feature(vocab_size=3),
        binary_feature(),
        numerical_feature(),
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # set_feature(vocab_size=3),
        # vector_feature()
    ]
    backend = LocalTestBackend()
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'training': {'epochs': 2}
    }

    # Generate training data
    training_data_csv_path = generate_data(input_features, output_features,
                                  data_csv_path)


    # Train Ludwig (Pythonic) model:
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    # Create graph inference model (Tensorflow) from trained Ludwig model.
    # Note that Tensorflow is running with eager execution enabled:
    ludwig_tf_model = ludwig_model.create_inference_graph()
    ludwig_tf_model.compile()

    # Generate test data:
    pred_data = list(build_synthetic_dataset(1, input_features))
    # Note: needed because graph model's binary feature input expects strings:
    pred_data[1] = [str(v) + ""
                    if isinstance(v, np.bool_) else v
                    for v in pred_data[1]]

    # Convert test dataset to a dict to be consumed:
    tensor_inputs_dict = {
        c: tf.convert_to_tensor(
            [v],
            dtype=ludwig_model.model.input_features[c].get_inference_input_dtype()
        )
        for c, v in zip(pred_data[0], pred_data[1])
    }

    # Apply both models to the test data and compare their output:
    ludwig_model_predictions_df = ludwig_model.predict(tensor_inputs_dict)[0]
    ludwig_tf_model_predictions_dict = ludwig_tf_model(tensor_inputs_dict)
    # for k, v in results.items():
    #     print(k, v)
    #     print(v['predictions'].numpy())

    print("\n\n\n" + "-"*100 + "LUDWIG PREDICTIONS" + "-"*100)
    print(ludwig_model_predictions_df)
    # print(ludwig_model_predictions[0].keys())
    print()
    print(ludwig_model_predictions_df[f'{list(ludwig_tf_model_predictions_dict.keys())[0]}_predictions'])
    # print(type(ludwig_model_predictions[0]))
    # print(type(ludwig_model_predictions[1]))
    print("\n\n\n" + "-"*100 + "TF PREDICTIONS" + "-"*100)
    print(ludwig_tf_model_predictions_dict.items())
    print()
    print(ludwig_tf_model_predictions_dict[list(ludwig_tf_model_predictions_dict.keys())[0]]["predictions"].numpy())

    t1 = ludwig_model_predictions_df[f'{list(ludwig_tf_model_predictions_dict.keys())[0]}_predictions'].to_numpy()
    t2 = np.array([s.decode('UTF-8') for s in ludwig_tf_model_predictions_dict[list(ludwig_tf_model_predictions_dict.keys())[0]]["predictions"].numpy()])

    assert(t1 == t2)


    # Print statements for debugging:
    # print("\n\n\n" + "-"*100 + "INPUT FEATURES" + "-"*100)
    # print(inputs)
    # print("\n\n\n" + "-"*100 + "GRAPH APPLIED TO INPUT FEATURES)" + "-"*100)
    # tf.print(ludwig_tf_model.predict(inputs))
    # print("\n\n\n" + "-"*100 + "GRAPH_MODEL.INPUTS" + "-"*100)
    # print(ludwig_tf_model.inputs)