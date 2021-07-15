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


@pytest.mark.distributed
def test_pure_tf_model(csv_filename, tmpdir):
    dir_path = tmpdir
    data_csv_path = os.path.join(tmpdir, csv_filename)
    # image_dest_folder = os.path.join(tmpdir, 'generated_images')
    # audio_dest_folder = os.path.join(tmpdir, 'generated_audio')

    # Single sequence input, single category output
    input_features = [
        # binary_feature(),
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
        # binary_feature(),
        numerical_feature(),
        # sequence_feature(vocab_size=3),
        # text_feature(vocab_size=3),
        # set_feature(vocab_size=3),
        # vector_feature()
    ]

    # Generate test data
    data_csv_path = generate_data(input_features, output_features,
                                  data_csv_path)

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

    ludwig_tf_model = ludwig_model.create_inference_graph()
    print(ludwig_tf_model)

    # pred_data = list(build_synthetic_dataset(1, input_features))
    #
    # inputs = {
    #     c: tf.convert_to_tensor(
    #         [v],
    #         dtype=ludwig_model.model.input_features[c].get_inference_input_dtype()
    #     )
    #     for c, v in zip(pred_data[0], pred_data[1])
    # }
    # print(inputs)
    #
    # print(ludwig_tf_model(inputs))
