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
from tensorflow.python.client import session

from ludwig.api import LudwigModel
from ludwig.data.dataset_synthesizer import build_synthetic_dataset
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils.strings_utils import SpaceStringToListTokenizer, \
    SpacePunctuationStringToListTokenizer, UnderscoreStringToListTokenizer, \
    CommaStringToListTokenizer

from tests.integration_tests.utils import category_feature, binary_feature, \
    numerical_feature, text_feature, vector_feature, image_feature, \
    audio_feature, timeseries_feature, date_feature, h3_feature, set_feature, \
    bag_feature, LocalTestBackend
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature

import tensorflow.keras.backend as keras_backend

# def test_pure_tf_tokenizers():
#     tokenizersToTest = {
#         'space': SpaceStringToListTokenizer,
#         'space_punct': SpacePunctuationStringToListTokenizer,
#         'underscore': UnderscoreStringToListTokenizer,
#         'comma': CommaStringToListTokenizer,
#     }
#     for tokenizer in tokenizersToTest:     
#         tokenizerFlag = {
#             'preprocessing': {
#                 'word_tokenizer': '',
#             },
#         }
#     text_feature(vocab_size=3

def test_pure_tf_model_tokenizers(jjj):
    pass

def test_pure_tf_model(csv_filename, tmpdir):
    tf.config.run_functions_eagerly(True)
    tf.compat.v1.enable_eager_execution() 

    dir_path = tmpdir
    data_csv_path = os.path.join(tmpdir, csv_filename)
    # image_dest_folder = os.path.join(tmpdir, 'generated_images')
    # audio_dest_folder = os.path.join(tmpdir, 'generated_audio')

    # Single sequence input, single category output
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

    # Generate test data
    data_csv_path = generate_data(input_features, output_features,
                                  data_csv_path)

    # load into datafram
    # take column of text data -> convert to text of type to be tokenized

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

    pred_data = list(build_synthetic_dataset(1, input_features))

    # needed because ludwig_tf_model binary features expect strings
    pred_data[1] = [str(v) + ""
                    if isinstance(v, np.bool_) else v
                    for v in pred_data[1]]

    inputs = {
        c: tf.convert_to_tensor(
            [v],
            dtype=ludwig_model.model.input_features[c].get_inference_input_dtype()
        )
        for c, v in zip(pred_data[0], pred_data[1])
    }

    ludwig_tf_model.compile()

    # print(inputs)
    print("\n\n\n" + "-"*100 + "INPUT FEATURES" + "-"*100)
    print(inputs)
    print("\n\n\n" + "-"*100 + "MODEL APPLIED TO INPUT FEATURES)" + "-"*100)
    tf.print(ludwig_tf_model.predict(inputs))
    print("\n\n\n" + "-"*100 + "MODEL.INPUTS" + "-"*100)
    print(ludwig_tf_model.inputs)
    print("\n\n\n" + "-"*100 + "NUMPY TEST" + "-"*100)
    # print(keras_backend.eval(ludwig_tf_model.inputs[3]))
    # print(keras_backend.eval(ludwig_tf_model.inputs[3]))
    #func = keras_backend.function(ludwig_tf_model.inputs[3], [ludwig_tf_model.inputs[3].output])
    #result = func(inputs[3])
    results = ludwig_tf_model(inputs)
    for k, v in results.items():
        print(k, v)
        print(v['predictions'].numpy())


    ### VALIDATE TOKENIZERS - DIFFERENT ONES, and INVALID CASE

    ### COMPARE MODEL OUTPUTS