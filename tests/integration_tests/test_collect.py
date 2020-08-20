# -*- coding: utf-8 -*-
# Copyright (c) 2020 Uber Technologies, Inc.
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

import multiprocessing
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

from ludwig.api import LudwigModel, Trainer
from ludwig.collect import collect_activations, collect_weights
from tests.integration_tests.utils import category_feature, generate_data, \
    sequence_feature, ENCODERS


def _prepare_data(csv_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    input_features[0]['encoder'] = ENCODERS[0]

    # Generate test data
    data_csv = generate_data(input_features, output_features, csv_filename)
    return input_features, output_features, data_csv


def _train(input_features, output_features, data_csv, **kwargs):
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)
    model.train(
        data_csv=data_csv,
        **kwargs
    )
    return model


def _get_layers_subprocess(model_path, queue):
    model = Trainer.load(model_path)
    keras_model = model.model.get_connected_model()
    layers = [layer.name for layer in keras_model.layers]
    queue.put(layers)


def _get_layers(model_path):
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(
        target=_get_layers_subprocess,
        args=(model_path, queue))
    p.start()
    p.join()
    layers = queue.get()
    return layers


def test_collect_weights(csv_filename):
    model = None
    try:
        # This will reset the layer numbering scheme TensorFlow uses.
        # Otherwise, when we load the model, its layer names will be appended
        # with "_1".
        tf.keras.backend.reset_uids()

        model = _train(*_prepare_data(csv_filename))
        model_path = os.path.join(model.exp_dir_name, 'model')

        weights = model.model.collect_weights()
        # todo fix: this function was returning 11 tensors originally.
        #  1 for the encoder (embeddings),
        #  2 for the decoder classifier (w and b)
        #  2 for the eval loss, for the eval accuracy and eval hits at k
        #  and 2 that I suppose are for the training loss but had names like
        #  mean and count without a namespace.
        #  After making metrics and losses class properties
        #  (metrics_functions in type output features and
        #  train_loss_function and eval_loss_function in output feature class)
        #  the number of weights should be 3 (just encoder and decoder weights)
        #  but it currently is 5, those mean and count that I believe to be
        #  the train loss function ones are still there.
        #  We should figure out why they appear here as they shouldn't
        assert len(weights) == 5

        # todo fix: this was :5 originally and made it so that we were missing
        #  the fact that the weights loaded in collect_weights() where a subset
        #  of all the weights. Specifically, the weights that are not loaded
        #  are the weights of hte output feature classifier (w and b).
        #  If the previous point is fixed, there should be only 3 elements
        #  in tensors (now they are 5) and collect_weights() should return 3
        #  filenames: 1 for encoder embeddings and 2 for classifier w and b,
        #  but right now it is returning 3 filenames, but they are
        #  for embedding, mean and count.
        tensors = [name for name, w in weights]
        assert len(tensors) == 3

        tf.keras.backend.reset_uids()
        with tempfile.TemporaryDirectory() as output_directory:
            filenames = collect_weights(model_path, tensors, output_directory)
            assert len(filenames) == 3

            for (name, weight), filename in zip(weights, filenames):
                saved_weight = np.load(filename)
                assert np.allclose(weight.numpy(), saved_weight), name
    finally:
        if model and model.exp_dir_name:
            shutil.rmtree(model.exp_dir_name, ignore_errors=True)


def test_collect_activations(csv_filename):
    model = None
    try:
        # This will reset the layer numbering scheme TensorFlow uses.
        # Otherwise, when we load the model, its layer names will be appended
        # with "_1".
        tf.keras.backend.reset_uids()

        model = _train(*_prepare_data(csv_filename))
        model_path = os.path.join(model.exp_dir_name, 'model')

        layers = _get_layers(model_path)
        assert len(layers) > 0

        tf.keras.backend.reset_uids()
        with tempfile.TemporaryDirectory() as output_directory:
            filenames = collect_activations(model_path, layers,
                                            data_csv=csv_filename,
                                            output_directory=output_directory)
            assert len(filenames) > len(layers)
    finally:
        if model and model.exp_dir_name:
            shutil.rmtree(model.exp_dir_name, ignore_errors=True)
