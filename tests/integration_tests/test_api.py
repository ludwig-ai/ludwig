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

import numpy as np

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import ENCODERS, run_api_experiment
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


def run_api_experiment_separated_datasets(
        input_features,
        output_features,
        data_csv
):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(config)

    # Training with dataframe
    data_df = read_csv(data_csv)
    train_df = data_df.sample(frac=0.8)
    test_df = data_df.drop(train_df.index).sample(frac=0.5)
    validation_df = data_df.drop(train_df.index).drop(test_df.index)

    basename, ext = os.path.splitext(data_csv)
    train_fname = basename + '.train' + ext
    val_fname = basename + '.validation' + ext
    test_fname = basename + '.test' + ext
    output_dirs = []

    try:
        train_df.to_csv(train_fname)
        validation_df.to_csv(val_fname)
        test_df.to_csv(test_fname)

        # Training with csv
        _, _, output_dir = model.train(
            training_set=train_fname,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        output_dirs.append(output_dir)

        _, _, output_dir = model.train(
            training_set=train_fname,
            validation_set=val_fname,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        output_dirs.append(output_dir)

        _, _, output_dir = model.train(
            training_set=train_fname,
            validation_set=val_fname,
            test_set=test_fname,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        output_dirs.append(output_dir)

        _, output_dir = model.predict(dataset=test_fname)
        output_dirs.append(output_dir)

    finally:
        # Remove results/intermediate data saved to disk
        os.remove(train_fname)
        os.remove(val_fname)
        os.remove(test_fname)
        for output_dir in output_dirs:
            shutil.rmtree(output_dir, ignore_errors=True)

    output_dirs = []
    try:
        _, _, output_dir = model.train(
            training_set=train_df,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        output_dirs.append(output_dir)

        _, _, output_dir = model.train(
            training_set=train_df,
            validation_set=validation_df,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        output_dirs.append(output_dir)

        _, _, output_dir = model.train(
            training_set=train_df,
            validation_set=validation_df,
            test_set=test_df,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        output_dirs.append(output_dir)

        _, output_dir = model.predict(dataset=data_df)
        output_dirs.append(output_dir)

    finally:
        for output_dir in output_dirs:
            shutil.rmtree(output_dir, ignore_errors=True)


def test_api_intent_classification(csv_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in ENCODERS:
        input_features[0]['encoder'] = encoder
        run_api_experiment(input_features, output_features, data_csv=rel_path)


def test_api_intent_classification_separated(csv_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    rel_path = generate_data(input_features, output_features, csv_filename)
    for encoder in ENCODERS:
        input_features[0]['encoder'] = encoder
        run_api_experiment_separated_datasets(
            input_features, output_features, data_csv=rel_path
        )


def test_api_train_online(csv_filename):
    input_features = [sequence_feature(reduce_output='sum')]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]
    data_csv = generate_data(input_features, output_features, csv_filename)

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
    }
    model = LudwigModel(config)

    for i in range(2):
        model.train_online(dataset=data_csv)
    model.predict(dataset=data_csv)


def test_api_training_set(csv_filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [sequence_feature(reduce_output='sum')]
        output_features = [category_feature(vocab_size=2, reduce_input='sum')]

        data_csv = generate_data(input_features, output_features, csv_filename)
        val_csv = shutil.copyfile(data_csv,
                                  os.path.join(tmpdir, 'validation.csv'))
        test_csv = shutil.copyfile(data_csv, os.path.join(tmpdir, 'test.csv'))

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
        }
        model = LudwigModel(config)
        model.train(training_set=data_csv,
                    validation_set=val_csv,
                    test_set=test_csv)
        model.predict(dataset=test_csv)

        # Train again, this time the HDF5 cache will be used
        model.train(training_set=data_csv,
                    validation_set=val_csv,
                    test_set=test_csv)


def test_api_training_determinism(csv_filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [sequence_feature(reduce_output='sum')]
        output_features = [category_feature(vocab_size=2, reduce_input='sum')]

        data_csv = generate_data(input_features, output_features, csv_filename)

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
        }

        # Train the model 3 times:
        #
        # 1. seed x
        # 2. seed y
        # 3. seed x
        #
        # Check that models (1) and (3) produce the same weights,
        # but (1) and (2) do not
        rand_x = 42
        rand_y = 24

        model_1 = LudwigModel(config)
        model_1.train(dataset=data_csv, output_directory=tmpdir, random_seed=rand_x)

        model_2 = LudwigModel(config)
        model_2.train(dataset=data_csv, output_directory=tmpdir, random_seed=rand_y)

        model_3 = LudwigModel(config)
        model_3.train(dataset=data_csv, output_directory=tmpdir, random_seed=rand_x)

        model_weights_1 = model_1.model.get_weights()
        model_weights_2 = model_2.model.get_weights()
        model_weights_3 = model_3.model.get_weights()

        divergence = False
        for weight_1, weight_2 in zip(model_weights_1, model_weights_2):
            if not np.allclose(weight_1, weight_2):
                divergence = True
                break
        assert divergence, 'model_1 and model_2 have identical weights with different seeds!'

        for weight_1, weight_3 in zip(model_weights_1, model_weights_3):
            assert np.allclose(weight_1, weight_3)
