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

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import read_csv
from tests.integration_tests.utils import ENCODERS
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import sequence_feature


def run_api_experiment(input_features, output_features, data_csv):
    """
    Helper method to avoid code repetition in running an experiment
    :param input_features: input schema
    :param output_features: output schema
    :param data_csv: path to data
    :return: None
    """
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)
    output_dir = None

    try:
        # Training with csv
        _, _, output_dir = model.train(
            dataset=data_csv,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        model.predict(dataset=data_csv)

        model_dir = os.path.join(model.exp_dir_name, 'model')
        loaded_model = LudwigModel.load(model_dir)

        model_weights = model.model.get_weights()
        loaded_weights = loaded_model.model.get_weights()
        for model_weight, loaded_weight in zip(model_weights, loaded_weights):
            assert np.allclose(model_weight, loaded_weight)
    finally:
        # Remove results/intermediate data saved to disk
        shutil.rmtree(output_dir, ignore_errors=True)

    try:
        # Training with dataframe
        data_df = read_csv(data_csv)
        _, _, output_dir = model.train(
            dataset=data_df,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        model.predict(dataset=data_df)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


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
    model_definition = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    model = LudwigModel(model_definition)

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
