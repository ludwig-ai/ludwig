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

    # Training with csv
    model.train(
        data_csv=data_csv,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )

    model.predict(data_csv=data_csv)

    # Remove results/intermediate data saved to disk
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)

    # Training with dataframe
    data_df = read_csv(data_csv)
    model.train(
        data_df=data_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_df=data_df)
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)


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

    train_df.to_csv(data_csv + '.train')
    validation_df.to_csv(data_csv + '.validation')
    test_df.to_csv(data_csv + '.test')

    # Training with csv
    model.train(
        data_train_csv=data_csv + '.train',
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.train(
        data_train_csv=data_csv + '.train',
        data_validation_df=data_csv + '.validation',
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.train(
        data_train_csv=data_csv + '.train',
        data_validation_df=data_csv + '.validation',
        data_test_csv=data_csv + '.test',
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )

    model.predict(data_csv=data_csv + '.test')

    # Remove results/intermediate data saved to disk
    os.remove(data_csv + '.train')
    os.remove(data_csv + '.validation')
    os.remove(data_csv + '.test')

    shutil.rmtree(model.exp_dir_name, ignore_errors=True)

    model.train(
        data_train_df=train_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.train(
        data_train_df=train_df,
        data_validation_df=validation_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.train(
        data_train_df=train_df,
        data_validation_df=validation_df,
        data_vtest_df=test_df,
        skip_save_processed_input=True,
        skip_save_progress=True,
        skip_save_unprocessed_output=True
    )
    model.predict(data_df=data_df)
    shutil.rmtree(model.exp_dir_name, ignore_errors=True)

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
