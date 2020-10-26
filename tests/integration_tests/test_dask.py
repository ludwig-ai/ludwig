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
import numpy as np
import os
import shutil
import tempfile

from ludwig.api import LudwigModel
from ludwig.backend import LocalBackend
from ludwig.backend.dask import DaskBackend
from ludwig.utils.data_utils import read_parquet

from tests.integration_tests.utils import create_data_set_to_use, run_api_experiment
from tests.integration_tests.utils import bag_feature
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import vector_feature


def run_api_experiment(input_features, output_features, data_parquet):
    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
    }

    backend = DaskBackend()
    # backend = LocalBackend()
    model = LudwigModel(config, backend=backend)
    output_dir = None

    try:
        # Training with csv
        _, _, output_dir = model.train(
            dataset=data_parquet,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )
        model.predict(dataset=data_parquet)

        model_dir = os.path.join(output_dir, 'model')
        loaded_model = LudwigModel.load(model_dir)

        # Necessary before call to get_weights() to materialize the weights
        loaded_model.predict(dataset=data_parquet)

        model_weights = model.model.get_weights()
        loaded_weights = loaded_model.model.get_weights()
        for model_weight, loaded_weight in zip(model_weights, loaded_weights):
            assert np.allclose(model_weight, loaded_weight)
    finally:
        # Remove results/intermediate data saved to disk
        shutil.rmtree(output_dir, ignore_errors=True)

    try:
        # Training with dataframe
        data_df = read_parquet(data_parquet, df_lib=backend.processor.df_lib)
        _, _, output_dir = model.train(
            dataset=data_df,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True
        )

        data_df = backend.processor.compute(data_df)
        model.predict(dataset=data_df)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_dask_tabular():
    # Single sequence input, single category output
    input_features = [
        sequence_feature(reduce_output='sum'),
        numerical_feature(normalization='zscore'),
        set_feature(),
        text_feature(),
        binary_feature(),
        bag_feature(),
        vector_feature()
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]

    # Generate test data
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, 'dataset.csv')
        dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=1000)
        dataset_parquet = create_data_set_to_use('parquet', dataset_csv)
        run_api_experiment(input_features, output_features, data_parquet=dataset_parquet)
