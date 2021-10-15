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
import contextlib
import os
import tempfile

import pytest
import ray
import tensorflow as tf

from ludwig.backend.ray import RayBackend, get_horovod_kwargs

from tests.integration_tests.utils import create_data_set_to_use, spawn
from tests.integration_tests.utils import bag_feature
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import date_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import h3_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import train_with_backend
from tests.integration_tests.utils import vector_feature


@contextlib.contextmanager
def ray_start_2_cpus():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = ray.init(
            num_cpus=2,
            include_dashboard=False,
            object_store_memory=150 * 1024 * 1024,
            # _temp_dir=tmpdir,
        )
        try:
            yield res
        finally:
            ray.shutdown()


def run_api_experiment(config, data_parquet):
    # Sanity check that we get 4 slots over 1 host
    kwargs = get_horovod_kwargs()
    # assert kwargs.get('num_workers') == 2

    # Train on Parquet
    dask_backend = RayBackend(
        processor={
            'parallelism': 2,
        },
        trainer={
            'num_workers': 2,
            'resources_per_worker': {
                'CPU': 0,
            }
        }
    )
    train_with_backend(dask_backend, config, dataset=data_parquet, evaluate=False)


@spawn
def run_test_parquet(
    input_features,
    output_features,
    num_examples=100,
    run_fn=run_api_experiment,
    expect_error=False
):
    tf.config.experimental_run_functions_eagerly(True)
    with ray_start_2_cpus():
        config = {
            'input_features': input_features,
            'output_features': output_features,
            'combiner': {'type': 'concat', 'fc_size': 14},
            'training': {'epochs': 2, 'batch_size': 8}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_filename = os.path.join(tmpdir, 'dataset.csv')
            dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=num_examples)
            dataset_parquet = create_data_set_to_use('parquet', dataset_csv)

            if expect_error:
                with pytest.raises(ValueError):
                    run_fn(config, data_parquet=dataset_parquet)
            else:
                run_fn(config, data_parquet=dataset_parquet)


@pytest.mark.distributed
def test_ray_tabular():
    input_features = [
        sequence_feature(reduce_output='sum'),
        numerical_feature(normalization='zscore'),
        set_feature(),
        binary_feature(),
        bag_feature(),
        vector_feature(),
        h3_feature(),
        date_feature(),
    ]
    output_features = [
        category_feature(vocab_size=2, reduce_input='sum'),
        binary_feature(),
        set_feature(max_len=3, vocab_size=5),
        numerical_feature(normalization='zscore'),
        vector_feature(),
    ]
    run_test_parquet(input_features, output_features)


@pytest.mark.distributed
def test_ray_text():
    input_features = [
        text_feature(),
    ]
    output_features = [
        text_feature(reduce_input=None, decoder='tagger'),
    ]
    run_test_parquet(input_features, output_features)


@pytest.mark.distributed
def test_ray_sequence():
    input_features = [
        sequence_feature(
            max_len=10,
            encoder='rnn',
            cell_type='lstm',
            reduce_output=None
        )
    ]
    output_features = [
        sequence_feature(
            max_len=10,
            decoder='tagger',
            attention=False,
            reduce_input=None
        )
    ]
    run_test_parquet(input_features, output_features)
