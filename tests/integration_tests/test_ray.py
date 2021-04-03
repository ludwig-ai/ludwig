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
import tempfile

import pytest
import ray
import tensorflow as tf

from ludwig.backend.ray import RayBackend, get_horovod_kwargs

from tests.integration_tests.utils import create_data_set_to_use
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


@pytest.fixture
def ray_start_4_cpus():
    res = ray.init(num_cpus=4)
    try:
        yield res
    finally:
        ray.shutdown()


def run_api_experiment(config, data_parquet):
    # Sanity check that we get 4 slots over 1 host
    kwargs = get_horovod_kwargs()
    assert kwargs.get('num_hosts') == 1
    assert kwargs.get('num_slots') == 4

    # Train on Parquet
    dask_backend = RayBackend()
    train_with_backend(dask_backend, config, dataset=data_parquet)


def run_test_parquet(
    input_features,
    output_features,
    num_examples=100,
    run_fn=run_api_experiment,
    expect_error=False
):
    tf.config.experimental_run_functions_eagerly(True)

    config = {
        'input_features': input_features,
        'output_features': output_features,
        'combiner': {'type': 'concat', 'fc_size': 14},
        'training': {'epochs': 2}
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
def test_ray_tabular(ray_start_4_cpus):
    input_features = [
        sequence_feature(reduce_output='sum'),
        numerical_feature(normalization='zscore'),
        set_feature(),
        text_feature(),
        binary_feature(),
        bag_feature(),
        vector_feature(),
        h3_feature(),
        date_feature(),
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]
    run_test_parquet(input_features, output_features)
