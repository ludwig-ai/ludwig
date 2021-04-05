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
import tensorflow as tf

from ludwig.backend import LOCAL_BACKEND
from ludwig.backend.dask import DaskBackend
from ludwig.utils.data_utils import read_parquet

from tests.integration_tests.utils import create_data_set_to_use
from tests.integration_tests.utils import audio_feature
from tests.integration_tests.utils import bag_feature
from tests.integration_tests.utils import binary_feature
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import date_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import h3_feature
from tests.integration_tests.utils import image_feature
from tests.integration_tests.utils import numerical_feature
from tests.integration_tests.utils import sequence_feature
from tests.integration_tests.utils import set_feature
from tests.integration_tests.utils import spawn
from tests.integration_tests.utils import text_feature
from tests.integration_tests.utils import timeseries_feature
from tests.integration_tests.utils import train_with_backend
from tests.integration_tests.utils import vector_feature


def split(data_parquet):
    data_df = read_parquet(data_parquet, LOCAL_BACKEND.df_engine.df_lib)
    train_df = data_df.sample(frac=0.8)
    test_df = data_df.drop(train_df.index).sample(frac=0.5)
    validation_df = data_df.drop(train_df.index).drop(test_df.index)

    basename, ext = os.path.splitext(data_parquet)
    train_fname = basename + '.train' + ext
    val_fname = basename + '.validation' + ext
    test_fname = basename + '.test' + ext

    train_df.to_parquet(train_fname)
    validation_df.to_parquet(val_fname)
    test_df.to_parquet(test_fname)
    return train_fname, val_fname, test_fname


def run_api_experiment(config, data_parquet):
    # Train on Parquet
    dask_backend = DaskBackend()
    train_with_backend(dask_backend, config, dataset=data_parquet)

    # Train on DataFrame directly
    data_df = read_parquet(data_parquet, df_lib=dask_backend.df_engine.df_lib)
    train_with_backend(dask_backend, config, dataset=data_df)


def run_split_api_experiment(config, data_parquet):
    backend = DaskBackend()

    train_fname, val_fname, test_fname = split(data_parquet)

    # Train
    train_with_backend(backend, config,
                       training_set=train_fname)

    # Train + Validation
    train_with_backend(backend, config,
                       training_set=train_fname,
                       validation_set=val_fname)

    # Train + Validation + Test
    train_with_backend(backend, config,
                       training_set=train_fname,
                       validation_set=val_fname,
                       test_set=test_fname)


@spawn
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
def test_dask_tabular():
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


@pytest.mark.distributed
def test_dask_split():
    input_features = [
        numerical_feature(normalization='zscore'),
        set_feature(),
        binary_feature(),
    ]
    output_features = [category_feature(vocab_size=2, reduce_input='sum')]
    run_test_parquet(input_features, output_features, run_fn=run_split_api_experiment)


@pytest.mark.distributed
def test_dask_timeseries():
    input_features = [timeseries_feature()]
    output_features = [numerical_feature()]
    run_test_parquet(input_features, output_features)


@pytest.mark.distributed
def test_dask_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dest_folder = os.path.join(tmpdir, 'generated_audio')
        input_features = [audio_feature(folder=audio_dest_folder)]
        output_features = [binary_feature()]
        run_test_parquet(input_features, output_features, num_examples=25)


@pytest.mark.distributed
def test_dask_lazy_load_audio_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dest_folder = os.path.join(tmpdir, 'generated_audio')
        input_features = [
            audio_feature(
                folder=audio_dest_folder,
                preprocessing={
                    'in_memory': False,
                }
            )
        ]
        output_features = [binary_feature()]
        run_test_parquet(input_features, output_features, expect_error=True)


@pytest.mark.distributed
def test_dask_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, 'generated_images')
        input_features = [
            image_feature(
                folder=image_dest_folder,
                encoder='resnet',
                preprocessing={
                    'in_memory': True,
                    'height': 12,
                    'width': 12,
                    'num_channels': 3,
                    'num_processes': 5
                },
                fc_size=16,
                num_filters=8
            ),
        ]
        output_features = [binary_feature()]
        run_test_parquet(input_features, output_features, num_examples=50)


@pytest.mark.distributed
def test_dask_lazy_load_image_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, 'generated_images')
        input_features = [
            image_feature(
                folder=image_dest_folder,
                encoder='resnet',
                preprocessing={
                    'in_memory': False,
                    'height': 12,
                    'width': 12,
                    'num_channels': 3,
                    'num_processes': 5
                },
                fc_size=16,
                num_filters=8
            ),
        ]
        output_features = [binary_feature()]
        run_test_parquet(input_features, output_features, expect_error=True)
