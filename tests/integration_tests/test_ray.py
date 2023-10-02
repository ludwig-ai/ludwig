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
import copy
import os
import tempfile
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.backend import create_ray_backend, initialize_backend, LOCAL_BACKEND
from ludwig.constants import (
    AUDIO,
    BAG,
    BALANCE_PERCENTAGE_TOLERANCE,
    BATCH_SIZE,
    BFILL,
    BINARY,
    CATEGORY,
    COLUMN,
    DATE,
    H3,
    IMAGE,
    MAX_BATCH_SIZE_DATASET_FRACTION,
    NAME,
    NUMBER,
    PREPROCESSING,
    SEQUENCE,
    SET,
    SPLIT,
    TEXT,
    TIMESERIES,
    TRAINER,
    VECTOR,
)
from ludwig.data.preprocessing import balance_data
from ludwig.data.split import DEFAULT_PROBABILITIES
from ludwig.utils.data_utils import read_parquet
from ludwig.utils.misc_utils import merge_dict
from tests.integration_tests.utils import (
    audio_feature,
    augment_dataset_with_none,
    bag_feature,
    binary_feature,
    category_feature,
    create_data_set_to_use,
    date_feature,
    generate_data,
    h3_feature,
    image_feature,
    number_feature,
    RAY_BACKEND_CONFIG,
    sequence_feature,
    set_feature,
    text_feature,
    timeseries_feature,
    train_with_backend,
    vector_feature,
)

ray = pytest.importorskip("ray")  # noqa

# Mark the entire module as distributed
pytestmark = [pytest.mark.distributed, pytest.mark.integration_tests_a]

import dask  # noqa: E402
import ray  # noqa: E402
import ray.exceptions  # noqa: E402
from ray.air.config import DatasetConfig  # noqa: E402
from ray.data import Dataset, DatasetPipeline  # noqa: E402
from ray.train._internal.dataset_spec import DataParallelIngestSpec  # noqa: E402

from ludwig.backend.ray import get_trainer_kwargs, RayBackend  # noqa: E402
from ludwig.data.dataframe.dask import DaskEngine  # noqa: E402

try:
    import modin  # noqa: E402
except ImportError:
    modin = None


@ray.remote(num_cpus=1, num_gpus=1)
def train_gpu(config, dataset, output_directory):
    model = LudwigModel(config, backend="local")
    _, _, output_dir = model.train(dataset, output_directory=output_directory)
    return os.path.join(output_dir, "model")


@ray.remote(num_cpus=1, num_gpus=0)
def predict_cpu(model_dir, dataset):
    model = LudwigModel.load(model_dir, backend="local")
    model.predict(dataset)


def run_api_experiment(
    config,
    dataset,
    backend_config,
    predict=False,
    evaluate=True,
    skip_save_processed_input=True,
    skip_save_predictions=True,
    required_metrics=None,
):
    # Sanity check that we get 4 slots over 1 host
    kwargs = get_trainer_kwargs()
    if torch.cuda.device_count() > 0:
        assert kwargs.get("num_workers") == torch.cuda.device_count(), kwargs
        assert kwargs.get("use_gpu"), kwargs
    else:
        assert kwargs.get("num_workers") == 1, kwargs
        assert not kwargs.get("use_gpu"), kwargs

    # Train on Parquet
    model = train_with_backend(
        backend_config,
        config,
        dataset=dataset,
        evaluate=evaluate,
        predict=predict,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_predictions=skip_save_predictions,
        required_metrics=required_metrics,
    )

    assert isinstance(model.backend, RayBackend)
    if isinstance(model.backend.df_engine, DaskEngine):
        assert model.backend.df_engine.parallelism == backend_config["processor"]["parallelism"]

    return model


def run_split_api_experiment(config, data_parquet, backend_config):
    train_fname, val_fname, test_fname = split(data_parquet)

    # Train
    train_with_backend(backend_config, config, training_set=train_fname, evaluate=False, predict=True)

    # Train + Validation
    train_with_backend(
        backend_config, config, training_set=train_fname, validation_set=val_fname, evaluate=False, predict=False
    )

    # Train + Validation + Test
    train_with_backend(
        backend_config,
        config,
        training_set=train_fname,
        validation_set=val_fname,
        test_set=test_fname,
        evaluate=False,
        predict=False,
    )


def run_preprocessing(
    tmpdir,
    df_engine,
    input_features,
    output_features,
    dataset_type="parquet",
    num_examples_per_split=20,
    nan_percent=0.0,
    first_row_none=False,
    last_row_none=False,
    nan_cols=None,
):
    # Split the dataset manually to avoid randomness in splitting
    split_to_df = {}
    for split in range(3):
        csv_filename = os.path.join(tmpdir, f"{split}_dataset.csv")
        dataset_csv_path = generate_data(
            input_features,
            output_features,
            csv_filename,
            num_examples=num_examples_per_split,
        )
        dataset_df = pd.read_csv(dataset_csv_path)
        dataset_df[SPLIT] = split
        dataset_df.to_csv(dataset_csv_path, index=False)
        split_to_df[split] = dataset_df
    full_df_path = os.path.join(tmpdir, "dataset.csv")
    pd.concat(split_to_df.values()).to_csv(full_df_path, index=False)
    dataset = create_data_set_to_use(dataset_type, full_df_path, nan_percent=nan_percent)
    dataset = augment_dataset_with_none(dataset, first_row_none, last_row_none, nan_cols)

    # Configure ray backend
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, "batch_size": 8},
        PREPROCESSING: {
            SPLIT: {
                "type": "fixed",
            },
        },
    }
    backend_config = {**RAY_BACKEND_CONFIG}
    if df_engine:
        backend_config["processor"]["type"] = df_engine

    # Run preprocessing with ray backend
    ray_model = LudwigModel(config, backend=backend_config)
    *ray_datasets, ray_training_set_metadata = ray_model.preprocess(
        skip_save_processed_input=False,  # Save the processed input to test pyarrow write/read
        dataset=dataset,
    )

    # Run preprocessing with local backend using the ray_training_set_metadata to ensure parity of
    # token assignments, etc.
    local_model = LudwigModel(config, backend=LOCAL_BACKEND)
    *local_datasets, _ = local_model.preprocess(
        training_set_metadata=ray_training_set_metadata,
        dataset=dataset,
    )

    for ray_dataset, local_dataset in zip(ray_datasets, local_datasets):
        ray_df = ray_model.backend.df_engine.compute(ray_dataset.to_df())
        local_df = local_model.backend.df_engine.compute(local_dataset.to_df())
        check_preprocessed_df_equal(local_df, ray_df)


def check_preprocessed_df_equal(df1, df2):
    for column in df1.columns:
        vals1 = df1[column].values
        vals2 = df2[column].values

        if any(feature_name in column for feature_name in [BINARY, CATEGORY]):
            is_equal = np.all(vals1 == vals2)
        elif any(feature_name in column for feature_name in [NUMBER]):
            is_equal = np.allclose(vals1, vals2)
        elif any(feature_name in column for feature_name in [SET, BAG, H3, DATE, TEXT, SEQUENCE, TIMESERIES, VECTOR]):
            is_equal = np.all([np.all(rv == lv) for rv, lv in zip(vals1, vals2)])
        elif any(feature_name in column for feature_name in [AUDIO, IMAGE]):
            is_equal = True
            for v1, v2 in zip(vals1, vals2):
                # We reshape both because there is a difference after preprocessing across the two backends.
                # With the distributed backend, the data is flattened and then later reshaped to its original shape
                # during training. With the local backend, the data is kept its original shape throughout.
                # TODO: Determine whether this is desired behavior. Tracked here:
                # https://github.com/ludwig-ai/ludwig/issues/2645
                v1 = v1.reshape(-1)
                v2 = v2.reshape(-1)
                is_equal &= np.allclose(v1, v2, atol=1e-5)
                if not is_equal:
                    break
        assert is_equal, f"Column {column} is not equal. Expected {vals1[:2]}, got {vals2[:2]}"


def split(data_parquet):
    data_df = read_parquet(data_parquet, LOCAL_BACKEND.df_engine.df_lib)
    train_df = data_df.sample(frac=0.8)
    test_df = data_df.drop(train_df.index).sample(frac=0.5)
    validation_df = data_df.drop(train_df.index).drop(test_df.index)

    basename, ext = os.path.splitext(data_parquet)
    train_fname = basename + ".train" + ext
    val_fname = basename + ".validation" + ext
    test_fname = basename + ".test" + ext

    train_df.to_parquet(train_fname)
    validation_df.to_parquet(val_fname)
    test_df.to_parquet(test_fname)
    return train_fname, val_fname, test_fname


def run_test_with_features(
    input_features,
    output_features,
    num_examples=100,
    run_fn=run_api_experiment,
    expect_error=False,
    df_engine=None,
    dataset_type="parquet",
    predict=False,
    skip_save_processed_input=True,
    skip_save_predictions=True,
    nan_percent=0.0,
    preprocessing=None,
    first_row_none=False,
    last_row_none=False,
    nan_cols=None,
    required_metrics=None,
    backend_kwargs=None,
):
    preprocessing = preprocessing or {}
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, "batch_size": 8},
    }
    if preprocessing:
        config[PREPROCESSING] = preprocessing

    backend_kwargs = copy.deepcopy(backend_kwargs or {})
    backend_config = merge_dict(RAY_BACKEND_CONFIG, backend_kwargs)
    if df_engine:
        backend_config["processor"]["type"] = df_engine

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, "dataset.csv")
        dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=num_examples)
        dataset = create_data_set_to_use(dataset_type, dataset_csv, nan_percent=nan_percent)
        dataset = augment_dataset_with_none(dataset, first_row_none, last_row_none, nan_cols)

        if expect_error:
            with pytest.raises((RuntimeError, ray.exceptions.RayTaskError)):
                run_fn(
                    config,
                    dataset=dataset,
                    backend_config=backend_config,
                    predict=predict,
                    skip_save_processed_input=skip_save_processed_input,
                    skip_save_predictions=skip_save_predictions,
                    required_metrics=required_metrics,
                )
        else:
            run_fn(
                config,
                dataset=dataset,
                backend_config=backend_config,
                predict=predict,
                skip_save_processed_input=skip_save_processed_input,
                skip_save_predictions=skip_save_predictions,
                required_metrics=required_metrics,
            )


@pytest.mark.parametrize("df_engine", ["pandas", "dask"])
@pytest.mark.distributed
def test_ray_read_binary_files(tmpdir, df_engine, ray_cluster_2cpu):
    preprocessing_params = {
        "audio_file_length_limit_in_s": 3.0,
        "missing_value_strategy": BFILL,
        "in_memory": True,
        "padding_value": 0,
        "norm": "per_file",
        "audio_feature": {
            "type": "fbank",
            "window_length_in_s": 0.04,
            "window_shift_in_s": 0.02,
            "num_filter_bands": 80,
        },
    }
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")
    audio_params = audio_feature(folder=audio_dest_folder, preprocessing=preprocessing_params)

    dataset_path = os.path.join(tmpdir, "dataset.csv")
    dataset_path = generate_data([audio_params], [], dataset_path, num_examples=10)
    dataset_path = create_data_set_to_use("csv", dataset_path, nan_percent=0.1)

    backend_config = {**RAY_BACKEND_CONFIG}
    backend_config["processor"]["type"] = df_engine
    backend = initialize_backend(backend_config)
    df = backend.df_engine.df_lib.read_csv(dataset_path)
    series = df[audio_params[COLUMN]]
    proc_col = backend.read_binary_files(series)
    proc_col = backend.df_engine.compute(proc_col)

    backend = initialize_backend(LOCAL_BACKEND)
    df = backend.df_engine.df_lib.read_csv(dataset_path)
    series = df[audio_params[COLUMN]]
    proc_col_expected = backend.read_binary_files(series)

    assert proc_col.equals(proc_col_expected)


@pytest.mark.slow
@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
@pytest.mark.parametrize(
    "trainer_strategy",
    [
        pytest.param("ddp", id="ddp", marks=pytest.mark.distributed),
        pytest.param("horovod", id="horovod", marks=[pytest.mark.distributed, pytest.mark.horovod]),
    ],
)
def test_ray_outputs(dataset_type, trainer_strategy, ray_cluster_2cpu):
    input_features = [
        binary_feature(),
    ]
    binary_feature_config = binary_feature()
    category_feature_config = category_feature(output_feature=True)
    output_features = [
        number_feature(),
        category_feature_config,
        binary_feature_config,
        # TODO: feature type not yet supported
        # text_feature(decoder={"vocab_size": 3}),  # Error having to do with a missing key (#2586)
        # sequence_feature(decoder={"vocab_size": 3}),  # Error having to do with a missing key (#2586)
    ]
    # NOTE: This test runs without NaNs because having multiple output features with DROP_ROWS strategy leads to
    # flakiness in the test having to do with uneven allocation of samples between Ray workers.
    run_test_with_features(
        input_features,
        output_features,
        df_engine="dask",
        dataset_type=dataset_type,
        predict=True,
        skip_save_predictions=False,
        required_metrics={
            binary_feature_config[NAME]: {"roc_auc"},
            category_feature_config[NAME]: {"roc_auc"},
        },  # ensures that these metrics are not omitted.
        backend_kwargs={
            TRAINER: {"strategy": trainer_strategy},
        },
    )


@pytest.mark.skip(reason="Occasional metadata mismatch error: https://github.com/ludwig-ai/ludwig/issues/2889")
@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
@pytest.mark.distributed
def test_ray_set_and_vector_outputs(dataset_type, ray_cluster_2cpu):
    input_features = [
        binary_feature(),
    ]
    # The synthetic set feature generator inserts between 0 and `vocab_size` entities per entry. 0 entities creates a
    # null (NaN) entry. The default behavior for such entries in output features is to DROP_ROWS. This leads to poorly
    # handled non-determinism when comparing the metrics between the local and Ray backends. We work around this by
    # setting the `missing_value_strategy` to `fill_with_const` and setting the `fill_value` to the empty string.
    set_feature_config = set_feature(
        decoder={"vocab_size": 3},
        preprocessing={"missing_value_strategy": "fill_with_const", "fill_value": ""},
    )
    output_features = [
        vector_feature(),
        set_feature_config,
    ]
    # NOTE: This test runs without NaNs because having multiple output features with DROP_ROWS strategy leads to
    # flakiness in the test having to do with uneven allocation of samples between Ray workers.
    run_test_with_features(
        input_features,
        output_features,
        df_engine="dask",
        dataset_type=dataset_type,
        predict=True,
        skip_save_predictions=False,
        required_metrics={set_feature_config[NAME]: {"jaccard"}},  # ensures that the metric is not omitted.
    )


@pytest.mark.distributed
@pytest.mark.parametrize(
    "df_engine",
    [
        "dask",
        pytest.param(
            "modin",
            marks=[
                pytest.mark.skipif(modin is None, reason="modin not installed"),
                pytest.mark.skip(reason="https://github.com/ludwig-ai/ludwig/issues/2643"),
            ],
        ),
    ],
)
def test_ray_tabular(tmpdir, df_engine, ray_cluster_2cpu):
    input_features = [
        category_feature(encoder={"vocab_size": 2}, reduce_input="sum"),
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
        bag_feature(),
        h3_feature(),
        date_feature(),
    ]
    output_features = [
        binary_feature(bool2str=["No", "Yes"]),
        binary_feature(),
        number_feature(normalization="zscore"),
    ]
    run_preprocessing(
        tmpdir,
        df_engine,
        input_features,
        output_features,
    )


@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
@pytest.mark.distributed
def test_ray_tabular_save_inputs(tmpdir, dataset_type, ray_cluster_2cpu):
    input_features = [
        category_feature(encoder={"vocab_size": 2}, reduce_input="sum"),
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
        bag_feature(),
        date_feature(
            preprocessing={"fill_value": "2020-01-01"}
        ),  # fill_value must be set to achieve parity between backends (otherwise fill value would be "now")
        # TODO: feature type not yet supported
        # h3_feature(),  # ValueError casting large int strings (e.g. '5.864041857092157e+17') to int (#2588)
    ]
    output_features = [
        category_feature(decoder={"vocab_size": 5}),  # Regression test for #1991 requires multi-class predictions.
    ]
    run_preprocessing(
        tmpdir,
        "dask",
        input_features,
        output_features,
        dataset_type=dataset_type,
        nan_percent=0.1,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
def test_ray_text_sequence_timeseries(tmpdir, dataset_type, ray_cluster_2cpu):
    input_features = [
        text_feature(),
        sequence_feature(encoder={"reduce_output": "sum"}),
        timeseries_feature(),
    ]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmpdir,
        "dask",
        input_features,
        output_features,
        dataset_type=dataset_type,
        nan_percent=0.1,
    )


@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
@pytest.mark.distributed
def test_ray_vector(tmpdir, dataset_type, ray_cluster_2cpu):
    input_features = [
        vector_feature(),
    ]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmpdir,
        "dask",
        input_features,
        output_features,
        dataset_type=dataset_type,
        nan_percent=0.0,  # NaN handling not supported for vectors.
    )


@pytest.mark.parametrize("dataset_type", ["csv", "parquet"])
@pytest.mark.distributed
def test_ray_audio(tmp_path, dataset_type, ray_cluster_2cpu):
    preprocessing_params = {
        "audio_file_length_limit_in_s": 3.0,
        "missing_value_strategy": BFILL,
        "in_memory": True,
        "padding_value": 0,
        "norm": "per_file",
        "type": "fbank",
        "window_length_in_s": 0.04,
        "window_shift_in_s": 0.02,
        "num_filter_bands": 80,
    }
    audio_dest_folder = os.path.join(tmp_path, "generated_audio")
    input_features = [audio_feature(folder=audio_dest_folder, preprocessing=preprocessing_params)]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmp_path,
        "dask",
        input_features,
        output_features,
        dataset_type=dataset_type,
        nan_percent=0.1,
    )


@pytest.mark.parametrize("dataset_type", ["csv", "parquet", "pandas+numpy_images"])
@pytest.mark.distributed
def test_ray_image(tmpdir, dataset_type, ray_cluster_2cpu):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
            encoder={"output_size": 16, "num_filters": 8},
        ),
    ]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmpdir,
        "dask",
        input_features,
        output_features,
        dataset_type=dataset_type,
        nan_percent=0.1,
    )


@pytest.mark.parametrize(
    "settings",
    [(True, False, "ffill"), (False, True, "bfill"), (True, True, "bfill"), (True, True, "ffill")],
    ids=["first_row_none", "last_row_none", "first_and_last_row_none_bfill", "first_and_last_row_none_ffill"],
)
@pytest.mark.distributed
def test_ray_image_with_fill_strategy_edge_cases(tmpdir, settings, ray_cluster_2cpu):
    first_row_none, last_row_none, missing_value_strategy = settings
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [
        image_feature(
            folder=image_dest_folder,
            preprocessing={
                "in_memory": True,
                "height": 12,
                "width": 12,
                "num_channels": 3,
                "num_processes": 5,
                "missing_value_strategy": missing_value_strategy,
            },
            encoder={"output_size": 16, "num_filters": 8},
        ),
    ]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmpdir,
        "dask",
        input_features,
        output_features,
        dataset_type="pandas+numpy_images",
        first_row_none=first_row_none,
        last_row_none=last_row_none,
        nan_cols=[input_features[0][NAME]],
    )


# TODO(geoffrey): Fold modin tests into test_ray_image as @pytest.mark.parametrized once tests are optimized


@pytest.mark.distributed
@pytest.mark.skipif(modin is None, reason="modin not installed")
@pytest.mark.skip(reason="https://github.com/ludwig-ai/ludwig/issues/2643")
def test_ray_image_modin(tmpdir, ray_cluster_2cpu):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder={
                "type": "stacked_cnn",
                "output_size": 16,
            },
            preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
        ),
    ]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmpdir,
        "modin",
        input_features,
        output_features,
        dataset_type="csv",
        nan_percent=0.1,
    )


@pytest.mark.distributed
def test_ray_image_multiple_features(tmpdir, ray_cluster_2cpu):
    input_features = [
        image_feature(
            folder=os.path.join(tmpdir, "generated_images_1"),
            preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
            encoder={"output_size": 16, "num_filters": 8},
        ),
        image_feature(
            folder=os.path.join(tmpdir, "generated_images_2"),
            preprocessing={"in_memory": True, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
            encoder={"output_size": 16, "num_filters": 8},
        ),
    ]
    output_features = [
        binary_feature(),
    ]
    run_preprocessing(
        tmpdir,
        "dask",
        input_features,
        output_features,
        dataset_type="csv",
        nan_percent=0.1,
    )


@pytest.mark.skip(reason="flaky: ray is running out of resources")
@pytest.mark.distributed
def test_ray_split(ray_cluster_2cpu):
    input_features = [
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
    ]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]
    run_test_with_features(
        input_features,
        output_features,
        run_fn=run_split_api_experiment,
    )


@pytest.mark.distributed
def test_ray_lazy_load_audio_error(tmpdir, ray_cluster_2cpu):
    audio_dest_folder = os.path.join(tmpdir, "generated_audio")
    input_features = [
        audio_feature(
            folder=audio_dest_folder,
            preprocessing={
                "in_memory": False,
            },
        )
    ]
    output_features = [
        binary_feature(),
    ]
    run_test_with_features(input_features, output_features, expect_error=True)


@pytest.mark.slow
@pytest.mark.distributed
def test_ray_lazy_load_image_works(tmpdir, ray_cluster_2cpu):
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [
        image_feature(
            folder=image_dest_folder,
            encoder={
                "type": "stacked_cnn",
                "output_size": 16,
            },
            preprocessing={"in_memory": False, "height": 12, "width": 12, "num_channels": 3, "num_processes": 5},
        ),
    ]
    output_features = [
        binary_feature(),
    ]
    run_test_with_features(input_features, output_features, expect_error=False)


# TODO(travis): move this to separate gpu module so we only have one ray cluster running at a time
# @pytest.mark.skipif(torch.cuda.device_count() == 0, reason="test requires at least 1 gpu")
# @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu support")
# @pytest.mark.distributed
# def test_train_gpu_load_cpu(ray_cluster_2cpu):
#     input_features = [
#         category_feature(encoder={"vocab_size": 2}, reduce_input="sum"),
#         number_feature(normalization="zscore"),
#     ]
#     output_features = [
#         binary_feature(),
#     ]
#     run_test_with_features(input_features, output_features, run_fn=_run_train_gpu_load_cpu, num_gpus=1)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "method, balance",
    [
        ("oversample_minority", 0.25),
        ("oversample_minority", 0.5),
        ("oversample_minority", 0.75),
        ("undersample_majority", 0.25),
        ("undersample_majority", 0.5),
        ("undersample_majority", 0.75),
    ],
)
def test_balance_ray(method, balance, ray_cluster_2cpu):
    config = {
        "input_features": [
            {"name": "Index", "proc_column": "Index", "type": "number"},
            {"name": "random_1", "proc_column": "random_1", "type": "number"},
            {"name": "random_2", "proc_column": "random_2", "type": "number"},
        ],
        "output_features": [{"name": "Label", "proc_column": "Label", "type": "binary"}],
        "preprocessing": {"oversample_minority": None, "undersample_majority": None},
    }
    input_df = pd.DataFrame(
        {
            "Index": np.arange(0, 200, 1),
            "random_1": np.random.randint(0, 50, 200),
            "random_2": np.random.choice(["Type A", "Type B", "Type C", "Type D"], 200),
            "Label": np.concatenate((np.zeros(180), np.ones(20))),
            "split": np.zeros(200),
        }
    )
    config["preprocessing"][method] = balance
    target = config["output_features"][0][NAME]

    backend = create_ray_backend()
    input_df = backend.df_engine.from_pandas(input_df)
    test_df = balance_data(input_df, config["output_features"], config["preprocessing"], backend, 42)

    majority_class = test_df[target].value_counts().compute()[test_df[target].value_counts().compute().idxmax()]
    minority_class = test_df[target].value_counts().compute()[test_df[target].value_counts().compute().idxmin()]
    new_class_balance = round(minority_class / majority_class, 2)

    assert abs(balance - new_class_balance) < BALANCE_PERCENTAGE_TOLERANCE


def _run_train_gpu_load_cpu(config, data_parquet):
    with tempfile.TemporaryDirectory() as output_dir:
        model_dir = ray.get(train_gpu.remote(config, data_parquet, output_dir))
        ray.get(predict_cpu.remote(model_dir, data_parquet))


# TODO(geoffrey): add a GPU test for batch size tuning


@pytest.mark.distributed
@pytest.mark.parametrize(
    ("max_batch_size", "expected_final_learning_rate"),
    [(256, 0.001), (8, 0.001)],
)
def test_tune_batch_size_lr_cpu(tmpdir, ray_cluster_2cpu, max_batch_size, expected_final_learning_rate):
    config = {
        "input_features": [
            number_feature(normalization="zscore"),
            set_feature(),
            binary_feature(),
        ],
        "output_features": [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")],
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {
            "epochs": 2,
            "batch_size": "auto",
            "learning_rate": "auto",
            "max_batch_size": max_batch_size,
        },
    }

    backend_config = copy.deepcopy(RAY_BACKEND_CONFIG)

    num_samples = 200
    csv_filename = os.path.join(tmpdir, "dataset.csv")
    dataset_csv = generate_data(
        config["input_features"], config["output_features"], csv_filename, num_examples=num_samples
    )
    dataset_parquet = create_data_set_to_use("parquet", dataset_csv)
    model = run_api_experiment(config, dataset=dataset_parquet, backend_config=backend_config, evaluate=False)

    num_train_samples = num_samples * DEFAULT_PROBABILITIES[0]
    max_batch_size_by_train_examples = MAX_BATCH_SIZE_DATASET_FRACTION * num_train_samples
    max_batch_size = (
        max_batch_size_by_train_examples
        if max_batch_size is None
        else min(max_batch_size_by_train_examples, max_batch_size)
    )
    assert 2 < model.config[TRAINER]["batch_size"] <= max_batch_size
    assert model.config[TRAINER]["learning_rate"] == expected_final_learning_rate


@pytest.mark.slow
@pytest.mark.parametrize("calibration", [True, False])
@pytest.mark.distributed
def test_ray_calibration(calibration, ray_cluster_2cpu):
    input_features = [
        number_feature(normalization="zscore"),
        set_feature(),
        binary_feature(),
    ]
    output_features = [
        binary_feature(calibration=calibration),
        category_feature(decoder={"vocab_size": 3}, calibration=calibration),
    ]
    run_test_with_features(input_features, output_features)


@pytest.mark.slow
@pytest.mark.distributed
def test_ray_distributed_predict(ray_cluster_2cpu):
    preprocessing_params = {
        "audio_file_length_limit_in_s": 3.0,
        "missing_value_strategy": BFILL,
        "in_memory": True,
        "padding_value": 0,
        "norm": "per_file",
        "type": "fbank",
        "window_length_in_s": 0.04,
        "window_shift_in_s": 0.02,
        "num_filter_bands": 80,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features = [audio_feature(folder=audio_dest_folder, preprocessing=preprocessing_params)]
        output_features = [
            binary_feature(),
        ]

        config = {
            "input_features": input_features,
            "output_features": output_features,
            TRAINER: {"epochs": 2, "batch_size": 8},
        }
        # Deep copy RAY_BACKEND_CONFIG to avoid shallow copy modification
        backend_config = copy.deepcopy(RAY_BACKEND_CONFIG)
        # Manually override num workers to 2 for distributed training and distributed predict
        backend_config["trainer"]["num_workers"] = 2
        csv_filename = os.path.join(tmpdir, "dataset.csv")
        dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=100)
        dataset = create_data_set_to_use("csv", dataset_csv, nan_percent=0.0)
        model = LudwigModel(config, backend=backend_config)

        _, _, _ = model.train(
            dataset=dataset,
            training_set=dataset,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            skip_save_log=True,
        )

        preds, _ = model.predict(dataset=dataset)

        # compute the predictions
        preds = preds.compute()
        assert preds.iloc[1].name != preds.iloc[42].name


@pytest.mark.slow
@pytest.mark.distributed
def test_ray_preprocessing_placement_group(ray_cluster_2cpu):
    preprocessing_params = {
        "audio_file_length_limit_in_s": 3.0,
        "missing_value_strategy": BFILL,
        "in_memory": True,
        "padding_value": 0,
        "norm": "per_file",
        "type": "fbank",
        "window_length_in_s": 0.04,
        "window_shift_in_s": 0.02,
        "num_filter_bands": 80,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_dest_folder = os.path.join(tmpdir, "generated_audio")
        input_features = [audio_feature(folder=audio_dest_folder, preprocessing=preprocessing_params)]
        output_features = [
            binary_feature(),
        ]

        config = {
            "input_features": input_features,
            "output_features": output_features,
            TRAINER: {"epochs": 2, "batch_size": 8},
        }

        backend_config = {**RAY_BACKEND_CONFIG}
        backend_config["preprocessor_kwargs"] = {"num_cpu": 1}
        csv_filename = os.path.join(tmpdir, "dataset.csv")
        dataset_csv = generate_data(input_features, output_features, csv_filename, num_examples=100)
        dataset = create_data_set_to_use("csv", dataset_csv, nan_percent=0.0)
        model = LudwigModel(config, backend=backend_config)
        _, _, output_dir = model.train(
            dataset=dataset,
            training_set=dataset,
            skip_save_processed_input=True,
            skip_save_progress=True,
            skip_save_unprocessed_output=True,
            skip_save_log=True,
        )
        preds, _ = model.predict(dataset=dataset)


@pytest.mark.skip(reason="This test needs a rewrite with Ray 2.3")
@pytest.mark.distributed
class TestDatasetWindowAutosizing:
    """Test dataset windowing with different dataset sizes and settings.

    Note that for these tests to run efficiently, windowing must be triggered while remaining within the object store
    memory size. The current heuristic is to trigger windowing when the dataset exceeds
    `ray.cluster_resources()['object_store_memory'] // 5` bytes.
    """

    @property
    def object_store_size(self):
        """The amount of object store memory available to the cluster fixture."""
        return int(ray.cluster_resources()["object_store_memory"])

    @property
    def auto_window_size(self):
        """The heuristic size of the automatic window in bytes."""
        return int(self.object_store_size // 5)

    @property
    def num_partitions(self):
        """The number of Dask dataframe partitions to create."""
        return 100

    def create_dataset_pipeline(
        self, size: int, window_size_bytes: Optional[Union[int, Literal["auto"]]] = None
    ) -> "DatasetPipeline":
        """Create a dataset of specified size to test auto-sizing.

        Args:
            size: Total size of the dataset in bytes
            window_size_bytes: Pass to override the auto_window size

        Returns:
            A Ludwig RayDataset of the specified size.
        """
        # Create a dataset of the specified size with 100 partitions.
        # This translates to 100 blocks within the `ray.data.Dataset`.
        df = pd.DataFrame(
            {
                "in_column": np.random.randint(0, 1, size=(size // 2,), dtype=np.uint8),
                "out_column": np.random.randint(0, 1, size=(size // 2,), dtype=np.uint8),
            }
        )
        df = dask.dataframe.from_pandas(df, npartitions=self.num_partitions)

        # Create a model with the dataset and
        config = {
            "input_features": [{"name": "in_column", "type": "binary"}],
            "output_features": [{"name": "out_column", "type": "binary"}],
            TRAINER: {"epochs": 1, BATCH_SIZE: 128},
        }
        backend_config = copy.deepcopy(RAY_BACKEND_CONFIG)
        backend_config["loader"] = {"window_size_bytes": window_size_bytes}
        backend_config["preprocessor_kwargs"] = {"num_cpu": 1}
        model = LudwigModel(config, backend=backend_config)

        # Create a dataset using the model backend to ensure it
        # is initialized correctly.
        ds = model.backend.dataset_manager.create(df, config=model.config, training_set_metadata={})

        # To window without using a training session, we configure `DataParallelIngestSpec` to use the specified window
        # size and turn off other features (e.g., shuffle) that may incur computational overhead.
        dataset_config = DatasetConfig(
            fit=False,
            split=False,
            transform=False,
            use_stream_api=True,
            stream_window_size=ds.window_size_bytes,
            global_shuffle=False,
        )
        spec = DataParallelIngestSpec({"train": dataset_config})

        # These two must be called in sequence so that the dataset is tracked internally. No preprocessing is applied.
        # The dummy argument `[1]` is used to indicate that the dataset should not be split. Normally, this argument
        # would correspond with Ray Actor metadata to distribute the preprocessed data.
        spec.preprocess_datasets(None, {"train": ds.ds})
        pipe = spec.get_dataset_shards([1])[0]["train"]
        return pipe

    def window_gen(self, pipe: "DatasetPipeline") -> "Dataset":
        """Convenient access to individual windows in a dataset pipeline."""
        for window in pipe._base_iterable:
            yield window()

    def test_small_dataset(self, ray_cluster_2cpu):
        """A small dataset should not trigger automatic window sizing.

        Without automatic window sizing, the number of blocks in the pipeline should match the number of partitions in
        the Dask dataframe.
        """
        pipe = self.create_dataset_pipeline(self.auto_window_size // 2, window_size_bytes="auto")
        window = next(self.window_gen(pipe))
        assert window.num_blocks() == self.num_partitions

    def test_large_dataset(self, ray_cluster_2cpu):
        """A large dataset should trigger windowing."""
        pipe = self.create_dataset_pipeline(self.auto_window_size * 2, window_size_bytes="auto")
        for i, window in enumerate(self.window_gen(pipe)):
            assert window.num_blocks() < self.num_partitions
            if i > 100:
                break

    def test_window_autosizing_disabled(self, ray_cluster_2cpu):
        """If window autosizing is disabled, no datasets should be windowed."""
        pipe = self.create_dataset_pipeline(self.auto_window_size * 2, window_size_bytes=None)
        window = next(self.window_gen(pipe))
        assert window.num_blocks() == self.num_partitions

    def test_user_window_size(self, ray_cluster_2cpu):
        """If the user supplies a window size, do not autosize."""
        auto_pipe = self.create_dataset_pipeline(self.auto_window_size * 2, window_size_bytes="auto")
        user_pipe = self.create_dataset_pipeline(self.auto_window_size * 2, window_size_bytes=self.auto_window_size * 4)
        windows = zip(self.window_gen(auto_pipe), self.window_gen(user_pipe))

        for i, (auto_window, user_window) in enumerate(windows):
            assert auto_window.num_blocks() < user_window.num_blocks()
            if i > 100:
                break
