import os
import tempfile
from typing import List, Set
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import COLUMN, ENCODER, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PREPROCESSING, SPLIT, TYPE
from ludwig.schema.model_types.base import ModelConfig
from ludwig.types import FeatureConfigDict, ModelConfigDict
from ludwig.utils.misc_utils import merge_dict
from tests.integration_tests.utils import (
    binary_feature,
    category_feature,
    generate_data,
    image_feature,
    minio_test_creds,
    number_feature,
    private_param,
    remote_tmpdir,
    text_feature,
)

ray = pytest.importorskip("ray")

import dask.dataframe as dd  # noqa
from ray.tune.experiment.trial import Trial  # noqa

from ludwig.automl import auto_train, create_auto_config, train_with_config  # noqa
from ludwig.hyperopt.execution import RayTuneExecutor  # noqa

pytestmark = [pytest.mark.distributed, pytest.mark.integration_tests_c]


def to_name_set(features: List[FeatureConfigDict]) -> Set[str]:
    """Returns the list of feature names."""
    return {feature[NAME] for feature in features}


def merge_lists(a_features: List, b_features: List):
    for idx in range(max(len(a_features), len(b_features))):
        if idx >= len(a_features):
            a_features.append(b_features[idx])
        elif idx < len(b_features):
            a_features[idx] = merge_dict(a_features[idx], b_features[idx])


def merge_dict_with_features(a: ModelConfigDict, b: ModelConfigDict) -> ModelConfigDict:
    merge_lists(a[INPUT_FEATURES], b.get(INPUT_FEATURES, []))
    merge_lists(a[OUTPUT_FEATURES], b.get(OUTPUT_FEATURES, []))

    b = b.copy()
    if INPUT_FEATURES in b:
        del b[INPUT_FEATURES]
    if OUTPUT_FEATURES in b:
        del b[OUTPUT_FEATURES]

    return merge_dict(a, b)


def check_types(
    config: ModelConfigDict, input_features: List[FeatureConfigDict], output_features: List[FeatureConfigDict]
):
    actual_features = config.get(INPUT_FEATURES, []) + config.get(OUTPUT_FEATURES, [])
    expected_features = {f[NAME]: f for f in input_features + output_features}
    assert len(actual_features) == len(expected_features)
    for actual_feature in actual_features:
        expected_feature = expected_features[actual_feature[NAME]]
        assert (
            actual_feature[TYPE] == expected_feature[TYPE]
        ), f"{actual_feature[NAME]}: actual type {actual_feature[TYPE]} != {expected_feature[TYPE]}"


@pytest.fixture(scope="module")
def test_data_tabular_large():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [
            number_feature(),
            number_feature(),
            category_feature(encoder={"vocab_size": 3}),
            category_feature(encoder={"vocab_size": 3}),
        ]
        output_features = [category_feature(decoder={"vocab_size": 3})]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100
        )
        yield input_features, output_features, dataset_csv


@pytest.fixture(scope="module")
def test_data_tabular_small():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [
            number_feature(),
            category_feature(encoder={"vocab_size": 3}),
        ]
        output_features = [category_feature(decoder={"vocab_size": 3})]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100
        )
        yield input_features, output_features, dataset_csv


@pytest.fixture(scope="module")
def test_data_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, "generated_images")
        input_features = [
            image_feature(folder=image_dest_folder),
        ]
        output_features = [binary_feature()]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=20
        )
        yield input_features, output_features, dataset_csv


@pytest.fixture(scope="module")
def test_data_text():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_features = [
            text_feature(preprocessing={"tokenizer": "space"}),
        ]
        output_features = [binary_feature()]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=20
        )
        yield input_features, output_features, dataset_csv


@pytest.fixture(scope="module")
def test_data_multimodal():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dest_folder = os.path.join(tmpdir, "generated_images")
        input_features = [
            image_feature(folder=image_dest_folder),
            text_feature(preprocessing={"tokenizer": "space"}),
            number_feature(),
            category_feature(encoder={"vocab_size": 3}),
            category_feature(encoder={"vocab_size": 5}),
        ]
        output_features = [binary_feature()]
        dataset_csv = generate_data(
            input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=20
        )
        yield input_features, output_features, dataset_csv


@pytest.mark.distributed
@pytest.mark.parametrize(
    "test_data,expectations",
    [
        ("test_data_tabular_large", {"combiner": {"type": "tabnet"}}),
        ("test_data_tabular_small", {"combiner": {"type": "concat"}}),
        ("test_data_image", {"combiner": {"type": "concat"}}),
        (
            "test_data_text",
            {
                "input_features": [{"type": "text", "encoder": {"type": "bert"}}],
                "combiner": {"type": "concat"},
                "trainer": {
                    "batch_size": "auto",
                    "learning_rate": 1e-05,
                    "epochs": 10,
                    "optimizer": {"type": "adamw"},
                    "learning_rate_scheduler": {"warmup_fraction": 0.1},
                    "use_mixed_precision": True,
                },
                "defaults": {
                    "text": {
                        "encoder": {
                            "type": "bert",
                            "trainable": True,
                        }
                    }
                },
            },
        ),
        (
            "test_data_multimodal",
            {
                "input_features": [{"type": "image"}, {"type": "text", "encoder": {"type": "embed"}}],
                "combiner": {"type": "concat"},
            },
        ),
    ],
    ids=["tabular_large", "tabular_small", "image", "text", "multimodal"],
)
def test_create_auto_config(test_data, expectations, ray_cluster_2cpu, request):
    test_data = request.getfixturevalue(test_data)
    input_features, output_features, dataset_csv = test_data
    targets = [feature[NAME] for feature in output_features]
    df = dd.read_csv(dataset_csv)
    config = create_auto_config(df, targets, time_limit_s=600, backend="ray")

    # Ensure our configs are using the latest Ludwig schema
    ModelConfig.from_dict(config)

    assert to_name_set(config[INPUT_FEATURES]) == to_name_set(input_features)
    assert to_name_set(config[OUTPUT_FEATURES]) == to_name_set(output_features)
    check_types(config, input_features, output_features)

    expected = merge_dict_with_features(config, expectations)
    assert config == expected


def _get_sample_df(class_probs):
    nrows = 1000
    thresholds = np.cumsum((class_probs * nrows).astype(int))

    df = pd.DataFrame(np.random.randint(0, 100, size=(nrows, 3)), columns=["A", "B", "C"])

    def get_category(v):
        if v < thresholds[0]:
            return 0
        if thresholds[0] <= v < thresholds[1]:
            return 1
        return 2

    df["category"] = df.index.map(get_category).astype(np.int8)
    return df


@pytest.mark.distributed
def test_autoconfig_preprocessing_balanced():
    df = _get_sample_df(np.array([0.33, 0.33, 0.34]))

    config = create_auto_config(dataset=df, target="category", time_limit_s=1)

    # Ensure our configs are using the latest Ludwig schema
    ModelConfig.from_dict(config)

    assert PREPROCESSING not in config


@pytest.mark.distributed
def test_autoconfig_preprocessing_imbalanced():
    df = _get_sample_df(np.array([0.6, 0.2, 0.2]))

    config = create_auto_config(dataset=df, target="category", time_limit_s=1)

    # Ensure our configs are using the latest Ludwig schema
    ModelConfig.from_dict(config)

    assert PREPROCESSING in config
    assert SPLIT in config[PREPROCESSING]
    assert config[PREPROCESSING][SPLIT] == {TYPE: "stratify", COLUMN: "category"}


@pytest.mark.distributed
def test_autoconfig_preprocessing_text_image(tmpdir):
    image_dest_folder = os.path.join(tmpdir, "generated_images")

    input_features = [text_feature(preprocessing={"tokenizer": "space"}), image_feature(folder=image_dest_folder)]
    output_features = [category_feature(output_feature=True)]

    # Generate Dataset
    rel_path = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"))
    df = pd.read_csv(rel_path)
    target = df.columns[-1]

    config = create_auto_config(dataset=df, target=target, time_limit_s=1)

    # Ensure our configs are using the latest Ludwig schema
    ModelConfig.from_dict(config)

    # Check no features shuffled around
    assert len(input_features) == 2
    assert len(output_features) == 1

    # Check encoders are properly nested
    assert isinstance(config[INPUT_FEATURES][0][ENCODER], dict)
    assert isinstance(config[INPUT_FEATURES][1][ENCODER], dict)

    # Check automl default encoders are properly set
    assert config[INPUT_FEATURES][0][ENCODER][TYPE] == "bert"
    assert config[INPUT_FEATURES][1][ENCODER][TYPE] == "stacked_cnn"


@pytest.mark.slow
@pytest.mark.distributed
@pytest.mark.parametrize("time_budget", [200, 1], ids=["high", "low"])
def test_train_with_config(time_budget, test_data_tabular_large, ray_cluster_2cpu, tmpdir):
    _run_train_with_config(time_budget, test_data_tabular_large, tmpdir)


@pytest.mark.distributed
def test_auto_train(test_data_tabular_large, ray_cluster_2cpu, tmpdir):
    _, ofeatures, dataset_csv = test_data_tabular_large
    results = auto_train(
        dataset=dataset_csv,
        target=ofeatures[0][NAME],
        time_limit_s=120,
        user_config={"hyperopt": {"executor": {"num_samples": 2}}},
    )

    analysis = results.experiment_analysis
    for trial in analysis.trials:
        assert trial.status != Trial.ERROR, f"Error in trial {trial}"


@pytest.mark.slow
@pytest.mark.parametrize("fs_protocol,bucket", [private_param(("s3", "ludwig-tests"))], ids=["s3"])
def test_train_with_config_remote(fs_protocol, bucket, test_data_tabular_large, ray_cluster_2cpu):
    backend = {
        "type": "local",
        "credentials": {
            "artifacts": minio_test_creds(),
        },
    }

    with remote_tmpdir(fs_protocol, bucket) as tmpdir:
        _run_train_with_config(200, test_data_tabular_large, tmpdir, backend=backend)


def _run_train_with_config(time_budget, test_data, tmpdir, **kwargs):
    input_features, output_features, dataset_csv = test_data
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "trainer": {"epochs": 2},
        "hyperopt": {
            "search_alg": {
                "type": "hyperopt",
                "random_state_seed": 42,
            },
            "executor": {
                "type": "ray",
                "time_budget_s": time_budget,
                "cpu_resources_per_trial": 1,
                "scheduler": {
                    "type": "async_hyperband",
                    "max_t": time_budget,
                    "time_attr": "time_total_s",
                    "grace_period": min(72, time_budget),
                    "reduction_factor": 5,
                },
            },
            "parameters": {
                "trainer.batch_size": {
                    "space": "choice",
                    "categories": [64, 128, 256],
                },
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
            },
        },
    }

    fn = RayTuneExecutor._evaluate_best_model
    with mock.patch("ludwig.hyperopt.execution.RayTuneExecutor._evaluate_best_model") as mock_fn:
        # We need to check that _evaluate_best_model is called when the time_budget is low
        # as this code path should be triggered when the trial was early stopped
        mock_fn.side_effect = fn

        outdir = os.path.join(tmpdir, "output")
        results = train_with_config(dataset_csv, config, output_directory=outdir, **kwargs)
        try:
            best_model = results.best_model
        except ValueError:
            # ValueError is raised when best_model can't be found. This typically
            # happens when the time_budget is low and the trial is stopped early,
            # resulting in no evaluations happening (and no scores being reported back to RayTune).
            # So RayTune has no way of determining what the best model is.
            best_model = None

        if time_budget > 1:
            assert isinstance(best_model, LudwigModel)
            assert best_model.config_obj.trainer.early_stop == -1
            # assert mock_fn.call_count == 1
        else:
            assert best_model is None
            assert mock_fn.call_count == 0
