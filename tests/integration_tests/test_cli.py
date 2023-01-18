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

import json
import os
import os.path
import pathlib
import shutil
import subprocess
from typing import List, Set

import pytest
import yaml

from ludwig.constants import BATCH_SIZE, COMBINER, INPUT_FEATURES, NAME, OUTPUT_FEATURES, PREPROCESSING, TRAINER
from ludwig.types import FeatureConfigDict
from ludwig.utils.data_utils import load_yaml
from tests.integration_tests.utils import category_feature, generate_data, number_feature, sequence_feature


def _run_commands(commands, **ludwig_kwargs):
    for arg_name, value in ludwig_kwargs.items():
        commands += ["--" + arg_name, value]
    cmdline = " ".join(commands)
    print(cmdline)
    completed_process = subprocess.run(cmdline, shell=True, stdout=subprocess.PIPE, env=os.environ.copy())
    assert completed_process.returncode == 0

    return completed_process


def _run_ludwig(command, **ludwig_kwargs):
    commands = ["ludwig", command]
    return _run_commands(commands, **ludwig_kwargs)


def _run_ludwig_horovod(command, **ludwig_kwargs):
    commands = ["horovodrun", "-np", "2", "ludwig", command]
    return _run_commands(commands, **ludwig_kwargs)


def _prepare_data(csv_filename, config_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    dataset_filename = generate_data(input_features, output_features, csv_filename)

    # generate config file
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    return dataset_filename


def _prepare_hyperopt_data(csv_filename, config_filename):
    # Single sequence input, single category output
    input_features = [sequence_feature(encoder={"reduce_output": "sum"})]
    output_features = [category_feature(decoder={"vocab_size": 2}, reduce_input="sum")]

    # Generate test data
    dataset_filename = generate_data(input_features, output_features, csv_filename)

    # generate config file
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 4},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
        "hyperopt": {
            "parameters": {
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.0001,
                    "upper": 0.01,
                }
            },
            "goal": "minimize",
            "output_feature": output_features[0]["name"],
            "validation_metrics": "loss",
            "executor": {
                "type": "ray",
                "num_samples": 2,
            },
            "search_alg": {
                "type": "variant_generator",
            },
        },
    }

    with open(config_filename, "w") as f:
        yaml.dump(config, f)

    return dataset_filename


def test_train_cli_dataset(tmpdir, csv_filename):
    """Test training using `ludwig train --dataset`."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))


def test_train_cli_gpu_memory_limit(tmpdir, csv_filename):
    """Test training using `ludwig train --dataset --gpu_memory_limit`."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig(
        "train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir), gpu_memory_limit="0.5"
    )


def test_train_cli_training_set(tmpdir, csv_filename):
    """Test training using `ludwig train --training_set`."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    validation_filename = shutil.copyfile(dataset_filename, os.path.join(tmpdir, "validation.csv"))
    test_filename = shutil.copyfile(dataset_filename, os.path.join(tmpdir, "test.csv"))
    _run_ludwig(
        "train",
        training_set=dataset_filename,
        validation_set=validation_filename,
        test_set=test_filename,
        config=config_filename,
        output_directory=str(tmpdir),
    )


@pytest.mark.distributed
def test_train_cli_horovod(tmpdir, csv_filename):
    """Test training using `horovodrun -np 2 ludwig train --dataset`."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig_horovod(
        "train",
        dataset=dataset_filename,
        config=config_filename,
        output_directory=str(tmpdir),
        experiment_name="horovod_experiment",
    )

    # Check that `model_load_path` works correctly
    _run_ludwig_horovod(
        "train",
        dataset=dataset_filename,
        config=config_filename,
        output_directory=str(tmpdir),
        model_load_path=os.path.join(tmpdir, "horovod_experiment_run", "model"),
    )


@pytest.mark.skip(reason="Issue #1451: Use torchscript.")
def test_export_neuropod_cli(tmpdir, csv_filename):
    """Test exporting Ludwig model to neuropod format."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=tmpdir)
    _run_ludwig(
        "export_neuropod",
        model=os.path.join(tmpdir, "experiment_run", "model"),
        output_path=os.path.join(tmpdir, "neuropod"),
    )


def test_export_torchscript_cli(tmpdir, csv_filename):
    """Test exporting Ludwig model to torchscript format."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))
    _run_ludwig(
        "export_torchscript",
        model_path=os.path.join(tmpdir, "experiment_run", "model"),
        output_path=os.path.join(tmpdir, "torchscript"),
    )


def test_export_mlflow_cli(tmpdir, csv_filename):
    """Test export_mlflow cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))
    _run_ludwig(
        "export_mlflow",
        model_path=os.path.join(tmpdir, "experiment_run", "model"),
        output_path=os.path.join(tmpdir, "data/results/mlflow"),
    )


def test_experiment_cli(tmpdir, csv_filename):
    """Test experiment cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("experiment", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))


def test_predict_cli(tmpdir, csv_filename):
    """Test predict cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))
    _run_ludwig(
        "predict",
        dataset=dataset_filename,
        model=os.path.join(tmpdir, "experiment_run", "model"),
        output_directory=os.path.join(tmpdir, "predictions"),
    )


def test_evaluate_cli(tmpdir, csv_filename):
    """Test evaluate cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))
    _run_ludwig(
        "evaluate",
        dataset=dataset_filename,
        model=os.path.join(tmpdir, "experiment_run", "model"),
        output_directory=os.path.join(tmpdir, "predictions"),
    )


@pytest.mark.distributed
def test_hyperopt_cli(tmpdir, csv_filename):
    """Test hyperopt cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_hyperopt_data(csv_filename, config_filename)
    _run_ludwig("hyperopt", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))


def test_visualize_cli(tmpdir, csv_filename):
    """Test Ludwig 'visualize' cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))
    _run_ludwig(
        "visualize",
        visualization="learning_curves",
        model_names="run",
        training_statistics=os.path.join(tmpdir, "experiment_run", "training_statistics.json"),
        output_directory=os.path.join(tmpdir, "visualizations"),
    )


def test_collect_summary_activations_weights_cli(tmpdir, csv_filename):
    """Test collect_summary cli."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("train", dataset=dataset_filename, config=config_filename, output_directory=str(tmpdir))
    assert _run_ludwig("collect_summary", model=os.path.join(tmpdir, "experiment_run", "model"))


def test_synthesize_dataset_cli(tmpdir, csv_filename):
    """Test synthesize_data cli."""
    # test depends on default setting of --dataset_size
    # if this parameter is specified, _run_ludwig fails when
    # attempting to build the cli parameter structure
    _run_ludwig(
        "synthesize_dataset",
        output_path=os.path.join(tmpdir, csv_filename),
        features="'[ \
                {name: text, type: text}, \
                {name: category, type: category}, \
                {name: number, type: number}, \
                {name: binary, type: binary}, \
                {name: set, type: set}, \
                {name: bag, type: bag}, \
                {name: sequence, type: sequence}, \
                {name: timeseries, type: timeseries}, \
                {name: date, type: date}, \
                {name: h3, type: h3}, \
                {name: vector, type: vector}, \
                {name: audio, type: audio}, \
                {name: image, type: image} \
            ]'",
    )


def test_preprocess_cli(tmpdir, csv_filename):
    """Test preprocess `ludwig preprocess."""
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)
    _run_ludwig("preprocess", dataset=dataset_filename, preprocessing_config=config_filename)


@pytest.mark.parametrize("second_seed_offset", [0, 1])
@pytest.mark.parametrize("random_seed", [1919, 31])
@pytest.mark.parametrize("type_of_run", ["train", "experiment"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("local", id="local"),
        pytest.param("horovod", id="horovod", marks=pytest.mark.distributed),
    ],
)
def test_reproducible_cli_runs(
    backend: str, type_of_run: str, random_seed: int, second_seed_offset: int, csv_filename: str, tmpdir: pathlib.Path
) -> None:
    """
    Test for reproducible training using `ludwig experiment|train --dataset`.
    Args:
        backend (str): backend to use
        type_of_run(str): type of run, either train or experiment
        csv_filename(str): file path of dataset to use
        random_seed(int): random seed integer to use for test
        second_seed_offset(int): zero to use same random seed for second test, non-zero to use a different
            seed for the second run.
        tmpdir (pathlib.Path): temporary directory path

    Returns: None
    """
    config_filename = os.path.join(tmpdir, "config.yaml")
    dataset_filename = _prepare_data(csv_filename, config_filename)

    if backend == "local":
        command_to_run = _run_ludwig
    else:
        command_to_run = _run_ludwig_horovod

    # run first model
    command_to_run(
        type_of_run,
        dataset=dataset_filename,
        config=config_filename,
        output_directory=str(tmpdir),
        skip_save_processed_input="",  # skip saving preprocessed inputs for reproducibility
        experiment_name="reproducible",
        model_name="run1",
        random_seed=str(random_seed),
    )

    # run second model with same seed
    command_to_run(
        type_of_run,
        dataset=dataset_filename,
        config=config_filename,
        output_directory=str(tmpdir),
        skip_save_processed_input="",  # skip saving preprocessed inputs for reproducibility
        experiment_name="reproducible",
        model_name="run2",
        random_seed=str(random_seed + second_seed_offset),
    )

    # retrieve training statistics and compare
    with open(os.path.join(tmpdir, "reproducible_run1", "training_statistics.json")) as f:
        training1 = json.load(f)
    with open(os.path.join(tmpdir, "reproducible_run2", "training_statistics.json")) as f:
        training2 = json.load(f)

    if second_seed_offset == 0:
        # same seeds should result in same output
        assert training1 == training2
    else:
        # non-zero second_seed_offset uses different seeds and should result in different output
        assert training1 != training2

    # if type_of_run is experiment check test statistics and compare
    if type_of_run == "experiment":
        with open(os.path.join(tmpdir, "reproducible_run1", "test_statistics.json")) as f:
            test1 = json.load(f)
        with open(os.path.join(tmpdir, "reproducible_run2", "test_statistics.json")) as f:
            test2 = json.load(f)

        if second_seed_offset == 0:
            # same seeds should result in same output
            assert test1 == test2
        else:
            # non-zero second_seed_offset uses different seeds and should result in different output
            assert test1 != test2


@pytest.mark.distributed
def test_init_config(tmpdir):
    """Test initializing a config from a dataset and a target."""
    input_features = [
        number_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [category_feature(decoder={"vocab_size": 3})]
    dataset_csv = generate_data(input_features, output_features, os.path.join(tmpdir, "dataset.csv"), num_examples=100)
    output_config_path = os.path.join(tmpdir, "config.yaml")

    _run_ludwig("init_config", dataset=dataset_csv, target=output_features[0][NAME], output=output_config_path)

    config = load_yaml(output_config_path)

    def to_name_set(features: List[FeatureConfigDict]) -> Set[str]:
        return {feature[NAME] for feature in features}

    assert to_name_set(config[INPUT_FEATURES]) == to_name_set(input_features)
    assert to_name_set(config[OUTPUT_FEATURES]) == to_name_set(output_features)


def test_render_config(tmpdir):
    """Test rendering a full config from a partial user config."""
    user_config_path = os.path.join(tmpdir, "config.yaml")
    input_features = [
        number_feature(),
        number_feature(),
        category_feature(encoder={"vocab_size": 3}),
        category_feature(encoder={"vocab_size": 3}),
    ]
    output_features = [category_feature(decoder={"vocab_size": 3})]

    user_config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
    }

    with open(user_config_path, "w") as f:
        yaml.dump(user_config, f)

    output_config_path = os.path.join(tmpdir, "rendered.yaml")
    _run_ludwig("render_config", config=user_config_path, output=output_config_path)

    rendered_config = load_yaml(output_config_path)
    assert len(rendered_config[INPUT_FEATURES]) == len(user_config[INPUT_FEATURES])
    assert len(rendered_config[OUTPUT_FEATURES]) == len(user_config[OUTPUT_FEATURES])
    assert TRAINER in rendered_config
    assert COMBINER in rendered_config
    assert PREPROCESSING in rendered_config
