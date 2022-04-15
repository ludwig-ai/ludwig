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
import logging
import os.path

import mlflow
import pandas as pd
import pytest
import ray
from mlflow.tracking import MlflowClient

from ludwig.constants import ACCURACY, TRAINER
from ludwig.contribs import MlflowCallback
from ludwig.hyperopt.execution import get_build_hyperopt_executor
from ludwig.hyperopt.results import RayTuneResults
from ludwig.hyperopt.run import hyperopt, update_hyperopt_params_with_defaults
from ludwig.hyperopt.sampling import get_build_hyperopt_sampler
from ludwig.utils.defaults import merge_with_defaults
from tests.integration_tests.utils import category_feature, generate_data, spawn, text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

HYPEROPT_CONFIG = {
    "parameters": {
        "trainer.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.num_fc_layers": {"space": "randint", "lower": 2, "upper": 6},
        "utterance.cell_type": {"space": "grid_search", "values": ["rnn", "gru"]},
        "utterance.bidirectional": {"space": "choice", "categories": [True, False]},
        "utterance.fc_layers": {
            "space": "choice",
            "categories": [
                [{"output_size": 64}, {"output_size": 32}],
                [{"output_size": 64}],
                [{"output_size": 32}],
            ],
        },
    },
    "goal": "minimize",
}


SAMPLERS = [
    {"type": "ray"},
    {"type": "ray", "num_samples": 2},
    {
        "type": "ray",
        "search_alg": {"type": "bohb"},
        "scheduler": {
            "type": "hb_bohb",
            "time_attr": "training_iteration",
            "reduction_factor": 4,
        },
        "num_samples": 3,
    },
]

EXECUTORS = [
    {"type": "ray"},
]


def _get_config(sampler, executor):
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum"),
    ]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    return {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
        "hyperopt": {
            **HYPEROPT_CONFIG,
            "executor": executor,
            "sampler": sampler,
        },
    }


@contextlib.contextmanager
def ray_start_4_cpus():
    res = ray.init(
        num_cpus=4,
        include_dashboard=False,
        object_store_memory=150 * 1024 * 1024,
    )
    try:
        yield res
    finally:
        ray.shutdown()


@spawn
def run_hyperopt_executor(
    sampler,
    executor,
    csv_filename,
    validate_output_feature=False,
    validation_metric=None,
    use_split=True,
):
    config = _get_config(sampler, executor)
    rel_path = generate_data(config["input_features"], config["output_features"], csv_filename)

    if not use_split:
        df = pd.read_csv(rel_path)
        df["split"] = 0
        df.to_csv(rel_path)

    config = merge_with_defaults(config)

    hyperopt_config = config["hyperopt"]

    if validate_output_feature:
        hyperopt_config["output_feature"] = config["output_features"][0]["name"]
    if validation_metric:
        hyperopt_config["validation_metric"] = validation_metric

    update_hyperopt_params_with_defaults(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    if sampler.get("search_alg", {}).get("type", "") == "bohb":
        # bohb does not support grid_search search space
        del parameters["utterance.cell_type"]

    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]

    hyperopt_sampler = get_build_hyperopt_sampler(sampler["type"])(goal, parameters, **sampler)

    hyperopt_executor = get_build_hyperopt_executor(executor["type"])(
        hyperopt_sampler, output_feature, metric, split, **executor
    )

    hyperopt_executor.execute(
        config,
        dataset=rel_path,
        backend="local",
    )


@pytest.mark.distributed
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("executor", EXECUTORS)
def test_hyperopt_executor(sampler, executor, csv_filename):
    with ray_start_4_cpus():
        run_hyperopt_executor(sampler, executor, csv_filename)


@pytest.mark.distributed
@pytest.mark.parametrize("use_split", [True, False], ids=["split", "no_split"])
def test_hyperopt_executor_with_metric(use_split, csv_filename):
    with ray_start_4_cpus():
        run_hyperopt_executor(
            {"type": "ray", "num_samples": 2},
            {"type": "ray"},
            csv_filename,
            validate_output_feature=True,
            validation_metric=ACCURACY,
            use_split=use_split,
        )


@pytest.mark.distributed
def test_hyperopt_run_hyperopt(csv_filename):
    with ray_start_4_cpus():
        input_features = [
            text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
            category_feature(vocab_size=2, reduce_input="sum"),
        ]

        output_features = [category_feature(vocab_size=2, reduce_input="sum")]

        rel_path = generate_data(input_features, output_features, csv_filename)

        config = {
            "input_features": input_features,
            "output_features": output_features,
            "combiner": {"type": "concat", "num_fc_layers": 2},
            TRAINER: {"epochs": 2, "learning_rate": 0.001},
        }

        output_feature_name = output_features[0]["name"]

        hyperopt_configs = {
            "parameters": {
                "trainer.learning_rate": {
                    "space": "loguniform",
                    "lower": 0.001,
                    "upper": 0.1,
                },
                output_feature_name + ".output_size": {"space": "randint", "lower": 32, "upper": 64},
                output_feature_name + ".num_fc_layers": {"space": "randint", "lower": 2, "upper": 6},
            },
            "goal": "minimize",
            "output_feature": output_feature_name,
            "validation_metrics": "loss",
            "executor": {"type": "ray"},
            "sampler": {"type": "ray", "num_samples": 2},
        }

        # add hyperopt parameter space to the config
        config["hyperopt"] = hyperopt_configs
        run_hyperopt(config, rel_path)


@pytest.mark.distributed
def test_hyperopt_ray_mlflow(csv_filename, tmpdir):
    with ray_start_4_cpus():
        mlflow_uri = f"file://{tmpdir}/mlruns"
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient(tracking_uri=mlflow_uri)

        num_samples = 2
        config = _get_config({"type": "ray", "num_samples": num_samples}, {"type": "ray"})

        rel_path = generate_data(config["input_features"], config["output_features"], csv_filename)

        exp_name = "mlflow_test"
        run_hyperopt(config, rel_path, experiment_name=exp_name, callbacks=[MlflowCallback(mlflow_uri)])

        experiment = client.get_experiment_by_name(exp_name)
        assert experiment is not None

        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) > 0

        for run in runs:
            artifacts = [f.path for f in client.list_artifacts(run.info.run_id, "")]
            assert "config.yaml" in artifacts
            assert "model" in artifacts


@spawn
def run_hyperopt(
    config,
    rel_path,
    experiment_name="ray_hyperopt",
    callbacks=None,
):
    hyperopt_results = hyperopt(
        config,
        dataset=rel_path,
        output_directory="results_hyperopt",
        experiment_name=experiment_name,
        callbacks=callbacks,
    )

    # check for return results
    assert isinstance(hyperopt_results, RayTuneResults)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(os.path.join("results_hyperopt", "hyperopt_statistics.json"))
