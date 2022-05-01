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
from typing import Optional

import pytest
import ray
import torch

from ludwig.constants import ACCURACY, TRAINER
from ludwig.hyperopt.execution import get_build_hyperopt_executor
from ludwig.hyperopt.results import HyperoptResults
from ludwig.hyperopt.run import hyperopt, update_hyperopt_params_with_defaults
from ludwig.hyperopt.sampling import get_build_hyperopt_sampler
from ludwig.utils.defaults import merge_with_defaults
from tests.integration_tests.utils import category_feature, generate_data, text_feature

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
        "combiner.fc_layers": {
            "space": "choice",
            "categories": [[{"output_size": 64}, {"output_size": 32}], [{"output_size": 64}], [{"output_size": 32}]],
        },
        "utterance.cell_type": {"space": "grid_search", "values": ["rnn", "gru"]},
        "utterance.bidirectional": {"space": "choice", "categories": [True, False]},
    },
    "goal": "minimize",
}

SAMPLERS = [{"type": "ray", "num_samples": 2}]

EXECUTORS = [
    {"type": "ray"},
]


@contextlib.contextmanager
def ray_start(num_cpus: Optional[int] = None, num_gpus: Optional[int] = None):
    res = ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        include_dashboard=False,
        object_store_memory=150 * 1024 * 1024,
    )
    try:
        yield res
    finally:
        ray.shutdown()


@pytest.mark.distributed
@pytest.mark.parametrize("sampler", SAMPLERS)
@pytest.mark.parametrize("executor", EXECUTORS)
def test_hyperopt_executor(sampler, executor, csv_filename, validate_output_feature=False, validation_metric=None):
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

    config = merge_with_defaults(config)

    hyperopt_config = HYPEROPT_CONFIG.copy()

    if validate_output_feature:
        hyperopt_config["output_feature"] = output_features[0]["name"]
    if validation_metric:
        hyperopt_config["validation_metric"] = validation_metric

    update_hyperopt_params_with_defaults(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]

    hyperopt_sampler = get_build_hyperopt_sampler(sampler["type"])(goal, parameters, **sampler)

    gpus = [i for i in range(torch.cuda.device_count())]
    with ray_start(num_gpus=len(gpus)):
        hyperopt_executor = get_build_hyperopt_executor(executor["type"])(
            hyperopt_sampler, output_feature, metric, split, **executor
        )

        hyperopt_executor.execute(config, dataset=rel_path)


@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename):
    test_hyperopt_executor(
        {"type": "ray", "num_samples": 2},
        {"type": "ray"},
        csv_filename,
        validate_output_feature=True,
        validation_metric=ACCURACY,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("samplers", SAMPLERS)
def test_hyperopt_run_hyperopt(csv_filename, samplers):
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
                "lower": 0.0001,
                "upper": 0.01,
                "space": "loguniform",
            },
            output_feature_name
            + ".fc_layers": {
                "space": "choice",
                "categories": [
                    [{"output_size": 64}, {"output_size": 32}],
                    [{"output_size": 64}],
                    [{"output_size": 32}],
                ],
            },
            output_feature_name + ".output_size": {"space": "choice", "categories": [16, 21, 26, 31, 36]},
            output_feature_name + ".num_fc_layers": {"space": "randint", "lower": 1, "upper": 6},
        },
        "goal": "minimize",
        "output_feature": output_feature_name,
        "validation_metrics": "loss",
        "executor": {"type": "ray"},
        "sampler": {"type": samplers["type"], "num_samples": 2},
    }

    # add hyperopt parameter space to the config
    config["hyperopt"] = hyperopt_configs

    with ray_start():
        hyperopt_results = hyperopt(config, dataset=rel_path, output_directory="results_hyperopt")

    # check for return results
    assert isinstance(hyperopt_results, HyperoptResults)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(os.path.join("results_hyperopt", "hyperopt_statistics.json"))

    if os.path.isfile(os.path.join("results_hyperopt", "hyperopt_statistics.json")):
        os.remove(os.path.join("results_hyperopt", "hyperopt_statistics.json"))


# TODO: Need to confirm if this is still needed with a RayTune only executors
@pytest.mark.distributed
@pytest.mark.skip("This is Serial only test?")
def test_hyperopt_executor_get_metric_score():
    executor = EXECUTORS[0]
    output_feature = "of_name"
    split = "validation"

    train_stats = {
        "training": {
            output_feature: {
                "loss": [0.58760345, 1.5066891],
                "accuracy": [0.6666667, 0.33333334],
                "hits_at_k": [1.0, 1.0],
            },
            "combined": {"loss": [0.58760345, 1.5066891]},
        },
        "validation": {
            output_feature: {"loss": [0.30233705, 2.6505466], "accuracy": [1.0, 0.0], "hits_at_k": [1.0, 1.0]},
            "combined": {"loss": [0.30233705, 2.6505466]},
        },
        "test": {
            output_feature: {"loss": [1.0876318, 1.4353828], "accuracy": [0.7, 0.5], "hits_at_k": [1.0, 1.0]},
            "combined": {"loss": [1.0876318, 1.4353828]},
        },
    }

    metric = "loss"
    hyperopt_executor = get_build_hyperopt_executor(executor["type"])(None, output_feature, metric, split, **executor)
    score = hyperopt_executor.get_metric_score(train_stats)
    assert score == 0.30233705

    metric = "accuracy"
    hyperopt_executor = get_build_hyperopt_executor(executor["type"])(None, output_feature, metric, split, **executor)
    score = hyperopt_executor.get_metric_score(train_stats)
    assert score == 1.0
