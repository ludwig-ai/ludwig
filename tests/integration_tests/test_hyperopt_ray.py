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
import logging
import os.path

import pytest

import ray

from ludwig.hyperopt.execution import get_build_hyperopt_executor
from ludwig.hyperopt.run import hyperopt
from ludwig.hyperopt.sampling import (get_build_hyperopt_sampler)
from ludwig.hyperopt.utils import update_hyperopt_params_with_defaults
from ludwig.utils.defaults import merge_with_defaults, ACCURACY
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import spawn
from tests.integration_tests.utils import text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

HYPEROPT_CONFIG = {
    "parameters": {
        "training.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
        "combiner.num_fc_layers": {
            "space": "randint",
            "lower": 2,
            "upper": 6
        },
        "utterance.cell_type": {
            "space": "grid_search",
            "values": ["rnn", "gru"]
        },
        "utterance.bidirectional": {
            "space": "choice",
            "categories": [True, False]
        },
        "utterance.fc_layers": {
            "space": "choice",
            "categories": [
                [{"fc_size": 512}, {"fc_size": 256}],
                [{"fc_size": 512}],
                [{"fc_size": 256}],
            ]
        }
    },
    "goal": "minimize"
}


SAMPLERS = [
    {"type": "ray"},
    {"type": "ray", "num_samples": 2},
    {
        "type": "ray",
        "search_alg": {
            "type": "bohb"
        },
        "scheduler": {
            "type": "hb_bohb",
            "time_attr": "training_iteration",
            "reduction_factor": 4,
        },
        "num_samples": 3
    },
]

EXECUTORS = [
    {"type": "ray"},
]


@pytest.fixture
def ray_start_4_cpus():
    address_info = ray.init(num_cpus=4)
    try:
        yield address_info
    finally:
        ray.shutdown()


@spawn
def run_hyperopt_executor(sampler, executor, csv_filename,
                          validate_output_feature=False,
                          validation_metric=None):
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum")]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 2, "learning_rate": 0.001},
        "hyperopt": {
            **HYPEROPT_CONFIG,
            "executor": executor,
            "sampler": sampler,
        },
    }

    config = merge_with_defaults(config)

    hyperopt_config = config["hyperopt"]

    if validate_output_feature:
        hyperopt_config['output_feature'] = output_features[0]['name']
    if validation_metric:
        hyperopt_config['validation_metric'] = validation_metric

    update_hyperopt_params_with_defaults(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    if sampler.get("search_alg", {}).get("type", "") == 'bohb':
        # bohb does not support grid_search search space
        del parameters['utterance.cell_type']

    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]

    hyperopt_sampler = get_build_hyperopt_sampler(
        sampler["type"])(goal, parameters, **sampler)

    hyperopt_executor = get_build_hyperopt_executor(executor["type"])(
        hyperopt_sampler, output_feature, metric, split, **executor)

    hyperopt_executor.execute(config, dataset=rel_path)


@pytest.mark.distributed
@pytest.mark.parametrize('sampler', SAMPLERS)
@pytest.mark.parametrize('executor', EXECUTORS)
def test_hyperopt_executor(sampler, executor, csv_filename, ray_start_4_cpus):
    run_hyperopt_executor(sampler, executor, csv_filename)


@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename, ray_start_4_cpus):
    run_hyperopt_executor({"type": "ray", "num_samples": 2},
                          {"type": "ray"},
                          csv_filename,
                          validate_output_feature=True,
                          validation_metric=ACCURACY)


@pytest.mark.distributed
def test_hyperopt_run_hyperopt(csv_filename, ray_start_4_cpus):
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum")]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 2, "learning_rate": 0.001}
    }

    output_feature_name = output_features[0]['name']

    hyperopt_configs = {
        "parameters": {
            "training.learning_rate": {
                "space": "loguniform",
                "lower": 0.001,
                "upper": 0.1,
            },
            output_feature_name + ".fc_size": {
                "space": "randint",
                "lower": 32,
                "upper": 256
            },
            output_feature_name + ".num_fc_layers": {
                "space": "randint",
                "lower": 2,
                "upper": 6
            }
        },
        "goal": "minimize",
        'output_feature': output_feature_name,
        'validation_metrics': 'loss',
        'executor': {'type': 'ray'},
        'sampler': {'type': 'ray', 'num_samples': 2}
    }

    # add hyperopt parameter space to the config
    config['hyperopt'] = hyperopt_configs
    run_hyperopt(config, rel_path)


@spawn
def run_hyperopt(config, rel_path):
    hyperopt_results = hyperopt(
        config,
        dataset=rel_path,
        output_directory='results_hyperopt'
    )

    # check for return results
    assert isinstance(hyperopt_results, list)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(
        os.path.join('results_hyperopt', 'hyperopt_statistics.json')
    )
