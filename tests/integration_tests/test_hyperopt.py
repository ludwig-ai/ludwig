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

import logging

from ludwig.hyperopt import update_hyperopt_params_with_defaults
from ludwig.utils.defaults import merge_with_defaults
from ludwig.utils.hyperopt_utils import (get_build_hyperopt_executor,
                                         get_build_hyperopt_strategy)
from tests.integration_tests.utils import category_feature
from tests.integration_tests.utils import generate_data
from tests.integration_tests.utils import text_feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("ludwig").setLevel(logging.INFO)

HYPEROPT_CONFIG = {
    "strategy": {
        "type": "pySOT",
        "num_samples": 10
    },
    "parameters": {
        "training.learning_rate": {
            "type": "real",
            "range": (0.0001, 0.1),
            "space": "log",
        },
        "combiner.num_fc_layers": {
            "type": "int",
            "range": (1, 4),
            "space": "linear",
        },
        "utterance.cell_type": {
            "type": "cat",
            "values": ["rnn", "gru", "lstm"]
        }
    },
    "goal": "minimize"
}

EXECUTORS = [
    {"type": "serial"},
    {"type": "parallel", "num_workers": 4},
    {"type": "fiber", "num_workers": 4},
]


def test_hyperopt_executor(csv_filename):
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum")]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    rel_path = generate_data(input_features, output_features, csv_filename)

    model_definition = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 2, "learning_rate": 0.001}
    }

    model_definition = merge_with_defaults(model_definition)

    input_features = model_definition["input_features"]
    output_features = model_definition["output_features"]
    hyperopt_config = HYPEROPT_CONFIG

    update_hyperopt_params_with_defaults(hyperopt_config)

    strategy = hyperopt_config["strategy"]
    parameters = hyperopt_config["parameters"]
    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]

    hyperopt_strategy = get_build_hyperopt_strategy(
        strategy["type"])(goal, parameters, **strategy)

    for executor in EXECUTORS:
        hyperopt_executor = get_build_hyperopt_executor(executor["type"])(
            hyperopt_strategy, output_feature, metric, split, **executor)

        hyperopt_executor.execute(model_definition, data_csv=rel_path)
