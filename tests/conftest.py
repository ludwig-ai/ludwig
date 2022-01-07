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
import uuid

import pytest

from ludwig.hyperopt.run import hyperopt
from tests.integration_tests.utils import category_feature, generate_data, text_feature


@pytest.fixture()
def csv_filename():
    """Yields a csv filename for holding temporary data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_filename = os.path.join(tmpdir, uuid.uuid4().hex[:10].upper() + ".csv")
        yield csv_filename


@pytest.fixture()
def yaml_filename():
    """Yields a yaml filename for holding a temporary config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_filename = os.path.join(tmpdir, "model_def_" + uuid.uuid4().hex[:10].upper() + ".yaml")
        yield yaml_filename


@pytest.fixture(scope="module")
def hyperopt_results():
    """This function generates hyperopt results."""
    input_features = [
        text_feature(name="utterance", cell_type="lstm", reduce_output="sum"),
        category_feature(vocab_size=2, reduce_input="sum"),
    ]

    output_features = [category_feature(vocab_size=2, reduce_input="sum")]

    csv_filename = uuid.uuid4().hex[:10].upper() + ".csv"
    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "num_fc_layers": 2},
        "training": {"epochs": 2, "learning_rate": 0.001},
    }

    output_feature_name = output_features[0]["name"]

    hyperopt_configs = {
        "parameters": {
            "training.learning_rate": {
                "type": "float",
                "low": 0.0001,
                "high": 0.01,
                "space": "log",
                "steps": 3,
            },
            output_feature_name + ".fc_size": {"type": "int", "low": 32, "high": 256, "steps": 5},
            output_feature_name + ".num_fc_layers": {"type": "int", "low": 1, "high": 5, "space": "linear", "steps": 4},
        },
        "goal": "minimize",
        "output_feature": output_feature_name,
        "validation_metrics": "loss",
        "executor": {"type": "serial"},
        "sampler": {"type": "random", "num_samples": 2},
    }

    # add hyperopt parameter space to the config
    config["hyperopt"] = hyperopt_configs

    hyperopt(config, dataset=rel_path, output_directory="results")

    return os.path.abspath("results")
