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
import json
import os.path
from typing import Any, Dict, Optional, Tuple, Union

import pytest
import torch
from packaging import version

from ludwig.constants import (
    ACCURACY,
    CATEGORY,
    COMBINER,
    DECODER,
    ENCODER,
    HYPEROPT,
    INPUT_FEATURES,
    NAME,
    OUTPUT_FEATURES,
    RAY,
    TEXT,
    TRAINER,
    TYPE,
)
from ludwig.globals import HYPEROPT_STATISTICS_FILE_NAME
from ludwig.hyperopt.results import HyperoptResults, RayTuneResults
from ludwig.hyperopt.run import hyperopt, update_hyperopt_params_with_defaults
from ludwig.utils.config_utils import get_feature_type_parameter_values_from_section
from ludwig.utils.defaults import merge_with_defaults
from tests.integration_tests.utils import category_feature, generate_data, text_feature

try:
    import ray

    from ludwig.hyperopt.execution import get_build_hyperopt_executor

    _ray113 = version.parse(ray.__version__) > version.parse("1.13")

except ImportError:
    ray = None
    _ray113 = None


RANDOM_SEARCH_SIZE = 4

HYPEROPT_CONFIG = {
    "parameters": {
        # using only float parameter as common in all search algorithms
        "trainer.learning_rate": {
            "space": "loguniform",
            "lower": 0.001,
            "upper": 0.1,
        },
    },
    "goal": "minimize",
    "executor": {"type": "ray", "num_samples": 2, "scheduler": {"type": "fifo"}},
    "search_alg": {"type": "variant_generator"},
}

SEARCH_ALGS_FOR_TESTING = [
    # None,
    # "variant_generator",
    "random",
    "bohb",
    # "hyperopt",
    # "ax",
    # "bayesopt",
    # "blendsearch",
    # "cfo",
    # "dragonfly",
    # "hebo",
    # "skopt",
    # "optuna",
]

SCHEDULERS_FOR_TESTING = [
    "fifo",
    "asynchyperband",
    # "async_hyperband",
    # "median_stopping_rule",
    # "medianstopping",
    # "hyperband",
    # "hb_bohb",
    # "pbt",
    # "pb2",  commented out for now: https://github.com/ray-project/ray/issues/24815
    # "resource_changing",
]


def _setup_ludwig_config(dataset_fp: str) -> Tuple[Dict, str]:
    input_features = [
        text_feature(name="utterance", encoder={"reduce_output": "sum"}),
        category_feature(encoder={"vocab_size": 3}),
    ]

    output_features = [category_feature(decoder={"vocab_size": 3})]

    rel_path = generate_data(input_features, output_features, dataset_fp)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {"type": "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
    }

    config = merge_with_defaults(config)

    return config, rel_path


def _setup_ludwig_config_with_shared_params(dataset_fp: str) -> Tuple[Dict, Any]:
    input_features = [
        text_feature(name="title", encoder={"type": "parallel_cnn"}),
        text_feature(name="summary"),
        category_feature(encoder={"vocab_size": 3}),
        category_feature(encoder={"vocab_size": 3}),
    ]

    output_features = [category_feature(decoder={"vocab_size": 3})]

    rel_path = generate_data(input_features, output_features, dataset_fp)

    num_filters_search_space = [4, 8]
    embedding_size_search_space = [4, 8]
    reduce_input_search_space = ["sum", "mean"]

    # Add default parameters in hyperopt parameter search space
    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
        HYPEROPT: {
            "parameters": {
                "trainer.learning_rate": {"lower": 0.0001, "upper": 0.01, "space": "loguniform"},
                "defaults.text.encoder.num_filters": {"space": "choice", "categories": num_filters_search_space},
                "defaults.category.encoder.embedding_size": {
                    "space": "choice",
                    "categories": embedding_size_search_space,
                },
                "defaults.category.decoder.reduce_input": {
                    "space": "choice",
                    "categories": reduce_input_search_space,
                },
            },
            "goal": "minimize",
            "output_feature": output_features[0][NAME],
            "validation_metrics": "loss",
            "executor": {"type": "ray", "num_samples": RANDOM_SEARCH_SIZE},
            "search_alg": {"type": "variant_generator"},
        },
    }

    return config, rel_path, num_filters_search_space, embedding_size_search_space, reduce_input_search_space


def _get_trial_parameter_value(parameter_key: str, trial_row: str) -> Union[str, None]:
    """Returns the parameter value from the Ray trial row, which has slightly different column names depending on
    the version of Ray. Returns None if the parameter key is not found.

    TODO(#2176): There are different key name delimiters depending on Ray version. The delimiter in future versions of
    Ray (> 1.13) will be '/' instead of '.' Simplify this as Ray is upgraded.
    """
    if _ray113:
        return trial_row[f"config/{parameter_key}"]
    return trial_row[f"config.{parameter_key}"]


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


@pytest.fixture(scope="module")
def ray_cluster():
    gpus = [i for i in range(torch.cuda.device_count())]
    with ray_start(num_gpus=len(gpus)):
        yield


@pytest.mark.distributed
@pytest.mark.parametrize("search_alg", SEARCH_ALGS_FOR_TESTING)
def test_hyperopt_search_alg(
    search_alg, csv_filename, tmpdir, ray_cluster, validate_output_feature=False, validation_metric=None
):
    config, rel_path = _setup_ludwig_config(csv_filename)

    hyperopt_config = HYPEROPT_CONFIG.copy()

    # finalize hyperopt config settings
    if search_alg == "dragonfly":
        hyperopt_config["search_alg"] = {
            "type": search_alg,
            "domain": "euclidean",
            "optimizer": "random",
        }
    elif search_alg is None:
        hyperopt_config["search_alg"] = {}
    else:
        hyperopt_config["search_alg"] = {
            "type": search_alg,
        }

    if validate_output_feature:
        hyperopt_config["output_feature"] = config[OUTPUT_FEATURES][0][NAME]
    if validation_metric:
        hyperopt_config["validation_metric"] = validation_metric

    update_hyperopt_params_with_defaults(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]
    executor = hyperopt_config["executor"]
    search_alg = hyperopt_config["search_alg"]

    hyperopt_executor = get_build_hyperopt_executor(RAY)(
        parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
    )
    raytune_results = hyperopt_executor.execute(config, dataset=rel_path, output_directory=tmpdir)
    assert isinstance(raytune_results, RayTuneResults)


@pytest.mark.distributed
def test_hyperopt_executor_with_metric(csv_filename, tmpdir, ray_cluster):
    test_hyperopt_search_alg(
        "variant_generator",
        csv_filename,
        tmpdir,
        ray_cluster,
        validate_output_feature=True,
        validation_metric=ACCURACY,
    )


@pytest.mark.distributed
@pytest.mark.parametrize("scheduler", SCHEDULERS_FOR_TESTING)
def test_hyperopt_scheduler(
    scheduler, csv_filename, tmpdir, ray_cluster, validate_output_feature=False, validation_metric=None
):
    config, rel_path = _setup_ludwig_config(csv_filename)

    hyperopt_config = HYPEROPT_CONFIG.copy()

    # finalize hyperopt config settings
    if scheduler == "pb2":
        # setup scheduler hyperparam_bounds parameter
        min = hyperopt_config["parameters"]["trainer.learning_rate"]["lower"]
        max = hyperopt_config["parameters"]["trainer.learning_rate"]["upper"]
        hyperparam_bounds = {
            "trainer.learning_rate": [min, max],
        }
        hyperopt_config["executor"]["scheduler"] = {
            "type": scheduler,
            "hyperparam_bounds": hyperparam_bounds,
        }
    else:
        hyperopt_config["executor"]["scheduler"] = {
            "type": scheduler,
        }

    if validate_output_feature:
        hyperopt_config["output_feature"] = config[OUTPUT_FEATURES][0][NAME]
    if validation_metric:
        hyperopt_config["validation_metric"] = validation_metric

    update_hyperopt_params_with_defaults(hyperopt_config)

    parameters = hyperopt_config["parameters"]
    split = hyperopt_config["split"]
    output_feature = hyperopt_config["output_feature"]
    metric = hyperopt_config["metric"]
    goal = hyperopt_config["goal"]
    executor = hyperopt_config["executor"]
    search_alg = hyperopt_config["search_alg"]

    # TODO: Determine if we still need this if-then-else construct
    if search_alg["type"] in {""}:
        with pytest.raises(ImportError):
            get_build_hyperopt_executor(RAY)(
                parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
            )
    else:
        hyperopt_executor = get_build_hyperopt_executor(RAY)(
            parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
        )
        raytune_results = hyperopt_executor.execute(config, dataset=rel_path, output_directory=tmpdir)
        assert isinstance(raytune_results, RayTuneResults)


@pytest.mark.distributed
@pytest.mark.parametrize("search_space", ["random", "grid"])
def test_hyperopt_run_hyperopt(csv_filename, search_space, tmpdir, ray_cluster):
    input_features = [
        text_feature(name="utterance", encoder={"reduce_output": "sum"}),
        category_feature(encoder={"vocab_size": 3}),
    ]

    output_features = [category_feature(decoder={"vocab_size": 3})]

    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
    }

    output_feature_name = output_features[0][NAME]

    if search_space == "random":
        # random search will be size of num_samples
        search_parameters = {
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
        }
    else:
        # grid search space will be product each parameter size
        search_parameters = {
            "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.005, 0.01]},
            output_feature_name + ".output_size": {"space": "grid_search", "values": [16, 21, 36]},
            output_feature_name + ".num_fc_layers": {"space": "grid_search", "values": [1, 3, 6]},
        }

    hyperopt_configs = {
        "parameters": search_parameters,
        "goal": "minimize",
        "output_feature": output_feature_name,
        "validation_metrics": "loss",
        "executor": {"type": "ray", "num_samples": 1 if search_space == "grid" else RANDOM_SEARCH_SIZE},
        "search_alg": {"type": "variant_generator"},
    }

    # add hyperopt parameter space to the config
    config["hyperopt"] = hyperopt_configs

    hyperopt_results = hyperopt(config, dataset=rel_path, output_directory=tmpdir, experiment_name="test_hyperopt")
    if search_space == "random":
        assert hyperopt_results.experiment_analysis.results_df.shape[0] == RANDOM_SEARCH_SIZE
    else:
        # compute size of search space for grid search
        grid_search_size = 1
        for k, v in search_parameters.items():
            grid_search_size *= len(v["values"])
        assert hyperopt_results.experiment_analysis.results_df.shape[0] == grid_search_size

    # check for return results
    assert isinstance(hyperopt_results, HyperoptResults)

    # check for existence of the hyperopt statistics file
    assert os.path.isfile(os.path.join(tmpdir, "test_hyperopt", HYPEROPT_STATISTICS_FILE_NAME))


def _test_hyperopt_with_shared_params_trial_table(
    hyperopt_results_df, num_filters_search_space, embedding_size_search_space, reduce_input_search_space
):
    # Check that hyperopt trials sample from defaults in the search space
    for _, trial_row in hyperopt_results_df.iterrows():
        embedding_size = _get_trial_parameter_value("defaults.category.encoder.embedding_size", trial_row)
        num_filters = _get_trial_parameter_value("defaults.text.encoder.num_filters", trial_row)
        reduce_input = _get_trial_parameter_value("defaults.category.decoder.reduce_input", trial_row).replace('"', "")
        assert embedding_size in embedding_size_search_space
        assert num_filters in num_filters_search_space
        assert reduce_input in reduce_input_search_space


def _test_hyperopt_with_shared_params_written_config(
    hyperopt_results_df, num_filters_search_space, embedding_size_search_space, reduce_input_search_space
):
    # Check that each hyperopt trial's written input/output configs got updated
    for _, trial_row in hyperopt_results_df.iterrows():
        model_parameters = json.load(
            open(os.path.join(trial_row["trial_dir"], "test_hyperopt_run", "model", "model_hyperparameters.json"))
        )

        # Check that num_filters got updated from the sampler correctly
        for input_feature in model_parameters[INPUT_FEATURES]:
            if input_feature[TYPE] == TEXT:
                assert input_feature[ENCODER]["num_filters"] in num_filters_search_space
            elif input_feature[TYPE] == CATEGORY:
                assert input_feature[ENCODER]["embedding_size"] in embedding_size_search_space

        # All text features with defaults should have the same num_filters for this trial
        text_input_num_filters = get_feature_type_parameter_values_from_section(
            model_parameters, INPUT_FEATURES, TEXT, "num_filters"
        )
        assert len(text_input_num_filters) == 1

        for output_feature in model_parameters[OUTPUT_FEATURES]:
            if output_feature[TYPE] == CATEGORY:
                assert output_feature[DECODER]["reduce_input"] in reduce_input_search_space

        # All category features with defaults should have the same embedding_size for this trial
        input_category_features_embedding_sizes = get_feature_type_parameter_values_from_section(
            model_parameters, INPUT_FEATURES, CATEGORY, "embedding_size"
        )

        assert len(input_category_features_embedding_sizes) == 1


@pytest.mark.distributed
def test_hyperopt_with_shared_params(csv_filename, tmpdir):
    (
        config,
        rel_path,
        num_filters_search_space,
        embedding_size_search_space,
        reduce_input_search_space,
    ) = _setup_ludwig_config_with_shared_params(csv_filename)

    hyperopt_results = hyperopt(config, dataset=rel_path, output_directory=tmpdir, experiment_name="test_hyperopt")
    hyperopt_results_df = hyperopt_results.experiment_analysis.results_df

    _test_hyperopt_with_shared_params_trial_table(
        hyperopt_results_df, num_filters_search_space, embedding_size_search_space, reduce_input_search_space
    )

    _test_hyperopt_with_shared_params_written_config(
        hyperopt_results_df, num_filters_search_space, embedding_size_search_space, reduce_input_search_space
    )
