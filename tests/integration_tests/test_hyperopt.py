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
    EXECUTOR,
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
from ludwig.hyperopt.results import HyperoptResults
from ludwig.hyperopt.run import hyperopt, update_hyperopt_params_with_defaults
from ludwig.utils.data_utils import load_json
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
        "trainer.learning_rate": {"space": "loguniform", "lower": 0.001, "upper": 0.1},
    },
    "goal": "minimize",
    "executor": {TYPE: "ray", "num_samples": 2, "scheduler": {TYPE: "fifo"}},
    "search_alg": {TYPE: "variant_generator"},
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
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
    }

    config = merge_with_defaults(config)

    return config, rel_path


def _setup_ludwig_config_with_shared_params(dataset_fp: str) -> Tuple[Dict, Any]:
    input_features = [
        text_feature(name="title", encoder={TYPE: "parallel_cnn"}),
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
            "executor": {TYPE: "ray", "num_samples": RANDOM_SEARCH_SIZE},
            "search_alg": {TYPE: "variant_generator"},
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
            TYPE: search_alg,
            "domain": "euclidean",
            "optimizer": "random",
        }
    elif search_alg is None:
        hyperopt_config["search_alg"] = {}
    else:
        hyperopt_config["search_alg"] = {
            TYPE: search_alg,
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
    results = hyperopt_executor.execute(config, dataset=rel_path, output_directory=tmpdir)
    assert isinstance(results, HyperoptResults)

    with hyperopt_executor._get_best_model_path(
        results.experiment_analysis.best_trial.logdir, results.experiment_analysis
    ) as path:
        assert path is not None
        assert isinstance(path, str)


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
            TYPE: scheduler,
            "hyperparam_bounds": hyperparam_bounds,
        }
    else:
        hyperopt_config["executor"]["scheduler"] = {
            TYPE: scheduler,
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
    if search_alg[TYPE] in {""}:
        with pytest.raises(ImportError):
            get_build_hyperopt_executor(RAY)(
                parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
            )
    else:
        hyperopt_executor = get_build_hyperopt_executor(RAY)(
            parameters, output_feature, metric, goal, split, search_alg=search_alg, **executor
        )
        raytune_results = hyperopt_executor.execute(config, dataset=rel_path, output_directory=tmpdir)
        assert isinstance(raytune_results, HyperoptResults)


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
        "executor": {TYPE: "ray", "num_samples": 1 if search_space == "grid" else RANDOM_SEARCH_SIZE},
        "search_alg": {TYPE: "variant_generator"},
    }

    # add hyperopt parameter space to the config
    config[HYPEROPT] = hyperopt_configs

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


@pytest.mark.distributed
def test_hyperopt_with_feature_specific_parameters(csv_filename, tmpdir, ray_cluster):
    input_features = [
        text_feature(name="utterance", reduce_output="sum"),
        category_feature(vocab_size=3),
    ]

    output_features = [category_feature(vocab_size=3)]

    rel_path = generate_data(input_features, output_features, csv_filename)

    filter_size_search_space = [5, 7]
    embedding_size_search_space = [4, 8, 12]

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat", "num_fc_layers": 2},
        TRAINER: {"epochs": 1, "learning_rate": 0.001},
        HYPEROPT: {
            "parameters": {
                input_features[0][NAME]
                + ".encoder.filter_size": {"space": "choice", "categories": filter_size_search_space},
                input_features[1][NAME]
                + ".encoder.embedding_size": {"space": "choice", "categories": embedding_size_search_space},
            },
            "goal": "minimize",
            "output_feature": output_features[0][NAME],
            "validation_metrics": "loss",
            "executor": {TYPE: "ray", "num_samples": 1},
            "search_alg": {TYPE: "variant_generator"},
        },
    }

    hyperopt_results = hyperopt(config, dataset=rel_path, output_directory=tmpdir, experiment_name="test_hyperopt")
    hyperopt_results_df = hyperopt_results.experiment_analysis.results_df

    model_parameters = json.load(
        open(
            os.path.join(
                hyperopt_results_df.iloc[0]["trial_dir"], "test_hyperopt_run", "model", "model_hyperparameters.json"
            )
        )
    )

    for input_feature in model_parameters[INPUT_FEATURES]:
        if input_feature[TYPE] == TEXT:
            assert input_feature["encoder"]["filter_size"] in filter_size_search_space
        elif input_feature[TYPE] == CATEGORY:
            assert input_feature["encoder"]["embedding_size"] in embedding_size_search_space


@pytest.mark.distributed
def test_hyperopt_old_config(csv_filename, tmpdir, ray_cluster):
    old_config = {
        "ludwig_version": "0.4",
        INPUT_FEATURES: [
            {"name": "cat1", TYPE: "category", "encoder": {"vocab_size": 2}},
            {"name": "num1", TYPE: "number"},
        ],
        OUTPUT_FEATURES: [
            {"name": "bin1", TYPE: "binary"},
        ],
        TRAINER: {"epochs": 2},
        HYPEROPT: {
            EXECUTOR: {
                TYPE: "ray",
                "time_budget_s": 200,
                "cpu_resources_per_trial": 1,
            },
            "sampler": {
                TYPE: "ray",
                "scheduler": {
                    TYPE: "async_hyperband",
                    "max_t": 200,
                    "time_attr": "time_total_s",
                    "grace_period": 72,
                    "reduction_factor": 5,
                },
                "search_alg": {
                    TYPE: HYPEROPT,
                    "random_state_seed": 42,
                },
                "num_samples": 2,
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

    input_features = old_config[INPUT_FEATURES]
    output_features = old_config[OUTPUT_FEATURES]
    rel_path = generate_data(input_features, output_features, csv_filename)

    hyperopt(old_config, dataset=rel_path, output_directory=tmpdir, experiment_name="test_hyperopt")


@pytest.mark.distributed
def test_hyperopt_nested_parameters(csv_filename, tmpdir, ray_cluster):
    config = {
        INPUT_FEATURES: [
            {"name": "cat1", TYPE: "category", "encoder": {"vocab_size": 2}},
            {"name": "num1", TYPE: "number"},
        ],
        OUTPUT_FEATURES: [
            {"name": "bin1", TYPE: "binary"},
        ],
        TRAINER: {"epochs": 2},
        HYPEROPT: {
            EXECUTOR: {
                TYPE: "ray",
                "time_budget_s": 200,
                "cpu_resources_per_trial": 1,
                "num_samples": 4,
                "scheduler": {TYPE: "fifo"},
            },
            "search_alg": {TYPE: "variant_generator"},
            "parameters": {
                ".": {
                    "space": "choice",
                    "categories": [
                        {
                            "combiner": {
                                "type": "tabnet",
                                "bn_virtual_bs": 256,
                            },
                            "trainer": {
                                "learning_rate_scaling": "sqrt",
                                "decay": True,
                                "decay_steps": 20000,
                                "decay_rate": 0.8,
                                "optimizer": {"type": "adam"},
                            },
                        },
                        {
                            "combiner": {
                                "type": "concat",
                                "num_fc_layers": 2,
                            },
                            "trainer": {
                                "learning_rate_scaling": "linear",
                            },
                        },
                    ],
                },
                "trainer.learning_rate": {"space": "choice", "categories": [0.7, 0.42]},
            },
        },
    }

    input_features = config[INPUT_FEATURES]
    output_features = config[OUTPUT_FEATURES]
    rel_path = generate_data(input_features, output_features, csv_filename)

    results = hyperopt(
        config,
        dataset=rel_path,
        output_directory=tmpdir,
        experiment_name="test_hyperopt_nested_params",
    )

    results_df = results.experiment_analysis.results_df
    assert len(results_df) == 4

    for _, trial_meta in results_df.iterrows():
        trial_dir = trial_meta["trial_dir"]
        trial_config = load_json(
            os.path.join(trial_dir, "test_hyperopt_nested_params_run", "model", "model_hyperparameters.json")
        )

        assert len(trial_config[INPUT_FEATURES]) == len(config[INPUT_FEATURES])
        assert len(trial_config[OUTPUT_FEATURES]) == len(config[OUTPUT_FEATURES])

        assert trial_config[COMBINER][TYPE] in {"tabnet", "concat"}
        if trial_config[COMBINER][TYPE] == "tabnet":
            assert trial_config[COMBINER]["bn_virtual_bs"] == 256
            assert trial_config[TRAINER]["learning_rate_scaling"] == "sqrt"
            assert trial_config[TRAINER]["decay"] is True
            assert trial_config[TRAINER]["decay_steps"] == 20000
            assert trial_config[TRAINER]["decay_rate"] == 0.8
            assert trial_config[TRAINER]["optimizer"]["type"] == "adam"
        else:
            assert trial_config[COMBINER]["num_fc_layers"] == 2
            assert trial_config[TRAINER]["learning_rate_scaling"] == "linear"

        assert trial_config[TRAINER]["learning_rate"] in {0.7, 0.42}
