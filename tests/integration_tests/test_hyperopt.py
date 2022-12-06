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
import json
import os.path
import uuid
from typing import Dict, Tuple

import pytest

from ludwig.backend import initialize_backend
from ludwig.constants import (
    ACCURACY,
    AUTO,
    CATEGORY,
    COMBINER,
    EXECUTOR,
    HYPEROPT,
    INPUT_FEATURES,
    MAX_CONCURRENT_TRIALS,
    NAME,
    OUTPUT_FEATURES,
    RAY,
    TEXT,
    TRAINER,
    TYPE,
)
from ludwig.globals import HYPEROPT_STATISTICS_FILE_NAME
from ludwig.hyperopt.results import HyperoptResults
from ludwig.hyperopt.run import hyperopt
from ludwig.hyperopt.utils import update_hyperopt_params_with_defaults
from ludwig.schema.model_config import ModelConfig
from ludwig.utils import fs_utils
from ludwig.utils.data_utils import load_json, use_credentials
from tests.integration_tests.utils import (
    category_feature,
    generate_data,
    minio_test_creds,
    private_param,
    remote_tmpdir,
    text_feature,
)

ray = pytest.importorskip("ray")

from ludwig.hyperopt.execution import get_build_hyperopt_executor  # noqa

pytestmark = pytest.mark.distributed

RANDOM_SEARCH_SIZE = 2

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
    input_features = [category_feature(encoder={"vocab_size": 3})]
    output_features = [category_feature(decoder={"vocab_size": 3})]

    rel_path = generate_data(input_features, output_features, dataset_fp)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat"},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
    }

    config = ModelConfig.from_dict(config).to_dict()

    return config, rel_path


@pytest.mark.parametrize("search_alg", SEARCH_ALGS_FOR_TESTING)
def test_hyperopt_search_alg(
    search_alg, csv_filename, tmpdir, ray_cluster_7cpu, validate_output_feature=False, validation_metric=None
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

    backend = initialize_backend("local")
    if hyperopt_config[EXECUTOR].get(MAX_CONCURRENT_TRIALS) == AUTO:
        hyperopt_config[EXECUTOR][MAX_CONCURRENT_TRIALS] = backend.max_concurrent_trials(hyperopt_config)

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


def test_hyperopt_executor_with_metric(csv_filename, tmpdir, ray_cluster_7cpu):
    test_hyperopt_search_alg(
        "variant_generator",
        csv_filename,
        tmpdir,
        ray_cluster_7cpu,
        validate_output_feature=True,
        validation_metric=ACCURACY,
    )


@pytest.mark.parametrize("scheduler", SCHEDULERS_FOR_TESTING)
def test_hyperopt_scheduler(
    scheduler, csv_filename, tmpdir, ray_cluster_7cpu, validate_output_feature=False, validation_metric=None
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

    backend = initialize_backend("local")
    update_hyperopt_params_with_defaults(hyperopt_config)
    if hyperopt_config[EXECUTOR].get(MAX_CONCURRENT_TRIALS) == AUTO:
        hyperopt_config[EXECUTOR][MAX_CONCURRENT_TRIALS] = backend.max_concurrent_trials(hyperopt_config)

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


def _run_hyperopt_run_hyperopt(csv_filename, search_space, tmpdir, backend, ray_cluster_7cpu):
    input_features = [category_feature(encoder={"vocab_size": 3})]
    output_features = [category_feature(decoder={"vocab_size": 3})]

    rel_path = generate_data(input_features, output_features, csv_filename)

    config = {
        INPUT_FEATURES: input_features,
        OUTPUT_FEATURES: output_features,
        COMBINER: {TYPE: "concat"},
        TRAINER: {"epochs": 2, "learning_rate": 0.001},
        "backend": backend,
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
        }
    else:
        # grid search space will be product each parameter size
        search_parameters = {
            "trainer.learning_rate": {"space": "grid_search", "values": [0.001, 0.01]},
            output_feature_name + ".output_size": {"space": "grid_search", "values": [16, 21]},
        }

    hyperopt_configs = {
        "parameters": search_parameters,
        "goal": "minimize",
        "output_feature": output_feature_name,
        "validation_metrics": "loss",
        "executor": {
            TYPE: "ray",
            "num_samples": 1 if search_space == "grid" else RANDOM_SEARCH_SIZE,
            "max_concurrent_trials": 1,
        },
        "search_alg": {TYPE: "variant_generator"},
    }

    # add hyperopt parameter space to the config
    config[HYPEROPT] = hyperopt_configs

    experiment_name = f"test_hyperopt_{uuid.uuid4().hex}"
    hyperopt_results = hyperopt(config, dataset=rel_path, output_directory=tmpdir, experiment_name=experiment_name)
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
    with use_credentials(minio_test_creds()):
        assert fs_utils.path_exists(os.path.join(tmpdir, experiment_name, HYPEROPT_STATISTICS_FILE_NAME))
        for trial in hyperopt_results.experiment_analysis.trials:
            assert fs_utils.path_exists(
                os.path.join(tmpdir, experiment_name, f"trial_{trial.trial_id}"),
            )


@pytest.mark.parametrize("search_space", ["random", "grid"])
def test_hyperopt_run_hyperopt(csv_filename, search_space, tmpdir, ray_cluster_7cpu):
    _run_hyperopt_run_hyperopt(csv_filename, search_space, tmpdir, "local", ray_cluster_7cpu)


@pytest.mark.parametrize("fs_protocol,bucket", [private_param(("s3", "ludwig-tests"))], ids=["s3"])
def test_hyperopt_sync_remote(fs_protocol, bucket, csv_filename, ray_cluster_7cpu):
    backend = {
        "type": "local",
        "credentials": {
            "artifacts": minio_test_creds(),
        },
    }

    with remote_tmpdir(fs_protocol, bucket) as tmpdir:
        _run_hyperopt_run_hyperopt(
            csv_filename,
            "random",
            tmpdir,
            backend,
            ray_cluster_7cpu,
        )


def test_hyperopt_with_feature_specific_parameters(csv_filename, tmpdir, ray_cluster_7cpu):
    input_features = [
        text_feature(name="utterance", reduce_output="sum"),
        category_feature(vocab_size=3),
    ]

    output_features = [category_feature(vocab_size=3, output_feature=True)]

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


def test_hyperopt_old_config(csv_filename, tmpdir, ray_cluster_7cpu):
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


def test_hyperopt_nested_parameters(csv_filename, tmpdir, ray_cluster_7cpu):
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
                "num_samples": 2,
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
                            "combiner": {"type": "concat"},
                            "trainer": {"learning_rate_scaling": "linear"},
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
    assert len(results_df) == 2

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
            assert trial_config[TRAINER]["learning_rate_scaling"] == "linear"

        assert trial_config[TRAINER]["learning_rate"] in {0.7, 0.42}
