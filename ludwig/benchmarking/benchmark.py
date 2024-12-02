import argparse
import importlib
import logging
import os
import shutil
from typing import Any, Dict, Tuple, Union

import ludwig.datasets
from ludwig.api import LudwigModel
from ludwig.benchmarking.artifacts import BenchmarkingResult, build_benchmarking_result
from ludwig.benchmarking.profiler_callbacks import LudwigProfilerCallback
from ludwig.benchmarking.utils import (
    create_default_config,
    delete_hyperopt_outputs,
    delete_model_checkpoints,
    export_artifacts,
    load_from_module,
    populate_benchmarking_config_with_defaults,
    propagate_global_parameters,
    save_yaml,
    validate_benchmarking_config,
)
from ludwig.contrib import add_contrib_callback_args
from ludwig.hyperopt.run import hyperopt
from ludwig.utils.data_utils import load_yaml

logger = logging.getLogger()


def setup_experiment(experiment: Dict[str, str]) -> Dict[Any, Any]:
    """Set up the backend and load the Ludwig config.

    Args:
        experiment: dictionary containing the dataset name, config path, and experiment name.
        Returns a Ludwig config.
    """
    shutil.rmtree(os.path.join(experiment["experiment_name"]), ignore_errors=True)
    if "config_path" not in experiment:
        experiment["config_path"] = create_default_config(experiment)
    model_config = load_yaml(experiment["config_path"])

    if experiment["process_config_file_path"]:
        process_config_spec = importlib.util.spec_from_file_location(
            "process_config_file_path.py", experiment["process_config_file_path"]
        )
        process_module = importlib.util.module_from_spec(process_config_spec)
        process_config_spec.loader.exec_module(process_module)
        model_config = process_module.process_config(model_config, experiment)
        experiment["config_path"] = experiment["config_path"].replace(
            ".yaml", "-" + experiment["experiment_name"] + "-modified.yaml"
        )
        save_yaml(experiment["config_path"], model_config)

    return model_config


def benchmark_one(experiment: Dict[str, Union[str, Dict[str, str]]]) -> None:
    """Run a Ludwig exepriment and track metrics given a dataset name.

    Args:
        experiment: dictionary containing the dataset name, config path, and experiment name.
    """
    logger.info(f"\nRunning experiment *{experiment['experiment_name']}* on dataset *{experiment['dataset_name']}*")

    # configuring backend and paths
    model_config = setup_experiment(experiment)

    # loading dataset
    # dataset_module = importlib.import_module(f"ludwig.datasets.{experiment['dataset_name']}")
    dataset_module = ludwig.datasets.get_dataset(experiment["dataset_name"])
    dataset = load_from_module(dataset_module, model_config["output_features"][0])

    if experiment["hyperopt"]:
        # run hyperopt
        hyperopt(
            config=model_config,
            dataset=dataset,
            output_directory=experiment["experiment_name"],
            skip_save_model=True,
            skip_save_training_statistics=True,
            skip_save_progress=True,
            skip_save_log=True,
            skip_save_processed_input=True,
            skip_save_unprocessed_output=True,
            skip_save_predictions=True,
            skip_save_training_description=True,
            hyperopt_log_verbosity=0,
        )
        delete_hyperopt_outputs(experiment["experiment_name"])
    else:
        backend = None
        ludwig_profiler_callbacks = None
        if experiment["profiler"]["enable"]:
            ludwig_profiler_callbacks = [LudwigProfilerCallback(experiment)]
            # Currently, only local backend is supported with LudwigProfiler.
            backend = "local"
            logger.info("Currently, only local backend is supported with LudwigProfiler.")
        # run model and capture metrics
        model = LudwigModel(
            config=model_config, callbacks=ludwig_profiler_callbacks, logging_level=logging.ERROR, backend=backend
        )
        model.experiment(
            dataset=dataset,
            output_directory=experiment["experiment_name"],
            skip_save_processed_input=True,
            skip_save_unprocessed_output=True,
            skip_save_predictions=True,
            skip_collect_predictions=True,
        )
        delete_model_checkpoints(experiment["experiment_name"])


def benchmark(benchmarking_config: Union[Dict[str, Any], str]) -> Dict[str, Tuple[BenchmarkingResult, Exception]]:
    """Launch benchmarking suite from a benchmarking config.

    Args:
        benchmarking_config: config or config path for the benchmarking tool. Specifies datasets and their
            corresponding Ludwig configs, as well as export options.
    """
    if isinstance(benchmarking_config, str):
        benchmarking_config = load_yaml(benchmarking_config)
    validate_benchmarking_config(benchmarking_config)
    benchmarking_config = populate_benchmarking_config_with_defaults(benchmarking_config)
    benchmarking_config = propagate_global_parameters(benchmarking_config)

    experiment_artifacts = {}
    for experiment_idx, experiment in enumerate(benchmarking_config["experiments"]):
        dataset_name = experiment["dataset_name"]
        try:
            benchmark_one(experiment)
            experiment_artifacts[dataset_name] = (build_benchmarking_result(benchmarking_config, experiment_idx), None)
        except Exception as e:
            logger.exception(
                f"Experiment *{experiment['experiment_name']}* on dataset *{experiment['dataset_name']}* failed"
            )
            experiment_artifacts[dataset_name] = (None, e)
        finally:
            if benchmarking_config["export"]["export_artifacts"]:
                export_base_path = benchmarking_config["export"]["export_base_path"]
                export_artifacts(experiment, experiment["experiment_name"], export_base_path)
    return experiment_artifacts


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script runs a ludwig experiment on datasets specified in the benchmark config and exports "
        "the experiment artifact for each of the datasets following the export parameters specified in"
        "the benchmarking config.",
        prog="ludwig benchmark",
        usage="%(prog)s [options]",
    )
    parser.add_argument("--benchmarking_config", type=str, help="The benchmarking config.")
    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)
    benchmark(args.benchmarking_config)
