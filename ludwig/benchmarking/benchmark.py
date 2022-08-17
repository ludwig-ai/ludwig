import argparse
import importlib
import logging
import os
import shutil
from typing import Any, Dict, Union

from ludwig.api import LudwigModel
from ludwig.benchmarking.utils import export_artifacts, load_from_module
from ludwig.contrib import add_contrib_callback_args
from ludwig.utils.data_utils import load_yaml

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_experiment(experiment: Dict[str, str]) -> Dict[Any, Any]:
    """Set up the backend and load the Ludwig config.

    experiment: dictionary containing the dataset name, config path, and experiment name.
    Returns a Ludwig config.
    """
    shutil.rmtree(os.path.join(experiment["dataset_name"]), ignore_errors=True)
    model_config = load_yaml(experiment["config_path"])
    model_config["backend"] = {}
    model_config["backend"]["type"] = "local"
    return model_config


def benchmark_one_local(experiment: Dict[str, str], export_artifacts_dict: Dict[str, Union[str, bool]]) -> None:
    """Run a Ludwig exepriment and track metrics given a dataset name.

    experiment: dictionary containing the dataset name, config path, and experiment name.
    export_artifacts_dict: dictionary containing an export boolean flag and a path to export to.
    """
    logging.info(f"\nRunning experiment *{experiment['experiment_name']}* on dataset *{experiment['dataset_name']}*")

    # configuring backend and paths
    model_config = setup_experiment(experiment)

    # loading dataset
    dataset_module = importlib.import_module(f"ludwig.datasets.{experiment['dataset_name']}")
    dataset = load_from_module(dataset_module, model_config["output_features"][0])

    # running model and capturing metrics
    model = LudwigModel(config=model_config, logging_level=logging.ERROR)
    _, _, _, output_directory = model.experiment(
        dataset=dataset,
        output_directory=experiment["dataset_name"],
        skip_save_processed_input=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=True,
        skip_collect_predictions=True,
    )
    if export_artifacts_dict["export_artifacts"]:
        export_base_path = export_artifacts_dict["export_base_path"]
        export_artifacts(experiment, output_directory, export_base_path)


def benchmark(bench_config_path: str) -> None:
    """Launch benchmarking suite from a benchmarking config.

    bench_config_path: config for the benchmarking tool. Specifies datasets and their
        corresponding Ludwig configs, as well as export options.
    """
    config = load_yaml(bench_config_path)
    for experiment in config["datasets"]:
        try:
            if "experiment_name" not in experiment:
                experiment["experiment_name"] = config["global_experiment_name"]
            benchmark_one_local(experiment, export_artifacts_dict=config["export"][0])
        except Exception:
            logging.exception(
                f"Experiment *{experiment['experiment_name']}* on dataset *{experiment['dataset_name']}* failed"
            )


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
    benchmark(args.config)
