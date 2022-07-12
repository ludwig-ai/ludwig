import importlib
import logging
import os
import shutil
import traceback
from typing import Any, Dict, Union

from ludwig_with_benchmarks import ludwig_experiment_with_benchmarks

# todo (Wael): to update once benchmarking utils PR merged.
from reporting import create_metrics_report
from utils import export_artifacts, load_from_module

from ludwig.api import LudwigModel
from ludwig.utils.data_utils import load_yaml

# todo (Wael): to update once api.py PR is merged.

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_experiment(experiment: Dict[str, str]) -> Dict[Any, Any]:
    """Set up the backend and load the Ludwig config.

    experiment: dictionary containing the dataset name, config path, and experiment name.
    Returns a Ludwig config.
    """
    shutil.rmtree(os.path.join(os.getcwd(), experiment["dataset_name"]), ignore_errors=True)
    model_config = load_yaml(os.path.join("configs", experiment["config_path"]))
    model_config["backend"] = {}
    model_config["backend"]["type"] = "local"
    model_config["backend"]["cache_dir"] = os.path.join(os.getcwd(), experiment["dataset_name"], "cache")
    os.makedirs(model_config["backend"]["cache_dir"], exist_ok=True)
    return model_config


def benchmark_one_local(experiment: Dict[str, str], export_artifacts_dict: Dict[str, Union[str, bool]]) -> None:
    """Run a Ludwig exepriment and track metrics given a dataset name.

    experiment: dictionary containing the dataset name, config path, and experiment name.
    export_artifacts_dict: dictionary containing an export boolean flag and a path to export to.
    """
    print("\nRunning", experiment["dataset_name"] + " " + experiment["experiment_name"])

    # configuring backend and paths
    model_config = setup_experiment(experiment)

    # loading dataset
    dataset_module = importlib.import_module("ludwig.datasets.{}".format(experiment["dataset_name"]))
    train_df, val_df, test_df, _ = load_from_module(dataset_module, model_config["output_features"][0])

    # running model and capturing metrics
    experiment_output_directory = os.path.join(os.getcwd(), experiment["dataset_name"])
    model = LudwigModel(config=model_config, logging_level=logging.ERROR)
    ludwig_experiment_with_benchmarks(
        model,
        training_set=train_df,
        validation_set=val_df,
        test_set=test_df,
        output_directory=experiment_output_directory,
    )

    # creating full report containing performance metrics (e.g. accuracy) and non-performance metrics (e.g. RAM usage)
    _, report_path = create_metrics_report(experiment["dataset_name"])

    # exporting the metrics report and experiment config to s3
    if export_artifacts_dict["export_artifacts"]:
        export_artifacts(
            experiment, report_path, experiment_output_directory, export_artifacts_dict["export_base_path"]
        )


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
        except Exception as e:
            print("Benchmarking {} {} failed".format(experiment["dataset_name"], experiment["experiment_name"]))
            print(traceback.format_exc())
