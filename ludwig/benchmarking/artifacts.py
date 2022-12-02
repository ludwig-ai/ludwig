import os
from dataclasses import dataclass
from typing import Any, Dict

from ludwig.types import ModelConfigDict, TrainingSetMetadataDict
from ludwig.utils.data_utils import load_json, load_yaml


@dataclass
class BenchmarkingResult:
    # The Ludwig benchmarking config.
    benchmarking_config: Dict[str, Any]

    # The config for one experiment.
    experiment_config: Dict[str, Any]

    # The Ludwig config used to run the experiment.
    ludwig_config: ModelConfigDict

    # The python script that is used to process the config before being used.
    process_config_file: str

    # Loaded `description.json` file.
    description: Dict[str, Any]

    # Loaded `test_statistics.json` file.
    test_statistics: Dict[str, Any]

    # Loaded `training_statistics.json` file.
    training_statistics: Dict[str, Any]

    # Loaded `model_hyperparameters.json` file.
    model_hyperparameters: Dict[str, Any]

    # Loaded `training_progress.json` file.
    training_progress: Dict[str, Any]

    # Loaded `training_set_metadata.json` file.
    training_set_metadata: TrainingSetMetadataDict


def build_benchmarking_result(benchmarking_config: dict, experiment_idx: int):
    experiment_config = benchmarking_config["experiments"][experiment_idx]
    process_config_file = ""
    if experiment_config["process_config_file_path"]:
        with open(experiment_config["process_config_file_path"]) as f:
            process_config_file = "".join(f.readlines())
    experiment_run_path = os.path.join(experiment_config["experiment_name"], "experiment_run")

    return BenchmarkingResult(
        benchmarking_config=benchmarking_config,
        experiment_config=experiment_config,
        ludwig_config=load_yaml(experiment_config["config_path"]),
        process_config_file=process_config_file,
        description=load_json(os.path.join(experiment_run_path, "description.json")),
        test_statistics=load_json(os.path.join(experiment_run_path, "test_statistics.json")),
        training_statistics=load_json(os.path.join(experiment_run_path, "training_statistics.json")),
        model_hyperparameters=load_json(os.path.join(experiment_run_path, "model", "model_hyperparameters.json")),
        training_progress=load_json(os.path.join(experiment_run_path, "model", "training_progress.json")),
        training_set_metadata=load_json(os.path.join(experiment_run_path, "model", "training_set_metadata.json")),
    )
