import asyncio
import functools
import logging
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, List, Tuple, Union

import fsspec
import pandas as pd
import yaml

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import BINARY, CATEGORY
from ludwig.datasets import model_configs_for_dataset
from ludwig.datasets.loaders.dataset_loader import DatasetLoader
from ludwig.globals import CONFIG_YAML
from ludwig.utils.data_utils import load_yaml
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import get_fs_and_path

HYPEROPT_OUTDIR_RETAINED_FILES = [
    "hyperopt_statistics.json",
    "params.json",
    "stderr",
    "stdout",
    "result.json",
    "error.txt",
]
logger = logging.getLogger()


def load_from_module(
    dataset_module: Union[DatasetLoader, ModuleType], output_feature: Dict[str, str], subsample_frac: float = 1
) -> pd.DataFrame:
    """Load the ludwig dataset, optionally subsamples it, and returns a repeatable split. A stratified split is
    used for classification datasets.

    Args:
        dataset_module: ludwig datasets module (e.g. ludwig.datasets.sst2, ludwig.datasets.ames_housing, etc.)
        subsample_frac: percentage of the total dataset to load.
    """
    dataset = dataset_module.load(split=False)
    if subsample_frac < 1:
        dataset = dataset.sample(frac=subsample_frac, replace=False, random_state=default_random_seed)

    if output_feature["type"] in [CATEGORY, BINARY]:
        return get_repeatable_train_val_test_split(
            dataset,
            stratify_colname=output_feature["name"],
            random_seed=default_random_seed,
        )
    else:
        return get_repeatable_train_val_test_split(dataset, random_seed=default_random_seed)


def export_artifacts(experiment: Dict[str, str], experiment_output_directory: str, export_base_path: str):
    """Save the experiment artifacts to the `bench_export_directory`.

    Args:
        experiment: experiment dict that contains "dataset_name" (e.g. ames_housing),
            "experiment_name" (specified by user), and "config_path" (path to experiment config.
            Relative to ludwig/benchmarks/configs).
        experiment_output_directory: path where the model, data, and logs of the experiment are saved.
        export_base_path: remote or local path (directory) where artifacts are
            exported. (e.g. s3://benchmarking.us-west-2.ludwig.com/bench/ or your/local/bench/)
    """
    protocol, _ = fsspec.core.split_protocol(export_base_path)
    fs, _ = get_fs_and_path(export_base_path)
    try:
        export_full_path = os.path.join(export_base_path, experiment["dataset_name"], experiment["experiment_name"])

        # override previous experiment with the same name
        if fs.exists(export_full_path):
            fs.rm(export_full_path, recursive=True)
        fs.put(experiment_output_directory, export_full_path, recursive=True)
        fs.put(
            os.path.join(experiment["config_path"]),
            os.path.join(export_full_path, CONFIG_YAML),
        )
        logger.info(f"Uploaded experiment artifact to\n\t{export_full_path}")
    except Exception:
        logger.exception(
            f"Failed to upload experiment artifacts for experiment *{experiment['experiment_name']}* on "
            f"dataset {experiment['dataset_name']}"
        )


def download_artifacts(
    bench_config_path: str,
    base_experiment: str,
    experimental_experiment: str,
    download_base_path: str,
    local_dir: str = "benchmarking_summaries",
) -> Tuple[str, List[str]]:
    """Download benchmarking artifacts for two experiments.

    Args:
        bench_config_path: bench config file path. Can be the same one that was used to run
            these experiments.
        base_experiment: name of the experiment we're comparing against.
        experimental_experiment: name of the experiment we're comparing.
        download_base_path: base path under which live the stored artifacts of
            the benchmarking experiments.
    """
    bench_config = load_yaml(bench_config_path)
    protocol, _ = fsspec.core.split_protocol(download_base_path)
    fs, _ = get_fs_and_path(download_base_path)
    os.makedirs(local_dir, exist_ok=True)

    coroutines = []
    for experiment in bench_config["experiments"]:
        dataset_name = experiment["dataset_name"]
        for experiment_name in [base_experiment, experimental_experiment]:
            coroutines.append(download_one(fs, download_base_path, dataset_name, experiment_name, local_dir))
    loop = asyncio.get_event_loop()
    futures = asyncio.gather(*coroutines, return_exceptions=True)
    downloaded_names = loop.run_until_complete(futures)

    dataset_names = [experiment_tuple[0] for experiment_tuple in set(downloaded_names) if experiment_tuple[0]]
    assert (
        len({experiment_tuple[1] for experiment_tuple in downloaded_names}) == 1 and downloaded_names[0][1] == local_dir
    ), "Experiments not downloaded to the same path"

    return local_dir, dataset_names


@DeveloperAPI
async def download_one(
    fs, download_base_path: str, dataset_name: str, experiment_name: str, local_dir: str
) -> Tuple[str, str]:
    """Download `config.yaml` and `report.json` for an experiment.

    Args:
        fs: filesystem to use to download.
        download_base_path: base path under which live the stored artifacts of
            the benchmarking experiments.
        dataset_name: name of the dataset we ran the experiments on.
        experiment_name: name of the experiment (e.g. `v0.5.3_with_bert`)
        local_dir: local directory under which the artifacts will be downloaded.
    """
    loop = asyncio.get_running_loop()
    local_experiment_dir = os.path.join(local_dir, dataset_name, experiment_name)
    remote_experiment_directory = os.path.join(download_base_path, dataset_name, experiment_name)
    os.makedirs(local_experiment_dir, exist_ok=True)
    try:
        with ThreadPoolExecutor() as pool:
            func = functools.partial(
                fs.get,
                remote_experiment_directory,
                local_experiment_dir,
                recursive=True,
            )
            await loop.run_in_executor(pool, func)
    except Exception:
        logger.exception(f"Couldn't download experiment *{experiment_name}* of dataset *{dataset_name}*.")
        return "", local_dir
    return dataset_name, local_dir


def validate_benchmarking_config(benchmarking_config: Dict[str, Any]) -> None:
    """Validates the parameters of the benchmarking config.

    Args:
        benchmarking_config: benchmarking config dictionary.

    Raises:
        ValueError if any of the expected parameters is not there.
    """
    if "experiment_name" not in benchmarking_config and not all(
        "experiment_name" in experiment for experiment in benchmarking_config["experiments"]
    ):
        raise ValueError("You must either specify a global experiment name or an experiment name for each experiment.")
    if "export" not in benchmarking_config:
        raise ValueError(
            """You must specify export parameters. Example:
            export:
              export_artifacts: true
              export_base_path: s3://benchmarking.us-west-2.ludwig.com/bench/    # include the slash at the end.
        """
        )
    if "experiments" not in benchmarking_config:
        raise ValueError("You must specify a list of experiments.")
    for experiment in benchmarking_config["experiments"]:
        if "dataset_name" not in experiment:
            raise ValueError("A Ludwig dataset must be specified.")


def populate_benchmarking_config_with_defaults(benchmarking_config: Dict[str, Any]) -> Dict[str, Any]:
    """Populates the parameters of the benchmarking config with defaults.

    Args:
        benchmarking_config: benchmarking config dictionary.
    """
    if "hyperopt" not in benchmarking_config:
        benchmarking_config["hyperopt"] = False
    if "process_config_file_path" not in benchmarking_config:
        benchmarking_config["process_config_file_path"] = None
    if "profiler" not in benchmarking_config:
        benchmarking_config["profiler"] = {"enable": False, "use_torch_profiler": False, "logging_interval": None}
    return benchmarking_config


def propagate_global_parameters(benchmarking_config: Dict[str, Any]) -> Dict[str, Any]:
    """Propagate the global parameters of the benchmarking config to local experiments.

    Args:
        benchmarking_config: benchmarking config dictionary.
    """
    for experiment in benchmarking_config["experiments"]:
        if "experiment_name" not in experiment:
            experiment["experiment_name"] = benchmarking_config["experiment_name"]
        if "export" not in experiment:
            experiment["export"] = benchmarking_config["export"]
        if "hyperopt" not in experiment:
            experiment["hyperopt"] = benchmarking_config["hyperopt"]
        if "process_config_file_path" not in experiment:
            experiment["process_config_file_path"] = benchmarking_config["process_config_file_path"]
        if "profiler" not in experiment:
            experiment["profiler"] = benchmarking_config["profiler"]
    return benchmarking_config


def create_default_config(experiment: Dict[str, Any]) -> str:
    """Create a Ludwig config that only contains input and output features.

    Args:
        experiment: experiment dictionary.

    Returns:
        path where the default config is saved.
    """
    model_config = model_configs_for_dataset(experiment["dataset_name"])["default"]

    # only keep input_features and output_features
    main_config_keys = list(model_config.keys())
    for key in main_config_keys:
        if key not in ["input_features", "output_features"]:
            del model_config[key]
    config_path = f"{experiment['dataset_name']}-{uuid.uuid4().hex}.yaml"
    save_yaml(config_path, model_config)
    return config_path


def delete_model_checkpoints(output_directory: str):
    """Deletes outputs of the experiment run that we don't want to save with the artifacts.

    Args:
        output_directory: output directory of the hyperopt run.
    """
    shutil.rmtree(os.path.join(output_directory, "model", "training_checkpoints"), ignore_errors=True)
    if os.path.isfile(os.path.join(output_directory, "model", "model_weights")):
        os.remove(os.path.join(output_directory, "model", "model_weights"))


def delete_hyperopt_outputs(output_directory: str):
    """Deletes outputs of the hyperopt run that we don't want to save with the artifacts.

    Args:
        output_directory: output directory of the hyperopt run.
    """
    for path, currentDirectory, files in os.walk(output_directory):
        for file in files:
            filename = os.path.join(path, file)
            if file not in HYPEROPT_OUTDIR_RETAINED_FILES:
                os.remove(filename)


def save_yaml(filename, dictionary):
    with open(filename, "w") as f:
        yaml.dump(dictionary, f, default_flow_style=False)


def format_time(time_us):
    """Defines how to format time in FunctionEvent.

    from https://github.com/pytorch/pytorch/blob/master/torch/autograd/profiler_util.py
    """
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return f"{time_us / US_IN_SECOND:.3f}s"
    if time_us >= US_IN_MS:
        return f"{time_us / US_IN_MS:.3f}ms"
    return f"{time_us:.3f}us"


def format_memory(nbytes):
    """Returns a formatted memory size string.

    from https://github.com/pytorch/pytorch/blob/master/torch/autograd/profiler_util.py
    """
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    else:
        return str(nbytes) + " b"
