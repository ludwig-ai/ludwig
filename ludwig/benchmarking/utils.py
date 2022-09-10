import asyncio
import functools
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Dict, List, Tuple, Union

import fsspec
import pandas as pd

from ludwig.constants import CATEGORY
from ludwig.datasets.base_dataset import BaseDataset
from ludwig.globals import CONFIG_YAML
from ludwig.utils.data_utils import load_yaml
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import get_fs_and_path


def load_from_module(
    dataset_module: Union[BaseDataset, ModuleType], output_feature: Dict[str, str], subsample_frac: float = 1
) -> pd.DataFrame:
    """Load the ludwig dataset, optionally subsamples it, and returns a repeatable split. A stratified split is
    used for classification datasets.

    dataset_module: ludwig datasets module (e.g. ludwig.datasets.sst2, ludwig.datasets.ames_housing, etc.)
    subsample_frac: percentage of the total dataset to load.
    """
    dataset = dataset_module.load(split=False)
    if subsample_frac < 1:
        dataset = dataset.sample(frac=subsample_frac, replace=False, random_state=default_random_seed)

    if output_feature["type"] == CATEGORY:
        return get_repeatable_train_val_test_split(
            dataset,
            stratify_colname=output_feature["name"],
            random_seed=default_random_seed,
        )
    else:
        return get_repeatable_train_val_test_split(dataset, random_seed=default_random_seed)


def export_artifacts(experiment: Dict[str, str], experiment_output_directory: str, export_base_path: str):
    """Save the experiment artifacts to the `bench_export_directory`.

    :param experiment: experiment dict that contains "dataset_name" (e.g. ames_housing),
        "experiment_name" (specified by user), and "config_path" (path to experiment config.
        Relative to ludwig/benchmarks/configs).
    :param experiment_output_directory: path where the model, data, and logs of the experiment are saved.
    :param export_base_path: remote or local path (directory) where artifacts are
        exported. (e.g. s3://benchmarking.us-west-2.ludwig.com/bench/ or your/local/bench/)
    """
    protocol, _ = fsspec.core.split_protocol(export_base_path)
    fs, _ = get_fs_and_path(export_base_path)
    try:
        export_full_path = os.path.join(export_base_path, experiment["dataset_name"], experiment["experiment_name"])
        fs.put(experiment_output_directory, export_full_path, recursive=True)
        fs.put(
            os.path.join("configs", experiment["config_path"]),
            os.path.join(export_full_path, CONFIG_YAML),
        )
        logging.info(f"Uploaded experiment artifact to\n\t{export_full_path}")
    except Exception:
        logging.exception(
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

    bench_config: bench config file. Can be the same one that was used to run
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
    for experiment in bench_config["datasets"]:
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


async def download_one(
    fs, download_base_path: str, dataset_name: str, experiment_name: str, local_dir: str
) -> Tuple[str, str]:
    """Download `config.yaml` and `report.json` for an experiment.

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
        logging.exception(f"Couldn't download experiment *{experiment_name}* of dataset *{dataset_name}*.")
        return "", local_dir
    return dataset_name, local_dir


def delete_model_checkpoints(output_directory: str):
    shutil.rmtree(os.path.join(output_directory, "model", "training_checkpoints"), ignore_errors=True)
    if os.path.isfile(os.path.join(output_directory, "model", "model_weights")):
        os.remove(os.path.join(output_directory, "model", "model_weights"))


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
