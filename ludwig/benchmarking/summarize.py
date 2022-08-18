import asyncio
import functools
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union, Set

import fsspec
from ludwig.benchmarking.summary_dataclasses import build_metrics_diff, MetricsDiff

from ludwig.utils.data_utils import load_yaml
from ludwig.utils.fs_utils import get_fs_and_path


def download_artifacts(
    bench_config_path: str, base_experiment: str, experimental_experiment: str, download_base_path: str
) -> Set[Union[Tuple[Any, Any]]]:
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
    local_dir = "benchmarking_summaries"
    os.makedirs(local_dir, exist_ok=True)

    coroutines = []
    for experiment in bench_config["datasets"]:
        dataset_name = experiment["dataset_name"]
        for experiment_name in [base_experiment, experimental_experiment]:
            coroutines.append(download_one(fs, download_base_path, dataset_name, experiment_name, local_dir))
    loop = asyncio.get_event_loop()
    futures = asyncio.gather(*coroutines, return_exceptions=True)
    downloaded_names = loop.run_until_complete(futures)
    loop.close()
    return set(downloaded_names)


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
    with ThreadPoolExecutor() as pool:
        func = functools.partial(
            fs.get,
            remote_experiment_directory,
            local_experiment_dir,
            recursive=True,
        )
        await loop.run_in_executor(pool, func)
    return dataset_name, local_dir


def summarize_metrics(
    bench_config_path: str, base_experiment: str, experimental_experiment: str, download_base_path: str
) -> List[MetricsDiff]:
    """Build summary and diffs of artifacts.

    bench_config_path: bench config file path. Can be the same one that was used to run
        these experiments.
    base_experiment: name of the experiment we're comparing against.
    experimental_experiment: name of the experiment we're comparing.
    download_base_path: base path under which live the stored artifacts of
        the benchmarking experiments.
    """
    downloaded_names = download_artifacts(bench_config_path, base_experiment, experimental_experiment, download_base_path)
    print(downloaded_names)
    experiment_diffs = []
    for experiment_tuple in downloaded_names:
        if isinstance(experiment_tuple, tuple) and len(experiment_tuple) == 2:
            (dataset_name, local_dir) = experiment_tuple
            try:
                e = build_metrics_diff(dataset_name, base_experiment, experimental_experiment, local_dir)
                experiment_diffs.append(e)
            except Exception:
                print("Exception encountered while creating diff summary for", dataset_name)
                print(traceback.format_exc())
    return experiment_diffs
