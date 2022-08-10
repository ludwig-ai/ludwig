import argparse
import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple, Union

import fsspec
import traceback
from summary_dataclasses import build_experiments_diff, ExperimentsDiff

from ludwig.globals import REPORT_JSON
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.utils.data_utils import load_yaml
from ludwig.utils.fs_utils import get_fs_and_path

# todo (Wael): to update once summary dataclasses PR is merged.


def download_artifacts(
    bench_config: Dict[str, Any], base_experiment: str, experimental_experiment: str, download_base_path: str
) -> List[Union[Tuple[str, str], Any]]:
    """Download benchmarking artifacts for two experiments.

    bench_config: bench config file. Can be the same one that was used to run
        these experiments.
    base_experiment: name of the experiment we're comparing against.
    experimental_experiment: name of the experiment we're comparing.
    download_base_path: base path under which live the stored artifacts of
        the benchmarking experiments.
    """
    protocol, _ = fsspec.core.split_protocol(download_base_path)
    fs, _ = get_fs_and_path(download_base_path)
    local_dir = os.path.join(os.getcwd(), "visualize-temp")
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
    return downloaded_names


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
    os.makedirs(local_experiment_dir, exist_ok=True)
    with ThreadPoolExecutor() as pool:
        remote_files = [file_dict["Key"] for file_dict in fs.listdir(os.path.join(download_base_path, dataset_name, experiment_name))]
        for remote_file in remote_files:
            func = functools.partial(
                fs.get,
                remote_file,
                os.path.join(local_experiment_dir, remote_file.split("/")[-1]),
                recursive=True,
            )
            await loop.run_in_executor(pool, func)
    return dataset_name, local_dir


def build_summary(
    bench_config_path: str, base_experiment: str, experimental_experiment: str, download_base_path: str
) -> List[ExperimentsDiff]:
    """Build summary and diffs of artifacts.

    bench_config_path: bench config file path. Can be the same one that was used to run
        these experiments.
    base_experiment: name of the experiment we're comparing against.
    experimental_experiment: name of the experiment we're comparing.
    download_base_path: base path under which live the stored artifacts of
        the benchmarking experiments.
    """
    config = load_yaml(bench_config_path)
    downloaded_names = set(download_artifacts(config, base_experiment, experimental_experiment, download_base_path))
    # print(downloaded_names)
    experiment_diffs = []
    for n in downloaded_names:
        if isinstance(n, tuple) and len(n) == 2:
            (dataset_name, local_dir) = n
            try:
                e = build_experiments_diff(dataset_name, base_experiment, experimental_experiment, local_dir)
                experiment_diffs.append(e)
            except Exception as e:
                print("Exception encountered while creating diff summary for", dataset_name)
                print(traceback.format_exc())
    return experiment_diffs


def export_summary(experiment_diffs: List[ExperimentsDiff]) -> None:
    """Print and export to .csv the summary of metrics and their diffs.

    experiment_diffs: list of `ExperimentsDiff` dataclass, containing summary
        of metrics.
    """
    example_diff = experiment_diffs[0]
    spacing_str = "{:<33} {:<20} {:<23} {:<13} {:<13} {:<13} {:<5}"
    csv_str = "{}, {}, {}, {}, {}, {}, {}\n"
    print(
        spacing_str.format(
            "Dataset Name",
            "Output Feature Name",
            "Metric Name",
            example_diff.base_experiment_name,
            example_diff.experimental_experiment_name,
            "Diff",
            "Diff Percentage",
        )
    )

    with open(f"report_{example_diff.base_experiment_name}_{example_diff.experimental_experiment_name}.csv", "w") as f:
        f.write(
            csv_str.format(
                "Dataset Name",
                "Output Feature Name",
                "Metric Name",
                example_diff.base_experiment_name,
                example_diff.experimental_experiment_name,
                "Diff",
                "Diff Percentage",
            )
        )
        for experiment_diff in experiment_diffs:
            for metric in experiment_diff.metrics:
                output_feature_name = experiment_diff.base_summary.output_feature_name
                metric_name = metric.name
                experiment1_val = round(metric.base_value, 3)
                experiment2_val = round(metric.experimental_value, 3)
                diff = round(metric.diff, 3)
                diff_percentage = round(metric.diff_percentage, 3)
                print(
                    spacing_str.format(
                        experiment_diff.dataset_name,
                        output_feature_name,
                        metric_name,
                        experiment1_val,
                        experiment2_val,
                        diff,
                        diff_percentage,
                    )
                )
                f.write(
                    csv_str.format(
                        experiment_diff.dataset_name,
                        output_feature_name,
                        metric_name,
                        experiment1_val,
                        experiment2_val,
                        diff,
                        diff_percentage,
                    )
                )

    print(
        "Exported report to",
        f"report_{example_diff.base_experiment_name}_{example_diff.experimental_experiment_name}.csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-one", type=str, help="Name of first experiment", default="0.5.3")
    parser.add_argument("--experiment-two", type=str, help="Name of first experiment", default="0.5.4")
    parser.add_argument(
        "--download-base-path",
        type=str,
        help="Base path under which benchmarking experiment artifacts (config.yaml, report.json, " "etc.) are saved",
        default="s3://benchmarking.us-west-2.predibase.com/bench/",
    )
    args, unknown = parser.parse_known_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.chdir("bench")

    summary = build_summary(
        "./configs/temp.yaml", args.experiment_one, args.experiment_two, download_base_path=args.download_base_path
    )
    print(summary)
    export_summary(summary)
