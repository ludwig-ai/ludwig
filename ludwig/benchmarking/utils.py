import copy
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import fsspec
import pandas as pd
from botocore.exceptions import ClientError
from globals import CONFIG_YAML, EXPERIMENT_RUN, REPORT_JSON
# todo (Wael): add to ludwig.globals
from s3fs.errors import translate_boto_error

from ludwig.constants import CATEGORY
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.datasets.base_dataset import BaseDataset
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.fs_utils import get_fs_and_path


def load_from_module(
    dataset_module: BaseDataset, output_feature: Dict[str, str], subsample_frac: float = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the ludwig dataset, optionally subsamples it, and returns a repeatable split.
    A stratified split is used for classification datasets.

    dataset_module: ludwig datasets module (e.g. ludwig.datasets.sst2, ludwig.datasets.ames_housing, etc.)
    subsample_frac: percentage of the total dataset to load.
    """
    dataset = dataset_module.load(split=False)
    if subsample_frac < 1:
        dataset = dataset.sample(frac=subsample_frac, replace=False, random_state=default_random_seed)

    if output_feature["type"] == CATEGORY:
        dataset = get_repeatable_train_val_test_split(dataset, stratify_colname=output_feature["name"],
                                                      random_seed=default_random_seed,)
    else:
        dataset = get_repeatable_train_val_test_split(dataset, random_seed=default_random_seed)

    train_df = dataset[dataset["split"] == 0]
    val_df = dataset[dataset["split"] == 1]
    test_df = dataset[dataset["split"] == 2]
    return train_df, val_df, test_df, dataset


def flatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


def export_artifacts(
    experiment: Dict[str, str], report_path: str, experiment_output_directory: str, export_base_path: str
) -> None:
    """Save the experiment artifacts to the `bench_export_directory`.

    experiment: experiment dict that contains "dataset_name" (e.g. ames_housing),
        "experiment_name" (specified by user), and "config_path" (path to experiment config.
        Relative to ludwig/benchmarks/configs).
    report_path: path where the experiment metrics report is
        saved.
    experiment_output_directory: path where the model, data,
        and logs of the experiment are saved.
    export_base_path: remote or local path (directory) where artifacts are
        exported. (e.g. s3://benchmarking.us-west-2.ludwig.com/bench/ or your/local/bench/)
    """
    protocol, _ = fsspec.core.split_protocol(export_base_path)
    fs, _ = get_fs_and_path(export_base_path)
    try:
        experiment_name = (
            experiment["experiment_name"]
            if experiment["experiment_name"]
            else datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
        )
        export_full_path = os.path.join(export_base_path, experiment["dataset_name"], experiment_name)
        fs.put(report_path, os.path.join(export_full_path, REPORT_JSON), recursive=True)
        fs.put(
            os.path.join("configs", experiment["config_path"]),
            os.path.join(export_full_path, CONFIG_YAML),
            recursive=True,
        )
        fs.put(
            os.path.join(experiment["dataset_name"], EXPERIMENT_RUN, "model", MODEL_HYPERPARAMETERS_FILE_NAME),
            os.path.join(export_full_path, MODEL_HYPERPARAMETERS_FILE_NAME),
            recursive=True,
        )

        # zip experiment directory to export
        try:
            shutil.make_archive("artifacts", "zip", experiment_output_directory)
            fs.put("artifacts.zip", os.path.join(export_full_path, "artifacts.zip"), recursive=True)
            os.remove("artifacts.zip")
        except Exception as e:
            logging.error(f"Couldn't export '{experiment_output_directory}' to bucket")
            logging.error(e)

        print("Uploaded metrics report and experiment config to\n\t", export_full_path)
    except ClientError as e:
        logging.error(translate_boto_error(e))
