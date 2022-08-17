import logging
import os
from types import ModuleType
from typing import Any, Dict, Union

import fsspec
import pandas as pd

from ludwig.constants import CATEGORY
from ludwig.datasets.base_dataset import BaseDataset
from ludwig.globals import CONFIG_YAML
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


def flatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient="records")
    return flat_dict


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
