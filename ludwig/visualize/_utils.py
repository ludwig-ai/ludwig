# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Shared private helpers and data-loading utilities for visualizations."""

import itertools
import logging
import os
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from yaml import warnings

from ludwig.api import EvaluationFrequency, TrainingStats
from ludwig.api_annotations import DeveloperAPI
from ludwig.backend import LOCAL_BACKEND
from ludwig.constants import SPLIT
from ludwig.utils import visualization_utils  # noqa: F401  (re-exported for submodules)
from ludwig.utils.data_utils import (
    CACHEABLE_FORMATS,
    data_reader_registry,
    figure_data_format_dataset,
    load_array,
    load_from_file,
    load_json,
    replace_file_extension,
)
from ludwig.utils.dataframe_utils import to_numpy_dataset, unflatten_df
from ludwig.utils.fs_utils import path_exists
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_PREDICTIONS_SUFFIX = "_predictions"
_PROBABILITIES_SUFFIX = "_probabilities"
_CSV_SUFFIX = "csv"
_PARQUET_SUFFIX = "parquet"


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------


def _convert_ground_truth(ground_truth, feature_metadata, ground_truth_apply_idx, positive_label):
    """Converts non-np.array representation to be np.array."""
    if "str2idx" in feature_metadata:
        # categorical output feature as binary
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

        # convert category index to binary representation
        ground_truth = ground_truth == positive_label
    else:
        # binary output feature
        if "str2bool" in feature_metadata:
            # non-standard boolean representation
            ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2bool"], ground_truth_apply_idx)
        else:
            # standard boolean representation
            ground_truth = ground_truth.values

        # ensure positive_label is 1 for binary feature
        positive_label = 1

    # convert to 0/1 representation and return
    return ground_truth.astype(int), positive_label


def _vectorize_ground_truth(
    ground_truth: pd.Series, str2idx: np.array, ground_truth_apply_idx: bool = True
) -> np.array:
    # raw hdf5 files generated during preprocessing don't need to be converted with str2idx
    if not ground_truth_apply_idx:
        return np.vectorize(lambda x, y: x)(ground_truth, str2idx)

    try:
        return np.vectorize(_encode_categorical_feature)(ground_truth, str2idx)
    except KeyError as e:
        logger.info(f"Unable to vectorize using str2idx with exception {e}. Falling back to ignoring str2idx")
        return np.vectorize(lambda x, y: x)(ground_truth, str2idx)


def _encode_categorical_feature(raw: np.array, str2idx: dict) -> np.array:
    """Encodes raw categorical string value to encoded numeric value.

    Args:
        raw: String categorical representation.
        str2idx: Dictionary that maps string representation to encoded value.

    Returns:
        Encoded numeric value.
    """
    return str2idx[raw]


def _get_ground_truth_df(ground_truth: str) -> DataFrame:
    # determine ground truth data format and get appropriate reader
    data_format = figure_data_format_dataset(ground_truth)
    if data_format not in CACHEABLE_FORMATS:
        raise ValueError(f"{data_format} is not supported for ground truth file, valid types are {CACHEABLE_FORMATS}")
    reader = get_from_registry(data_format, data_reader_registry)

    # retrieve ground truth from source data set
    if data_format in {"csv", "tsv"}:
        return reader(ground_truth, dtype=None, df_lib=pd)  # allow type inference
    return reader(ground_truth, df_lib=pd)


def _extract_ground_truth_values(
    ground_truth: "str | DataFrame",
    output_feature_name: str,
    ground_truth_split: int,
    split_file: "str | None" = None,
) -> pd.Series:
    """Helper function to extract ground truth values.

    Args:
        ground_truth: Path to source data containing ground truth or ground truth dataframe.
        output_feature_name: Output feature name for ground truth values.
        ground_truth_split: Dataset split to use for ground truth, defaults to 2.
        split_file: Optional file path to split values.

    Returns:
        Ground truth values from source data set.
    """
    ground_truth_df = _get_ground_truth_df(ground_truth) if isinstance(ground_truth, str) else ground_truth

    # extract ground truth for visualization
    if SPLIT in ground_truth_df:
        # get split value from source data set
        split = ground_truth_df[SPLIT]
        gt = ground_truth_df[output_feature_name][split == ground_truth_split]
    elif split_file is not None:
        # retrieve from split file
        if split_file.endswith(".csv"):
            # Legacy code path for previous split file format
            warnings.warn(
                "Using a CSV split file is deprecated and will be removed in a future version. "
                "Please retrain or convert to Parquet",
                DeprecationWarning,
            )
            split = load_array(split_file)
            mask = split == ground_truth_split
        else:
            split = pd.read_parquet(split_file)

            # Realign index from the split file with the ground truth to account for
            # dropped rows during preprocessing.
            # https://stackoverflow.com/a/65731168
            mask = split.iloc[:, 0] == ground_truth_split
            mask = mask.reindex(ground_truth_df.index, fill_value=False)

        gt = ground_truth_df[output_feature_name][mask]
    else:
        # use all the data in ground_truth
        gt = ground_truth_df[output_feature_name]

    return gt


def _get_cols_from_predictions(predictions_paths, cols, metadata):
    results_per_model = []
    for predictions_path in predictions_paths:
        pred_df = pd.read_parquet(predictions_path)

        shapes_fname = replace_file_extension(predictions_path, "shapes.json")
        if path_exists(shapes_fname):
            column_shapes = load_json(shapes_fname)
            pred_df = unflatten_df(pred_df, column_shapes, LOCAL_BACKEND.df_engine)

        for col in cols:
            # Convert categorical features back to indices
            if col.endswith(_PREDICTIONS_SUFFIX):
                feature_name = col[: -len(_PREDICTIONS_SUFFIX)]
                feature_metadata = metadata[feature_name]
                if "str2idx" in feature_metadata:
                    pred_df[col] = pred_df[col].map(lambda x: feature_metadata["str2idx"][x])

        pred_df = to_numpy_dataset(pred_df, LOCAL_BACKEND)
        results_per_model += [pred_df[col] for col in cols]

    return results_per_model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@DeveloperAPI
def validate_conf_thresholds_and_probabilities_2d_3d(probabilities, threshold_output_feature_names):
    """Ensure probabilities and threshold output_feature_names arrays have two members each.

    Args:
        probabilities: List of probabilities per model.
        threshold_output_feature_names: List of threshold output_feature_names per model.

    Raises:
        RuntimeError: If either list does not contain exactly two members.
    """
    validation_mapping = {
        "probabilities": probabilities,
        "threshold_output_feature_names": threshold_output_feature_names,
    }
    for item, value in validation_mapping.items():
        item_len = len(value)
        if item_len != 2:
            exception_message = f"Two {item} should be provided - {item_len} was given."
            logger.error(exception_message)
            raise RuntimeError(exception_message)


@DeveloperAPI
def load_data_for_viz(load_type, model_file_statistics, dtype=int, ground_truth_split=2) -> "dict[str, Any]":
    """Load JSON files (training stats, evaluation stats...) for a list of models.

    Args:
        load_type: Type of the data loader to be used.
        model_file_statistics: JSON file or list of json files containing any model experiment stats.

    Returns:
        List of training statistics loaded as json objects.
    """
    supported_load_types = {
        "load_json": load_json,
        "load_from_file": partial(load_from_file, dtype=dtype, ground_truth_split=ground_truth_split),
    }
    loader = supported_load_types[load_type]
    # Loads training stats from JSON file(s).
    try:
        stats_per_model = [loader(stats_f) for stats_f in model_file_statistics]
    except (TypeError, AttributeError):
        logger.exception(f"Unable to open model statistics file {model_file_statistics}!")
        raise
    return stats_per_model


def _load_training_stats(data: dict) -> TrainingStats:
    """Construct a TrainingStats from a dict loaded from JSON."""
    eval_freq = data.get("evaluation_frequency")
    if isinstance(eval_freq, dict):
        eval_freq = EvaluationFrequency(**eval_freq)
    elif eval_freq is None:
        eval_freq = EvaluationFrequency()
    return TrainingStats(
        training=data.get("training", {}),
        validation=data.get("validation", {}),
        test=data.get("test", {}),
        evaluation_frequency=eval_freq,
    )


@DeveloperAPI
def load_training_stats_for_viz(load_type, model_file_statistics, dtype=int, ground_truth_split=2) -> TrainingStats:
    """Load model file data (specifically training stats) for a list of models.

    Args:
        load_type: Type of the data loader to be used.
        model_file_statistics: JSON file or list of json files containing any model experiment stats.

    Returns:
        List of model statistics loaded as TrainingStats objects.
    """
    stats_per_model = load_data_for_viz(
        load_type, model_file_statistics, dtype=dtype, ground_truth_split=ground_truth_split
    )
    try:
        stats_per_model = [_load_training_stats(j) for j in stats_per_model]
    except Exception:
        logger.exception(f"Failed to load model statistics {model_file_statistics}!")
        raise
    return stats_per_model


@DeveloperAPI
def convert_to_list(item):
    """If item is not list class instance or None put inside a list.

    Args:
        item: Object to be checked and converted.

    Returns:
        Original item if it is a list instance or list containing the item.
    """
    return item if item is None or isinstance(item, list) else [item]


def _validate_output_feature_name_from_train_stats(output_feature_name, train_stats_per_model):
    """Validate prediction output_feature_name from model train stats and return it as list.

    Args:
        output_feature_name: Output feature name containing ground truth.
        train_stats_per_model: List of per model train stats.

    Returns:
        List of output_feature_name(s) containing ground truth.
    """
    output_feature_names_set = set()
    for train_stats in train_stats_per_model:
        for key in itertools.chain(train_stats.training.keys(), train_stats.validation.keys(), train_stats.test.keys()):
            output_feature_names_set.add(key)
    try:
        if output_feature_name in output_feature_names_set:
            return [output_feature_name]
        else:
            return output_feature_names_set
    # raised if output_feature_name is empty iterable (e.g. [] in set())
    except TypeError:
        return output_feature_names_set


def _validate_output_feature_name_from_test_stats(output_feature_name, test_stats_per_model):
    """Validate prediction output_feature_name from model test stats and return it as list.

    Args:
        output_feature_name: Output feature name containing ground truth.
        test_stats_per_model: List of per model test stats.

    Returns:
        List of output_feature_name(s) containing ground truth.
    """
    output_feature_names_set = set()
    for ls in test_stats_per_model:
        for key in ls:
            output_feature_names_set.add(key)
    try:
        if output_feature_name in output_feature_names_set:
            return [output_feature_name]
        else:
            return output_feature_names_set
    # raised if output_feature_name is empty iterable (e.g. [] in set())
    except TypeError:
        return output_feature_names_set


@DeveloperAPI
def generate_filename_template_path(output_dir, filename_template):
    """Ensure path to template file can be constructed given an output dir.

    Create output directory if it does not yet exist.

    Args:
        output_dir: Directory that will contain the filename_template file.
        filename_template: Name of the file template to be appended to the filename template path.

    Returns:
        Path to filename template inside the output dir or None if the output dir is None.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename_template)
    return None
