#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
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
import argparse
import itertools
import logging
import os
import sys
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import entropy
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from yaml import warnings

from ludwig.api import TrainingStats
from ludwig.api_annotations import DeveloperAPI, PublicAPI
from ludwig.backend import LOCAL_BACKEND
from ludwig.callbacks import Callback
from ludwig.constants import ACCURACY, EDIT_DISTANCE, HITS_AT_K, LOSS, PREDICTIONS, SPACE, SPLIT
from ludwig.contrib import add_contrib_callback_args
from ludwig.utils import visualization_utils
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
from ludwig.utils.print_utils import get_logging_level_registry
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)

_PREDICTIONS_SUFFIX = "_predictions"
_PROBABILITIES_SUFFIX = "_probabilities"
_CSV_SUFFIX = "csv"
_PARQUET_SUFFIX = "parquet"


def _convert_ground_truth(ground_truth, feature_metadata, ground_truth_apply_idx, positive_label):
    """converts non-np.array representation to be np.array."""
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


@DeveloperAPI
def validate_conf_thresholds_and_probabilities_2d_3d(probabilities, threshold_output_feature_names):
    """Ensure probabilities and threshold output_feature_names arrays have two members each.

    :param probabilities: List of probabilities per model
    :param threshhold_output_feature_names: List of threshhold output_feature_names per model
    :raise: RuntimeError
    """
    validation_mapping = {
        "probabilities": probabilities,
        "threshold_output_feature_names": threshold_output_feature_names,
    }
    for item, value in validation_mapping.items():
        item_len = len(value)
        if item_len != 2:
            exception_message = "Two {} should be provided - " "{} was given.".format(item, item_len)
            logger.error(exception_message)
            raise RuntimeError(exception_message)


@DeveloperAPI
def load_data_for_viz(load_type, model_file_statistics, dtype=int, ground_truth_split=2) -> Dict[str, Any]:
    """Load JSON files (training stats, evaluation stats...) for a list of models.

    :param load_type: type of the data loader to be used.
    :param model_file_statistics: JSON file or list of json files containing any
           model experiment stats.
    :return List of training statistics loaded as json objects.
    """
    supported_load_types = dict(
        load_json=load_json,
        load_from_file=partial(load_from_file, dtype=dtype, ground_truth_split=ground_truth_split),
    )
    loader = supported_load_types[load_type]
    # Loads training stats from JSON file(s).
    try:
        stats_per_model = [loader(stats_f) for stats_f in model_file_statistics]
    except (TypeError, AttributeError):
        logger.exception(f"Unable to open model statistics file {model_file_statistics}!")
        raise
    return stats_per_model


@DeveloperAPI
def load_training_stats_for_viz(load_type, model_file_statistics, dtype=int, ground_truth_split=2) -> TrainingStats:
    """Load model file data (specifically training stats) for a list of models.

    :param load_type: type of the data loader to be used.
    :param model_file_statistics: JSON file or list of json files containing any
           model experiment stats.
    :return List of model statistics loaded as TrainingStats objects.
    """
    stats_per_model = load_data_for_viz(
        load_type, model_file_statistics, dtype=dtype, ground_truth_split=ground_truth_split
    )
    try:
        stats_per_model = [TrainingStats.Schema().load(j) for j in stats_per_model]
    except Exception:
        logger.exception(f"Failed to load model statistics {model_file_statistics}!")
        raise
    return stats_per_model


@DeveloperAPI
def convert_to_list(item):
    """If item is not list class instance or None put inside a list.

    :param item: object to be checked and converted
    :return: original item if it is a list instance or list containing the item.
    """
    return item if item is None or isinstance(item, list) else [item]


def _validate_output_feature_name_from_train_stats(output_feature_name, train_stats_per_model):
    """Validate prediction output_feature_name from model train stats and return it as list.

    :param output_feature_name: output_feature_name containing ground truth
    :param train_stats_per_model: list of per model train stats
    :return output_feature_names: list of output_feature_name(s) containing ground truth
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

    :param output_feature_name: output_feature_name containing ground truth
    :param test_stats_per_model: list of per model test stats
    :return output_feature_names: list of output_feature_name(s) containing ground truth
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


def _encode_categorical_feature(raw: np.array, str2idx: dict) -> np.array:
    """encodes raw categorical string value to encoded numeric value.

    Args:
    :param raw: (np.array) string categorical representation
    :param str2idx: (dict) dictionary that maps string representation to
        encoded value.

    Returns:
        np.array
    """
    return str2idx[raw]


def _get_ground_truth_df(ground_truth: str) -> DataFrame:
    # determine ground truth data format and get appropriate reader
    data_format = figure_data_format_dataset(ground_truth)
    if data_format not in CACHEABLE_FORMATS:
        raise ValueError(
            "{} is not supported for ground truth file, " "valid types are {}".format(data_format, CACHEABLE_FORMATS)
        )
    reader = get_from_registry(data_format, data_reader_registry)

    # retrieve ground truth from source data set
    if data_format in {"csv", "tsv"}:
        return reader(ground_truth, dtype=None, df_lib=pd)  # allow type inference
    return reader(ground_truth, df_lib=pd)


def _extract_ground_truth_values(
    ground_truth: Union[str, DataFrame],
    output_feature_name: str,
    ground_truth_split: int,
    split_file: Union[str, None] = None,
) -> pd.Series:
    """Helper function to extract ground truth values.

    Args:
    :param ground_truth: (str, DataFrame) path to source data containing ground truth or ground truth dataframe
    :param output_feature_name: (str) output feature name for ground
        truth values.
    :param ground_truth_split: (int) dataset split to use for ground truth,
        defaults to 2.
    :param split_file: (Union[str, None]) optional file path to split values.

    # Return

    :return pd.Series: ground truth values from source data set
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
                "Using a CSV split file is deprecated and will be removed in v0.7. "
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


@DeveloperAPI
def generate_filename_template_path(output_dir, filename_template):
    """Ensure path to template file can be constructed given an output dir.

    Create output directory if yet does exist.
    :param output_dir: Directory that will contain the filename_template file
    :param filename_template: name of the file template to be appended to the
            filename template path
    :return: path to filename template inside the output dir or None if the
             output dir is None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename_template)
    return None


@DeveloperAPI
def compare_performance_cli(test_statistics: Union[str, List[str]], **kwargs: dict) -> None:
    """Load model data from files to be shown by compare_performance.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test
        statistics file.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    compare_performance(test_stats_per_model, **kwargs)


@DeveloperAPI
def learning_curves_cli(training_statistics: Union[str, List[str]], **kwargs: dict) -> None:
    """Load model data from files to be shown by learning_curves.

    # Inputs

    :param training_statistics: (Union[str, List[str]]) path to experiment
        training statistics file
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    train_stats_per_model = load_training_stats_for_viz("load_json", training_statistics)
    learning_curves(train_stats_per_model, **kwargs)


@DeveloperAPI
def compare_classifiers_performance_from_prob_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_classifiers_from_prob.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
        results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return
    :return None:
    """

    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # translate string to encoded numeric value
    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(
        ground_truth, output_feature_name, ground_truth_split, split_file=split_file
    )

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)

    compare_classifiers_performance_from_prob(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def compare_classifiers_performance_from_pred_cli(
    predictions: List[str],
    ground_truth: str,
    ground_truth_metadata: str,
    ground_truth_split: int,
    split_file: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_classifiers_from_pred.

    # Inputs

    :param predictions: (List[str]) list of prediction results file names
        to extract predictions from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
        results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PREDICTIONS_SUFFIX}"
    predictions_per_model = _get_cols_from_predictions(predictions, [col], metadata)

    compare_classifiers_performance_from_pred(
        predictions_per_model, ground_truth, metadata, output_feature_name, output_directory=output_directory, **kwargs
    )


@DeveloperAPI
def compare_classifiers_performance_subset_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_classifiers_subset.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)

    compare_classifiers_performance_subset(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def compare_classifiers_performance_changing_k_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_classifiers_changing_k.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    compare_classifiers_performance_changing_k(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def compare_classifiers_multiclass_multimetric_cli(
    test_statistics: Union[str, List[str]], ground_truth_metadata: str, **kwargs: dict
) -> None:
    """Load model data from files to be shown by compare_classifiers_multiclass.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test
        statistics file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    metadata = load_json(ground_truth_metadata)
    compare_classifiers_multiclass_multimetric(test_stats_per_model, metadata=metadata, **kwargs)


@DeveloperAPI
def compare_classifiers_predictions_cli(
    predictions: List[str],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_classifiers_predictions.

    # Inputs

    :param predictions: (List[str]) list of prediction results file names
        to extract predictions from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PREDICTIONS_SUFFIX}"
    predictions_per_model = _get_cols_from_predictions(predictions, [col], metadata)

    compare_classifiers_predictions(
        predictions_per_model, ground_truth, metadata, output_feature_name, output_directory=output_directory, **kwargs
    )


@DeveloperAPI
def compare_classifiers_predictions_distribution_cli(
    predictions: List[str],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_predictions_distribution.

    # Inputs

    :param predictions: (List[str]) list of prediction results file names
        to extract predictions from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PREDICTIONS_SUFFIX}"
    predictions_per_model = _get_cols_from_predictions(predictions, [col], metadata)
    compare_classifiers_predictions_distribution(
        predictions_per_model, ground_truth, metadata, output_feature_name, output_directory=output_directory, **kwargs
    )


@DeveloperAPI
def confidence_thresholding_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by confidence_thresholding.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    confidence_thresholding(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def confidence_thresholding_data_vs_acc_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by confidence_thresholding_data_vs_acc_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    confidence_thresholding_data_vs_acc(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def confidence_thresholding_data_vs_acc_subset_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by confidence_thresholding_data_vs_acc_subset.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    confidence_thresholding_data_vs_acc_subset(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def confidence_thresholding_data_vs_acc_subset_per_class_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_metadata: str,
    ground_truth_split: int,
    split_file: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by compare_classifiers_multiclass.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    confidence_thresholding_data_vs_acc_subset_per_class(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def confidence_thresholding_2thresholds_2d_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    threshold_output_feature_names: List[str],
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by confidence_thresholding_2thresholds_2d_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param threshold_output_feature_names: (List[str]) name of the output
        feature to visualizes.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth0 = _extract_ground_truth_values(
        ground_truth, threshold_output_feature_names[0], ground_truth_split, split_file
    )

    ground_truth1 = _extract_ground_truth_values(
        ground_truth, threshold_output_feature_names[1], ground_truth_split, split_file
    )

    cols = [f"{feature_name}{_PROBABILITIES_SUFFIX}" for feature_name in threshold_output_feature_names]
    probabilities_per_model = _get_cols_from_predictions(probabilities, cols, metadata)

    confidence_thresholding_2thresholds_2d(
        probabilities_per_model,
        [ground_truth0, ground_truth1],
        metadata,
        threshold_output_feature_names,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def confidence_thresholding_2thresholds_3d_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    threshold_output_feature_names: List[str],
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by confidence_thresholding_2thresholds_3d_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param threshold_output_feature_names: (List[str]) name of the output
        feature to visualizes.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth0 = _extract_ground_truth_values(
        ground_truth, threshold_output_feature_names[0], ground_truth_split, split_file
    )

    ground_truth1 = _extract_ground_truth_values(
        ground_truth, threshold_output_feature_names[1], ground_truth_split, split_file
    )

    cols = [f"{feature_name}{_PROBABILITIES_SUFFIX}" for feature_name in threshold_output_feature_names]
    probabilities_per_model = _get_cols_from_predictions(probabilities, cols, metadata)
    confidence_thresholding_2thresholds_3d(
        probabilities_per_model,
        [ground_truth0, ground_truth1],
        metadata,
        threshold_output_feature_names,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def binary_threshold_vs_metric_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by binary_threshold_vs_metric_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """

    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    binary_threshold_vs_metric(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def precision_recall_curves_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by precision_recall_curves_cli.

    Args

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    Return

    :return None:
    """
    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    precision_recall_curves(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def roc_curves_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by roc_curves_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file.
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """

    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    roc_curves(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def roc_curves_from_test_statistics_cli(test_statistics: Union[str, List[str]], **kwargs: dict) -> None:
    """Load model data from files to be shown by roc_curves_from_test_statistics_cli.

    # Inputs
    :param test_statistics: (Union[str, List[str]]) path to experiment test
        statistics file.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    roc_curves_from_test_statistics(test_stats_per_model, **kwargs)


@DeveloperAPI
def precision_recall_curves_from_test_statistics_cli(test_statistics: Union[str, List[str]], **kwargs: dict) -> None:
    """Load model data from files to be shown by precision_recall_curves_from_test_statistics_cli.

    Args:

    :param test_statistics: (Union[str, List[str]]) path to experiment test
        statistics file.
    :param kwargs: (dict) parameters for the requested visualizations.

    Return:

    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    precision_recall_curves_from_test_statistics(test_stats_per_model, **kwargs)


@DeveloperAPI
def calibration_1_vs_all_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    output_feature_proc_name: Optional[str] = None,
    ground_truth_apply_idx: bool = True,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by calibration_1_vs_all_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param output_feature_proc_name: (str) name of the output feature column in ground_truth. If ground_truth is a
        preprocessed parquet or hdf5 file, the column name will be <output_feature>_<hash>
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """

    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(
        ground_truth, output_feature_proc_name or output_feature_name, ground_truth_split, split_file
    )
    feature_metadata = metadata[output_feature_name]
    ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    calibration_1_vs_all(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def calibration_multiclass_cli(
    probabilities: Union[str, List[str]],
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    **kwargs: dict,
) -> None:
    """Load model data from files to be shown by calibration_multiclass_cli.

    # Inputs

    :param probabilities: (Union[str, List[str]]) list of prediction results file names
        to extract probabilities from.
    :param ground_truth: (str) path to ground truth file
    :param ground_truth_split: (str) type of ground truth split -
        `0` for training split, `1` for validation split or
        2 for `'test'` split.
    :param split_file: (str, None) file path to csv file containing split values
    :param ground_truth_metadata: (str) file path to feature metadata json file
        created during training.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param output_directory: (str) name of output directory containing training
         results.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """

    # retrieve feature metadata to convert raw predictions to encoded value
    metadata = load_json(ground_truth_metadata)

    # retrieve ground truth from source data set
    ground_truth = _extract_ground_truth_values(ground_truth, output_feature_name, ground_truth_split, split_file)

    col = f"{output_feature_name}{_PROBABILITIES_SUFFIX}"
    probabilities_per_model = _get_cols_from_predictions(probabilities, [col], metadata)
    calibration_multiclass(
        probabilities_per_model,
        ground_truth,
        metadata,
        output_feature_name,
        output_directory=output_directory,
        **kwargs,
    )


@DeveloperAPI
def confusion_matrix_cli(test_statistics: Union[str, List[str]], ground_truth_metadata: str, **kwargs: dict) -> None:
    """Load model data from files to be shown by confusion_matrix.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test
        statistics file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    metadata = load_json(ground_truth_metadata)
    confusion_matrix(test_stats_per_model, metadata, **kwargs)


@DeveloperAPI
def frequency_vs_f1_cli(test_statistics: Union[str, List[str]], ground_truth_metadata: str, **kwargs: dict) -> None:
    """Load model data from files to be shown by frequency_vs_f1.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test
        statistics file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param kwargs: (dict) parameters for the requested visualizations.

    # Return

    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    metadata = load_json(ground_truth_metadata)
    frequency_vs_f1(test_stats_per_model, metadata, **kwargs)


@DeveloperAPI
def learning_curves(
    train_stats_per_model: List[dict],
    output_feature_name: Union[str, None] = None,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    callbacks: List[Callback] = None,
    **kwargs,
) -> None:
    """Show how model metrics change over training and validation data epochs.

    For each model and for each output feature and metric of the model,
    it produces a line plot showing how that metric changed over the course
    of the epochs of training on the training and validation sets.

    # Inputs

    :param train_stats_per_model: (List[dict]) list containing dictionary of
        training statistics per model.
    :param output_feature_name: (Union[str, `None`], default: `None`) name of the output feature
        to use for the visualization.  If `None`, use all output features.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param callbacks: (list, default: `None`) a list of
        `ludwig.callbacks.Callback` objects that provide hooks into the
        Ludwig pipeline.

    # Return
    :return: (None)
    """
    filename_template = "learning_curves_{}_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    train_stats_per_model_list = convert_to_list(train_stats_per_model)
    model_names_list = convert_to_list(model_names)
    output_feature_names = _validate_output_feature_name_from_train_stats(
        output_feature_name, train_stats_per_model_list
    )

    metrics = [LOSS, ACCURACY, HITS_AT_K, EDIT_DISTANCE]
    for output_feature_name in output_feature_names:
        for metric in metrics:
            if metric in train_stats_per_model_list[0].training[output_feature_name]:
                filename = None
                if filename_template_path:
                    filename = filename_template_path.format(output_feature_name, metric)

                training_stats = [
                    learning_stats.training[output_feature_name][metric]
                    for learning_stats in train_stats_per_model_list
                ]

                validation_stats = []
                for learning_stats in train_stats_per_model_list:
                    if learning_stats.validation and output_feature_name in learning_stats.validation:
                        validation_stats.append(learning_stats.validation[output_feature_name][metric])
                    else:
                        validation_stats.append(None)

                evaluation_frequency = train_stats_per_model_list[0].evaluation_frequency

                visualization_utils.learning_curves_plot(
                    training_stats,
                    validation_stats,
                    metric,
                    x_label=evaluation_frequency.period,
                    x_step=evaluation_frequency.frequency,
                    algorithm_names=model_names_list,
                    title=f"Learning Curves {output_feature_name}",
                    filename=filename,
                    callbacks=callbacks,
                )


@DeveloperAPI
def compare_performance(
    test_stats_per_model: List[dict],
    output_feature_name: Union[str, None] = None,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Produces model comparison barplot visualization for each overall metric.

    For each model (in the aligned lists of test_statistics and model_names)
    it produces bars in a bar plot, one for each overall metric available
    in the test_statistics file for the specified output_feature_name.

    # Inputs

    :param test_stats_per_model: (List[dict]) dictionary containing evaluation
        performance statistics.
    :param output_feature_name: (Union[str, `None`], default: `None`) name of the output feature
        to use for the visualization.  If `None`, use all output features.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)

    # Example usage:

    ```python
    model_a = LudwigModel(config)
    model_a.train(dataset)
    a_evaluation_stats, _, _ = model_a.evaluate(eval_set)
    model_b = LudwigModel.load("path/to/model/")
    b_evaluation_stats, _, _ = model_b.evaluate(eval_set)
    compare_performance([a_evaluation_stats, b_evaluation_stats], model_names=["A", "B"])
    ```
    """
    ignore_names = ["overall_stats", "confusion_matrix", "per_class_stats", "predictions", "probabilities"]

    filename_template = "compare_performance_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)

    test_stats_per_model_list = convert_to_list(test_stats_per_model)
    model_names_list = convert_to_list(model_names)
    output_feature_names = _validate_output_feature_name_from_test_stats(output_feature_name, test_stats_per_model_list)

    for output_feature_name in output_feature_names:
        metric_names_sets = list(set(tspr[output_feature_name].keys()) for tspr in test_stats_per_model_list)
        metric_names = metric_names_sets[0]
        for metric_names_set in metric_names_sets:
            metric_names = metric_names.intersection(metric_names_set)
        metric_names.remove(LOSS)
        for name in ignore_names:
            if name in metric_names:
                metric_names.remove(name)
        metrics_dict = {name: [] for name in metric_names}

        for test_stats_per_model in test_stats_per_model_list:
            for metric_name in metric_names:
                metrics_dict[metric_name].append(test_stats_per_model[output_feature_name][metric_name])

        # are there any metrics to compare?
        if metrics_dict:
            metrics = []
            metrics_names = []
            min_val = float("inf")
            max_val = float("-inf")
            for metric_name, metric_vals in metrics_dict.items():
                if len(metric_vals) > 0:
                    metrics.append(metric_vals)
                    metrics_names.append(metric_name)
                    curr_min = min(metric_vals)
                    if curr_min < min_val:
                        min_val = curr_min
                    curr_max = max(metric_vals)
                    if curr_max > max_val:
                        max_val = curr_max

            filename = None

            if filename_template_path:
                filename = filename_template_path.format(output_feature_name)
                os.makedirs(output_directory, exist_ok=True)

            visualization_utils.compare_classifiers_plot(
                metrics,
                metrics_names,
                model_names_list,
                adaptive=min_val < 0 or max_val > 1,
                title=f"Performance comparison on {output_feature_name}",
                filename=filename,
            )


@DeveloperAPI
def compare_classifiers_performance_from_prob(
    probabilities_per_model: List[np.ndarray],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int = 0,
    top_n_classes: Union[List[int], int] = 3,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Produces model comparison barplot visualization from probabilities.

    For each model it produces bars in a bar plot, one for each overall metric
    computed on the fly from the probabilities of predictions for the specified
    `model_names`.

    # Inputs

    :param probabilities_per_model: (List[np.ndarray]) path to experiment
        probabilities file
    :param ground_truth: (pd.Series) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param top_n_classes: (List[int]) list containing the number of classes
        to plot.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """

    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    top_n_classes_list = convert_to_list(top_n_classes)
    k = top_n_classes_list[0]
    model_names_list = convert_to_list(model_names)
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit

    probs = probabilities_per_model
    accuracies = []
    hits_at_ks = []
    mrrs = []

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)
        top1 = prob[:, -1]
        topk = prob[:, -k:]

        accuracies.append((ground_truth == top1).sum() / len(ground_truth))

        hits_at_k = 0
        for j in range(len(ground_truth)):
            hits_at_k += np.in1d(ground_truth[j], topk[j])
        hits_at_ks.append(hits_at_k.item() / len(ground_truth))

        mrr = 0
        for j in range(len(ground_truth)):
            ground_truth_pos_in_probs = prob[j] == ground_truth[j]
            if np.any(ground_truth_pos_in_probs):
                mrr += 1 / -(np.argwhere(ground_truth_pos_in_probs).item() - prob.shape[1])
        mrrs.append(mrr / len(ground_truth))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "compare_classifiers_performance_from_prob." + file_format)

    visualization_utils.compare_classifiers_plot(
        [accuracies, hits_at_ks, mrrs], [ACCURACY, HITS_AT_K, "mrr"], model_names_list, filename=filename
    )


@DeveloperAPI
def compare_classifiers_performance_from_pred(
    predictions_per_model: List[np.ndarray],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Produces model comparison barplot visualization from predictions.

    For each model it produces bars in a bar plot, one for each overall metric
    computed on the fly from the predictions for the specified
    `model_names`.

    # Inputs

    :param predictions_per_model: (List[str]) path to experiment predictions file.
    :param ground_truth: (pd.Series) ground truth values
    :param metadata: (dict) feature metadata dictionary.
    :param output_feature_name: (str) name of the output feature to visualize.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """

    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    predictions_per_model = [np.ndarray.flatten(np.array(pred)) for pred in predictions_per_model]

    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit

    preds = predictions_per_model
    model_names_list = convert_to_list(model_names)
    mapped_preds = []
    try:
        for pred in preds:
            mapped_preds.append([metadata[output_feature_name]["str2idx"][val] for val in pred])
        preds = mapped_preds
    # If predictions are coming from npy file there is no need to convert to
    # numeric labels using metadata
    except (TypeError, KeyError):
        pass
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    for i, pred in enumerate(preds):
        accuracies.append(sklearn.metrics.accuracy_score(ground_truth, pred))
        precisions.append(sklearn.metrics.precision_score(ground_truth, pred, average="macro"))
        recalls.append(sklearn.metrics.recall_score(ground_truth, pred, average="macro"))
        f1s.append(sklearn.metrics.f1_score(ground_truth, pred, average="macro"))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "compare_classifiers_performance_from_pred." + file_format)

    visualization_utils.compare_classifiers_plot(
        [accuracies, precisions, recalls, f1s],
        [ACCURACY, "precision", "recall", "f1"],
        model_names_list,
        filename=filename,
    )


@DeveloperAPI
def compare_classifiers_performance_subset(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    top_n_classes: List[int],
    labels_limit: (int),
    subset: str,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Produces model comparison barplot visualization from train subset.

    For each model  it produces bars in a bar plot, one for each overall metric
     computed on the fly from the probabilities predictions for the
     specified `model_names`, considering only a subset of the full training set.
     The way the subset is obtained is using the `top_n_classes` and
     `subset` parameters.

     # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param top_n_classes: (List[int]) list containing the number of classes
        to plot.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param subset: (str) string specifying type of subset filtering.  Valid
        values are `ground_truth` or `predictions`.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    top_n_classes_list = convert_to_list(top_n_classes)
    k = top_n_classes_list[0]
    model_names_list = convert_to_list(model_names)
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit

    subset_indices = ground_truth > 0
    gt_subset = ground_truth
    if subset == "ground_truth":
        subset_indices = ground_truth < k
        gt_subset = ground_truth[subset_indices]
        logger.info(f"Subset is {len(gt_subset) / len(ground_truth) * 100:.2f}% of the data")

    probs = probabilities_per_model
    accuracies = []
    hits_at_ks = []

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        if subset == PREDICTIONS:
            subset_indices = np.argmax(prob, axis=1) < k
            gt_subset = ground_truth[subset_indices]
            logger.info(
                "Subset for model_name {} is {:.2f}% of the data".format(
                    model_names[i] if model_names and i < len(model_names) else i,
                    len(gt_subset) / len(ground_truth) * 100,
                )
            )
            model_names[i] = "{} ({:.2f}%)".format(
                model_names[i] if model_names and i < len(model_names) else i, len(gt_subset) / len(ground_truth) * 100
            )

        prob_subset = prob[subset_indices]

        prob_subset = np.argsort(prob_subset, axis=1)
        top1_subset = prob_subset[:, -1]
        top3_subset = prob_subset[:, -3:]

        accuracies.append(np.sum(gt_subset == top1_subset) / len(gt_subset))

        hits_at_k = 0
        for j in range(len(gt_subset)):
            hits_at_k += np.in1d(gt_subset[j], top3_subset[i, :])
        hits_at_ks.append(hits_at_k.item() / len(gt_subset))

    title = None
    if subset == "ground_truth":
        title = "Classifier performance on first {} class{} ({:.2f}%)".format(
            k, "es" if k > 1 else "", len(gt_subset) / len(ground_truth) * 100
        )
    elif subset == PREDICTIONS:
        title = "Classifier performance on first {} class{}".format(k, "es" if k > 1 else "")

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "compare_classifiers_performance_subset." + file_format)

    visualization_utils.compare_classifiers_plot(
        [accuracies, hits_at_ks], [ACCURACY, HITS_AT_K], model_names_list, title=title, filename=filename
    )


@DeveloperAPI
def compare_classifiers_performance_changing_k(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    top_k: int,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Produce lineplot that show Hits@K metric while k goes from 1 to `top_k`.

    For each model it produces a line plot that shows the Hits@K metric
    (that counts a prediction as correct if the model produces it among the
    first k) while changing k from 1 to top_k for the specified
    `output_feature_name`.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param top_k: (int) number of elements in the ranklist to consider.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    k = top_k
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
    probs = probabilities_per_model

    hits_at_ks = []
    model_names_list = convert_to_list(model_names)
    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)

        hits_at_k = [0.0] * k
        for g in range(len(ground_truth)):
            for j in range(k):
                hits_at_k[j] += np.in1d(ground_truth[g], prob[g, -j - 1 :])
        hits_at_ks.append(np.array(hits_at_k) / len(ground_truth))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "compare_classifiers_performance_changing_k." + file_format)

    visualization_utils.compare_classifiers_line_plot(
        np.arange(1, k + 1),
        hits_at_ks,
        "hits@k",
        model_names_list,
        title="Classifier comparison (hits@k)",
        filename=filename,
    )


@DeveloperAPI
def compare_classifiers_multiclass_multimetric(
    test_stats_per_model: List[dict],
    metadata: dict,
    output_feature_name: str,
    top_n_classes: List[int],
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show the precision, recall and F1 of the model for the specified output_feature_name.

    For each model it produces four plots that show the precision,
    recall and F1 of the model on several classes for the specified output_feature_name.

    # Inputs

    :param test_stats_per_model: (List[dict]) list containing dictionary of
        evaluation performance statistics
    :param metadata: (dict) intermediate preprocess structure created during
        training containing the mappings of the input dataset.
    :param output_feature_name: (Union[str, `None`]) name of the output feature
        to use for the visualization.  If `None`, use all output features.
    :param top_n_classes: (List[int]) list containing the number of classes
        to plot.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return
    :return: (None)
    """
    filename_template = "compare_classifiers_multiclass_multimetric_{}_{}_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)

    test_stats_per_model_list = convert_to_list(test_stats_per_model)
    model_names_list = convert_to_list(model_names)
    output_feature_names = _validate_output_feature_name_from_test_stats(output_feature_name, test_stats_per_model_list)

    for i, test_statistics in enumerate(test_stats_per_model_list):
        for output_feature_name in output_feature_names:
            model_name_name = model_names_list[i] if model_names_list is not None and i < len(model_names_list) else ""
            if "per_class_stats" not in test_statistics[output_feature_name]:
                logger.warning(
                    f"The output_feature_name {output_feature_name} in test statistics does not contain "
                    + "per_class_stats, skipping it."
                )
                break
            per_class_stats = test_statistics[output_feature_name]["per_class_stats"]
            precisions = []
            recalls = []
            f1_scores = []
            labels = []
            for _, class_name in sorted(
                ((metadata[output_feature_name]["str2idx"][key], key) for key in per_class_stats.keys()),
                key=lambda tup: tup[0],
            ):
                class_stats = per_class_stats[class_name]
                precisions.append(class_stats["precision"])
                recalls.append(class_stats["recall"])
                f1_scores.append(class_stats["f1_score"])
                labels.append(class_name)
            for k in top_n_classes:
                k = min(k, len(precisions)) if k > 0 else len(precisions)
                ps = precisions[0:k]
                rs = recalls[0:k]
                fs = f1_scores[0:k]
                ls = labels[0:k]

                filename = None
                if filename_template_path:
                    os.makedirs(output_directory, exist_ok=True)
                    filename = filename_template_path.format(model_name_name, output_feature_name, f"top{k}")

                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [ps, rs, fs],
                    ["precision", "recall", "f1 score"],
                    labels=ls,
                    title="{} Multiclass Precision / Recall / "
                    "F1 Score top {} {}".format(model_name_name, k, output_feature_name),
                    filename=filename,
                )

                p_np = np.nan_to_num(np.array(precisions, dtype=np.float32))
                r_np = np.nan_to_num(np.array(recalls, dtype=np.float32))
                f1_np = np.nan_to_num(np.array(f1_scores, dtype=np.float32))
                labels_np = np.nan_to_num(np.array(labels))

                sorted_indices = f1_np.argsort()
                higher_f1s = sorted_indices[-k:][::-1]
                filename = None
                if filename_template_path:
                    os.makedirs(output_directory, exist_ok=True)
                    filename = filename_template_path.format(model_name_name, output_feature_name, f"best{k}")
                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [p_np[higher_f1s], r_np[higher_f1s], f1_np[higher_f1s]],
                    ["precision", "recall", "f1 score"],
                    labels=labels_np[higher_f1s].tolist(),
                    title="{} Multiclass Precision / Recall / "
                    "F1 Score best {} classes {}".format(model_name_name, k, output_feature_name),
                    filename=filename,
                )
                lower_f1s = sorted_indices[:k]
                filename = None
                if filename_template_path:
                    filename = filename_template_path.format(model_name_name, output_feature_name, f"worst{k}")
                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [p_np[lower_f1s], r_np[lower_f1s], f1_np[lower_f1s]],
                    ["precision", "recall", "f1 score"],
                    labels=labels_np[lower_f1s].tolist(),
                    title=(
                        f"{model_name_name} Multiclass Precision / Recall / F1 Score worst "
                        + f"{k} classes {output_feature_name}"
                    ),
                    filename=filename,
                )

                filename = None
                if filename_template_path:
                    filename = filename_template_path.format(model_name_name, output_feature_name, "sorted")
                visualization_utils.compare_classifiers_multiclass_multimetric_plot(
                    [p_np[sorted_indices[::-1]], r_np[sorted_indices[::-1]], f1_np[sorted_indices[::-1]]],
                    ["precision", "recall", "f1 score"],
                    labels=labels_np[sorted_indices[::-1]].tolist(),
                    title=f"{model_name_name} Multiclass Precision / Recall / F1 Score {output_feature_name} sorted",
                    filename=filename,
                )

                logger.info("\n")
                logger.info(model_name_name)
                tmp_str = f"{output_feature_name} best 5 classes: "
                tmp_str += "{}"
                logger.info(tmp_str.format(higher_f1s))
                logger.info(f1_np[higher_f1s])
                tmp_str = f"{output_feature_name} worst 5 classes: "
                tmp_str += "{}"
                logger.info(tmp_str.format(lower_f1s))
                logger.info(f1_np[lower_f1s])
                tmp_str = f"{output_feature_name} number of classes with f1 score > 0: "
                tmp_str += "{}"
                logger.info(tmp_str.format(np.sum(f1_np > 0)))
                tmp_str = f"{output_feature_name} number of classes with f1 score = 0: "
                tmp_str += "{}"
                logger.info(tmp_str.format(np.sum(f1_np == 0)))


@DeveloperAPI
def compare_classifiers_predictions(
    predictions_per_model: List[list],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show two models comparison of their output_feature_name predictions.

    # Inputs

    :param predictions_per_model: (List[list]) list containing the model
        predictions for the specified output_feature_name.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    model_names_list = convert_to_list(model_names)
    name_c1 = model_names_list[0] if model_names is not None and len(model_names) > 0 else "c1"
    name_c2 = model_names_list[1] if model_names is not None and len(model_names) > 1 else "c2"

    pred_c1 = predictions_per_model[0]
    pred_c2 = predictions_per_model[1]

    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
        pred_c1[pred_c1 > labels_limit] = labels_limit
        pred_c2[pred_c2 > labels_limit] = labels_limit

    # TODO all shadows built in name - come up with a more descriptive name
    all = len(ground_truth)
    if all == 0:
        logger.error("No labels in the ground truth")
        return

    both_right = 0
    both_wrong_same = 0
    both_wrong_different = 0
    c1_right_c2_wrong = 0
    c1_wrong_c2_right = 0

    for i in range(all):
        if ground_truth[i] == pred_c1[i] and ground_truth[i] == pred_c2[i]:
            both_right += 1
        elif ground_truth[i] != pred_c1[i] and ground_truth[i] != pred_c2[i]:
            if pred_c1[i] == pred_c2[i]:
                both_wrong_same += 1
            else:
                both_wrong_different += 1
        elif ground_truth[i] == pred_c1[i] and ground_truth[i] != pred_c2[i]:
            c1_right_c2_wrong += 1
        elif ground_truth[i] != pred_c1[i] and ground_truth[i] == pred_c2[i]:
            c1_wrong_c2_right += 1

    one_right = c1_right_c2_wrong + c1_wrong_c2_right
    both_wrong = both_wrong_same + both_wrong_different

    logger.info(f"Test datapoints: {all}")
    logger.info(f"Both right: {both_right} {100 * both_right / all:.2f}%")
    logger.info(f"One right: {one_right} {100 * one_right / all:.2f}%")
    logger.info(
        "  {} right / {} wrong: {} {:.2f}% {:.2f}%".format(
            name_c1,
            name_c2,
            c1_right_c2_wrong,
            100 * c1_right_c2_wrong / all,
            100 * c1_right_c2_wrong / one_right if one_right > 0 else 0,
        )
    )
    logger.info(
        "  {} wrong / {} right: {} {:.2f}% {:.2f}%".format(
            name_c1,
            name_c2,
            c1_wrong_c2_right,
            100 * c1_wrong_c2_right / all,
            100 * c1_wrong_c2_right / one_right if one_right > 0 else 0,
        )
    )
    logger.info(f"Both wrong: {both_wrong} {100 * both_wrong / all:.2f}%")
    logger.info(
        "  same prediction: {} {:.2f}% {:.2f}%".format(
            both_wrong_same, 100 * both_wrong_same / all, 100 * both_wrong_same / both_wrong if both_wrong > 0 else 0
        )
    )
    logger.info(
        "  different prediction: {} {:.2f}% {:.2f}%".format(
            both_wrong_different,
            100 * both_wrong_different / all,
            100 * both_wrong_different / both_wrong if both_wrong > 0 else 0,
        )
    )

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, f"compare_classifiers_predictions_{name_c1}_{name_c2}.{file_format}")

    visualization_utils.donut(
        [both_right, one_right, both_wrong],
        ["both right", "one right", "both wrong"],
        [both_right, c1_right_c2_wrong, c1_wrong_c2_right, both_wrong_same, both_wrong_different],
        [
            "both right",
            f"{name_c1} right / {name_c2} wrong",
            f"{name_c1} wrong / {name_c2} right",
            "same prediction",
            "different prediction",
        ],
        [0, 1, 1, 2, 2],
        title=f"{name_c1} vs {name_c2}",
        tight_layout=kwargs.pop("tight_layout", True),
        filename=filename,
    )


@DeveloperAPI
def compare_classifiers_predictions_distribution(
    predictions_per_model: List[list],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show comparision of models predictions distribution for 10 output_feature_name classes.

    This visualization produces a radar plot comparing the distributions of
    predictions of the models for the first 10 classes of the specified
    output_feature_name.

    # Inputs

    :param predictions_per_model: (List[list]) list containing the model
        predictions for the specified output_feature_name.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    model_names_list = convert_to_list(model_names)
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
        for i in range(len(predictions_per_model)):
            predictions_per_model[i][predictions_per_model[i] > labels_limit] = labels_limit

    max_gt = max(ground_truth)
    max_pred = max(max(alg_predictions) for alg_predictions in predictions_per_model)
    max_val = max(max_gt, max_pred) + 1

    counts_gt = np.bincount(ground_truth, minlength=max_val)
    prob_gt = counts_gt / counts_gt.sum()

    counts_predictions = [np.bincount(alg_predictions, minlength=max_val) for alg_predictions in predictions_per_model]

    prob_predictions = [
        alg_count_prediction / alg_count_prediction.sum() for alg_count_prediction in counts_predictions
    ]

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "compare_classifiers_predictions_distribution." + file_format)

    visualization_utils.radar_chart(prob_gt, prob_predictions, model_names_list, filename=filename)


@DeveloperAPI
def confidence_thresholding(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models accuracy and data coverage while increasing treshold.

    For each model it produces a pair of lines indicating the accuracy of
    the model and the data coverage while increasing a threshold (x axis) on
    the probabilities of predictions for the specified output_feature_name.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        max_prob = np.max(prob, axis=1)
        predictions = np.argmax(prob, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = ground_truth[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = (filtered_gt == filtered_predictions).sum() / len(filtered_gt)

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(ground_truth))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "confidence_thresholding." + file_format)

    visualization_utils.confidence_filtering_plot(
        thresholds, accuracies, dataset_kept, model_names_list, title="Confidence_Thresholding", filename=filename
    )


@DeveloperAPI
def confidence_thresholding_data_vs_acc(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models comparison of confidence threshold data vs accuracy.

    For each model it produces a line indicating the accuracy of the model
    and the data coverage while increasing a threshold on the probabilities
    of predictions for the specified output_feature_name. The difference with
    confidence_thresholding is that it uses two axes instead of three,
    not visualizing the threshold and having coverage as x axis instead of
    the threshold.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return
    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        max_prob = np.max(prob, axis=1)
        predictions = np.argmax(prob, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = ground_truth[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = (filtered_gt == filtered_predictions).sum() / len(filtered_gt)

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(ground_truth))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "confidence_thresholding_data_vs_acc." + file_format)

    visualization_utils.confidence_filtering_data_vs_acc_plot(
        accuracies,
        dataset_kept,
        model_names_list,
        title="Confidence_Thresholding (Data vs Accuracy)",
        filename=filename,
    )


@DeveloperAPI
def confidence_thresholding_data_vs_acc_subset(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    top_n_classes: List[int],
    labels_limit: int,
    subset: str,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models comparison of confidence threshold data vs accuracy on a subset of data.

    For each model it produces a line indicating the accuracy of the model
    and the data coverage while increasing a threshold on the probabilities
    of predictions for the specified output_feature_name, considering only a subset of the
    full training set. The way the subset is obtained is using the `top_n_classes`
    and subset parameters.
     The difference with confidence_thresholding is that it uses two axes
     instead of three, not visualizing the threshold and having coverage as
     x axis instead of the threshold.

    If the values of subset is `ground_truth`, then only datapoints where the
    ground truth class is within the top n most frequent ones will be
    considered  as test set, and the percentage of datapoints that have been
    kept  from the original set will be displayed. If the values of subset is
     `predictions`, then only datapoints where the the model predicts a class
     that is within the top n most frequent ones will be considered as test set,
     and the percentage of datapoints that have been kept from the original set
     will be displayed for each model.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param top_n_classes: (List[int]) list containing the number of classes
        to plot.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param subset: (str) string specifying type of subset filtering.  Valid
        values are `ground_truth` or `predictions`.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    top_n_classes_list = convert_to_list(top_n_classes)
    k = top_n_classes_list[0]
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    subset_indices = ground_truth > 0
    gt_subset = ground_truth
    if subset == "ground_truth":
        subset_indices = ground_truth < k
        gt_subset = ground_truth[subset_indices]
        logger.info(f"Subset is {len(gt_subset) / len(ground_truth) * 100:.2f}% of the data")

    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        if subset == PREDICTIONS:
            subset_indices = np.argmax(prob, axis=1) < k
            gt_subset = ground_truth[subset_indices]
            logger.info(
                "Subset for model_name {} is {:.2f}% of the data".format(
                    model_names[i] if model_names and i < len(model_names) else i,
                    len(gt_subset) / len(ground_truth) * 100,
                )
            )

        prob_subset = prob[subset_indices]

        max_prob = np.max(prob_subset, axis=1)
        predictions = np.argmax(prob_subset, axis=1)

        accuracies_alg = []
        dataset_kept_alg = []

        for threshold in thresholds:
            threshold = threshold if threshold < 1 else 0.999
            filtered_indices = max_prob >= threshold
            filtered_gt = gt_subset[filtered_indices]
            filtered_predictions = predictions[filtered_indices]
            accuracy = (filtered_gt == filtered_predictions).sum() / len(filtered_gt)

            accuracies_alg.append(accuracy)
            dataset_kept_alg.append(len(filtered_gt) / len(ground_truth))

        accuracies.append(accuracies_alg)
        dataset_kept.append(dataset_kept_alg)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "confidence_thresholding_data_vs_acc_subset." + file_format)

    visualization_utils.confidence_filtering_data_vs_acc_plot(
        accuracies,
        dataset_kept,
        model_names_list,
        title="Confidence_Thresholding (Data vs Accuracy)",
        filename=filename,
    )


@DeveloperAPI
def confidence_thresholding_data_vs_acc_subset_per_class(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    top_n_classes: Union[int, List[int]],
    labels_limit: int,
    subset: str,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models comparison of confidence threshold data vs accuracy on a subset of data per class in top n
    classes.

    For each model (in the aligned lists of probabilities and model_names)
    it produces a line indicating the accuracy of the model and the data
    coverage while increasing a threshold on the probabilities of
    predictions for the specified output_feature_name, considering only a subset of the
    full training set. The way the subset is obtained is using the
    `top_n_classes`  and `subset` parameters.  The difference with
    confidence_thresholding is that it uses two axes instead of three,
    not visualizing the threshold and having coverage as x axis instead of
    the  threshold.

    If the values of subset is `ground_truth`, then only datapoints where the
    ground truth class is within the top n most frequent ones will be
    considered  as test set, and the percentage of datapoints that have been
    kept from the original set will be displayed. If the values of subset is
    `predictions`, then only datapoints where the the model predicts a class that
    is within the top n most frequent ones will be considered as test set, and
    the percentage of datapoints that have been kept from the original set will
    be displayed for each model.

    The difference with confidence_thresholding_data_vs_acc_subset is that it
    produces one plot per class within the top_n_classes.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) intermediate preprocess structure created during
        training containing the mappings of the input dataset.
    :param output_feature_name: (str) name of the output feature to use
        for the visualization.
    :param top_n_classes: (Union[int, List[int]]) number of top classes or list
        containing the number of top classes to plot.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param subset: (str) string specifying type of subset filtering.  Valid
        values are `ground_truth` or `predictions`.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return
    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    filename_template = "confidence_thresholding_data_vs_acc_subset_per_class_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    top_n_classes_list = convert_to_list(top_n_classes)
    k = top_n_classes_list[0]
    # If top_n_classes is greater than the maximum number of tokens, truncate to use max token size
    if k > len(metadata[output_feature_name]["idx2str"]):
        k = len(metadata[output_feature_name]["idx2str"])
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)

    thresholds = [t / 100 for t in range(0, 101, 5)]

    for curr_k in range(k):
        accuracies = []
        dataset_kept = []

        subset_indices = ground_truth > 0
        gt_subset = ground_truth
        if subset == "ground_truth":
            subset_indices = ground_truth == curr_k
            gt_subset = ground_truth[subset_indices]
            logger.info(f"Subset is {len(gt_subset) / len(ground_truth) * 100:.2f}% of the data")

        for i, prob in enumerate(probs):
            if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
                prob_limit = prob[:, : labels_limit + 1]
                prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
                prob = prob_limit

            if subset == PREDICTIONS:
                subset_indices = np.argmax(prob, axis=1) == curr_k
                gt_subset = ground_truth[subset_indices]
                logger.info(
                    "Subset for model_name {} is {:.2f}% of the data".format(
                        model_names_list[i] if model_names_list and i < len(model_names_list) else i,
                        len(gt_subset) / len(ground_truth) * 100,
                    )
                )

            prob_subset = prob[subset_indices]

            max_prob = np.max(prob_subset, axis=1)
            predictions = np.argmax(prob_subset, axis=1)

            accuracies_alg = []
            dataset_kept_alg = []

            for threshold in thresholds:
                threshold = threshold if threshold < 1 else 0.999
                filtered_indices = max_prob >= threshold
                filtered_gt = gt_subset[filtered_indices]
                filtered_predictions = predictions[filtered_indices]
                accuracy = (filtered_gt == filtered_predictions).sum() / len(filtered_gt) if len(filtered_gt) > 0 else 0

                accuracies_alg.append(accuracy)
                dataset_kept_alg.append(len(filtered_gt) / len(ground_truth))

            accuracies.append(accuracies_alg)
            dataset_kept.append(dataset_kept_alg)

        output_feature_name_name = metadata[output_feature_name]["idx2str"][curr_k]

        filename = None
        if filename_template_path:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template_path.format(output_feature_name_name)

        visualization_utils.confidence_filtering_data_vs_acc_plot(
            accuracies,
            dataset_kept,
            model_names_list,
            decimal_digits=2,
            title="Confidence_Thresholding (Data vs Accuracy) " "for class {}".format(output_feature_name_name),
            filename=filename,
        )


@DeveloperAPI
def confidence_thresholding_2thresholds_2d(
    probabilities_per_model: List[np.array],
    ground_truths: Union[List[np.array], List[pd.Series]],
    metadata,
    threshold_output_feature_names: List[str],
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show confidence threshold data vs accuracy for two output feature names.

    The first plot shows several semi transparent lines. They summarize the
    3d surfaces displayed by confidence_thresholding_2thresholds_3d that have
    thresholds on the confidence of the predictions of the two
    `threshold_output_feature_names`  as x and y axes and either the data
    coverage percentage or
    the accuracy as z axis. Each line represents a slice of the data
    coverage  surface projected onto the accuracy surface.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[List[np.array], List[pd.Series]]) containing
        ground truth data
    :param metadata: (dict) feature metadata dictionary
    :param threshold_output_feature_names: (List[str]) List containing two output
        feature names for visualization.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)
    """
    try:
        validate_conf_thresholds_and_probabilities_2d_3d(probabilities_per_model, threshold_output_feature_names)
    except RuntimeError:
        return
    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    filename_template = "confidence_thresholding_2thresholds_2d_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)

    if not isinstance(ground_truths[0], np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[threshold_output_feature_names[0]]
        vfunc = np.vectorize(_encode_categorical_feature)
        gt_1 = vfunc(ground_truths[0], feature_metadata["str2idx"])
        feature_metadata = metadata[threshold_output_feature_names[1]]
        gt_2 = vfunc(ground_truths[1], feature_metadata["str2idx"])
    else:
        gt_1 = ground_truths[0]
        gt_2 = ground_truths[1]

    if labels_limit > 0:
        gt_1[gt_1 > labels_limit] = labels_limit
        gt_2[gt_2 > labels_limit] = labels_limit

    thresholds = [t / 100 for t in range(0, 101, 5)]
    fixed_step_coverage = thresholds
    name_t1 = f"{threshold_output_feature_names[0]} threshold"
    name_t2 = f"{threshold_output_feature_names[1]} threshold"

    accuracies = []
    dataset_kept = []
    interps = []
    table = [[name_t1, name_t2, "coverage", ACCURACY]]

    if labels_limit > 0 and probs[0].shape[1] > labels_limit + 1:
        prob_limit = probs[0][:, : labels_limit + 1]
        prob_limit[:, labels_limit] = probs[0][:, labels_limit:].sum(1)
        probs[0] = prob_limit

    if labels_limit > 0 and probs[1].shape[1] > labels_limit + 1:
        prob_limit = probs[1][:, : labels_limit + 1]
        prob_limit[:, labels_limit] = probs[1][:, labels_limit:].sum(1)
        probs[1] = prob_limit

    max_prob_1 = np.max(probs[0], axis=1)
    predictions_1 = np.argmax(probs[0], axis=1)

    max_prob_2 = np.max(probs[1], axis=1)
    predictions_2 = np.argmax(probs[1], axis=1)

    for threshold_1 in thresholds:
        threshold_1 = threshold_1 if threshold_1 < 1 else 0.999
        curr_accuracies = []
        curr_dataset_kept = []

        for threshold_2 in thresholds:
            threshold_2 = threshold_2 if threshold_2 < 1 else 0.999

            filtered_indices = np.logical_and(max_prob_1 >= threshold_1, max_prob_2 >= threshold_2)

            filtered_gt_1 = gt_1[filtered_indices]
            filtered_predictions_1 = predictions_1[filtered_indices]
            filtered_gt_2 = gt_2[filtered_indices]
            filtered_predictions_2 = predictions_2[filtered_indices]

            coverage = len(filtered_gt_1) / len(gt_1)
            accuracy = (
                np.logical_and(filtered_gt_1 == filtered_predictions_1, filtered_gt_2 == filtered_predictions_2)
            ).sum() / len(filtered_gt_1)

            curr_accuracies.append(accuracy)
            curr_dataset_kept.append(coverage)
            table.append([threshold_1, threshold_2, coverage, accuracy])

        accuracies.append(curr_accuracies)
        dataset_kept.append(curr_dataset_kept)
        interps.append(
            np.interp(
                fixed_step_coverage, list(reversed(curr_dataset_kept)), list(reversed(curr_accuracies)), left=1, right=0
            )
        )

    logger.info("CSV table")
    for row in table:
        logger.info(",".join([str(e) for e in row]))

    # ===========#
    # Multiline #
    # ===========#
    filename = None
    if filename_template_path:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template_path.format("multiline")
    visualization_utils.confidence_filtering_data_vs_acc_multiline_plot(
        accuracies, dataset_kept, model_names_list, title="Coverage vs Accuracy, two thresholds", filename=filename
    )

    # ==========#
    # Max line #
    # ==========#
    filename = None
    if filename_template_path:
        filename = filename_template_path.format("maxline")
    max_accuracies = np.amax(np.array(interps), 0)
    visualization_utils.confidence_filtering_data_vs_acc_plot(
        [max_accuracies],
        [thresholds],
        model_names_list,
        title="Coverage vs Accuracy, two thresholds",
        filename=filename,
    )

    # ==========================#
    # Max line with thresholds #
    # ==========================#
    acc_matrix = np.array(accuracies)
    cov_matrix = np.array(dataset_kept)
    t1_maxes = [1]
    t2_maxes = [1]
    for i in range(len(fixed_step_coverage) - 1):
        lower = fixed_step_coverage[i]
        upper = fixed_step_coverage[i + 1]
        indices = np.logical_and(cov_matrix >= lower, cov_matrix < upper)
        selected_acc = acc_matrix.copy()
        selected_acc[np.logical_not(indices)] = -1
        threshold_indices = np.unravel_index(np.argmax(selected_acc, axis=None), selected_acc.shape)
        t1_maxes.append(thresholds[threshold_indices[0]])
        t2_maxes.append(thresholds[threshold_indices[1]])
    model_name = model_names_list[0] if model_names_list is not None and len(model_names_list) > 0 else ""

    filename = None
    if filename_template_path:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template_path.format("maxline_with_thresholds")

    visualization_utils.confidence_filtering_data_vs_acc_plot(
        [max_accuracies, t1_maxes, t2_maxes],
        [fixed_step_coverage, fixed_step_coverage, fixed_step_coverage],
        model_names=[model_name + " accuracy", name_t1, name_t2],
        dotted=[False, True, True],
        y_label="",
        title="Coverage vs Accuracy & Threshold",
        filename=filename,
    )


@DeveloperAPI
def confidence_thresholding_2thresholds_3d(
    probabilities_per_model: List[np.array],
    ground_truths: Union[List[np.array], List[pd.Series]],
    metadata,
    threshold_output_feature_names: List[str],
    labels_limit: int,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show 3d confidence threshold data vs accuracy for two output feature names.

    The plot shows the 3d surfaces displayed by
    confidence_thresholding_2thresholds_3d that have thresholds on the
    confidence of the predictions of the two `threshold_output_feature_names`
    as x and y axes and either the data coverage percentage or the accuracy
    as z axis.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[List[np.array], List[pd.Series]]) containing
        ground truth data
    :param metadata: (dict) feature metadata dictionary
    :param threshold_output_feature_names: (List[str]) List containing two output
        feature names for visualization.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)
    """
    try:
        validate_conf_thresholds_and_probabilities_2d_3d(probabilities_per_model, threshold_output_feature_names)
    except RuntimeError:
        return
    probs = probabilities_per_model

    if not isinstance(ground_truths[0], np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[threshold_output_feature_names[0]]
        vfunc = np.vectorize(_encode_categorical_feature)
        gt_1 = vfunc(ground_truths[0], feature_metadata["str2idx"])
        feature_metadata = metadata[threshold_output_feature_names[1]]
        gt_2 = vfunc(ground_truths[1], feature_metadata["str2idx"])
    else:
        gt_1 = ground_truths[0]
        gt_2 = ground_truths[1]

    if labels_limit > 0:
        gt_1[gt_1 > labels_limit] = labels_limit
        gt_2[gt_2 > labels_limit] = labels_limit

    thresholds = [t / 100 for t in range(0, 101, 5)]

    accuracies = []
    dataset_kept = []

    if labels_limit > 0 and probs[0].shape[1] > labels_limit + 1:
        prob_limit = probs[0][:, : labels_limit + 1]
        prob_limit[:, labels_limit] = probs[0][:, labels_limit:].sum(1)
        probs[0] = prob_limit

    if labels_limit > 0 and probs[1].shape[1] > labels_limit + 1:
        prob_limit = probs[1][:, : labels_limit + 1]
        prob_limit[:, labels_limit] = probs[1][:, labels_limit:].sum(1)
        probs[1] = prob_limit

    max_prob_1 = np.max(probs[0], axis=1)
    predictions_1 = np.argmax(probs[0], axis=1)

    max_prob_2 = np.max(probs[1], axis=1)
    predictions_2 = np.argmax(probs[1], axis=1)

    for threshold_1 in thresholds:
        threshold_1 = threshold_1 if threshold_1 < 1 else 0.999
        curr_accuracies = []
        curr_dataset_kept = []

        for threshold_2 in thresholds:
            threshold_2 = threshold_2 if threshold_2 < 1 else 0.999

            filtered_indices = np.logical_and(max_prob_1 >= threshold_1, max_prob_2 >= threshold_2)

            filtered_gt_1 = gt_1[filtered_indices]
            filtered_predictions_1 = predictions_1[filtered_indices]
            filtered_gt_2 = gt_2[filtered_indices]
            filtered_predictions_2 = predictions_2[filtered_indices]

            accuracy = (
                np.logical_and(filtered_gt_1 == filtered_predictions_1, filtered_gt_2 == filtered_predictions_2)
            ).sum() / len(filtered_gt_1)

            curr_accuracies.append(accuracy)
            curr_dataset_kept.append(len(filtered_gt_1) / len(gt_1))

        accuracies.append(curr_accuracies)
        dataset_kept.append(curr_dataset_kept)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "confidence_thresholding_2thresholds_3d." + file_format)

    visualization_utils.confidence_filtering_3d_plot(
        np.array(thresholds),
        np.array(thresholds),
        np.array(accuracies),
        np.array(dataset_kept),
        threshold_output_feature_names,
        title="Confidence_Thresholding, two thresholds",
        filename=filename,
    )


@DeveloperAPI
def binary_threshold_vs_metric(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    metrics: List[str],
    positive_label: int = 1,
    model_names: List[str] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show confidence of the model against metric for the specified output_feature_name.

    For each metric specified in metrics (options are `f1`, `precision`, `recall`,
    `accuracy`), this visualization produces a line chart plotting a threshold
    on  the confidence of the model against the metric for the specified
    output_feature_name.  If output_feature_name is a category feature,
    positive_label, which is specified as the numeric encoded value, indicates
    the class to be considered positive class and all others will be
    considered negative. To figure out the
    association between classes and numeric encoded values check the
    ground_truth_metadata JSON file.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param metrics: (List[str]) metrics to display (`'f1'`, `'precision'`,
        `'recall'`, `'accuracy'`).
    :param positive_label: (int, default: `1`) numeric encoded value for the
        positive class.
    :param model_names: (List[str], default: `None`) list of the names of the
        models to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (`None`)
    """

    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth, positive_label = _convert_ground_truth(
            ground_truth, feature_metadata, ground_truth_apply_idx, positive_label
        )

    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    metrics_list = convert_to_list(metrics)
    filename_template = "binary_threshold_vs_metric_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)

    thresholds = [t / 100 for t in range(0, 101, 5)]

    supported_metrics = {"f1", "precision", "recall", "accuracy"}

    for metric in metrics_list:
        if metric not in supported_metrics:
            logger.error(f"Metric {metric} not supported")
            continue

        scores = []

        for i, prob in enumerate(probs):
            scores_alg = []

            if len(prob.shape) == 2:
                if prob.shape[1] > positive_label:
                    prob = prob[:, positive_label]
                else:
                    raise Exception(
                        "the specified positive label {} is not " "present in the probabilities".format(positive_label)
                    )

            for threshold in thresholds:
                threshold = threshold if threshold < 1 else 0.99

                predictions = prob >= threshold

                if metric == "f1":
                    metric_score = sklearn.metrics.f1_score(ground_truth, predictions)
                elif metric == "precision":
                    metric_score = sklearn.metrics.precision_score(ground_truth, predictions)
                elif metric == "recall":
                    metric_score = sklearn.metrics.recall_score(ground_truth, predictions)
                elif metric == ACCURACY:
                    metric_score = sklearn.metrics.accuracy_score(ground_truth, predictions)

                scores_alg.append(metric_score)

            scores.append(scores_alg)

        filename = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template_path.format(metric)

        visualization_utils.threshold_vs_metric_plot(
            thresholds, scores, model_names_list, title=f"Binary threshold vs {metric}", filename=filename
        )


@DeveloperAPI
def precision_recall_curves(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    positive_label: int = 1,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show the precision recall curves for output features in the specified models.

    This visualization produces a line chart plotting a precision recall curve for the
    specified output feature name. If output feature name is a category feature,
    `positive_label` indicates which is the class to be considered positive
    class and all the others will be considered negative. `positive_label` is
    the encoded numeric value for category classes. The numeric value can be
    determined by association between classes and integers captured in the
    training metadata JSON file.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param positive_label: (int, default: `1`) numeric encoded value for the
        positive class.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth, positive_label = _convert_ground_truth(
            ground_truth, feature_metadata, ground_truth_apply_idx, positive_label
        )

    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    precision_recalls = []

    for _, prob in enumerate(probs):
        if len(prob.shape) > 1:
            prob = prob[:, positive_label]
        precision, recall, _ = sklearn.metrics.precision_recall_curve(ground_truth, prob, pos_label=positive_label)
        precision_recalls.append({"precisions": precision, "recalls": recall})

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "precision_recall_curve." + file_format)

    visualization_utils.precision_recall_curves_plot(
        precision_recalls, model_names_list, title="Precision Recall Curves", filename=filename
    )


@DeveloperAPI
def precision_recall_curves_from_test_statistics(
    test_stats_per_model: List[dict],
    output_feature_name: str,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show the PR curves for the specified models output binary `output_feature_name`.

    This visualization uses `output_feature_name`, `test_stats_per_model` and
    `model_names` parameters. `output_feature_name` needs to be binary feature.
    This visualization produces a line chart plotting the PR curves for the
    specified `output_feature_name`.

    Args:

    :param test_stats_per_model: (List[dict]) dictionary containing evaluation
        performance statistics.
    :param output_feature_name: (str) name of the output feature to use
        for the visualization.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    Return

    :return: (None)
    """
    model_names_list = convert_to_list(model_names)
    filename_template = "precision_recall_curves_from_prediction_statistics." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    precision_recalls = []
    for curr_test_statistics in test_stats_per_model:
        precisions = curr_test_statistics[output_feature_name]["precision_recall_curve"]["precisions"]
        recalls = curr_test_statistics[output_feature_name]["precision_recall_curve"]["recalls"]
        precision_recalls.append({"precisions": precisions, "recalls": recalls})

    visualization_utils.precision_recall_curves_plot(
        precision_recalls, model_names_list, title="Precision Recall Curves", filename=filename_template_path
    )


@DeveloperAPI
def roc_curves(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    positive_label: int = 1,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show the roc curves for output features in the specified models.

    This visualization produces a line chart plotting the roc curves for the
    specified output feature name. If output feature name is a category feature,
    `positive_label` indicates which is the class to be considered positive
    class and all the others will be considered negative. `positive_label` is
    the encoded numeric value for category classes. The numeric value can be
    determined by association between classes and integers captured in the
    training metadata JSON file.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param positive_label: (int, default: `1`) numeric encoded value for the
        positive class.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth, positive_label = _convert_ground_truth(
            ground_truth, feature_metadata, ground_truth_apply_idx, positive_label
        )

    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    fpr_tprs = []

    for i, prob in enumerate(probs):
        if len(prob.shape) > 1:
            prob = prob[:, positive_label]
        fpr, tpr, _ = sklearn.metrics.roc_curve(ground_truth, prob, pos_label=positive_label)
        fpr_tprs.append((fpr, tpr))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = os.path.join(output_directory, "roc_curves." + file_format)

    visualization_utils.roc_curves(fpr_tprs, model_names_list, title="ROC curves", filename=filename)


@DeveloperAPI
def roc_curves_from_test_statistics(
    test_stats_per_model: List[dict],
    output_feature_name: str,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show the roc curves for the specified models output binary `output_feature_name`.

    This visualization uses `output_feature_name`, `test_stats_per_model` and
    `model_names` parameters. `output_feature_name` needs to be binary feature.
    This visualization produces a line chart plotting the roc curves for the
    specified `output_feature_name`.

    # Inputs

    :param test_stats_per_model: (List[dict]) dictionary containing evaluation
        performance statistics.
    :param output_feature_name: (str) name of the output feature to use
        for the visualization.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)
    """
    model_names_list = convert_to_list(model_names)
    filename_template = "roc_curves_from_prediction_statistics." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    fpr_tprs = []
    for curr_test_statistics in test_stats_per_model:
        fpr = curr_test_statistics[output_feature_name]["roc_curve"]["false_positive_rate"]
        tpr = curr_test_statistics[output_feature_name]["roc_curve"]["true_positive_rate"]
        fpr_tprs.append((fpr, tpr))

    visualization_utils.roc_curves(fpr_tprs, model_names_list, title="ROC curves", filename=filename_template_path)


@DeveloperAPI
def calibration_1_vs_all(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    top_n_classes: List[int],
    labels_limit: int,
    model_names: List[str] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models probability of predictions for the specified output_feature_name.

    For each class or each of the k most frequent classes if top_k is
    specified,  it produces two plots computed on the fly from the
    probabilities  of predictions for the specified output_feature_name.

    The first plot is a calibration curve that shows the calibration of the
    predictions considering the current class to be the true one and all
    others  to be a false one, drawing one line for each model (in the
    aligned  lists of probabilities and model_names).

    The second plot shows the distributions of the predictions considering
    the  current class to be the true one and all others to be a false one,
    drawing the distribution for each model (in the aligned lists of
    probabilities and model_names).

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param top_n_classes: (list) List containing the number of classes to plot.
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (List[str], default: `None`) list of the names of the
        models to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # String

    :return: (None)
    """
    feature_metadata = metadata[output_feature_name]
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    filename_template = "calibration_1_vs_all_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit
    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            probs[i] = prob_limit

    num_classes = len(metadata[output_feature_name]["str2idx"])

    brier_scores = []

    classes = min(num_classes, top_n_classes[0]) if top_n_classes[0] > 0 else num_classes
    class_names = [feature_metadata["idx2str"][i] for i in range(classes)]

    for class_idx in range(classes):
        fraction_positives_class = []
        mean_predicted_vals_class = []
        probs_class = []
        brier_scores_class = []
        for prob in probs:
            # ground_truth is a vector of integers, each integer is a class
            # index to have a [0,1] vector we have to check if the value equals
            # the input class index and convert the resulting boolean vector
            # into an integer vector probabilities is a n x c matrix, n is the
            # number of datapoints and c number of classes; its values are the
            # probabilities of the ith datapoint to be classified as belonging
            # to the jth class according to the learned model. For this reason
            # we need to take only the column of predictions that is about the
            # class we are interested in, the input class index

            gt_class = (ground_truth == class_idx).astype(int)
            prob_class = prob[:, class_idx]

            (curr_fraction_positives, curr_mean_predicted_vals) = calibration_curve(gt_class, prob_class, n_bins=21)

            if len(curr_fraction_positives) < 2:
                curr_fraction_positives = np.concatenate((np.array([0.0]), curr_fraction_positives))
            if len(curr_mean_predicted_vals) < 2:
                curr_mean_predicted_vals = np.concatenate((np.array([0.0]), curr_mean_predicted_vals))

            fraction_positives_class.append(curr_fraction_positives)
            mean_predicted_vals_class.append(curr_mean_predicted_vals)
            probs_class.append(prob[:, class_idx])
            brier_scores_class.append(brier_score_loss(gt_class, prob_class, pos_label=1))

        brier_scores.append(brier_scores_class)

        filename = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template_path.format(class_idx)

        visualization_utils.calibration_plot(
            fraction_positives_class,
            mean_predicted_vals_class,
            model_names_list,
            class_name=class_names[class_idx],
            filename=filename,
        )

        filename = None
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
            filename = filename_template_path.format("prediction_distribution_" + str(class_idx))

        visualization_utils.predictions_distribution_plot(probs_class, model_names_list, filename=filename)

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template_path.format("brier")

    visualization_utils.brier_plot(
        np.array(brier_scores),
        algorithm_names=model_names_list,
        class_names=class_names,
        title="Brier scores for each class",
        filename=filename,
    )


@DeveloperAPI
def calibration_multiclass(
    probabilities_per_model: List[np.array],
    ground_truth: Union[pd.Series, np.ndarray],
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models probability of predictions for each class of the specified output_feature_name.

    # Inputs

    :param probabilities_per_model: (List[numpy.array]) list of model
        probabilities.
    :param ground_truth: (Union[pd.Series, np.ndarray]) ground truth values
    :param metadata: (dict) feature metadata dictionary
    :param output_feature_name: (str) output feature name
    :param labels_limit: (int) upper limit on the numeric encoded label value.
        Encoded numeric label values in dataset that are higher than
        `labels_limit` are considered to be "rare" labels.
    :param model_names: (List[str], default: `None`) list of the names of the
        models to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.
    :param ground_truth_apply_idx: (bool, default: `True`) whether to use
        metadata['str2idx'] in np.vectorize

    # Return

    :return: (None)
    """
    if not isinstance(ground_truth, np.ndarray):
        # not np array, assume we need to translate raw value to encoded value
        feature_metadata = metadata[output_feature_name]
        ground_truth = _vectorize_ground_truth(ground_truth, feature_metadata["str2idx"], ground_truth_apply_idx)

    probs = probabilities_per_model
    model_names_list = convert_to_list(model_names)
    filename_template = "calibration_multiclass{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    if labels_limit > 0:
        ground_truth[ground_truth > labels_limit] = labels_limit

    prob_classes = 0
    for i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            probs[i] = prob_limit
        if probs[i].shape[1] > prob_classes:
            prob_classes = probs[i].shape[1]

    gt_one_hot_dim_2 = max(prob_classes, max(ground_truth) + 1)
    gt_one_hot = np.zeros((len(ground_truth), gt_one_hot_dim_2))
    gt_one_hot[np.arange(len(ground_truth)), ground_truth] = 1
    gt_one_hot_flat = gt_one_hot.flatten()

    fraction_positives = []
    mean_predicted_vals = []
    brier_scores = []
    for prob in probs:
        # flatten probabilities to be compared to flatten ground truth
        prob_flat = prob.flatten()
        curr_fraction_positives, curr_mean_predicted_vals = calibration_curve(gt_one_hot_flat, prob_flat, n_bins=21)
        fraction_positives.append(curr_fraction_positives)
        mean_predicted_vals.append(curr_mean_predicted_vals)
        brier_scores.append(brier_score_loss(gt_one_hot_flat, prob_flat, pos_label=1))

    filename = None
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        filename = filename_template_path.format("")

    visualization_utils.calibration_plot(fraction_positives, mean_predicted_vals, model_names_list, filename=filename)

    filename = None
    if output_directory:
        filename = filename_template_path.format("_brier")

    visualization_utils.compare_classifiers_plot(
        [brier_scores], ["brier"], model_names, adaptive=True, decimals=8, filename=filename
    )

    for i, brier_score in enumerate(brier_scores):
        if i < len(model_names):
            tokenizer_name = f"{model_names[i]}: "
            tokenizer_name += "{}"
        else:
            tokenizer_name = "{}"
        logger.info(tokenizer_name.format(brier_score))


@DeveloperAPI
def confusion_matrix(
    test_stats_per_model: List[dict],
    metadata: dict,
    output_feature_name: Union[str, None],
    top_n_classes: List[int],
    normalize: bool,
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show confusion matrix in the models predictions for each `output_feature_name`.

    For each model (in the aligned lists of test_statistics and model_names)
    it  produces a heatmap of the confusion matrix in the predictions for
    each  output_feature_name that has a confusion matrix in test_statistics.
    The value of `top_n_classes` limits the heatmap to the n most frequent
    classes.

    # Inputs

    :param test_stats_per_model: (List[dict]) dictionary containing evaluation
      performance statistics.
    :param metadata: (dict) intermediate preprocess structure created during
        training containing the mappings of the input dataset.
    :param output_feature_name: (Union[str, `None`]) name of the output feature
        to use for the visualization.  If `None`, use all output features.
    :param top_n_classes: (List[int]) number of top classes or list
        containing the number of top classes to plot.
    :param normalize: (bool) flag to normalize rows in confusion matrix.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)
    """
    test_stats_per_model_list = test_stats_per_model
    model_names_list = convert_to_list(model_names)
    filename_template = "confusion_matrix_{}_{}_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    output_feature_names = _validate_output_feature_name_from_test_stats(output_feature_name, test_stats_per_model_list)

    confusion_matrix_found = False
    for i, test_statistics in enumerate(test_stats_per_model_list):
        for output_feature_name in output_feature_names:
            if "confusion_matrix" in test_statistics[output_feature_name]:
                confusion_matrix_found = True
                _confusion_matrix = np.array(test_statistics[output_feature_name]["confusion_matrix"])
                model_name_name = (
                    model_names_list[i] if (model_names_list is not None and i < len(model_names_list)) else ""
                )
                if (
                    metadata is not None
                    and output_feature_name in metadata
                    and ("idx2str" in metadata[output_feature_name] or "bool2str" in metadata[output_feature_name])
                ):
                    if "bool2str" in metadata[output_feature_name]:  # Handles the binary output case
                        labels = metadata[output_feature_name]["bool2str"]
                    else:
                        labels = metadata[output_feature_name]["idx2str"]
                else:
                    labels = list(range(len(_confusion_matrix)))

                for k in top_n_classes:
                    k = min(k, _confusion_matrix.shape[0]) if k > 0 else _confusion_matrix.shape[0]
                    cm = _confusion_matrix[:k, :k]
                    if normalize:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            cm_norm = np.true_divide(cm, cm.sum(1)[:, np.newaxis])
                            cm_norm[cm_norm == np.inf] = 0
                            cm_norm = np.nan_to_num(cm_norm)
                        cm = cm_norm

                    filename = None
                    if output_directory:
                        os.makedirs(output_directory, exist_ok=True)
                        filename = filename_template_path.format(model_name_name, output_feature_name, "top" + str(k))

                    visualization_utils.confusion_matrix_plot(
                        cm, labels[:k], output_feature_name=output_feature_name, filename=filename
                    )

                    entropies = []
                    for row in cm:
                        if np.count_nonzero(row) > 0:
                            entropies.append(entropy(row))
                        else:
                            entropies.append(0)
                    class_entropy = np.array(entropies)
                    class_desc_entropy = np.argsort(class_entropy)[::-1]
                    desc_entropy = class_entropy[class_desc_entropy]

                    filename = None
                    if output_directory:
                        filename = filename_template_path.format(
                            "entropy_" + model_name_name, output_feature_name, "top" + str(k)
                        )

                    visualization_utils.bar_plot(
                        class_desc_entropy,
                        desc_entropy,
                        labels=[labels[i] for i in class_desc_entropy],
                        title="Classes ranked by entropy of " "Confusion Matrix row",
                        filename=filename,
                    )
    if not confusion_matrix_found:
        logger.error("Cannot find confusion_matrix in evaluation data")
        raise FileNotFoundError("Cannot find confusion_matrix in evaluation " "data")


@DeveloperAPI
def frequency_vs_f1(
    test_stats_per_model: List[dict],
    metadata: dict,
    output_feature_name: Union[str, None],
    top_n_classes: List[int],
    model_names: Union[str, List[str]] = None,
    output_directory: str = None,
    file_format: str = "pdf",
    **kwargs,
):
    """Show prediction statistics for the specified `output_feature_name` for each model.

    For each model (in the aligned lists of `test_stats_per_model` and
    `model_names`), produces two plots statistics of predictions for the
    specified `output_feature_name`.

    The first plot is a line plot with one x axis representing the different
    classes and two vertical axes colored in orange and blue respectively.
    The orange one is the frequency of the class and an orange line is plotted
    to show the trend. The blue one is the F1 score for that class and a blue
    line is plotted to show the trend. The classes on the x axis are sorted by
    f1 score.

    The second plot has the same structure of the first one,
    but the axes are flipped and the classes on the x axis are sorted by
    frequency.

    # Inputs

    :param test_stats_per_model: (List[dict]) dictionary containing evaluation
        performance statistics.
    :param metadata: (dict) intermediate preprocess structure created during
        training containing the mappings of the input dataset.
    :param output_feature_name: (Union[str, `None`]) name of the output feature
        to use for the visualization.  If `None`, use all output features.
    :param top_n_classes: (List[int]) number of top classes or list
        containing the number of top classes to plot.
    :param model_names: (Union[str, List[str]], default: `None`) model name or
        list of the model names to use as labels.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)
    """
    test_stats_per_model_list = test_stats_per_model
    model_names_list = convert_to_list(model_names)
    filename_template = "frequency_vs_f1_{}_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)
    output_feature_names = _validate_output_feature_name_from_test_stats(output_feature_name, test_stats_per_model_list)
    k = top_n_classes[0]

    for i, test_stats in enumerate(test_stats_per_model_list):
        for of_name in output_feature_names:
            # Figure out model name
            model_name = model_names_list[i] if model_names_list is not None and i < len(model_names_list) else ""

            # setup directory and filename
            filename = None
            if output_directory:
                os.makedirs(output_directory, exist_ok=True)
                filename = filename_template_path.format(model_name, of_name)

            # setup local variables
            per_class_stats = test_stats[of_name]["per_class_stats"]
            class_names = metadata[of_name]["idx2str"]

            # get np arrays of frequencies, f1s and labels
            idx2freq = {metadata[of_name]["str2idx"][key]: val for key, val in metadata[of_name]["str2freq"].items()}
            freq_np = np.array([idx2freq[class_id] for class_id in sorted(idx2freq)], dtype=np.int32)

            if k > 0:
                class_names = class_names[:k]
                freq_np = freq_np[:k]

            f1_scores = []
            labels = []

            for class_name in class_names:
                class_stats = per_class_stats[class_name]
                f1_scores.append(class_stats["f1_score"])
                labels.append(class_name)

            f1_np = np.nan_to_num(np.array(f1_scores, dtype=np.float32))
            labels_np = np.array(labels)

            # sort by f1
            f1_sort_idcs = f1_np.argsort()[::-1]
            len_f1_sort_idcs = len(f1_sort_idcs)

            freq_sorted_by_f1 = freq_np[f1_sort_idcs]
            freq_sorted_by_f1 = freq_sorted_by_f1[:len_f1_sort_idcs]
            f1_sorted_by_f1 = f1_np[f1_sort_idcs]
            f1_sorted_by_f1 = f1_sorted_by_f1[:len_f1_sort_idcs]
            labels_sorted_by_f1 = labels_np[f1_sort_idcs]
            labels_sorted_by_f1 = labels_sorted_by_f1[:len_f1_sort_idcs]

            # create viz sorted by f1
            visualization_utils.double_axis_line_plot(
                f1_sorted_by_f1,
                freq_sorted_by_f1,
                "F1 score",
                "frequency",
                labels=labels_sorted_by_f1,
                title=f"{model_name} F1 Score vs Frequency {of_name}",
                filename=filename,
            )

            # sort by freq
            freq_sort_idcs = freq_np.argsort()[::-1]
            len_freq_sort_idcs = len(freq_sort_idcs)

            freq_sorted_by_freq = freq_np[freq_sort_idcs]
            freq_sorted_by_freq = freq_sorted_by_freq[:len_freq_sort_idcs]
            f1_sorted_by_freq = f1_np[freq_sort_idcs]
            f1_sorted_by_freq = f1_sorted_by_freq[:len_freq_sort_idcs]
            labels_sorted_by_freq = labels_np[freq_sort_idcs]
            labels_sorted_by_freq = labels_sorted_by_freq[:len_freq_sort_idcs]

            # create viz sorted by freq
            visualization_utils.double_axis_line_plot(
                freq_sorted_by_freq,
                f1_sorted_by_freq,
                "frequency",
                "F1 score",
                labels=labels_sorted_by_freq,
                title=f"{model_name} F1 Score vs Frequency {of_name}",
                filename=filename,
            )


@DeveloperAPI
def hyperopt_report_cli(hyperopt_stats_path, output_directory=None, file_format="pdf", **kwargs) -> None:
    """Produces a report about hyperparameter optimization creating one graph per hyperparameter to show the
    distribution of results and one additional graph of pairwise hyperparameters interactions.

    :param hyperopt_stats_path: path to the hyperopt results JSON file
    :param output_directory: path where to save the output plots
    :param file_format: format of the output plot, pdf or png
    :return:
    """

    hyperopt_report(hyperopt_stats_path, output_directory=output_directory, file_format=file_format)


@DeveloperAPI
def hyperopt_report(hyperopt_stats_path: str, output_directory: str = None, file_format: str = "pdf", **kwargs) -> None:
    """Produces a report about hyperparameter optimization creating one graph per hyperparameter to show the
    distribution of results and one additional graph of pairwise hyperparameters interactions.

    # Inputs

    :param hyperopt_stats_path: (str) path to the hyperopt results JSON file.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window.
    :param file_format: (str, default: `'pdf'`) file format of output plots -
        `'pdf'` or `'png'`.

    # Return

    :return: (None)
    """
    filename_template = "hyperopt_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)

    hyperopt_stats = load_json(hyperopt_stats_path)

    visualization_utils.hyperopt_report(
        hyperopt_stats["hyperopt_config"]["parameters"],
        hyperopt_results_to_dataframe(
            hyperopt_stats["hyperopt_results"],
            hyperopt_stats["hyperopt_config"]["parameters"],
            hyperopt_stats["hyperopt_config"]["metric"],
        ),
        metric=hyperopt_stats["hyperopt_config"]["metric"],
        filename_template=filename_template_path,
    )


@DeveloperAPI
def hyperopt_hiplot_cli(hyperopt_stats_path, output_directory=None, **kwargs):
    """Produces a parallel coordinate plot about hyperparameter optimization creating one HTML file and optionally
    a CSV file to be read by hiplot.

    :param hyperopt_stats_path: path to the hyperopt results JSON file
    :param output_directory: path where to save the output plots
    :return:
    """

    hyperopt_hiplot(hyperopt_stats_path, output_directory=output_directory)


@DeveloperAPI
def hyperopt_hiplot(hyperopt_stats_path, output_directory=None, **kwargs):
    """Produces a parallel coordinate plot about hyperparameter optimization creating one HTML file and optionally
    a CSV file to be read by hiplot.

    # Inputs

    :param hyperopt_stats_path: (str) path to the hyperopt results JSON file.
    :param output_directory: (str, default: `None`) directory where to save
        plots. If not specified, plots will be displayed in a window.

    # Return

    :return: (None)
    """
    filename = "hyperopt_hiplot.html"
    filename_path = generate_filename_template_path(output_directory, filename)

    hyperopt_stats = load_json(hyperopt_stats_path)
    hyperopt_df = hyperopt_results_to_dataframe(
        hyperopt_stats["hyperopt_results"],
        hyperopt_stats["hyperopt_config"]["parameters"],
        hyperopt_stats["hyperopt_config"]["metric"],
    )
    visualization_utils.hyperopt_hiplot(
        hyperopt_df,
        filename=filename_path,
    )


def _convert_space_to_dtype(space: str) -> str:
    if space in visualization_utils.RAY_TUNE_FLOAT_SPACES:
        return "float"
    elif space in visualization_utils.RAY_TUNE_INT_SPACES:
        return "int"
    else:
        return "object"


@DeveloperAPI
def hyperopt_results_to_dataframe(hyperopt_results, hyperopt_parameters, metric):
    df = pd.DataFrame([{metric: res["metric_score"], **res["parameters"]} for res in hyperopt_results])
    df = df.astype(
        {hp_name: _convert_space_to_dtype(hp_params[SPACE]) for hp_name, hp_params in hyperopt_parameters.items()}
    )
    return df


@DeveloperAPI
def get_visualizations_registry() -> Dict[str, Callable]:
    return {
        "compare_performance": compare_performance_cli,
        "compare_classifiers_performance_from_prob": compare_classifiers_performance_from_prob_cli,
        "compare_classifiers_performance_from_pred": compare_classifiers_performance_from_pred_cli,
        "compare_classifiers_performance_subset": compare_classifiers_performance_subset_cli,
        "compare_classifiers_performance_changing_k": compare_classifiers_performance_changing_k_cli,
        "compare_classifiers_multiclass_multimetric": compare_classifiers_multiclass_multimetric_cli,
        "compare_classifiers_predictions": compare_classifiers_predictions_cli,
        "compare_classifiers_predictions_distribution": compare_classifiers_predictions_distribution_cli,
        "confidence_thresholding": confidence_thresholding_cli,
        "confidence_thresholding_data_vs_acc": confidence_thresholding_data_vs_acc_cli,
        "confidence_thresholding_data_vs_acc_subset": confidence_thresholding_data_vs_acc_subset_cli,
        "confidence_thresholding_data_vs_acc_subset_per_class": confidence_thresholding_data_vs_acc_subset_per_class_cli,  # noqa: E501
        "confidence_thresholding_2thresholds_2d": confidence_thresholding_2thresholds_2d_cli,
        "confidence_thresholding_2thresholds_3d": confidence_thresholding_2thresholds_3d_cli,
        "binary_threshold_vs_metric": binary_threshold_vs_metric_cli,
        "roc_curves": roc_curves_cli,
        "roc_curves_from_test_statistics": roc_curves_from_test_statistics_cli,
        "precision_recall_curves": precision_recall_curves_cli,
        "precision_recall_curves_from_test_statistics": precision_recall_curves_from_test_statistics_cli,
        "calibration_1_vs_all": calibration_1_vs_all_cli,
        "calibration_multiclass": calibration_multiclass_cli,
        "confusion_matrix": confusion_matrix_cli,
        "frequency_vs_f1": frequency_vs_f1_cli,
        "learning_curves": learning_curves_cli,
        "hyperopt_report": hyperopt_report_cli,
        "hyperopt_hiplot": hyperopt_hiplot_cli,
    }


@PublicAPI
def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script analyzes results and shows some nice plots.",
        prog="ludwig visualize",
        usage="%(prog)s [options]",
    )

    parser.add_argument("-g", "--ground_truth", help="ground truth file")
    parser.add_argument("-gm", "--ground_truth_metadata", help="input metadata JSON file")
    parser.add_argument(
        "-sf",
        "--split_file",
        default=None,
        help="file containing split values used in conjunction with " "ground truth file.",
    )

    parser.add_argument(
        "-od",
        "--output_directory",
        help="directory where to save plots." "If not specified, plots will be displayed in a window",
    )
    parser.add_argument(
        "-ff", "--file_format", help="file format of output plots", default="pdf", choices=["pdf", "png"]
    )

    parser.add_argument(
        "-v",
        "--visualization",
        choices=sorted(list(get_visualizations_registry().keys())),
        help="type of visualization to generate",
        required=True,
    )

    parser.add_argument("-ofn", "--output_feature_name", default=[], help="name of the output feature to visualize")
    parser.add_argument(
        "-gts", "--ground_truth_split", default=2, help="ground truth split - 0:train, 1:validation, 2:test split"
    )
    parser.add_argument(
        "-tf",
        "--threshold_output_feature_names",
        default=[],
        nargs="+",
        help="names of output features for 2d threshold",
    )
    parser.add_argument("-pred", "--predictions", default=[], nargs="+", type=str, help="predictions files")
    parser.add_argument("-prob", "--probabilities", default=[], nargs="+", type=str, help="probabilities files")
    parser.add_argument("-trs", "--training_statistics", default=[], nargs="+", type=str, help="training stats files")
    parser.add_argument("-tes", "--test_statistics", default=[], nargs="+", type=str, help="test stats files")
    parser.add_argument("-hs", "--hyperopt_stats_path", default=None, type=str, help="hyperopt stats file")
    parser.add_argument(
        "-mn", "--model_names", default=[], nargs="+", type=str, help="names of the models to use as labels"
    )
    parser.add_argument("-tn", "--top_n_classes", default=[0], nargs="+", type=int, help="number of classes to plot")
    parser.add_argument("-k", "--top_k", default=3, type=int, help="number of elements in the ranklist to consider")
    parser.add_argument(
        "-ll",
        "--labels_limit",
        default=0,
        type=int,
        help="maximum numbers of labels. Encoded numeric label values in dataset that are higher than "
        'labels_limit are considered to be "rare" labels',
    )
    parser.add_argument(
        "-ss",
        "--subset",
        default="ground_truth",
        choices=["ground_truth", PREDICTIONS],
        help="type of subset filtering",
    )
    parser.add_argument(
        "-n", "--normalize", action="store_true", default=False, help="normalize rows in confusion matrix"
    )
    parser.add_argument(
        "-m", "--metrics", default=["f1"], nargs="+", type=str, help="metrics to display in threshold_vs_metric"
    )
    parser.add_argument(
        "-pl", "--positive_label", type=int, default=1, help="label of the positive class for the roc curve"
    )
    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("visualize", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.visualize")

    try:
        vis_func = get_visualizations_registry()[args.visualization]
    except KeyError:
        logger.info("Visualization argument not recognized")
        raise
    vis_func(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
