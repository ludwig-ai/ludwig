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
"""Confidence thresholding and binary threshold visualizations."""

import logging
import os

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ACCURACY, PREDICTIONS
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json
from ludwig.visualize._utils import (
    _convert_ground_truth,
    _encode_categorical_feature,
    _extract_ground_truth_values,
    _get_cols_from_predictions,
    _PROBABILITIES_SUFFIX,
    _vectorize_ground_truth,
    convert_to_list,
    generate_filename_template_path,
    validate_conf_thresholds_and_probabilities_2d_3d,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
def confidence_thresholding_cli(
    probabilities: "str | list[str]",
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
def confidence_thresholding(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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

    for _i, prob in enumerate(probs):
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
def confidence_thresholding_data_vs_acc_cli(
    probabilities: "str | list[str]",
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
def confidence_thresholding_data_vs_acc(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    labels_limit: int,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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

    for _i, prob in enumerate(probs):
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
def confidence_thresholding_data_vs_acc_subset_cli(
    probabilities: "str | list[str]",
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
def confidence_thresholding_data_vs_acc_subset(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    top_n_classes: "list[int]",
    labels_limit: int,
    subset: str,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models comparison of confidence threshold data vs accuracy on a subset of data.

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
                f"Subset for model_name {model_names[i] if model_names and i < len(model_names) else i} is {len(gt_subset) / len(ground_truth) * 100:.2f}% of the data"
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
def confidence_thresholding_data_vs_acc_subset_per_class_cli(
    probabilities: "str | list[str]",
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
def confidence_thresholding_data_vs_acc_subset_per_class(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    top_n_classes: "int | list[int]",
    labels_limit: int,
    subset: str,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models comparison of confidence threshold data vs accuracy on a subset of data per class in top n classes.

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
                    f"Subset for model_name {model_names_list[i] if model_names_list and i < len(model_names_list) else i} is {len(gt_subset) / len(ground_truth) * 100:.2f}% of the data"
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
            title=f"Confidence_Thresholding (Data vs Accuracy) for class {output_feature_name_name}",
            filename=filename,
        )


@DeveloperAPI
def confidence_thresholding_2thresholds_2d_cli(
    probabilities: "str | list[str]",
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    threshold_output_feature_names: "list[str]",
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
def confidence_thresholding_2thresholds_2d(
    probabilities_per_model: "list[np.array]",
    ground_truths: "list[np.array] | list[pd.Series]",
    metadata,
    threshold_output_feature_names: "list[str]",
    labels_limit: int,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show confidence threshold data vs accuracy for two output feature names.

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
def confidence_thresholding_2thresholds_3d_cli(
    probabilities: "str | list[str]",
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    threshold_output_feature_names: "list[str]",
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
def confidence_thresholding_2thresholds_3d(
    probabilities_per_model: "list[np.array]",
    ground_truths: "list[np.array] | list[pd.Series]",
    metadata,
    threshold_output_feature_names: "list[str]",
    labels_limit: int,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show 3d confidence threshold data vs accuracy for two output feature names.

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
def binary_threshold_vs_metric_cli(
    probabilities: "str | list[str]",
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
def binary_threshold_vs_metric(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    metrics: "list[str]",
    positive_label: int = 1,
    model_names: "list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show confidence of the model against metric for the specified output_feature_name.

    For each metric specified in metrics (options are `f1`, `precision`, `recall`,
    `accuracy`), this visualization produces a line chart plotting a threshold
    on  the confidence of the model against the metric for the specified
    output_feature_name.

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

        for _i, prob in enumerate(probs):
            scores_alg = []

            if len(prob.shape) == 2:
                if prob.shape[1] > positive_label:
                    prob = prob[:, positive_label]
                else:
                    raise Exception(
                        f"the specified positive label {positive_label} is not present in the probabilities"
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
