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
"""ROC, precision-recall, and calibration curve visualizations."""

import logging
import os

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from scipy.stats import entropy  # noqa: F401 - keep import available
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json
from ludwig.visualize._utils import (
    _convert_ground_truth,
    _extract_ground_truth_values,
    _get_cols_from_predictions,
    _PROBABILITIES_SUFFIX,
    _vectorize_ground_truth,
    convert_to_list,
    generate_filename_template_path,
    load_data_for_viz,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
def precision_recall_curves_cli(
    probabilities: "str | list[str]",
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
def precision_recall_curves(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    positive_label: int = 1,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show the precision recall curves for output features in the specified models.

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
def precision_recall_curves_from_test_statistics_cli(test_statistics: "str | list[str]", **kwargs: dict) -> None:
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
def precision_recall_curves_from_test_statistics(
    test_stats_per_model: "list[dict]",
    output_feature_name: str,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show the PR curves for the specified models output binary `output_feature_name`.

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
def roc_curves_cli(
    probabilities: "str | list[str]",
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
def roc_curves(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    positive_label: int = 1,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show the roc curves for output features in the specified models.

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

    for _i, prob in enumerate(probs):
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
def roc_curves_from_test_statistics_cli(test_statistics: "str | list[str]", **kwargs: dict) -> None:
    """Load model data from files to be shown by roc_curves_from_test_statistics_cli.

    # Inputs
    :param test_statistics: (Union[str, List[str]]) path to experiment test statistics file.
    :param kwargs: (dict) parameters for the requested visualizations.  # Return
    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    roc_curves_from_test_statistics(test_stats_per_model, **kwargs)


@DeveloperAPI
def roc_curves_from_test_statistics(
    test_stats_per_model: "list[dict]",
    output_feature_name: str,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    **kwargs,
) -> None:
    """Show the roc curves for the specified models output binary `output_feature_name`.

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
def calibration_1_vs_all_cli(
    probabilities: "str | list[str]",
    ground_truth: str,
    ground_truth_split: int,
    split_file: str,
    ground_truth_metadata: str,
    output_feature_name: str,
    output_directory: str,
    output_feature_proc_name: "str | None" = None,
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
def calibration_1_vs_all(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    top_n_classes: "list[int]",
    labels_limit: int,
    model_names: "list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    ground_truth_apply_idx: bool = True,
    **kwargs,
) -> None:
    """Show models probability of predictions for the specified output_feature_name.

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
            gt_class = (ground_truth == class_idx).astype(int)
            prob_class = prob[:, class_idx]

            curr_fraction_positives, curr_mean_predicted_vals = calibration_curve(gt_class, prob_class, n_bins=21)

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
def calibration_multiclass_cli(
    probabilities: "str | list[str]",
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
def calibration_multiclass(
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
        [brier_scores], ["brier"], model_names_list, adaptive=True, decimals=8, filename=filename
    )

    for i, brier_score in enumerate(brier_scores):
        if i < len(model_names_list):
            tokenizer_name = f"{model_names_list[i]}: "
            tokenizer_name += "{}"
        else:
            tokenizer_name = "{}"
        logger.info(tokenizer_name.format(brier_score))
