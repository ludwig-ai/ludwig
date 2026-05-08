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
"""Performance comparison and frequency visualizations."""

import logging
import os

import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import ACCURACY, HITS_AT_K, LOSS, PREDICTIONS
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json
from ludwig.visualize._utils import (
    _extract_ground_truth_values,
    _get_cols_from_predictions,
    _PREDICTIONS_SUFFIX,
    _PROBABILITIES_SUFFIX,
    _validate_output_feature_name_from_test_stats,
    _vectorize_ground_truth,
    convert_to_list,
    generate_filename_template_path,
    load_data_for_viz,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
def compare_performance_cli(test_statistics: "str | list[str]", **kwargs: dict) -> None:
    """Load model data from files to be shown by compare_performance.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test statistics file.
    :param kwargs: (dict) parameters for the requested visualizations.  # Return
    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    compare_performance(test_stats_per_model, **kwargs)


@DeveloperAPI
def compare_performance(
    test_stats_per_model: "list[dict]",
    output_feature_name: "str | None" = None,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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
    ignore_names = {
        "overall_stats",
        "confusion_matrix",
        "per_class_stats",
        "predictions",
        "probabilities",
        "roc_curve",
        "precision_recall_curve",
        LOSS,
    }

    filename_template = "compare_performance_{}." + file_format
    filename_template_path = generate_filename_template_path(output_directory, filename_template)

    test_stats_per_model_list = convert_to_list(test_stats_per_model)
    model_names_list = convert_to_list(model_names)
    output_feature_names = _validate_output_feature_name_from_test_stats(output_feature_name, test_stats_per_model_list)

    for output_feature_name in output_feature_names:
        metric_names_sets = [set(tspr[output_feature_name].keys()) for tspr in test_stats_per_model_list]
        metric_names = metric_names_sets[0]
        for metric_names_set in metric_names_sets:
            metric_names = metric_names.intersection(metric_names_set)
        metric_names = metric_names - ignore_names
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
def compare_classifiers_performance_from_prob_cli(
    probabilities: "str | list[str]",
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
def compare_classifiers_performance_from_prob(
    probabilities_per_model: "list[np.ndarray]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    labels_limit: int = 0,
    top_n_classes: "list[int] | int" = 3,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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

    for _i, prob in enumerate(probs):
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
            hits_at_k += np.isin(ground_truth[j], topk[j])
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
def compare_classifiers_performance_from_pred_cli(
    predictions: "list[str]",
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
def compare_classifiers_performance_from_pred(
    predictions_per_model: "list[np.ndarray]",
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

    for _i, pred in enumerate(preds):
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
def compare_classifiers_performance_subset_cli(
    probabilities: "str | list[str]",
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
def compare_classifiers_performance_subset(
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
                f"Subset for model_name {model_names[i] if model_names and i < len(model_names) else i} is {len(gt_subset) / len(ground_truth) * 100:.2f}% of the data"
            )
            model_names[i] = (
                f"{model_names[i] if model_names and i < len(model_names) else i} ({len(gt_subset) / len(ground_truth) * 100:.2f}%)"
            )

        prob_subset = prob[subset_indices]

        prob_subset = np.argsort(prob_subset, axis=1)
        top1_subset = prob_subset[:, -1]
        top3_subset = prob_subset[:, -3:]

        accuracies.append(np.sum(gt_subset == top1_subset) / len(gt_subset))

        hits_at_k = 0
        for j in range(len(gt_subset)):
            hits_at_k += np.isin(gt_subset[j], top3_subset[i, :])
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
def compare_classifiers_performance_changing_k_cli(
    probabilities: "str | list[str]",
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
def compare_classifiers_performance_changing_k(
    probabilities_per_model: "list[np.array]",
    ground_truth: "pd.Series | np.ndarray",
    metadata: dict,
    output_feature_name: str,
    top_k: int,
    labels_limit: int,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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
    for _i, prob in enumerate(probs):
        if labels_limit > 0 and prob.shape[1] > labels_limit + 1:
            prob_limit = prob[:, : labels_limit + 1]
            prob_limit[:, labels_limit] = prob[:, labels_limit:].sum(1)
            prob = prob_limit

        prob = np.argsort(prob, axis=1)

        hits_at_k = [0.0] * k
        for g in range(len(ground_truth)):
            for j in range(k):
                hits_at_k[j] += np.isin(ground_truth[g], prob[g, -j - 1 :])
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
def compare_classifiers_multiclass_multimetric_cli(
    test_statistics: "str | list[str]", ground_truth_metadata: str, **kwargs: dict
) -> None:
    """Load model data from files to be shown by compare_classifiers_multiclass.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test statistics file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param kwargs: (dict) parameters for the requested visualizations.  # Return
    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    metadata = load_json(ground_truth_metadata)
    compare_classifiers_multiclass_multimetric(test_stats_per_model, metadata=metadata, **kwargs)


@DeveloperAPI
def compare_classifiers_multiclass_multimetric(
    test_stats_per_model: "list[dict]",
    metadata: dict,
    output_feature_name: str,
    top_n_classes: "list[int]",
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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
                ((metadata[output_feature_name]["str2idx"][key], key) for key in per_class_stats),
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
                    title=f"{model_name_name} Multiclass Precision / Recall / F1 Score top {k} {output_feature_name}",
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
                    title=f"{model_name_name} Multiclass Precision / Recall / "
                    f"F1 Score best {k} classes {output_feature_name}",
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
def compare_classifiers_predictions_cli(
    predictions: "list[str]",
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
def compare_classifiers_predictions(
    predictions_per_model: "list[list]",
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
        f"  {name_c1} right / {name_c2} wrong: {c1_right_c2_wrong} {100 * c1_right_c2_wrong / all:.2f}% {100 * c1_right_c2_wrong / one_right if one_right > 0 else 0:.2f}%"
    )
    logger.info(
        f"  {name_c1} wrong / {name_c2} right: {c1_wrong_c2_right} {100 * c1_wrong_c2_right / all:.2f}% {100 * c1_wrong_c2_right / one_right if one_right > 0 else 0:.2f}%"
    )
    logger.info(f"Both wrong: {both_wrong} {100 * both_wrong / all:.2f}%")
    logger.info(
        f"  same prediction: {both_wrong_same} {100 * both_wrong_same / all:.2f}% {100 * both_wrong_same / both_wrong if both_wrong > 0 else 0:.2f}%"
    )
    logger.info(
        f"  different prediction: {both_wrong_different} {100 * both_wrong_different / all:.2f}% {100 * both_wrong_different / both_wrong if both_wrong > 0 else 0:.2f}%"
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
def compare_classifiers_predictions_distribution_cli(
    predictions: "list[str]",
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
def compare_classifiers_predictions_distribution(
    predictions_per_model: "list[list]",
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
    """Show comparison of models predictions distribution for 10 output_feature_name classes.

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
def frequency_vs_f1_cli(test_statistics: "str | list[str]", ground_truth_metadata: str, **kwargs: dict) -> None:
    """Load model data from files to be shown by frequency_vs_f1.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test statistics file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param kwargs: (dict) parameters for the requested visualizations.  # Return
    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    metadata = load_json(ground_truth_metadata)
    frequency_vs_f1(test_stats_per_model, metadata, **kwargs)


@DeveloperAPI
def frequency_vs_f1(
    test_stats_per_model: "list[dict]",
    metadata: dict,
    output_feature_name: "str | None",
    top_n_classes: "list[int]",
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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
