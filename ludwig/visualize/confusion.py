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
"""Confusion matrix visualization."""

import logging
import os

import numpy as np
from scipy.stats import entropy

from ludwig.api_annotations import DeveloperAPI
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json
from ludwig.visualize._utils import (
    _validate_output_feature_name_from_test_stats,
    convert_to_list,
    generate_filename_template_path,
    load_data_for_viz,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
def confusion_matrix_cli(test_statistics: "str | list[str]", ground_truth_metadata: str, **kwargs: dict) -> None:
    """Load model data from files to be shown by confusion_matrix.

    # Inputs

    :param test_statistics: (Union[str, List[str]]) path to experiment test statistics file.
    :param ground_truth_metadata: (str) path to ground truth metadata file.
    :param kwargs: (dict) parameters for the requested visualizations.  # Return
    :return None:
    """
    test_stats_per_model = load_data_for_viz("load_json", test_statistics)
    metadata = load_json(ground_truth_metadata)
    confusion_matrix(test_stats_per_model, metadata, **kwargs)


@DeveloperAPI
def confusion_matrix(
    test_stats_per_model: "list[dict]",
    metadata: dict,
    output_feature_name: "str | None",
    top_n_classes: "list[int]",
    normalize: bool,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
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
                        title="Classes ranked by entropy of Confusion Matrix row",
                        filename=filename,
                    )
    if not confusion_matrix_found:
        logger.error("Cannot find confusion_matrix in evaluation data")
        raise FileNotFoundError("Cannot find confusion_matrix in evaluation data")
