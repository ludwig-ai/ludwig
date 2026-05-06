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
"""Learning-curve visualizations."""

import logging

from ludwig.api_annotations import DeveloperAPI
from ludwig.callbacks import Callback
from ludwig.constants import ACCURACY, EDIT_DISTANCE, HITS_AT_K, LOSS
from ludwig.utils import visualization_utils
from ludwig.visualize._utils import (
    _validate_output_feature_name_from_train_stats,
    convert_to_list,
    generate_filename_template_path,
    load_training_stats_for_viz,
)

logger = logging.getLogger(__name__)


@DeveloperAPI
def learning_curves_cli(training_statistics: "str | list[str]", **kwargs: dict) -> None:
    """Load model data from files to be shown by learning_curves.

    # Inputs

    :param training_statistics: (Union[str, List[str]]) path to experiment training statistics file
    :param kwargs: (dict) parameters for the requested visualizations.  # Return
    :return None:
    """
    train_stats_per_model = load_training_stats_for_viz("load_json", training_statistics)
    learning_curves(train_stats_per_model, **kwargs)


@DeveloperAPI
def learning_curves(
    train_stats_per_model: "list[dict]",
    output_feature_name: "str | None" = None,
    model_names: "str | list[str] | None" = None,
    output_directory: "str | None" = None,
    file_format: str = "pdf",
    callbacks: "list[Callback] | None" = None,
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
