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
"""Hyperparameter optimization visualizations."""

import logging

import pandas as pd

from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import SPACE
from ludwig.utils import visualization_utils
from ludwig.utils.data_utils import load_json
from ludwig.visualize._utils import generate_filename_template_path

logger = logging.getLogger(__name__)


@DeveloperAPI
def hyperopt_report_cli(hyperopt_stats_path, output_directory=None, file_format="pdf", **kwargs) -> None:
    """Produces a report about hyperparameter optimization creating one graph per hyperparameter to show the
    distribution of results and one additional graph of pairwise hyperparameters interactions.

    Args:
        hyperopt_stats_path: Path to the hyperopt results JSON file.
        output_directory: Path where to save the output plots.
        file_format: Format of the output plot, pdf or png.
    """

    hyperopt_report(hyperopt_stats_path, output_directory=output_directory, file_format=file_format)


@DeveloperAPI
def hyperopt_report(
    hyperopt_stats_path: str, output_directory: "str | None" = None, file_format: str = "pdf", **kwargs
) -> None:
    """Produces a report about hyperparameter optimization creating one graph per hyperparameter to show the
    distribution of results and one additional graph of pairwise hyperparameters interactions.

    Args:
        hyperopt_stats_path: Path to the hyperopt results JSON file.
        output_directory: Directory where to save plots. If not specified, plots will be displayed in a window.
        file_format: File format of output plots — 'pdf' or 'png'.
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

    Args:
        hyperopt_stats_path: Path to the hyperopt results JSON file.
        output_directory: Path where to save the output plots.
    """

    hyperopt_hiplot(hyperopt_stats_path, output_directory=output_directory)


@DeveloperAPI
def hyperopt_hiplot(hyperopt_stats_path, output_directory=None, **kwargs):
    """Produces a parallel coordinate plot about hyperparameter optimization creating one HTML file and optionally
    a CSV file to be read by hiplot.

    Args:
        hyperopt_stats_path: Path to the hyperopt results JSON file.
        output_directory: Directory where to save plots. If not specified, plots will be displayed in a window.
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
