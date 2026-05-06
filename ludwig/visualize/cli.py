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
"""CLI entry point and visualizations registry."""

import argparse
import logging
import sys
from collections.abc import Callable

from ludwig.api_annotations import PublicAPI
from ludwig.constants import PREDICTIONS
from ludwig.contrib import add_contrib_callback_args
from ludwig.utils.print_utils import get_logging_level_registry
from ludwig.visualize.confusion import confusion_matrix_cli
from ludwig.visualize.curves import (
    calibration_1_vs_all_cli,
    calibration_multiclass_cli,
    precision_recall_curves_cli,
    precision_recall_curves_from_test_statistics_cli,
    roc_curves_cli,
    roc_curves_from_test_statistics_cli,
)
from ludwig.visualize.hyperopt import hyperopt_hiplot_cli, hyperopt_report_cli
from ludwig.visualize.performance import (
    compare_classifiers_multiclass_multimetric_cli,
    compare_classifiers_performance_changing_k_cli,
    compare_classifiers_performance_from_pred_cli,
    compare_classifiers_performance_from_prob_cli,
    compare_classifiers_performance_subset_cli,
    compare_classifiers_predictions_cli,
    compare_classifiers_predictions_distribution_cli,
    compare_performance_cli,
    frequency_vs_f1_cli,
)
from ludwig.visualize.threshold import (
    binary_threshold_vs_metric_cli,
    confidence_thresholding_2thresholds_2d_cli,
    confidence_thresholding_2thresholds_3d_cli,
    confidence_thresholding_cli,
    confidence_thresholding_data_vs_acc_cli,
    confidence_thresholding_data_vs_acc_subset_cli,
    confidence_thresholding_data_vs_acc_subset_per_class_cli,
)
from ludwig.visualize.training import learning_curves_cli

logger = logging.getLogger(__name__)


@PublicAPI
def get_visualizations_registry() -> "dict[str, Callable]":
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
        "confidence_thresholding_data_vs_acc_subset_per_class": confidence_thresholding_data_vs_acc_subset_per_class_cli,
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
        help="file containing split values used in conjunction with ground truth file.",
    )

    parser.add_argument(
        "-od",
        "--output_directory",
        help="directory where to save plots.If not specified, plots will be displayed in a window",
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
