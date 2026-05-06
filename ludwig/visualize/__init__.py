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
"""ludwig.visualize package — backward-compatible re-exports.

All public and private symbols that were importable from ``ludwig.visualize``
when it was a single module remain importable from this package.
"""

# ---------------------------------------------------------------------------
# Private helpers / data-loading utilities
# ---------------------------------------------------------------------------
from ludwig.visualize._utils import (
    _convert_ground_truth,
    _CSV_SUFFIX,
    _encode_categorical_feature,
    _extract_ground_truth_values,
    _get_cols_from_predictions,
    _get_ground_truth_df,
    _load_training_stats,
    _PARQUET_SUFFIX,
    _PREDICTIONS_SUFFIX,
    _PROBABILITIES_SUFFIX,
    _validate_output_feature_name_from_test_stats,
    _validate_output_feature_name_from_train_stats,
    _vectorize_ground_truth,
    convert_to_list,
    generate_filename_template_path,
    load_data_for_viz,
    load_training_stats_for_viz,
    validate_conf_thresholds_and_probabilities_2d_3d,
)

# ---------------------------------------------------------------------------
# CLI entry point + registry
# ---------------------------------------------------------------------------
from ludwig.visualize.cli import (
    cli,
    get_visualizations_registry,
)

# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------
from ludwig.visualize.confusion import (
    confusion_matrix,
    confusion_matrix_cli,
)

# ---------------------------------------------------------------------------
# ROC, precision-recall, calibration curves
# ---------------------------------------------------------------------------
from ludwig.visualize.curves import (
    calibration_1_vs_all,
    calibration_1_vs_all_cli,
    calibration_multiclass,
    calibration_multiclass_cli,
    precision_recall_curves,
    precision_recall_curves_cli,
    precision_recall_curves_from_test_statistics,
    precision_recall_curves_from_test_statistics_cli,
    roc_curves,
    roc_curves_cli,
    roc_curves_from_test_statistics,
    roc_curves_from_test_statistics_cli,
)

# ---------------------------------------------------------------------------
# Hyperopt
# ---------------------------------------------------------------------------
from ludwig.visualize.hyperopt import (
    _convert_space_to_dtype,
    hyperopt_hiplot,
    hyperopt_hiplot_cli,
    hyperopt_report,
    hyperopt_report_cli,
    hyperopt_results_to_dataframe,
)

# ---------------------------------------------------------------------------
# Performance comparisons + frequency
# ---------------------------------------------------------------------------
from ludwig.visualize.performance import (
    compare_classifiers_multiclass_multimetric,
    compare_classifiers_multiclass_multimetric_cli,
    compare_classifiers_performance_changing_k,
    compare_classifiers_performance_changing_k_cli,
    compare_classifiers_performance_from_pred,
    compare_classifiers_performance_from_pred_cli,
    compare_classifiers_performance_from_prob,
    compare_classifiers_performance_from_prob_cli,
    compare_classifiers_performance_subset,
    compare_classifiers_performance_subset_cli,
    compare_classifiers_predictions,
    compare_classifiers_predictions_cli,
    compare_classifiers_predictions_distribution,
    compare_classifiers_predictions_distribution_cli,
    compare_performance,
    compare_performance_cli,
    frequency_vs_f1,
    frequency_vs_f1_cli,
)

# ---------------------------------------------------------------------------
# Confidence thresholding + binary threshold
# ---------------------------------------------------------------------------
from ludwig.visualize.threshold import (
    binary_threshold_vs_metric,
    binary_threshold_vs_metric_cli,
    confidence_thresholding,
    confidence_thresholding_2thresholds_2d,
    confidence_thresholding_2thresholds_2d_cli,
    confidence_thresholding_2thresholds_3d,
    confidence_thresholding_2thresholds_3d_cli,
    confidence_thresholding_cli,
    confidence_thresholding_data_vs_acc,
    confidence_thresholding_data_vs_acc_cli,
    confidence_thresholding_data_vs_acc_subset,
    confidence_thresholding_data_vs_acc_subset_cli,
    confidence_thresholding_data_vs_acc_subset_per_class,
    confidence_thresholding_data_vs_acc_subset_per_class_cli,
)

# ---------------------------------------------------------------------------
# Training / learning curves
# ---------------------------------------------------------------------------
from ludwig.visualize.training import (
    learning_curves,
    learning_curves_cli,
)

__all__ = [
    # constants
    "_PREDICTIONS_SUFFIX",
    "_PROBABILITIES_SUFFIX",
    "_CSV_SUFFIX",
    "_PARQUET_SUFFIX",
    # private helpers
    "_convert_ground_truth",
    "_vectorize_ground_truth",
    "_encode_categorical_feature",
    "_get_ground_truth_df",
    "_extract_ground_truth_values",
    "_get_cols_from_predictions",
    "_load_training_stats",
    "_validate_output_feature_name_from_train_stats",
    "_validate_output_feature_name_from_test_stats",
    # public utils
    "validate_conf_thresholds_and_probabilities_2d_3d",
    "load_data_for_viz",
    "load_training_stats_for_viz",
    "convert_to_list",
    "generate_filename_template_path",
    # training
    "learning_curves_cli",
    "learning_curves",
    # performance
    "compare_performance_cli",
    "compare_performance",
    "compare_classifiers_performance_from_prob_cli",
    "compare_classifiers_performance_from_prob",
    "compare_classifiers_performance_from_pred_cli",
    "compare_classifiers_performance_from_pred",
    "compare_classifiers_performance_subset_cli",
    "compare_classifiers_performance_subset",
    "compare_classifiers_performance_changing_k_cli",
    "compare_classifiers_performance_changing_k",
    "compare_classifiers_multiclass_multimetric_cli",
    "compare_classifiers_multiclass_multimetric",
    "compare_classifiers_predictions_cli",
    "compare_classifiers_predictions",
    "compare_classifiers_predictions_distribution_cli",
    "compare_classifiers_predictions_distribution",
    "frequency_vs_f1_cli",
    "frequency_vs_f1",
    # threshold
    "confidence_thresholding_cli",
    "confidence_thresholding",
    "confidence_thresholding_data_vs_acc_cli",
    "confidence_thresholding_data_vs_acc",
    "confidence_thresholding_data_vs_acc_subset_cli",
    "confidence_thresholding_data_vs_acc_subset",
    "confidence_thresholding_data_vs_acc_subset_per_class_cli",
    "confidence_thresholding_data_vs_acc_subset_per_class",
    "confidence_thresholding_2thresholds_2d_cli",
    "confidence_thresholding_2thresholds_2d",
    "confidence_thresholding_2thresholds_3d_cli",
    "confidence_thresholding_2thresholds_3d",
    "binary_threshold_vs_metric_cli",
    "binary_threshold_vs_metric",
    # curves
    "precision_recall_curves_cli",
    "precision_recall_curves",
    "precision_recall_curves_from_test_statistics_cli",
    "precision_recall_curves_from_test_statistics",
    "roc_curves_cli",
    "roc_curves",
    "roc_curves_from_test_statistics_cli",
    "roc_curves_from_test_statistics",
    "calibration_1_vs_all_cli",
    "calibration_1_vs_all",
    "calibration_multiclass_cli",
    "calibration_multiclass",
    # confusion
    "confusion_matrix_cli",
    "confusion_matrix",
    # hyperopt
    "hyperopt_report_cli",
    "hyperopt_report",
    "hyperopt_hiplot_cli",
    "hyperopt_hiplot",
    "_convert_space_to_dtype",
    "hyperopt_results_to_dataframe",
    # cli
    "get_visualizations_registry",
    "cli",
]
