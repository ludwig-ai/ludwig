import logging
from typing import Dict, List

from tabulate import tabulate

from ludwig.constants import COMBINED, LOSS
from ludwig.utils.metric_utils import TrainerMetric

logger = logging.getLogger(__name__)


def get_metric_value_or_empty(metrics_log: Dict[str, List[TrainerMetric]], metric_name: str):
    """Returns the metric value if it exists or empty."""
    if metric_name not in metrics_log:
        return ""
    return metrics_log[metric_name][-1][-1]


def print_table_for_single_output_feature(
    train_metrics_log: Dict[str, List[TrainerMetric]],
    validation_metrics_log: Dict[str, List[TrainerMetric]],
    test_metrics_log: Dict[str, List[TrainerMetric]],
    combined_loss_for_each_split: List[float],
) -> None:
    """Prints the metrics table for a single output feature.

    Args:
        train_metrics_log: Dict from metric name to list of TrainerMetric.
        validation_metrics_log: Dict from metric name to list of TrainerMetric.
        test_metrics_log: Dict from metric name to list of TrainerMetric.
    """
    # Get the superset of metric names across all splits.
    all_metric_names = set()
    all_metric_names.update(train_metrics_log.keys())
    all_metric_names.update(validation_metrics_log.keys())
    all_metric_names.update(test_metrics_log.keys())
    all_metric_names = sorted(list(all_metric_names))

    # Assemble the printed table.
    # Each item in the printed_table corresponds to a row in the printed table.
    printed_table = [["train", "validation", "test"]]
    for metric_name in all_metric_names:
        metrics_for_each_split = [
            get_metric_value_or_empty(train_metrics_log, metric_name),
            get_metric_value_or_empty(validation_metrics_log, metric_name),
            get_metric_value_or_empty(test_metrics_log, metric_name),
        ]
        printed_table.append([metric_name] + metrics_for_each_split)

    # Add combined loss.
    printed_table.append(["combined_loss"] + combined_loss_for_each_split)

    logger.info(tabulate(printed_table, headers="firstrow", tablefmt="fancy_grid", floatfmt=".4f"))


def print_metrics_table(
    output_features: Dict[str, "OutputFeature"],  # noqa
    train_metrics_log: Dict[str, Dict[str, List[TrainerMetric]]],
    validation_metrics_log: Dict[str, Dict[str, List[TrainerMetric]]],
    test_metrics_log: Dict[str, Dict[str, List[TrainerMetric]]],
):
    """Prints a table of metrics table for each output feature, for each split.

    Example:
    ╒═══════════════╤═════════╤══════════════╤════════╕
    │               │   train │   validation │   test │
    ╞═══════════════╪═════════╪══════════════╪════════╡
    │ accuracy      │  0.8157 │       0.6966 │ 0.8090 │
    ├───────────────┼─────────┼──────────────┼────────┤
    │ loss          │  0.4619 │       0.5039 │ 0.4488 │
    ├───────────────┼─────────┼──────────────┼────────┤
    │ precision     │  0.8274 │       0.6250 │ 0.7818 │
    ├───────────────┼─────────┼──────────────┼────────┤
    │ recall        │  0.6680 │       0.4545 │ 0.6615 │
    ├───────────────┼─────────┼──────────────┼────────┤
    │ roc_auc       │  0.8471 │       0.7706 │ 0.8592 │
    ├───────────────┼─────────┼──────────────┼────────┤
    │ specificity   │  0.9105 │       0.8393 │ 0.8938 │
    ├───────────────┼─────────┼──────────────┼────────┤
    │ combined_loss │  0.4619 │       0.5039 │ 0.4488 │
    ╘═══════════════╧═════════╧══════════════╧════════╛
    """
    # Obtain the combined loss, which is the same across all output features.
    combined_loss_for_each_split = [
        get_metric_value_or_empty(train_metrics_log[COMBINED], LOSS),
        get_metric_value_or_empty(validation_metrics_log[COMBINED], LOSS),
        get_metric_value_or_empty(test_metrics_log[COMBINED], LOSS),
    ]

    for output_feature_name in sorted(output_features.keys()):
        if output_feature_name == COMBINED:
            # Skip the combined output feature. The combined loss will be added to each output feature's table.
            continue
        print_table_for_single_output_feature(
            train_metrics_log[output_feature_name],
            validation_metrics_log[output_feature_name],
            test_metrics_log[output_feature_name],
            combined_loss_for_each_split,
        )
