import logging
from collections import OrderedDict
from typing import Dict

from tabulate import tabulate

from ludwig.constants import COMBINED, LOSS
from ludwig.utils.metric_utils import TrainerMetric

logger = logging.getLogger(__name__)


class MetricsPrintedTable:
    """Maintains a table data structure used for logging metrics.

    ╒════════════╤════════════╤════════╤═════════════╤══════════╤═══════════╕
    │ Survived   │   accuracy │   loss │   precision │   recall │   roc_auc │
    ╞════════════╪════════════╪════════╪═════════════╪══════════╪═══════════╡
    │ train      │     0.7420 │ 0.7351 │      0.7107 │   0.5738 │    0.7659 │
    ├────────────┼────────────┼────────┼─────────────┼──────────┼───────────┤
    │ validation │     0.7079 │ 0.9998 │      0.6061 │   0.6061 │    0.7354 │
    ├────────────┼────────────┼────────┼─────────────┼──────────┼───────────┤
    │ test       │     0.7360 │ 0.7620 │      0.6667 │   0.5538 │    0.7358 │
    ╘════════════╧════════════╧════════╧═════════════╧══════════╧═══════════╛
    ╒════════════╤════════╕
    │ combined   │   loss │
    ╞════════════╪════════╡
    │ train      │ 0.7351 │
    ├────────────┼────────┤
    │ validation │ 0.9998 │
    ├────────────┼────────┤
    │ test       │ 0.7620 │
    ╘════════════╧════════╛
    """

    def __init__(self, output_features: Dict[str, "OutputFeature"]):  # noqa
        self.printed_table = OrderedDict()
        for output_feature_name, output_feature in output_features.items():
            self.printed_table[output_feature_name] = [[output_feature_name] + output_feature.metric_names]
        self.printed_table[COMBINED] = [[COMBINED, LOSS]]

        # Establish the printed table's order of metrics (used for appending metrics in the right order).
        self.metrics_headers = {}
        for output_feature_name in output_features.keys():
            # [0]: The header is the first row, which contains names of metrics.
            # [1:]: Skip the first column as it's just the name of the output feature, not an actual metric name.
            self.metrics_headers[output_feature_name] = self.printed_table[output_feature_name][0][1:]
        self.metrics_headers[COMBINED] = [LOSS]

    def add_metrics_to_printed_table(self, metrics_log: Dict[str, Dict[str, TrainerMetric]], split_name: str):
        """Add metrics to tables by the order of the table's metric header."""
        for output_feature_name, output_feature_metrics in metrics_log.items():
            printed_metrics = []
            for metric_name in self.metrics_headers[output_feature_name]:
                # Metrics may be missing if should_evaluate_train is False.
                if metric_name in output_feature_metrics and output_feature_metrics[metric_name]:
                    printed_metrics.append(output_feature_metrics[metric_name][-1][-1])
                else:
                    printed_metrics.append("")
            self.printed_table[output_feature_name].append([split_name] + printed_metrics)

    def log_info(self):
        for output_feature, table in self.printed_table.items():
            logger.info(tabulate(table, headers="firstrow", tablefmt="fancy_grid", floatfmt=".4f"))
