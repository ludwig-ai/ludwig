import os
from collections import OrderedDict
from pprint import pformat

from ludwig.constants import TYPE, COMBINED
from ludwig.features.feature_registries import output_type_registry
from ludwig.predict import logger
from ludwig.utils.data_utils import save_csv, save_json
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.print_utils import repr_ordered_dict


def calculate_overall_stats(test_stats, output_features, dataset,
                            train_set_metadata):
    for output_feature in output_features:
        feature = get_from_registry(
            output_feature[TYPE],
            output_type_registry
        )
        feature.calculate_overall_stats(
            test_stats, output_feature, dataset, train_set_metadata
        )


def save_prediction_outputs(
        postprocessed_output,
        experiment_dir_name,
        skip_output_types=None
):
    if skip_output_types is None:
        skip_output_types = set()
    csv_filename = os.path.join(experiment_dir_name, '{}_{}.csv')
    for output_field, outputs in postprocessed_output.items():
        for output_type, values in outputs.items():
            if output_type not in skip_output_types:
                save_csv(
                    csv_filename.format(output_field, output_type),
                    values
                )


def save_test_statistics(test_stats, experiment_dir_name):
    test_stats_fn = os.path.join(
        experiment_dir_name,
        'test_statistics.json'
    )
    save_json(test_stats_fn, test_stats)


def print_test_results(test_stats):
    for output_field, result in test_stats.items():
        if (output_field != COMBINED or
                (output_field == COMBINED and len(test_stats) > 2)):
            logger.info('\n===== {} ====='.format(output_field))
            for metric in sorted(list(result)):
                if metric != 'confusion_matrix' and metric != 'roc_curve':
                    value = result[metric]
                    if isinstance(value, OrderedDict):
                        value_repr = repr_ordered_dict(value)
                    else:
                        value_repr = pformat(result[metric], indent=2)
                    logger.info(
                        '{0}: {1}'.format(
                            metric,
                            value_repr
                        )
                    )
