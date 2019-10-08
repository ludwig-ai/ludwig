#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2019 Uber Technologies, Inc.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import sys
from collections import OrderedDict
from pprint import pformat

from ludwig.contrib import contrib_command
from ludwig.data.postprocessing import postprocess
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.features.feature_registries import output_type_registry
from ludwig.globals import LUDWIG_VERSION, is_on_master, set_on_master
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.models.model import load_model_and_definition
from ludwig.utils.data_utils import save_csv
from ludwig.utils.data_utils import save_json
from ludwig.utils.misc import get_from_registry, \
    find_non_existing_dir_by_adding_suffix
from ludwig.utils.print_utils import logging_level_registry, repr_ordered_dict
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.print_utils import print_ludwig


logger = logging.getLogger(__name__)


def full_predict(
        model_path,
        data_csv=None,
        data_hdf5=None,
        split='test',
        batch_size=128,
        skip_save_unprocessed_output=False,
        skip_save_test_predictions=False,
        skip_save_test_statistics=False,
        output_directory='results',
        evaluate_performance=True,
        gpus=None,
        gpu_fraction=1.0,
        use_horovod=False,
        debug=False,
        **kwargs
):
    if is_on_master():
        logger.info('Dataset path: {}'.format(
            data_csv if data_csv is not None else data_hdf5))
        logger.info('Model path: {}'.format(model_path))
        logger.info('')

    train_set_metadata_json_fp = os.path.join(
        model_path,
        TRAIN_SET_METADATA_FILE_NAME
    )

    # preprocessing
    dataset, train_set_metadata = preprocess_for_prediction(
        model_path,
        split,
        data_csv,
        data_hdf5,
        train_set_metadata_json_fp,
        evaluate_performance
    )

    # run the prediction
    if is_on_master():
        print_boxed('LOADING MODEL')
    model, model_definition = load_model_and_definition(model_path,
                                                        use_horovod=use_horovod)

    prediction_results = predict(
        dataset,
        train_set_metadata,
        model,
        model_definition,
        batch_size,
        evaluate_performance,
        gpus,
        gpu_fraction,
        debug
    )
    model.close_session()

    if is_on_master():
        # setup directories and file names
        experiment_dir_name = find_non_existing_dir_by_adding_suffix(output_directory)

        # if we are skipping all saving,
        # there is no need to create a directory that will remain empty
        should_create_exp_dir = not (
                skip_save_unprocessed_output and
                skip_save_test_predictions and
                skip_save_test_statistics
        )
        if should_create_exp_dir:
                os.makedirs(experiment_dir_name)

        # postprocess
        postprocessed_output = postprocess(
            prediction_results,
            model_definition['output_features'],
            train_set_metadata,
            experiment_dir_name,
            skip_save_unprocessed_output or not is_on_master()
        )

        if not skip_save_test_predictions:
            save_prediction_outputs(postprocessed_output, experiment_dir_name)

        if evaluate_performance:
            print_test_results(prediction_results)
            if not skip_save_test_statistics:
                save_test_statistics(prediction_results, experiment_dir_name)

        logger.info('Saved to: {0}'.format(experiment_dir_name))


def predict(
        dataset,
        train_set_metadata,
        model,
        model_definition,
        batch_size=128,
        evaluate_performance=True,
        gpus=None,
        gpu_fraction=1.0,
        debug=False
):
    """Computes predictions based on the computed model.
        :param dataset: Dataset containing the data to calculate
               the predictions from.
        :type dataset: Dataset
        :param model: The trained model used to produce the predictions.
        :type model: Model
        :param model_definition: The model definition of the model to use
               for obtaining predictions
        :type model_definition: Dictionary
        :param batch_size: The size of batches when computing the predictions.
        :type batch_size: Integer
        :param evaluate_performance: If this parameter is False, only the predictions
               will be returned, if it is True, also performance metrics
               will be calculated on the predictions. It requires the data
               to contain also ground truth for the output features, otherwise
               the metrics cannot be computed.
        :type evaluate_performance: Bool
        :type gpus: List
        :type gpu_fraction: Integer
        :param debug: If true turns on tfdbg with inf_or_nan checks.
        :type debug: Boolean

        :returns: A dictionary containing the predictions of each output feature,
                  alongside with statistics on the quality of those predictions
                  (if evaluate_performance is True).
        """
    if is_on_master():
        print_boxed('PREDICT')
    test_stats = model.predict(
        dataset,
        batch_size,
        evaluate_performance=evaluate_performance,
        gpus=gpus,
        gpu_fraction=gpu_fraction
    )

    if evaluate_performance:
        calculate_overall_stats(
            test_stats,
            model_definition['output_features'],
            dataset,
            train_set_metadata
        )

    return test_stats


def calculate_overall_stats(test_stats, output_features, dataset,
                            train_set_metadata):
    for output_feature in output_features:
        feature = get_from_registry(
            output_feature['type'],
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
                save_csv(csv_filename.format(output_field, output_type), values)


def save_test_statistics(test_stats, experiment_dir_name):
    test_stats_fn = os.path.join(
        experiment_dir_name,
        'test_statistics.json'
    )
    save_json(test_stats_fn, test_stats)


def print_test_results(test_stats):
    for output_field, result in test_stats.items():
        if (output_field != 'combined' or
                (output_field == 'combined' and len(test_stats) > 2)):
            logger.info('\n===== {} ====='.format(output_field))
            for measure in sorted(list(result)):
                if measure != 'confusion_matrix' and measure != 'roc_curve':
                    value = result[measure]
                    if isinstance(value, OrderedDict):
                        value_repr = repr_ordered_dict(value)
                    else:
                        value_repr = pformat(result[measure], indent=2)
                    logger.info(
                        '{0}: {1}'.format(
                            measure,
                            value_repr
                        )
                    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and uses it to predict',
        prog='ludwig predict',
        usage='%(prog)s [options]'
    )

    # ---------------
    # Data parameters
    # ---------------
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--data_csv',
        help='input data CSV file. '
             'If it has a split column, it will be used for splitting '
             '(0: train, 1: validation, 2: test), '
             'otherwise the dataset will be randomly split'
    )
    group.add_argument(
        '--data_hdf5',
        help='input data HDF5 file. It is an intermediate preprocess version of'
             ' the input CSV created the first time a CSV file is used in the '
             'same directory with the same name and a hdf5 extension'
    )
    parser.add_argument(
        '--train_set_metadata_json',
        help='input metadata JSON file. It is an intermediate preprocess file '
             'containing the mappings of the input CSV created the first time '
             'a CSV file is used in the same directory with the same name and '
             'a json extension'
    )

    parser.add_argument(
        '-s',
        '--split',
        default='test',
        choices=['training', 'validation', 'test', 'full'],
        help='the split to test the model on'
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument(
        '-m',
        '--model_path',
        help='model to load',
        required=True
    )

    # -------------------------
    # Output results parameters
    # -------------------------
    parser.add_argument(
        '-od',
        '--output_directory',
        type=str,
        default='results',
        help='directory that contains the results'
    )
    parser.add_argument(
        '-ssuo',
        '--skip_save_unprocessed_output',
        help='skips saving intermediate NPY output files',
        action='store_true', default=False
    )
    parser.add_argument(
        '-sstp',
        '--skip_save_test_predictions',
        help='skips saving test predictions CSV files',
        action='store_true', default=False
    )
    parser.add_argument(
        '-sstes',
        '--skip_save_test_statistics',
        help='skips saving test statistics JSON file',
        action='store_true', default=False
    )


    # ------------------
    # Generic parameters
    # ------------------
    parser.add_argument(
        '-bs',
        '--batch_size',
        type=int,
        default=128,
        help='size of batches'
    )

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument(
        '-g',
        '--gpus',
        type=int,
        default=0,
        help='list of gpu to use'
    )
    parser.add_argument(
        '-gf',
        '--gpu_fraction',
        type=float,
        default=1.0,
        help='fraction of gpu memory to initialize the process with'
    )
    parser.add_argument(
        '-uh',
        '--use_horovod',
        action='store_true',
        default=False,
        help='uses horovod for distributed training'
    )
    parser.add_argument(
        '-dbg',
        '--debug',
        action='store_true',
        default=False,
        help='enables debugging mode'
    )
    parser.add_argument(
        '-l',
        '--logging_level',
        default='info',
        help='the level of logging to use',
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    args = parser.parse_args(sys_argv)
    args.evaluate_performance = False

    logging.getLogger('ludwig').setLevel(
        logging_level_registry[args.logging_level]
    )
    set_on_master(args.use_horovod)

    if is_on_master():
        print_ludwig('Predict', LUDWIG_VERSION)

    full_predict(**vars(args))


if __name__ == '__main__':
    contrib_command("predict", *sys.argv)
    cli(sys.argv[1:])
