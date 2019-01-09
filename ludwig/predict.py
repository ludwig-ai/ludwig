#! /usr/bin/env python
# coding=utf-8
# Copyright 2019 The Ludwig Authors. All Rights Reserved.
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
from pprint import pformat

from ludwig.data.postprocessing import postprocess
from ludwig.data.preprocessing import preprocess_for_prediction
from ludwig.features.feature_registries import output_type_registry
from ludwig.globals import LUDWIG_VERSION
from ludwig.models.model import load_model_and_definition
from ludwig.utils.data_utils import save_json, save_csv
from ludwig.utils.misc import get_from_registry
from ludwig.utils.print_utils import logging_level_registry
from ludwig.utils.print_utils import print_boxed
from ludwig.utils.print_utils import print_ludwig


def full_predict(
        model_path,
        data_csv=None,
        data_hdf5=None,
        metadata_json=None,
        dataset_type='generic',
        split='test',
        batch_size=128,
        skip_save_unprocessed_output=False,
        output_directory='results',
        only_predictions=False,
        gpus=None,
        gpu_fraction=1.0,
        debug=False,
        **kwargs
):
    # setup directories and file names
    experiment_dir_name = output_directory
    suffix = 0
    while os.path.exists(experiment_dir_name):
        experiment_dir_name = output_directory + '_' + str(suffix)
        suffix += 1

    logging.info('Dataset type: {}'.format(dataset_type))
    logging.info('Dataset path: {}'.format(
        data_csv if data_csv is not None else data_hdf5))
    logging.info('Model path: {}'.format(model_path))
    logging.info('Output path: {}'.format(experiment_dir_name))
    logging.info('')

    # preprocessing
    dataset, metadata = preprocess_for_prediction(
        model_path,
        split,
        dataset_type,
        data_csv,
        data_hdf5,
        metadata_json,
        only_predictions
    )

    # run the prediction
    print_boxed("LOADING MODEL")
    model, model_definition = load_model_and_definition(model_path)

    prediction_results = predict(
        dataset,
        model,
        model_definition,
        batch_size,
        only_predictions,
        gpus,
        gpu_fraction,
        debug
    )
    model.close_session()

    os.mkdir(experiment_dir_name)

    # postprocess
    postprocessed_output = postprocess(
        prediction_results,
        model_definition['output_features'],
        metadata,
        experiment_dir_name,
        skip_save_unprocessed_output
    )

    save_prediction_outputs(postprocessed_output, experiment_dir_name)

    if not only_predictions:
        print_prediction_results(prediction_results)
        save_prediction_statistics(prediction_results, experiment_dir_name)

    logging.info('Saved to: {0}'.format(experiment_dir_name))


def predict(
        dataset,
        model,
        model_definition,
        batch_size=128,
        only_predictions=False,
        gpus=None,
        gpu_fraction=1.0,
        debug=False
):
    """Predicts the $\hat{y}$ based on the computed Model.
        :param dataset: The dataset to test on
        :type dataset: # TODO
        :param model: The model trained that is now being evaluated.
        :type model:
        :model_definition: Model definition file read in through a yaml parser
        :type model_definition: Dictionary
        :param batch_size: The size of each batch that is being used
        :type batch_size: Integer
        :param only_predictions: # TODO
        :type only_predictions: Bool
        :param gpus: List of GPUs to use
        :type gpus: List
        :param gpu_fraction: Percentage of gpu memory to use
        :type gpu_fraction: Float
        :param debug: Whether the function is being debugged or not
        :type debug: Boolean

        :returns: Test Statistics for inference and evaluation of the model
                  performance.
        """
    print_boxed("PREDICT")
    test_stats = model.predict(
        dataset,
        batch_size,
        only_predictions=only_predictions,
        gpus=gpus,
        gpu_fraction=gpu_fraction
    )

    if not only_predictions:
        calculate_overall_stats(
            test_stats,
            model_definition['output_features'],
            dataset
        )

    return test_stats


def calculate_overall_stats(test_stats, output_features, dataset):
    for output_feature in output_features:
        feature = get_from_registry(
            output_feature['type'],
            output_type_registry
        )
        feature.calculate_overall_stats(
            test_stats, output_feature, dataset
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


def save_prediction_statistics(prediction_stats, experiment_dir_name):
    test_stats_fn = os.path.join(
        experiment_dir_name,
        'prediction_statistics.json'
    )
    save_json(test_stats_fn, prediction_stats)


def print_prediction_results(prediction_stats):
    for output_field, result in prediction_stats.items():
        if (output_field != 'combined' or
                (output_field == 'combined' and len(prediction_stats) > 2)):
            logging.info('===== {} ====='.format(output_field))
            for measure in sorted(list(result)):
                if measure != 'confusion_matrix' and measure != 'roc_curve':
                    logging.info(
                        '{0}: {1}'.format(
                            measure,
                            pformat(result[measure], indent=2)
                        )
                    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script loads a pretrained model '
                    'and uses it to predict.',
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
        '--metadata_json',
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
    parser.add_argument(
        '-op',
        '--only_predictions',
        action='store_true',
        default=False,
        help='skip metrics calculation'
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
        choices=["critical", "error", "warning", "info", "debug", "notset"]
    )

    args = parser.parse_args(sys_argv)

    logging.basicConfig(
        stream=sys.stdout,
        # filename='log.log', TODO - remove these?
        # filemode='w',
        level=logging_level_registry[args.logging_level],
        format='%(message)s'
    )

    print_ludwig('Predict', LUDWIG_VERSION)

    full_predict(**vars(args))


if __name__ == '__main__':
    cli(sys.argv[1:])
