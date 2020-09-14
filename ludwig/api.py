# !/usr/bin/env python
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
"""
    File name: LudwigModel.py
    Author: Piero Molino
    Date created: 5/21/2019
    Date last modified: 5/21/2019
    Python Version: 3+
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import tempfile
from pprint import pformat

import numpy as np
import pandas as pd

import ludwig.contrib

ludwig.contrib.contrib_import()

from ludwig.constants import PREPROCESSING, TRAINING, VALIDATION, TEST
from ludwig.contrib import contrib_command
from ludwig.data.postprocessing import convert_predictions, postprocess
from ludwig.data.preprocessing import preprocess_for_training, \
    preprocess_for_prediction, load_metadata
from ludwig.features.feature_registries import \
    update_model_definition_with_metadata
from ludwig.globals import set_on_master, is_on_master, \
    TRAIN_SET_METADATA_FILE_NAME, MODEL_HYPERPARAMETERS_FILE_NAME, \
    MODEL_WEIGHTS_FILE_NAME, set_disable_progressbar
from ludwig.models.ecd import ECD
from ludwig.models.predictor import Predictor, save_prediction_outputs, \
    calculate_overall_stats, print_evaluation_stats, save_evaluation_stats
from ludwig.models.trainer import Trainer
from ludwig.modules.metric_modules import get_best_function
from ludwig.utils.data_utils import save_json, override_in_memory_flag, \
    load_json, generate_kfold_splits
from ludwig.utils.horovod_utils import should_use_horovod
from ludwig.utils.misc_utils import get_experiment_dir_name, get_file_names, \
    get_experiment_description, find_non_existing_dir_by_adding_suffix
from ludwig.utils.tf_utils import initialize_tensorflow

import yaml

from ludwig.utils.defaults import default_random_seed, merge_with_defaults
from ludwig.utils.print_utils import logging_level_registry, print_boxed

logger = logging.getLogger(__name__)


class LudwigModel:

    def __init__(self,
                 model_definition=None,
                 model_definition_fp=None,
                 logging_level=logging.ERROR,
                 use_horovod=None,
                 gpus=None,
                 gpu_memory_limit=None,
                 allow_parallel_threads=True,
                 random_seed=default_random_seed):
        # check for model_definition and model_definition_file
        if model_definition is None and model_definition_fp is None:
            raise ValueError(
                'Either model_definition of model_definition_file have to be'
                'not None to initialize a LudwigModel'
            )
        if model_definition is not None and model_definition_fp is not None:
            raise ValueError(
                'Only one between model_definition and '
                'model_definition_file can be provided'
            )

        # merge model definition with defaults
        if model_definition_fp is not None:
            with open(model_definition_fp, 'r') as def_file:
                raw_model_definition = yaml.safe_load(def_file)
        else:
            raw_model_definition = copy.deepcopy(model_definition)
        self.model_definition = merge_with_defaults(raw_model_definition)
        self.model_definition_fp = model_definition_fp

        # setup horovod
        self._horovod = None
        if should_use_horovod(use_horovod):
            import horovod.tensorflow
            self._horovod = horovod.tensorflow
            self._horovod.init()
        # todo refactoring: figure out it this belongs here or
        #  in Trainer and Predictor. It probably belongs here
        set_on_master(use_horovod)

        # setup logging
        self.set_logging_level(logging_level)

        # setup TensorFlow
        initialize_tensorflow(gpus, gpu_memory_limit, allow_parallel_threads,
                              self._horovod)
        # todo refactoring: decide where to put this,
        #  here or at the beginning of training.
        #  Either way make sure it is called before the model is initialized.
        # tf.random.set_seed(random_seed)

        # setup model
        self.model = None
        self.training_set_metadata = None
        self.exp_dir_name = ''

    # def train_pseudo(self, data, training_params):
    #     # process_data ignores self.training_set_metadata if it's None and computes a new one from the actual data
    #     # or uses the procided one and does not compute a new one if it is not None
    #     preproc_data, training_set_metadata = preprocess_data(
    #         data,
    #         self.training_set_metadata
    #     )
    #     self.training_set_metadata = training_set_metadata
    #
    #     # this is done only if the model is not loaded
    #     if not self.model:
    #         update_model_definition_with_metadata(
    #             self.model_definition,
    #             training_set_metadata
    #         )
    #         self.model = ECD(self.model_definition)
    #
    #     trainer = Trainer(training_params)
    #     training_stats = trainer.train(self.model, preproc_data)
    #     return training_stats

    def train(
            self,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name='api_experiment',
            model_name='run',
            model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            output_directory='results',
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        """This function is used to perform a full training of the model on the
           specified dataset.

        # Inputs

        :param data_df: (DataFrame) dataframe containing data. If it has a split
               column, it will be used for splitting (0: train, 1: validation,
               2: test), otherwise the dataset will be randomly split
        :param data_train_df: (DataFrame) dataframe containing training data
        :param data_validation_df: (DataFrame) dataframe containing validation
               data
        :param data_test_df: (DataFrame dataframe containing test data
        :param data_csv: (string) input data CSV file. If it has a split column,
               it will be used for splitting (0: train, 1: validation, 2: test),
               otherwise the dataset will be randomly split
        :param data_train_csv: (string) input train data CSV file
        :param data_validation_csv: (string) input validation data CSV file
        :param data_test_csv: (string) input test data CSV file
        :param data_hdf5: (string) input data HDF5 file. It is an intermediate
               preprocess  version of the input CSV created the first time a CSV
               file is used in the same directory with the same name and a hdf5
               extension
        :param data_train_hdf5: (string) input train data HDF5 file. It is an
               intermediate preprocess  version of the input CSV created the
               first time a CSV file is used in the same directory with the same
               name and a hdf5 extension
        :param data_validation_hdf5: (string) input validation data HDF5 file.
               It is an intermediate preprocess version of the input CSV created
               the first time a CSV file is used in the same directory with the
               same name and a hdf5 extension
        :param data_test_hdf5: (string) input test data HDF5 file. It is an
               intermediate preprocess  version of the input CSV created the
               first time a CSV file is used in the same directory with the same
               name and a hdf5 extension
        :param data_dict: (dict) input data dictionary. It is expected to
               contain one key for each field and the values have to be lists of
               the same length. Each index in the lists corresponds to one
               datapoint. For example a data set consisting of two datapoints
               with a text and a class may be provided as the following dict
               `{'text_field_name': ['text of the first datapoint', text of the
               second datapoint'], 'class_filed_name': ['class_datapoints_1',
               'class_datapoints_2']}`.
        :param data_train_dict: (dict) input training data dictionary. It is
               expected to contain one key for each field and the values have
               to be lists of the same length. Each index in the lists
               corresponds to one datapoint. For example a data set consisting
               of two datapoints with a text and a class may be provided as the
               following dict:
               `{'text_field_name': ['text of the first datapoint', 'text of the
               second datapoint'], 'class_field_name': ['class_datapoint_1',
               'class_datapoint_2']}`.
        :param data_validation_dict: (dict) input validation data dictionary. It
               is expected to contain one key for each field and the values have
               to be lists of the same length. Each index in the lists
               corresponds to one datapoint. For example a data set consisting
               of two datapoints with a text and a class may be provided as the
               following dict:
               `{'text_field_name': ['text of the first datapoint', 'text of the
               second datapoint'], 'class_field_name': ['class_datapoint_1',
               'class_datapoint_2']}`.
        :param data_test_dict: (dict) input test data dictionary. It is
               expected to contain one key for each field and the values have
               to be lists of the same length. Each index in the lists
               corresponds to one datapoint. For example a data set consisting
               of two datapoints with a text and a class may be provided as the
               following dict:
               `{'text_field_name': ['text of the first datapoint', 'text of the
               second datapoint'], 'class_field_name': ['class_datapoint_1',
               'class_datapoint_2']}`.
        :param training_set_metadata_json: (string) input metadata JSON file. It is an
               intermediate preprocess file containing the mappings of the input
               CSV created the first time a CSV file is used in the same
               directory with the same name and a json extension
        :param experiment_name: (string) a name for the experiment, used for the save
               directory
        :param model_name: (string) a name for the model, used for the save
               directory
        :param model_load_path: (string) path of a pretrained model to load as
               initialization
        :param model_resume_path: (string) path of a the model directory to
               resume training of
        :param skip_save_training_description: (bool, default: `False`) disables
               saving the description JSON file.
        :param skip_save_training_statistics: (bool, default: `False`) disables
               saving training statistics JSON file.
        :param skip_save_model: (bool, default: `False`) disables
               saving model weights and hyperparameters each time the model
               improves. By default Ludwig saves model weights after each epoch
               the validation metric imrpvoes, but if the model is really big
               that can be time consuming if you do not want to keep
               the weights and just find out what performance can a model get
               with a set of hyperparameters, use this parameter to skip it,
               but the model will not be loadable later on.
        :param skip_save_progress: (bool, default: `False`) disables saving
               progress each epoch. By default Ludwig saves weights and stats
               after each epoch for enabling resuming of training, but if
               the model is really big that can be time consuming and will uses
               twice as much space, use this parameter to skip it, but training
               cannot be resumed later on.
        :param skip_save_log: (bool, default: `False`) disables saving TensorBoard
               logs. By default Ludwig saves logs for the TensorBoard, but if it
               is not needed turning it off can slightly increase the
               overall speed.
        :param skip_save_processed_input: (bool, default: `False`) skips saving
               intermediate HDF5 and JSON files
        :param output_directory: (string, default: `'results'`) directory that
               contains the results
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_memory_limit: (int: default: `None`) maximum memory in MB to allocate
              per GPU device.
        :param allow_parallel_threads: (bool, default: `True`) allow TensorFlow to use
               multithreading parallelism to improve performance at the cost of
               determinism.
        :param random_seed: (int, default`42`) a random seed that is going to be
               used anywhere there is a call to a random number generator: data
               splitting, parameter initialization and training set shuffling
        :param debug: (bool, default: `False`) enables debugging mode

        There are three ways to provide data: by dataframes using the `_df`
        parameters, by CSV using the `_csv` parameters and by HDF5 and JSON,
        using `_hdf5` and `_json` parameters.
        The DataFrame approach uses data previously obtained and put in a
        dataframe, the CSV approach loads data from a CSV file, while HDF5 and
        JSON load previously preprocessed HDF5 and JSON files (they are saved in
        the same directory of the CSV they are obtained from).
        For all three approaches either a full dataset can be provided (which
        will be split randomly according to the split probabilities defined in
        the model definition, by default 70% training, 10% validation and 20%
        test) or, if it contanins a plit column, it will be plit according to
        that column (interpreting 0 as training, 1 as validation and 2 as test).
        Alternatively separated dataframes / CSV / HDF5 files can beprovided
        for each split.

        During training the model and statistics will be saved in a directory
        `[output_dir]/[experiment_name]_[model_name]_n` where all variables are
        resolved to user spiecified ones and `n` is an increasing number
        starting from 0 used to differentiate different runs.


        # Return

        :return: (dict) a dictionary containing training statistics for each
        output feature containing loss and metrics values for each epoch.
        """
        # setup directories and file names
        experiment_dir_name = None
        if model_resume_path is not None:
            if os.path.exists(model_resume_path):
                experiment_dir_name = model_resume_path
            else:
                if is_on_master():
                    logger.info(
                        'Model resume path does not exists, '
                        'starting training from scratch'
                    )
                model_resume_path = None

        if model_resume_path is None:
            if is_on_master():
                experiment_dir_name = get_experiment_dir_name(
                    output_directory,
                    experiment_name,
                    model_name
                )
            else:
                experiment_dir_name = None

        # if we are skipping all saving,
        # there is no need to create a directory that will remain empty
        should_create_exp_dir = not (
                skip_save_training_description and
                skip_save_training_statistics and
                skip_save_model and
                skip_save_progress and
                skip_save_log and
                skip_save_processed_input
        )

        description_fn = training_stats_fn = model_dir = None
        if is_on_master():
            if should_create_exp_dir:
                if not os.path.exists(experiment_dir_name):
                    os.makedirs(experiment_dir_name, exist_ok=True)
            description_fn, training_stats_fn, model_dir = get_file_names(
                experiment_dir_name)

        # save description
        if is_on_master():
            # todo refactoring: fix this
            description = get_experiment_description(
                self.model_definition,
                dataset=dataset,
                training_set=training_set,
                validation_set=validation_set,
                test_set=test_set,
                training_set_metadata=training_set_metadata,
                data_format=data_format,
                random_seed=random_seed
            )
            if not skip_save_training_description:
                save_json(description_fn, description)
            # print description
            logger.info('Experiment name: {}'.format(experiment_name))
            logger.info('Model name: {}'.format(model_name))
            logger.info('Output path: {}'.format(experiment_dir_name))
            logger.info('\n')
            for key, value in description.items():
                logger.info('{}: {}'.format(key, pformat(value, indent=4)))
            logger.info('\n')

        # preprocess
        preprocessed_data = preprocess_for_training(
            self.model_definition,
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=self.model_definition[PREPROCESSING],
            random_seed=random_seed
        )

        (training_set,
         validation_set,
         test_set,
         training_set_metadata) = preprocessed_data
        self.training_set_metadata = training_set_metadata

        if is_on_master():
            logger.info('Training set: {0}'.format(training_set.size))
            if validation_set is not None:
                logger.info('Validation set: {0}'.format(validation_set.size))
            if test_set is not None:
                logger.info('Test set: {0}'.format(test_set.size))

        if is_on_master():
            if not skip_save_model:
                # save train set metadata
                os.makedirs(model_dir, exist_ok=True)
                save_json(
                    os.path.join(
                        model_dir,
                        TRAIN_SET_METADATA_FILE_NAME
                    ),
                    training_set_metadata
                )

        contrib_command("train_init", experiment_directory=experiment_dir_name,
                        experiment_name=experiment_name, model_name=model_name,
                        output_directory=output_directory,
                        resume=model_resume_path is not None)

        # Build model if not provided
        # if it was provided it means it was already loaded
        if not self.model:
            if is_on_master():
                print_boxed('MODEL', print_fun=logger.debug)
            # update model definition with metadata properties
            update_model_definition_with_metadata(
                self.model_definition,
                training_set_metadata
            )
            self.model = LudwigModel.create_model(self.model_definition)

        # init trainer
        trainer = Trainer(
            **self.model_definition[TRAINING],
            debug=debug
        )

        contrib_command("train_model", self.model, self.model_definition,
                        self.model_definition_fp)

        # train model
        if is_on_master():
            print_boxed('TRAINING')
            if not skip_save_model:
                self.save_model_definition(model_dir)

        train_stats = trainer.train(
            self.model,
            training_set,
            validation_set=validation_set,
            test_set=test_set,
            save_path=model_dir,
        )

        train_trainset_stats, train_valiset_stats, train_testset_stats = train_stats
        train_stats = {
            TRAINING: train_trainset_stats,
            VALIDATION: train_valiset_stats,
            TEST: train_testset_stats
        }

        # save training statistics
        if is_on_master():
            if not skip_save_training_statistics:
                save_json(training_stats_fn, train_stats)

        # grab the results of the model with highest validation test performance
        validation_field = self.model_definition[TRAINING]['validation_field']
        validation_metric = self.model_definition[TRAINING][
            'validation_metric']
        validation_field_result = train_valiset_stats[validation_field]

        best_function = get_best_function(validation_metric)
        # results of the model with highest validation test performance
        if is_on_master() and validation_set is not None:
            epoch_best_vali_metric, best_vali_metric = best_function(
                enumerate(validation_field_result[validation_metric]),
                key=lambda pair: pair[1]
            )
            logger.info(
                'Best validation model epoch: {0}'.format(
                    epoch_best_vali_metric + 1)
            )
            logger.info(
                'Best validation model {0} on validation set {1}: {2}'.format(
                    validation_metric, validation_field, best_vali_metric
                ))
            if test_set is not None:
                best_vali_metric_epoch_test_metric = train_testset_stats[
                    validation_field][validation_metric][
                    epoch_best_vali_metric]

                logger.info(
                    'Best validation model {0} on test set {1}: {2}'.format(
                        validation_metric,
                        validation_field,
                        best_vali_metric_epoch_test_metric
                    )
                )
            logger.info(
                '\nFinished: {0}_{1}'.format(experiment_name, model_name))
            logger.info('Saved to: {0}'.format(experiment_dir_name))

        contrib_command("train_save", experiment_dir_name)

        self.training_set_metadata = training_set_metadata
        self.exp_dir_name = experiment_dir_name

        return train_stats, preprocessed_data

    # todo refactoring: reintroduce the train_online functionality?
    def train_online(self):
        pass

    # def predict_pseudo(self, data):
    #     preproc_data = preprocess_data(data)
    #     preds = self.model.batch_predict(preproc_data)
    #     postproc_preds = postprocess_data(preds)
    #     return postproc_preds

    def predict(
            self,
            dataset=None,
            data_format=None,
            batch_size=128,
            skip_save_unprocessed_output=True,
            skip_save_predictions=True,
            output_directory='results',
            return_type=pd.DataFrame,
            debug=False,
            **kwargs
    ):
        self._check_initialization()
        if is_on_master() and not self.exp_dir_name:
            # setup directories and file names
            self.exp_dir_name = find_non_existing_dir_by_adding_suffix(
                output_directory)

        logger.debug('Preprocessing')
        # Added [:] to next line, before I was just assigning,
        # this way I'm copying the list. If you don't do it, you are actually
        # modifying the input feature list when you add output features,
        # which you definitely don't want to do
        features_to_load = self.model_definition['input_features'][:]

        # todo refactoring: this is needed for image features as we expect all
        #  inputs to predict to be in memory, but doublecheck
        num_overrides = override_in_memory_flag(
            self.model_definition['input_features'],
            True
        )
        if num_overrides > 0:
            logger.warning(
                'Using in_memory = False is not supported for Ludwig API.'
            )

        # preprocessing
        dataset, training_set_metadata = preprocess_for_prediction(
            self.model_definition,
            dataset=dataset,
            data_format=data_format,
            training_set_metadata=self.training_set_metadata,
            include_outputs=False,
        )

        logger.debug('Predicting')
        predictor = Predictor(
            batch_size=batch_size, horovod=self._horovod, debug=debug
        )
        predictions = predictor.batch_predict(
            self.model,
            dataset,
        )

        logger.debug('Postprocessing')
        postproc_predictions = convert_predictions(
            postprocess(
                predictions,
                self.model.output_features,
                self.training_set_metadata,
                experiment_dir_name=self.exp_dir_name,
                skip_save_unprocessed_output=skip_save_unprocessed_output
                                             or not is_on_master(),
            ),
            self.model.output_features,
            self.training_set_metadata,
            return_type=return_type
        )

        if is_on_master():
            # if we are skipping all saving,
            # there is no need to create a directory that will remain empty
            should_create_exp_dir = not (
                    skip_save_unprocessed_output and skip_save_predictions
            )
            if should_create_exp_dir:
                os.makedirs(self.exp_dir_name, exist_ok=True)

            if not skip_save_predictions:
                save_prediction_outputs(postproc_predictions,
                                        self.exp_dir_name)

                logger.info('Saved to: {0}'.format(self.exp_dir_name))

        return postproc_predictions

    # def evaluate_pseudo(self, data, return_preds=False):
    #     preproc_data = preprocess_data(data)
    #     if return_preds:
    #         eval_stats, preds = self.model.batch_evaluate(
    #             preproc_data, return_preds=return_preds
    #         )
    #         postproc_preds = postprocess_data(preds)
    #         return eval_stats, postproc_preds
    #     else:
    #         eval_stats = self.model.batch_evaluate(
    #             preproc_data, return_preds=return_preds
    #         )
    #         return eval_stats

    def evaluate(
            self,
            dataset=None,
            data_format=None,
            batch_size=128,
            skip_save_unprocessed_output=True,
            skip_save_predictions=True,
            skip_save_eval_stats=True,
            collect_predictions=False,
            collect_overall_stats=False,
            output_directory='results',
            return_type=pd.DataFrame,
            debug=False,
            **kwargs
    ):
        self._check_initialization()
        if is_on_master() and not self.exp_dir_name:
            # setup directories and file names
            self.exp_dir_name = find_non_existing_dir_by_adding_suffix(
                output_directory)

        logger.debug('Preprocessing')
        # Added [:] to next line, before I was just assigning,
        # this way I'm copying the list. If you don't do it, you are actually
        # modifying the input feature list when you add output features,
        # which you definitely don't want to do
        features_to_load = self.model_definition['input_features'] + \
                           self.model_definition['output_features']

        num_overrides = override_in_memory_flag(
            self.model_definition['input_features'],
            True
        )
        if num_overrides > 0:
            logger.warning(
                'Using in_memory = False is not supported for Ludwig API.'
            )

        # preprocessing
        # todo refactoring: maybe replace the self.model_definition paramter
        #  here with features_to_load
        dataset, training_set_metadata = preprocess_for_prediction(
            self.model_definition,
            dataset=dataset,
            data_format=data_format,
            training_set_metadata=self.training_set_metadata,
            include_outputs=True,
        )

        logger.debug('Predicting')
        predictor = Predictor(
            batch_size=batch_size, horovod=self._horovod, debug=debug
        )
        stats, predictions = predictor.batch_evaluation(
            self.model,
            dataset,
            collect_predictions=collect_predictions or collect_overall_stats,
        )

        # calculate the overall metrics
        if collect_overall_stats:
            overall_stats = calculate_overall_stats(
                self.model.output_features,
                predictions,
                dataset,
                training_set_metadata
            )
            stats = {of_name: {**stats[of_name], **overall_stats[of_name]}
            # account for presence of 'combined' key
            if of_name in overall_stats else {**stats[of_name]}
                     for of_name in stats}

        if collect_predictions:
            logger.debug('Postprocessing')
            print('predictions: {} {}'.format(type(predictions), predictions))
            postproc_predictions = postprocess(
                predictions,
                self.model.output_features,
                self.training_set_metadata,
                experiment_dir_name=self.exp_dir_name,
                skip_save_unprocessed_output=skip_save_unprocessed_output
                                             or not is_on_master(),
            )
        else:
            postproc_predictions = predictions  # = {}

        if is_on_master():
            # if we are skipping all saving,
            # there is no need to create a directory that will remain empty
            should_create_exp_dir = not (
                    skip_save_unprocessed_output and
                    skip_save_predictions and
                    skip_save_eval_stats
            )
            if should_create_exp_dir:
                os.makedirs(self.exp_dir_name, exist_ok=True)

            if postproc_predictions is not None and not skip_save_predictions:
                save_prediction_outputs(postproc_predictions,
                                        self.exp_dir_name)

            print_evaluation_stats(stats)
            if not skip_save_eval_stats:
                save_evaluation_stats(stats, self.exp_dir_name)

            if not skip_save_predictions or not skip_save_eval_stats:
                logger.info('Saved to: {0}'.format(self.exp_dir_name))

        if collect_predictions:
            postproc_predictions = convert_predictions(
                postproc_predictions,
                self.model.output_features,
                self.training_set_metadata,
                return_type=return_type)

        return stats, postproc_predictions

    def experiment(
            self,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            data_format=None,
            experiment_name='experiment',
            model_name='run',
            model_load_path=None,
            model_resume_path=None,
            skip_save_training_description=False,
            skip_save_training_statistics=False,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            skip_save_unprocessed_output=False,  # skipcq: PYL-W0613
            skip_save_predictions=False,
            skip_save_eval_stats=False,
            skip_collect_predictions=False,
            skip_collect_overall_stats=False,
            output_directory='results',
            gpus=None,
            gpu_memory_limit=None,
            allow_parallel_threads=True,
            use_horovod=None,
            random_seed=default_random_seed,
            debug=False,
            **kwargs
    ):
        (
            train_stats,
            preprocessed_data
        ) = self.train(
            dataset=dataset,
            training_set=training_set,
            validation_set=validation_set,
            test_set=test_set,
            training_set_metadata=training_set_metadata,
            data_format=data_format,
            experiment_name=experiment_name,
            model_name=model_name,
            model_load_path=model_load_path,
            model_resume_path=model_resume_path,
            skip_save_training_description=skip_save_training_description,
            skip_save_training_statistics=skip_save_training_statistics,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            skip_save_processed_input=skip_save_processed_input,
            output_directory=output_directory,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
            use_horovod=use_horovod,
            random_seed=random_seed,
            debug=debug,
        )

        (_,  # training_set
         _,  # validation_set
         test_set,
         training_set_metadata) = preprocessed_data

        if test_set is not None:
            if self.model_definition[TRAINING]['eval_batch_size'] > 0:
                batch_size = self.model_definition[TRAINING]['eval_batch_size']
            else:
                batch_size = self.model_definition[TRAINING]['batch_size']

            # todo tf2 refactor: figure out where this goes given NewLudwigModel
            # if a model was saved on disk, reload it
            # model_dir = os.path.join(self.exp_dir_name, 'model')
            # if is_model_dir(model_dir):
            #     model = NewLudwigModel.load(model_dir,
            #                          use_horovod=use_horovod,
            #                          gpus=gpus,
            #                          gpu_memory_limit=gpu_memory_limit,
            #                          allow_parallel_threads=allow_parallel_threads)

            # predict
            test_results = self.evaluate(
                test_set,
                data_format=data_format,
                batch_size=batch_size,
                skip_save_predictions=skip_save_predictions,
                skip_save_eval_stats=skip_save_eval_stats,
                collect_predictions=not skip_collect_predictions,
                collect_overall_stats=not skip_collect_overall_stats,
                debug=debug
            )
        else:
            test_results = None

        return (
            test_results,
            train_stats,
            preprocessed_data
        )

    def collect_weights(
            self,
            tensor_names=None,
            **kwargs
    ):
        self._check_initialization()
        collected_tensors = self.model.collect_weights(tensor_names)
        return collected_tensors

    def collect_activations(
            self,
            layer_names,
            dataset,
            data_format=None,
            batch_size=128,
            # output_directory='results',
            debug=False,
            **kwargs
    ):
        self._check_initialization()
        logger.debug('Preprocessing')
        # Added [:] to next line, before I was just assigning,
        # this way I'm copying the list. If you don't do it, you are actually
        # modifying the input feature list when you add output features,
        # which you definitely don't want to do
        features_to_load = self.model_definition['input_features'][:]

        # todo refactoring: this is needed for image features as we expect all
        #  inputs to predict to be in memory, but doublecheck
        num_overrides = override_in_memory_flag(
            self.model_definition['input_features'],
            True
        )
        if num_overrides > 0:
            logger.warning(
                'Using in_memory = False is not supported for Ludwig API.'
            )

        # preprocessing
        dataset, training_set_metadata = preprocess_for_prediction(
            self.model_definition,
            dataset=dataset,
            data_format=data_format,
            training_set_metadata=self.training_set_metadata,
            # todo refactoring: we shouldn't need to include the outputs
            #  we want to collect activations also using loaded models
            #  on unseen test sets
            include_outputs=True,
        )

        logger.debug('Predicting')
        predictor = Predictor(
            batch_size=batch_size, horovod=self._horovod, debug=debug
        )
        activations = predictor.batch_collect_activations(
            self.model,
            layer_names,
            dataset,
        )

        return activations

    @staticmethod
    def load(model_dir,
             logging_level=logging.ERROR,
             use_horovod=None,
             gpus=None,
             gpu_memory_limit=None,
             allow_parallel_threads=True):
        """This function allows for loading pretrained models


        # Inputs

        :param model_dir: (string) path to the directory containing the model.
               If the model was trained by the `train` or `experiment` command,
               the model is in `results_dir/experiment_dir/model`.
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_memory_limit: (int: default: `None`) maximum memory in MB to allocate
              per GPU device.
        :param allow_parallel_threads: (bool, default: `True`) allow TensorFlow to use
               multithreading parallelism to improve performance at the cost of
               determinism.

        # Return

        :return: (LudwigModel) a LudwigModel object


        # Example usage

        ```python
        ludwig_model = LudwigModel.load(model_dir)
        ```

        """
        # load model definition
        model_definition = load_json(
            os.path.join(
                model_dir,
                MODEL_HYPERPARAMETERS_FILE_NAME
            )
        )

        # initialize model
        ludwig_model = LudwigModel(
            model_definition,
            logging_level=logging_level,
            use_horovod=use_horovod,
            gpus=gpus,
            gpu_memory_limit=gpu_memory_limit,
            allow_parallel_threads=allow_parallel_threads,
        )

        # generate model from definition
        ludwig_model.model = LudwigModel.create_model(model_definition)

        # load model weights
        weights_save_path = os.path.join(
            model_dir,
            MODEL_WEIGHTS_FILE_NAME
        )
        ludwig_model.model.load_weights(weights_save_path)

        # load train set metadata
        ludwig_model.training_set_metadata = load_metadata(
            os.path.join(
                model_dir,
                TRAIN_SET_METADATA_FILE_NAME
            )
        )

        return ludwig_model

    def save(self, save_path):
        """This function allows to save models on disk

        # Inputs

        :param  save_path: (string) path to the directory where the model is
                going to be saved. Both a JSON file containing the model
                architecture hyperparameters and checkpoints files containing
                model weights will be saved.


        # Example usage

        ```python
        ludwig_model.save(save_path)
        ```

        """
        self._check_initialization()

        # save model definition
        self.save_model_definition(save_path)

        # save model weights
        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)
        self.model.save_weights(model_weights_path)

        # save training set metadata
        training_set_metadata_path = os.path.join(
            save_path,
            TRAIN_SET_METADATA_FILE_NAME
        )
        save_json(training_set_metadata_path, self.training_set_metadata)

    def save_model_definition(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        model_hyperparameters_path = os.path.join(
            save_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )
        save_json(model_hyperparameters_path, self.model_definition)

    def save_for_serving(self, save_path):
        """This function allows to save models on disk

        # Inputs

        :param  save_path: (string) path to the directory where the SavedModel
                is going to be saved.


        # Example usage

        ```python
        ludwig_model.save_for_serving(save_path)
        ```

        """
        self._check_initialization()
        self.model.save_savedmodel(save_path)

    def _check_initialization(self):
        if self.model is None or \
                self.model_definition is None or \
                self.training_set_metadata is None:
            raise ValueError('Model has not been trained or loaded')

    @staticmethod
    def create_model(model_definition):
        # TODO: support loading other model types based on definition
        return ECD(
            input_features_def=model_definition['input_features'],
            combiner_def=model_definition['combiner'],
            output_features_def=model_definition['output_features'],
        )

    @staticmethod
    def set_logging_level(logging_level):
        """
        :param logging_level: Set/Update the logging level. Use logging
        constants like `logging.DEBUG` , `logging.INFO` and `logging.ERROR`.

        :return: None
        """
        logging.getLogger('ludwig').setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)
        else:
            set_disable_progressbar(False)


def kfold_cross_validate(
        num_folds,
        model_definition=None,
        model_definition_file=None,
        data_csv=None,
        skip_save_training_description=False,
        skip_save_training_statistics=False,
        skip_save_model=False,
        skip_save_progress=False,
        skip_save_log=False,
        skip_save_processed_input=False,
        skip_save_predictions=False,
        skip_save_eval_stats=False,
        skip_collect_predictions=False,
        skip_collect_overall_stats=False,
        output_directory='results',
        random_seed=default_random_seed,
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        use_horovod=None,
        logging_level=logging.INFO,
        debug=False,
        **kwargs
):
    """Performs k-fold cross validation and returns result data structures.

    # Inputs

    :param num_folds: (int) number of folds to create for the cross-validation
    :param model_definition: (dict, default: None) a dictionary containing
           information needed to build a model. Refer to the
           [User Guide](http://ludwig.ai/user_guide/#model-definition)
           for details.
    :param model_definition_file: (string, optional, default: `None`) path to
           a YAML file containing the model definition. If available it will be
           used instead of the model_definition dict.
    :param data_csv: (dataframe, default: None)
    :param data_csv: (string, default: None)
    :param output_directory: (string, default: 'results')
    :param random_seed: (int) Random seed used k-fold splits.

    # Return

    :return: (tuple(kfold_cv_stats, kfold_split_indices), dict) a tuple of
            dictionaries `kfold_cv_stats`: contains metrics from cv run.
             `kfold_split_indices`: indices to split training data into
             training fold and test fold.
    """
    set_on_master(use_horovod)

    # check for k_fold
    if num_folds is None:
        raise ValueError(
            'k_fold parameter must be specified'
        )

    # check for model_definition and model_definition_file
    if model_definition is None and model_definition_file is None:
        raise ValueError(
            'Either model_definition of model_definition_file have to be'
            'not None to initialize a LudwigModel'
        )
    if model_definition is not None and model_definition_file is not None:
        raise ValueError(
            'Only one between model_definition and '
            'model_definition_file can be provided'
        )

    logger.info('starting {:d}-fold cross validation'.format(num_folds))

    # create output_directory if not available
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # read in data to split for the folds
    data_df = pd.read_csv(data_csv)

    # place each fold in a separate directory
    data_dir = os.path.dirname(data_csv)

    kfold_cv_stats = {}
    kfold_split_indices = {}

    for train_indices, test_indices, fold_num in \
            generate_kfold_splits(data_df, num_folds, random_seed):
        with tempfile.TemporaryDirectory(dir=data_dir) as temp_dir_name:
            curr_train_df = data_df.iloc[train_indices]
            curr_test_df = data_df.iloc[test_indices]

            kfold_split_indices['fold_' + str(fold_num)] = {
                'training_indices': train_indices,
                'test_indices': test_indices
            }

            # train and validate model on this fold
            logger.info("training on fold {:d}".format(fold_num))

            model = LudwigModel(
                    model_definition=model_definition,
                    model_definition_fp=model_definition_file,
                    logging_level=logging_level,
                    use_horovod=use_horovod,
                    gpus=gpus,
                    gpu_memory_limit=gpu_memory_limit,
                    allow_parallel_threads=allow_parallel_threads,
                    random_seed=random_seed
                )
            (
                test_results,
                train_stats,
                preprocessed_data
            ) = model.experiment(
                training_set=curr_train_df,
                test_set=curr_test_df,
                experiment_name='cross_validation',
                model_name='fold_' + str(fold_num),
                skip_save_training_description=skip_save_training_description,
                skip_save_training_statistics=skip_save_training_statistics,
                skip_save_model=skip_save_model,
                skip_save_progress=skip_save_progress,
                skip_save_log=skip_save_log,
                skip_save_processed_input=skip_save_processed_input,
                skip_save_predictions=skip_save_predictions,
                skip_save_eval_stats=skip_save_eval_stats,
                skip_collect_predictions=skip_collect_predictions,
                skip_collect_overall_stats=skip_collect_overall_stats,
                output_directory=os.path.join(temp_dir_name, 'results'),
                random_seed=random_seed,
                debug=debug,
            )

            # todo if we want to save the csv of predictions uncomment block
            # if is_on_master():
            #     print_test_results(test_results)
            #     if not skip_save_predictions:
            #         save_prediction_outputs(
            #             postprocessed_output,
            #             experiment_dir_name
            #         )
            #     if not skip_save_eval_stats:
            #         save_test_statistics(test_results, experiment_dir_name)

            # augment the training statistics with scoring metric from
            # the hold out fold
            train_stats['fold_test_results'] = test_results[0]

            # collect training statistics for this fold
            kfold_cv_stats['fold_' + str(fold_num)] = train_stats

    # consolidate raw fold metrics across all folds
    raw_kfold_stats = {}
    for fold_name in kfold_cv_stats:
        curr_fold_test_results = kfold_cv_stats[fold_name]['fold_test_results']
        for of_name in curr_fold_test_results:
            if of_name not in raw_kfold_stats:
                raw_kfold_stats[of_name] = {}
            fold_test_results_of = curr_fold_test_results[of_name]

            for metric in fold_test_results_of:
                if metric not in {
                    'predictions',
                    'probabilities',
                    'confusion_matrix',
                    'overall_stats',
                    'per_class_stats',
                    'roc_curve',
                    'precision_recall_curve'
                }:
                    if metric not in raw_kfold_stats[of_name]:
                        raw_kfold_stats[of_name][metric] = []
                    raw_kfold_stats[of_name][metric].append(
                        fold_test_results_of[metric]
                    )

    # calculate overall kfold statistics
    overall_kfold_stats = {}
    for of_name in raw_kfold_stats:
        overall_kfold_stats[of_name] = {}
        for metric in raw_kfold_stats[of_name]:
            mean = np.mean(raw_kfold_stats[of_name][metric])
            std = np.std(raw_kfold_stats[of_name][metric])
            overall_kfold_stats[of_name][metric + '_mean'] = mean
            overall_kfold_stats[of_name][metric + '_std'] = std

    kfold_cv_stats['overall'] = overall_kfold_stats

    logger.info('completed {:d}-fold cross validation'.format(num_folds))

    return kfold_cv_stats, kfold_split_indices

# todo refactoring: this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def test_train(
        data_csv,
        model_definition,
        batch_size=128,
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        debug=False,
        logging_level=logging.ERROR,
        **kwargs
):
    model = LudwigModel(
        model_definition=model_definition,
        logging_level=logging_level,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
    )

    train_stats, _ = model.train(
        dataset=data_csv,
        debug=debug,
    )

    logger.critical(train_stats)

    # predict
    predictions = model.predict(
        dataset=data_csv,
        batch_size=batch_size
    )

    logger.critical(predictions)


# todo refactoring: this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def test_train_online(
        data_csv,
        model_definition,
        batch_size=128,
        debug=False,
        logging_level=logging.ERROR,
        **kwargs
):
    # todo refactoring
    # model_definition = merge_with_defaults(model_definition)
    # data, train_set_metadata = build_dataset(
    #     data_csv,
    #     (model_definition['input_features'] +
    #      model_definition['output_features']),
    #     model_definition['preprocessing']
    # )
    #
    # ludwig_model = NewLudwigModel(model_definition, logging_level=logging_level)
    # ludwig_model.initialize_model(train_set_metadata=train_set_metadata)
    #
    # ludwig_model.train_online(
    #     data_csv=data_csv,
    #     batch_size=128,
    # )
    # ludwig_model.train_online(
    #     data_csv=data_csv,
    #     batch_size=128,
    # )
    #
    # # predict
    # predictions = ludwig_model.predict(
    #     data_csv=data_csv,
    #     batch_size=batch_size,
    # )
    # ludwig_model.close()
    # logger.critical(predictions)
    pass


# todo refactoring: this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def test_predict(
        data_csv,
        model_path,
        batch_size=128,
        gpus=None,
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        logging_level=logging.ERROR,
        **kwargs
):
    ludwig_model = LudwigModel.load(
        model_path,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        logging_level=logging_level)

    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
    )

    ludwig_model.close()
    logger.critical(predictions)

    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
    )

    logger.critical(predictions)


# todo(refactoring): this shouldn't exist,
#  all api tests should be done in a proper integration test,
#  move there if needed
def main(sys_argv):
    parser = argparse.ArgumentParser(
        description='This script tests ludwig APIs.'
    )

    parser.add_argument(
        '-t',
        '--test',
        default='train',
        choices=['train', 'train_online', 'predict'],
        help='which test to run'
    )

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument('--data_csv', help='input data CSV file')
    parser.add_argument(
        '--train_set_metadata_json',
        help='input metadata JSON file'
    )

    # ----------------
    # Model parameters
    # ----------------
    parser.add_argument('-m', '--model_path', help='model to load')
    parser.add_argument(
        '-md',
        '--model_definition',
        type=yaml.safe_load,
        help='model definition'
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
        default=None,
        help='list of gpu to use'
    )
    parser.add_argument(
        '-gml',
        '--gpu_memory_limit',
        type=int,
        default=None,
        help='maximum memory in MB to allocate per GPU device'
    )
    parser.add_argument(
        '-dpt',
        '--disable_parallel_threads',
        action='store_false',
        dest='allow_parallel_threads',
        help='disable TensorFlow from using multithreading for reproducibility'
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
    args.logging_level = logging_level_registry[args.logging_level]

    if args.test == 'train':
        test_train(**vars(args))
    elif args.test == 'train_online':
        test_train_online(**vars(args))
    elif args.test == 'predict':
        test_predict(**vars(args))
    else:
        logger.info('Unsupported test type')


if __name__ == '__main__':
    main(sys.argv[1:])


