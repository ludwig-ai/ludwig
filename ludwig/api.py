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
import logging
import os
import sys

import ludwig.contrib

ludwig.contrib.contrib_import()

import pandas as pd
import yaml

from ludwig.data.dataset import Dataset
from ludwig.data.postprocessing import postprocess_df, postprocess
from ludwig.data.preprocessing import build_data
from ludwig.data.preprocessing import build_dataset
from ludwig.data.preprocessing import load_metadata
from ludwig.data.preprocessing import replace_text_feature_level
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.globals import MODEL_WEIGHTS_FILE_NAME
from ludwig.globals import TRAIN_SET_METADATA_FILE_NAME
from ludwig.globals import set_disable_progressbar
from ludwig.models.model import Model
from ludwig.models.model import load_model_and_definition
from ludwig.predict import calculate_overall_stats
from ludwig.train import full_train
from ludwig.train import update_model_definition_with_metadata
from ludwig.utils.data_utils import read_csv
from ludwig.utils.data_utils import save_json
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.defaults import merge_with_defaults
from ludwig.utils.print_utils import logging_level_registry


class LudwigModel:
    """Class that allows access to high level Ludwig functionalities.

    # Inputs

    :param model_definition: (dict) a dictionary containing information needed
           to build a model. Refer to the [User Guide]
           (http://ludwig.ai/user_guide/#model-definition) for details.
    :param model_definition_file: (string, optional, default: `None`) path to
           a YAML file containing the model definition. If available it will be
           used instead of the model_definition dict.
    :param logging_level: (int, default: `logging.ERROR`) logging level to use
           for logging. Use logging constants like `logging.DEBUG`,
           `logging.INFO` and `logging.ERROR`. By default only errors will be
           printed.


    # Example usage:

    ```python
    from ludwig.api import LudwigModel
    ```

    Train a model:

    ```python
    model_definition = {...}
    ludwig_model = LudwigModel(model_definition)
    train_stats = ludwig_model.train(data_csv=csv_file_path)
    ```

    or

    ```python
    train_stats = ludwig_model.train(data_df=dataframe)
    ```

    If you have already trained a model you can load it and use it to predict

    ```python
    ludwig_model = LudwigModel.load(model_dir)
    ```

    Predict:

    ```python
    predictions = ludwig_model.predict(data_csv=csv_file_path)
    ```

    or

    ```python
    predictions = ludwig_model.predict(data_df=dataframe)
    ```

    Finally in order to release resources:

    ```python
    model.close()
    ```
    """

    def __init__(
            self,
            model_definition,
            model_definition_file=None,
            logging_level=logging.ERROR
    ):
        logging.getLogger().setLevel(logging_level)
        if model_definition_file is not None:
            with open(model_definition_file, 'r') as def_file:
                self.model_definition = merge_with_defaults(
                    yaml.safe_load(def_file)
                )
        else:
            self.model_definition = merge_with_defaults(model_definition)
        self.train_set_metadata = None
        self.model = None

    @staticmethod
    def _read_data(data_csv, data_dict):
        """
        :param data_csv: path to the csv data
        :param data_dict: raw data
        :return: pandas dataframe with the data
        """
        if data_csv is not None:
            data_df = read_csv(data_csv)
        elif data_dict is not None:
            data_df = pd.DataFrame(data_dict)
        else:
            raise ValueError(
                'No input data specified. '
                'One of data_df, data_csv or data_dict must be provided'
            )

        return data_df

    @staticmethod
    def load(model_dir, logging_level=logging.ERROR):
        """This function allows for loading pretrained models


        # Inputs

        :param model_dir: (string) path to the directory containing the model.
               If the model was trained by the `train` or `experiment` command,
               the model is in `results_dir/experiment_dir/model`.
        :param logging_level: (int, default: `logging.ERROR`) logging level to
               use for logging. Use logging constants like `logging.DEBUG`,
               `logging.INFO` and `logging.ERROR`. By default only errors will
               be printed.


        # Return

        :return: (LudwigModel) a LudwigModel object


        # Example usage

        ```python
        ludwig_model = LudwigModel.load(model_dir)
        ```

        """

        logging.getLogger().setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)

        model, model_definition = load_model_and_definition(model_dir)
        ludwig_model = LudwigModel(model_definition)
        ludwig_model.model = model
        ludwig_model.train_set_metadata = load_metadata(
            os.path.join(
                model_dir,
                TRAIN_SET_METADATA_FILE_NAME
            )
        )
        return ludwig_model

    def save(self, save_path):
        """This function allows for loading pretrained models

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
        if (self.model is None or self.model.session is None or
                self.model_definition is None or self.train_set_metadata is None):
            raise ValueError('Model has not been initialized or loaded')

        model_weights_path = os.path.join(save_path, MODEL_WEIGHTS_FILE_NAME)

        model_hyperparameters_path = os.path.join(
            save_path,
            MODEL_HYPERPARAMETERS_FILE_NAME
        )

        self.model.save_weights(self.model.session, model_weights_path)

        train_set_metadata_path = os.path.join(
            save_path,
            TRAIN_SET_METADATA_FILE_NAME
        )
        save_json(train_set_metadata_path, self.train_set_metadata)

        self.model.save_hyperparameters(
            self.model.hyperparameters,
            model_hyperparameters_path
        )

    def close(self):
        """Closes an open LudwigModel (closing the session running it).
        It should be called once done with the model to release resources.
        """
        if self.model is not None:
            self.model.close_session()

    def train(
            self,
            data_df=None,
            data_train_df=None,
            data_validation_df=None,
            data_test_df=None,
            data_csv=None,
            data_train_csv=None,
            data_validation_csv=None,
            data_test_csv=None,
            data_hdf5=None,
            data_train_hdf5=None,
            data_validation_hdf5=None,
            data_test_hdf5=None,
            data_dict=None,
            train_set_metadata_json=None,
            model_name='run',
            model_load_path=None,
            model_resume_path=None,
            skip_save_model=False,
            skip_save_progress=False,
            skip_save_log=False,
            skip_save_processed_input=False,
            output_directory='results',
            gpus=None,
            gpu_fraction=1.0,
            use_horovod=False,
            random_seed=42,
            logging_level=logging.ERROR,
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
               ``{'text_field_name': ['text of the first datapoint', text of the
               second datapoint'], 'class_filed_name': ['class_datapoints_1',
               'class_datapoints_2']}`.
        :param train_set_metadata_json: (string) input metadata JSON file. It is an
               intermediate preprocess file containing the mappings of the input
               CSV created the first time a CSV file is used in the same
               directory with the same name and a json extension
        :param model_name: (string) a name for the model, user for the save
               directory
        :param model_load_path: (string) path of a pretrained model to load as
               initialization
        :param model_resume_path: (string) path of a the model directory to
               resume training of
        :param skip_save_model: (bool, default: `False`) disables
               saving model weights and hyperparameters each time the model
               improves. By default Ludwig saves model weights after each epoch
               the validation measure imrpvoes, but if the model is really big
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
        :param gpu_fraction: (float, default `1.0`) fraction of gpu memory to
               initialize the process with
        :param random_seed: (int, default`42`) a random seed that is going to be
               used anywhere there is a call to a random number generator: data
               splitting, parameter initialization and training set shuffling
        :param debug: (bool, default: `False`) enables debugging mode
        :param logging_level: (int, default: `logging.ERROR`) logging level to
               use for logging. Use logging constants like `logging.DEBUG`,
               `logging.INFO` and `logging.ERROR`. By default only errors will
               be printed.

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
        output feature containing loss and measures values for each epoch.

        """
        logging.getLogger().setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)

        if data_df is None and data_dict is not None:
            data_df = pd.DataFrame(data_dict)

        (
            self.model,
            preprocessed_data,
            _,
            train_stats,
            self.model_definition
        ) = full_train(
            self.model_definition,
            data_df=data_df,
            data_train_df=data_train_df,
            data_validation_df=data_validation_df,
            data_test_df=data_test_df,
            data_csv=data_csv,
            data_train_csv=data_train_csv,
            data_validation_csv=data_validation_csv,
            data_test_csv=data_test_csv,
            data_hdf5=data_hdf5,
            data_train_hdf5=data_train_hdf5,
            data_validation_hdf5=data_validation_hdf5,
            data_test_hdf5=data_test_hdf5,
            train_set_metadata_json=train_set_metadata_json,
            experiment_name='api_experiment',
            model_name=model_name,
            model_load_path=model_load_path,
            model_resume_path=model_resume_path,
            skip_save_model=skip_save_model,
            skip_save_progress=skip_save_progress,
            skip_save_log=skip_save_log,
            skip_save_processed_input=skip_save_processed_input,
            output_directory=output_directory,
            gpus=gpus,
            gpu_fraction=gpu_fraction,
            use_horovod=use_horovod,
            random_seed=random_seed,
            debug=debug,
        )

        self.train_set_metadata = preprocessed_data[-1]

        return train_stats

    def initialize_model(
            self,
            train_set_metadata=None,
            train_set_metadata_json=None,
            gpus=None,
            gpu_fraction=1,
            random_seed=default_random_seed,
            logging_level=logging.ERROR,
            debug=False,
            **kwargs
    ):
        """This function initializes a model. It is need for performing online
        learning, so it has to be called before `train_online`.
        `train` initialize the model under the hood, so there is no need to call
        this function if you don't use `train_online`.

        # Inputs

        :param train_set_metadata: (dict) it contains metadata information for
               the input and output features the model is going to be trained
               on. It's the same content of the metadata json file that is
               created while training.
        :param train_set_metadata_json: (string)  path to the JSON metadata file
               created while training. it contains metadata information for the
               input and output features the model is going to be trained on
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_fraction: (float, default `1.0`) fraction of GPU memory to
               initialize the process with
        :param random_seed: (int, default`42`) a random seed that is going to be
               used anywhere there is a call to a random number generator: data
               splitting, parameter initialization and training set shuffling
        :param logging_level: (int, default: `logging.ERROR`) logging level to
               use for logging. Use logging constants like `logging.DEBUG`,
               `logging.INFO` and `logging.ERROR`. By default only errors will
               be printed.
        :param debug: (bool, default: `False`) enables debugging mode
        """
        logging.getLogger().setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)

        if train_set_metadata is None and train_set_metadata_json is None:
            raise ValueError(
                'train_set_metadata or train_set_metadata_json must not None.'
            )
        if train_set_metadata_json is not None:
            train_set_metadata = load_metadata(train_set_metadata_json)

        # update model definition with metadata properties
        update_model_definition_with_metadata(
            self.model_definition,
            train_set_metadata)

        # build model
        model = Model(
            self.model_definition['input_features'],
            self.model_definition['output_features'],
            self.model_definition['combiner'],
            self.model_definition['training'],
            self.model_definition['preprocessing'],
            random_seed=random_seed,
            debug=debug
        )
        model.initialize_session(gpus=gpus, gpu_fraction=gpu_fraction)

        # set parameters
        self.model = model
        self.train_set_metadata = train_set_metadata

    def train_online(
            self,
            data_df=None,
            data_csv=None,
            data_dict=None,
            batch_size=None,
            learning_rate=None,
            regularization_lambda=None,
            dropout_rate=None,
            bucketing_field=None,
            gpus=None,
            gpu_fraction=1,
            logging_level=logging.ERROR,
    ):
        """This function is used to perform one epoch of training of the model 
        on the specified dataset.

        # Inputs

        :param data_df: (DataFrame) dataframe containing data.
        :param data_csv: (string) input data CSV file.
        :param data_dict: (dict) input data dictionary. It is expected to 
               contain one key for each field and the values have to be lists of 
               the same length. Each index in the lists corresponds to one 
               datapoint. For example a data set consisting of two datapoints 
               with a text and a class may be provided as the following dict 
               ``{'text_field_name': ['text of the first datapoint', text of the
               second datapoint'], 'class_filed_name': ['class_datapoints_1', 
               'class_datapoints_2']}`.
        :param batch_size: (int) the batch size to use for training. By default 
               it's the one specified in the model definition.
        :param learning_rate: (float) the learning rate to use for training. By
               default the values is the one specified in the model definition.
        :param regularization_lambda: (float) the regularization lambda
               parameter to use for training. By default the values is the one
               specified in the model definition.
        :param dropout_rate: (float) the dropout rate to use for training. By
               default the values is the one specified in the model definition.
        :param bucketing_field: (string) the bucketing field to use for
               bucketing the data. By default the values is one specified in the
               model definition.
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_fraction: (float, default `1.0`) fraction of GPU memory to
               initialize the process with
        :param logging_level: (int, default: `logging.ERROR`) logging level to
               use for logging. Use logging constants like `logging.DEBUG`,
               `logging.INFO` and `logging.ERROR`. By default only errors will
               be printed.

        There are three ways to provide data: by dataframes using the `data_df`
        parameter, by CSV using the `data_csv` parameter and by dictionary,
        using the `data_dict` parameter.

        The DataFrame approach uses data previously obtained and put in a
        dataframe, the CSV approach loads data from a CSV file, while dict
        approach uses data organized by keys representing columns and values
        that are lists of the datapoints for each. For example a data set
        consisting of two datapoints with a text and a class may be provided as
        the following dict ``{'text_field_name}: ['text of the first datapoint',
        text of the second datapoint'], 'class_filed_name':
        ['class_datapoints_1', 'class_datapoints_2']}`.
        """
        logging.getLogger().setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)

        if (self.model is None or self.model_definition is None
                or self.train_set_metadata is None):
            raise ValueError('Model has not been initialized or loaded')

        if data_df is None:
            data_df = self._read_data(data_csv, data_dict)
            data_df.csv = data_csv

        if batch_size is None:
            batch_size = self.model_definition['training']['batch_size']
        if learning_rate is None:
            learning_rate = self.model_definition['training']['learning_rate']
        if regularization_lambda is None:
            regularization_lambda = self.model_definition['training'][
                'regularization_lambda'
            ]
        if dropout_rate is None:
            dropout_rate = self.model_definition['training']['dropout_rate'],
        if bucketing_field is None:
            bucketing_field = self.model_definition['training'][
                'bucketing_field'
            ]

        logging.debug('Preprocessing {} datapoints'.format(len(data_df)))
        features_to_load = (self.model_definition['input_features'] +
                            self.model_definition['output_features'])
        preprocessed_data = build_data(
            data_df,
            features_to_load,
            self.train_set_metadata,
            self.model_definition['preprocessing']
        )
        replace_text_feature_level(
            self.model_definition['input_features'] +
            self.model_definition['output_features'],
            [preprocessed_data]
        )
        dataset = Dataset(
            preprocessed_data,
            self.model_definition['input_features'],
            self.model_definition['output_features'],
            None
        )

        logging.debug('Training batch')
        self.model.train_online(
            dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            regularization_lambda=regularization_lambda,
            dropout_rate=dropout_rate,
            bucketing_field=bucketing_field,
            gpus=gpus,
            gpu_fraction=gpu_fraction)

    def _predict(
            self,
            data_df=None,
            data_csv=None,
            data_dict=None,
            return_type=pd.DataFrame,
            batch_size=128,
            gpus=None,
            gpu_fraction=1,
            evaluate_performance=False,
            logging_level=logging.ERROR,
    ):
        logging.getLogger().setLevel(logging_level)
        if logging_level in {logging.WARNING, logging.ERROR, logging.CRITICAL}:
            set_disable_progressbar(True)

        if (self.model is None or self.model_definition is None or
                self.train_set_metadata is None):
            raise ValueError('Model has not been trained or loaded')

        if data_df is None:
            data_df = self._read_data(data_csv, data_dict)

        logging.debug('Preprocessing {} datapoints'.format(len(data_df)))
        features_to_load = self.model_definition['input_features']
        if evaluate_performance:
            output_features = self.model_definition['output_features']
        else:
            output_features = []
        features_to_load += output_features

        preprocessed_data = build_data(
            data_df,
            features_to_load,
            self.train_set_metadata,
            self.model_definition['preprocessing']
        )
        replace_text_feature_level(
            features_to_load,
            [preprocessed_data]
        )
        dataset = Dataset(
            preprocessed_data,
            self.model_definition['input_features'],
            output_features,
            None
        )

        logging.debug('Predicting')
        predict_results = self.model.predict(
            dataset,
            batch_size,
            evaluate_performance=evaluate_performance,
            gpus=gpus, gpu_fraction=gpu_fraction,
            session=getattr(self.model, 'session', None)
        )

        if evaluate_performance:
            calculate_overall_stats(
                predict_results,
                self.model_definition['output_features'],
                dataset,
                self.train_set_metadata
            )

        logging.debug('Postprocessing')
        if (
                return_type == 'dict' or
                return_type == 'dictionary' or
                return_type == dict
        ):
            postprocessed_predictions = postprocess(
                predict_results,
                self.model_definition['output_features'],
                self.train_set_metadata
            )
        elif (
                return_type == 'dataframe' or
                return_type == 'df' or
                return_type == pd.DataFrame
        ):
            postprocessed_predictions = postprocess_df(
                predict_results,
                self.model_definition['output_features'],
                self.train_set_metadata
            )
        else:
            logging.warning(
                'Unrecognized return_type: {}. '
                'Returning DataFrame.'.format(return_type)
            )
            postprocessed_predictions = postprocess(
                predict_results,
                self.model_definition['output_features'],
                self.train_set_metadata
            )

        return postprocessed_predictions, predict_results

    def predict(
            self,
            data_df=None,
            data_csv=None,
            data_dict=None,
            return_type=pd.DataFrame,
            batch_size=128,
            gpus=None,
            gpu_fraction=1,
            logging_level=logging.ERROR,
    ):
        """This function is used to predict the output variables given the input
           variables using the trained model.

        # Inputs

        :param data_df: (DataFrame) dataframe containing data. Only the input
               features defined in the model definition need to be present in
               the dataframe.
        :param data_csv: (string) input data CSV file. Only the input features
               defined in the model definition need to be present in the CSV.
        :param data_dict: (dict) input data dictionary. It is expected to
               contain one key for each field and the values have to be lists
               of the same length. Each index in the lists corresponds to one
               datapoint. Only the input features defined in the model
               definition need to be present in the dataframe. For example a
               data set consisting of two datapoints with a input text may be
               provided as the following dict ``{'text_field_name}: ['text of
               the first datapoint', text of the second datapoint']}`.
        :param return_type: (strng or type, default: `DataFrame`)
               string describing the type of the returned prediction object.
               `'dataframe'`, `'df'` and `DataFrame` will return a pandas
               DataFrame , while `'dict'`, ''dictionary'` and `dict` will
               return a dictionary.
        :param batch_size: (int, default: `128`) batch size
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_fraction: (float, default `1.0`) fraction of gpu memory to
               initialize the process with
        :param logging_level: (int, default: `logging.ERROR`) logging level to
               use for logging. Use logging constants like `logging.DEBUG`,
               `logging.INFO` and `logging.ERROR`. By default only errors will
               be printed.


        # Return

        :return: (DataFrame or dict) a dataframe containing the predictions for
                 each output feature and their probabilities (for types that
                 return them) will be returned. For instance in a 3 way
                 multiclass classification problem with a category field names
                 `class` as output feature with possible values `one`, `two`
                 and `three`, the dataframe will have as many rows as input
                 datapoints and five columns: `class_predictions`,
                 `class_UNK_probability`, `class_one_probability`,
                 `class_two_probability`, `class_three_probability`. (The UNK
                 class is always present in categorical features).
                 If the `return_type` is a dictionary, the returned object be
                 a dictionary contaning one entry for each output feature.
                 Each entry is itself a dictionary containing aligned
                 arrays of predictions and probabilities / scores.
        """
        predictions, _ = self._predict(
            data_df=data_df,
            data_csv=data_csv,
            data_dict=data_dict,
            return_type=return_type,
            batch_size=batch_size,
            gpus=gpus,
            gpu_fraction=gpu_fraction,
            evaluate_performance=False,
            logging_level=logging_level,
        )

        return predictions

    def test(
            self,
            data_df=None,
            data_csv=None,
            data_dict=None,
            return_type=pd.DataFrame,
            batch_size=128,
            gpus=None,
            gpu_fraction=1,
            logging_level=logging.ERROR,
    ):
        """This function is used to predict the output variables given the input
        variables using the trained model and compute test statistics like
        performance measures, confusion matrices and the like.


        # Inputs

        :param data_df: (DataFrame) dataframe containing data. Both input and
               output features defined in the model definition need to be
               present in the dataframe.
        :param data_csv: (string) input data CSV file. Both input and output
               features defined in the model definition need to be present in
               the CSV.
        :param data_dict: (dict) input data dictionary. It is expected to
               contain one key for each field and the values have to be lists
               of the same length. Each index in the lists corresponds to one
               datapoint. Both input and output features defined in the model
               definition need to be present in the dataframe. For example a
               data set consisting of two datapoints with a input text may be
               provided as the following dict ``{'text_field_name}: ['text of
               the first datapoint', text of the second datapoint']}`.
        :param return_type: (strng or type, default: `DataFrame`)
               string describing the type of the returned prediction object.
               `'dataframe'`, `'df'` and `DataFrame` will return a pandas
               DataFrame , while `'dict'`, ''dictionary'` and `dict` will
               return a dictionary.
        :param batch_size: (int, default: `128`) batch size
        :param gpus: (string, default: `None`) list of GPUs to use (it uses the
               same syntax of CUDA_VISIBLE_DEVICES)
        :param gpu_fraction: (float, default `1.0`) fraction of GPU memory to
               initialize the process with
        :param logging_level: (int, default: `logging.ERROR`) logging level to
               use for logging. Use logging constants like `logging.DEBUG`,
               `logging.INFO` and `logging.ERROR`. By default only errors will
               be printed.


        # Return

        :return: (tuple((DataFrame or dict), dict)) a tuple of a dataframe and a
                 dictionary. The dataframe contains the predictions for each
                 output feature and their probabilities (for types that return
                 them) will be returned. For instance in a 3 way multiclass
                 classification problem with a category field names `class` as
                 output feature with possible values `one`, `two` and `three`,
                 the dataframe will have as many rows as input datapoints and
                 five columns: `class_predictions`, `class_UNK_probability`,
                 `class_one_probability`, `class_two_probability`,
                 `class_three_probability`. (The UNK class is always present in
                 categorical features).
                 If the `return_type` is a dictionary, the first object
                 of the tuple will be a dictionary contaning one entry
                 for each output feature.
                 Each entry is itself a dictionary containing aligned
                 arrays of predictions and probabilities / scores.
                 The second object of the tuple is a dictionary that contains
                 the test statistics, with each key being the name of an output
                 feature and the values being dictionaries containing measures
                 names and their values.
        """
        predictions, test_stats = self._predict(
            data_df=data_df,
            data_csv=data_csv,
            data_dict=data_dict,
            return_type=return_type,
            batch_size=batch_size,
            gpus=gpus,
            gpu_fraction=gpu_fraction,
            evaluate_performance=True,
            logging_level=logging_level,
        )

        return predictions, test_stats


def test_train(
        data_csv,
        model_definition,
        batch_size=128,
        gpus=None,
        gpu_fraction=1,
        debug=False,
        logging_level=logging.ERROR,
        **kwargs
):
    ludwig_model = LudwigModel(
        model_definition,
        logging_level=logging_level
    )

    train_stats = ludwig_model.train(
        data_csv=data_csv,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level,
        debug=debug
    )

    logging.critical(train_stats)

    # predict
    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level
    )

    ludwig_model.close()
    logging.critical(predictions)


def test_train_online(
        data_csv,
        model_definition,
        batch_size=128,
        gpus=None,
        gpu_fraction=1,
        debug=False,
        logging_level=logging.ERROR,
        **kwargs
):
    model_definition = merge_with_defaults(model_definition)
    data, train_set_metadata = build_dataset(
        data_csv,
        (model_definition['input_features'] +
         model_definition['output_features']),
        model_definition['preprocessing']
    )

    ludwig_model = LudwigModel(model_definition, logging_level=logging_level)
    ludwig_model.initialize_model(
        train_set_metadata=train_set_metadata,
        logging_level=logging_level
    )

    ludwig_model.train_online(
        data_csv=data_csv,
        batch_size=128,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level
    )
    ludwig_model.train_online(
        data_csv=data_csv,
        batch_size=128,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level
    )

    # predict
    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level
    )
    ludwig_model.close()
    logging.critical(predictions)


def test_predict(
        data_csv,
        model_path,
        batch_size=128,
        gpus=None,
        gpu_fraction=1,
        logging_level=logging.ERROR,
        **kwargs
):
    ludwig_model = LudwigModel.load(
        model_path,
        logging_level=logging_level
    )

    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level
    )

    ludwig_model.close()
    logging.critical(predictions)

    predictions = ludwig_model.predict(
        data_csv=data_csv,
        batch_size=batch_size,
        gpus=gpus,
        gpu_fraction=gpu_fraction,
        logging_level=logging_level
    )

    logging.critical(predictions)


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
        choices=['critical', 'error', 'warning', 'info', 'debug', 'notset']
    )

    args = parser.parse_args(sys_argv)
    args.logging_level = logging_level_registry[args.logging_level]

    logging.basicConfig(
        stream=sys.stdout,
        # filename='log.log',
        # filemode='w',
        level=args.logging_level,
        format='%(message)s'
    )

    if args.test == 'train':
        test_train(**vars(args))
    elif args.test == 'train_online':
        test_train_online(**vars(args))
    elif args.test == 'predict':
        test_predict(**vars(args))
    else:
        logging.info('Unsupported test type')


if __name__ == '__main__':
    main(sys.argv[1:])
