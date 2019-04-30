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
import argparse
import logging
import os

import h5py
import numpy as np
import yaml

from ludwig.constants import *
from ludwig.constants import TEXT
from ludwig.data.concatenate_datasets import concatenate_csv
from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.data.dataset import Dataset
from ludwig.features.feature_registries import base_type_registry
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME
from ludwig.utils import data_utils
from ludwig.utils.data_utils import collapse_rare_labels
from ludwig.utils.data_utils import load_json
from ludwig.utils.data_utils import read_csv
from ludwig.utils.data_utils import replace_file_extension
from ludwig.utils.data_utils import split_dataset_tvt
from ludwig.utils.data_utils import text_feature_data_field
from ludwig.utils.defaults import default_preprocessing_parameters
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.misc import get_from_registry
from ludwig.utils.misc import merge_dict
from ludwig.utils.misc import set_random_seed


def build_dataset(
        dataset_csv,
        features,
        global_preprocessing_parameters,
        train_set_metadata=None,
        random_seed=default_random_seed,
        **kwargs
):
    dataset_df = read_csv(dataset_csv)
    dataset_df.csv = dataset_csv
    return build_dataset_df(
        dataset_df,
        features,
        global_preprocessing_parameters,
        train_set_metadata,
        random_seed,
        **kwargs
    )


def build_dataset_df(
        dataset_df,
        features,
        global_preprocessing_parameters,
        train_set_metadata=None,
        random_seed=default_random_seed,
        **kwargs
):
    global_preprocessing_parameters = merge_dict(
        default_preprocessing_parameters,
        global_preprocessing_parameters
    )

    if train_set_metadata is None:
        train_set_metadata = build_metadata(
            dataset_df,
            features,
            global_preprocessing_parameters
        )

    data_val = build_data(
        dataset_df,
        features,
        train_set_metadata,
        global_preprocessing_parameters
    )

    data_val['split'] = get_split(
        dataset_df,
        force_split=global_preprocessing_parameters['force_split'],
        split_probabilities=global_preprocessing_parameters[
            'split_probabilities'
        ],
        stratify=global_preprocessing_parameters['stratify'],
        random_seed=random_seed
    )

    return data_val, train_set_metadata


def build_metadata(dataset_df, features, global_preprocessing_parameters):
    train_set_metadata = {}
    for feature in features:
        get_feature_meta = get_from_registry(
            feature['type'],
            base_type_registry
        ).get_feature_meta
        if 'preprocessing' in feature:
            preprocessing_parameters = merge_dict(
                global_preprocessing_parameters[feature['type']],
                feature['preprocessing']
            )
        else:
            preprocessing_parameters = global_preprocessing_parameters[
                feature['type']
            ]
        train_set_metadata[feature['name']] = get_feature_meta(
            dataset_df[feature['name']].astype(str),
            preprocessing_parameters
        )
    return train_set_metadata


def build_data(
        dataset_df,
        features,
        train_set_metadata,
        global_preprocessing_parameters
):
    data = {}
    for feature in features:
        add_feature_data = get_from_registry(
            feature['type'],
            base_type_registry
        ).add_feature_data
        if 'preprocessing' in feature:
            preprocessing_parameters = merge_dict(
                global_preprocessing_parameters[feature['type']],
                feature['preprocessing']
            )
        else:
            preprocessing_parameters = global_preprocessing_parameters[
                feature['type']
            ]
        handle_missing_values(
            dataset_df,
            feature,
            preprocessing_parameters
        )
        if feature['name'] not in train_set_metadata:
            train_set_metadata[feature['name']] = {}
        train_set_metadata[
            feature['name']
        ]['preprocessing'] = preprocessing_parameters
        add_feature_data(
            feature,
            dataset_df,
            data,
            train_set_metadata,
            preprocessing_parameters
        )
    return data


def handle_missing_values(dataset_df, feature, preprocessing_parameters):
    missing_value_strategy = preprocessing_parameters['missing_value_strategy']

    if missing_value_strategy == FILL_WITH_CONST:
        dataset_df[feature['name']] = dataset_df[feature['name']].fillna(
            preprocessing_parameters['fill_value'],
        )
    elif missing_value_strategy == FILL_WITH_MODE:
        dataset_df[feature['name']] = dataset_df[feature['name']].fillna(
            dataset_df[feature['name']].value_counts().index[0],
        )
    elif missing_value_strategy == FILL_WITH_MEAN:
        if feature['type'] != NUMERICAL:
            raise ValueError(
                'Filling missing values with mean is supported '
                'only for numerical types',
            )
        dataset_df[feature['name']] = dataset_df[feature['name']].fillna(
            dataset_df[feature['name']].mean(),
        )
    elif missing_value_strategy in ['backfill', 'bfill', 'pad', 'ffill']:
        dataset_df[feature['name']] = dataset_df[feature['name']].fillna(
            method=missing_value_strategy,
        )
    else:
        raise ValueError('Invalid missing value strategy')


def get_split(
        dataset_df,
        force_split=False,
        split_probabilities=(0.7, 0.1, 0.2),
        stratify=None,
        random_seed=default_random_seed,
):
    if 'split' in dataset_df and not force_split:
        split = dataset_df['split']
    else:
        set_random_seed(random_seed)
        if stratify is None or stratify not in dataset_df:
            split = np.random.choice(
                3,
                len(dataset_df),
                p=split_probabilities,
            ).astype(np.int8)
        else:
            split = np.zeros(len(dataset_df))
            for val in dataset_df[stratify].unique():
                idx_list = (
                    dataset_df.index[dataset_df[stratify] == val].tolist()
                )
                val_list = np.random.choice(
                    3,
                    len(idx_list),
                    p=split_probabilities,
                ).astype(np.int8)
                split[idx_list] = val_list
    return split


def load_data(
        hdf5_file_path,
        input_features,
        output_features,
        split_data=True,
        shuffle_training=False
):
    logging.info('Loading data from: {0}'.format(hdf5_file_path))
    # Load data from file
    hdf5_data = h5py.File(hdf5_file_path, 'r')
    dataset = {}
    for input_feature in input_features:
        if input_feature['type'] == TEXT:
            text_data_field = text_feature_data_field(input_feature)
            dataset[text_data_field] = hdf5_data[text_data_field].value
        else:
            dataset[input_feature['name']] = hdf5_data[
                input_feature['name']
            ].value
    for output_feature in output_features:
        if output_feature['type'] == TEXT:
            dataset[text_feature_data_field(output_feature)] = hdf5_data[
                text_feature_data_field(output_feature)
            ].value
        else:
            dataset[output_feature['name']] = hdf5_data[
                output_feature['name']].value
        if 'limit' in output_feature:
            dataset[output_feature['name']] = collapse_rare_labels(
                dataset[output_feature['name']],
                output_feature['limit']
            )

    if not split_data:
        hdf5_data.close()
        return dataset

    split = hdf5_data['split'].value
    hdf5_data.close()
    training_set, test_set, validation_set = split_dataset_tvt(dataset, split)

    # shuffle up
    if shuffle_training:
        training_set = data_utils.shuffle_dict_unison_inplace(training_set)

    return training_set, test_set, validation_set


def load_metadata(metadata_file_path):
    logging.info('Loading metadata from: {0}'.format(metadata_file_path))
    return data_utils.load_json(metadata_file_path)


def preprocess_for_training(
        model_definition,
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
        train_set_metadata_json=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    # Sanity Check to make sure some data source is provided
    data_sources_provided = [data_df, data_train_df, data_csv, data_train_csv,
                             data_hdf5, data_train_hdf5]
    data_sources_not_none = [x is not None for x in data_sources_provided]
    if not any(data_sources_not_none):
        raise ValueError('No training data is provided!')

    # Check if hdf5 and json already exist. If they do, use the hdf5 data,
    # instead of the csvs
    data_hdf5_fp = None

    if data_csv is not None:
        data_hdf5_fp = replace_file_extension(data_csv, 'hdf5')
        train_set_metadata_json_fp = replace_file_extension(data_csv, 'json')
        if os.path.isfile(data_hdf5_fp) and os.path.isfile(
                train_set_metadata_json_fp):
            logging.info(
                'Found hdf5 and json with the same filename '
                'of the csv, using them instead'
            )
            data_csv = None
            data_hdf5 = data_hdf5_fp
            train_set_metadata_json = train_set_metadata_json_fp

    if data_train_csv is not None:
        data_train_hdf5_fp = replace_file_extension(data_train_csv, 'hdf5')
        train_set_metadata_json_fp = replace_file_extension(
            data_train_csv,
            'json',
        )

        if os.path.isfile(data_train_hdf5_fp) and os.path.isfile(
                train_set_metadata_json_fp):
            logging.info(
                'Found hdf5 and json with the same filename of '
                'the train csv, using them instead'
            )
            data_train_csv = None
            data_train_hdf5 = data_train_hdf5_fp
            train_set_metadata_json = train_set_metadata_json_fp

    if data_validation_csv is not None:
        data_validation_hdf5_fp = replace_file_extension(
            data_validation_csv,
            'hdf5'
        )
        if os.path.isfile(data_validation_hdf5_fp):
            logging.info(
                'Found hdf5 with the same filename of '
                'the validation csv, using it instead'
            )
            data_validation_csv = None
            data_validation_hdf5 = data_validation_hdf5_fp

    if data_test_csv is not None:
        data_test_hdf5_fp = replace_file_extension(data_test_csv, 'hdf5')
        if os.path.isfile(data_test_hdf5_fp):
            logging.info(
                'Found hdf5 with the same filename of '
                'the test csv, using it instead'
            )
            data_test_csv = None
            data_test_hdf5 = data_test_hdf5_fp

    model_definition['data_hdf5_fp'] = data_hdf5_fp

    # Decide if to preprocess or just load
    features = (model_definition['input_features'] +
                model_definition['output_features'])

    if data_df is not None or data_train_df is not None:
        # Preprocess data frames
        (
            training_set,
            test_set,
            validation_set,
            train_set_metadata
        ) = _preprocess_df_for_training(
            features,
            data_df,
            data_train_df,
            data_validation_df,
            data_test_df,
            preprocessing_params,
            random_seed
        )
    elif data_csv is not None or data_train_csv is not None:
        # Preprocess csv data
        (
            training_set,
            test_set,
            validation_set,
            train_set_metadata
        ) = _preprocess_csv_for_training(
            features,
            data_csv,
            data_train_csv,
            data_validation_csv,
            data_test_csv,
            skip_save_processed_input,
            preprocessing_params,
            random_seed
        )

    elif data_hdf5 is not None and train_set_metadata_json is not None:
        # use data and train set metadata
        # does not need preprocessing, just load
        logging.info('Using full hdf5 and json')
        training_set, test_set, validation_set = load_data(
            data_hdf5,
            model_definition['input_features'],
            model_definition['output_features'],
            shuffle_training=True
        )
        train_set_metadata = load_metadata(train_set_metadata_json)

    elif data_train_hdf5 is not None and train_set_metadata_json is not None:
        # use data and train set metadata
        # does not need preprocessing, just load
        logging.info('Using hdf5 and json')
        training_set = load_data(
            data_train_hdf5,
            model_definition['input_features'],
            model_definition['output_features'],
            split_data=False
        )
        train_set_metadata = load_metadata(train_set_metadata_json)
        if data_validation_hdf5 is not None:
            validation_set = load_data(
                data_validation_hdf5,
                model_definition['input_features'],
                model_definition['output_features'],
                split_data=False
            )
        else:
            validation_set = None
        if data_test_hdf5 is not None:
            test_set = load_data(
                data_test_hdf5,
                model_definition['input_features'],
                model_definition['output_features'],
                split_data=False
            )
        else:
            test_set = None

    else:
        raise RuntimeError('Insufficient input parameters')

    replace_text_feature_level(
        model_definition['input_features'] +
        model_definition['output_features'],
        [training_set, validation_set, test_set]
    )

    training_dataset = Dataset(
        training_set,
        model_definition['input_features'],
        model_definition['output_features'],
        data_hdf5_fp
    )

    validation_dataset = None
    if validation_set is not None:
        validation_dataset = Dataset(
            validation_set,
            model_definition['input_features'],
            model_definition['output_features'],
            data_hdf5_fp
        )

    test_dataset = None
    if test_set is not None:
        test_dataset = Dataset(
            test_set,
            model_definition['input_features'],
            model_definition['output_features'],
            data_hdf5_fp
        )

    return (
        training_dataset,
        validation_dataset,
        test_dataset,
        train_set_metadata
    )


def _preprocess_csv_for_training(
        features,
        data_csv=None,
        data_train_csv=None,
        data_validation_csv=None,
        data_test_csv=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    """
    Method to pre-process csv data
    :param features: list of all features (input + output)
    :param data_csv: path to the csv data
    :param data_train_csv:  training csv data
    :param data_validation_csv: validation csv data
    :param data_test_csv: test csv data
    :param skip_save_processed_input: if False, the pre-processed data is saved
    as .hdf5 files in the same location as the csvs with the same names.
    :param preprocessing_params: preprocessing parameters
    :param random_seed: random seed
    :return: training, test, validation datasets, training metadata
    """
    if data_csv is not None:
        # Use data and ignore _train, _validation and _test.
        # Also ignore data and train set metadata needs preprocessing
        logging.info(
            'Using full raw csv, no hdf5 and json file '
            'with the same name have been found'
        )
        logging.info('Building dataset (it may take a while)')
        data, train_set_metadata = build_dataset(
            data_csv,
            features,
            preprocessing_params,
            random_seed=random_seed
        )
        if not skip_save_processed_input:
            logging.info('Writing dataset')
            data_hdf5_fp = replace_file_extension(data_csv, 'hdf5')
            data_utils.save_hdf5(data_hdf5_fp, data, train_set_metadata)
            logging.info('Writing train set metadata with vocabulary')

            train_set_metadata_json_fp = replace_file_extension(
                data_csv,
                'json'
            )
            data_utils.save_json(
                train_set_metadata_json_fp, train_set_metadata)

        training_set, test_set, validation_set = split_dataset_tvt(
            data,
            data['split']
        )

    elif data_train_csv is not None:
        # use data_train (including _validation and _test if they are present)
        # and ignore data and train set metadata
        # needs preprocessing
        logging.info(
            'Using training raw csv, no hdf5 and json '
            'file with the same name have been found'
        )
        logging.info('Building dataset (it may take a while)')
        concatenated_df = concatenate_csv(
            data_train_csv,
            data_validation_csv,
            data_test_csv
        )
        concatenated_df.csv = data_train_csv
        data, train_set_metadata = build_dataset_df(
            concatenated_df,
            features,
            preprocessing_params,
            random_seed=random_seed
        )
        training_set, test_set, validation_set = split_dataset_tvt(
            data,
            data['split']
        )
        if not skip_save_processed_input:
            logging.info('Writing dataset')
            data_train_hdf5_fp = replace_file_extension(data_train_csv, 'hdf5')
            data_utils.save_hdf5(
                data_train_hdf5_fp,
                training_set,
                train_set_metadata
            )
            if validation_set is not None:
                data_validation_hdf5_fp = replace_file_extension(
                    data_validation_csv,
                    'hdf5'
                )
                data_utils.save_hdf5(
                    data_validation_hdf5_fp,
                    validation_set,
                    train_set_metadata
                )
            if test_set is not None:
                data_test_hdf5_fp = replace_file_extension(data_test_csv,
                                                           'hdf5')
                data_utils.save_hdf5(
                    data_test_hdf5_fp,
                    test_set,
                    train_set_metadata
                )
            logging.info('Writing train set metadata with vocabulary')
            train_set_metadata_json_fp = replace_file_extension(data_train_csv,
                                                                'json')
            data_utils.save_json(train_set_metadata_json_fp, train_set_metadata)

    return training_set, test_set, validation_set, train_set_metadata


def _preprocess_df_for_training(
        features,
        data_df=None,
        data_train_df=None,
        data_validation_df=None,
        data_test_df=None,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    """ Method to pre-process dataframes. This doesn't have the optoin to save the
    processed data as hdf5 as we don't expect users to do this as the data can
    be processed in memory
    """

    if data_df is not None:
        # needs preprocessing
        logging.info('Using full dataframe')
        logging.info('Building dataset (it may take a while)')

    elif data_train_df is not None:
        # needs preprocessing
        logging.info('Using training dataframe')
        logging.info('Building dataset (it may take a while)')
        data_df = concatenate_df(
            data_train_df,
            data_validation_df,
            data_test_df
        )

    data, train_set_metadata = build_dataset_df(
        data_df,
        features,
        preprocessing_params,
        random_seed=random_seed
    )
    training_set, test_set, validation_set = split_dataset_tvt(
        data,
        data['split']
    )
    return training_set, test_set, validation_set, train_set_metadata


def preprocess_for_prediction(
        model_path,
        split,
        data_csv=None,
        data_hdf5=None,
        train_set_metadata=None,
        evaluate_performance=True
):
    """Preprocesses the dataset to parse it into a format that is usable by the
    Ludwig core
        :param model_path: The input data that is joined with the model
               hyperparameter file to create the model definition file
        :type model_path: Str
        :param split: Splits the data into the train and test sets
        :param data_csv: The CSV input data file
        :param data_hdf5: The hdf5 data file if there is no csv data file
        :param train_set_metadata: Train set metadata for the input features
        :param evaluate_performance: If False does not load output features
        :returns: Dataset, Train set metadata
        """
    model_definition = load_json(
        os.path.join(model_path, MODEL_HYPERPARAMETERS_FILE_NAME)
    )
    for input_feature in model_definition['input_features']:
        if 'preprocessing' in input_feature:
            if 'in_memory' in input_feature['preprocessing']:
                if not input_feature['preprocessing']['in_memory']:
                    logging.warning(
                        'WARNING: When running predict in_memory flag should '
                        'be true. Overriding and setting it to true for '
                        'feature <{}>'.format(input_feature['name'])
                    )
                    input_feature['preprocessing']['in_memory'] = True
    preprocessing_params = merge_dict(
        default_preprocessing_parameters,
        model_definition['preprocessing']
    )
    output_features = model_definition['output_features'] if evaluate_performance else []
    features = model_definition['input_features'] + output_features

    # Check if hdf5 file already exists
    if data_csv is not None:
        data_hdf5_fp = replace_file_extension(data_csv, 'hdf5')
        if os.path.isfile(data_hdf5_fp):
            logging.info('Found hdf5 with the same filename of the csv, '
                         'using it instead')
            data_csv = None
            data_hdf5 = data_hdf5_fp
    else:
        data_hdf5_fp = None

    # Load data
    train_set_metadata = load_metadata(train_set_metadata)
    if split == 'full':
        if data_hdf5 is not None:
            dataset = load_data(
                data_hdf5,
                model_definition['input_features'],
                output_features,
                split_data=False, shuffle_training=False
            )
        else:
            dataset, train_set_metadata = build_dataset(
                data_csv,
                features,
                preprocessing_params,
                train_set_metadata=train_set_metadata
            )
    else:
        if data_hdf5 is not None:
            training, test, validation = load_data(
                data_hdf5,
                model_definition['input_features'],
                output_features,
                shuffle_training=False
            )

            if split == 'training':
                dataset = training
            elif split == 'validation':
                dataset = validation
            else:  # if split == 'test':
                dataset = test
        else:
            dataset, train_set_metadata = build_dataset(
                data_csv,
                features,
                preprocessing_params,
                train_set_metadata=train_set_metadata
            )

    replace_text_feature_level(
        features,
        [dataset]
    )

    dataset = Dataset(
        dataset,
        model_definition['input_features'],
        output_features,
        data_hdf5_fp,
    )

    return dataset, train_set_metadata


def replace_text_feature_level(features, datasets):
    for feature in features:
        if feature['type'] == TEXT:
            for dataset in datasets:
                if dataset is not None:
                    dataset[feature['name']] = dataset[
                        '{}_{}'.format(
                            feature['name'],
                            feature['level']
                        )
                    ]
                    for level in ('word', 'char'):
                        name_level = '{}_{}'.format(
                            feature['name'],
                            level)
                        if name_level in dataset:
                            del dataset[name_level]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script takes csv files as input and outputs a HDF5 '
                    'and JSON file containing  a dataset and the train set '
                    'metadata associated with it'
    )

    parser.add_argument(
        '-id',
        '--dataset_csv',
        help='CSV containing contacts',
        required=True
    )
    parser.add_argument(
        '-ime',
        '--train_set_metadata_json',
        help='Input JSON containing metadata'
    )
    parser.add_argument(
        '-od',
        '--output_dataset_h5',
        help='HDF5 containing output data',
        required=True
    )
    parser.add_argument(
        '-ome',
        '--output_metadata_json',
        help='JSON containing metadata',
        required=True
    )

    parser.add_argument(
        '-f',
        '--features',
        type=yaml.safe_load,
        help='list of features in the CSV to map to hdf5 and JSON files'
    )

    parser.add_argument(
        '-p',
        '--preprocessing_parameters',
        type=yaml.safe_load,
        default='{}',
        help='the parameters for preprocessing the different features'
    )

    parser.add_argument(
        '-rs',
        '--random_seed',
        type=int,
        default=42,
        help='a random seed that is going to be used anywhere there is a call '
             'to a random number generator: data splitting, parameter '
             'initialization and training set shuffling'
    )

    args = parser.parse_args()

    data, train_set_metadata = build_dataset(
        args.dataset_csv,
        args.train_set_metadata_json,
        args.features,
        args.preprocessing_parameters,
        args.random_seed
    )

    # write train set metadata, dataset
    logging.info('Writing train set metadata with vocabulary')
    data_utils.save_json(args.output_metadata_json, train_set_metadata)
    logging.info('Writing dataset')
    data_utils.save_hdf5(args.output_dataset_h5, data, train_set_metadata)
