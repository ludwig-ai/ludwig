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
import argparse
import logging
import os

import h5py
import numpy as np
import pandas as pd
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
        metadata_val=None,
        random_seed=default_random_seed,
        **kwargs
):
    dataset_df = pd.read_csv(dataset_csv)
    dataset_df.csv = dataset_csv
    return build_dataset_df(
        dataset_df,
        features,
        global_preprocessing_parameters,
        metadata_val,
        random_seed,
        **kwargs
    )


def build_dataset_df(
        dataset_df,
        features,
        global_preprocessing_parameters,
        metadata_val=None,
        random_seed=default_random_seed,
        **kwargs
):
    global_preprocessing_parameters = merge_dict(
        default_preprocessing_parameters,
        global_preprocessing_parameters
    )

    if metadata_val is None:
        metadata_val = build_metadata(
            dataset_df,
            features,
            global_preprocessing_parameters
        )

    data_val = build_data(
        dataset_df,
        features,
        metadata_val,
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

    return data_val, metadata_val


def build_metadata(dataset_df, features, global_preprocessing_parameters):
    metadata_val = {}
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
        metadata_val[feature['name']] = get_feature_meta(
            dataset_df[feature['name']].astype(str),
            preprocessing_parameters
        )
    return metadata_val


def build_data(
        dataset_df,
        features,
        metadata_val,
        global_preprocessing_parameters
):
    data_val = {}
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
        if feature['name'] not in metadata_val:
            metadata_val[feature['name']] = {}
        metadata_val[
            feature['name']
        ]['preprocessing'] = preprocessing_parameters
        add_feature_data(
            feature,
            dataset_df,
            data_val,
            metadata_val,
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
        if stratify is None:
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
        validation_split=None,  # TODO validation_split is not used
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


def get_dataset_fun(dataset_type):
    return get_from_registry(
        dataset_type,
        dataset_type_registry
    )


dataset_type_registry = {
    'generic': (
        concatenate_csv,
        concatenate_df,
        build_dataset,
        build_dataset_df
    )
}


def preprocess_for_training(
        model_definition,
        dataset_type='generic',
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
        metadata_json=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    # Check if hdf5 and json already exist
    data_hdf5_fp = None
    data_train_hdf5_fp = None
    data_validation_hdf5_fp = None
    data_test_hdf5_fp = None
    metadata_json_fp = 'metadata.json'
    if data_csv is not None:
        data_hdf5_fp = data_csv.replace('csv', 'hdf5')
        metadata_json_fp = data_csv.replace('csv', 'json')
        if os.path.isfile(data_hdf5_fp) and os.path.isfile(metadata_json_fp):
            logging.info(
                'Found hdf5 and json with the same filename '
                'of the csv, using them instead'
            )
            data_csv = None
            data_hdf5 = data_hdf5_fp
            metadata_json = metadata_json_fp

    if data_train_csv is not None:
        data_train_hdf5_fp = data_train_csv.replace('csv', 'hdf5')
        metadata_json_fp = data_train_csv.replace('csv', 'json')
        if (os.path.isfile(data_train_hdf5_fp) and
                os.path.isfile(metadata_json_fp)):
            logging.info(
                'Found hdf5 and json with the same filename of '
                'the train csv, using them instead'
            )
            data_train_csv = None
            data_train_hdf5 = data_train_hdf5_fp
            metadata_json = metadata_json_fp

    if data_validation_csv is not None:
        data_validation_hdf5_fp = data_validation_csv.replace('csv', 'hdf5')
        if os.path.isfile(data_validation_hdf5_fp):
            logging.info(
                'Found hdf5 with the same filename of '
                'the validation csv, using it instead'
            )
            data_validation_csv = None
            data_validation_hdf5 = data_validation_hdf5_fp

    if data_test_csv is not None:
        data_test_hdf5_fp = data_test_csv.replace('csv', 'hdf5')
        if os.path.isfile(data_test_hdf5_fp):
            logging.info(
                'Found hdf5 with the same filename of '
                'the validation csv, using it instead'
            )
            data_test_csv = None
            data_test_hdf5 = data_test_hdf5_fp

    model_definition['data_hdf5_fp'] = data_hdf5_fp

    # Decide if to preprocess or just load
    features = (model_definition['input_features'] +
                model_definition['output_features'])
    (
        concatenate_csv,  # TODO - all of these shadow names from outer scope
        concatenate_df,
        build_dataset,
        build_dataset_df
    ) = get_dataset_fun(dataset_type)

    if data_df is not None:
        # needs preprocessing
        logging.info('Using full dataframe')
        logging.info('Building dataset (it may take a while)')
        # TODO these two shadow names from outer scope
        data, metadata = build_dataset_df(
            data_df,
            features,
            preprocessing_params,
            random_seed=random_seed
        )
        if not skip_save_processed_input:
            logging.info('Writing dataset')
            data_utils.save_hdf5(data_hdf5_fp, data, metadata)
        logging.info('Writing metadata with vocabulary')
        data_utils.save_json(metadata_json_fp, metadata)
        training_set, test_set, validation_set = split_dataset_tvt(
            data,
            data['split']
        )

    elif data_train_df is not None:
        # needs preprocessing
        logging.info('Using training dataframe')
        logging.info('Building dataset (it may take a while)')
        concatenated_df = concatenate_df(
            data_train_df,
            data_validation_df,
            data_test_df
        )
        # TODO these two shadow names from outer scope
        data, metadata = build_dataset_df(
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
            data_utils.save_hdf5(data_train_hdf5_fp, training_set, metadata)
            if validation_set is not None:
                data_utils.save_hdf5(
                    data_validation_hdf5_fp,
                    validation_set,
                    metadata
                )
            if test_set is not None:
                data_utils.save_hdf5(data_test_hdf5_fp, test_set, metadata)
        logging.info('Writing metadata with vocabulary')
        data_utils.save_json(metadata_json_fp, metadata)

    elif data_csv is not None:
        # Use data and ignore _train, _validation and _test.
        # Also ignore data and metadata needs preprocessing
        logging.info(
            'Using full raw csv, no hdf5 and json file '
            'with the same name have been found'
        )
        logging.info('Building dataset (it may take a while)')
        # TODO these two shadow names from outer scope
        data, metadata = build_dataset(
            data_csv,
            features,
            preprocessing_params,
            random_seed=random_seed
        )
        if not skip_save_processed_input:
            logging.info('Writing dataset')
            data_utils.save_hdf5(data_hdf5_fp, data, metadata)
            logging.info('Writing metadata with vocabulary')
            data_utils.save_json(metadata_json_fp, metadata)
        training_set, test_set, validation_set = split_dataset_tvt(
            data,
            data['split']
        )

    elif data_train_csv is not None:
        # use data_train (including _validation and _test if they are present)
        # and ignore data and metadata
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
        # TODO these two shadow names from outer scope
        data, metadata = build_dataset_df(
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
            data_utils.save_hdf5(data_train_hdf5_fp, training_set, metadata)
            if validation_set is not None:
                data_utils.save_hdf5(
                    data_validation_hdf5_fp,
                    validation_set,
                    metadata
                )
            if test_set is not None:
                data_utils.save_hdf5(data_test_hdf5_fp, test_set, metadata)
            logging.info('Writing metadata with vocabulary')
            data_utils.save_json(metadata_json_fp, metadata)

    elif data_hdf5 is not None and metadata_json is not None:
        # use data and metadata
        # doesn't need preprocessing, just load
        logging.info('Using full hdf5 and json')
        training_set, test_set, validation_set = load_data(
            data_hdf5,
            model_definition['input_features'],
            model_definition['output_features'],
            shuffle_training=True
        )
        # TODO - shadows name from outer scope
        metadata = load_metadata(metadata_json)

    elif data_train_hdf5 is not None and metadata_json is not None:
        # use data and metadata
        # doesn't need preprocessing, just load
        logging.info('Using hdf5 and json')
        training_set = load_data(
            data_train_hdf5,
            model_definition['input_features'],
            model_definition['output_features'],
            split_data=False
        )
        # TODO - shadows name from outer scope
        metadata = load_metadata(metadata_json)
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
        model_definition,
        [training_set, validation_set, test_set]
    )

    training_dataset = Dataset(
        training_set,
        model_definition['input_features'],
        model_definition['output_features'],
        data_hdf5_fp
    )
    validation_dataset = Dataset(
        validation_set,
        model_definition['input_features'],
        model_definition['output_features'],
        data_hdf5_fp
    )
    test_dataset = Dataset(
        test_set,
        model_definition['input_features'],
        model_definition['output_features'],
        data_hdf5_fp
    )

    return training_dataset, validation_dataset, test_dataset, metadata


def preprocess_for_prediction(
        model_path,
        split,
        dataset_type='generic',
        data_csv=None,
        data_hdf5=None,
        metadata=None,  # TODO shadows name from outer scope
        only_predictions=False
):
    """Preprocesses the dataset to parse it into a format that is usable by the
    Ludwig core
        :param model_path: The input data that is joined with the model
               hyperparameter file to create the model definition file
        :type model_path: Str
        :param dataset_type: Generic
        :type: Str
        :param split: Splits the data into the train and test sets
        :param data_csv: The CSV input data file
        :param data_hdf5: The hdf5 data file if there is no csv data file
        :param metadata: Metadata for the input features
        :param only_predictions: TODO
        :returns: Dataset, Metadata
        """
    model_definition = load_json(
        os.path.join(model_path, MODEL_HYPERPARAMETERS_FILE_NAME)
    )
    preprocessing_params = merge_dict(
        default_preprocessing_parameters,
        model_definition['preprocessing']
    )

    # Check if hdf5 and json already exist
    if data_csv is not None:
        data_hdf5_fp = data_csv.replace('csv', 'hdf5')
        if os.path.isfile(data_hdf5_fp):
            logging.info(
                'Found hdf5 with the same filename of the csv, using it instead'
            )
            data_csv = None
            data_hdf5 = data_hdf5_fp

    # Load data
    # TODO build_dataset shadows name from outer scope
    _, _, build_dataset, _ = get_dataset_fun(dataset_type)
    metadata = load_metadata(metadata)  # TODO shadows name from outer scope
    features = (model_definition['input_features'] +
                ([] if only_predictions
                 else model_definition['output_features']))
    if split == 'full':
        if data_hdf5 is not None:
            dataset = load_data(
                data_hdf5,
                model_definition['input_features'],
                [] if only_predictions else model_definition['output_features'],
                split_data=False, shuffle_training=False
            )
        else:
            # TODO shadows name from outer scope
            dataset, metadata = build_dataset(
                data_csv,
                features,
                preprocessing_params,
                metadata=metadata
            )
    else:
        if data_hdf5 is not None:
            training, test, validation = load_data(
                data_hdf5,
                model_definition['input_features'],
                [] if only_predictions else model_definition['output_features'],
                shuffle_training=False
            )

            if split == 'training':
                dataset = training
            elif split == 'validation':
                dataset = validation
            else:  # if split == 'test':
                dataset = test
        else:
            # TODO shadows name from outer scope
            dataset, metadata = build_dataset(
                data_csv,
                features,
                preprocessing_params,
                metadata=metadata
            )

    replace_text_feature_level(model_definition, [dataset])

    dataset = Dataset(
        dataset,
        model_definition['input_features'],
        [] if only_predictions else model_definition['output_features'],
        data_hdf5,
    )

    return dataset, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script takes csv files as input and outputs a HDF5 '
                    'and JSON file containing  a dataset and the metadata '
                    'associated with it'
    )

    parser.add_argument(
        '-id',
        '--dataset_csv',
        help='CSV containing contacts',
        required=True
    )
    parser.add_argument(
        '-ime',
        '--metadata_json',
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
        type=yaml.load,
        help='list of features in the CSV to map to hdf5 and JSON files'
    )

    parser.add_argument(
        '-p',
        '--preprocessing_parameters',
        type=yaml.load,
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

    data, metadata = build_dataset(
        args.dataset_csv,
        args.metadata_json,
        args.features,
        args.preprocessing_parameters,
        args.random_seed
    )

    # write metadata, dataset
    logging.info('Writing metadata with vocabulary')
    data_utils.save_json(args.output_metadata_json, metadata)
    logging.info('Writing dataset')
    data_utils.save_hdf5(args.output_dataset_h5, data, metadata)


def replace_text_feature_level(model_definition, datasets):
    for feature in (model_definition['input_features'] +
                    model_definition['output_features']):
        if feature['type'] == TEXT:
            for dataset in datasets:
                dataset[feature['name']] = dataset[
                    '{}_{}'.format(
                        feature['name'],
                        feature['level']
                    )
                ]
                for level in ('word', 'char'):
                    del dataset[
                        '{}_{}'.format(
                            feature['name'],
                            level)
                    ]
