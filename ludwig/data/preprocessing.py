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
from abc import ABC, abstractmethod

import h5py
import numpy as np
import pandas as pd
import yaml

from ludwig.constants import *
from ludwig.constants import TEXT
from ludwig.data.concatenate_datasets import concatenate_csv
from ludwig.data.concatenate_datasets import concatenate_df
from ludwig.data.dataset import Dataset
from ludwig.features.feature_registries import base_type_registry, \
    input_type_registry
from ludwig.utils import data_utils
from ludwig.utils.data_utils import figure_data_format, \
    DATA_TRAIN_HDF5_FP, DICT_FORMATS, DATAFRAME_FORMATS, CSV_FORMATS, \
    HDF5_FORMATS, override_in_memory_flag, TSV_FORMATS, JSON_FORMATS, \
    JSONL_FORMATS, read_tsv, read_jsonl, read_json, EXCEL_FORMATS, read_excel
from ludwig.utils.data_utils import file_exists_with_diff_extension
from ludwig.utils.data_utils import read_csv
from ludwig.utils.data_utils import replace_file_extension
from ludwig.utils.data_utils import split_dataset_ttv
from ludwig.utils.data_utils import text_feature_data_field
from ludwig.utils.defaults import default_preprocessing_parameters, \
    merge_with_defaults
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.horovod_utils import is_on_master
from ludwig.utils.misc_utils import get_from_registry, resolve_pointers
from ludwig.utils.misc_utils import merge_dict
from ludwig.utils.misc_utils import set_random_seed

logger = logging.getLogger(__name__)


class DataFormatPreprocessor(ABC):

    @staticmethod
    @abstractmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        pass

    @staticmethod
    @abstractmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        pass


class DictPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        num_overrides = override_in_memory_flag(features, True)
        if num_overrides > 0:
            logger.warning(
                'Using in_memory = False is not supported '
                'with {} data format.'.format('dict')
            )

        if dataset is not None:
            dataset = pd.DataFrame(dataset)
        if training_set_metadata is not None:
            training_set = pd.DataFrame(training_set)
        if validation_set is not None:
            validation_set = pd.DataFrame(validation_set)
        if test_set is not None:
            test_set = pd.DataFrame(test_set)

        return _preprocess_df_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            training_set_metadata=training_set_metadata,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset, training_set_metadata = build_dataset(
            pd.DataFrame(dataset),
            features,
            preprocessing_params,
            metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class DataFramePreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        num_overrides = override_in_memory_flag(features, True)
        if num_overrides > 0:
            logger.warning(
                'Using in_memory = False is not supported '
                'with {} data format.'.format('dataframe')
            )

        return _preprocess_df_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            training_set_metadata=training_set_metadata,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset, training_set_metadata = build_dataset(
            dataset,
            features,
            preprocessing_params,
            metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class CSVPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        return _preprocess_file_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_csv,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset_df = read_csv(dataset)
        dataset_df.src = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            training_set_metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class TSVPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        return _preprocess_file_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_tsv,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset_df = read_tsv(dataset)
        dataset_df.src = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            training_set_metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class JSONPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        return _preprocess_file_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_json,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset_df = read_json(dataset)
        dataset_df.src = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            training_set_metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class JSONLPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        return _preprocess_file_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_jsonl,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset_df = read_jsonl(dataset)
        dataset_df.src = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            training_set_metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class ExcelPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        return _preprocess_file_for_training(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_excel,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            random_seed=random_seed
        )

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        dataset_df = read_excel(dataset)
        dataset_df.src = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            training_set_metadata=training_set_metadata
        )
        return dataset, training_set_metadata, None


class HDF5Preprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
            features,
            dataset=None,
            training_set=None,
            validation_set=None,
            test_set=None,
            training_set_metadata=None,
            skip_save_processed_input=False,
            preprocessing_params=default_preprocessing_parameters,
            random_seed=default_random_seed
    ):
        if not training_set_metadata:
            raise ValueError('When providing HDF5 data, '
                             'training_set_metadata must not be None.')

        logger.info('Using full hdf5 and json')

        if DATA_TRAIN_HDF5_FP not in training_set_metadata:
            logger.warning(
                'data_train_hdf5_fp not present in training_set_metadata. '
                'Adding it with the current HDF5 file path {}'.format(dataset)
            )
            training_set_metadata[DATA_TRAIN_HDF5_FP] = dataset
        elif training_set_metadata[DATA_TRAIN_HDF5_FP] != dataset:
            logger.warning(
                'data_train_hdf5_fp in training_set_metadata is {}, '
                'different from the current HDF5 file path {}. '
                'Replacing it'.format(
                    training_set_metadata[DATA_TRAIN_HDF5_FP],
                    dataset
                )
            )
            training_set_metadata[DATA_TRAIN_HDF5_FP] = dataset

        if dataset is not None:
            training_set, test_set, validation_set = load_hdf5(
                dataset,
                features,
                shuffle_training=True
            )
        elif training_set is not None:
            kwargs = dict(features=features, split_data=False)
            training_set = load_hdf5(training_set,
                                     shuffle_training=True,
                                     **kwargs)
            if validation_set is not None:
                validation_set = load_hdf5(validation_set,
                                           shuffle_training=False,
                                           **kwargs)
            if test_set is not None:
                test_set = load_hdf5(test_set,
                                     shuffle_training=False,
                                     **kwargs)
        else:
            raise ValueError(
                'One of `dataset` or `training_set` must be not None')

        return training_set, test_set, validation_set, training_set_metadata

    @staticmethod
    def preprocess_for_prediction(
            dataset,
            features,
            preprocessing_params,
            training_set_metadata
    ):
        hdf5_fp = dataset
        dataset = load_hdf5(
            dataset,
            features,
            split_data=False,
            shuffle_training=False
        )
        return dataset, training_set_metadata, hdf5_fp


data_format_preprocessor_registry = {
    **{fmt: DictPreprocessor for fmt in DICT_FORMATS},
    **{fmt: DataFramePreprocessor for fmt in DATAFRAME_FORMATS},
    **{fmt: CSVPreprocessor for fmt in CSV_FORMATS},
    **{fmt: TSVPreprocessor for fmt in TSV_FORMATS},
    **{fmt: JSONPreprocessor for fmt in JSON_FORMATS},
    **{fmt: JSONLPreprocessor for fmt in JSONL_FORMATS},
    **{fmt: ExcelPreprocessor for fmt in EXCEL_FORMATS},
    **{fmt: HDF5Preprocessor for fmt in HDF5_FORMATS},
}


def build_dataset(
        dataset_df,
        features,
        global_preprocessing_parameters,
        metadata=None,
        random_seed=default_random_seed,
        **kwargs
):
    global_preprocessing_parameters = merge_dict(
        default_preprocessing_parameters,
        global_preprocessing_parameters
    )

    if metadata is None:
        metadata = build_metadata(
            dataset_df,
            features,
            global_preprocessing_parameters
        )

    dataset = build_data(
        dataset_df,
        features,
        metadata,
        global_preprocessing_parameters
    )

    dataset[SPLIT] = get_split(
        dataset_df,
        force_split=global_preprocessing_parameters['force_split'],
        split_probabilities=global_preprocessing_parameters[
            'split_probabilities'
        ],
        stratify=global_preprocessing_parameters['stratify'],
        random_seed=random_seed
    )

    return dataset, metadata


def build_metadata(dataset_df, features, global_preprocessing_parameters):
    metadata = {}
    for feature in features:
        if PREPROCESSING in feature:
            preprocessing_parameters = merge_dict(
                global_preprocessing_parameters[feature[TYPE]],
                feature[PREPROCESSING]
            )
        else:
            preprocessing_parameters = global_preprocessing_parameters[
                feature[TYPE]
            ]

        # deal with encoders that have fixed preprocessing
        if 'encoder' in feature:
            encoders_registry = get_from_registry(
                feature[TYPE],
                input_type_registry
            ).encoder_registry
            encoder_class = encoders_registry[feature['encoder']]
            if hasattr(encoder_class, 'fixed_preprocessing_parameters'):
                encoder_fpp = encoder_class.fixed_preprocessing_parameters

                preprocessing_parameters = merge_dict(
                    preprocessing_parameters,
                    resolve_pointers(encoder_fpp, feature, 'feature.')
                )

        handle_missing_values(
            dataset_df,
            feature,
            preprocessing_parameters
        )

        get_feature_meta = get_from_registry(
            feature[TYPE],
            base_type_registry
        ).get_feature_meta
        metadata[feature[NAME]] = get_feature_meta(
            dataset_df[feature[NAME]].astype(str),
            preprocessing_parameters
        )
    return metadata


def build_data(
        dataset_df,
        features,
        training_set_metadata,
        global_preprocessing_parameters
):
    dataset = {}
    for feature in features:
        if PREPROCESSING in feature:
            preprocessing_parameters = merge_dict(
                global_preprocessing_parameters[feature[TYPE]],
                feature[PREPROCESSING]
            )
        else:
            preprocessing_parameters = global_preprocessing_parameters[
                feature[TYPE]
            ]

        # deal with encoders that have fixed preprocessing
        if 'encoder' in feature:
            encoders_registry = get_from_registry(
                feature[TYPE],
                input_type_registry
            ).encoder_registry

            encoder_class = encoders_registry[feature['encoder']]
            if hasattr(encoder_class, 'fixed_preprocessing_parameters'):
                encoder_fpp = encoder_class.fixed_preprocessing_parameters

                preprocessing_parameters = merge_dict(
                    preprocessing_parameters,
                    resolve_pointers(encoder_fpp, feature, 'feature.')
                )

        handle_missing_values(
            dataset_df,
            feature,
            preprocessing_parameters
        )
        if feature[NAME] not in training_set_metadata:
            training_set_metadata[feature[NAME]] = {}
        training_set_metadata[
            feature[NAME]
        ][PREPROCESSING] = preprocessing_parameters

        add_feature_data = get_from_registry(
            feature[TYPE],
            base_type_registry
        ).add_feature_data
        add_feature_data(
            feature,
            dataset_df,
            dataset,
            training_set_metadata,
            preprocessing_parameters
        )
    return dataset


def handle_missing_values(dataset_df, feature, preprocessing_parameters):
    missing_value_strategy = preprocessing_parameters['missing_value_strategy']

    if missing_value_strategy == FILL_WITH_CONST:
        dataset_df[feature[NAME]] = dataset_df[feature[NAME]].fillna(
            preprocessing_parameters['fill_value'],
        )
    elif missing_value_strategy == FILL_WITH_MODE:
        dataset_df[feature[NAME]] = dataset_df[feature[NAME]].fillna(
            dataset_df[feature[NAME]].value_counts().index[0],
        )
    elif missing_value_strategy == FILL_WITH_MEAN:
        if feature[TYPE] != NUMERICAL:
            raise ValueError(
                'Filling missing values with mean is supported '
                'only for numerical types',
            )
        dataset_df[feature[NAME]] = dataset_df[feature[NAME]].fillna(
            dataset_df[feature[NAME]].mean(),
        )
    elif missing_value_strategy in ['backfill', 'bfill', 'pad', 'ffill']:
        dataset_df[feature[NAME]] = dataset_df[feature[NAME]].fillna(
            method=missing_value_strategy,
        )
    elif missing_value_strategy == DROP_ROW:
        dataset_df.dropna(subset=[feature[NAME]], inplace=True)
    else:
        raise ValueError('Invalid missing value strategy')


def get_split(
        dataset_df,
        force_split=False,
        split_probabilities=(0.7, 0.1, 0.2),
        stratify=None,
        random_seed=default_random_seed,
):
    if SPLIT in dataset_df and not force_split:
        split = dataset_df[SPLIT]
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


def load_hdf5(
        hdf5_file_path,
        features,
        split_data=True,
        shuffle_training=False
):
    logger.info('Loading data from: {0}'.format(hdf5_file_path))
    # Load data from file
    hdf5_data = h5py.File(hdf5_file_path, 'r')
    dataset = {}
    for feature in features:
        if feature[TYPE] == TEXT:
            text_data_field = text_feature_data_field(feature)
            dataset[text_data_field] = hdf5_data[text_data_field][()]
        else:
            dataset[feature[NAME]] = hdf5_data[feature[NAME]][()]

    if not split_data:
        hdf5_data.close()
        if shuffle_training:
            dataset = data_utils.shuffle_dict_unison_inplace(dataset)
        return dataset

    split = hdf5_data[SPLIT][()]
    hdf5_data.close()
    training_set, test_set, validation_set = split_dataset_ttv(dataset, split)

    if shuffle_training:
        training_set = data_utils.shuffle_dict_unison_inplace(training_set)

    return training_set, test_set, validation_set


def load_metadata(metadata_file_path):
    logger.info('Loading metadata from: {0}'.format(metadata_file_path))
    return data_utils.load_json(metadata_file_path)


def preprocess_for_training(
        model_definition,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        data_format=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    # sanity check to make sure some data source is provided
    if dataset is None and training_set is None:
        raise ValueError('No training data is provided!')

    # determine data format if not provided or auto
    if not data_format or data_format == 'auto':
        data_format = figure_data_format(
            dataset, training_set, validation_set, test_set
        )

    # if training_set_metadata is a string, assume it's a path to load the json
    if training_set_metadata and isinstance(training_set_metadata, str):
        training_set_metadata = load_metadata(training_set_metadata)

    # setup
    features = (model_definition['input_features'] +
                model_definition['output_features'])

    # in case data_format is csv,
    # check if there's a cached hdf5 file with hte same name,
    # and in case move on with the hdf5 branch
    if (data_format in CSV_FORMATS or data_format in TSV_FORMATS or
            data_format in JSON_FORMATS or data_format in JSONL_FORMATS or
            data_format in EXCEL_FORMATS):
        if dataset:
            if (file_exists_with_diff_extension(dataset, 'hdf5') and
                    file_exists_with_diff_extension(dataset, 'meta.json')):
                logger.info(
                    'Found hdf5 and meta.json with the same filename '
                    'of the input file, using them instead'
                )
                dataset = replace_file_extension(dataset, 'hdf5')
                training_set_metadata_fp = replace_file_extension(dataset,
                                                                  'meta.json')
                training_set_metadata = data_utils.load_json(
                    training_set_metadata_fp)
                model_definition['data_hdf5_fp'] = dataset
                data_format = 'hdf5'

        elif training_set:
            if (file_exists_with_diff_extension(training_set, 'hdf5') and
                    file_exists_with_diff_extension(training_set,
                                                    'meta.json')):
                logger.info(
                    'Found hdf5 and json with the same filename '
                    'of the csv, using them instead'
                )
                training_set = replace_file_extension(training_set, 'hdf5')
                training_set_metadata_fp = replace_file_extension(training_set,
                                                                  'meta.json')
                training_set_metadata = data_utils.load_json(
                    training_set_metadata_fp
                )
                validation_set = replace_file_extension(validation_set, 'hdf5')
                test_set = replace_file_extension(test_set, 'hdf5')
                model_definition['data_hdf5_fp'] = training_set
                data_format = 'hdf5'

    data_format_processor = get_from_registry(
        data_format,
        data_format_preprocessor_registry
    )
    processed = data_format_processor.preprocess_for_training(
        features,
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        skip_save_processed_input=skip_save_processed_input,
        preprocessing_params=preprocessing_params,
        random_seed=random_seed
    )
    training_set, test_set, validation_set, training_set_metadata = processed

    replace_text_feature_level(
        features,
        [training_set, validation_set, test_set]
    )

    training_dataset = Dataset(
        training_set,
        model_definition['input_features'],
        model_definition['output_features'],
        training_set_metadata.get(DATA_TRAIN_HDF5_FP)
    )

    validation_dataset = None
    if validation_set is not None:
        validation_dataset = Dataset(
            validation_set,
            model_definition['input_features'],
            model_definition['output_features'],
            training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        )

    test_dataset = None
    if test_set is not None:
        test_dataset = Dataset(
            test_set,
            model_definition['input_features'],
            model_definition['output_features'],
            training_set_metadata.get(DATA_TRAIN_HDF5_FP)
        )

    return (
        training_dataset,
        validation_dataset,
        test_dataset,
        training_set_metadata
    )


def _preprocess_file_for_training(
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        read_fn=read_csv,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    """
    Method to pre-process csv data
    :param features: list of all features (input + output)
    :param dataset: path to the csv data
    :param training_set:  training csv data
    :param validation_set: validation csv data
    :param test_set: test csv data
    :param training_set_metadata: train set metadata
    :param skip_save_processed_input: if False, the pre-processed data is saved
    as .hdf5 files in the same location as the csvs with the same names.
    :param preprocessing_params: preprocessing parameters
    :param random_seed: random seed
    :return: training, test, validation datasets, training metadata
    """
    if dataset:
        # Use data and ignore _train, _validation and _test.
        # Also ignore data and train set metadata needs preprocessing
        logger.info(
            'Using full raw csv, no hdf5 and json file '
            'with the same name have been found'
        )
        logger.info('Building dataset (it may take a while)')

        dataset_df = read_fn(dataset)
        dataset_df.src = dataset
        data, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            training_set_metadata=training_set_metadata,
            random_seed=random_seed
        )

        if is_on_master() and not skip_save_processed_input:
            logger.info('Writing prprocessed dataset cache')
            data_hdf5_fp = replace_file_extension(dataset, 'hdf5')
            data_utils.save_hdf5(data_hdf5_fp, data, training_set_metadata)
            training_set_metadata[DATA_TRAIN_HDF5_FP] = data_hdf5_fp
            logger.info('Writing train set metadata with vocabulary')
            training_set_metadata_fp = replace_file_extension(dataset,
                                                              'meta.json')
            data_utils.save_json(training_set_metadata_fp,
                                 training_set_metadata)

        training_data, test_data, validation_data = split_dataset_ttv(
            data,
            data[SPLIT]
        )

    elif training_set:
        # use data_train (including _validation and _test if they are present)
        # and ignore data and train set metadata
        # needs preprocessing
        logger.info(
            'Using training raw csv, no hdf5 and json '
            'file with the same name have been found'
        )
        logger.info('Building dataset (it may take a while)')

        concatenated_df = concatenate_csv(
            training_set,
            validation_set,
            test_set
        )
        concatenated_df.src = training_set

        data, training_set_metadata = build_dataset(
            concatenated_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            random_seed=random_seed
        )

        training_data, test_data, validation_data = split_dataset_ttv(
            data,
            data[SPLIT]
        )

        if is_on_master() and not skip_save_processed_input:
            logger.info('Writing prprocessed dataset cache')
            data_train_hdf5_fp = replace_file_extension(training_set, 'hdf5')
            data_utils.save_hdf5(
                data_train_hdf5_fp,
                training_data,
                training_set_metadata
            )
            training_set_metadata[DATA_TRAIN_HDF5_FP] = data_train_hdf5_fp

            if validation_set is not None:
                data_validation_hdf5_fp = replace_file_extension(
                    validation_set,
                    'hdf5'
                )
                data_utils.save_hdf5(
                    data_validation_hdf5_fp,
                    validation_data,
                    training_set_metadata
                )

            if test_set is not None:
                data_test_hdf5_fp = replace_file_extension(
                    test_set,
                    'hdf5'
                )
                data_utils.save_hdf5(
                    data_test_hdf5_fp,
                    test_data,
                    training_set_metadata
                )

            logger.info('Writing train set metadata with vocabulary')
            training_set_metadata_fp = replace_file_extension(
                training_set,
                'meta.json'
            )
            data_utils.save_json(
                training_set_metadata_fp,
                training_set_metadata,
            )

    else:
        raise ValueError('either data or data_train have to be not None')

    return training_data, test_data, validation_data, training_set_metadata


def _preprocess_df_for_training(
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        preprocessing_params=default_preprocessing_parameters,
        random_seed=default_random_seed
):
    """ Method to pre-process dataframes. This doesn't have the option to save the
    processed data as hdf5 as we don't expect users to do this as the data can
    be processed in memory
    """
    if dataset is not None:
        # needs preprocessing
        logger.info('Using full dataframe')
        logger.info('Building dataset (it may take a while)')

    elif training_set is not None:
        # needs preprocessing
        logger.info('Using training dataframe')
        logger.info('Building dataset (it may take a while)')
        dataset = concatenate_df(
            training_set,
            validation_set,
            test_set
        )

    dataset, training_set_metadata = build_dataset(
        dataset,
        features,
        preprocessing_params,
        metadata=training_set_metadata,
        random_seed=random_seed
    )
    training_set, test_set, validation_set = split_dataset_ttv(
        dataset,
        dataset[SPLIT]
    )
    return training_set, test_set, validation_set, training_set_metadata


def preprocess_for_prediction(
        model_definition,
        dataset,
        training_set_metadata=None,
        data_format=None,
        include_outputs=True,
):
    """Preprocesses the dataset to parse it into a format that is usable by the
    Ludwig core
        :param model_path: The input data that is joined with the model
               hyperparameter file to create the model definition file
        :type model_path: Str
        :param split: Splits the data into the train and test sets
        :param data_csv: The CSV input data file
        :param data_hdf5: The hdf5 data file if there is no csv data file
        :param training_set_metadata: Train set metadata for the input features
        :param evaluate_performance: If False does not load output features
        :returns: Dataset, Train set metadata
        """
    # Sanity Check to make sure some data source is provided
    if dataset is None:
        raise ValueError('No training data is provided!')

    if isinstance(dataset, Dataset):
        return dataset, training_set_metadata

    # determine data format if not provided or auto
    if not data_format or data_format == 'auto':
        data_format = figure_data_format(dataset)

    # manage the in_memory parameter
    if data_format not in HDF5_FORMATS:
        num_overrides = override_in_memory_flag(
            model_definition['input_features'],
            True
        )
        if num_overrides > 0:
            logger.warning(
                'Using in_memory = False is not supported '
                'with {} data format.'.format(data_format)
            )

    preprocessing_params = merge_dict(
        default_preprocessing_parameters,
        model_definition[PREPROCESSING]
    )

    # if training_set_metadata is a string, assume it's a path to load the json
    if training_set_metadata and isinstance(training_set_metadata, str):
        training_set_metadata = load_metadata(training_set_metadata)

    hdf5_fp = training_set_metadata.get(DATA_TRAIN_HDF5_FP, None)

    # setup
    output_features = []
    if include_outputs:
        output_features += model_definition['output_features']
    features = model_definition['input_features'] + output_features

    # in case data_format is csv,
    # check if there's a cached hdf5 file with hte same name,
    # and in case move on with the hdf5 branch
    if (data_format in CSV_FORMATS or data_format in TSV_FORMATS or
            data_format in JSON_FORMATS or data_format in JSONL_FORMATS or
            data_format in EXCEL_FORMATS):
        if (file_exists_with_diff_extension(dataset, 'hdf5') and
                file_exists_with_diff_extension(dataset, 'meta.json')):
            logger.info(
                'Found hdf5 and meta.json with the same filename '
                'of the input file, using them instead'
            )
            dataset = replace_file_extension(dataset, 'hdf5')
            model_definition['data_hdf5_fp'] = dataset
            data_format = 'hdf5'

    data_format_processor = get_from_registry(
        data_format,
        data_format_preprocessor_registry
    )
    processed = data_format_processor.preprocess_for_prediction(
        dataset,
        features,
        preprocessing_params,
        training_set_metadata
    )
    dataset, training_set_metadata, new_hdf5_fp = processed
    if new_hdf5_fp:
        hdf5_fp = new_hdf5_fp

    replace_text_feature_level(features, [dataset])

    dataset = Dataset(
        dataset,
        model_definition['input_features'],
        output_features,
        hdf5_fp
    )

    return dataset, training_set_metadata


def replace_text_feature_level(features, datasets):
    for feature in features:
        if feature[TYPE] == TEXT:
            for dataset in datasets:
                if dataset is not None:
                    dataset[feature[NAME]] = dataset[
                        '{}_{}'.format(
                            feature[NAME],
                            feature['level']
                        )
                    ]
                    for level in ('word', 'char'):
                        name_level = '{}_{}'.format(
                            feature[NAME],
                            level)
                        if name_level in dataset:
                            del dataset[name_level]


def get_preprocessing_params(model_definition):
    model_definition = merge_with_defaults(model_definition)

    global_preprocessing_parameters = model_definition[PREPROCESSING]
    features = (
            model_definition['input_features'] +
            model_definition['output_features']
    )

    global_preprocessing_parameters = merge_dict(
        default_preprocessing_parameters,
        global_preprocessing_parameters
    )

    merged_preprocessing_params = []
    for feature in features:
        if PREPROCESSING in feature:
            local_preprocessing_parameters = merge_dict(
                global_preprocessing_parameters[feature[TYPE]],
                feature[PREPROCESSING]
            )
        else:
            local_preprocessing_parameters = global_preprocessing_parameters[
                feature[TYPE]
            ]
        merged_preprocessing_params.append(
            (feature[NAME], feature[TYPE], local_preprocessing_parameters)
        )

    return merged_preprocessing_params


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
        '--training_set_metadata_json',
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

    dataset_df = read_csv(args.dataset_csv)
    dataset_df.src = args.dataset_csv
    dataset, training_set_metadata = build_dataset(
        dataset_df,
        args.training_set_metadata_json,
        args.features,
        args.preprocessing_parameters,
        args.random_seed
    )

    # write train set metadata, dataset
    logger.info('Writing train set metadata with vocabulary')
    data_utils.save_json(args.output_metadata_json, training_set_metadata)
    logger.info('Writing dataset')
    data_utils.save_hdf5(args.output_dataset_h5, dataset,
                         training_set_metadata)
