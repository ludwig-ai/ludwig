#! /usr/bin/env python
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
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ludwig.backend import Backend, LOCAL_BACKEND
from ludwig.constants import (
    BACKFILL,
    BFILL,
    BINARY,
    CHECKSUM,
    COLUMN,
    DROP_ROW,
    FFILL,
    FILL_WITH_CONST,
    FILL_WITH_FALSE,
    FILL_WITH_MEAN,
    FILL_WITH_MODE,
    FULL,
    NAME,
    NUMBER,
    PAD,
    PREPROCESSING,
    PROC_COLUMN,
    SPLIT,
    SRC,
    TEST,
    TRAINING,
    TYPE,
    VALIDATION,
)
from ludwig.data.cache.types import wrap
from ludwig.data.concatenate_datasets import concatenate_df, concatenate_files
from ludwig.data.dataset.base import Dataset
from ludwig.encoders.registry import get_encoder_cls
from ludwig.features.feature_registries import base_type_registry
from ludwig.features.feature_utils import compute_feature_hash
from ludwig.utils import data_utils, strings_utils
from ludwig.utils.data_utils import (
    CACHEABLE_FORMATS,
    CSV_FORMATS,
    DATA_TRAIN_HDF5_FP,
    DATAFRAME_FORMATS,
    DICT_FORMATS,
    EXCEL_FORMATS,
    FEATHER_FORMATS,
    figure_data_format,
    FWF_FORMATS,
    get_split_path,
    HDF5_FORMATS,
    HTML_FORMATS,
    JSON_FORMATS,
    JSONL_FORMATS,
    ORC_FORMATS,
    override_in_memory_flag,
    PARQUET_FORMATS,
    PICKLE_FORMATS,
    read_csv,
    read_excel,
    read_feather,
    read_fwf,
    read_html,
    read_json,
    read_jsonl,
    read_orc,
    read_parquet,
    read_pickle,
    read_sas,
    read_spss,
    read_stata,
    read_tsv,
    SAS_FORMATS,
    save_array,
    split_dataset_ttv,
    SPSS_FORMATS,
    STATA_FORMATS,
    TSV_FORMATS,
    use_credentials,
)
from ludwig.utils.defaults import default_preprocessing_parameters, default_random_seed
from ludwig.utils.fs_utils import file_lock, path_exists
from ludwig.utils.misc_utils import get_from_registry, merge_dict, resolve_pointers, set_random_seed
from ludwig.utils.type_utils import Column
from ludwig.utils.types import DataFrame

logger = logging.getLogger(__name__)


class DataFormatPreprocessor(ABC):
    @staticmethod
    @abstractmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        pass

    @staticmethod
    @abstractmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        pass

    @staticmethod
    @abstractmethod
    def prepare_processed_data(
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
    ):
        pass


class DictPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        num_overrides = override_in_memory_flag(features, True)
        if num_overrides > 0:
            logger.warning("Using in_memory = False is not supported " "with {} data format.".format("dict"))

        df_engine = backend.df_engine
        if dataset is not None:
            dataset = df_engine.from_pandas(pd.DataFrame(dataset))
        if training_set is not None:
            training_set = df_engine.from_pandas(pd.DataFrame(training_set))
        if validation_set is not None:
            validation_set = df_engine.from_pandas(pd.DataFrame(validation_set))
        if test_set is not None:
            test_set = df_engine.from_pandas(pd.DataFrame(test_set))

        return _preprocess_df_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            training_set_metadata=training_set_metadata,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset, training_set_metadata = build_dataset(
            pd.DataFrame(dataset),
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class DataFramePreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        num_overrides = override_in_memory_flag(features, True)
        if num_overrides > 0:
            logger.warning("Using in_memory = False is not supported " "with {} data format.".format("dataframe"))

        if isinstance(dataset, pd.DataFrame):
            dataset = backend.df_engine.from_pandas(dataset)

        return _preprocess_df_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            training_set_metadata=training_set_metadata,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        if isinstance(dataset, pd.DataFrame):
            dataset = backend.df_engine.from_pandas(dataset)

        dataset, training_set_metadata = build_dataset(
            dataset,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class CSVPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_csv,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_csv(dataset, df_lib=backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class TSVPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_tsv,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_tsv(dataset, df_lib=backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class JSONPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_json,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_json(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class JSONLPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_jsonl,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_jsonl(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class ExcelPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_excel,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_excel(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class ParquetPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_parquet,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_parquet(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None

    @staticmethod
    def prepare_processed_data(
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
    ):
        test_set = test_set if test_set and path_exists(test_set) else None
        validation_set = validation_set if validation_set and path_exists(validation_set) else None
        return training_set, test_set, validation_set, training_set_metadata


class PicklePreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_pickle,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_pickle(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class FatherPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_feather,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_feather(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class FWFPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_fwf,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_fwf(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class HTMLPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_html,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_html(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class ORCPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_orc,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_orc(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class SASPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_sas,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_sas(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class SPSSPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_spss,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_spss(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class StataPreprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return _preprocess_file_for_training(
            config,
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            read_fn=read_stata,
            training_set_metadata=training_set_metadata,
            skip_save_processed_input=skip_save_processed_input,
            preprocessing_params=preprocessing_params,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        dataset_df = read_stata(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset
        dataset, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            callbacks=callbacks,
            mode="prediction",
        )
        return dataset, training_set_metadata, None


class HDF5Preprocessor(DataFormatPreprocessor):
    @staticmethod
    def preprocess_for_training(
        config,
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
        callbacks=None,
    ):
        return HDF5Preprocessor.prepare_processed_data(
            features,
            dataset,
            training_set,
            validation_set,
            test_set,
            training_set_metadata,
            skip_save_processed_input,
            preprocessing_params,
            backend,
            random_seed,
        )

    @staticmethod
    def preprocess_for_prediction(dataset, features, preprocessing_params, training_set_metadata, backend, callbacks):
        hdf5_fp = dataset
        dataset = load_hdf5(dataset, features, split_data=False, shuffle_training=False)
        return dataset, training_set_metadata, hdf5_fp

    @staticmethod
    def prepare_processed_data(
        features,
        dataset=None,
        training_set=None,
        validation_set=None,
        test_set=None,
        training_set_metadata=None,
        skip_save_processed_input=False,
        preprocessing_params=default_preprocessing_parameters,
        backend=LOCAL_BACKEND,
        random_seed=default_random_seed,
    ):
        if dataset is None and training_set is None:
            raise ValueError("One of `dataset` or `training_set` must be not None")
        not_none_set = dataset if dataset is not None else training_set

        if not training_set_metadata:
            raise ValueError("When providing HDF5 data, " "training_set_metadata must not be None.")

        logger.info("Using full hdf5 and json")

        if DATA_TRAIN_HDF5_FP not in training_set_metadata:
            logger.warning(
                "data_train_hdf5_fp not present in training_set_metadata. "
                "Adding it with the current HDF5 file path {}".format(not_none_set)
            )
            training_set_metadata[DATA_TRAIN_HDF5_FP] = not_none_set

        elif training_set_metadata[DATA_TRAIN_HDF5_FP] != not_none_set:
            logger.warning(
                "data_train_hdf5_fp in training_set_metadata is {}, "
                "different from the current HDF5 file path {}. "
                "Replacing it".format(training_set_metadata[DATA_TRAIN_HDF5_FP], not_none_set)
            )
            training_set_metadata[DATA_TRAIN_HDF5_FP] = not_none_set

        if dataset is not None:
            training_set, test_set, validation_set = load_hdf5(dataset, features, shuffle_training=True)

        elif training_set is not None:
            kwargs = dict(features=features, split_data=False)
            training_set = load_hdf5(training_set, shuffle_training=True, **kwargs)

            if validation_set is not None:
                validation_set = load_hdf5(validation_set, shuffle_training=False, **kwargs)

            if test_set is not None:
                test_set = load_hdf5(test_set, shuffle_training=False, **kwargs)

        return training_set, test_set, validation_set, training_set_metadata


data_format_preprocessor_registry = {
    **{fmt: DictPreprocessor for fmt in DICT_FORMATS},
    **{fmt: DataFramePreprocessor for fmt in DATAFRAME_FORMATS},
    **{fmt: CSVPreprocessor for fmt in CSV_FORMATS},
    **{fmt: TSVPreprocessor for fmt in TSV_FORMATS},
    **{fmt: JSONPreprocessor for fmt in JSON_FORMATS},
    **{fmt: JSONLPreprocessor for fmt in JSONL_FORMATS},
    **{fmt: ExcelPreprocessor for fmt in EXCEL_FORMATS},
    **{fmt: ParquetPreprocessor for fmt in PARQUET_FORMATS},
    **{fmt: PicklePreprocessor for fmt in PICKLE_FORMATS},
    **{fmt: FWFPreprocessor for fmt in FWF_FORMATS},
    **{fmt: FatherPreprocessor for fmt in FEATHER_FORMATS},
    **{fmt: HTMLPreprocessor for fmt in HTML_FORMATS},
    **{fmt: ORCPreprocessor for fmt in ORC_FORMATS},
    **{fmt: SASPreprocessor for fmt in SAS_FORMATS},
    **{fmt: SPSSPreprocessor for fmt in SPSS_FORMATS},
    **{fmt: StataPreprocessor for fmt in STATA_FORMATS},
    **{fmt: HDF5Preprocessor for fmt in HDF5_FORMATS},
}


def build_dataset(
    dataset_df,
    features,
    global_preprocessing_parameters,
    metadata=None,
    backend=LOCAL_BACKEND,
    random_seed=default_random_seed,
    skip_save_processed_input=False,
    callbacks=None,
    mode=None,
):
    df_engine = backend.df_engine
    dataset_df = df_engine.parallelize(dataset_df)

    sample_ratio = global_preprocessing_parameters["sample_ratio"]
    if sample_ratio < 1.0:
        logger.debug(f"sample {sample_ratio} of data")
        dataset_df = dataset_df.sample(frac=sample_ratio)

    # If persisting DataFrames in memory is enabled, we want to do this after
    # each batch of parallel ops in order to avoid redundant computation
    dataset_df = df_engine.persist(dataset_df)

    global_preprocessing_parameters = merge_dict(default_preprocessing_parameters, global_preprocessing_parameters)

    # Get all the unique preprocessing features to compute
    feature_configs = []
    feature_hashes = set()
    for feature in features:
        if PROC_COLUMN not in feature:
            feature[PROC_COLUMN] = compute_feature_hash(feature)
        if feature[PROC_COLUMN] not in feature_hashes:
            feature_configs.append(feature)
            feature_hashes.add(feature[PROC_COLUMN])

    logger.debug("cast columns")
    dataset_cols = cast_columns(dataset_df, feature_configs, backend)

    for callback in callbacks or []:
        callback.on_build_metadata_start(dataset_df, mode)

    logger.debug("build metadata")
    metadata = build_metadata(metadata, dataset_cols, feature_configs, global_preprocessing_parameters, backend)

    for callback in callbacks or []:
        callback.on_build_metadata_end(dataset_df, mode)

    for callback in callbacks or []:
        callback.on_build_data_start(dataset_df, mode)

    logger.debug("build data")
    proc_cols = build_data(dataset_cols, feature_configs, metadata, backend, skip_save_processed_input)

    for callback in callbacks or []:
        callback.on_build_data_end(dataset_df, mode)

    logger.debug("get split")
    split = get_split(
        dataset_df,
        force_split=global_preprocessing_parameters["force_split"],
        split_probabilities=global_preprocessing_parameters["split_probabilities"],
        stratify=global_preprocessing_parameters["stratify"],
        backend=backend,
        random_seed=random_seed,
    )

    if split is not None:
        proc_cols[SPLIT] = split

    # TODO ray: this is needed because ray 1.7 doesn't support Dask to RayDataset
    #  conversion with Tensor columns. Can remove for 1.8.
    if backend.df_engine.partitioned:
        for feature in features:
            name = feature[NAME]
            proc_column = feature[PROC_COLUMN]
            reshape = metadata[name].get("reshape")
            if reshape is not None:
                proc_cols[proc_column] = backend.df_engine.map_objects(proc_cols[proc_column], lambda x: x.reshape(-1))

    # Implements an outer join of proc_cols
    dataset = backend.df_engine.df_like(dataset_df, proc_cols)

    # At this point, there should be no missing values left in the dataframe, unless
    # the DROP_ROW preprocessing option was selected, in which case we need to drop those
    # rows.
    dataset = dataset.dropna()

    # NaNs introduced by outer join change dtype of dataset cols (upcast to float64), so we need to cast them back.
    col_name_to_dtype = {}
    for col_name, col in proc_cols.items():
        # if col is a list of list-like objects, we assume the internal dtype of each col[i] remains unchanged.
        if type(col) == list and type(col[0]) in {list, np.ndarray, torch.Tensor}:
            continue
        col_name_to_dtype[col_name] = col.dtype
    dataset = dataset.astype(col_name_to_dtype)

    return dataset, metadata


def cast_columns(dataset_df, features, backend) -> Dict[str, DataFrame]:
    """Copies each column of the dataset to a dataframe, with potential type casting."""
    dataset_cols = {}
    for feature in features:
        # todo figure out if additional parameters are needed
        #  for the cast_column function
        try:
            dataset_cols[feature[COLUMN]] = get_from_registry(feature[TYPE], base_type_registry).cast_column(
                dataset_df[feature[COLUMN]], backend
            )
        except KeyError as e:
            raise KeyError(
                f"Feature name {e} specified in the config was not found in dataset with columns: "
                + f"{list(dataset_df.columns)}"
            )

    return dataset_cols


def merge_preprocessing(
    feature_config: Dict[str, Any], global_preprocessing_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    if PREPROCESSING not in feature_config:
        return global_preprocessing_parameters[feature_config[TYPE]]

    return merge_dict(global_preprocessing_parameters[feature_config[TYPE]], feature_config[PREPROCESSING])


def build_metadata(
    metadata: Dict[str, Any],
    dataset_cols: Dict[str, Column],
    feature_configs: List[Dict[str, Any]],
    global_preprocessing_parameters: Dict[str, Any],
    backend: Backend,
) -> Dict[str, Any]:
    for feature_config in feature_configs:
        if feature_config[NAME] in metadata:
            continue

        preprocessing_parameters = merge_preprocessing(feature_config, global_preprocessing_parameters)

        # deal with encoders that have fixed preprocessing
        if "encoder" in feature_config:
            encoder_class = get_encoder_cls(feature_config[TYPE], feature_config["encoder"])
            if hasattr(encoder_class, "fixed_preprocessing_parameters"):
                encoder_fpp = encoder_class.fixed_preprocessing_parameters

                preprocessing_parameters = merge_dict(
                    preprocessing_parameters, resolve_pointers(encoder_fpp, feature_config, "feature.")
                )

        fill_value = precompute_fill_value(dataset_cols, feature_config, preprocessing_parameters, backend)

        if fill_value is not None:
            preprocessing_parameters = {"computed_fill_value": fill_value, **preprocessing_parameters}

        handle_missing_values(dataset_cols, feature_config, preprocessing_parameters)

        column = dataset_cols[feature_config[COLUMN]]
        if column.dtype == object:
            column = column.astype(str)

        metadata[feature_config[NAME]] = get_from_registry(feature_config[TYPE], base_type_registry).get_feature_meta(
            column, preprocessing_parameters, backend
        )

        metadata[feature_config[NAME]][PREPROCESSING] = preprocessing_parameters

    return metadata


def build_data(
    input_cols: DataFrame,
    feature_configs: List[Dict],
    training_set_metadata: Dict,
    backend: Backend,
    skip_save_processed_input: bool,
) -> Dict[str, DataFrame]:
    """Preprocesses the input dataframe columns, handles missing values, and potentially adds metadata to
    training_set_metadata.

    Args:
        input_cols: Input dataframe to be processed.
        feature_configs: List of feature configs.
        training_set_metadata: Training set metadata. Additional fields may be added.
        backend: Backend for data processing.
        skip_save_processed_input: (bool) Whether to skip saving the processed input.

    Returns:
        Dictionary of (feature name) -> (processed data).
    """
    proc_cols = {}
    for feature_config in feature_configs:
        preprocessing_parameters = training_set_metadata[feature_config[NAME]][PREPROCESSING]
        handle_missing_values(input_cols, feature_config, preprocessing_parameters)
        get_from_registry(feature_config[TYPE], base_type_registry).add_feature_data(
            feature_config,
            input_cols,
            proc_cols,
            training_set_metadata,
            preprocessing_parameters,
            backend,
            skip_save_processed_input,
        )

    return proc_cols


def balance_data(dataset_df: DataFrame, output_features: List[Dict], preprocessing_parameters: Dict, backend: Backend):
    """The purpose of this function is to balance the training dataset using either over-sampling or under-
    sampling.

    Args:
        dataset_df: Input dataframe to be over-sampled or under-sampled.
        output_features: List of feature configs.
        preprocessing_parameters: Dictionary of the global preprocessing parameters.
        backend: Backend for data processing.

    Returns: An over-sampled or under-sampled training dataset.
    """

    if len(output_features) != 1:
        raise ValueError("Class balancing is only available for datasets with a single output feature")
    if output_features[0][TYPE] != BINARY:
        raise ValueError("Class balancing is only supported for binary output types")

    target = output_features[0][PROC_COLUMN]

    if backend.df_engine.partitioned:
        majority_class = backend.df_engine.compute(dataset_df[target].value_counts()).idxmax()
        minority_class = backend.df_engine.compute(dataset_df[target].value_counts()).idxmin()
    else:
        majority_class = dataset_df[target].value_counts().idxmax()
        minority_class = dataset_df[target].value_counts().idxmin()
    majority_df = dataset_df[dataset_df[target] == majority_class]
    minority_df = dataset_df[dataset_df[target] == minority_class]

    if preprocessing_parameters["oversample_minority"] and preprocessing_parameters["undersample_majority"]:
        raise ValueError(
            "Cannot balance data if both oversampling an undersampling are specified in the config. "
            "Must specify only one method"
        )

    if preprocessing_parameters["oversample_minority"]:
        sample_fraction = (len(majority_df) * preprocessing_parameters["oversample_minority"]) / len(minority_df)
        minority_df = minority_df.sample(frac=sample_fraction, replace=True)
    elif preprocessing_parameters["undersample_majority"]:
        sample_fraction = int(len(minority_df) / preprocessing_parameters["undersample_majority"]) / len(majority_df)
        majority_df = majority_df.sample(frac=sample_fraction, replace=False)

    balanced_df = backend.df_engine.concat([minority_df, majority_df])

    return balanced_df


def precompute_fill_value(dataset_cols, feature, preprocessing_parameters, backend):
    missing_value_strategy = preprocessing_parameters["missing_value_strategy"]
    if missing_value_strategy == FILL_WITH_CONST:
        return preprocessing_parameters["fill_value"]
    elif missing_value_strategy == FILL_WITH_MODE:
        return dataset_cols[feature[COLUMN]].value_counts().index[0]
    elif missing_value_strategy == FILL_WITH_MEAN:
        if feature[TYPE] != NUMBER:
            raise ValueError(
                f"Filling missing values with mean is supported "
                f"only for number types, not for type {feature[TYPE]}.",
            )
        return backend.df_engine.compute(dataset_cols[feature[COLUMN]].mean())
    elif missing_value_strategy == FILL_WITH_FALSE:
        distinct_values = backend.df_engine.compute(
            dataset_cols[feature[COLUMN]].drop_duplicates().dropna()
        ).values.tolist()
        if len(distinct_values) > 2:
            raise ValueError(
                f"Missing value strategy `fill_with_false` "
                f"for column {feature[COLUMN]} expects 2 distinct values, "
                f"found: {len(distinct_values)} (ex: {distinct_values[:10]})"
            )

        # Determine the False label.
        # Distinct values are sorted in reverse to mirror the selection of the default fallback_true_label (in
        # binary_feature.get_feature_meta) for binary columns with unconventional boolean values, "human"/"bot".
        for v in sorted(distinct_values, reverse=True):
            fallback_true_label = preprocessing_parameters.get("fallback_true_label", "true")
            if strings_utils.str2bool(v, fallback_true_label) is False:
                return v
        raise ValueError(
            f"Unable to determine False value for column {feature[COLUMN]} with distinct values: {distinct_values}."
        )
    # Otherwise, we cannot precompute the fill value for this dataset
    return None


def handle_missing_values(dataset_cols, feature, preprocessing_parameters):
    missing_value_strategy = preprocessing_parameters["missing_value_strategy"]

    # Check for the precomputed fill value in the metadata
    computed_fill_value = preprocessing_parameters.get("computed_fill_value")

    if computed_fill_value is not None:
        dataset_cols[feature[COLUMN]] = dataset_cols[feature[COLUMN]].fillna(
            computed_fill_value,
        )
    elif missing_value_strategy in [BACKFILL, BFILL, PAD, FFILL]:
        dataset_cols[feature[COLUMN]] = dataset_cols[feature[COLUMN]].fillna(
            method=missing_value_strategy,
        )
    elif missing_value_strategy == DROP_ROW:
        # Here we only drop from this series, but after preprocessing we'll do a second
        # round of dropping NA values from the entire output dataframe, which will
        # result in the removal of the rows.
        dataset_cols[feature[COLUMN]] = dataset_cols[feature[COLUMN]].dropna()
    else:
        raise ValueError("Invalid missing value strategy")


def get_split(
    dataset_df,
    force_split=False,
    split_probabilities=(0.7, 0.1, 0.2),
    stratify=None,
    backend=LOCAL_BACKEND,
    random_seed=default_random_seed,
):
    if SPLIT in dataset_df and not force_split:
        split = dataset_df[SPLIT]
    else:
        set_random_seed(random_seed)
        if stratify is None or stratify not in dataset_df:
            if backend.df_engine.partitioned:
                # This approach is very inefficient for partitioned backends, which
                # can split by partition
                return

            split = (
                dataset_df.index.to_series()
                .map(lambda x: np.random.choice(3, 1, p=split_probabilities))
                .astype(np.int8)
            )
        else:
            split = np.zeros(len(dataset_df))
            for val in dataset_df[stratify].unique():
                # TODO dask: find a way to better parallelize this operation
                idx_list = dataset_df.index[dataset_df[stratify] == val].tolist()
                array_lib = backend.df_engine.array_lib
                val_list = array_lib.random.choice(
                    3,
                    len(idx_list),
                    p=split_probabilities,
                ).astype(np.int8)
                split[idx_list] = val_list
    return split


def load_hdf5(hdf5_file_path, features, split_data=True, shuffle_training=False):
    # TODO dask: this needs to work with DataFrames
    logger.info(f"Loading data from: {hdf5_file_path}")

    def shuffle(df):
        return df.sample(frac=1).reset_index(drop=True)

    dataset = data_utils.load_hdf5(hdf5_file_path)
    if not split_data:
        if shuffle_training:
            dataset = shuffle(dataset)
        return dataset

    training_set, test_set, validation_set = split_dataset_ttv(dataset, SPLIT)

    if shuffle_training:
        training_set = shuffle(training_set)

    return training_set, test_set, validation_set


def load_metadata(metadata_file_path):
    logger.info(f"Loading metadata from: {metadata_file_path}")
    return data_utils.load_json(metadata_file_path)


def preprocess_for_training(
    config,
    dataset=None,
    training_set=None,
    validation_set=None,
    test_set=None,
    training_set_metadata=None,
    data_format=None,
    skip_save_processed_input=False,
    preprocessing_params=default_preprocessing_parameters,
    backend=LOCAL_BACKEND,
    random_seed=default_random_seed,
    callbacks=None,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, Any]]:
    """Returns training, val and test datasets with training set metadata."""

    # sanity check to make sure some data source is provided
    if dataset is None and training_set is None:
        raise ValueError("No training data is provided!")

    # determine data format if not provided or auto
    if not data_format or data_format == "auto":
        data_format = figure_data_format(dataset, training_set, validation_set, test_set)

    # Wrap dataset into a form we can use to manage within the cache
    dataset = wrap(dataset)
    training_set = wrap(training_set)
    validation_set = wrap(validation_set)
    test_set = wrap(test_set)

    try:
        lock_path = backend.cache.get_cache_directory(dataset)
    except (TypeError, ValueError):
        lock_path = None
    with file_lock(lock_path, lock_file=".lock_preprocessing"):
        # if training_set_metadata is a string, assume it's a path to load the json
        training_set_metadata = training_set_metadata or {}
        if training_set_metadata and isinstance(training_set_metadata, str):
            training_set_metadata = load_metadata(training_set_metadata)

        # setup
        features = config["input_features"] + config["output_features"]

        # in case data_format is one of the cacheable formats,
        # check if there's a cached hdf5 file with the same name,
        # and in case move on with the hdf5 branch.
        cached = False
        cache = backend.cache.get_dataset_cache(config, dataset, training_set, test_set, validation_set)

        # Unwrap dataset into the form used for preprocessing
        dataset = dataset.unwrap() if dataset is not None else None
        training_set = training_set.unwrap() if training_set is not None else None
        validation_set = validation_set.unwrap() if validation_set is not None else None
        test_set = test_set.unwrap() if test_set is not None else None

        if data_format in CACHEABLE_FORMATS:
            with use_credentials(backend.cache.credentials):
                cache_results = cache.get()
                if cache_results is not None:
                    valid, *cache_values = cache_results
                    if valid:
                        logger.info(
                            "Found cached dataset and meta.json with the same filename "
                            "of the dataset, using them instead"
                        )
                        training_set_metadata, training_set, test_set, validation_set = cache_values
                        config["data_hdf5_fp"] = training_set
                        data_format = backend.cache.data_format
                        cached = True
                        dataset = None
                    else:
                        logger.info(
                            "Found cached dataset and meta.json with the same filename "
                            "of the dataset, but checksum don't match, "
                            "if saving of processed input is not skipped "
                            "they will be overridden"
                        )
                        cache.delete()

        training_set_metadata[CHECKSUM] = cache.checksum
        data_format_processor = get_from_registry(data_format, data_format_preprocessor_registry)

        if cached or data_format == "hdf5":
            with use_credentials(backend.cache.credentials):
                # Always interpret hdf5 files as preprocessed, even if missing from the cache
                processed = data_format_processor.prepare_processed_data(
                    features,
                    dataset=dataset,
                    training_set=training_set,
                    validation_set=validation_set,
                    test_set=test_set,
                    training_set_metadata=training_set_metadata,
                    skip_save_processed_input=skip_save_processed_input,
                    preprocessing_params=preprocessing_params,
                    backend=backend,
                    random_seed=random_seed,
                )
                training_set, test_set, validation_set, training_set_metadata = processed
        else:
            processed = data_format_processor.preprocess_for_training(
                config,
                features,
                dataset=dataset,
                training_set=training_set,
                validation_set=validation_set,
                test_set=test_set,
                training_set_metadata=training_set_metadata,
                skip_save_processed_input=skip_save_processed_input,
                preprocessing_params=preprocessing_params,
                backend=backend,
                random_seed=random_seed,
                callbacks=callbacks,
            )
            training_set, test_set, validation_set, training_set_metadata = processed
            processed = (training_set, test_set, validation_set, training_set_metadata)

            # cache the dataset
            if backend.cache.can_cache(skip_save_processed_input):
                with use_credentials(backend.cache.credentials):
                    logger.debug("cache processed data")
                    processed = cache.put(*processed)
                    # set cached=True to ensure credentials are used correctly below
                    cached = True
            training_set, test_set, validation_set, training_set_metadata = processed

        with use_credentials(backend.cache.credentials if cached else None):
            logger.debug("create training dataset")
            training_dataset = backend.dataset_manager.create(training_set, config, training_set_metadata)

            validation_dataset = None
            if validation_set is not None:
                logger.debug("create validation dataset")
                validation_dataset = backend.dataset_manager.create(validation_set, config, training_set_metadata)

            test_dataset = None
            if test_set is not None:
                logger.debug("create test dataset")
                test_dataset = backend.dataset_manager.create(test_set, config, training_set_metadata)

        return (training_dataset, validation_dataset, test_dataset, training_set_metadata)


def _preprocess_file_for_training(
    config,
    features,
    dataset=None,
    training_set=None,
    validation_set=None,
    test_set=None,
    training_set_metadata=None,
    read_fn=read_csv,
    skip_save_processed_input=False,
    preprocessing_params=default_preprocessing_parameters,
    backend=LOCAL_BACKEND,
    random_seed=default_random_seed,
    callbacks=None,
):
    """Method to pre-process csv data.

    :param features: list of all features (input + output)
    :param dataset: path to the data
    :param training_set:  training data
    :param validation_set: validation data
    :param test_set: test data
    :param training_set_metadata: train set metadata
    :param skip_save_processed_input: if False, the pre-processed data is saved
    as .hdf5 files in the same location as the csv files with the same names.
    :param preprocessing_params: preprocessing parameters
    :param random_seed: random seed
    :return: training, test, validation datasets, training metadata
    """
    if dataset:
        # Use data and ignore _train, _validation and _test.
        # Also ignore data and train set metadata needs preprocessing
        logger.info("Using full raw dataset, no hdf5 and json file " "with the same name have been found")
        logger.info("Building dataset (it may take a while)")

        dataset_df = read_fn(dataset, backend.df_engine.df_lib)
        training_set_metadata[SRC] = dataset

        data, training_set_metadata = build_dataset(
            dataset_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            random_seed=random_seed,
            skip_save_processed_input=skip_save_processed_input,
            callbacks=callbacks,
            mode="training",
        )

        # TODO(travis): implement saving split for Ray
        if backend.is_coordinator() and not skip_save_processed_input and SPLIT in data.columns:
            # save split values for use by visualization routines
            split_fp = get_split_path(dataset)
            save_array(split_fp, data[SPLIT])

    elif training_set:
        # use data_train (including _validation and _test if they are present)
        # and ignore data and train set metadata
        # needs preprocessing
        logger.info("Using training raw csv, no hdf5 and json " "file with the same name have been found")
        logger.info("Building dataset (it may take a while)")

        concatenated_df = concatenate_files(training_set, validation_set, test_set, read_fn, backend)
        training_set_metadata[SRC] = training_set

        data, training_set_metadata = build_dataset(
            concatenated_df,
            features,
            preprocessing_params,
            metadata=training_set_metadata,
            backend=backend,
            random_seed=random_seed,
            callbacks=callbacks,
            mode="training",
        )

    else:
        raise ValueError("either data or data_train have to be not None")

    logger.info("Building dataset: DONE")

    if SPLIT in data.columns:
        logger.debug("split on split column")
        training_data, test_data, validation_data = split_dataset_ttv(data, SPLIT)
    else:
        logger.debug("split randomly by partition")
        training_data, test_data, validation_data = data.random_split(preprocessing_params["split_probabilities"])

    if preprocessing_params["oversample_minority"] or preprocessing_params["undersample_majority"]:
        training_data = balance_data(training_data, config["output_features"], preprocessing_params, backend)

    return training_data, test_data, validation_data, training_set_metadata


def _preprocess_df_for_training(
    config,
    features,
    dataset=None,
    training_set=None,
    validation_set=None,
    test_set=None,
    training_set_metadata=None,
    preprocessing_params=default_preprocessing_parameters,
    backend=LOCAL_BACKEND,
    random_seed=default_random_seed,
    callbacks=None,
):
    """Method to pre-process dataframes.

    This doesn't have the option to save the processed data as hdf5 as we don't expect users to do this as the data can
    be processed in memory
    """
    if dataset is not None:
        # needs preprocessing
        logger.info("Using full dataframe")
    elif training_set is not None:
        # needs preprocessing
        logger.info("Using training dataframe")
        dataset = concatenate_df(training_set, validation_set, test_set, backend)
    logger.info("Building dataset (it may take a while)")

    dataset, training_set_metadata = build_dataset(
        dataset,
        features,
        preprocessing_params,
        metadata=training_set_metadata,
        random_seed=random_seed,
        backend=backend,
        callbacks=callbacks,
        mode="training",
    )

    logger.info("Building dataset: DONE")

    if SPLIT in dataset.columns:
        logger.debug("split on split column")
        training_set, test_set, validation_set = split_dataset_ttv(dataset, SPLIT)
    else:
        logger.debug("split randomly by partition")
        training_set, test_set, validation_set = dataset.random_split(preprocessing_params["split_probabilities"])

    if preprocessing_params["oversample_minority"] or preprocessing_params["undersample_majority"]:
        training_set = balance_data(training_set, config["output_features"], preprocessing_params, backend)

    return training_set, test_set, validation_set, training_set_metadata


def preprocess_for_prediction(
    config,
    dataset,
    training_set_metadata=None,
    data_format=None,
    split=FULL,
    include_outputs=True,
    backend=LOCAL_BACKEND,
    callbacks=None,
):
    """Preprocesses the dataset to parse it into a format that is usable by the Ludwig core.

    :param model_path: The input data that is joined with the model
           hyperparameter file to create the config file
    :param data_csv: The CSV input data file
    :param data_hdf5: The hdf5 data file if there is no csv data file
    :param training_set_metadata: Train set metadata for the input features
    :param split: the split of dataset to return
    :returns: Dataset, Train set metadata
    """
    # Sanity Check to make sure some data source is provided
    if dataset is None:
        raise ValueError("No training data is provided!")

    if isinstance(dataset, Dataset):
        return dataset, training_set_metadata

    # determine data format if not provided or auto
    if not data_format or data_format == "auto":
        data_format = figure_data_format(dataset)

    # manage the in_memory parameter
    if data_format not in HDF5_FORMATS:
        num_overrides = override_in_memory_flag(config["input_features"], True)
        if num_overrides > 0:
            logger.warning("Using in_memory = False is not supported " "with {} data format.".format(data_format))

    preprocessing_params = merge_dict(default_preprocessing_parameters, config[PREPROCESSING])

    # if training_set_metadata is a string, assume it's a path to load the json
    if training_set_metadata and isinstance(training_set_metadata, str):
        training_set_metadata = load_metadata(training_set_metadata)

    # setup
    output_features = []
    if include_outputs:
        output_features += config["output_features"]
    features = config["input_features"] + output_features

    # Check the cache for an already preprocessed dataset. This only
    # applies to scenarios where the user wishes to predict on a split
    # of the full dataset, where we preprocess the whole dataset together
    # during training. If the user wishes to predict on the full dataset,
    # it is assumed they are predicting on unseen data. This is done
    # because the cached data is stored in its split form, and would be
    # expensive to recombine, requiring further caching.
    cached = False

    dataset = wrap(dataset)
    cache = backend.cache.get_dataset_cache(config, dataset)
    dataset = dataset.unwrap()

    training_set = test_set = validation_set = None
    if data_format in CACHEABLE_FORMATS and split != FULL:
        with use_credentials(backend.cache.credentials):
            cache_results = cache.get()
            if cache_results is not None:
                valid, *cache_values = cache_results
                if valid:
                    logger.info(
                        "Found cached dataset and meta.json with the same filename "
                        "of the input file, using them instead"
                    )
                    training_set_metadata, training_set, test_set, validation_set = cache_values
                    config["data_hdf5_fp"] = training_set
                    data_format = backend.cache.data_format
                    cached = True

    data_format_processor = get_from_registry(data_format, data_format_preprocessor_registry)
    if cached:
        with use_credentials(backend.cache.credentials):
            processed = data_format_processor.prepare_processed_data(
                features,
                dataset=dataset,
                training_set=training_set,
                validation_set=validation_set,
                test_set=test_set,
                training_set_metadata=training_set_metadata,
                preprocessing_params=preprocessing_params,
                backend=backend,
            )
            training_set, test_set, validation_set, training_set_metadata = processed
    else:
        processed = data_format_processor.preprocess_for_prediction(
            dataset, features, preprocessing_params, training_set_metadata, backend, callbacks
        )
        dataset, training_set_metadata, new_hdf5_fp = processed
        training_set_metadata = training_set_metadata.copy()

        if new_hdf5_fp:
            training_set_metadata[DATA_TRAIN_HDF5_FP] = new_hdf5_fp

        if split != FULL:
            training_set, test_set, validation_set = split_dataset_ttv(dataset, SPLIT)

    if split == TRAINING:
        dataset = training_set
    elif split == VALIDATION:
        dataset = validation_set
    elif split == TEST:
        dataset = test_set

    config = {
        **config,
        "output_features": output_features,
    }

    with use_credentials(backend.cache.credentials if cached else None):
        dataset = backend.dataset_manager.create(
            dataset,
            config,
            training_set_metadata,
        )

    return dataset, training_set_metadata
