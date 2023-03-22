#! /usr/bin/env python
# Copyright (c) 2022 Predibase, Inc.
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

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, TYPE_CHECKING
from zlib import crc32

import numpy as np
from sklearn.model_selection import train_test_split

from ludwig.api_annotations import DeveloperAPI
from ludwig.backend.base import Backend
from ludwig.constants import BINARY, CATEGORY, DATE, MIN_DATASET_SPLIT_ROWS, SPLIT
from ludwig.error import ConfigValidationError
from ludwig.schema.split import (
    DateTimeSplitConfig,
    FixedSplitConfig,
    HashSplitConfig,
    RandomSplitConfig,
    StratifySplitConfig,
)
from ludwig.types import ModelConfigDict, PreprocessingConfigDict
from ludwig.utils.data_utils import hash_dict, split_dataset_ttv
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.registry import Registry
from ludwig.utils.types import DataFrame

if TYPE_CHECKING:
    from ludwig.schema.model_config import ModelConfig

split_registry = Registry()
logger = logging.getLogger(__name__)

TMP_SPLIT_COL = "__SPLIT__"
DEFAULT_PROBABILITIES = (0.7, 0.1, 0.2)


class Splitter(ABC):
    @abstractmethod
    def split(
        self, df: DataFrame, backend: Backend, random_seed: int = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        pass

    def validate(self, config: ModelConfigDict):
        pass

    def has_split(self, split_index: int) -> bool:
        return True

    @property
    def required_columns(self) -> List[str]:
        """Returns the list of columns that are required for splitting."""
        return []


def _make_divisions_ensure_minimum_rows(
    divisions: List[int],
    n_examples: int,
    min_val_rows: int = MIN_DATASET_SPLIT_ROWS,
    min_test_rows: int = MIN_DATASET_SPLIT_ROWS,
) -> List[int]:
    """Revises divisions to ensure no dataset split has too few examples."""
    result = list(divisions)
    n = [dn - dm for dm, dn in zip((0,) + divisions, divisions + (n_examples,))]  # Number of examples in each split.
    if 0 < n[2] < min_test_rows and n[0] > 0:
        # Test set is nonempty but too small, take examples from training set.
        shift = min(min_test_rows - n[2], n[0])
        result = [d - shift for d in result]
    if 0 < n[1] < min_val_rows and n[0] > 0:
        # Validation set is nonempty but too small, take examples from training set.
        result[0] -= min(min_val_rows - n[1], result[0])
    return result


def _split_divisions_with_min_rows(n_rows: int, probabilities: List[float]) -> List[int]:
    """Generates splits for a dataset of n_rows into train, validation, and test sets according to split
    probabilities, also ensuring that at least min_val_rows or min_test_rows are present in each nonempty split.

    Returns division indices to split on.
    """
    d1 = int(np.ceil(probabilities[0] * n_rows))
    if probabilities[-1] > 0:
        n2 = int(probabilities[1] * n_rows)
        d2 = d1 + n2
    else:
        # If the last probability is 0, then use the entire remaining dataset for validation.
        d2 = n_rows
    return _make_divisions_ensure_minimum_rows((d1, d2), n_rows)


@split_registry.register("random", default=True)
class RandomSplitter(Splitter):
    def __init__(self, probabilities: List[float] = DEFAULT_PROBABILITIES, **kwargs):
        self.probabilities = probabilities

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        probabilities = self.probabilities
        if not backend.df_engine.partitioned:
            divisions = _split_divisions_with_min_rows(len(df), probabilities)
            shuffled_df = df.sample(frac=1, random_state=random_seed)
            return (
                shuffled_df.iloc[: divisions[0]],  # Train
                shuffled_df.iloc[divisions[0] : divisions[1]],  # Validation
                shuffled_df.iloc[divisions[1] :],  # Test
            )

        # The above approach is very inefficient for partitioned backends, which can split by partition.
        # This does not give exact guarantees on split size but is much more efficient for large datasets.
        return df.random_split(self.probabilities, random_state=random_seed)

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0

    @staticmethod
    def get_schema_cls():
        return RandomSplitConfig


@split_registry.register("fixed")
class FixedSplitter(Splitter):
    def __init__(self, column: str = SPLIT, **kwargs):
        self.column = column

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        df[self.column] = df[self.column].astype(np.int8)
        dfs = split_dataset_ttv(df, self.column)
        train, test, val = tuple(df.drop(columns=self.column) if df is not None else None for df in dfs)
        return train, val, test

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    @staticmethod
    def get_schema_cls():
        return FixedSplitConfig


def stratify_split_dataframe(
    df: DataFrame, column: str, probabilities: List[float], backend: Backend, random_seed: float
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Splits a dataframe into train, validation, and test sets based on the values of a column.

    The column must be categorical (including binary). The split is stratified, meaning that the proportion of each
    category in each split is the same as in the original dataset.
    """

    frac_train, frac_val, frac_test = probabilities

    def _safe_stratify(df, column, test_size):
        # Get the examples with cardinality of 1
        df_cadinalities = df.groupby(column)[column].size()
        low_cardinality_elems = df_cadinalities.loc[lambda x: x == 1]
        df_low_card = df[df[column].isin(low_cardinality_elems.index)]
        df = df[~df[column].isin(low_cardinality_elems.index)]
        y = df[[column]]

        df_train, df_temp, _, _ = train_test_split(df, y, stratify=y, test_size=test_size, random_state=random_seed)

        # concat the examples with cardinality of 1 to the training DF.
        if len(df_low_card.index) > 0:
            df_train = backend.df_engine.concat([df_train, df_low_card])

        return df_train, df_temp

    df_train, df_temp = _safe_stratify(df, column, 1.0 - frac_train)

    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test = _safe_stratify(df_temp, column, relative_frac_test)

    return df_train, df_val, df_test


@split_registry.register("stratify")
class StratifySplitter(Splitter):
    def __init__(self, column: str, probabilities: List[float] = DEFAULT_PROBABILITIES, **kwargs):
        self.column = column
        self.probabilities = probabilities

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        if not backend.df_engine.partitioned:
            return stratify_split_dataframe(df, self.column, self.probabilities, backend, random_seed)

        # For a partitioned dataset, we can stratify split each partition individually
        # to obtain a global stratified split.

        def split_partition(partition: DataFrame) -> DataFrame:
            """Splits a single partition into train, val, test.

            Returns a single DataFrame with the split column populated. Assumes that the split column is already present
            in the partition and has a default value of 0 (train).
            """
            _, val, test = stratify_split_dataframe(partition, self.column, self.probabilities, backend, random_seed)
            # Split column defaults to train, so only need to update val and test
            partition.loc[val.index, TMP_SPLIT_COL] = 1
            partition.loc[test.index, TMP_SPLIT_COL] = 2
            return partition

        df[TMP_SPLIT_COL] = 0
        df = backend.df_engine.map_partitions(df, split_partition, meta=df)

        df_train = df[df[TMP_SPLIT_COL] == 0].drop(columns=TMP_SPLIT_COL)
        df_val = df[df[TMP_SPLIT_COL] == 1].drop(columns=TMP_SPLIT_COL)
        df_test = df[df[TMP_SPLIT_COL] == 2].drop(columns=TMP_SPLIT_COL)

        return df_train, df_val, df_test

    def validate(self, config: "ModelConfig"):  # noqa: F821
        features = [f for f in config.input_features] + [f for f in config.output_features]
        feature_cols = {f.column for f in features}
        if self.column not in feature_cols:
            logging.info(
                f"Stratify column {self.column} is not among the features. "
                f"Cannot establish if it is a binary or category feature."
            )
        elif [f for f in features if f.column == self.column][0].type not in {BINARY, CATEGORY}:
            raise ConfigValidationError(f"Feature for stratify column {self.column} must be binary or category")

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    @staticmethod
    def get_schema_cls():
        return StratifySplitConfig


@split_registry.register("datetime")
class DatetimeSplitter(Splitter):
    def __init__(
        self,
        column: str,
        probabilities: List[float] = DEFAULT_PROBABILITIES,
        datetime_format: Optional[str] = None,
        fill_value: str = "",
        **kwargs,
    ):
        self.column = column
        self.probabilities = probabilities
        self.datetime_format = datetime_format
        self.fill_value = fill_value

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # In case the split column was preprocessed by Ludwig into a list, convert it back to a
        # datetime string for the sort and split
        def list_to_date_str(x):
            if not isinstance(x, list):
                if not isinstance(x, str):
                    # Convert timestamps, etc. to strings and return so it can direct cast to epoch time
                    return str(x)

                if len(x) != 9:
                    # Strings not in the expected format, so assume it's a formatted datetime and return
                    return x

            return f"{x[0]}-{x[1]}-{x[2]} {x[5]}:{x[6]}:{x[7]}"

        df[TMP_SPLIT_COL] = backend.df_engine.map_objects(df[self.column], list_to_date_str)

        # Convert datetime to int64 to workaround Dask limitation
        # https://github.com/dask/dask/issues/9003
        df[TMP_SPLIT_COL] = backend.df_engine.df_lib.to_datetime(df[TMP_SPLIT_COL]).values.astype("int64")

        # Sort by ascending datetime and drop the temporary column
        df = df.sort_values(TMP_SPLIT_COL).drop(columns=TMP_SPLIT_COL)

        # Split using different methods based on the underlying df engine.
        # For Pandas, split by row index.
        # For Dask, split by partition, as splitting by row is very inefficient.
        return tuple(backend.df_engine.split(df, self.probabilities))

    def validate(self, config: "ModelConfig"):  # noqa: F821
        features = [f for f in config.input_features] + [f for f in config.output_features]
        feature_cols = {f.column for f in features}
        if self.column not in feature_cols:
            logging.info(
                f"Datetime split column {self.column} is not among the features. "
                f"Cannot establish if it is a valid datetime."
            )
        elif [f for f in features if f.column == self.column][0].type not in {DATE}:
            raise ConfigValidationError(f"Feature for datetime split column {self.column} must be a datetime")

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    @staticmethod
    def get_schema_cls():
        return DateTimeSplitConfig


@split_registry.register("hash")
class HashSplitter(Splitter):
    def __init__(
        self,
        column: str,
        probabilities: List[float] = DEFAULT_PROBABILITIES,
        **kwargs,
    ):
        self.column = column
        self.probabilities = probabilities

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # Maximum value of the hash function crc32
        max_value = 2**32
        thresholds = [v * max_value for v in self.probabilities]

        def hash_column(x):
            value = hash_dict({"value": x}, max_length=None)
            hash_value = crc32(value)
            if hash_value < thresholds[0]:
                return 0
            elif hash_value < (thresholds[0] + thresholds[1]):
                return 1
            else:
                return 2

        df[TMP_SPLIT_COL] = backend.df_engine.map_objects(df[self.column], hash_column).astype(np.int8)
        dfs = split_dataset_ttv(df, TMP_SPLIT_COL)
        train, test, val = tuple(df.drop(columns=TMP_SPLIT_COL) if df is not None else None for df in dfs)
        return train, val, test

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0

    @property
    def required_columns(self) -> List[str]:
        return [self.column]

    @staticmethod
    def get_schema_cls():
        return HashSplitConfig


@DeveloperAPI
def get_splitter(type: Optional[str] = None, **kwargs) -> Splitter:
    splitter_cls = split_registry.get(type)
    if splitter_cls is None:
        return ValueError(f"Invalid split type: {type}")
    return splitter_cls(**kwargs)


@DeveloperAPI
def split_dataset(
    df: DataFrame,
    global_preprocessing_parameters: PreprocessingConfigDict,
    backend: Backend,
    random_seed: float = default_random_seed,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    splitter = get_splitter(**global_preprocessing_parameters.get(SPLIT, {}))
    datasets: Tuple[DataFrame, DataFrame, DataFrame] = splitter.split(df, backend, random_seed)
    if len(datasets[0].columns) == 0:
        raise ValueError(
            "Encountered an empty training set while splitting data. Please double check the preprocessing split "
            "configuration."
        )

    # Remove partitions that are empty after splitting
    datasets = [None if dataset is None else backend.df_engine.remove_empty_partitions(dataset) for dataset in datasets]
    return datasets
