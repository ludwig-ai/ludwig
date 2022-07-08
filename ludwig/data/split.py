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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from ludwig.backend.base import Backend
from ludwig.constants import BINARY, CATEGORY, COLUMN, DATE, SPLIT, TYPE
from ludwig.utils.data_utils import split_dataset_ttv
from ludwig.utils.registry import Registry
from ludwig.utils.types import DataFrame

split_registry = Registry()
default_random_seed = 42


TMP_SPLIT_COL = "__SPLIT__"
DEFAULT_PROBABILITIES = (0.7, 0.1, 0.2)


class Splitter(ABC):
    @abstractmethod
    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        pass

    def validate(self, config: Dict[str, Any]):
        pass

    def has_split(self, split_index: int) -> bool:
        return True

    @property
    def required_columns(self) -> List[str]:
        return []


@split_registry.register("random", default=True)
class RandomSplitter(Splitter):
    def __init__(self, probabilities: List[float] = DEFAULT_PROBABILITIES, **kwargs):
        self.probabilities = probabilities

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        if backend.df_engine.partitioned:
            # The below approach is very inefficient for partitioned backends, which
            # can split by partition. This may not be exact in all cases, but is much more efficient.
            return df.random_split(self.probabilities, random_state=random_seed)

        n = len(df)
        d1 = int(self.probabilities[0] * n)
        d2 = d1 + int(self.probabilities[1] * n)
        return np.split(df.sample(frac=1, random_state=random_seed), [d1, d2])

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0


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


@split_registry.register("stratify")
class StratifySplitter(Splitter):
    def __init__(self, column: str, probabilities: List[float] = DEFAULT_PROBABILITIES, **kwargs):
        self.column = column
        self.probabilities = probabilities

    def split(
        self, df: DataFrame, backend: Backend, random_seed: float = default_random_seed
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        if backend.df_engine.partitioned:
            # TODO dask: find a way to support this method
            raise ValueError('Split type "stratify" is not supported with a partitioned dataset.')

        frac_train, frac_val, frac_test = self.probabilities

        # Dataframe of just the column on which to stratify
        y = df[[self.column]].astype(np.int8)
        df_train, df_temp, _, y_temp = train_test_split(
            df, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_seed
        )
        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, _, _ = train_test_split(
            df_temp, y_temp, stratify=y_temp, test_size=relative_frac_test, random_state=random_seed
        )

        return df_train, df_val, df_test

    def validate(self, config: Dict[str, Any]):
        features = config["input_features"] + config["output_features"]
        feature_names = {f[COLUMN] for f in features}
        if self.column not in feature_names:
            logging.info(
                f"Stratify column {self.column} is not among the features. "
                f"Cannot establish if it is a binary or category"
            )
        elif [f for f in features if f[COLUMN] == self.column][0][TYPE] not in {BINARY, CATEGORY}:
            raise ValueError(f"Feature for stratify column {self.column} must be binary or category")

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0

    @property
    def required_columns(self) -> List[str]:
        return [self.column]


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
            if not isinstance(x, list) and len(x) != 9:
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

    def validate(self, config: Dict[str, Any]):
        features = config["input_features"] + config["output_features"]
        feature_names = {f[COLUMN] for f in features}
        if self.column not in feature_names:
            logging.info(
                f"Datetime split column {self.column} is not among the features. "
                f"Cannot establish if it is a valid datetime."
            )
        elif [f for f in features if f[COLUMN] == self.column][0][TYPE] not in {DATE}:
            raise ValueError(f"Feature for datetime split column {self.column} must be a datetime")

    def has_split(self, split_index: int) -> bool:
        return self.probabilities[split_index] > 0

    @property
    def required_columns(self) -> List[str]:
        return [self.column]


def get_splitter(type: Optional[str] = None, **kwargs) -> Splitter:
    splitter_cls = split_registry.get(type)
    if splitter_cls is None:
        return ValueError(f"Invalid split type: {type}")
    return splitter_cls(**kwargs)


def split_dataset(
    df: DataFrame,
    global_preprocessing_parameters: Dict[str, Any],
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
    return datasets
