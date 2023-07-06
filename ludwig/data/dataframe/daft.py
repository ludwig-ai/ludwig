import logging
from typing import Dict, List

import daft
import pandas as pd
from ray.data import Dataset

from ludwig.api_annotations import DeveloperAPI
from ludwig.data.dataframe.base import DataFrameEngine

logger = logging.getLogger(__name__)


@DeveloperAPI
class DaftEngine(DataFrameEngine):
    def __init__(self, parallelism: int | None):
        self._parallelism = parallelism

    def set_parallelism(self, parallelism):
        raise NotImplementedError(
            "Not implemented for DaftEngine - this does not appear to be called anywhere in Ludwig"
        )

    def df_like(self, df: daft.DataFrame, proc_cols: Dict[str, daft.Expression]) -> daft.DataFrame:
        for col_name, expr in proc_cols.items():
            df = df.with_column(col_name, expr)
        return df

    def parallelize(self, data: daft.DataFrame) -> daft.DataFrame:
        if self._parallelism:
            return data.into_partitions(self._parallelism)
        return data

    def persist(self, data: daft.DataFrame) -> daft.DataFrame:
        # TODO(jay): Currently just a no-op. Let's see how far we can take it without persisting the dataframe at all
        # It seems like the Dask implementation defaults to (and is likely always just) persist=True, but it is unclear
        # why we need to persist.
        return data

    def concat(self, dfs: List[daft.DataFrame]) -> daft.DataFrame:
        if len(dfs) == 0:
            raise ValueError("Cannot concat a list of empty dataframes")
        elif len(dfs) == 1:
            return dfs[0]
        else:
            df = dfs[0]
            for i in range(1, len(dfs)):
                df = df.concat(dfs[i])
            return df

    def compute(self, data) -> pd.DataFrame:
        return data.to_pandas()

    def from_pandas(self, df):
        parallelism = self._parallelism or 1
        return daft.from_pydict({column: daft.Series.from_pandas(df[column]) for column in df.columns}).into_partitions(
            parallelism
        )

    def map_objects(self, series, map_fn, meta=None):
        raise NotImplementedError("TODO: Needs implementation!")

    def map_partitions(self, series, map_fn, meta=None):
        raise NotImplementedError("TODO: Needs implementation!")

    def map_batches(self, series, map_fn, enable_tensor_extension_casting=True):
        raise NotImplementedError("TODO: Needs implementation!")

    def apply_objects(self, df, apply_fn, meta=None):
        raise NotImplementedError("TODO: Needs implementation!")

    def reduce_objects(self, series, reduce_fn):
        raise NotImplementedError("TODO: Needs implementation!")

    def split(self, df, probabilities):
        raise NotImplementedError("TODO: Needs implementation!")

    def remove_empty_partitions(self, df):
        raise NotImplementedError("TODO: Needs implementation!")

    def to_parquet(self, df, path, index=False):
        raise NotImplementedError("TODO: Needs implementation!")

    def write_predictions(self, df: daft.DataFrame, path: str):
        raise NotImplementedError("TODO: Needs implementation!")

    def read_predictions(self, path: str) -> daft.DataFrame:
        raise NotImplementedError("TODO: Needs implementation!")

    def to_ray_dataset(self, df: daft.DataFrame) -> Dataset:
        return df.to_ray_dataset()

    def from_ray_dataset(self, dataset: Dataset) -> daft.DataFrame:
        return daft.from_ray_dataset(dataset)

    def reset_index(self, df):
        # Daft has no concept of indices so this is a no-op
        return df

    @property
    def array_lib(self):
        raise NotImplementedError("Daft has no array library - this does not seemt o be called from anywhere")

    @property
    def df_lib(self):
        return daft

    @property
    def parallelism(self):
        return self._parallelism

    @property
    def partitioned(self):
        return True
