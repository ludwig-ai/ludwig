import logging
from typing import Callable, Dict, List

import daft
import pandas as pd
from ray.data import Dataset

from ludwig.api_annotations import DeveloperAPI
from ludwig.data.dataframe.base import DataFrameEngine

logger = logging.getLogger(__name__)


class DaftDataframeShim:
    """Shim layer on top of a daft.Dataframe to make it behave like a Pandas Dataframe object."""

    def __init__(self, df: daft.DataFrame):
        self._df = df

    @property
    def inner(self) -> daft.DataFrame:
        return self._df

    def __setitem__(self, key: str, val: "DaftSeriesShim") -> None:
        self._df = self._df.with_column(key, val.expr)

    def __getitem__(self, key) -> "DaftSeriesShim":
        return DaftSeriesShim(self, self.inner[key])


class DaftSeriesShim:
    """Shim layer on top of a daft.Expression to make it behave like a Pandas Series object."""

    def __init__(self, src: DaftDataframeShim, expr: daft.Expression):
        self._expr = expr
        self._src = src

    @property
    def expr(self):
        return self._expr

    @property
    def source_dataframe(self) -> daft.DataFrame:
        return self._src.inner


@DeveloperAPI
class DaftEngine(DataFrameEngine):
    def __init__(self, parallelism: int | None):
        self._parallelism = parallelism

    def set_parallelism(self, parallelism):
        raise NotImplementedError(
            "Not implemented for DaftEngine - this does not appear to be called anywhere in Ludwig"
        )

    def df_like(self, df: DaftDataframeShim, proc_cols: Dict[str, DaftSeriesShim]) -> DaftDataframeShim:
        df = df.inner
        for col_name, series in proc_cols.items():
            df = df.with_column(col_name, series.expr)
        return DaftDataframeShim(df)

    def parallelize(self, data: DaftDataframeShim) -> DaftDataframeShim:
        if self._parallelism:
            return DaftDataframeShim(data.inner.into_partitions(self._parallelism))
        return data

    def persist(self, data: DaftDataframeShim) -> DaftDataframeShim:
        # TODO(jay): Currently just a no-op. Let's see how far we can take it without persisting the dataframe at all
        # It seems like the Dask implementation defaults to (and is likely always just) persist=True, but it is unclear
        # why we need to persist.
        return data

    def concat(self, dfs: List[DaftDataframeShim]) -> DaftDataframeShim:
        if len(dfs) == 0:
            raise ValueError("Cannot concat a list of empty dataframes")
        elif len(dfs) == 1:
            return dfs[0]
        else:
            df = dfs[0].inner
            for i in range(1, len(dfs)):
                df = df.concat(dfs[i].inner)
            return DaftDataframeShim(df)

    def compute(self, data: DaftDataframeShim) -> pd.DataFrame:
        return data.inner.to_pandas()

    def from_pandas(self, df: pd.DataFrame) -> DaftDataframeShim:
        parallelism = self._parallelism or 1
        return DaftDataframeShim(
            daft.from_pydict({column: daft.Series.from_pandas(df[column]) for column in df.columns}).into_partitions(
                parallelism
            )
        )

    def map_objects(self, series: DaftSeriesShim, map_fn: Callable[[object], object], meta=None) -> DaftSeriesShim:
        # TODO(jay): If the user can supply the return dtype (e.g. daft.DataType.string()), this operation
        # can be much more optimized in terms of memory usage
        return DaftSeriesShim(series.source_dataframe, series.expr.apply(map_fn, return_dtype=daft.DataType.python()))

    def map_partitions(self, obj: DaftSeriesShim, map_fn: Callable[[pd.Series], pd.Series], meta=None):
        if isinstance(obj, DaftDataframeShim):
            raise NotImplementedError("map_partitions not implemented for Daft Dataframes, only on Daft Series")
        elif isinstance(obj, DaftSeriesShim):
            # TODO(jay): IMPLEMENT.
            pass
        else:
            raise NotImplementedError(f"map_partitions not implemented for object of type: {type(obj)}")

    def map_batches(
        self,
        obj: DaftDataframeShim,
        map_fn: Callable[[pd.DataFrame], pd.DataFrame],
        enable_tensor_extension_casting=True,
    ):
        if isinstance(obj, DaftDataframeShim):
            # TODO(jay): IMPLEMENT.
            pass
        elif isinstance(obj, DaftSeriesShim):
            # TODO(jay): It appears that even though this API is annotated as taking the Series as input, it is actually
            # used on DataFrames, not Series. This branch should never be hit and is not used anywhere in Ludwig.
            raise NotImplementedError("map_batches not implemented for Daft Series")
        else:
            raise NotImplementedError(f"map_batches not implemented for object of type: {type(obj)}")

    def apply_objects(self, df, apply_fn, meta=None):
        raise NotImplementedError(
            "Not implemented for DaftEngine - this does not appear to be called anywhere in Ludwig"
        )

    def reduce_objects(self, series, reduce_fn):
        raise NotImplementedError("TODO: Needs implementation!")

    def split(self, df, probabilities):
        raise NotImplementedError("TODO: Needs implementation!")

    def remove_empty_partitions(self, df):
        raise NotImplementedError("TODO: Needs implementation!")

    def to_parquet(self, df, path, index=False):
        raise NotImplementedError("TODO: Needs implementation!")

    def write_predictions(self, df: DaftDataframeShim, path: str):
        raise NotImplementedError("TODO: Needs implementation!")

    def read_predictions(self, path: str) -> DaftDataframeShim:
        raise NotImplementedError("TODO: Needs implementation!")

    def to_ray_dataset(self, df: DaftDataframeShim) -> Dataset:
        return df.inner.to_ray_dataset()

    def from_ray_dataset(self, dataset: Dataset) -> DaftDataframeShim:
        return DaftDataframeShim(daft.from_ray_dataset(dataset))

    def reset_index(self, df):
        # Daft has no concept of indices so this is a no-op
        return df

    @property
    def array_lib(self):
        raise NotImplementedError(
            "Not implemented for DaftEngine - this does not appear to be called anywhere in Ludwig"
        )

    @property
    def df_lib(self):
        return daft

    @property
    def parallelism(self):
        return self._parallelism

    @property
    def partitioned(self):
        return True
