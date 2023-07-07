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
        return DaftSeriesShim(self.inner[key])


class DaftSeriesShim:
    """Shim layer on top of a daft.Expression to make it behave like a Pandas Series object."""

    def __init__(self, expr: daft.Expression):
        self._expr = expr

    @property
    def expr(self):
        return self._expr


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
        return DaftDataframeShim(data.inner.collect())

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
        # NOTE: If the user can supply the return dtype (e.g. daft.DataType.string()), this operation
        # can be much more optimized in terms of memory usage
        return DaftSeriesShim(series.expr.apply(map_fn, return_dtype=daft.DataType.python()))

    def map_partitions(self, obj: DaftSeriesShim, map_fn: Callable[[pd.Series], pd.Series], meta=None):
        # NOTE: Although the function signature indicates that this function takes in a Series, in practice
        # it appears that this function is often used interchangeably to run on both Series and DataFrames
        if isinstance(obj, DaftDataframeShim):
            raise NotImplementedError("TODO: Implementation")
        elif isinstance(obj, DaftSeriesShim):
            raise NotImplementedError("TODO: Implementation")
        else:
            raise NotImplementedError(f"map_partitions not implemented for object of type: {type(obj)}")

    def map_batches(
        self,
        obj: DaftDataframeShim,
        map_fn: Callable[[pd.DataFrame], pd.DataFrame],
        enable_tensor_extension_casting=True,
    ):
        # NOTE: This is only used in preprocessing code to run "postprocess_batch", which is a specific function
        # per feature type that is defined in the config
        #
        # NOTE: This is fairly inefficient in Daft because when calling a black-box map_fn function, Daft
        # cannot understand which columns are actually being used in `map_fn`, and cannot perform optimizations
        # using that information.
        #
        # Instead, if each postprocessing step can define what columns it needs to run, then we can supply
        # that to Daft and Daft will provide just those columns that it needs.
        assert isinstance(obj, DaftDataframeShim), "map_batches should only be called on DataFrames, not Series"
        raise NotImplementedError("TODO: Implementation")

    def apply_objects(self, df, apply_fn, meta=None):
        raise NotImplementedError(
            "Not implemented for DaftEngine - this does not appear to be called anywhere in Ludwig"
        )

    def reduce_objects(self, series, reduce_fn):
        raise NotImplementedError(
            "Not implemented for DaftEngine - this is only used in audio_feature.py and is much better "
            "expressed as a DataFrame aggregation using the provided dataframe APIs for mean/max/min/stddev etc. "
            "As a workaround, users can run .map_partitions() and then just .to_pandas() to perform reductions locally."
        )

    def split(self, df, probabilities):
        raise NotImplementedError("Requires some new APIs in Daft to support")

    def remove_empty_partitions(self, df: DaftDataframeShim):
        # This is a no-op in the DaftEngine - we stick to the specified parallelism and users can
        # call a df.into_partitions(self._parallelism) instead to rebalance the data.
        logger.warning(
            "Ignoring `.remove_empty_partitions()`: DaftEngine has a fixed number of partitions. "
            "You may wish to rebalance the dataframe instead with `.into_partitions(parallelism)`"
        )
        return df

    def to_parquet(self, df, path, index=False):
        if index:
            logger.warning(
                "Ignoring `index=True`: DaftEngine has no concept of an index and cannot write indices to Parquet"
            )
        df.inner.write_parquet(path)

    def write_predictions(self, df: DaftDataframeShim, path: str):
        self.to_parquet(df, path)

    def read_predictions(self, path: str) -> DaftDataframeShim:
        return DaftDataframeShim(daft.read_parquet(path))

    def to_ray_dataset(self, df: DaftDataframeShim) -> Dataset:
        return df.inner.to_ray_dataset()

    def from_ray_dataset(self, dataset: Dataset) -> DaftDataframeShim:
        return DaftDataframeShim(daft.from_ray_dataset(dataset))

    def reset_index(self, df):
        # Daft has no concept of indices so this is a no-op
        logger.warning("Ignoring `.reset_index()`: DaftEngine has no concept indices")
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
