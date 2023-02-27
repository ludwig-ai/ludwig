#! /usr/bin/env python
# Copyright (c) 2020 Uber Technologies, Inc.
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

import collections
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Tuple, Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
from dask.diagnostics import ProgressBar
from packaging import version
from pyarrow.fs import FSSpecHandler, PyFileSystem
from ray.data import Dataset, read_parquet
from ray.data.block import Block, BlockAccessor
from ray.data.extensions import ArrowTensorType, TensorDtype
from ray.util.client.common import ClientObjectRef

from ludwig.api_annotations import DeveloperAPI
from ludwig.data.dataframe.base import DataFrameEngine
from ludwig.utils.data_utils import get_pa_schema, get_parquet_filename, split_by_slices
from ludwig.utils.dataframe_utils import set_index_name
from ludwig.utils.fs_utils import get_fs_and_path

TMP_COLUMN = "__TMP_COLUMN__"

# This is to be compatible with pyarrow.lib.schema
PandasBlockSchema = collections.namedtuple("PandasBlockSchema", ["names", "types"])

logger = logging.getLogger(__name__)


_ray_230 = version.parse(ray.__version__) >= version.parse("2.3.0")


@DeveloperAPI
def set_scheduler(scheduler):
    dask.config.set(scheduler=scheduler)


@DeveloperAPI
def reset_index_across_all_partitions(df):
    """Compute a monotonically increasing index across all partitions.

    This differs from dd.reset_index, which computes an independent index for each partition.
    Source: https://stackoverflow.com/questions/61395351/how-to-reset-index-on-concatenated-dataframe-in-dask
    """
    # Create temporary column of ones
    df = df.assign(**{TMP_COLUMN: 1})

    # Set the index to the cumulative sum of TMP_COLUMN, which we know to be sorted; this improves efficiency.
    df = df.set_index(df[TMP_COLUMN].cumsum() - 1, sorted=True)

    # Drop temporary column and ensure the index is not named TMP_COLUMN
    df = df.drop(columns=TMP_COLUMN)
    df = df.map_partitions(lambda pd_df: set_index_name(pd_df, None))
    return df


@DeveloperAPI
class DaskEngine(DataFrameEngine):
    def __init__(self, parallelism=None, persist=True, _use_ray=True, **kwargs):
        from ray.util.dask import ray_dask_get

        self._parallelism = parallelism
        self._persist = persist
        if _use_ray:
            set_scheduler(ray_dask_get)

    def set_parallelism(self, parallelism):
        self._parallelism = parallelism

    def df_like(self, df: dd.DataFrame, proc_cols: Dict[str, dd.Series]):
        """Outer joins the given DataFrame with the given processed columns.

        NOTE: If any of the processed columns have been repartitioned, the original index is replaced with a
        monotonically increasing index, which is used to define the new divisions and align the various partitions.
        """
        # Our goal is to preserve the index of the input dataframe but to drop
        # all its columns. Because to_frame() creates a column from the index,
        # we need to drop it immediately following creation.
        dataset = df.index.to_frame(name=TMP_COLUMN).drop(columns=TMP_COLUMN)

        repartitioned_cols = {}
        for k, v in proc_cols.items():
            if v.npartitions == dataset.npartitions:
                # Outer join cols with equal partitions
                v.divisions = dataset.divisions
                dataset[k] = v
            else:
                # If partitions have changed (e.g. due to conversion from Ray dataset), we handle separately
                repartitioned_cols[k] = v

        # Assumes that there is a globally unique index (see preprocessing.build_dataset)
        if repartitioned_cols:
            if not dataset.known_divisions:
                # Sometimes divisions are unknown despite having a usable index– set_index to know divisions
                dataset = dataset.assign(**{TMP_COLUMN: dataset.index})
                dataset = dataset.set_index(TMP_COLUMN, drop=True)
                dataset = dataset.map_partitions(lambda pd_df: set_index_name(pd_df, dataset.index.name))

            # Find the divisions of the column with the largest number of partitions
            proc_col_with_max_npartitions = max(repartitioned_cols.values(), key=lambda x: x.npartitions)
            new_divisions = proc_col_with_max_npartitions.divisions

            # Repartition all columns to have the same divisions
            dataset = dataset.repartition(new_divisions)
            repartitioned_cols = {k: v.repartition(new_divisions) for k, v in repartitioned_cols.items()}

            # Outer join the remaining columns
            for k, v in repartitioned_cols.items():
                dataset[k] = v

        return dataset

    def parallelize(self, data):
        if self.parallelism:
            return data.repartition(self.parallelism)
        return data

    def persist(self, data):
        # No graph optimizations to prevent dropping custom annotations
        # https://github.com/dask/dask/issues/7036
        return data.persist(optimize_graph=False) if self._persist else data

    def concat(self, dfs):
        return self.df_lib.multi.concat(dfs)

    def compute(self, data):
        return data.compute()

    def from_pandas(self, df):
        parallelism = self._parallelism or 1
        return dd.from_pandas(df, npartitions=parallelism)

    def map_objects(self, series, map_fn, meta=None):
        meta = meta if meta is not None else ("data", "object")
        return series.map(map_fn, meta=meta)

    def map_partitions(self, series, map_fn, meta=None):
        meta = meta if meta is not None else ("data", "object")
        return series.map_partitions(map_fn, meta=meta)

    def map_batches(self, series, map_fn, enable_tensor_extension_casting=True):
        """Map a function over batches of a Dask Series.

        Args:
            series: Dask Series
            map_fn: Function to apply to each batch
            enable_tensor_extension_casting: Whether to enable tensor extension casting at the end of the Ray Datasets
                map_batches call. This is useful in cases where the output is not supported by the ray Tensor dtype
                extension, such as when the output consists of ragged tensors.
        """
        import ray.data

        with tensor_extension_casting(enable_tensor_extension_casting):
            ds = ray.data.from_dask(series)
            ds = ds.map_batches(map_fn, batch_format="pandas")
            return self._to_dask(ds)

    def apply_objects(self, df, apply_fn, meta=None):
        meta = meta if meta is not None else ("data", "object")
        return df.apply(apply_fn, axis=1, meta=meta)

    def reduce_objects(self, series, reduce_fn):
        return series.reduction(reduce_fn, aggregate=reduce_fn, meta=("data", "object")).compute()[0]

    def split(self, df, probabilities):
        # Split the DataFrame proprotionately along partitions. This is an inexact solution designed
        # to speed up the split process, as splitting within partitions would be significantly
        # more expensive.
        # TODO(travis): revisit in the future to make this more precise

        # First ensure that every split receives at least one partition.
        # If not, we need to increase the number of partitions to satisfy this constraint.
        min_prob = min(probabilities)
        min_partitions = int(1 / min_prob)
        if df.npartitions < min_partitions:
            df = df.repartition(min_partitions)

        n = df.npartitions
        slices = df.partitions
        return split_by_slices(slices, n, probabilities)

    def remove_empty_partitions(self, df):
        # Reference: https://stackoverflow.com/questions/47812785/remove-empty-partitions-in-dask
        ll = list(df.map_partitions(len).compute())
        if all([ll_i > 0 for ll_i in ll]):
            return df

        df_delayed = df.to_delayed()
        df_delayed_new = list()
        empty_partition = None
        for ix, n in enumerate(ll):
            if n == 0:
                empty_partition = df.get_partition(ix)
            else:
                df_delayed_new.append(df_delayed[ix])
        df = dd.from_delayed(df_delayed_new, meta=empty_partition)
        return df

    def to_parquet(self, df, path, index=False):
        schema = get_pa_schema(df)
        with ProgressBar():
            df.to_parquet(
                path,
                engine="pyarrow",
                write_index=index,
                schema=schema,
                name_function=get_parquet_filename,
            )

    def write_predictions(self, df: dd.DataFrame, path: str):
        ds = self.to_ray_dataset(df)
        # We disable tensor extension casting here because we are writing out to Parquet and there is no need
        # to cast to the ray Tensor dtype extension before doing so (they will be written out as object dtype as if
        # we were writing to parquet using dask).
        with tensor_extension_casting(False):
            fs, path = get_fs_and_path(path)
            ds.write_parquet(path, filesystem=PyFileSystem(FSSpecHandler(fs)))

    def read_predictions(self, path: str) -> dd.DataFrame:
        fs, path = get_fs_and_path(path)
        ds = read_parquet(path, filesystem=PyFileSystem(FSSpecHandler(fs)))
        return self.from_ray_dataset(ds)

    def to_ray_dataset(self, df) -> Dataset:
        from ray.data import from_dask

        return from_dask(df)

    def from_ray_dataset(self, dataset) -> dd.DataFrame:
        return self._to_dask(dataset)

    def reset_index(self, df):
        return reset_index_across_all_partitions(df)

    def _to_dask(
        self,
        dataset: Dataset,
        meta: Union[
            pd.DataFrame,
            pd.Series,
            Dict[str, Any],
            Iterable[Any],
            Tuple[Any],
            None,
        ] = None,
    ) -> dd.DataFrame:
        """Custom Ray to_dask() conversion implementation with meta inference added for compatibility with Ray 2.0
        and Ray 2.1. Useful for Ray Datasets that have image and audio features.

        TODO(Arnav): Remove in Ray 2.2
        """
        if _ray_230:
            return dataset.to_dask()

        @dask.delayed
        def block_to_df(block: Block):
            if isinstance(block, (ray.ObjectRef, ClientObjectRef)):
                raise ValueError(
                    "Dataset.to_dask() must be used with Dask-on-Ray, please "
                    "set the Dask scheduler to ray_dask_get (located in "
                    "ray.util.dask)."
                )
            block = BlockAccessor.for_block(block)
            return block.to_pandas()

        # Infer Dask metadata from Datasets schema.
        schema = dataset.schema()
        if isinstance(schema, PandasBlockSchema):
            meta = pd.DataFrame(
                {
                    col: pd.Series(dtype=(dtype if not isinstance(dtype, TensorDtype) else np.object_))
                    for col, dtype in zip(schema.names, schema.types)
                }
            )
        elif isinstance(schema, pa.Schema):
            if any(isinstance(type_, ArrowTensorType) for type_ in schema.types):
                meta = pd.DataFrame(
                    {
                        col: pd.Series(
                            dtype=(dtype.to_pandas_dtype() if not isinstance(dtype, ArrowTensorType) else np.object_)
                        )
                        for col, dtype in zip(schema.names, schema.types)
                    }
                )
            else:
                meta = schema.empty_table().to_pandas()

        ddf = dd.from_delayed(
            [block_to_df(block) for block in dataset.get_internal_block_refs()],
            meta=meta,
        )
        return ddf

    @property
    def array_lib(self):
        return da

    @property
    def df_lib(self):
        return dd

    @property
    def parallelism(self):
        return self._parallelism

    @property
    def partitioned(self):
        return True


@contextmanager
def tensor_extension_casting(enforced: bool):
    """This context manager is used to enforce or disable tensor extension casting.

    Ray Datasets will automatically cast tensor columns to the ray Tensor dtype extension at the end of
    map_batches calls and before writing to Parquet. This context manager can be used to disable this behavior
    and keep the tensor columns as object dtype. This is useful for writing to Parquet using dask.

    Args:
        enforced (bool): Whether to enforce tensor extension casting.
    """
    from ray.data.context import DatasetContext

    ctx = DatasetContext.get_current()
    prev_enable_tensor_extension_casting = ctx.enable_tensor_extension_casting
    try:
        ctx.enable_tensor_extension_casting = enforced
        yield
    finally:
        ctx.enable_tensor_extension_casting = prev_enable_tensor_extension_casting
