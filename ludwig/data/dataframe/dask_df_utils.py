import json

import os

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.delayed import Delayed
from dask.utils import apply

from ludwig.data.dataframe.pandas import pandas_df_to_tfrecords


def dask_to_tfrecords(
        df,
        folder,
        compression_type="GZIP",
        compression_level=9):
    """Store Dask.dataframe to TFRecord files."""

    os.makedirs(folder, exist_ok=True)
    compression_ext = '.gz' if compression_type else ''
    filenames = [f"part.{str(i).zfill(5)}.tfrecords{compression_ext}" for i in range(df.npartitions)]

    # Also write a meta data file.
    meta = dict()
    meta["size"] = len(df.index)
    meta_path = os.path.join(folder, "meta.json")
    with open(meta_path, "w") as outfile:
        json.dump(meta, outfile)

    dsk = {}
    name = "to-tfrecord-" + tokenize(df, folder)
    part_tasks = []
    kwargs = {}

    for d, filename in enumerate(filenames):
        dsk[(name, d)] = (
            apply,
            pandas_df_to_tfrecords,
            [
                (df._name, d),
                os.path.join(folder, filename),
                compression_type,
                compression_level
            ],
            kwargs
        )
        part_tasks.append((name, d))

    dsk[name] = (lambda x: None, part_tasks)

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[df])
    out = Delayed(name, graph)
    out = out.compute()
    return out
