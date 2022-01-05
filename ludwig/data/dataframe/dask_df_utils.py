import os

from dask.base import tokenize
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.utils import apply

from ludwig.data.dataframe.pandas import pandas_df_to_tfrecords, write_meta
from ludwig.data.dataset.tfrecord import get_compression_ext, get_part_filename
from ludwig.utils.fs_utils import makedirs


def dask_to_tfrecords(df, folder, compression_type="GZIP", compression_level=9):
    """Store Dask.dataframe to TFRecord files."""
    makedirs(folder, exist_ok=True)
    compression_ext = get_compression_ext(compression_type)
    filenames = [get_part_filename(i, compression_ext) for i in range(df.npartitions)]

    # Also write a meta data file
    write_meta(df, folder, compression_type)

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
                compression_level,
            ],
            kwargs,
        )
        part_tasks.append((name, d))

    dsk[name] = (lambda x: None, part_tasks)

    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[df])
    out = Delayed(name, graph)
    out = out.compute()
    return out
