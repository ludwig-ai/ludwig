import contextlib
import os
import time
import shutil

from contextlib import AbstractContextManager
from pathlib import Path

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.delayed import Delayed
from dask.utils import apply
from filelock import FileLock

from ludwig.data.dataframe.pandas import pandas_df_to_tfrecords, write_meta
from ludwig.data.dataset.tfrecord import get_part_filename, get_compression_ext
from ludwig.utils.fs_utils import makedirs, has_remote_protocol


def dask_to_tfrecords(
        df,
        folder,
        compression_type="GZIP",
        compression_level=9):
    """Store Dask.dataframe to TFRecord files."""
    makedirs(folder, exist_ok=True)
    compression_ext = get_compression_ext(compression_type)
    filenames = [get_part_filename(i, compression_ext)
                 for i in range(df.npartitions)]

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


@contextlib.contextmanager
def file_lock(path: str):
    """Simple file lock based on creating and removing a lock file."""
    if not has_remote_protocol(path):
        with FileLock(f'{path}.lock'):
            yield
    else:
        yield
