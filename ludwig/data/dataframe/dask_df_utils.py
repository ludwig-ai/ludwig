import json

import os
import tempfile

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.delayed import Delayed
from dask.utils import apply

from ludwig.data.dataframe.pandas import pandas_df_to_tfrecords, write_meta
from ludwig.data.dataset.tfrecord import get_part_filename, get_compression_ext
from ludwig.utils.data_utils import save_json


def dask_to_tfrecords(
        df,
        folder,
        compression_type="GZIP",
        compression_level=9):
    """Store Dask.dataframe to TFRecord files."""

    use_s3 = folder.startswith("s3")
    local_folder = tempfile.mkdtemp() if use_s3 else folder

    os.makedirs(local_folder, exist_ok=True)
    compression_ext = get_compression_ext(compression_type)
    filenames = [get_part_filename(i, compression_ext) for i in range(df.npartitions)]

    # Also write a meta data file
    write_meta(df, local_folder, compression_type)

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
                os.path.join(local_folder, filename),
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

    # Move to s3
    if use_s3:
        try:
            import s3fs
            s3_fs = s3fs.S3FileSystem()
            s3_fs.put(os.path.join(local_folder, "meta.json"), "{}/{}".format(folder, "meta.json"))
            for filename in filenames:
                s3_fs.put(os.path.join(local_folder, filename), "{}/{}".format(folder, filename))
        except ImportError:
           raise ImportError("Writing to S3 requires `s3fs` support. "
                             "Please install s3fs following: "
                             "https://github.com/s3fs-fuse/s3fs-fuse#installation.")

    return out
