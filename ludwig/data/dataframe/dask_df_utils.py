import os
import time
import shutil

from contextlib import AbstractContextManager
from pathlib import Path

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.delayed import Delayed
from dask.utils import apply

from ludwig.data.dataframe.pandas import pandas_df_to_tfrecords, write_meta
from ludwig.data.dataset.tfrecord import get_part_filename, get_compression_ext
from ludwig.utils.fs_utils import makedirs


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


class file_lock(AbstractContextManager):
    def __init__(self, path, timeout=None, remove_file=False) -> None:
        self.path = Path(path) if "://" not in path else None
        self.timeout = timeout
        self.remove_file = remove_file
        self.lock_name = f".lock_{self.path.name}" if path else None
        self.lock_path = self.path.parent.joinpath(self.lock_name) if path else None

    def __enter__(self):
        if not self.path:
            return
        start_time = time.time()
        while self.lock_path.exists():
            print(f"{self.lock_path} is locked")
            time.sleep(0.1)
            if self.timeout and (time.time() - start_time) > self.timeout:
                raise TimeoutError()
        print(f"creating lock on {self.lock_path}")
        open(self.lock_path, "w").close()
        if self.remove_file:
            if self.path.is_dir():
                shutil.rmtree(self.path)
            elif self.path.exists():
                os.remove(str(self.path))

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.path:
            return
        print(f"releasing lock on {self.lock_path}")
        os.remove(str(self.lock_path))
