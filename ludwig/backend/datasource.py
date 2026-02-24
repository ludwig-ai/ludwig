"""Custom Ray datasource utilities for reading binary files with None handling."""
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

import pandas as pd
import ray
import urllib3

from ludwig.utils.fs_utils import get_bytes_obj_from_http_path, is_http

if TYPE_CHECKING:
    import pyarrow

logger = logging.getLogger(__name__)


def read_binary_files_with_index(
    paths_and_idxs: list[tuple[str | None, int]],
    filesystem: Optional["pyarrow.fs.FileSystem"] = None,
) -> "ray.data.Dataset":
    """Read binary files into a Ray Dataset, handling None paths and HTTP URLs.

    Each row in the resulting dataset has columns:
    - "data": the raw bytes of the file (or None if path was None/failed)
    - "idx": the original index for reordering

    Args:
        paths_and_idxs: List of (path, index) tuples. Path can be None.
        filesystem: PyArrow filesystem for reading non-HTTP files.

    Returns:
        A ray.data.Dataset with "data" and "idx" columns.
    """

    def _read_file(path: str | None, idx: int) -> dict:
        if path is None:
            return {"data": None, "idx": idx}
        elif is_http(path):
            try:
                data = get_bytes_obj_from_http_path(path)
            except urllib3.exceptions.HTTPError as e:
                logger.warning(e)
                data = None
            return {"data": data, "idx": idx}
        else:
            try:
                with filesystem.open_input_stream(path) as f:
                    data = f.read()
            except Exception as e:
                logger.warning(f"Failed to read file {path}: {e}")
                data = None
            return {"data": data, "idx": idx}

    # Create a dataset from the paths and indices, then map to read files
    records = [{"path": p, "idx": i} for p, i in paths_and_idxs]
    ds = ray.data.from_items(records)

    def read_batch(batch: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in batch.iterrows():
            result = _read_file(row["path"], row["idx"])
            results.append(result)
        return pd.DataFrame(results)

    ds = ds.map_batches(read_batch, batch_format="pandas")
    return ds
