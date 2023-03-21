import contextlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING, Union

import ray
import urllib3
from packaging import version
from ray.data.block import Block
from ray.data.context import DatasetContext
from ray.data.datasource.binary_datasource import BinaryDatasource
from ray.data.datasource.datasource import Datasource, ReadTask
from ray.data.datasource.file_based_datasource import (
    _check_pyarrow_version,
    _resolve_paths_and_filesystem,
    _S3FileSystemWrapper,
    _wrap_s3_serialization_workaround,
    BaseFileMetadataProvider,
    BlockOutputBuffer,
    DefaultFileMetadataProvider,
)

from ludwig.utils.fs_utils import get_bytes_obj_from_http_path, is_http

_ray113 = version.parse("1.13") <= version.parse(ray.__version__) == version.parse("1.13.0")

if TYPE_CHECKING:
    import pyarrow

    if _ray113:
        # Only implemented starting in Ray 1.13
        from ray.data.datasource.partitioning import PathPartitionFilter

logger = logging.getLogger(__name__)


class BinaryIgnoreNoneTypeDatasource(BinaryDatasource):
    """Binary datasource, for reading and writing binary files. Ignores None values.

    Examples:
        >>> import ray
        >>> from ray.data.datasource import BinaryDatasource
        >>> source = BinaryDatasource() # doctest: +SKIP
        >>> ray.data.read_datasource( # doctest: +SKIP
        ...     source, paths=["/path/to/dir", None]).take()
        [b"file_data", ...]
    """

    def create_reader(self, **kwargs):
        return _BinaryIgnoreNoneTypeDatasourceReader(self, **kwargs)

    def prepare_read(
        self,
        parallelism: int,
        path_and_idxs: Union[str, List[str], Tuple[str, int], List[Tuple[str, int]]],
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        schema: Optional[Union[type, "pyarrow.lib.Schema"]] = None,
        open_stream_args: Optional[Dict[str, Any]] = None,
        meta_provider: BaseFileMetadataProvider = DefaultFileMetadataProvider(),
        partition_filter: "PathPartitionFilter" = None,
        # TODO(ekl) deprecate this once read fusion is available.
        _block_udf: Optional[Callable[[Block], Block]] = None,
        **reader_args,
    ) -> List[ReadTask]:
        """Creates and returns read tasks for a file-based datasource.

        If `paths` is a tuple, The resulting dataset will have an `idx` key containing the second item in the tuple.
        Useful for tracking the order of files in the dataset.
        """
        reader = self.create_reader(
            paths=path_and_idxs,
            filesystem=filesystem,
            schema=schema,
            open_stream_args=open_stream_args,
            meta_provider=meta_provider,
            partition_filter=partition_filter,
            _block_udf=_block_udf,
            **reader_args,
        )
        return reader.get_read_tasks(parallelism)

    def _open_input_source(
        self,
        filesystem: "pyarrow.fs.FileSystem",
        path: str,
        **open_args,
    ) -> "pyarrow.NativeFile":
        """Opens a source path for reading and returns the associated Arrow NativeFile.

        The default implementation opens the source path as a sequential input stream.

        Implementations that do not support streaming reads (e.g. that require random
        access) should override this method.
        """
        if path is None or is_http(path):
            return contextlib.nullcontext()
        return filesystem.open_input_stream(path, **open_args)

    def _read_file(
        self,
        f: Union["pyarrow.NativeFile", contextlib.nullcontext],
        path_and_idx: Tuple[str, int] = None,
        **reader_args,
    ):
        include_paths = reader_args.get("include_paths", False)

        path, idx = path_and_idx
        if path is None:
            data = None
        elif is_http(path):
            try:
                data = get_bytes_obj_from_http_path(path)
            except urllib3.exceptions.HTTPError as e:
                logger.warning(e)
                data = None
        else:
            super_result = super()._read_file(f, path, **reader_args)[0]
            if include_paths:
                _, data = super_result
            else:
                data = super_result

        result = {"data": data}
        if include_paths:
            result["path"] = path
        if idx is not None:
            result["idx"] = idx
        return [result]


# TODO(geoffrey): ensure this subclasses ray.data.datasource.Reader in ray 1.14
class _BinaryIgnoreNoneTypeDatasourceReader:
    def __init__(
        self,
        delegate: Datasource,
        path_and_idxs: Union[str, List[str], Tuple[str, int], List[Tuple[str, int]]],
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        schema: Optional[Union[type, "pyarrow.lib.Schema"]] = None,
        open_stream_args: Optional[Dict[str, Any]] = None,
        meta_provider: BaseFileMetadataProvider = DefaultFileMetadataProvider(),
        partition_filter: "PathPartitionFilter" = None,
        # TODO(ekl) deprecate this once read fusion is available.
        _block_udf: Optional[Callable[[Block], Block]] = None,
        **reader_args,
    ):
        _check_pyarrow_version()
        self._delegate = delegate
        self._schema = schema
        self._open_stream_args = open_stream_args
        self._meta_provider = meta_provider
        self._partition_filter = partition_filter
        self._block_udf = _block_udf
        self._reader_args = reader_args

        has_idx = isinstance(path_and_idxs[0], tuple)  # include idx if paths is a list of Tuple[str, int]
        raw_paths_and_idxs = path_and_idxs if has_idx else [(path, None) for path in path_and_idxs]

        self._paths = []
        self._file_sizes = []
        for raw_path, idx in raw_paths_and_idxs:
            # Paths must be resolved and expanded
            if raw_path is None or is_http(raw_path):
                read_path = raw_path
                file_size = None  # unknown file size is None
            else:
                resolved_path, filesystem = _resolve_paths_and_filesystem([raw_path], filesystem)
                read_path, file_size = meta_provider.expand_paths(resolved_path, filesystem)
                # expand_paths returns two lists, so get the first element of each
                read_path = read_path[0]
                file_size = file_size[0]

            self._paths.append((read_path, idx))
            self._file_sizes.append(file_size)
        self._filesystem = filesystem

    def estimate_inmemory_data_size(self) -> Optional[int]:
        total_size = 0
        for sz in self._file_sizes:
            if sz is not None:
                total_size += sz
        return total_size

    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        import numpy as np

        open_stream_args = self._open_stream_args
        reader_args = self._reader_args
        _block_udf = self._block_udf

        paths, file_sizes = self._paths, self._file_sizes
        if self._partition_filter is not None:
            raise ValueError("partition_filter is not currently supported by this class")

        read_stream = self._delegate._read_stream
        filesystem = _wrap_s3_serialization_workaround(self._filesystem)

        if open_stream_args is None:
            open_stream_args = {}

        open_input_source = self._delegate._open_input_source

        def read_files(
            read_paths_and_idxs: List[Tuple[str, int]],
            fs: Union["pyarrow.fs.FileSystem", _S3FileSystemWrapper],
        ) -> Iterable[Block]:
            logger.debug(f"Reading {len(read_paths)} files.")
            if isinstance(fs, _S3FileSystemWrapper):
                fs = fs.unwrap()
            ctx = DatasetContext.get_current()
            output_buffer = BlockOutputBuffer(block_udf=_block_udf, target_max_block_size=ctx.target_max_block_size)
            for read_path_and_idx in read_paths_and_idxs:
                read_path, _ = read_path_and_idx
                # Get reader_args and open_stream_args only if valid path.
                if read_path is not None:
                    compression = open_stream_args.pop("compression", None)
                    if compression is None:
                        import pyarrow as pa

                        try:
                            # If no compression manually given, try to detect
                            # compression codec from path.
                            compression = pa.Codec.detect(read_path).name
                        except (ValueError, TypeError):
                            # Arrow's compression inference on the file path
                            # doesn't work for Snappy, so we double-check ourselves.
                            import pathlib

                            suffix = pathlib.Path(read_path).suffix
                            if suffix and suffix[1:] == "snappy":
                                compression = "snappy"
                            else:
                                compression = None
                    if compression == "snappy":
                        # Pass Snappy compression as a reader arg, so datasource subclasses
                        # can manually handle streaming decompression in
                        # self._read_stream().
                        reader_args["compression"] = compression
                        reader_args["filesystem"] = fs
                    elif compression is not None:
                        # Non-Snappy compression, pass as open_input_stream() arg so Arrow
                        # can take care of streaming decompression for us.
                        open_stream_args["compression"] = compression

                with open_input_source(fs, read_path, **open_stream_args) as f:
                    for data in read_stream(f, read_path_and_idx, **reader_args):
                        output_buffer.add_block(data)
                        if output_buffer.has_next():
                            yield output_buffer.next()
            output_buffer.finalize()
            if output_buffer.has_next():
                yield output_buffer.next()

        # fix https://github.com/ray-project/ray/issues/24296
        parallelism = min(parallelism, len(paths))

        read_tasks = []
        for read_paths_and_idxs in np.array_split(paths, parallelism):
            if len(read_paths_and_idxs) <= 0:
                continue

            read_paths, _ = zip(*read_paths_and_idxs)
            meta = self._meta_provider(
                read_paths,
                self._schema,
                rows_per_file=self._delegate._rows_per_file(),
                file_sizes=file_sizes,
            )

            read_task = ReadTask(
                lambda read_paths_and_idxs=read_paths_and_idxs: read_files(read_paths_and_idxs, filesystem), meta
            )
            read_tasks.append(read_task)

        return read_tasks
