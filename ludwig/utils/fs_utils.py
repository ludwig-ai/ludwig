#! /usr/bin/env python
# Copyright (c) 2021 Linux Foundation.
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

import contextlib
import errno
import functools
import logging
import os
import pathlib
import shutil
import tempfile
import uuid
from typing import Optional
from urllib.parse import unquote, urlparse

import certifi
import fsspec
import h5py
import pyarrow.fs
import urllib3
from filelock import FileLock
from fsspec.core import split_protocol

from ludwig.api_annotations import DeveloperAPI

logger = logging.getLogger(__name__)


@DeveloperAPI
def get_fs_and_path(url):
    protocol, path = split_protocol(url)
    # Parse the url to get only the escaped url path
    path = unquote(urlparse(path).path)
    # Create a windows compatible path from url path
    path = os.fspath(pathlib.PurePosixPath(path))
    fs = fsspec.filesystem(protocol)
    return fs, path


@DeveloperAPI
def has_remote_protocol(url):
    protocol, _ = split_protocol(url)
    return protocol and protocol != "file"


@DeveloperAPI
def is_http(urlpath):
    protocol, _ = split_protocol(urlpath)
    return protocol == "http" or protocol == "https"


@DeveloperAPI
def upgrade_http(urlpath):
    protocol, url = split_protocol(urlpath)
    if protocol == "http":
        return "https://" + url
    return None


@DeveloperAPI
@functools.lru_cache(maxsize=32)
def get_bytes_obj_from_path(path: str) -> Optional[bytes]:
    if is_http(path):
        try:
            return get_bytes_obj_from_http_path(path)
        except Exception as e:
            logger.warning(e)
            return None
    else:
        try:
            with open_file(path) as f:
                return f.read()
        except OSError as e:
            logger.warning(e)
            return None


@DeveloperAPI
def stream_http_get_request(path: str) -> urllib3.response.HTTPResponse:
    if upgrade_http(path):
        http = urllib3.PoolManager()
    else:
        http = urllib3.PoolManager(ca_certs=certifi.where())
    resp = http.request("GET", path, preload_content=False)
    return resp


@DeveloperAPI
@functools.lru_cache(maxsize=32)
def get_bytes_obj_from_http_path(path: str) -> bytes:
    resp = stream_http_get_request(path)
    if resp.status == 404:
        upgraded = upgrade_http(path)
        if upgraded:
            logger.info(f"reading url {path} failed. upgrading to https and retrying")
            return get_bytes_obj_from_http_path(upgraded)
        else:
            raise urllib3.exceptions.HTTPError(f"reading url {path} failed and cannot be upgraded to https")

    # stream data
    data = b""
    for chunk in resp.stream(1024):
        data += chunk
    return data


@DeveloperAPI
def find_non_existing_dir_by_adding_suffix(directory_name):
    fs, _ = get_fs_and_path(directory_name)
    suffix = 0
    curr_directory_name = directory_name
    while fs.exists(curr_directory_name):
        curr_directory_name = directory_name + "_" + str(suffix)
        suffix += 1
    return curr_directory_name


@DeveloperAPI
def abspath(url):
    protocol, _ = split_protocol(url)
    if protocol is not None:
        # we assume any path containing an explicit protovol is fully qualified
        return url
    return os.path.abspath(url)


@DeveloperAPI
def path_exists(url):
    fs, path = get_fs_and_path(url)
    return fs.exists(path)


@DeveloperAPI
def listdir(url):
    fs, path = get_fs_and_path(url)
    return fs.listdir(path)


@DeveloperAPI
def safe_move_file(src, dst):
    """Rename a file from `src` to `dst`. Inspired by: https://alexwlchan.net/2019/03/atomic-cross-filesystem-
    moves-in-python/

    *   Moves must be atomic.  `shutil.move()` is not atomic.

    *   Moves must work across filesystems.  Sometimes temp directories and the
        model directories live on different filesystems.  `os.replace()` will
        throw errors if run across filesystems.

    So we try `os.replace()`, but if we detect a cross-filesystem copy, we
    switch to `shutil.move()` with some wrappers to make it atomic.
    """
    try:
        os.replace(src, dst)
    except OSError as err:
        if err.errno == errno.EXDEV:
            # Generate a unique ID, and copy `<src>` to the target directory with a temporary name `<dst>.<ID>.tmp`.
            # Because we're copying across a filesystem boundary, this initial copy may not be atomic.  We insert a
            # random UUID so if different processes are copying into `<dst>`, they don't overlap in their tmp copies.
            copy_id = uuid.uuid4()
            tmp_dst = f"{dst}.{copy_id}.tmp"
            shutil.copyfile(src, tmp_dst)

            # Atomic replace file onto the new name, and clean up original source file.
            os.replace(tmp_dst, dst)
            os.unlink(src)
        else:
            raise


@DeveloperAPI
def safe_move_directory(src, dst):
    """Recursively moves files from src directory to dst directory and removes src directory.

    If dst directory does not exist, it will be created.
    """
    try:
        os.replace(src, dst)
    except OSError as err:
        if err.errno == errno.EXDEV:
            # Generate a unique ID, and copy `<src>` to the target directory with a temporary name `<dst>.<ID>.tmp`.
            # Because we're copying across a filesystem boundary, this initial copy may not be atomic.  We insert a
            # random UUID so if different processes are copying into `<dst>`, they don't overlap in their tmp copies.
            copy_id = uuid.uuid4()
            tmp_dst = f"{dst}.{copy_id}.tmp"
            shutil.copytree(src, tmp_dst)

            # Atomic replace directory name onto the new name, and clean up original source directory.
            os.replace(tmp_dst, dst)
            os.unlink(src)
        else:
            raise


@DeveloperAPI
def rename(src, tgt):
    protocol, _ = split_protocol(tgt)
    if protocol is not None:
        fs = fsspec.filesystem(protocol)
        fs.mv(src, tgt, recursive=True)
    else:
        safe_move_file(src, tgt)


@DeveloperAPI
def upload_file(src, tgt):
    protocol, _ = split_protocol(tgt)
    fs = fsspec.filesystem(protocol)
    fs.put(src, tgt)


@DeveloperAPI
def copy(src, tgt, recursive=False):
    protocol, _ = split_protocol(tgt)
    fs = fsspec.filesystem(protocol)
    fs.copy(src, tgt, recursive=recursive)


@DeveloperAPI
def makedirs(url, exist_ok=False):
    fs, path = get_fs_and_path(url)
    fs.makedirs(path, exist_ok=exist_ok)


@DeveloperAPI
def delete(url, recursive=False):
    fs, path = get_fs_and_path(url)
    return fs.delete(path, recursive=recursive)


@DeveloperAPI
def upload(lpath, rpath):
    fs, path = get_fs_and_path(rpath)
    pyarrow.fs.copy_files(lpath, path, destination_filesystem=pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs)))


@DeveloperAPI
def download(rpath, lpath):
    fs, path = get_fs_and_path(rpath)
    pyarrow.fs.copy_files(path, lpath, source_filesystem=pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs)))


@DeveloperAPI
def checksum(url):
    fs, path = get_fs_and_path(url)
    return fs.checksum(path)


@DeveloperAPI
def to_url(path):
    protocol, _ = split_protocol(path)
    if protocol is not None:
        return path
    return pathlib.Path(os.path.abspath(path)).as_uri()


@DeveloperAPI
@contextlib.contextmanager
def upload_output_directory(url):
    if url is None:
        yield None, None
        return

    protocol, _ = split_protocol(url)
    if protocol is not None:
        # To avoid extra network load, write all output files locally at runtime,
        # then upload to the remote fs at the end.
        with tempfile.TemporaryDirectory() as tmpdir:
            fs, remote_path = get_fs_and_path(url)

            # In cases where we are resuming from a previous run, we first need to download
            # the artifacts from the remote filesystem
            if path_exists(url):
                fs.get(url, tmpdir + "/", recursive=True)

            def put_fn():
                # Use pyarrow API here as fs.put() is inconsistent in where it uploads the file
                # See: https://github.com/fsspec/filesystem_spec/issues/1062
                pyarrow.fs.copy_files(
                    tmpdir, remote_path, destination_filesystem=pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(fs))
                )

            # Write to temp directory locally
            yield tmpdir, put_fn

            # Upload to remote when finished
            put_fn()
    else:
        makedirs(url, exist_ok=True)
        # Just use the output directory directly if using a local filesystem
        yield url, None


@DeveloperAPI
@contextlib.contextmanager
def open_file(url, *args, **kwargs):
    fs, path = get_fs_and_path(url)
    with fs.open(path, *args, **kwargs) as f:
        yield f


@DeveloperAPI
@contextlib.contextmanager
def download_h5(url):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, os.path.basename(url))
        fs, path = get_fs_and_path(url)
        fs.get(path, local_path)
        with h5py.File(local_path, "r") as f:
            yield f


@DeveloperAPI
@contextlib.contextmanager
def upload_h5(url):
    with upload_output_file(url) as local_fname:
        mode = "w"
        if url == local_fname and path_exists(url):
            mode = "r+"

        with h5py.File(local_fname, mode) as f:
            yield f


@DeveloperAPI
@contextlib.contextmanager
def upload_output_file(url):
    """Takes a remote URL as input, returns a temp filename, then uploads it when done."""
    protocol, _ = split_protocol(url)
    if protocol is not None:
        fs = fsspec.filesystem(protocol)
        with tempfile.TemporaryDirectory() as tmpdir:
            local_fname = os.path.join(tmpdir, "tmpfile")
            yield local_fname
            fs.put(local_fname, url, recursive=True)
    else:
        yield url


@DeveloperAPI
class file_lock(contextlib.AbstractContextManager):
    """File lock based on filelock package."""

    def __init__(self, path: str, ignore_remote_protocol: bool = True, lock_file: str = ".lock") -> None:
        if not isinstance(path, (str, os.PathLike, pathlib.Path)):
            self.lock = None
        else:
            path = os.path.join(path, lock_file) if os.path.isdir(path) else f"{path}./{lock_file}"
            if ignore_remote_protocol and has_remote_protocol(path):
                self.lock = None
            else:
                self.lock = FileLock(path, timeout=-1)

    def __enter__(self, *args, **kwargs):
        if self.lock:
            return self.lock.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        if self.lock:
            return self.lock.__exit__(*args, **kwargs)
