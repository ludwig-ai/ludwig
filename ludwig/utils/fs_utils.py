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
import functools
import logging
import os
import pathlib
import tempfile
from typing import Any, Optional, Union
from urllib.parse import unquote, urlparse

import fsspec
import h5py
import requests
from filelock import FileLock
from fsspec.core import split_protocol


def get_fs_and_path(url):
    protocol, path = split_protocol(url)
    # Parse the url to get only the escaped url path
    path = unquote(urlparse(path).path)
    # Create a windows compatible path from url path
    path = os.fspath(pathlib.PurePosixPath(path))
    fs = fsspec.filesystem(protocol)
    return fs, path


def has_remote_protocol(url):
    protocol, _ = split_protocol(url)
    return protocol and protocol != "file"


def is_http(urlpath):
    protocol, _ = split_protocol(urlpath)
    return protocol == "http" or protocol == "https"


def upgrade_http(urlpath):
    protocol, url = split_protocol(urlpath)
    if protocol == "http":
        return "https://" + url
    return None


def get_bytes_obj_if_path(path: Any) -> Union[Any, Optional[bytes]]:
    """Gets bytes string if `path` is a path (e.g. a string).

    If it is not a path, return as-is.
    """
    if not isinstance(path, str):
        return path
    return get_bytes_obj_from_path(path)


@functools.lru_cache(maxsize=32)
def get_bytes_obj_from_path(path: str) -> Optional[bytes]:
    if is_http(path):
        try:
            return get_bytes_obj_from_http_path(path)
        except requests.exceptions.RequestException as e:
            logging.warning(e)
            return None
    else:
        try:
            with open_file(path) as f:
                return f.read()
        except OSError as e:
            logging.warning(e)
            return None


@functools.lru_cache(maxsize=32)
def get_bytes_obj_from_http_path(path: str) -> bytes:
    data = requests.get(path, stream=True)
    if data.status_code == 404:
        upgraded = upgrade_http(path)
        if upgraded:
            logging.info(f"reading url {path} failed. upgrading to https and retrying")
            return get_bytes_obj_from_http_path(upgraded)
        else:
            raise requests.exceptions.HTTPError(f"reading url {path} failed and cannot be upgraded to https")
    return data.raw.read()


def find_non_existing_dir_by_adding_suffix(directory_name):
    fs, _ = get_fs_and_path(directory_name)
    suffix = 0
    curr_directory_name = directory_name
    while fs.exists(curr_directory_name):
        curr_directory_name = directory_name + "_" + str(suffix)
        suffix += 1
    return curr_directory_name


def path_exists(url):
    fs, path = get_fs_and_path(url)
    return fs.exists(path)


def rename(src, tgt):
    protocol, _ = split_protocol(tgt)
    if protocol is not None:
        fs = fsspec.filesystem(protocol)
        fs.mv(src, tgt, recursive=True)
    else:
        os.rename(src, tgt)


def makedirs(url, exist_ok=False):
    fs, path = get_fs_and_path(url)
    fs.makedirs(path, exist_ok=exist_ok)
    if not path_exists(url):
        with fsspec.open(url, mode="wb"):
            pass


def delete(url, recursive=False):
    fs, path = get_fs_and_path(url)
    return fs.delete(path, recursive=recursive)


def checksum(url):
    fs, path = get_fs_and_path(url)
    return fs.checksum(path)


def to_url(path):
    protocol, _ = split_protocol(path)
    if protocol is not None:
        return path
    return pathlib.Path(os.path.abspath(path)).as_uri()


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
            if path_exists(url):
                fs.get(url, tmpdir + "/", recursive=True)

            def put_fn():
                fs.put(tmpdir, remote_path, recursive=True)

            # Write to temp directory locally
            yield tmpdir, put_fn

            # Upload to remote when finished
            put_fn()
    else:
        makedirs(url, exist_ok=True)
        # Just use the output directory directly if using a local filesystem
        yield url, None


@contextlib.contextmanager
def open_file(url, *args, **kwargs):
    fs, path = get_fs_and_path(url)
    with fs.open(path, *args, **kwargs) as f:
        yield f


@contextlib.contextmanager
def download_h5(url):
    local_path = fsspec.open_local(url)
    with h5py.File(local_path, "r") as f:
        yield f


@contextlib.contextmanager
def upload_h5(url):
    with upload_output_file(url) as local_fname:
        mode = "w"
        if url == local_fname and path_exists(url):
            mode = "r+"

        with h5py.File(local_fname, mode) as f:
            yield f


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
