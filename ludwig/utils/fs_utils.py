#! /usr/bin/env python
# coding=utf-8
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
import os
import pathlib
import tempfile

import fsspec
import h5py
from fsspec.core import split_protocol


def get_fs_mode(mode):
    if mode == 'r':
        return 'rb'
    elif mode == 'w':
        return 'wb'
    return mode


def get_fs_and_path(url):
    protocol, path = split_protocol(url)
    fs = fsspec.filesystem(protocol)
    return fs, path


def find_non_existing_dir_by_adding_suffix(directory_name):
    fs, _ = get_fs_and_path(directory_name)
    suffix = 0
    curr_directory_name = directory_name
    while fs.exists(curr_directory_name):
        curr_directory_name = directory_name + '_' + str(suffix)
        suffix += 1
    return curr_directory_name


def path_exists(url):
    fs, path = get_fs_and_path(url)
    return fs.exists(path)


def makedirs(url, exist_ok=False):
    fs, path = get_fs_and_path(url)
    return fs.makedirs(path, exist_ok=exist_ok)


def to_url(path):
    protocol, _ = split_protocol(path)
    if protocol is not None:
        return path
    return pathlib.Path(os.path.abspath(path)).as_uri()


@contextlib.contextmanager
def prepare_output_directory(url):
    if url is None:
        yield None
        return

    protocol, _ = split_protocol(url)
    if protocol is not None:
        # To avoid extra network load, write all output files locally at runtime,
        # then upload to the remote fs at the end.
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write to temp directory locally
            yield tmpdir

            # Upload to remote when finished
            fs, remote_path = get_fs_and_path(url)
            fs.put(tmpdir, remote_path, recursive=True)
    else:
        # Just use the output directory directly if using a local filesystem
        yield url


@contextlib.contextmanager
def open_file(url, *args, **kwargs):
    of = fsspec.open(url, *args, **kwargs)
    with of as f:
        yield f


@contextlib.contextmanager
def open_h5(url, mode):
    with open_file(url, get_fs_mode(mode)) as fh:
        with h5py.File(fh, mode) as f:
            yield f
