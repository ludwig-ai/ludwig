#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Linux Foundation
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
import fsspec


DELIMITER = '://'


def get_fs_with_path(path):
    protocol = None
    fname = path
    parts = path.split(DELIMITER)
    if len(parts) == 2:
        protocol, fname = parts
    fs = fsspec.filesystem(protocol)
    return fs, fname


@contextlib.contextmanager
def fs_open(path, *args, **kwargs):
    fs, path = get_fs_with_path(path)
    with fs.open(path, *args, **kwargs) as f:
        yield f


def makedirs(path, *args, **kwargs):
    fs, path = get_fs_with_path(path)
    fs.mkdirs(path, *args, **kwargs)

