#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2020 Uber Technologies, Inc.
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

import os
import shutil
import tempfile
import zipfile


def read_directory_buffer(dir_name):
    """Reads the contents of a directory into an in-memory buffer for RPC."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_name = os.path.join(tmpdir, 'archive')
        ext = 'zip'
        shutil.make_archive(archive_name, ext, dir_name)
        with open(archive_name + '.' + ext, 'rb') as f:
            return f.read()


def write_directory_buffer(directory_buffer, out_dir_name):
    """Writes the in-memory buffer as contents of the output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_name = os.path.join(tmpdir, 'archive')
        with open(archive_name, 'wb') as f:
            f.write(directory_buffer)

        with zipfile.ZipFile(archive_name, 'r') as zip_ref:
            zip_ref.extractall(out_dir_name)
