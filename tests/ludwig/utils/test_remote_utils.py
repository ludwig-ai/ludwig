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
import tempfile

from ludwig.utils.remote_utils import read_directory_buffer, write_directory_buffer


def test_remote_archive():
    basename = 'file.txt'
    text = 'Hello world!'

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, basename), 'w') as f:
            f.write(text)

        buffer = read_directory_buffer(tmpdir)

    assert len(buffer) > 0
    assert isinstance(buffer, bytes)

    with tempfile.TemporaryDirectory() as tmpdir:
        write_directory_buffer(buffer, tmpdir)

        fnames = os.listdir(tmpdir)
        assert len(fnames) == 1
        assert fnames[0] == basename

        with open(os.path.join(tmpdir, basename), 'r') as f:
            assert text == f.read()
