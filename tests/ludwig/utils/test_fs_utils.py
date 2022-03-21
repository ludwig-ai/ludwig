# Copyright (c) 2019 Uber Technologies, Inc.
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

from ludwig.utils.fs_utils import get_fs_and_path


def assert_and_create_file(url, expected_path):
    _, path = get_fs_and_path(url)
    assert path == expected_path, f"Expected path {expected_path}"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, path)
            os.makedirs(os.path.dirname(file_path))
            with open(file_path, "w"):
                assert True
    except OSError as exc:
        # OSError if file exists or is invalid
        assert False, f"OS exception {exc}"


def test_get_fs_and_path_simple():
    assert_and_create_file("http://a/b.jpg", os.path.join("a", "b.jpg"))


def test_get_fs_and_path_query_string():
    assert_and_create_file("http://a/b.jpg?c=d", os.path.join("a", "b.jpg"))


def test_get_fs_and_path_decode():
    assert_and_create_file("http://a//b%20c.jpg", os.path.join("a", "b c.jpg"))


def test_get_fs_and_path_unicode():
    assert_and_create_file("http://a/æ.jpg", "a/æ.jpg")
