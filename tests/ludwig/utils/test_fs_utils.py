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
import logging
import os
import platform
import tempfile
from urllib.parse import quote

import pytest

from ludwig.utils.fs_utils import get_fs_and_path


def create_file(url):
    _, path = get_fs_and_path(url)
    logging.info(f"saving url '{url}' to path '{path}'")
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, path)
        os.makedirs(os.path.dirname(file_path))
        with open(file_path, "w"):
            return path


@pytest.mark.filesystem
def test_get_fs_and_path_simple():
    assert create_file("http://a/b.jpg") == os.path.join("a", "b.jpg")


@pytest.mark.filesystem
def test_get_fs_and_path_query_string():
    assert create_file("http://a/b.jpg?c=d") == os.path.join("a", "b.jpg")


@pytest.mark.filesystem
def test_get_fs_and_path_decode():
    assert create_file("http://a//b%20c.jpg") == os.path.join("a", "b c.jpg")


@pytest.mark.filesystem
def test_get_fs_and_path_unicode():
    assert create_file("http://a/æ.jpg") == "a/æ.jpg"


@pytest.mark.filesystem
@pytest.mark.skipif(platform.system() == "Windows", reason="Skipping if windows.")
def test_get_fs_and_path_invalid_linux():
    invalid_chars = {
        "\x00": ValueError,
        "/": FileExistsError,
    }
    for c, e in invalid_chars.items():
        url = f"http://a/{quote(c)}"
        with pytest.raises(e):
            create_file(url)


@pytest.mark.filesystem
@pytest.mark.skipif(platform.system() != "Windows", reason="Skipping if not windows.")
def test_get_fs_and_path_invalid_windows():
    invalid_chars = {
        "\x00": ValueError,
        "\\": FileExistsError,
        "/": OSError,
        ":": OSError,
        "*": OSError,
        "?": OSError,
        '"': OSError,
        "<": OSError,
        ">": OSError,
        "|": OSError,
    }
    for c, e in invalid_chars.items():
        url = f"http://a/{quote(c)}"
        with pytest.raises(e):
            create_file(url)
