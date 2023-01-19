import logging
import os
import platform
import tempfile
from urllib.parse import quote

import pytest

from ludwig.utils.fs_utils import get_fs_and_path, safe_move_directory

logger = logging.getLogger(__name__)


def create_file(url):
    _, path = get_fs_and_path(url)
    logger.info(f"saving url '{url}' to path '{path}'")
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


@pytest.mark.filesystem
def test_safe_move_directory(tmpdir):
    src_dir = os.path.join(tmpdir, "src")
    dst_dir = os.path.join(tmpdir, "dst")

    os.mkdir(src_dir)
    os.mkdir(dst_dir)

    with open(os.path.join(src_dir, "file.txt"), "w") as f:
        f.write("test")

    safe_move_directory(src_dir, dst_dir)

    assert not os.path.exists(src_dir)
    assert os.path.exists(os.path.join(dst_dir, "file.txt"))
    with open(os.path.join(dst_dir, "file.txt")) as f:
        assert f.read() == "test"
