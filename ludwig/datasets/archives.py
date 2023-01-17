#! /usr/bin/env python
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
import gzip
import logging
import os
import shutil
import tarfile
from enum import Enum
from typing import List, Optional
from zipfile import ZipFile

from ludwig.utils.fs_utils import upload_output_directory

logger = logging.getLogger(__name__)


class ArchiveType(str, Enum):
    """The type of file archive."""

    UNKNOWN = "unknown"
    ZIP = "zip"
    GZIP = "gz"
    TAR = "tar"
    TAR_ZIP = "tar.z"
    TAR_BZ2 = "tar.bz2"
    TAR_GZ = "tar.gz"


def infer_archive_type(archive_path):
    """Try to infer archive type from file extension."""
    # Get the path extension including multiple extensions, ex. ".tar.gz"
    extension = ".".join(["", *os.path.basename(archive_path).split(".")[1:]])
    extension = extension.lower()
    if extension.endswith(".tar.z") or extension.endswith(".tar.zip"):
        return ArchiveType.TAR_ZIP
    elif extension.endswith(".tar.bz2") or extension.endswith(".tbz2"):
        return ArchiveType.TAR_BZ2
    elif extension.endswith(".tar.gz") or extension.endswith(".tgz"):
        return ArchiveType.TAR_GZ
    elif extension.endswith(".tar"):
        return ArchiveType.TAR
    elif extension.endswith(".zip") or extension.endswith(".zipx"):
        return ArchiveType.ZIP
    elif extension.endswith(".gz") or extension.endswith(".gzip"):
        return ArchiveType.GZIP
    else:
        return ArchiveType.UNKNOWN


def is_archive(path):
    """Does this path a supported archive type."""
    return infer_archive_type(path) != ArchiveType.UNKNOWN


def list_archive(archive_path, archive_type: Optional[ArchiveType] = None) -> List[str]:
    """Return list of files extracted in an archive (without extracting them)."""
    if archive_type is None:
        archive_type = infer_archive_type(archive_path)
    if archive_type == ArchiveType.UNKNOWN:
        logger.error(
            f"Could not infer type of archive {archive_path}.  May be an unsupported archive type."
            "Specify archive_type in the dataset config if this file has an unknown file extension."
        )
        return []
    if archive_type == ArchiveType.ZIP:
        with ZipFile(archive_path) as zfile:
            return zfile.namelist()
    elif archive_type == ArchiveType.GZIP:
        return [".".join(archive_path.split(".")[:-1])]  # Path minus the .gz extension
    elif archive_type in {ArchiveType.TAR, ArchiveType.TAR_ZIP, ArchiveType.TAR_BZ2, ArchiveType.TAR_GZ}:
        with tarfile.open(archive_path) as tar_file:
            return tar_file.getnames()
    else:
        logger.error(f"Unsupported archive: {archive_path}")
    return []


def extract_archive(archive_path: str, archive_type: Optional[ArchiveType] = None) -> List[str]:
    """Extracts files from archive (into the same directory), returns a list of extracted files.

    Args:
        archive_path - The full path to the archive.

    Returns A list of the files extracted.
    """
    if archive_type is None:
        archive_type = infer_archive_type(archive_path)
    if archive_type == ArchiveType.UNKNOWN:
        logger.error(
            f"Could not infer type of archive {archive_path}.  May be an unsupported archive type."
            "Specify archive_type in the dataset config if this file has an unknown file extension."
        )
        return []
    archive_directory = os.path.dirname(archive_path)
    directory_contents_before = os.listdir(archive_directory)
    with upload_output_directory(archive_directory) as (tmpdir, _):
        if archive_type == ArchiveType.ZIP:
            with ZipFile(archive_path) as zfile:
                zfile.extractall(tmpdir)
        elif archive_type == ArchiveType.GZIP:
            gzip_content_file = ".".join(archive_path.split(".")[:-1])  # Path minus the .gz extension
            with gzip.open(archive_path) as gzfile:
                with open(os.path.join(tmpdir, gzip_content_file), "wb") as output:
                    shutil.copyfileobj(gzfile, output)
        elif archive_type in {ArchiveType.TAR, ArchiveType.TAR_ZIP, ArchiveType.TAR_BZ2, ArchiveType.TAR_GZ}:
            with tarfile.open(archive_path) as tar_file:

                def is_within_directory(directory, target):
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")

                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar_file, path=tmpdir)
        else:
            logger.error(f"Unsupported archive: {archive_path}")
    directory_contents_after = set(os.listdir(archive_directory))
    return directory_contents_after.difference(directory_contents_before)
