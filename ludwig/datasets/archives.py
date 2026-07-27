#! /usr/bin/env python
# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
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
from zipfile import ZipFile

from ludwig.error import UnsafeArchiveError
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


def list_archive(archive_path, archive_type: ArchiveType | None = None) -> list[str]:
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


def is_within_directory(directory: str, target: str) -> bool:
    """Does target resolve to a location inside directory (or to directory itself)?

    Args:
        directory: The containing directory.
        target: The path to test for containment.

    Returns:
        True if target is directory or lies beneath it, False otherwise.

    Example:
        is_within_directory("/data/dest", "/data/dest/train.csv") -> True
        is_within_directory("/data/dest", "/data/dest-evil/train.csv") -> False
    """
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    try:
        # commonpath compares path components. os.path.commonprefix, which this previously used, compares characters,
        # so it reports "/data/dest-evil" as living inside "/data/dest".
        return os.path.commonpath([abs_directory, abs_target]) == abs_directory
    except ValueError:
        # Raised when the paths cannot be compared at all, e.g. different drives on Windows.
        return False


def _resolve_link_destination(path: str, member: tarfile.TarInfo) -> str:
    """Return the path a link member would point at once extracted under path.

    Args:
        path: The destination directory the archive is extracted into.
        member: A tar member for which issym() or islnk() is True.

    Returns:
        The absolute path the link would resolve to.

    Example:
        For a symlink member named "a/link" with linkname "../../etc", under path "/dest",
        this returns "/etc".
    """
    if member.issym():
        # A symlink's target is interpreted relative to the directory containing the link.
        return os.path.join(path, os.path.dirname(member.name), member.linkname)
    # A hard link's target names another member of the archive, so it resolves from the archive root.
    return os.path.join(path, member.linkname)


def safe_extract_tar(tar: tarfile.TarFile, path: str = ".", members=None, *, numeric_owner: bool = False) -> None:
    """Extract a tar archive, refusing any member that would write outside path.

    Checking member names alone is not enough: a member name is validated before extraction, when a symlink created
    earlier in the same archive does not exist yet. An archive can therefore ship a symlink pointing outside the
    destination followed by a regular file whose name resolves "through" that link, and both members pass a
    name-only check. Link targets are validated here, and extraction additionally uses the stdlib "data" filter,
    which independently refuses escaping links.

    Args:
        tar: The open tar archive to extract.
        path: Destination directory to extract into.
        members: Optional subset of members to extract, as accepted by TarFile.extractall.
        numeric_owner: Whether to use numeric owner/group ids rather than names.

    Returns:
        None. Files are written under path.

    Raises:
        UnsafeArchiveError: If any member name or link target escapes path.

    Example:
        with tarfile.open("dataset.tar.gz") as tar:
            safe_extract_tar(tar, path="/data/dest")
    """
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise UnsafeArchiveError(
                f"Refusing to extract archive member {member.name!r}: it resolves to {os.path.abspath(member_path)!r}, "
                f"which is outside the destination directory {os.path.abspath(path)!r}."
            )
        if member.issym() or member.islnk():
            link_destination = _resolve_link_destination(path, member)
            if not is_within_directory(path, link_destination):
                link_kind = "symlink" if member.issym() else "hard link"
                raise UnsafeArchiveError(
                    f"Refusing to extract archive member {member.name!r}: it is a {link_kind} pointing at "
                    f"{os.path.abspath(link_destination)!r}, which is outside the destination directory "
                    f"{os.path.abspath(path)!r}."
                )

    # filter="data" is the stdlib default from Python 3.14 on. Passing it explicitly keeps the same protections on
    # 3.12 and 3.13, where the default is still the unrestricted "fully_trusted" behavior.
    tar.extractall(path, members, numeric_owner=numeric_owner, filter="data")


def extract_archive(archive_path: str, archive_type: ArchiveType | None = None) -> list[str]:
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
            # Join on the file name, not the full path: os.path.join discards tmpdir when its second argument is
            # absolute. That lands in the right place only because tmpdir is the archive directory itself for local
            # paths; for a remote destination tmpdir is a staging directory whose contents are uploaded afterwards,
            # and writing outside it would silently drop the extracted file.
            gzip_content_name = ".".join(os.path.basename(archive_path).split(".")[:-1])  # Name minus the .gz suffix
            with gzip.open(archive_path) as gzfile:
                with open(os.path.join(tmpdir, gzip_content_name), "wb") as output:
                    shutil.copyfileobj(gzfile, output)
        elif archive_type in {ArchiveType.TAR, ArchiveType.TAR_ZIP, ArchiveType.TAR_BZ2, ArchiveType.TAR_GZ}:
            with tarfile.open(archive_path) as tar_file:
                safe_extract_tar(tar_file, path=tmpdir)
        else:
            logger.error(f"Unsupported archive: {archive_path}")
    directory_contents_after = set(os.listdir(archive_directory))
    return directory_contents_after.difference(directory_contents_before)
