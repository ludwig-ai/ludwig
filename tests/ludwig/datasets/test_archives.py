"""Tests that archive extraction cannot write outside its destination directory.

Regression coverage for two escapes in the tar extraction path:
  1. A symlink member pointing outside the destination, followed by a regular file written through it. Validating
     member names alone misses this, because the symlink does not exist yet at validation time.
  2. os.path.commonprefix treating "<dest>-evil" as living inside "<dest>", since it compares characters rather than
     path components.
"""

import contextlib
import gzip
import io
import os
import tarfile

import pytest

from ludwig.datasets import archives
from ludwig.datasets.archives import extract_archive, is_within_directory, safe_extract_tar
from ludwig.error import UnsafeArchiveError


def _add_symlink(tar: tarfile.TarFile, name: str, linkname: str) -> None:
    member = tarfile.TarInfo(name)
    member.type = tarfile.SYMTYPE
    member.linkname = linkname
    tar.addfile(member)


def _add_file(tar: tarfile.TarFile, name: str, content: bytes = b"payload\n") -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    tar.addfile(member, io.BytesIO(content))


class TestIsWithinDirectory:
    def test_accepts_path_inside_directory(self, tmpdir):
        dest = str(tmpdir)
        assert is_within_directory(dest, os.path.join(dest, "train.csv"))

    def test_accepts_directory_itself(self, tmpdir):
        dest = str(tmpdir)
        assert is_within_directory(dest, dest)

    def test_rejects_sibling_directory_sharing_a_name_prefix(self, tmpdir):
        """The commonprefix bug: "<dest>-evil" is not inside "<dest>"."""
        dest = os.path.join(str(tmpdir), "dest")
        sibling = os.path.join(str(tmpdir), "dest-evil", "loot.txt")
        assert not is_within_directory(dest, sibling)

    def test_rejects_parent_traversal(self, tmpdir):
        dest = os.path.join(str(tmpdir), "dest")
        assert not is_within_directory(dest, os.path.join(dest, "..", "..", "escaped.txt"))


class TestSafeExtractTar:
    def test_extracts_benign_archive(self, tmpdir):
        """Ordinary archives must still extract, including symlinks that stay inside."""
        archive_path = os.path.join(str(tmpdir), "benign.tar")
        dest = os.path.join(str(tmpdir), "dest")
        os.makedirs(dest)

        with tarfile.open(archive_path, "w") as tar:
            _add_file(tar, "data/train.csv", b"a,b\n1,2\n")
            _add_symlink(tar, "data/latest.csv", "train.csv")

        with tarfile.open(archive_path) as tar:
            safe_extract_tar(tar, path=dest)

        extracted = os.path.join(dest, "data", "train.csv")
        assert os.path.exists(extracted)
        assert open(extracted).read() == "a,b\n1,2\n"
        assert os.path.islink(os.path.join(dest, "data", "latest.csv"))

    def test_rejects_symlink_escape(self, tmpdir):
        """A symlink out of the destination plus a file written through it must be refused.

        This is the bypass that name-only validation missed: both member names resolve inside the destination.
        """
        archive_path = os.path.join(str(tmpdir), "evil.tar")
        dest = os.path.join(str(tmpdir), "dest")
        outside = os.path.join(str(tmpdir), "outside")
        os.makedirs(dest)
        os.makedirs(outside)

        with tarfile.open(archive_path, "w") as tar:
            _add_symlink(tar, "escape", outside)
            _add_file(tar, "escape/pwned.txt")

        with tarfile.open(archive_path) as tar:
            with pytest.raises(UnsafeArchiveError, match="outside the destination directory"):
                safe_extract_tar(tar, path=dest)

        assert not os.path.exists(os.path.join(outside, "pwned.txt"))

    def test_rejects_relative_symlink_escape(self, tmpdir):
        """The same escape expressed with a relative link target rather than an absolute one."""
        archive_path = os.path.join(str(tmpdir), "evil_relative.tar")
        dest = os.path.join(str(tmpdir), "dest")
        outside = os.path.join(str(tmpdir), "outside")
        os.makedirs(dest)
        os.makedirs(outside)

        with tarfile.open(archive_path, "w") as tar:
            _add_symlink(tar, "escape", "../outside")
            _add_file(tar, "escape/pwned.txt")

        with tarfile.open(archive_path) as tar:
            with pytest.raises(UnsafeArchiveError, match="outside the destination directory"):
                safe_extract_tar(tar, path=dest)

        assert not os.path.exists(os.path.join(outside, "pwned.txt"))

    def test_rejects_hard_link_escape(self, tmpdir):
        archive_path = os.path.join(str(tmpdir), "evil_hardlink.tar")
        dest = os.path.join(str(tmpdir), "dest")
        os.makedirs(dest)

        with tarfile.open(archive_path, "w") as tar:
            member = tarfile.TarInfo("escape")
            member.type = tarfile.LNKTYPE
            member.linkname = "../../etc/passwd"
            tar.addfile(member)

        with tarfile.open(archive_path) as tar:
            with pytest.raises(UnsafeArchiveError, match="outside the destination directory"):
                safe_extract_tar(tar, path=dest)

    def test_rejects_parent_traversal_member_name(self, tmpdir):
        """The classic CVE-2007-4559 case, still refused."""
        archive_path = os.path.join(str(tmpdir), "traversal.tar")
        dest = os.path.join(str(tmpdir), "dest")
        os.makedirs(dest)

        with tarfile.open(archive_path, "w") as tar:
            _add_file(tar, "../escaped.txt")

        with tarfile.open(archive_path) as tar:
            with pytest.raises(UnsafeArchiveError, match="outside the destination directory"):
                safe_extract_tar(tar, path=dest)

        assert not os.path.exists(os.path.join(str(tmpdir), "escaped.txt"))

    def test_rejects_absolute_member_name(self, tmpdir):
        archive_path = os.path.join(str(tmpdir), "absolute.tar")
        dest = os.path.join(str(tmpdir), "dest")
        outside = os.path.join(str(tmpdir), "outside")
        os.makedirs(dest)
        os.makedirs(outside)

        with tarfile.open(archive_path, "w") as tar:
            _add_file(tar, os.path.join(outside, "pwned.txt"))

        with tarfile.open(archive_path) as tar:
            with pytest.raises(UnsafeArchiveError, match="outside the destination directory"):
                safe_extract_tar(tar, path=dest)

        assert not os.path.exists(os.path.join(outside, "pwned.txt"))


class TestExtractGzip:
    def test_writes_decompressed_file_beside_the_archive(self, tmpdir):
        """The local case, where the staging directory is the archive directory itself."""
        archive_directory = str(tmpdir)
        archive_path = os.path.join(archive_directory, "data.csv.gz")
        with gzip.open(archive_path, "wb") as gzfile:
            gzfile.write(b"a,b\n1,2\n")

        extracted = extract_archive(archive_path)

        assert "data.csv" in extracted
        assert open(os.path.join(archive_directory, "data.csv")).read() == "a,b\n1,2\n"

    def test_writes_into_a_separate_staging_directory(self, tmpdir, monkeypatch):
        """The remote case: the decompressed file must land in the staging directory that gets uploaded.

        Joining tmpdir with the archive's full path silently discarded tmpdir, so the file was written next to the
        source archive and never uploaded.
        """
        archive_directory = os.path.join(str(tmpdir), "archives")
        staging_directory = os.path.join(str(tmpdir), "staging")
        os.makedirs(archive_directory)
        os.makedirs(staging_directory)

        archive_path = os.path.join(archive_directory, "data.csv.gz")
        with gzip.open(archive_path, "wb") as gzfile:
            gzfile.write(b"a,b\n1,2\n")

        @contextlib.contextmanager
        def staging_output_directory(url):
            yield staging_directory, None

        monkeypatch.setattr(archives, "upload_output_directory", staging_output_directory)

        extract_archive(archive_path)

        staged_file = os.path.join(staging_directory, "data.csv")
        assert os.path.exists(staged_file), "decompressed file was not written into the staging directory"
        assert open(staged_file).read() == "a,b\n1,2\n"
