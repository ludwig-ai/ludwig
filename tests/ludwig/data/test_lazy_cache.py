"""Tests for lazy media caching utilities.

Covers:
- ``_cache_audio_column_to_disk`` in ``ludwig.features.audio_feature``
- ``_cache_image_column_to_disk`` in ``ludwig.features.image_feature``
- ``resolve_lazy_cache_dir`` / ``get_default_lazy_cache_dir`` in ``ludwig.data.lazy_utils``
"""

from __future__ import annotations

import importlib
import os
import wave
from pathlib import Path

import numpy as np
import pytest

from ludwig.data.lazy_utils import resolve_lazy_cache_dir

# ---------------------------------------------------------------------------
# Conditional skip markers
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None

requires_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
requires_pil = pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(path: str, num_samples: int = 1000, num_channels: int = 1, sample_rate: int = 16_000) -> None:
    """Write a minimal valid WAV file to *path*."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        data = (np.zeros(num_samples * num_channels, dtype=np.int16)).tobytes()
        wf.writeframes(data)


def _write_png(path: str, width: int = 8, height: int = 8) -> None:
    """Write a minimal valid PNG file to *path*."""
    from PIL import Image as PILImage

    PILImage.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(path, format="PNG")


# ---------------------------------------------------------------------------
# Audio caching tests
# ---------------------------------------------------------------------------


@requires_torch
class TestCacheAudioToDisk:
    """Tests for ``_cache_audio_column_to_disk``."""

    def test_reuses_existing_hf_path(self, tmp_path: Path) -> None:
        """If audio dict has a 'path' pointing to an existing WAV, reuse it."""
        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        wav_path = str(tmp_path / "existing.wav")
        _write_wav(wav_path)

        entry = {"array": np.zeros(16_000, dtype=np.float32), "sampling_rate": 16_000, "path": wav_path}
        result = _cache_audio_column_to_disk([entry], tmp_path, "audio")

        assert result == [wav_path]
        # No extra files should have been created in cache_dir besides existing.wav
        wav_files = list(tmp_path.glob("audio_*.wav"))
        assert len(wav_files) == 0

    def test_writes_new_wav_for_dict_without_path(self, tmp_path: Path) -> None:
        """Audio dict without existing 'path' gets written to cache_dir."""
        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        entry = {"array": np.zeros(8_000, dtype=np.float32), "sampling_rate": 8_000}
        result = _cache_audio_column_to_disk([entry], tmp_path, "audio")

        assert len(result) == 1
        dest = result[0]
        assert os.path.isfile(dest)
        assert dest.endswith(".wav")

    def test_writes_new_wav_for_dict_with_nonexistent_path(self, tmp_path: Path) -> None:
        """Audio dict whose 'path' does not exist on disk triggers a new WAV write."""
        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        entry = {
            "array": np.zeros(8_000, dtype=np.float32),
            "sampling_rate": 8_000,
            "path": "/nonexistent/path/audio.wav",
        }
        result = _cache_audio_column_to_disk([entry], tmp_path, "audio")

        assert len(result) == 1
        assert os.path.isfile(result[0])
        assert result[0] != "/nonexistent/path/audio.wav"

    def test_idempotent_does_not_overwrite(self, tmp_path: Path) -> None:
        """Running cache twice does not overwrite existing files."""

        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        entry = {"array": np.zeros(8_000, dtype=np.float32), "sampling_rate": 16_000}
        result1 = _cache_audio_column_to_disk([entry], tmp_path, "audio")
        mtime_before = os.path.getmtime(result1[0])

        result2 = _cache_audio_column_to_disk([entry], tmp_path, "audio")
        mtime_after = os.path.getmtime(result2[0])

        assert result1 == result2
        assert mtime_before == mtime_after

    def test_tensor_input_cached(self, tmp_path: Path) -> None:
        """torch.Tensor input is written to a WAV file."""
        import torch

        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        waveform = torch.zeros(1, 8_000)
        result = _cache_audio_column_to_disk([waveform], tmp_path, "audio")

        assert len(result) == 1
        assert os.path.isfile(result[0])

    def test_1d_tensor_input_cached(self, tmp_path: Path) -> None:
        """1-D torch.Tensor is promoted to (1, samples) before saving."""
        import torch

        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        waveform = torch.zeros(8_000)  # shape (8000,) — no channel dim
        result = _cache_audio_column_to_disk([waveform], tmp_path, "audio")

        assert len(result) == 1
        assert os.path.isfile(result[0])

    def test_multiple_entries_produce_ordered_paths(self, tmp_path: Path) -> None:
        """Multiple entries produce one path per entry in order."""
        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        n = 5
        entries = [{"array": np.zeros(4_000, dtype=np.float32), "sampling_rate": 16_000} for _ in range(n)]
        result = _cache_audio_column_to_disk(entries, tmp_path, "audio")

        assert len(result) == n
        for path in result:
            assert os.path.isfile(path)

    def test_invalid_entry_type_raises(self, tmp_path: Path) -> None:
        """An unrecognised entry type raises ValueError."""
        from ludwig.features.audio_feature import _cache_audio_column_to_disk

        with pytest.raises(ValueError, match="unrecognised type"):
            _cache_audio_column_to_disk(["not_a_dict_or_tensor"], tmp_path, "audio")


# ---------------------------------------------------------------------------
# Image caching tests
# ---------------------------------------------------------------------------


@requires_pil
class TestCacheImageToDisk:
    """Tests for ``_cache_image_column_to_disk``."""

    def test_reuses_pil_filename(self, tmp_path: Path) -> None:
        """PIL Image with .filename pointing to existing file → reuse that path."""
        from PIL import Image as PILImage

        from ludwig.features.image_feature import _cache_image_column_to_disk

        png_path = str(tmp_path / "existing.png")
        _write_png(png_path)

        img = PILImage.open(png_path)
        # PIL sets .filename on images opened from disk
        result = _cache_image_column_to_disk([img], tmp_path, "image")

        assert result == [png_path]
        # No cached copies should have been written
        cached = list(tmp_path.glob("image_*.png"))
        assert len(cached) == 0

    def test_writes_png_for_pil_without_filename(self, tmp_path: Path) -> None:
        """In-memory PIL Image (no filename) → written to cache_dir as PNG."""
        from PIL import Image as PILImage

        from ludwig.features.image_feature import _cache_image_column_to_disk

        img = PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        result = _cache_image_column_to_disk([img], tmp_path, "image")

        assert len(result) == 1
        assert os.path.isfile(result[0])
        assert result[0].endswith(".png")

    def test_bytes_input_cached(self, tmp_path: Path) -> None:
        """Raw bytes image input is decoded and saved as PNG."""
        import io

        from PIL import Image as PILImage

        from ludwig.features.image_feature import _cache_image_column_to_disk

        buf = io.BytesIO()
        PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
        raw_bytes = buf.getvalue()

        result = _cache_image_column_to_disk([raw_bytes], tmp_path, "image")

        assert len(result) == 1
        assert os.path.isfile(result[0])

    def test_numpy_array_cached(self, tmp_path: Path) -> None:
        """A numpy (H, W, 3) array is saved as PNG."""
        from ludwig.features.image_feature import _cache_image_column_to_disk

        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        result = _cache_image_column_to_disk([arr], tmp_path, "image")

        assert len(result) == 1
        assert os.path.isfile(result[0])

    def test_numpy_chw_array_cached(self, tmp_path: Path) -> None:
        """A numpy (3, H, W) channel-first array is transposed and saved."""
        from ludwig.features.image_feature import _cache_image_column_to_disk

        arr = np.zeros((3, 8, 8), dtype=np.uint8)
        result = _cache_image_column_to_disk([arr], tmp_path, "image")

        assert len(result) == 1
        assert os.path.isfile(result[0])

    def test_hf_dict_with_bytes_cached(self, tmp_path: Path) -> None:
        """HuggingFace-style dict with 'bytes' key is decoded and saved."""
        import io

        from PIL import Image as PILImage

        from ludwig.features.image_feature import _cache_image_column_to_disk

        buf = io.BytesIO()
        PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(buf, format="PNG")
        entry = {"bytes": buf.getvalue(), "path": None}

        result = _cache_image_column_to_disk([entry], tmp_path, "image")

        assert len(result) == 1
        assert os.path.isfile(result[0])

    def test_hf_dict_with_existing_path_reused(self, tmp_path: Path) -> None:
        """HuggingFace-style dict with a valid 'path' → reuse that path."""
        from ludwig.features.image_feature import _cache_image_column_to_disk

        png_path = str(tmp_path / "hf_cached.png")
        _write_png(png_path)

        entry = {"bytes": None, "path": png_path}
        result = _cache_image_column_to_disk([entry], tmp_path, "image")

        assert result == [png_path]

    def test_idempotent(self, tmp_path: Path) -> None:
        """Running cache twice does not overwrite existing files."""
        from PIL import Image as PILImage

        from ludwig.features.image_feature import _cache_image_column_to_disk

        img = PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        result1 = _cache_image_column_to_disk([img], tmp_path, "image")
        mtime_before = os.path.getmtime(result1[0])

        result2 = _cache_image_column_to_disk([img], tmp_path, "image")
        mtime_after = os.path.getmtime(result2[0])

        assert result1 == result2
        assert mtime_before == mtime_after

    def test_multiple_entries(self, tmp_path: Path) -> None:
        """Multiple entries produce one path per entry in order."""
        from PIL import Image as PILImage

        from ludwig.features.image_feature import _cache_image_column_to_disk

        imgs = [PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)) for _ in range(4)]
        result = _cache_image_column_to_disk(imgs, tmp_path, "image")

        assert len(result) == 4
        for path in result:
            assert os.path.isfile(path)

    def test_invalid_entry_type_raises(self, tmp_path: Path) -> None:
        """An unrecognised entry type raises ValueError."""
        from ludwig.features.image_feature import _cache_image_column_to_disk

        with pytest.raises(ValueError, match="unrecognised type"):
            _cache_image_column_to_disk([12345], tmp_path, "image")


# ---------------------------------------------------------------------------
# resolve_lazy_cache_dir / get_default_lazy_cache_dir tests
# ---------------------------------------------------------------------------


class TestResolveLazyCacheDir:
    """Tests for ``resolve_lazy_cache_dir`` and ``get_default_lazy_cache_dir``."""

    def test_uses_explicit_param(self, tmp_path: Path) -> None:
        """When cache_dir_param is given, the resolved path is inside it."""
        explicit = str(tmp_path / "my_cache")
        result = resolve_lazy_cache_dir(explicit, "audio_feat")

        assert result == Path(explicit) / "audio_feat"
        assert result.is_dir()

    def test_falls_back_to_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When cache_dir_param is None, the default root is used.

        We monkeypatch ``Path.home`` inside the lazy_utils module so the test
        does not pollute the real ``~/.cache`` and does not require a module
        reload (which would invalidate other imported names like ``LazyColumn``).
        """
        import ludwig.data.lazy_utils as lu

        monkeypatch.setattr(lu, "get_default_lazy_cache_dir", lambda: tmp_path)
        result = lu.resolve_lazy_cache_dir(None, "my_feat")

        assert result == tmp_path / "my_feat"
        assert result.is_dir()

    def test_creates_directory(self, tmp_path: Path) -> None:
        """resolve_lazy_cache_dir creates the directory if it does not exist."""
        nested = str(tmp_path / "does" / "not" / "exist")
        result = resolve_lazy_cache_dir(nested, "feat")

        assert result.is_dir()

    def test_get_default_lazy_cache_dir_creates_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_default_lazy_cache_dir creates a 'lazy_media' directory under home."""
        import ludwig.data.lazy_utils as lu

        # Redirect Path.home() inside the module so we don't pollute ~/.cache
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        root = lu.get_default_lazy_cache_dir()

        assert root.is_dir()
        assert root.name == "lazy_media"

    def test_feature_name_used_as_subdirectory(self, tmp_path: Path) -> None:
        """The feature name becomes the leaf directory under cache_dir_param."""
        result = resolve_lazy_cache_dir(str(tmp_path), "my_audio_feature")

        assert result.name == "my_audio_feature"
        assert result.parent == tmp_path
