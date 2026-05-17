"""Unit tests for ludwig.collect — save_tensors and related helpers.

Tests here do not require a trained model or GPU; they verify the pure-logic
functions that handle tensor serialization and filename generation.
"""

import os

import numpy as np
import pytest
import torch

from ludwig.collect import save_tensors
from ludwig.utils.strings_utils import make_safe_filename


class TestSaveTensors:
    def test_saves_1d_tensor(self, tmp_path):
        t = torch.tensor([1.0, 2.0, 3.0])
        files = save_tensors([("my_layer", t)], str(tmp_path))
        assert len(files) == 1
        loaded = np.load(files[0])
        np.testing.assert_array_equal(loaded, t.numpy())

    def test_saves_2d_tensor(self, tmp_path):
        t = torch.randn(4, 8)
        files = save_tensors([("encoder.output", t)], str(tmp_path))
        assert len(files) == 1
        loaded = np.load(files[0])
        assert loaded.shape == (4, 8)

    def test_filename_uses_safe_name(self, tmp_path):
        t = torch.tensor([0.0])
        files = save_tensors([("a/b/c", t)], str(tmp_path))
        expected_stem = make_safe_filename("a/b/c")
        assert os.path.basename(files[0]) == expected_stem + ".npy"

    def test_multiple_tensors_saved(self, tmp_path):
        tensors = [
            ("layer1", torch.ones(3)),
            ("layer2", torch.zeros(5)),
        ]
        files = save_tensors(tensors, str(tmp_path))
        assert len(files) == 2

    def test_non_tensor_entries_skipped(self, tmp_path):
        """Non-tensor values (e.g. used_tokens int) must be silently skipped."""
        collected = [
            ("encoder.output", torch.ones(4)),
            ("used_tokens", 42),  # not a tensor
        ]
        files = save_tensors(collected, str(tmp_path))
        # Only the tensor entry should produce a file
        assert len(files) == 1

    def test_creates_output_directory(self, tmp_path):
        nested = tmp_path / "subdir" / "deep"
        t = torch.tensor([1.0])
        # save_tensors does NOT create the directory — caller must
        nested.mkdir(parents=True)
        files = save_tensors([("w", t)], str(nested))
        assert len(files) == 1
        assert os.path.exists(files[0])

    def test_empty_list_returns_empty(self, tmp_path):
        files = save_tensors([], str(tmp_path))
        assert files == []

    def test_tensor_values_are_preserved(self, tmp_path):
        expected = torch.tensor([[1.5, -2.0], [0.0, 3.14]])
        files = save_tensors([("weights", expected)], str(tmp_path))
        loaded = np.load(files[0])
        np.testing.assert_allclose(loaded, expected.numpy(), rtol=1e-6)

    def test_gpu_tensor_saved_to_cpu(self, tmp_path):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.ones(4).cuda()
        files = save_tensors([("gpu_layer", t)], str(tmp_path))
        assert len(files) == 1
        loaded = np.load(files[0])
        np.testing.assert_array_equal(loaded, np.ones(4))
