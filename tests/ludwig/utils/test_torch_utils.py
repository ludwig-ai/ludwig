import contextlib
import os
from typing import List
from unittest.mock import patch

import pytest
import torch

from ludwig.utils.torch_utils import (
    _get_torch_init_params,
    _set_torch_init_params,
    initialize_pytorch,
    sequence_length_2D,
    sequence_length_3D,
)


@pytest.mark.parametrize("input_sequence", [[[0, 1, 1], [2, 0, 0], [3, 3, 3]]])
@pytest.mark.parametrize("expected_output", [[3, 2, 3]])
def test_sequence_length_2D(input_sequence: List[List[int]], expected_output: List[int]):
    output_seq_length = sequence_length_2D(torch.tensor(input_sequence))
    assert torch.equal(torch.tensor(expected_output), output_seq_length)


@pytest.mark.parametrize("input_sequence", [[[[-1, 0, 1], [1, -2, 0]], [[0, 0, 0], [3, 0, -2]]]])
@pytest.mark.parametrize("expected_output", [[2, 1]])
def test_sequence_length_3D(input_sequence: List[List[List[int]]], expected_output: List[int]):
    input_sequence = torch.tensor(input_sequence, dtype=torch.int32)
    expected_output = torch.tensor(expected_output, dtype=torch.int32)
    output_seq_length = sequence_length_3D(input_sequence)
    assert torch.equal(expected_output, output_seq_length)


@contextlib.contextmanager
def clean_params():
    prev = _get_torch_init_params()
    try:
        _set_torch_init_params(None)
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        yield
    finally:
        _set_torch_init_params(prev)


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_only_once(mock_torch):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4
    with clean_params():
        # During first time initialization, set pytorch parallelism
        initialize_pytorch(allow_parallel_threads=False)
        mock_torch.set_num_threads.assert_called_once()
        mock_torch.set_num_interop_threads.assert_called_once()

        # Reset call counts on all threading calls
        mock_torch.reset_mock()

        # In the second call to initialization, avoid calling these methods again, as pytorch
        # will raise an exception
        initialize_pytorch(allow_parallel_threads=False)
        mock_torch.set_num_threads.assert_not_called()
        mock_torch.set_num_interop_threads.assert_not_called()

    # No GPUs were specified, so this should not have been called even once
    mock_torch.cuda.memory.set_per_process_memory_fraction.assert_not_called()


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_with_gpu_list(mock_torch):
    # For test purposes, these devices can be anything, we just need to be able to uniquely
    # identify them.
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4
    with clean_params():
        initialize_pytorch(gpus=[1, 2])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_with_gpu_string(mock_torch):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4
    with clean_params():
        initialize_pytorch(gpus="1,2")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,2"


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_with_gpu_int(mock_torch):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4
    with clean_params():
        initialize_pytorch(gpus=1)
    mock_torch.cuda.set_device.assert_called_with(1)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_without_gpu(mock_torch):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4
    with clean_params():
        initialize_pytorch(gpus=-1)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_with_distributed(mock_torch):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4

    with clean_params():
        initialize_pytorch(local_rank=1, local_size=4)

    mock_torch.cuda.set_device.assert_called_with(1)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@patch("ludwig.utils.torch_utils.warnings")
@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_with_distributed_bad_local_rank(mock_torch, mock_warnings):
    """In this scenario, the local_size 5 is out of the bounds of the GPU indices."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4

    with clean_params():
        initialize_pytorch(local_rank=1, local_size=5)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    mock_warnings.warn.assert_called()


@patch("ludwig.utils.torch_utils.torch")
def test_initialize_pytorch_with_distributed_explicit_gpus(mock_torch):
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 4

    with clean_params():
        initialize_pytorch(gpus="-1", local_rank=1, local_size=4)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
