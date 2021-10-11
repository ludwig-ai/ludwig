from typing import List
import pytest

import torch

from ludwig.utils.torch_utils import sequence_length_2D, sequence_length_3D


@pytest.mark.parametrize('input_sequence', [[[0, 1, 1], [2, 0, 0], [3, 3, 3]]])
@pytest.mark.parametrize('expected_output', [[2, 1, 3]])
def test_sequence_length_2D(
        input_sequence: List[List[int]],
        expected_output: List[int]
):
    output_seq_length = sequence_length_2D(torch.tensor(input_sequence))
    assert torch.equal(torch.tensor(expected_output), output_seq_length)


@pytest.mark.parametrize(
    'input_sequence', [[[[-1, 0, 1], [1, -2, 0]], [[0, 0, 0], [3, 0, -2]]]])
@pytest.mark.parametrize('expected_output', [[2, 1]])
def test_sequence_length_3D(
        input_sequence: List[List[List[int]]],
        expected_output: List[int]
):
    input_sequence = torch.tensor(input_sequence, dtype=torch.int32)
    expected_output = torch.tensor(expected_output, dtype=torch.int32)
    output_seq_length = sequence_length_3D(input_sequence)
    assert torch.equal(expected_output, output_seq_length)
