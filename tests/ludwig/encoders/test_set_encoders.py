from typing import List
import pytest

import torch

from ludwig.encoders.set_encoders import SetSparseEncoder


@pytest.mark.parametrize('vocab', [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']])
@pytest.mark.parametrize('embedding_size', [10])
@pytest.mark.parametrize('representation', ['sparse'])
def test_set_encoder(
        vocab: List[str],
        embedding_size: int,
        representation: str
):
    set_encoder = SetSparseEncoder(
        vocab=vocab,
        representation=representation,
        embedding_size=embedding_size
    )
    inputs = torch.randint(0, 2, size=(2, len(vocab))).bool()
    outputs = set_encoder(inputs)
    assert outputs.shape[1:] == set_encoder.output_shape
