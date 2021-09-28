from typing import List
import pytest

import torch

from ludwig.encoders.bag_encoders import BagEmbedWeightedEncoder


@pytest.mark.parametrize('vocab', [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']])
@pytest.mark.parametrize('embedding_size', [10])
@pytest.mark.parametrize('representation', ['dense', 'sparse'])
def test_set_encoder(
        vocab: List[str],
        embedding_size: int,
        representation: str
):
    bag_encoder = BagEmbedWeightedEncoder(
        vocab=vocab,
        representation=representation,
        embedding_size=embedding_size,
    )
    inputs = torch.randint(0, 9, size=(2, len(vocab)))
    outputs = bag_encoder(inputs)
    assert outputs.shape[1:] == bag_encoder.output_shape
