from typing import List

import pytest
import torch

from ludwig.encoders.bag_encoders import BagEmbedWeightedEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("vocab", [["a", "b", "c", "d", "e", "f", "g", "h"]])
@pytest.mark.parametrize("embedding_size", [10])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_set_encoder(vocab: List[str], embedding_size: int, representation: str):
    bag_encoder = BagEmbedWeightedEncoder(
        vocab=vocab,
        representation=representation,
        embedding_size=embedding_size,
    ).to(DEVICE)
    inputs = torch.randint(0, 9, size=(2, len(vocab))).to(DEVICE)
    outputs = bag_encoder(inputs)
    assert outputs.shape[1:] == bag_encoder.output_shape
