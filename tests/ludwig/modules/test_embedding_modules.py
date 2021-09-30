import pytest
import torch
from typing import List

from ludwig.modules.embedding_modules import Embed, EmbedSequence,\
    EmbedWeighted, TokenAndPositionEmbedding


@pytest.mark.parametrize('vocab', [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']])
@pytest.mark.parametrize('embedding_size', [5, 10])
@pytest.mark.parametrize('representation', ['dense'])
@pytest.mark.parametrize('force_embedding_size', [False])
def test_embed(
        vocab: List[str],
        embedding_size: int,
        representation: str,
        force_embedding_size: bool
):
    embed = Embed(
        vocab=vocab,
        embedding_size=embedding_size,
        representation=representation,
        force_embedding_size=force_embedding_size
    )
    inputs = torch.randint(0, 2, size=(2, len(vocab)))
    outputs = embed(inputs)
    print(f'OUTPUTS SHAPE: {outputs.shape}')
    assert outputs.shape[1:] == embed.output_shape
