from typing import List

import pytest
import torch

from ludwig.modules.embedding_modules import Embed, EmbedSequence, EmbedSet, EmbedWeighted, TokenAndPositionEmbedding
from ludwig.utils.torch_utils import get_torch_device

DEVICE = get_torch_device()


@pytest.mark.parametrize("vocab", [["a", "b", "c"]])
@pytest.mark.parametrize("embedding_size", [2])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_embed(
    vocab: List[str],
    embedding_size: int,
    representation: str,
):
    embed = Embed(
        vocab=vocab,
        embedding_size=embedding_size,
        representation=representation,
    ).to(DEVICE)
    inputs = torch.randint(0, 2, size=(2, 1)).bool().to(DEVICE)
    outputs = embed(inputs)
    assert outputs.shape[1:] == embed.output_shape


@pytest.mark.parametrize("vocab", [["a", "b", "c", "d"]])
@pytest.mark.parametrize("embedding_size", [3])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_embed_set(
    vocab: List[str],
    embedding_size: int,
    representation: str,
):
    embed = EmbedSet(
        vocab=vocab,
        embedding_size=embedding_size,
        representation=representation,
    ).to(DEVICE)
    inputs = torch.randint(0, 2, size=(2, len(vocab))).bool().to(DEVICE)
    outputs = embed(inputs)
    assert outputs.shape[1:] == embed.output_shape


@pytest.mark.parametrize("vocab", [["a", "b", "c", "d", "e", "f", "g", "h"]])
@pytest.mark.parametrize("embedding_size", [5, 10])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_embed_weighted(
    vocab: List[str],
    embedding_size: int,
    representation: str,
):
    embed_weighted = EmbedWeighted(vocab=vocab, embedding_size=embedding_size, representation=representation).to(DEVICE)
    inputs = torch.randint(0, 2, size=(2, len(vocab))).bool().to(DEVICE)
    outputs = embed_weighted(inputs)
    assert outputs.shape[1:] == embed_weighted.output_shape


@pytest.mark.parametrize("vocab", [["a", "b", "c"]])
@pytest.mark.parametrize("embedding_size", [2])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_embed_sequence(
    vocab: List[str],
    embedding_size: int,
    representation: str,
):
    embed = EmbedSequence(
        vocab=vocab,
        embedding_size=embedding_size,
        max_sequence_length=10,
        representation=representation,
    ).to(DEVICE)
    inputs = torch.randint(0, 2, size=(2, 10)).to(DEVICE)
    outputs = embed(inputs)
    assert outputs.shape[1:] == embed.output_shape


@pytest.mark.parametrize("vocab", [["a", "b", "c"]])
@pytest.mark.parametrize("embedding_size", [10])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_token_and_position_embedding(
    vocab: List[str],
    embedding_size: int,
    representation: str,
):
    embed = TokenAndPositionEmbedding(
        vocab=vocab,
        embedding_size=embedding_size,
        max_sequence_length=10,
        representation=representation,
    ).to(DEVICE)
    inputs = torch.randint(0, 2, size=(2, 10)).to(DEVICE)
    outputs = embed(inputs)
    assert outputs.shape[1:] == embed.output_shape
