import pytest

import torch

from ludwig.encoders.category_encoders import CategoricalEmbedEncoder
from ludwig.encoders.category_encoders import CategoricalSparseEncoder


@pytest.mark.parametrize('vocab', [['red', 'orange', 'yellow', 'green', 'blue', 'violet'],
                                   ['a', 'b', 'c']])
@pytest.mark.parametrize('embedding_size', [4, 6, 10])
def test_categorical_dense_encoder(vocab, embedding_size):
    dense_encoder = CategoricalEmbedEncoder(vocab=vocab, embedding_size=embedding_size)
    inputs = torch.randint(len(vocab), (10,))  # Chooses 10 items from vocab with replacement.
    inputs = torch.unsqueeze(inputs, 1)
    outputs = dense_encoder(inputs)
    # Ensures that if embedding_size is larger than len(vocab), len(vocab) is used as the embedding size.
    assert outputs.shape[-1] == min(embedding_size, len(vocab))
    # Ensures output shape matches encoder expected output shape.
    assert outputs.shape[1:] == dense_encoder.output_shape


@pytest.mark.parametrize('vocab', [['red', 'orange', 'yellow', 'green', 'blue', 'violet'],
                                   ['a', 'b', 'c']]
)
def test_categorical_sparse_encoder(vocab):
    sparse_encoder = CategoricalSparseEncoder(vocab=vocab)
    inputs = torch.randint(len(vocab), (10,))  # Chooses 10 items from vocab with replacement.
    inputs = torch.unsqueeze(inputs, 1)
    outputs = sparse_encoder(inputs)
    # In sparse mode, embedding_size will always be equal vocab_size.
    assert outputs.shape[-1] == len(vocab)
    # Ensures output shape matches encoder expected output shape.
    assert outputs.shape[1:] == sparse_encoder.output_shape
