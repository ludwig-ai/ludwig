from typing import List

import pytest
import torch

from ludwig.encoders.category_encoders import CategoricalEmbedEncoder, CategoricalSparseEncoder
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919
DEVICE = get_torch_device()


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("vocab", [["red", "orange", "yellow", "green", "blue", "violet"], ["a", "b", "c"]])
@pytest.mark.parametrize("embedding_size", [4, 6, 10])
def test_categorical_dense_encoder(vocab: List[str], embedding_size: int, trainable: bool):
    # make repeatable
    torch.manual_seed(RANDOM_SEED)

    dense_encoder = CategoricalEmbedEncoder(
        vocab=vocab,
        embedding_size=embedding_size,
        embeddings_trainable=trainable,
    ).to(DEVICE)
    inputs = torch.randint(len(vocab), (10,)).to(DEVICE)  # Chooses 10 items from vocab with replacement.
    inputs = torch.unsqueeze(inputs, 1)
    outputs = dense_encoder(inputs)
    # In dense mode, the embedding size should be less than or equal to vocab size.
    assert outputs.shape[-1] == min(embedding_size, len(vocab))
    # Ensures output shape matches encoder expected output shape.
    assert outputs.shape[1:] == dense_encoder.output_shape

    # check for parameter updating
    target = torch.randn(outputs.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(dense_encoder, (inputs,), target)

    if trainable:
        assert fpc == 0, "Embedding layer should be trainable, but found to be frozen."
    else:
        assert fpc == 1, "Embedding layer should be frozen, but found to be trainable."

    assert upc == tpc, f"Not all parameters updated.  Parameters not updated: {not_updated}.\nModule: {dense_encoder}"


@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("vocab", [["red", "orange", "yellow", "green", "blue", "violet"], ["a", "b", "c"]])
def test_categorical_sparse_encoder(vocab: List[str], trainable: bool):
    # make repeatable
    torch.manual_seed(RANDOM_SEED)

    sparse_encoder = CategoricalSparseEncoder(vocab=vocab, embeddings_trainable=trainable).to(DEVICE)
    inputs = torch.randint(len(vocab), (10,)).to(DEVICE)  # Chooses 10 items from vocab with replacement.
    inputs = torch.unsqueeze(inputs, 1)
    outputs = sparse_encoder(inputs)
    # In sparse mode, embedding_size will always be equal to vocab size.
    assert outputs.shape[-1] == len(vocab)
    # Ensures output shape matches encoder expected output shape.
    assert outputs.shape[1:] == sparse_encoder.output_shape

    # check for parameter updating
    target = torch.randn(outputs.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(sparse_encoder, (inputs,), target)

    if trainable:
        assert fpc == 0, "Embedding layer should be trainable, but found to be frozen."
    else:
        assert fpc == 1, "Embedding layer should be frozen, but found to be trainable."

    assert upc == tpc, f"Not all parameters updated.  Parameters not updated: {not_updated}.\nModule: {sparse_encoder}"
