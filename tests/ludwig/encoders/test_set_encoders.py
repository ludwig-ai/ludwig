from typing import List

import pytest
import torch

from ludwig.encoders.set_encoders import SetSparseEncoder
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919
DEVICE = get_torch_device()


@pytest.mark.parametrize("num_fc_layers", [0, 2])
@pytest.mark.parametrize("vocab", [["a", "b", "c", "d", "e", "f", "g", "h"]])
@pytest.mark.parametrize("embedding_size", [10])
@pytest.mark.parametrize("representation", ["sparse"])
def test_set_encoder(
    vocab: List[str],
    embedding_size: int,
    representation: str,
    num_fc_layers: int,
):
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # setup encoder to test
    set_encoder = SetSparseEncoder(
        vocab=vocab,
        representation=representation,
        embedding_size=embedding_size,
        num_fc_layers=num_fc_layers,
    ).to(DEVICE)
    inputs = torch.randint(0, 2, size=(2, len(vocab))).bool().to(DEVICE)
    outputs = set_encoder(inputs)
    assert outputs.shape[1:] == set_encoder.output_shape

    # check for parameter updating
    target = torch.randn(outputs.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(set_encoder, (inputs,), target)
    assert tpc == upc, f"Failed to update parameters. Parameters not updated: {not_updated}"
