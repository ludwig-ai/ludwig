from typing import List

import pytest
import torch

from ludwig.constants import ENCODER_OUTPUT
from ludwig.encoders.bag_encoders import BagEmbedWeightedEncoder
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919
DEVICE = get_torch_device()


@pytest.mark.parametrize("dropout", [0, 0.9])
@pytest.mark.parametrize("num_fc_layers", [0, 2])
@pytest.mark.parametrize("vocab", [["a", "b", "c", "d", "e", "f", "g", "h"]])
@pytest.mark.parametrize("embedding_size", [10])
@pytest.mark.parametrize("representation", ["dense", "sparse"])
def test_set_encoder(vocab: List[str], embedding_size: int, representation: str, num_fc_layers: int, dropout: float):
    # make repeatable
    torch.manual_seed(RANDOM_SEED)

    bag_encoder = BagEmbedWeightedEncoder(
        vocab=vocab,
        representation=representation,
        embedding_size=embedding_size,
        num_fc_layers=num_fc_layers,
        dropout=dropout,
    ).to(DEVICE)
    inputs = torch.randint(0, 9, size=(2, len(vocab))).to(DEVICE)
    outputs = bag_encoder(inputs)[ENCODER_OUTPUT]
    assert outputs.shape[1:] == bag_encoder.output_shape

    # check for parameter updating
    target = torch.randn(outputs.shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(bag_encoder, (inputs,), target)

    if dropout == 0:
        assert upc == tpc, f"Not all parameters updated.  Parameters not updated: {not_updated}.\nModule: {bag_encoder}"
    else:
        # given random seed and configuration, non-zero dropout can take various values
        assert (upc == tpc) or (
            upc == 0
        ), f"Not all parameterss updated.  Parameters not updated: {not_updated}.\nModule: {bag_encoder}"
