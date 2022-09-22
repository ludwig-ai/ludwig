import logging

import torch

from ludwig.encoders import h3_encoders
from ludwig.utils.misc_utils import set_random_seed
from ludwig.utils.torch_utils import get_torch_device
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated

RANDOM_SEED = 1919
DEVICE = get_torch_device()

logger = logging.getLogger(__name__)


def test_h3_embed():
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # setup encoder to test
    embed = h3_encoders.H3Embed().to(DEVICE)
    inputs = torch.tensor(
        [
            [2, 0, 14, 102, 7, 0, 3, 5, 0, 5, 5, 0, 5, 7, 7, 7, 7, 7, 7],
            [2, 0, 14, 102, 7, 0, 3, 5, 0, 5, 5, 0, 5, 7, 7, 7, 7, 7, 7],
        ],
        dtype=torch.int32,
    ).to(DEVICE)
    outputs = embed(inputs)
    assert outputs["encoder_output"].size()[1:] == embed.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(embed, (inputs,), target)
    assert tpc == upc, f"Failed to update parameters. Parameters not updated: {not_updated}"


def test_h3_weighted_sum():
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # setup encoder to test
    embed = h3_encoders.H3WeightedSum().to(DEVICE)
    inputs = torch.tensor(
        [
            [2, 0, 14, 102, 7, 0, 3, 5, 0, 5, 5, 0, 5, 7, 7, 7, 7, 7, 7],
            [2, 0, 14, 102, 7, 0, 3, 5, 0, 5, 5, 0, 5, 7, 7, 7, 7, 7, 7],
        ],
        dtype=torch.int32,
    ).to(DEVICE)
    outputs = embed(inputs)
    assert outputs["encoder_output"].size()[1:] == embed.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(embed, (inputs,), target)
    assert tpc == upc, f"Failed to update parameters. Parameters not updated: {not_updated}"


def test_h3_rnn_embed():
    # make repeatable
    set_random_seed(RANDOM_SEED)

    # setup encoder to test
    embed = h3_encoders.H3RNN().to(DEVICE)
    inputs = torch.tensor(
        [
            [2, 0, 14, 102, 7, 0, 3, 5, 0, 5, 5, 0, 5, 7, 7, 7, 7, 7, 7],
            [2, 0, 14, 102, 7, 0, 3, 5, 0, 5, 5, 0, 5, 7, 7, 7, 7, 7, 7],
        ],
        dtype=torch.int32,
    ).to(DEVICE)
    outputs = embed(inputs)
    assert outputs["encoder_output"].size()[1:] == embed.output_shape

    # check for parameter updating
    target = torch.randn(outputs["encoder_output"].shape)
    fpc, tpc, upc, not_updated = check_module_parameters_updated(embed, (inputs,), target)
    assert tpc == upc, f"Failed to update parameters. Parameters not updated: {not_updated}"
